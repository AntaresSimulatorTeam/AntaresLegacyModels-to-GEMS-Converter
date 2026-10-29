# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
"""One legacy simulation, two outputs: the mc-ind/mc-all tree and the views.

1. antares-solver runs the legacy study with `--output=all`: it writes the legacy
   outputs, the simulation tables (legacy variables and extra-outputs) and
   `system-for-views.yml`, the GEMS system of the legacy components.
2. GEMS-ViewsBuilder builds the views of the converter view configs from these
   simulation tables, with the converter libraries, catalogs and taxonomy.
"""

import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml
from antares.craft import Study, read_study_local
from antares.craft.model.output import Output

SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "antares_gems_converter"
LIBRARIES_DIR = SRC_DIR / "libs" / "antares_historic"
CATALOGS_DIR = SRC_DIR / "catalogs"
VIEW_CONFIGS_DIR = SRC_DIR / "views_config"
# The view builder doesn't resolve parent categories, so it is given the flattened taxonomy.
TAXONOMY_FILE = SRC_DIR / "taxonomies" / "antares_legacy_taxonomy_flattened.yml"

# The patterns the view configs are filled with, and the legacy outputs they match:
# mc-ind hourly, mc-ind annual, mc-all annual.
AGGREGATION_PATTERNS = [
    {"id": "hourly_per_scenario", "time-granularity": "hour", "scenario": False},
    {"id": "annual_per_scenario", "time-granularity": "year", "scenario": False},
    {"id": "annual_all_scenario", "time-granularity": "year", "scenario": True},
]

# First hour of the calendar: 1st January of a year starting on a Monday, as
# the default study calendar.
CALENDAR_START = datetime(2018, 1, 1)

logger = logging.getLogger(__name__)


@dataclass
class SimulationResults:
    study: Study
    output: Output
    output_dir: Path
    system_file: Path
    # (view id, aggregation pattern id) -> view
    views: dict[tuple[str, str], pl.DataFrame] = field(default_factory=dict)
    # Model properties the system-for-views lacked (component id -> property ids)
    missing_properties: dict[str, list[str]] = field(default_factory=dict)


def run_legacy_simulation(study: Study, antares_bin_dir: Path) -> tuple[Output, Path]:
    """Run antares-solver on the study; return its output, read by antares-craft, and folder."""
    study_path = Path(study.path)
    command = [
        str(antares_bin_dir / "antares-solver"),
        str(study_path),
        "--output=all",
        "--parquet",
        # mc-ind holds the numbers of the last resolution stage
        "--simulation-table-stages=last",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"antares-solver failed ({result.returncode}):\n{result.stdout[-5000:]}"
        )
    outputs = read_study_local(study_path).get_outputs()
    assert len(outputs) == 1, f"expected one output, got {list(outputs)}"
    output = next(iter(outputs.values()))
    return output, study_path / "output" / output.name


def complete_system_for_views(
    system_file: Path, study: Study, completed_file: Path
) -> dict[str, list[str]]:
    """Copy the system, adding the `name` property the libraries declare but Antares
    may not write (fixed in Antares after 10.2: thermal clusters and short-term
    storages). The value is the cluster id, as in the converter templates.

    Return the properties added, by component.
    """
    system = yaml.safe_load(system_file.read_text(encoding="utf-8"))
    declared = _declared_properties()
    cluster_ids = _cluster_ids_by_component(study)
    added: dict[str, list[str]] = {}
    for component in system["system"]["components"]:
        properties = component.setdefault("properties", [])
        present = {prop["id"] for prop in properties}
        missing = [
            prop for prop in declared.get(component["model"], []) if prop not in present
        ]
        if missing:
            added[component["id"]] = missing
        for prop in missing:
            if prop != "name" or component["id"] not in cluster_ids:
                raise ValueError(
                    f"{system_file}: component {component['id']!r} lacks the "
                    f"property {prop!r} of model {component['model']!r}"
                )
            properties.append({"id": "name", "value": cluster_ids[component["id"]]})
    completed_file.write_text(yaml.safe_dump(system, sort_keys=False), encoding="utf-8")
    return added


def _declared_properties() -> dict[str, list[str]]:
    """`<library id>.<model id>` -> the property ids the model declares."""
    declared = {}
    for library_file in LIBRARIES_DIR.glob("*.yml"):
        library = yaml.safe_load(library_file.read_text(encoding="utf-8"))["library"]
        for model in library["models"]:
            declared[f"{library['id']}.{model['id']}"] = [
                prop["id"] for prop in model.get("properties") or []
            ]
    return declared


def _cluster_ids_by_component(study: Study) -> dict[str, str]:
    ids = {}
    for area_id, area in study.get_areas().items():
        for thermal_id in area.get_thermals():
            ids[f"{area_id}_thermal_{thermal_id}"] = thermal_id
        for storage_id in area.get_st_storages():
            ids[f"{area_id}_short_term_storage_{storage_id}"] = storage_id
    return ids


def write_calendar(simulation_tables: list[Path], calendar_file: Path) -> None:
    """Hourly calendar covering the time steps (and blocks) of the simulation tables."""
    steps = (
        pl.concat(
            [
                pl.scan_parquet(table).select("absolute_time_index", "block")
                for table in simulation_tables
            ]
        )
        .drop_nulls()
        .unique()
        .sort("absolute_time_index")
        .collect()
    )
    calendar = steps.with_columns(
        granular_date=pl.lit(CALENDAR_START)
        + pl.duration(hours=pl.col("absolute_time_index"))
    )
    calendar.select("absolute_time_index", "block", "granular_date").write_csv(
        calendar_file
    )


def write_view_configs(view_configs_dir: Path) -> None:
    """Copy the converter view configs with their aggregations-patterns filled."""
    view_configs_dir.mkdir(parents=True, exist_ok=True)
    for view_file in sorted(VIEW_CONFIGS_DIR.glob("*.yml")):
        view = yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"]
        view["aggregations-patterns"] = AGGREGATION_PATTERNS
        (view_configs_dir / view_file.name).write_text(
            yaml.safe_dump({"view": view}, sort_keys=False), encoding="utf-8"
        )


def build_views(
    system_file: Path, simulation_tables_pattern: str, calendar: Path, work_dir: Path
) -> dict[tuple[str, str], pl.DataFrame]:
    """Run the gems-views-builder command line, as a user would."""
    view_configs_dir = work_dir / "view_configs"
    views_dir = work_dir / "views"
    logs_dir = work_dir / "logs"
    write_view_configs(view_configs_dir)
    views_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "gems_views_builder",
        "--libraries-dir",
        str(LIBRARIES_DIR),
        "--catalogs-dir",
        str(CATALOGS_DIR),
        "--system",
        str(system_file),
        "--taxonomy",
        str(TAXONOMY_FILE),
        "--calendar",
        str(calendar),
        "--view-configs",
        str(view_configs_dir / "*.yml"),
        "--simulation-tables",
        simulation_tables_pattern,
        "--output-dir",
        str(views_dir),
        "--log-dir",
        str(logs_dir),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gems-views-builder failed ({result.returncode}):\n{result.stderr[-5000:]}"
        )
    views = {}
    # Views are named <view id>--<aggregation pattern id>--<timestamp>.parquet
    for view_file in views_dir.glob("*.parquet"):
        view_id, pattern_id, _ = view_file.stem.split("--")
        views[(view_id, pattern_id)] = pl.read_parquet(view_file)
    return views


def simulate(study: Study, antares_bin_dir: Path, work_dir: Path) -> SimulationResults:
    output, output_dir = run_legacy_simulation(study, antares_bin_dir)
    simulation_tables = sorted(output_dir.glob("simulation-table-*.parquet"))
    assert simulation_tables, f"no simulation table in {output_dir}"
    work_dir.mkdir(parents=True, exist_ok=True)

    system_file = work_dir / "system-for-views.yml"
    missing_properties = complete_system_for_views(
        output_dir / "system-for-views.yml", study, system_file
    )
    if missing_properties:
        logger.warning(
            "system-for-views.yml lacks model properties, added: %s",
            missing_properties,
        )
    calendar = work_dir / "calendar.csv"
    write_calendar(simulation_tables, calendar)
    views = build_views(
        system_file,
        str(output_dir / "simulation-table-*.parquet"),
        calendar,
        work_dir,
    )
    return SimulationResults(
        study, output, output_dir, system_file, views, missing_properties
    )
