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
"""Read the legacy scenario builder (SB) entry of the legacy object a
component reads, clear the entries of deleted objects, and read, copy or write
modeler-scenariobuilder.dat."""
import logging
import shutil
from pathlib import Path
from typing import Optional

from antares.craft.model.scenario_builder import ScenarioBuilder
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import LEGACY_TYPE_TO_SB_ATTR
from antares_gems_converter.input_converter.src.parsing import (
    ComponentConversionConfig,
    ObjectProperties,
)

MODELER_SCENARIO_BUILDER_FILE = "modeler-scenariobuilder.dat"


def get_legacy_sb(study: Study, logger: logging.Logger) -> ScenarioBuilder | None:
    # Returns None (instead of raising) when the study has no SB or it cannot be read,
    # so the rest of the conversion proceeds without SB data.
    try:
        return study.get_scenario_builder()
    except Exception as e:
        logger.warning(f"Could not read legacy scenario builder: {e}")
        return None


def legacy_sb_source(
    template_name: str, component: ComponentConversionConfig
) -> Optional[ObjectProperties]:
    """Return the legacy object whose legacy SB entry gives the scenario group
    of `component`, an instantiated component of the template named
    `template_name`: the one object with a legacy SB entry that its
    scenario-dependent parameters read, or None if they read none.

    Parameters reading several such objects raise a ValueError.
    """
    sources: dict[tuple, ObjectProperties] = {}
    for parameter in component.parameters:
        source = parameter.value.object_properties
        if (
            parameter.scenario_dependent
            and source is not None
            and source.type in LEGACY_TYPE_TO_SB_ATTR
        ):
            # Several fields of one object have the same entry
            key = (source.type, source.area, source.cluster, source.link)
            sources.setdefault(key, source)
    if len(sources) > 1:
        raise ValueError(
            f"Template '{template_name}', component '{component.id}': its "
            f"scenario-dependent parameters read several legacy objects with a "
            f"legacy scenario builder entry {sorted(sources)}, so its scenario "
            f"group is ambiguous"
        )
    return next(iter(sources.values()), None)


def read_year_ts(
    legacy_sb: ScenarioBuilder | None, source: ObjectProperties
) -> dict[int, int]:
    """Return {year_index: ts_index} for all years that have an explicit SB
    entry (non-None) for `source`, a legacy object of a type with a legacy SB
    entry."""
    if legacy_sb is None:
        return {}
    sb_matrices = getattr(legacy_sb, LEGACY_TYPE_TO_SB_ATTR[source.type])
    if source.type == "link":
        # SB link ids are sorted alphabetically
        sb_matrix = sb_matrices.get_link(source.link)
    elif source.type in ("thermal", "renewable"):
        sb_matrix = sb_matrices.get_cluster(source.area, source.cluster)
    else:
        sb_matrix = sb_matrices.get_area(source.area)
    return {
        year: ts for year, ts in enumerate(sb_matrix.get_scenario()) if ts is not None
    }


def read_scenario_groups(sb_file: Path) -> set[str]:
    """Return the scenario groups that `sb_file`, a modeler-scenariobuilder.dat,
    defines: its lines are `<group>, <year> = <ts>`, blank or `#` comments."""
    return {
        line.split(",", 1)[0].strip()
        for line in sb_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def clear_legacy_sb_entries(
    study: Study, sb_cleanups: list[tuple[str, str]], logger: logging.Logger
) -> None:
    """Clear the legacy SB entries of each (area, SB attribute) in `sb_cleanups`."""
    if not sb_cleanups:
        return
    try:
        legacy_sb = study.get_scenario_builder()
        for area, type_ in sb_cleanups:
            sb_area = getattr(legacy_sb, type_)
            sb_matrix = sb_area.get_area(area)
            sb_matrix.set_new_scenario([None] * len(sb_matrix.get_scenario()))
        study.set_scenario_builder(legacy_sb)
    except Exception as e:
        logger.warning(f"Failed to persist cleaned legacy scenario builder: {e}")


def copy_scenario_builder_file(
    source_file: Path, series_folder: Path, logger: logging.Logger
) -> None:
    series_folder.mkdir(parents=True, exist_ok=True)

    dest_file = series_folder / MODELER_SCENARIO_BUILDER_FILE  # enforce name

    try:
        shutil.copy2(source_file, dest_file)
        logger.info(f"Copied scenario builder file to {dest_file}")
    except Exception as e:
        logger.warning(f"Failed to copy scenario builder file: {e}")


def write_scenario_builder_file(
    group_data: dict[str, dict[int, int]],
    series_folder: Path,
    logger: logging.Logger,
) -> None:
    series_folder.mkdir(parents=True, exist_ok=True)

    dest_file = series_folder / MODELER_SCENARIO_BUILDER_FILE

    lines = []
    for group in sorted(group_data):
        for year, ts_index in sorted(group_data[group].items()):
            lines.append(f"{group}, {year} = {ts_index}")
    dest_file.write_text("\n".join(lines))

    logger.info(f"Generated scenario builder file at {dest_file}")
