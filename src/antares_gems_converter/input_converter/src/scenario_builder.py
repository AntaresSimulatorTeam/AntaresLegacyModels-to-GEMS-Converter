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
"""Read the legacy scenario builder (SB) of each study object, clear the
entries of deleted objects, and write modeler-scenariobuilder.dat."""
import logging
import shutil
from pathlib import Path

from antares.craft.model.scenario_builder import ScenarioBuilder
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import (
    CLUSTER_TYPE_TO_SB_ATTR,
    LINK_TYPE_TO_SCENARIO_BUILDER_ATTR,
    MATRIX_TYPES_TO_SCENARIO_BUILDER_ATTR,
)
from antares_gems_converter.input_converter.src.parsing import ObjectKind
from antares_gems_converter.input_converter.src.templates import StudyObject

MODELER_SCENARIO_BUILDER_FILE = "modeler-scenariobuilder.dat"


def get_legacy_sb(study: Study, logger: logging.Logger) -> ScenarioBuilder | None:
    # Returns None (instead of raising) when the study has no SB or it cannot be read,
    # so the rest of the conversion proceeds without SB data.
    try:
        return study.get_scenario_builder()
    except Exception as e:
        logger.warning(f"Could not read legacy scenario builder: {e}")
        return None


def read_year_ts(
    legacy_sb: ScenarioBuilder | None,
    template_name: str,
    study_object: StudyObject,
) -> dict[int, int]:
    """Return {year_index: ts_index} for all years that have an explicit SB
    entry (non-None) for `study_object`, an object that the template named
    `template_name` is instantiated for."""
    if legacy_sb is None:
        return {}
    # Links, cluster types (thermal/renewable) and area matrix types
    # (wind/solar/load/hydro/ror) have their own SB attribute
    if study_object.kind == ObjectKind.LINK:
        attr = LINK_TYPE_TO_SCENARIO_BUILDER_ATTR.get(template_name)
    elif study_object.kind == ObjectKind.AREA:
        attr = MATRIX_TYPES_TO_SCENARIO_BUILDER_ATTR.get(template_name)
    else:
        attr = CLUSTER_TYPE_TO_SB_ATTR.get(study_object.kind)
    if attr is None:
        return {}

    sb_matrices = getattr(legacy_sb, attr)
    if study_object.kind == ObjectKind.LINK:
        # SB link ids are sorted alphabetically
        sb_matrix = sb_matrices.get_link(study_object.link)
    elif study_object.kind == ObjectKind.AREA:
        sb_matrix = sb_matrices.get_area(study_object.area)
    else:
        sb_matrix = sb_matrices.get_cluster(study_object.area, study_object.cluster)
    return {
        year: ts for year, ts in enumerate(sb_matrix.get_scenario()) if ts is not None
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
