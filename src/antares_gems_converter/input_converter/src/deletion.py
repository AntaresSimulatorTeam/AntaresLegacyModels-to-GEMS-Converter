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
"""Hybrid mode: delete or blank the legacy objects replaced by GEMS components."""
import logging
from typing import Mapping

import pandas as pd
from antares.craft.exceptions.exceptions import ReferencedObjectDeletionNotAllowed
from antares.craft.model.area import Area
from antares.craft.model.hydro import HydroPropertiesUpdate
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import (
    HYDRO_SB_SHARED_FIELDS,
    HYDRO_TYPE_TO_SET_METHOD,
    LEGACY_TYPE_TO_SB_ATTR,
    MATRIX_TYPES_TO_SET_METHOD,
    STUDY_LEVEL_DELETION,
    STUDY_LEVEL_GET,
    TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
)
from antares_gems_converter.input_converter.src.parsing import ObjectProperties
from antares_gems_converter.input_converter.src.scenario_builder import (
    clear_legacy_sb_entries,
)


def delete_legacy_objects(
    study: Study,
    areas: Mapping[str, Area],
    legacy_objects: list[ObjectProperties],
    logger: logging.Logger,
) -> None:
    # SB is read and written once at the end to avoid overwriting antares-craft's own SB updates
    sb_cleanups: list[tuple[str, str]] = []
    # "mod_inflows" and "ror" share one "hydro" SB bucket per area; only clear it when BOTH are deleted
    hydro_sb_fields_deleted: dict[str, set[str]] = {}
    undeletable_objects: list[str] = []

    for legacy_component in legacy_objects:
        try:
            if legacy_component.type in STUDY_LEVEL_DELETION:
                if legacy_component.type == "area":
                    id = legacy_component.area
                elif legacy_component.type == "link":
                    id = legacy_component.link
                elif legacy_component.type == "binding_constraint":
                    id = legacy_component.binding_constraint_id
                else:
                    continue
                target = getattr(study, STUDY_LEVEL_GET[legacy_component.type])()[id]
                delete_method = STUDY_LEVEL_DELETION[legacy_component.type]
                if legacy_component.type == "binding_constraint":
                    getattr(study, delete_method)([target])
                else:
                    getattr(study, delete_method)(target)
            elif (
                legacy_component.type in TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD
                and legacy_component.area is not None
            ):
                getattr(
                    areas[legacy_component.area],
                    TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD[legacy_component.type],
                )(
                    getattr(
                        areas[legacy_component.area],
                        TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[legacy_component.type],
                    )()[legacy_component.cluster]
                )
            elif (
                legacy_component.type in MATRIX_TYPES_TO_SET_METHOD
                and legacy_component.area is not None
            ):
                # To "delete" legacy wind, solar or load object, we simply set an empty timeseries
                getattr(
                    areas[legacy_component.area],
                    MATRIX_TYPES_TO_SET_METHOD[legacy_component.type],
                )(pd.DataFrame())
                if legacy_component.type in LEGACY_TYPE_TO_SB_ATTR:
                    sb_cleanups.append(
                        (
                            legacy_component.area,
                            LEGACY_TYPE_TO_SB_ATTR[legacy_component.type],
                        )
                    )
            elif (
                legacy_component.type == "hydro"
                and legacy_component.area is not None
                and legacy_component.field is not None
            ):
                if legacy_component.field in HYDRO_TYPE_TO_SET_METHOD:
                    getattr(
                        areas[legacy_component.area].hydro,
                        HYDRO_TYPE_TO_SET_METHOD[legacy_component.field],
                    )(pd.DataFrame())
                    # Wait the post-loop gate to clear the shared SB bucket
                    if legacy_component.field in HYDRO_SB_SHARED_FIELDS:
                        hydro_sb_fields_deleted.setdefault(
                            legacy_component.area, set()
                        ).add(legacy_component.field)
                else:
                    areas[legacy_component.area].hydro.update_properties(
                        HydroPropertiesUpdate(**{legacy_component.field: False})
                    )
            else:
                raise NotImplementedError
        except ReferencedObjectDeletionNotAllowed as e:
            undeletable_objects.append(str(e))
        except (KeyError, FileNotFoundError):
            logger.warning(
                f"Item {legacy_component} not found in study; skipping deletion"
            )
        except NotImplementedError:
            logger.warning(
                f"Failure to delete {legacy_component} because the method is not implemented yet on antares craft"
            )

    # A legacy object left in the study would be counted twice, along with
    # the GEMS component replacing it, and Antares >= 10.2 rejects a GEMS
    # component that reuses the id of a legacy object.
    if undeletable_objects:
        raise ValueError(
            "Hybrid conversion failed: some converted legacy objects cannot be"
            " deleted. Remove these binding constraints from the study, or do"
            " not convert these objects.\n" + "\n".join(undeletable_objects)
        )

    # Clear the shared "hydro" SB bucket only when both siblings were deleted this run
    for area, fields in hydro_sb_fields_deleted.items():
        if HYDRO_SB_SHARED_FIELDS <= fields:
            sb_cleanups.append((area, "hydro"))

    clear_legacy_sb_entries(study, sb_cleanups, logger)
