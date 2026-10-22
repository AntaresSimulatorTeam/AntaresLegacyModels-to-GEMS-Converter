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
import logging
import shutil
from pathlib import Path
from typing import Any, Optional, Union, List

import pandas as pd
from antares.craft.exceptions.exceptions import ReferencedObjectDeletionNotAllowed
from antares.craft.model.link import Link
from antares.craft.model.scenario_builder import ScenarioBuilder
from antares.craft.model.study import Study, read_study_local
from antares.craft.model.hydro import HydroPropertiesUpdate

from antares_gems_converter.input_converter.src.config import (
    BLANKED_LEGACY_OBJECT_TYPES,
    CLUSTER_TYPE_TO_SB_ATTR,
    HYDRO_SB_SHARED_FIELDS,
    HYDRO_TYPE_TO_SET_METHOD,
    LINK_TYPE_TO_SCENARIO_BUILDER_ATTR,
    LINK_TYPES,
    MATRIX_TYPES,
    MATRIX_TYPES_TO_SET_METHOD,
    MATRIX_TYPES_TO_SCENARIO_BUILDER_ATTR,
    MODEL_NAME_TO_FILE_NAME,
    STUDY_LEVEL_DELETION,
    STUDY_LEVEL_GET,
    TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
)
from antares_gems_converter.input_converter.src.data_preprocessing.data_classes import (
    ConversionMode,
)
from antares_gems_converter.input_converter.src.data_preprocessing.preprocessing import (
    ModelConversionPreprocessor,
)
from antares_gems_converter.input_converter.src.parsing import (
    ConversionTemplate,
    ObjectProperties,
    VirtualObjectsRepository,
    parse_conversion_template,
)
from antares_gems_converter.input_converter.src.utils import (
    dump_to_yaml,
    resolve_path,
)
from gems_craft.model.parsing import LibrarySchema, ModelSchema, parse_yaml_library
from gems_craft.study.parsing import (
    ComponentParameterSchema,
    ComponentPropertySchema,
    ComponentSchema,
    PortConnectionsSchema,
    SystemSchema,
)
from gems_craft_hybrid.study.parsing import AreaConnectionsSchema, HybridSystemSchema

MODEL_TEMPLATE_FOLDER = Path(__file__).parents[1] / "data" / "model_configuration"
LIBS_FOLDER = "model-libraries"
SERIES_FOLDER = "data-series"
# Antares >= 10.2 requires this file next to system.yml. Default content: no
# per-model setting, so the solver defaults apply (sequential-subproblems,
# cyclic out-of-bounds processing). Also valid for GemsPy.
OPTIM_CONFIG_FILE = "optim-config.yml"
DEFAULT_OPTIM_CONFIG = "models: []\n"
# Antares >= 10.2 gives legacy objects the same ids as the templates (e.g.
# <area>_thermal_<cluster>, <area>_load) and rejects GEMS components that reuse
# the id of a legacy object still in the study. Hybrid mode deletes the replaced
# clusters and links, so their GEMS components keep the legacy id, but it only
# blanks area-level objects: the GEMS components replacing them get this suffix.
HYBRID_COMPONENT_ID_SUFFIX = "_gems"

# TODO: Move all global variables in a config class, that is used in AntaresStudyConverter constructor


class AntaresStudyConverter:
    def __init__(
        self,
        study_input: Union[Path, Study],
        logger: logging.Logger,
        mode: str = "full",
        output_folder: Path = Path("/tmp/"),
        period: Optional[int] = None,
        lib_paths: Optional[list[str]] = None,
        models_to_convert: Optional[list[str]] = None,
        modeler_scenario_builder_file: Optional[Path] = None,
    ):
        """
        Initialize processor
        """
        self.logger = logger
        self.period: int = period if period else 168
        self.lib_paths: list[str] = lib_paths if lib_paths else []
        self.modeler_scenario_builder_file = modeler_scenario_builder_file
        try:
            self.mode = ConversionMode(mode)
        except ValueError:
            raise ValueError(
                f"Invalid conversion mode: {mode}, possible values are {[conv_mode.value for conv_mode in ConversionMode]}"
            )

        self.models_to_convert = list(
            models_to_convert
            if models_to_convert is not None
            else [m for m in MODEL_NAME_TO_FILE_NAME.keys() if m != "area"]
        )
        if self.mode == ConversionMode.FULL and "area" not in self.models_to_convert:
            self.logger.warning(
                "'area' is not in models_to_convert but conversion mode is 'full': adding it automatically."
            )
            self.models_to_convert.append("area")
        if self.mode == ConversionMode.HYBRID and "area" in self.models_to_convert:
            self.logger.warning(
                "'area' is in models_to_convert but conversion mode is 'hybrid': removing it automatically."
            )
            self.models_to_convert.remove("area")

        # TODO: The logic is still too complicated, needs more refacto / to understand why thermal preprocessing sometimes needs different paths
        study_input_path = (
            study_input.path.stem
            if isinstance(study_input, Study)
            else study_input.stem
        )
        self.output_folder = output_folder / study_input_path

        if self.mode == ConversionMode.HYBRID:
            # In hybrid mode, the output is the input study from which we replace converted components by Gems ones, hence we copy the original study
            shutil.copytree(
                study_input.path if isinstance(study_input, Study) else study_input,
                self.output_folder,
                dirs_exist_ok=True,
            )
            if isinstance(study_input, Path):
                study_input = self.output_folder
        else:
            # In full mode, the output is a full Gems study so no need to copy the original study, we start "from scratch"
            self.output_folder.mkdir(parents=True, exist_ok=True)

        if isinstance(study_input, Study):
            # We have a different way of managing thermal preprocessing files, because in this case we want to modify the study_path.
            # But in the same moment we dont want the preprocessing files in the modified study_path
            study_input.path = self.output_folder
            self.study = study_input
        else:
            self.study = read_study_local(resolve_path(study_input))
        self.output_system_path = self.output_folder / "input" / "system.yml"
        self.output_optim_config_path = self.output_folder / "input" / OPTIM_CONFIG_FILE
        self.areas = self.study.get_areas()
        self.legacy_objects: list[ObjectProperties] = []

    def _delete_legacy_objects(self) -> None:
        # SB is read and written once at the end to avoid overwriting antares-craft's own SB updates
        sb_cleanups: list[tuple[str, str]] = []
        # "mod_inflows" and "ror" share one "hydro" SB bucket per area; only clear it when BOTH are deleted
        hydro_sb_fields_deleted: dict[str, set[str]] = {}
        undeletable_objects: list[str] = []

        for legacy_component in self.legacy_objects:
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
                    target = getattr(
                        self.study, STUDY_LEVEL_GET[legacy_component.type]
                    )()[id]
                    delete_method = STUDY_LEVEL_DELETION[legacy_component.type]
                    if legacy_component.type == "binding_constraint":
                        getattr(self.study, delete_method)([target])
                    else:
                        getattr(self.study, delete_method)(target)
                elif (
                    legacy_component.type in TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD
                    and legacy_component.area is not None
                ):
                    getattr(
                        self.areas[legacy_component.area],
                        TEMPLATE_CLUSTER_TYPE_TO_DELETE_METHOD[legacy_component.type],
                    )(
                        getattr(
                            self.areas[legacy_component.area],
                            TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[legacy_component.type],
                        )()[legacy_component.cluster]
                    )
                elif (
                    legacy_component.type in MATRIX_TYPES_TO_SET_METHOD
                    and legacy_component.area is not None
                ):
                    # To "delete" legacy wind, solar or load object, we simply set an empty timeseries
                    getattr(
                        self.areas[legacy_component.area],
                        MATRIX_TYPES_TO_SET_METHOD[legacy_component.type],
                    )(pd.DataFrame())
                    if legacy_component.type in MATRIX_TYPES_TO_SCENARIO_BUILDER_ATTR:
                        sb_cleanups.append(
                            (legacy_component.area, legacy_component.type)
                        )
                elif (
                    legacy_component.type == "hydro"
                    and legacy_component.area is not None
                    and legacy_component.field is not None
                ):
                    if legacy_component.field in HYDRO_TYPE_TO_SET_METHOD:
                        getattr(
                            self.areas[legacy_component.area].hydro,
                            HYDRO_TYPE_TO_SET_METHOD[legacy_component.field],
                        )(pd.DataFrame())
                        # Wait the post-loop gate to clear the shared SB bucket
                        if legacy_component.field in HYDRO_SB_SHARED_FIELDS:
                            hydro_sb_fields_deleted.setdefault(
                                legacy_component.area, set()
                            ).add(legacy_component.field)
                    else:
                        self.areas[legacy_component.area].hydro.update_properties(
                            HydroPropertiesUpdate(**{legacy_component.field: False})
                        )
                else:
                    raise NotImplementedError
            except ReferencedObjectDeletionNotAllowed as e:
                undeletable_objects.append(str(e))
            except (KeyError, FileNotFoundError):
                self.logger.warning(
                    f"Item {legacy_component} not found in study; skipping deletion"
                )
            except NotImplementedError:
                self.logger.warning(
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

        if sb_cleanups:
            try:
                legacy_sb = self.study.get_scenario_builder()
                for area, type_ in sb_cleanups:
                    sb_area = getattr(legacy_sb, type_)
                    sb_matrix = sb_area.get_area(area)
                    sb_matrix.set_new_scenario([None] * len(sb_matrix.get_scenario()))
                self.study.set_scenario_builder(legacy_sb)
            except Exception as e:
                self.logger.warning(
                    f"Failed to persist cleaned legacy scenario builder: {e}"
                )

        self.legacy_objects = []

    def _iterate_through_model(
        self,
        resolved_conversion_template: ConversionTemplate,
        components: list,
        connections: list,
        area_connections: list,
        mp: ModelConversionPreprocessor,
    ) -> None:
        components_connections_to_avoid: List[str] = []
        id_suffix = self._component_id_suffix(resolved_conversion_template)
        for comp in resolved_conversion_template.components:
            if resolved_conversion_template.name in MATRIX_TYPES:
                if not all(
                    mp.check_timeseries_validity(param.value)
                    for param in comp.parameters
                ):
                    components_connections_to_avoid.append(comp.id)
                    continue
            parameters = [
                ComponentParameterSchema(
                    id=param.id,
                    time_dependent=bool(param.time_dependent),
                    scenario_dependent=bool(param.scenario_dependent),
                    value=mp.convert_param_value(
                        param.id,
                        param.value,
                        comp.id,
                    ),
                )
                for param in comp.parameters
            ]
            properties = (
                [
                    ComponentPropertySchema(
                        id=prop.id,
                        value=str(mp.convert_param_value(prop.id, prop.value, comp.id)),
                    )
                    for prop in comp.properties
                ]
                if comp.properties
                else None
            )
            components.append(
                ComponentSchema(
                    id=self._output_component_id(comp.id, id_suffix),
                    model=resolved_conversion_template.model,
                    scenario_group=comp.scenario_group,
                    parameters=parameters,
                    properties=properties,
                )
            )

        if self.mode == ConversionMode.HYBRID:
            for area_connection in resolved_conversion_template.area_connections:
                # TODO: Improve logic
                if area_connection.component not in components_connections_to_avoid:
                    if "." in area_connection.component:
                        component_parts = area_connection.component.split(".")
                        component_value = getattr(
                            self.study.get_links()[component_parts[0]],
                            component_parts[1],
                        )
                    else:
                        component_value = area_connection.component
                    component_value = self._output_component_id(
                        component_value, id_suffix
                    )
                    if "." in area_connection.area:
                        area_parts = area_connection.area.split(".")
                        area_value = getattr(
                            self.study.get_links()[area_parts[0]], area_parts[1]
                        )
                    else:
                        area_value = area_connection.area
                    area_value = area_value.replace(" / ", "_")
                    area_connections.append(
                        AreaConnectionsSchema(
                            component=component_value,
                            port=area_connection.port,
                            area=area_value,
                        )
                    )
            # TODO : Simplify usage, use directly legacy_objectis_to_delete
            for item in resolved_conversion_template.legacy_objects_to_delete:
                self.legacy_objects.append(item.object_properties)
        else:
            for connection in resolved_conversion_template.connections:
                # TODO: Factorize logic with previous connections
                if (
                    connection.component1 not in components_connections_to_avoid
                    and connection.component2 not in components_connections_to_avoid
                ):
                    treated_components = []
                    for component in [connection.component1, connection.component2]:
                        if "." in component:
                            component_parts = component.split(".")
                            component_value = getattr(
                                self.study.get_links()[component_parts[0]],
                                component_parts[1],
                            ) + ".".join(component_parts[2:])
                        else:
                            component_value = component
                        treated_components.append(component_value.replace(" / ", "_"))

                    connections.append(
                        PortConnectionsSchema(
                            component1=treated_components[0],
                            port1=connection.port1,
                            component2=treated_components[1],
                            port2=connection.port2,
                        )
                    )

    def _component_id_suffix(self, conversion_template: ConversionTemplate) -> str:
        # In hybrid mode, a blanked (not deleted) legacy object stays in the study
        # under the id that the GEMS components replacing it would have.
        if self.mode == ConversionMode.HYBRID and any(
            item.object_properties.type in BLANKED_LEGACY_OBJECT_TYPES
            for item in conversion_template.legacy_objects_to_delete
        ):
            return HYBRID_COMPONENT_ID_SUFFIX
        return ""

    @staticmethod
    def _output_component_id(component_id: str, suffix: str) -> str:
        # " / " in component ids comes from link IDs (e.g. "at / fr") — normalize to "_".
        return component_id.replace(" / ", "_") + suffix

    @staticmethod
    def _read_sb_year_ts(sb_matrix: Any) -> dict[int, int]:
        # Returns {year_index: ts_index} for all years that have an explicit SB entry (non-None).
        return {
            year: ts
            for year, ts in enumerate(sb_matrix.get_scenario())
            if ts is not None
        }

    @staticmethod
    def _apply_sb_groups(
        resolved_template: ConversionTemplate,
        year_ts: dict[int, int],
        group_data: dict[str, dict[int, int]],
    ) -> ConversionTemplate:
        # For each component that carries a scenario_group placeholder from the YAML template:
        # - if year_ts is non-empty: keep the group name, register its {year: ts} data in group_data
        #   (used later to write modeler-scenariobuilder.dat)
        # - if year_ts is empty (no SB entries, or no legacy SB at all): clear to None so the
        #   placeholder value does not leak into the output system
        # " / " in group names comes from link IDs (e.g. "at / fr") — normalize to "_".
        new_components = []
        changed = False
        for comp in resolved_template.components:
            if comp.scenario_group is not None:
                normalized = comp.scenario_group.replace(" / ", "_")
                if year_ts:
                    group_data[normalized] = year_ts
                    new_comp = comp.model_copy(update={"scenario_group": normalized})
                else:
                    new_comp = comp.model_copy(update={"scenario_group": None})
                new_components.append(new_comp)
                changed = True
            else:
                new_components.append(comp)
        if changed:
            return resolved_template.model_copy(update={"components": new_components})
        return resolved_template

    def _convert_model_to_component_list(
        self,
        conversion_template: ConversionTemplate,
        virtual_objects: VirtualObjectsRepository = VirtualObjectsRepository(),
        legacy_sb: ScenarioBuilder | None = None,
    ) -> tuple[
        list[ComponentSchema],
        list[PortConnectionsSchema],
        list[AreaConnectionsSchema],
        dict[str, dict[int, int]],
    ]:
        components: list[ComponentSchema] = []
        connections: list[PortConnectionsSchema] = []
        area_connections: list[AreaConnectionsSchema] = []
        group_data: dict[str, dict[int, int]] = {}

        model_area_pattern = f"${{{conversion_template.template_parameters[0].name}}}"

        model_preprocessor = ModelConversionPreprocessor(
            self.study, self.mode, self.output_folder
        )

        model_name = conversion_template.name

        try:
            if model_name in LINK_TYPES:
                # Link SB attribute
                sb_link_attr = LINK_TYPE_TO_SCENARIO_BUILDER_ATTR.get(model_name)
                for link in self.study.get_links().values():
                    if not self.is_virtual_link(link, virtual_objects):
                        resolved_template = conversion_template.resolve_template(
                            model_area_pattern, link.id
                        )
                        # Link IDs are sorted alphabetically
                        if legacy_sb is not None and sb_link_attr is not None:
                            sb_obj = getattr(legacy_sb, sb_link_attr)
                            year_ts = self._read_sb_year_ts(sb_obj.get_link(link.id))
                        else:
                            year_ts = {}
                        # Always call _apply_sb_groups: clears scenario_group to None when year_ts is empty
                        resolved_template = self._apply_sb_groups(
                            resolved_template, year_ts, group_data
                        )
                        self._iterate_through_model(
                            resolved_template,
                            components,
                            connections,
                            area_connections,
                            model_preprocessor,
                        )
            else:
                cluster_type = next(
                    (
                        template.cluster_type
                        for template in conversion_template.template_parameters
                    ),
                    None,
                )
                # Resolve the SB attribute name for cluster types (thermal/renewable)
                # or area matrix types (wind/solar/load/hydro/ror)
                sb_area_attr = (
                    CLUSTER_TYPE_TO_SB_ATTR.get(cluster_type)
                    if cluster_type
                    else MATRIX_TYPES_TO_SCENARIO_BUILDER_ATTR.get(model_name)
                )
                for area in self.areas.values():
                    if area.id not in virtual_objects.areas:
                        area_resolved_template = conversion_template.resolve_template(
                            model_area_pattern, area.id
                        )
                        resolved_virtual_objects = virtual_objects.resolve_template(
                            "${area}", area.id
                        )
                        if cluster_type:
                            clusters = getattr(
                                area, TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[cluster_type]
                            )()
                            for cluster_id, cluster in clusters.items():
                                if cluster_id not in resolved_virtual_objects.thermals:
                                    if (
                                        cluster_type == "renewable"
                                        and not cluster.properties.enabled
                                    ):
                                        self.logger.warning(
                                            f"Renewable cluster '{cluster_id}' in area "
                                            f"'{area.id}' is disabled; skipping conversion."
                                        )
                                        continue
                                    # We have already resolved areas, now need to resolve cluster ids
                                    cluster_resolved_template = (
                                        area_resolved_template.resolve_template(
                                            f"${{{cluster_type}}}", cluster_id
                                        )
                                    )
                                    if (
                                        legacy_sb is not None
                                        and sb_area_attr is not None
                                    ):
                                        sb_obj = getattr(legacy_sb, sb_area_attr)
                                        year_ts = self._read_sb_year_ts(
                                            sb_obj.get_cluster(area.id, cluster_id)
                                        )
                                    else:
                                        year_ts = {}
                                    # Always call _apply_sb_groups: clears scenario_group to None when year_ts is empty
                                    cluster_resolved_template = self._apply_sb_groups(
                                        cluster_resolved_template, year_ts, group_data
                                    )
                                    self._iterate_through_model(
                                        cluster_resolved_template,
                                        components,
                                        connections,
                                        area_connections,
                                        model_preprocessor,
                                    )
                        else:
                            if legacy_sb is not None and sb_area_attr is not None:
                                sb_obj = getattr(legacy_sb, sb_area_attr)
                                year_ts = self._read_sb_year_ts(
                                    sb_obj.get_area(area.id)
                                )
                            else:
                                year_ts = {}
                            # Always call _apply_sb_groups: clears scenario_group to None when year_ts is empty
                            area_resolved_template = self._apply_sb_groups(
                                area_resolved_template, year_ts, group_data
                            )
                            self._iterate_through_model(
                                area_resolved_template,
                                components,
                                connections,
                                area_connections,
                                model_preprocessor,
                            )
        except (KeyError, FileNotFoundError) as e:
            # Not raised: a template is applied to every area, so the battery
            # template fails on any area without the battery artefact (e.g. a
            # study with no battery, while battery is converted by default).
            # This also stops the model's conversion at the first such area.
            self.logger.error(
                f"Error while converting model '{model_name}' to component list: "
                f"{e!r}. Please check the model configuration file."
            )

        return components, connections, area_connections, group_data

    @staticmethod
    def is_virtual_link(link: Link, virtual_objects: VirtualObjectsRepository) -> bool:
        return (
            link.id in virtual_objects.links
            or link.area_from_id in virtual_objects.areas
            or link.area_to_id in virtual_objects.areas
        )

    @staticmethod
    def _parse_library(path: str) -> LibrarySchema:
        try:
            with Path(path).open("r", encoding="utf-8") as file:
                return parse_yaml_library(file)
        except Exception as e:
            raise ValueError(f"Failed to parse library file {path}: {e}") from e

    def _check_converted_models_are_in_libs(
        self, model_conversion_templates: dict[str, ConversionTemplate]
    ) -> None:
        lib_to_models: dict[str, dict[str, ModelSchema]] = {}
        for lib_path in self.lib_paths:
            lib_data = self._parse_library(lib_path)
            lib_to_models[lib_data.id] = {model.id: model for model in lib_data.models}

        mismatches: list[str] = []
        for model in self.models_to_convert:
            conversion_template = model_conversion_templates[model]
            lib_id, model_id = conversion_template.model.split(".")
            if lib_id not in lib_to_models:
                raise ValueError(
                    f"Library {lib_id} has not been found in provided libraries"
                )
            if model_id not in lib_to_models[lib_id]:
                raise ValueError(
                    f"Model {model_id} has not been found in library {lib_id}"
                )
            mismatches += self._template_mismatches_with_model(
                conversion_template, lib_to_models[lib_id][model_id]
            )

        # A mismatch would otherwise only surface later, in GEMS or the solver,
        # far from the template, or go unnoticed
        if mismatches:
            raise ValueError(
                "Conversion templates do not match the parameters and properties"
                " of their library models:\n" + "\n".join(mismatches)
            )

    @staticmethod
    def _template_mismatches_with_model(
        conversion_template: ConversionTemplate, model: ModelSchema
    ) -> list[str]:
        model_ids = {
            "parameter": {param.id for param in model.parameters},
            "property": {prop.id for prop in model.properties},
        }
        mismatches = []
        for comp in conversion_template.components:
            template_ids = {
                "parameter": {param.id for param in comp.parameters},
                "property": {prop.id for prop in comp.properties},
            }
            for kind in ("parameter", "property"):
                for issue, ids in (
                    ("unknown", template_ids[kind] - model_ids[kind]),
                    ("missing", model_ids[kind] - template_ids[kind]),
                ):
                    if ids:
                        mismatches.append(
                            f"- template '{conversion_template.name}', component"
                            f" '{comp.id}', model '{conversion_template.model}':"
                            f" {issue} {kind}(s) {sorted(ids)}"
                        )
        return mismatches

    # TODO: Does not depend on self for now, but will be once the config is a class attribute
    def _get_model_conversion_template(self, model: str) -> ConversionTemplate:
        model_conversion_template_file = (
            MODEL_TEMPLATE_FOLDER / MODEL_NAME_TO_FILE_NAME[model]
        )
        if not model_conversion_template_file.exists():
            raise FileNotFoundError(
                f"The model configuration file for {model} has not been found at the location {model_conversion_template_file}"
            )
        # TODO: Parse yaml file with Pydantic to return a proper Python object rather than a generic dict
        with model_conversion_template_file.open() as template:
            return parse_conversion_template(template)

    def _copy_libs_to_model_librairies(self) -> None:
        # Retrieve library files and put it in the output study (as fro now libs must be contained in modeler studies)
        dest_dir = self.output_folder / "input" / LIBS_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in self.lib_paths:
            shutil.copy2(path, dest_dir)

    def _create_dataseries_dir(self) -> None:
        dest_dir = self.output_folder / "input" / SERIES_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)

    def _convert_single_model(
        self,
        conversion_template: ConversionTemplate,
        virtual_objects: VirtualObjectsRepository,
        components: list[ComponentSchema],
        connections: list[PortConnectionsSchema],
        area_connections: list[AreaConnectionsSchema],
        legacy_sb: ScenarioBuilder | None,
    ) -> dict[str, dict[int, int]]:
        self.logger.info(
            f"Converting components of model {conversion_template.name}..."
        )

        (
            components_from_model,
            connections_from_model,
            area_connections_from_model,
            group_data,
        ) = self._convert_model_to_component_list(
            conversion_template, virtual_objects, legacy_sb
        )

        components.extend(components_from_model)
        connections.extend(connections_from_model)
        area_connections.extend(area_connections_from_model)
        return group_data

    def _get_legacy_sb(self) -> ScenarioBuilder | None:
        # Returns None (instead of raising) when the study has no SB or it cannot be read,
        # so the rest of the conversion proceeds without SB data.
        try:
            return self.study.get_scenario_builder()
        except Exception as e:
            self.logger.warning(f"Could not read legacy scenario builder: {e}")
            return None

    def convert_study_to_input_system(self) -> SystemSchema | HybridSystemSchema:
        self._copy_libs_to_model_librairies()
        # Copy a pre-existing modeler SB file if one was provided (skips auto-generation below)
        self._copy_scenario_builder()
        self._create_dataseries_dir()
        model_conversion_templates = self._build_model_conversion_templates()
        self._check_converted_models_are_in_libs(model_conversion_templates)
        virtual_objects = self._build_virtual_objects_repo(model_conversion_templates)
        components: list[ComponentSchema] = []
        connections: list[PortConnectionsSchema] = []
        area_connections: list[AreaConnectionsSchema] = []

        # Read the legacy SB once and pass it to every model conversion
        legacy_sb = self._get_legacy_sb()
        # Accumulates {group_name: {year: ts_index}} across all converted models
        all_group_data: dict[str, dict[int, int]] = {}

        for model in self.models_to_convert:
            conversion_template = model_conversion_templates[model]
            group_data = self._convert_single_model(
                conversion_template,
                virtual_objects,
                components,
                connections,
                area_connections,
                legacy_sb,
            )
            all_group_data.update(group_data)

        # Auto-generate modeler-scenariobuilder.dat only when no file was provided and there is SB data
        if not self.modeler_scenario_builder_file and all_group_data:
            self._generate_scenario_builder_file(all_group_data)

        if self.mode == ConversionMode.HYBRID:
            self._delete_legacy_objects()
            system = HybridSystemSchema(
                id=self.study.name,
                components=components,
                connections=connections or None,
                area_connections=area_connections or None,
            )
            # exclude_unset keeps defaults (e.g. integer_strategy) unset, so they are not dumped
            data = system.model_dump(exclude_none=True, exclude_unset=True)
            return HybridSystemSchema(**data)
        system = SystemSchema(
            id=self.study.name,
            components=components,
            connections=connections or None,
        )
        data = system.model_dump(exclude_none=True, exclude_unset=True)
        return SystemSchema(**data)

    def _build_model_conversion_templates(self) -> dict[str, ConversionTemplate]:
        model_conversion_templates: dict[str, ConversionTemplate] = {}
        for model in self.models_to_convert:
            model_conversion_templates[model] = self._get_model_conversion_template(
                model
            )
        return model_conversion_templates

    def _build_virtual_objects_repo(
        self, model_conversion_templates: dict[str, ConversionTemplate]
    ) -> VirtualObjectsRepository:
        virtual_objects = VirtualObjectsRepository()
        for model in self.models_to_convert:
            virtual_objects_this_model = model_conversion_templates[
                model
            ].get_excluded_objects_ids()
            virtual_objects.add(virtual_objects_this_model)
        return virtual_objects

    def process_all(self) -> None:
        system = self.convert_study_to_input_system()
        self.logger.info("Dumping input system into yaml file...")
        dump_to_yaml(model=system, output_path=self.output_system_path)
        self._write_optim_config()

    def _write_optim_config(self) -> None:
        # Keep an existing optim-config.yml (e.g. from a hybrid input study)
        if self.output_optim_config_path.exists():
            return
        self.output_optim_config_path.write_text(DEFAULT_OPTIM_CONFIG)
        self.logger.info(
            f"Generated optim-config file at {self.output_optim_config_path}"
        )

    def _copy_scenario_builder(self) -> None:
        if not self.modeler_scenario_builder_file:
            return

        dest = self.output_folder / "input" / "data-series"
        dest.mkdir(parents=True, exist_ok=True)

        dest_file = dest / "modeler-scenariobuilder.dat"  # enforce name

        try:
            shutil.copy2(self.modeler_scenario_builder_file, dest_file)
            self.logger.info(f"Copied scenario builder file to {dest_file}")
        except Exception as e:
            self.logger.warning(f"Failed to copy scenario builder file: {e}")

    def _generate_scenario_builder_file(
        self, group_year_scanned: dict[str, dict[int, int]]
    ) -> None:
        dest = self.output_folder / "input" / SERIES_FOLDER
        dest.mkdir(parents=True, exist_ok=True)

        dest_file = dest / "modeler-scenariobuilder.dat"

        lines = []
        for group in sorted(group_year_scanned):
            for year, ts_index in sorted(group_year_scanned[group].items()):
                lines.append(f"{group}, {year} = {ts_index}")
        dest_file.write_text("\n".join(lines))

        self.logger.info(f"Generated scenario builder file at {dest_file}")
