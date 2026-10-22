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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from antares.craft.exceptions.exceptions import ReferencedObjectDeletionNotAllowed
from antares.craft.model.scenario_builder import ScenarioBuilder
from antares.craft.model.study import Study, read_study_local
from antares.craft.model.hydro import HydroPropertiesUpdate

from antares_gems_converter.input_converter.src.checks import (
    check_converted_models_are_in_libs,
)
from antares_gems_converter.input_converter.src.config import (
    BLANKED_LEGACY_OBJECT_TYPES,
    HYDRO_SB_SHARED_FIELDS,
    HYDRO_TYPE_TO_SET_METHOD,
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
from antares_gems_converter.input_converter.src.optim_config import (
    OPTIM_CONFIG_FILE,
    write_optim_config,
)
from antares_gems_converter.input_converter.src.parsing import (
    ConversionTemplate,
    ObjectProperties,
    VirtualObjectsRepository,
)
from antares_gems_converter.input_converter.src.scenario_builder import (
    apply_sb_groups,
    copy_scenario_builder_file,
    get_legacy_sb,
    read_year_ts,
    write_scenario_builder_file,
)
from antares_gems_converter.input_converter.src.templates import (
    build_virtual_objects_repo,
    load_conversion_templates,
    resolved_templates,
)
from antares_gems_converter.input_converter.src.utils import (
    dump_to_yaml,
    resolve_path,
)
from gems_craft.study.parsing import (
    ComponentParameterSchema,
    ComponentPropertySchema,
    ComponentSchema,
    PortConnectionsSchema,
    SystemSchema,
)
from gems_craft_hybrid.study.parsing import AreaConnectionsSchema, HybridSystemSchema

LIBS_FOLDER = "model-libraries"
SERIES_FOLDER = "data-series"
# Antares >= 10.2 gives legacy objects the same ids as the templates (e.g.
# <area>_thermal_<cluster>, <area>_load) and rejects GEMS components that reuse
# the id of a legacy object still in the study. Hybrid mode deletes the replaced
# clusters and links, so their GEMS components keep the legacy id, but it only
# blanks area-level objects: the GEMS components replacing them get this suffix.
HYBRID_COMPONENT_ID_SUFFIX = "_gems"

# TODO: Move all global variables in a config class, that is used in AntaresStudyConverter constructor


@dataclass
class SystemFragment:
    """Part of the output system: what one or more converted templates add to it."""

    components: list[ComponentSchema] = field(default_factory=list)
    connections: list[PortConnectionsSchema] = field(default_factory=list)
    area_connections: list[AreaConnectionsSchema] = field(default_factory=list)
    # {scenario group: {year: ts index}}, written to modeler-scenariobuilder.dat
    group_data: dict[str, dict[int, int]] = field(default_factory=dict)

    def merge(self, other: "SystemFragment") -> None:
        self.components.extend(other.components)
        self.connections.extend(other.connections)
        self.area_connections.extend(other.area_connections)
        self.group_data.update(other.group_data)


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

    def _convert_resolved_template(
        self,
        resolved_conversion_template: ConversionTemplate,
        mp: ModelConversionPreprocessor,
        fragment: SystemFragment,
    ) -> None:
        id_suffix = self._component_id_suffix(resolved_conversion_template)
        skipped_components = self._add_components(
            resolved_conversion_template, mp, id_suffix, fragment
        )
        if self.mode == ConversionMode.HYBRID:
            self._add_area_connections(
                resolved_conversion_template, skipped_components, id_suffix, fragment
            )
            # TODO : Simplify usage, use directly legacy_objectis_to_delete
            for item in resolved_conversion_template.legacy_objects_to_delete:
                self.legacy_objects.append(item.object_properties)
        else:
            self._add_port_connections(
                resolved_conversion_template, skipped_components, id_suffix, fragment
            )

    def _add_components(
        self,
        resolved_conversion_template: ConversionTemplate,
        mp: ModelConversionPreprocessor,
        id_suffix: str,
        fragment: SystemFragment,
    ) -> list[str]:
        """Add the template's components to `fragment` and return the ids of
        those skipped because their time series are empty or all zeros."""
        skipped_components: list[str] = []
        for comp in resolved_conversion_template.components:
            if resolved_conversion_template.name in MATRIX_TYPES:
                if not all(
                    mp.check_timeseries_validity(param.value)
                    for param in comp.parameters
                ):
                    skipped_components.append(comp.id)
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
            fragment.components.append(
                ComponentSchema(
                    id=self._output_component_id(comp.id, id_suffix),
                    model=resolved_conversion_template.model,
                    scenario_group=comp.scenario_group,
                    parameters=parameters,
                    properties=properties,
                )
            )
        return skipped_components

    def _add_area_connections(
        self,
        resolved_conversion_template: ConversionTemplate,
        skipped_components: list[str],
        id_suffix: str,
        fragment: SystemFragment,
    ) -> None:
        for area_connection in resolved_conversion_template.area_connections:
            if area_connection.component in skipped_components:
                continue
            fragment.area_connections.append(
                AreaConnectionsSchema(
                    component=self._resolve_ref(area_connection.component, id_suffix),
                    port=area_connection.port,
                    area=self._resolve_ref(area_connection.area),
                )
            )

    def _add_port_connections(
        self,
        resolved_conversion_template: ConversionTemplate,
        skipped_components: list[str],
        id_suffix: str,
        fragment: SystemFragment,
    ) -> None:
        for connection in resolved_conversion_template.connections:
            if (
                connection.component1 in skipped_components
                or connection.component2 in skipped_components
            ):
                continue
            fragment.connections.append(
                PortConnectionsSchema(
                    component1=self._resolve_ref(connection.component1, id_suffix),
                    port1=connection.port1,
                    component2=self._resolve_ref(connection.component2, id_suffix),
                    port2=connection.port2,
                )
            )

    def _resolve_ref(self, ref: str, suffix: str = "") -> str:
        # A reference "<link id>.<attribute>[.<rest>]" is replaced by that
        # attribute of the link, directly followed by the rest: with
        # the link "at / fr", "at / fr.area_from_id._node" becomes "at_node".
        # Link ids must not contain "."
        if "." in ref:
            link_id, attribute, *rest = ref.split(".")
            ref = getattr(self.study.get_links()[link_id], attribute) + ".".join(rest)
        return self._output_component_id(ref, suffix)

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

    def convert_template(
        self,
        conversion_template: ConversionTemplate,
        virtual_objects: Optional[VirtualObjectsRepository] = None,
        legacy_sb: ScenarioBuilder | None = None,
    ) -> SystemFragment:
        """Convert the study objects that `conversion_template` applies to.

        In hybrid mode, also record the legacy objects to delete. Objects in
        `virtual_objects` are skipped; `legacy_sb` gives the scenario groups.
        """
        if virtual_objects is None:
            virtual_objects = VirtualObjectsRepository()
        fragment = SystemFragment()
        model_preprocessor = ModelConversionPreprocessor(
            self.study, self.mode, self.output_folder
        )
        try:
            for resolved_template, study_object in resolved_templates(
                conversion_template,
                self.study,
                self.areas,
                virtual_objects,
                self.logger,
            ):
                year_ts = read_year_ts(legacy_sb, conversion_template, study_object)
                # Always called: clears scenario_group to None when year_ts is empty
                resolved_template = apply_sb_groups(
                    resolved_template, year_ts, fragment.group_data
                )
                self._convert_resolved_template(
                    resolved_template, model_preprocessor, fragment
                )
        except (KeyError, FileNotFoundError) as e:
            # Not raised: a template is applied to every area, so the battery
            # template fails on any area without the battery artefact (e.g. a
            # study with no battery, while battery is converted by default).
            # This also stops the model's conversion at the first such area.
            self.logger.error(
                f"Error while converting model '{conversion_template.name}' to "
                f"component list: {e!r}. Please check the model configuration file."
            )
        return fragment

    def _copy_libs_to_model_librairies(self) -> None:
        # Retrieve library files and put it in the output study (as fro now libs must be contained in modeler studies)
        dest_dir = self.output_folder / "input" / LIBS_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in self.lib_paths:
            shutil.copy2(path, dest_dir)

    def _create_dataseries_dir(self) -> None:
        dest_dir = self.output_folder / "input" / SERIES_FOLDER
        dest_dir.mkdir(parents=True, exist_ok=True)

    def convert_study_to_input_system(self) -> SystemSchema | HybridSystemSchema:
        self._copy_libs_to_model_librairies()
        # Copy a pre-existing modeler SB file if one was provided (skips auto-generation below)
        if self.modeler_scenario_builder_file:
            copy_scenario_builder_file(
                self.modeler_scenario_builder_file,
                self.output_folder / "input" / SERIES_FOLDER,
                self.logger,
            )
        self._create_dataseries_dir()
        model_conversion_templates = load_conversion_templates(self.models_to_convert)
        check_converted_models_are_in_libs(model_conversion_templates, self.lib_paths)
        virtual_objects = build_virtual_objects_repo(model_conversion_templates)

        # Read the legacy SB once and pass it to every model conversion
        legacy_sb = get_legacy_sb(self.study, self.logger)
        converted = SystemFragment()
        for model in self.models_to_convert:
            conversion_template = model_conversion_templates[model]
            self.logger.info(
                f"Converting components of model {conversion_template.name}..."
            )
            converted.merge(
                self.convert_template(conversion_template, virtual_objects, legacy_sb)
            )

        # Auto-generate modeler-scenariobuilder.dat only when no file was provided and there is SB data
        if not self.modeler_scenario_builder_file and converted.group_data:
            write_scenario_builder_file(
                converted.group_data,
                self.output_folder / "input" / SERIES_FOLDER,
                self.logger,
            )

        if self.mode == ConversionMode.HYBRID:
            self._delete_legacy_objects()
            system = HybridSystemSchema(
                id=self.study.name,
                components=converted.components,
                connections=converted.connections or None,
                area_connections=converted.area_connections or None,
            )
            # exclude_unset keeps defaults (e.g. integer_strategy) unset, so they are not dumped
            data = system.model_dump(exclude_none=True, exclude_unset=True)
            return HybridSystemSchema(**data)
        system = SystemSchema(
            id=self.study.name,
            components=converted.components,
            connections=converted.connections or None,
        )
        data = system.model_dump(exclude_none=True, exclude_unset=True)
        return SystemSchema(**data)

    def process_all(self) -> None:
        system = self.convert_study_to_input_system()
        self.logger.info("Dumping input system into yaml file...")
        dump_to_yaml(model=system, output_path=self.output_system_path)
        write_optim_config(self.output_optim_config_path, self.logger)
