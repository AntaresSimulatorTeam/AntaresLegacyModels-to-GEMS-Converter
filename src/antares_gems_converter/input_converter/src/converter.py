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
from typing import Optional, Sequence, Union

from antares.craft.model.scenario_builder import ScenarioBuilder
from antares.craft.model.study import Study, read_study_local

from antares_gems_converter.input_converter.src.checks import (
    check_converted_models_are_in_libs,
)
from antares_gems_converter.input_converter.src.config import (
    BLANKED_LEGACY_OBJECT_TYPES,
    MATRIX_TYPES,
    MODEL_NAME_TO_FILE_NAME,
)
from antares_gems_converter.input_converter.src.data_preprocessing.data_classes import (
    ConversionMode,
)
from antares_gems_converter.input_converter.src.data_preprocessing.preprocessing import (
    ModelConversionPreprocessor,
)
from antares_gems_converter.input_converter.src.deletion import delete_legacy_objects
from antares_gems_converter.input_converter.src.optim_config import (
    OPTIM_CONFIG_FILE,
    write_optim_config,
)
from antares_gems_converter.input_converter.src.parsing import (
    ConversionTemplate,
    ObjectProperties,
    TemplateBody,
    TemplateInstance,
)
from antares_gems_converter.input_converter.src.scenario_builder import (
    apply_sb_groups,
    copy_scenario_builder_file,
    get_legacy_sb,
    read_year_ts,
    write_scenario_builder_file,
)
from antares_gems_converter.input_converter.src.templates import (
    load_conversion_templates,
    study_objects,
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

    def _convert_instance(
        self,
        instance: TemplateInstance,
        mp: ModelConversionPreprocessor,
        fragment: SystemFragment,
    ) -> None:
        id_suffix = self._component_id_suffix(instance)
        skipped_components = self._add_components(instance, mp, id_suffix, fragment)
        if self.mode == ConversionMode.HYBRID:
            self._add_area_connections(
                instance, skipped_components, id_suffix, fragment
            )
            # TODO : Simplify usage, use directly legacy_objectis_to_delete
            for item in instance.legacy_objects_to_delete:
                self.legacy_objects.append(item.object_properties)
        else:
            self._add_port_connections(
                instance, skipped_components, id_suffix, fragment
            )

    def _add_components(
        self,
        instance: TemplateInstance,
        mp: ModelConversionPreprocessor,
        id_suffix: str,
        fragment: SystemFragment,
    ) -> list[str]:
        """Add the instance's components to `fragment` and return the ids of
        those skipped because their time series are empty or all zeros."""
        skipped_components: list[str] = []
        for comp in instance.components:
            if instance.name in MATRIX_TYPES:
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
                    model=instance.model,
                    scenario_group=comp.scenario_group,
                    parameters=parameters,
                    properties=properties,
                )
            )
        return skipped_components

    def _add_area_connections(
        self,
        instance: TemplateInstance,
        skipped_components: list[str],
        id_suffix: str,
        fragment: SystemFragment,
    ) -> None:
        for area_connection in instance.area_connections:
            if area_connection.component in skipped_components:
                continue
            fragment.area_connections.append(
                AreaConnectionsSchema(
                    component=self._output_component_id(
                        area_connection.component, id_suffix
                    ),
                    port=area_connection.port,
                    area=self._output_component_id(area_connection.area, ""),
                )
            )

    def _add_port_connections(
        self,
        instance: TemplateInstance,
        skipped_components: list[str],
        id_suffix: str,
        fragment: SystemFragment,
    ) -> None:
        for connection in instance.connections:
            if (
                connection.component1 in skipped_components
                or connection.component2 in skipped_components
            ):
                continue
            fragment.connections.append(
                PortConnectionsSchema(
                    component1=self._output_component_id(
                        connection.component1, id_suffix
                    ),
                    port1=connection.port1,
                    component2=self._output_component_id(
                        connection.component2, id_suffix
                    ),
                    port2=connection.port2,
                )
            )

    def _component_id_suffix(self, conversion_template: TemplateBody) -> str:
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
        exclusions: Sequence[ObjectProperties] = (),
        legacy_sb: ScenarioBuilder | None = None,
    ) -> SystemFragment:
        """Convert the study objects that `conversion_template` applies to.

        In hybrid mode, also record the legacy objects to delete. Objects
        matching `exclusions` are skipped; `legacy_sb` gives the scenario groups.
        """
        fragment = SystemFragment()
        model_preprocessor = ModelConversionPreprocessor(
            self.study, self.mode, self.output_folder
        )
        try:
            for study_object in study_objects(
                conversion_template,
                self.study,
                self.areas,
                exclusions,
                self.logger,
            ):
                instance = conversion_template.instantiate(study_object.substitutions())
                year_ts = read_year_ts(
                    legacy_sb, conversion_template.name, study_object
                )
                # Always called: clears scenario_group to None when year_ts is empty
                instance = apply_sb_groups(instance, year_ts, fragment.group_data)
                self._convert_instance(instance, model_preprocessor, fragment)
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

    def load_and_check_templates(
        self,
    ) -> tuple[dict[str, ConversionTemplate], list[ObjectProperties]]:
        """Load the templates of `models_to_convert`, check them against the
        libraries, and return them with the objects they exclude.

        An exclusion applies to all the converted templates, not only to the
        one declaring it, and only if that one is converted: e.g. converting
        battery excludes its legacy artefacts (z_batteries area, injection
        clusters) from the area, thermal and link conversions, which would
        otherwise count the battery twice.
        """
        model_conversion_templates = load_conversion_templates(self.models_to_convert)
        check_converted_models_are_in_libs(model_conversion_templates, self.lib_paths)
        exclusions = [
            excluded
            for conversion_template in model_conversion_templates.values()
            for excluded in conversion_template.excluded_objects
        ]
        return model_conversion_templates, exclusions

    def system_and_scenario_groups(
        self,
        model_conversion_templates: dict[str, ConversionTemplate],
        exclusions: Sequence[ObjectProperties],
    ) -> tuple[SystemSchema | HybridSystemSchema, dict[str, dict[int, int]]]:
        """Convert the study to a GEMS system, and return it with the scenario
        groups of its components: {scenario group: {year: ts index}}.

        In hybrid mode, also record the legacy objects to delete in
        `self.legacy_objects`.
        """
        # Read the legacy SB once and pass it to every model conversion
        legacy_sb = get_legacy_sb(self.study, self.logger)
        self.legacy_objects = []
        converted = SystemFragment()
        for conversion_template in model_conversion_templates.values():
            self.logger.info(
                f"Converting components of model {conversion_template.name}..."
            )
            converted.merge(
                self.convert_template(conversion_template, exclusions, legacy_sb)
            )

        if self.mode == ConversionMode.HYBRID:
            system = HybridSystemSchema(
                id=self.study.name,
                components=converted.components,
                connections=converted.connections or None,
                area_connections=converted.area_connections or None,
            )
            # exclude_unset keeps defaults (e.g. integer_strategy) unset, so they are not dumped
            data = system.model_dump(exclude_none=True, exclude_unset=True)
            return HybridSystemSchema(**data), converted.group_data
        system = SystemSchema(
            id=self.study.name,
            components=converted.components,
            connections=converted.connections or None,
        )
        data = system.model_dump(exclude_none=True, exclude_unset=True)
        return SystemSchema(**data), converted.group_data

    def process_all(self) -> None:
        self._copy_libs_to_model_librairies()
        series_folder = self.output_folder / "input" / SERIES_FOLDER
        # Copy a pre-existing modeler SB file if one was provided (skips auto-generation below)
        if self.modeler_scenario_builder_file:
            copy_scenario_builder_file(
                self.modeler_scenario_builder_file, series_folder, self.logger
            )
        self._create_dataseries_dir()

        model_conversion_templates, exclusions = self.load_and_check_templates()
        system, scenario_groups = self.system_and_scenario_groups(
            model_conversion_templates, exclusions
        )

        # Auto-generate modeler-scenariobuilder.dat only when no file was provided and there is SB data
        if not self.modeler_scenario_builder_file and scenario_groups:
            write_scenario_builder_file(scenario_groups, series_folder, self.logger)

        if self.mode == ConversionMode.HYBRID:
            delete_legacy_objects(
                self.study, self.areas, self.legacy_objects, self.logger
            )
            self.legacy_objects = []

        self.logger.info("Dumping input system into yaml file...")
        dump_to_yaml(model=system, output_path=self.output_system_path)
        write_optim_config(self.output_optim_config_path, self.logger)
