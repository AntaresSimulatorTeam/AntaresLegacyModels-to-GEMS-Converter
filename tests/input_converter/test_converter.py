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
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import MODEL_NAME_TO_FILE_NAME
from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter
from antares_gems_converter.input_converter.src.logger import Logger
from antares_gems_converter.input_converter.src.parsing import (
    Operation,
    parse_conversion_template,
)
from antares_gems_converter.input_converter.src.utils import (
    check_file_exists,
    dump_to_yaml,
)
from gems.model.resolve_library import resolve_library
from gems.study.parsing import (
    AreaConnectionsSchema,
    ComponentParameterSchema,
    ComponentSchema,
    ComponentPropertySchema,
    PortConnectionsSchema,
    SystemSchema,
    parse_yaml_components,
)
from gems.study.resolve_components import resolve_system
from tests.input_converter.conftest import create_dataframe_from_constant
import numpy as np

RESOURCES_FOLDER = (
    Path(__file__).parents[2]
    / "src"
    / "antares_gems_converter"
    / "input_converter"
    / "data"
    / "model_configuration"
)
LOCAL_PATH = "mini_test_batterie_BP23"
DATAFRAME_PREPRO_SERIES = (create_dataframe_from_constant(lines=8760),)  # series

DATAFRAME_PREPRO_THERMAL_CONFIG = (
    create_dataframe_from_constant(lines=8760, columns=4),  # modulation
    create_dataframe_from_constant(lines=8760),  # series
)

DATAFRAME_PREPRO_BC_CONFIG = (
    create_dataframe_from_constant(lines=8760, columns=6),  # modulation
    create_dataframe_from_constant(lines=8760, columns=4),  # series
)
LIB_PATHS = [
    "src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml",
    "src/antares_gems_converter/libs/reference_models/andromede_v1_models.yml",
]
MODEL_LIST_WITH_BASE = [str(Path(os.getcwd()) / suffix) for suffix in LIB_PATHS]
DATAFRAME_MISC_GEN_CONFIG = pd.DataFrame(np.zeros((3, 8), dtype=int))
DATAFRAME_MISC_GEN_CONFIG.iloc[:, 1] = 1


class TestConverter:
    def _init_converter_from_study(
        self,
        local_study,
        model_list: list[str] = list(MODEL_NAME_TO_FILE_NAME.keys()),
        mode: str = "full",
    ):
        logger = Logger(__name__, local_study.path)
        converter: AntaresStudyConverter = AntaresStudyConverter(
            study_input=local_study,
            logger=logger,
            mode=mode,
            lib_paths=LIB_PATHS,
            models_to_convert=model_list,
            output_folder=local_study.path.parent / "converter_output",
        )
        return converter

    def _init_converter_from_path(
        self,
        input_path: Path,
        output_path: Path,
        mode: str = "full",
        lib_paths: list = None,
        model_list: list = list(MODEL_NAME_TO_FILE_NAME.keys()),
    ):
        logger = Logger(__name__, str(input_path))
        converter: AntaresStudyConverter = AntaresStudyConverter(
            study_input=input_path,
            logger=logger,
            mode=mode,
            output_folder=output_path,
            lib_paths=lib_paths,
            models_to_convert=model_list,
        )
        return converter

    def test_convert_study_to_input_study(self, local_study_w_areas: Study):
        converter = self._init_converter_from_study(local_study_w_areas, model_list=[])
        input_study = converter.convert_study_to_input_system()

        expected_input_study = SystemSchema(
            id="studyTest",
            components=[
                ComponentSchema(
                    id="fr_node",
                    model="antares_legacy_models.area",
                    scenario_group=None,
                    parameters=[
                        ComponentParameterSchema(
                            id="unsupplied_energy_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        ComponentParameterSchema(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                    properties=[
                        ComponentPropertySchema(id="carrier", value="electricity")
                    ],
                ),
                ComponentSchema(
                    id="it_node",
                    model="antares_legacy_models.area",
                    scenario_group=None,
                    parameters=[
                        ComponentParameterSchema(
                            id="unsupplied_energy_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        ComponentParameterSchema(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                    properties=[
                        ComponentPropertySchema(id="carrier", value="electricity")
                    ],
                ),
            ],
        )
        assert input_study == expected_input_study

    def test_convert_area_to_component(self, local_study_w_areas: Study):
        converter = self._init_converter_from_study(local_study_w_areas, model_list=[])
        path_area = RESOURCES_FOLDER / "area.yaml"
        with path_area.open() as template:
            resource_content = parse_conversion_template(template)
        (area_components, _, _) = converter._convert_model_to_component_list(
            resource_content
        )

        expected_area_components = [
            ComponentSchema(
                id="fr_node",
                model="antares_legacy_models.area",
                parameters=[
                    ComponentParameterSchema(
                        id="unsupplied_energy_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.5,
                    ),
                    ComponentParameterSchema(
                        id="spillage_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                ],
                properties=[ComponentPropertySchema(id="carrier", value="electricity")],
            ),
            ComponentSchema(
                id="it_node",
                model="antares_legacy_models.area",
                parameters=[
                    ComponentParameterSchema(
                        id="unsupplied_energy_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.5,
                    ),
                    ComponentParameterSchema(
                        id="spillage_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                ],
                properties=[ComponentPropertySchema(id="carrier", value="electricity")],
            ),
        ]

        assert area_components == expected_area_components

    def test_convert_area_to_yaml(self, local_study_w_areas: Study):
        converter = self._init_converter_from_study(local_study_w_areas, model_list=[])
        path_area = RESOURCES_FOLDER / "area.yaml"
        with path_area.open() as template:
            resource_content = parse_conversion_template(template)
        (area_components, _, _) = converter._convert_model_to_component_list(
            resource_content
        )
        input_study = SystemSchema(id=converter.study.name, components=area_components)

        # Dump model into yaml file
        yaml_path = converter.output_folder / "study_path.yaml"
        dump_to_yaml(model=input_study, output_path=yaml_path)
        # Open yaml file to validate
        with open(yaml_path, "r", encoding="utf-8") as yaml_file:
            validated_data = parse_yaml_components(yaml_file)

        expected_validated_data = SystemSchema(
            id="studyTest",
            components=[
                ComponentSchema(
                    id="fr_node",
                    model="antares_legacy_models.area",
                    scenario_group=None,
                    parameters=[
                        ComponentParameterSchema(
                            id="unsupplied_energy_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        ComponentParameterSchema(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                    properties=[
                        ComponentPropertySchema(id="carrier", value="electricity")
                    ],
                ),
                ComponentSchema(
                    id="it_node",
                    model="antares_legacy_models.area",
                    scenario_group=None,
                    parameters=[
                        ComponentParameterSchema(
                            id="unsupplied_energy_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=0.5,
                        ),
                        ComponentParameterSchema(
                            id="spillage_cost",
                            time_dependent=False,
                            scenario_dependent=False,
                            scenario_group=None,
                            value=1.0,
                        ),
                    ],
                    properties=[
                        ComponentPropertySchema(id="carrier", value="electricity")
                    ],
                ),
            ],
        )

        assert validated_data == expected_validated_data

    def test_convert_st_storages_to_component(
        self, local_study_with_st_storage: Study, lib_id: str
    ):
        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_study(
            local_study_with_st_storage, model_list=[]
        )
        path_load = RESOURCES_FOLDER / "st-storage.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            storage_components,
            storage_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        inflows_path = "inflows_fr_short_term_storage_storage_1"
        lower_rule_curve_path = "lower_rule_curve_fr_short_term_storage_storage_1"
        pmax_injection_path = "max_injection_modulation_fr_short_term_storage_storage_1"
        pmax_withdrawal_path = (
            "max_withdrawal_modulation_fr_short_term_storage_storage_1"
        )
        upper_rule_curve_path = "upper_rule_curve_fr_short_term_storage_storage_1"
        cost_injection_path = "injection_cost_fr_short_term_storage_storage_1"
        cost_withdrawal_path = "withdrawal_cost_fr_short_term_storage_storage_1"
        cost_level_path = "level_cost_fr_short_term_storage_storage_1"
        expected_storage_connections = [
            PortConnectionsSchema(
                component1="fr_short_term_storage_storage_1",
                port1="balance_port",
                component2="fr_node",
                port2="balance_port",
            )
        ]
        expected_storage_component = [
            ComponentSchema(
                id="fr_short_term_storage_storage_1",
                model=f"{lib_id}.short_term_storage",
                scenario_group=None,
                parameters=[
                    ComponentParameterSchema(
                        id="reservoir_capacity",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    ComponentParameterSchema(
                        id="nominal_injection_capacity",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=10.0,
                    ),
                    ComponentParameterSchema(
                        id="nominal_withdrawal_capacity",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=10.0,
                    ),
                    ComponentParameterSchema(
                        id="injection_efficiency",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1,
                    ),
                    ComponentParameterSchema(
                        id="withdrawal_efficiency",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1,
                    ),
                    ComponentParameterSchema(
                        id="lower_rule_curve",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{lower_rule_curve_path}",
                    ),
                    ComponentParameterSchema(
                        id="upper_rule_curve",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{upper_rule_curve_path}",
                    ),
                    ComponentParameterSchema(
                        id="max_injection_modulation",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{pmax_injection_path}",
                    ),
                    ComponentParameterSchema(
                        id="max_withdrawal_modulation",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{pmax_withdrawal_path}",
                    ),
                    ComponentParameterSchema(
                        id="inflows",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{inflows_path}",
                    ),
                    ComponentParameterSchema(
                        id="initial_level",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.5,
                    ),
                    ComponentParameterSchema(
                        id="injection_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{cost_injection_path}",
                    ),
                    ComponentParameterSchema(
                        id="withdrawal_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{cost_withdrawal_path}",
                    ),
                    ComponentParameterSchema(
                        id="level_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{cost_level_path}",
                    ),
                    ComponentParameterSchema(
                        id="is_overflow_allowed",
                        value=0,
                    ),
                    ComponentParameterSchema(
                        id="is_injection_variation_penalized",
                        value=0,
                    ),
                    ComponentParameterSchema(
                        id="is_withdrawal_variation_penalized",
                        value=0,
                    ),
                    ComponentParameterSchema(
                        id="injection_variation_penalty",
                        time_dependent=True,
                        scenario_dependent=False,
                        value=f"injection_variation_penalty_fr_short_term_storage_storage_1",
                    ),
                    ComponentParameterSchema(
                        id="withdrawal_variation_penalty",
                        time_dependent=True,
                        scenario_dependent=False,
                        value=f"withdrawal_variation_penalty_fr_short_term_storage_storage_1",
                    ),
                    ComponentParameterSchema(
                        id="overflow_cost",
                        value=2,
                    ),
                ],
                properties=[
                    ComponentPropertySchema(id="carrier", value="electricity"),
                    ComponentPropertySchema(id="group", value="other1"),
                ],
            )
        ]

        assert storage_components == expected_storage_component
        assert storage_connections == expected_storage_connections

    # This parametrize allows to pass the parameter "DATAFRAME_PREPRO_THERMAL_CONFIG" inside the fixture
    # To specify the modulation and series dataframes
    @pytest.mark.parametrize(
        "local_study_w_thermal",
        [DATAFRAME_PREPRO_THERMAL_CONFIG],
        indirect=True,
    )
    def test_convert_thermals_to_component(
        self,
        local_study_w_thermal: Study,
    ):
        converter = self._init_converter_from_study(
            local_study_w_thermal, model_list=[]
        )
        path_load = RESOURCES_FOLDER / "thermal.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            thermals_components,
            thermals_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        # study_path = converter.output_folder
        # series_path = study_path / "input" / "thermal" / "series" / "fr" / "gaz"
        print(thermals_components)
        expected_thermals_connections = [
            PortConnectionsSchema(
                component1="fr_thermal_gaz",
                port1="balance_port",
                component2="fr_node",
                port2="balance_port",
            )
        ]
        expected_thermals_components = [
            ComponentSchema(
                id="fr_thermal_gaz",
                model="antares_legacy_models.thermal",
                scenario_group=None,
                parameters=[
                    ComponentParameterSchema(
                        id="cluster_min_gen_modulation",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value="cluster_min_gen_modulation_fr_thermal_gaz",
                    ),
                    ComponentParameterSchema(
                        id="cluster_max_generation",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value="cluster_max_generation_fr_thermal_gaz",
                    ),
                    ComponentParameterSchema(
                        id="min_power_per_unit",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    ComponentParameterSchema(
                        id="max_power_per_unit",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=2.0,
                    ),
                    ComponentParameterSchema(
                        id="generation_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    ComponentParameterSchema(
                        id="market_bid_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    ComponentParameterSchema(
                        id="startup_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    ComponentParameterSchema(
                        id="fixed_cost",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0.0,
                    ),
                    ComponentParameterSchema(
                        id="min_up_duration",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                    ComponentParameterSchema(
                        id="min_down_duration",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                    ComponentParameterSchema(
                        id="num_units",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=1.0,
                    ),
                    ComponentParameterSchema(
                        id="spinning",
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0,
                    ),
                ]
                + [
                    ComponentParameterSchema(
                        id=pollutant,
                        time_dependent=False,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=0,
                    )
                    for pollutant in [
                        "co2_emissions_rate",
                        "nh3_emissions_rate",
                        "so2_emissions_rate",
                        "nox_emissions_rate",
                        "pm2_5_emissions_rate",
                        "pm5_emissions_rate",
                        "pm10_emissions_rate",
                        "nmvoc_emissions_rate",
                        "op1_emissions_rate",
                        "op2_emissions_rate",
                        "op3_emissions_rate",
                        "op4_emissions_rate",
                        "op5_emissions_rate",
                    ]
                ],
                properties=[
                    ComponentPropertySchema(id="carrier", value="electricity"),
                    ComponentPropertySchema(id="technology", value="other 1"),
                    ComponentPropertySchema(id="plant", value="gaz"),
                ],
            )
        ]
        # TODO preprocessing + nouveaux parametres liées a la nouvelle version antarescraft
        assert thermals_components == expected_thermals_components
        assert thermals_connections == expected_thermals_connections

    def test_convert_hydro_to_component(
        self,
        local_study_with_hydro: Study,
    ):
        converter = self._init_converter_from_study(
            local_study_with_hydro, model_list=[]
        )
        path_load = RESOURCES_FOLDER / "hydro.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            hydro_components,
            hydro_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        hydro_fr_component = next(
            (comp for comp in hydro_components if comp.id == "fr_hydro_storage"), None
        )
        hydro_fr_connection = next(
            (
                conn
                for conn in hydro_connections
                if conn.component1 == "fr_hydro_storage"
            ),
            None,
        )

        expected_hydro_connection = PortConnectionsSchema(
            component1="fr_hydro_storage",
            port1="balance_port",
            component2="fr_node",
            port2="balance_port",
        )
        expected_hydro_component = ComponentSchema(
            id="fr_hydro_storage",
            model="antares_legacy_models.long_term_storage",
            scenario_group=None,
            parameters=[
                ComponentParameterSchema(
                    id="reservoir_capacity",
                    time_dependent=False,
                    scenario_dependent=False,
                    scenario_group=None,
                    value=1000,
                ),
                ComponentParameterSchema(
                    id="nominal_pumping_capacity",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value="nominal_pumping_capacity_fr_hydro_storage",
                ),
                ComponentParameterSchema(
                    id="nominal_generation_capacity",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value="nominal_generation_capacity_fr_hydro_storage",
                ),
                ComponentParameterSchema(
                    id="minimum_generation",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value="minimum_generation_fr_hydro_storage",
                ),
                ComponentParameterSchema(
                    id="pumping_efficiency",
                    time_dependent=False,
                    scenario_dependent=False,
                    scenario_group=None,
                    value=0.75,
                ),
                ComponentParameterSchema(
                    id="lower_rule_curve",
                    time_dependent=True,
                    scenario_dependent=False,
                    scenario_group=None,
                    value="lower_rule_curve_fr_hydro_storage",
                ),
                ComponentParameterSchema(
                    id="upper_rule_curve",
                    time_dependent=True,
                    scenario_dependent=False,
                    scenario_group=None,
                    value="upper_rule_curve_fr_hydro_storage",
                ),
                ComponentParameterSchema(
                    id="inflows",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value="inflows_fr_hydro_storage",
                ),
                ComponentParameterSchema(
                    id="initial_and_final_level",
                    time_dependent=True,
                    scenario_dependent=True,
                    scenario_group=None,
                    value="initial_and_final_level_fr_hydro_storage",
                ),
                ComponentParameterSchema(
                    id="overflow_cost",
                    time_dependent=False,
                    scenario_dependent=False,
                    scenario_group=None,
                    value=1,
                ),
            ],
            properties=[
                ComponentPropertySchema(id="carrier", value="electricity"),
            ],
        )
        assert hydro_fr_connection == expected_hydro_connection
        assert hydro_fr_component == expected_hydro_component

    def test_convert_load_to_component_from_path(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH

        output_path = local_path / "reference.yaml"
        with open(output_path) as system_file:
            expected_data = parse_yaml_components(system_file)

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_path(
            input_path, output_path, "full", model_list=[]
        )
        path_load = RESOURCES_FOLDER / "load.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            load_components,
            load_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        assert load_connections == [
            c for c in expected_data.connections if c.component1 == "fr_load"
        ]
        assert load_components == [
            c for c in expected_data.components if c.id == "fr_load"
        ]
        # TODO enrich

    @pytest.mark.parametrize(
        "fr_solar",
        [DATAFRAME_PREPRO_SERIES],
        indirect=True,
    )
    def test_convert_solar_to_component_from_study(self, fr_solar: None):
        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_study(fr_solar, model_list=[])

        path_load = RESOURCES_FOLDER / "solar.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            solar_components,
            solar_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        solar_fr_component = next(
            (comp for comp in solar_components if comp.id == "fr_solar"), None
        )
        solar_fr_connection = next(
            (conn for conn in solar_connections if conn.component1 == "fr_solar"), None
        )
        solar_timeseries = "available_power_fr_solar"
        expected_solar_connection = PortConnectionsSchema(
            component1="fr_solar",
            port1="balance_port",
            component2="fr_node",
            port2="balance_port",
        )

        expected_solar_components = ComponentSchema(
            id="fr_solar",
            model="antares_legacy_models.renewable",
            scenario_group=None,
            parameters=[
                ComponentParameterSchema(
                    id="nominal_capacity",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                    scenario_group=None,
                ),
                ComponentParameterSchema(
                    id="num_units",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                    scenario_group=None,
                ),
                ComponentParameterSchema(
                    id="available_power",
                    time_dependent=True,
                    scenario_dependent=True,
                    value=f"{solar_timeseries}",
                    scenario_group=None,
                ),
            ],
            properties=[
                ComponentPropertySchema(id="carrier", value="electricity"),
                ComponentPropertySchema(id="technology", value="solar"),
            ],
        )
        assert solar_fr_connection == expected_solar_connection
        assert solar_fr_component == expected_solar_components

    def test_convert_load_to_component_from_study(self, fr_load: None):
        converter = self._init_converter_from_study(fr_load)
        path_load = RESOURCES_FOLDER / "load.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            load_components,
            load_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        load_fr_component = next(
            (comp for comp in load_components if comp.id == "fr_load"), None
        )
        load_fr_connection = next(
            (conn for conn in load_connections if conn.component1 == "fr_load"), None
        )

        load_timeseries = "load_fr_load"
        expected_load_connection = PortConnectionsSchema(
            component1="fr_load",
            port1="balance_port",
            component2="fr_node",
            port2="balance_port",
        )
        expected_load_components = ComponentSchema(
            id="fr_load",
            model="antares_legacy_models.load",
            scenario_group=None,
            parameters=[
                ComponentParameterSchema(
                    id="load",
                    time_dependent=True,
                    scenario_dependent=True,
                    value=f"{load_timeseries}",
                    scenario_group=None,
                ),
            ],
            properties=[ComponentPropertySchema(id="carrier", value="electricity")],
        )
        assert load_fr_connection == expected_load_connection
        assert load_fr_component == expected_load_components

    @pytest.mark.parametrize(
        "fr_wind",
        [DATAFRAME_PREPRO_SERIES],
        indirect=True,
    )
    def test_convert_wind_to_component_from_study(self, fr_wind: Study):
        converter = self._init_converter_from_study(fr_wind, model_list=[])

        path_load = RESOURCES_FOLDER / "wind.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            wind_components,
            wind_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        wind_fr_component = next(
            (comp for comp in wind_components if comp.id == "fr_wind"), None
        )
        wind_fr_connection = next(
            (conn for conn in wind_connections if conn.component1 == "fr_wind"), None
        )

        expected_wind_connection = PortConnectionsSchema(
            component1="fr_wind",
            port1="balance_port",
            component2="fr_node",
            port2="balance_port",
        )
        expected_wind_components = ComponentSchema(
            id="fr_wind",
            model="antares_legacy_models.renewable",
            scenario_group="wind_group",
            parameters=[
                ComponentParameterSchema(
                    id="nominal_capacity",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                ),
                ComponentParameterSchema(
                    id="num_units",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                ),
                ComponentParameterSchema(
                    id="available_power",
                    time_dependent=True,
                    scenario_dependent=True,
                    value="available_power_fr_wind",
                ),
            ],
            properties=[
                ComponentPropertySchema(id="carrier", value="electricity"),
                ComponentPropertySchema(id="technology", value="wind"),
            ],
        )
        assert wind_fr_connection == expected_wind_connection
        assert wind_fr_component == expected_wind_components

    @pytest.mark.parametrize(
        "fr_wind",
        [
            pd.DataFrame(),  # DataFrame empty
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_empty_file(
        self,
        fr_wind: object,
    ):
        converter = self._init_converter_from_study(fr_wind, model_list=[])

        path_load = RESOURCES_FOLDER / "wind.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            wind_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        assert wind_components == []

    @pytest.mark.parametrize(
        "fr_wind",
        [
            pd.DataFrame([0, 0, 0]),  # DataFrame full of 0
        ],
        indirect=True,
    )
    def test_convert_wind_to_component_zero_values(self, fr_wind: int):
        converter = self._init_converter_from_study(fr_wind, model_list=[])

        path_load = RESOURCES_FOLDER / "wind.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            wind_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        assert wind_components == []

    @pytest.mark.parametrize(
        "fr_misc_gen",
        [create_dataframe_from_constant(lines=8760, columns=8)],
        indirect=True,
    )
    def test_convert_misc_gen_to_components_from_study(self, fr_misc_gen: Study):
        converter = self._init_converter_from_study(fr_misc_gen, model_list=[])

        path_load = RESOURCES_FOLDER / "misc_gen.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            misc_gen_components,
            misc_gen_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        expected_misc_gen_connections = [
            PortConnectionsSchema(
                component1=f"fr_{generation_type}",
                port1="balance_port",
                component2="fr_node",
                port2="balance_port",
            )
            for generation_type in [
                "combined_heat_power",
                "biomass",
                "biogas",
                "waste",
                "geothermal",
                "other",
                "pumped_storage_power",
                "rest_world",
            ]
        ]
        expected_misc_gen_components = [
            ComponentSchema(
                id=f"fr_{generation_type}",
                model="antares_legacy_models.miscellaneous_generation",
                parameters=[
                    ComponentParameterSchema(
                        id="available_power",
                        time_dependent=True,
                        scenario_dependent=False,
                        value=f"available_power_fr_{generation_type}",
                    )
                ],
                properties=[
                    ComponentPropertySchema(id="carrier", value="electricity"),
                    ComponentPropertySchema(id="technology", value=generation_type),
                    ComponentPropertySchema(
                        id="miscellaneous_type",
                        value="misc_ndg"
                        if generation_type not in ["pumped_storage_power", "rest_world"]
                        else generation_type,
                    ),
                ],
            )
            for generation_type in [
                "combined_heat_power",
                "biomass",
                "biogas",
                "waste",
                "geothermal",
                "other",
                "pumped_storage_power",
                "rest_world",
            ]
        ]
        assert misc_gen_connections == expected_misc_gen_connections
        assert misc_gen_components == expected_misc_gen_components

    @pytest.mark.parametrize(
        "fr_misc_gen",
        [
            pd.DataFrame(),  # DataFrame empty
        ],
        indirect=True,
    )
    def test_convert_misc_gen_to_components_empty_file(
        self,
        fr_misc_gen: object,
    ):
        converter = self._init_converter_from_study(fr_misc_gen, model_list=[])

        path_load = RESOURCES_FOLDER / "misc_gen.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            misc_gen_components,
            misc_gen_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        assert misc_gen_components == []
        assert misc_gen_connections == []

    @pytest.mark.parametrize(
        "fr_misc_gen",
        [DATAFRAME_MISC_GEN_CONFIG],
        indirect=True,
    )
    def test_convert_misc_gen_to_components_zero_values(self, fr_misc_gen: int):
        converter = self._init_converter_from_study(fr_misc_gen, model_list=[])

        path_load = RESOURCES_FOLDER / "misc_gen.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            misc_gen_components,
            misc_gen_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        expected_misc_gen_connections = [
            PortConnectionsSchema(
                component1=f"fr_biomass",
                port1="balance_port",
                component2="fr_node",
                port2="balance_port",
            )
        ]
        expected_misc_gen_components = [
            ComponentSchema(
                id=f"fr_biomass",
                model="antares_legacy_models.miscellaneous_generation",
                parameters=[
                    ComponentParameterSchema(
                        id="available_power",
                        time_dependent=True,
                        scenario_dependent=False,
                        value=f"available_power_fr_biomass",
                    )
                ],
                properties=[
                    ComponentPropertySchema(id="carrier", value="electricity"),
                    ComponentPropertySchema(id="technology", value="biomass"),
                    ComponentPropertySchema(id="miscellaneous_type", value="misc_ndg"),
                ],
            )
        ]
        assert misc_gen_connections == expected_misc_gen_connections
        assert misc_gen_components == expected_misc_gen_components

    @pytest.mark.parametrize(
        "fr_ror",
        [DATAFRAME_PREPRO_SERIES],
        indirect=True,
    )
    def test_convert_ror_to_component_from_study(self, fr_ror: Study):
        converter = self._init_converter_from_study(fr_ror, model_list=[])

        path_load = RESOURCES_FOLDER / "ror.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            ror_components,
            ror_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        ror_fr_component = next(
            (comp for comp in ror_components if comp.id == "fr_run_of_river"), None
        )
        ror_fr_connection = next(
            (conn for conn in ror_connections if conn.component1 == "fr_run_of_river"),
            None,
        )

        expected_ror_connection = PortConnectionsSchema(
            component1="fr_run_of_river",
            port1="balance_port",
            component2="fr_node",
            port2="balance_port",
        )
        expected_ror_component = ComponentSchema(
            id="fr_run_of_river",
            model="antares_legacy_models.renewable",
            scenario_group=None,
            parameters=[
                ComponentParameterSchema(
                    id="nominal_capacity",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                ),
                ComponentParameterSchema(
                    id="num_units",
                    time_dependent=False,
                    scenario_dependent=False,
                    value=1.0,
                ),
                ComponentParameterSchema(
                    id="available_power",
                    time_dependent=True,
                    scenario_dependent=True,
                    value="available_power_fr_run_of_river",
                ),
            ],
            properties=[
                ComponentPropertySchema(id="carrier", value="electricity"),
                ComponentPropertySchema(id="technology", value="run_of_river"),
            ],
        )
        assert ror_fr_connection == expected_ror_connection
        assert ror_fr_component == expected_ror_component

    @pytest.mark.parametrize(
        "fr_ror",
        [
            pd.DataFrame(),  # DataFrame empty
        ],
        indirect=True,
    )
    def test_convert_ror_to_component_empty_file(self, fr_ror: object):
        converter = self._init_converter_from_study(fr_ror, model_list=[])

        path_load = RESOURCES_FOLDER / "ror.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            ror_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        assert ror_components == []

    @pytest.mark.parametrize(
        "fr_ror",
        [
            pd.DataFrame([0, 0, 0]),  # DataFrame full of 0
        ],
        indirect=True,
    )
    def test_convert_ror_to_component_zero_values(self, fr_ror: object):
        converter = self._init_converter_from_study(fr_ror, model_list=[])

        path_load = RESOURCES_FOLDER / "ror.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            ror_components,
            _,
            _,
        ) = converter._convert_model_to_component_list(resource_content)
        assert ror_components == []

    def test_convert_links_to_component(self, local_study_w_links: Study, lib_id: str):
        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_study(local_study_w_links, model_list=[])
        path_load = RESOURCES_FOLDER / "link.yaml"

        with path_load.open() as template:
            resource_content = parse_conversion_template(template)

        (
            links_components,
            links_connections,
            _,
        ) = converter._convert_model_to_component_list(resource_content)

        fr_it_direct_links_timeseries = "direct_capacity_fr_it_link"
        fr_it_indirect_links_timeseries = "indirect_capacity_fr_it_link"
        fr_it_direct_costs_timeseries = "direct_hurdle_cost_fr_it_link"
        fr_it_indirect_costs_timeseries = "indirect_hurdle_cost_fr_it_link"
        at_fr_direct_links_timeseries = "direct_capacity_at_fr_link"
        at_fr_indirect_links_timeseries = "indirect_capacity_at_fr_link"
        at_it_direct_links_timeseries = "direct_capacity_at_it_link"
        at_it_indirect_links_timeseries = "indirect_capacity_at_it_link"
        at_fr_direct_costs_timeseries = "direct_hurdle_cost_at_fr_link"
        at_fr_indirect_costs_timeseries = "indirect_hurdle_cost_at_fr_link"
        at_it_direct_costs_timeseries = "direct_hurdle_cost_at_it_link"
        at_it_indirect_costs_timeseries = "indirect_hurdle_cost_at_it_link"
        fr_it_loop_flow_timeseries = "loop_flow_fr_it_link"
        at_fr_loop_flow_timeseries = "loop_flow_at_fr_link"
        at_it_loop_flow_timeseries = "loop_flow_at_it_link"
        expected_link_component = [
            ComponentSchema(
                id="fr_it_link",
                model="antares_legacy_models.link",
                scenario_group=None,
                parameters=[
                    ComponentParameterSchema(
                        id="direct_capacity",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{fr_it_direct_links_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="indirect_capacity",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{fr_it_indirect_links_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="direct_hurdle_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{fr_it_direct_costs_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="indirect_hurdle_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{fr_it_indirect_costs_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="loop_flow",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{fr_it_loop_flow_timeseries}",
                    ),
                ],
                properties=[
                    ComponentPropertySchema(id="carrier", value="electricity"),
                ],
            ),
            ComponentSchema(
                id="at_fr_link",
                model="antares_legacy_models.link",
                scenario_group=None,
                parameters=[
                    ComponentParameterSchema(
                        id="direct_capacity",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_fr_direct_links_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="indirect_capacity",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_fr_indirect_links_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="direct_hurdle_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_fr_direct_costs_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="indirect_hurdle_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_fr_indirect_costs_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="loop_flow",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_fr_loop_flow_timeseries}",
                    ),
                ],
                properties=[
                    ComponentPropertySchema(id="carrier", value="electricity"),
                ],
            ),
            ComponentSchema(
                id="at_it_link",
                model="antares_legacy_models.link",
                scenario_group=None,
                parameters=[
                    ComponentParameterSchema(
                        id="direct_capacity",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_it_direct_links_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="indirect_capacity",
                        time_dependent=True,
                        scenario_dependent=True,
                        scenario_group=None,
                        value=f"{at_it_indirect_links_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="direct_hurdle_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_it_direct_costs_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="indirect_hurdle_cost",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_it_indirect_costs_timeseries}",
                    ),
                    ComponentParameterSchema(
                        id="loop_flow",
                        time_dependent=True,
                        scenario_dependent=False,
                        scenario_group=None,
                        value=f"{at_it_loop_flow_timeseries}",
                    ),
                ],
                properties=[
                    ComponentPropertySchema(id="carrier", value="electricity"),
                ],
            ),
        ]

        expected_link_connections = [
            PortConnectionsSchema(
                component1="at_fr_link",
                port1="in_port",
                component2="at_node",
                port2="balance_port",
            ),
            PortConnectionsSchema(
                component1="at_fr_link",
                port1="out_port",
                component2="fr_node",
                port2="balance_port",
            ),
            PortConnectionsSchema(
                component1="at_it_link",
                port1="in_port",
                component2="at_node",
                port2="balance_port",
            ),
            PortConnectionsSchema(
                component1="at_it_link",
                port1="out_port",
                component2="it_node",
                port2="balance_port",
            ),
            PortConnectionsSchema(
                component1="fr_it_link",
                port1="in_port",
                component2="fr_node",
                port2="balance_port",
            ),
            PortConnectionsSchema(
                component1="fr_it_link",
                port1="out_port",
                component2="it_node",
                port2="balance_port",
            ),
        ]
        assert sorted(links_components, key=lambda x: x.id) == sorted(
            expected_link_component, key=lambda x: x.id
        )
        assert links_connections == expected_link_connections

    @staticmethod
    def _match_area_pattern(object, param_values: dict[str, str], pattern: str) -> any:
        if isinstance(object, dict):
            return {
                TestConverter._match_area_pattern(
                    k, param_values, pattern
                ): TestConverter._match_area_pattern(v, param_values, pattern)
                for k, v in object.items()
            }
        elif isinstance(object, list):
            return [
                TestConverter._match_area_pattern(elem, param_values, pattern)
                for elem in object
            ]
        elif isinstance(object, str):
            return object.replace(pattern, param_values)
        else:
            return object

    def test_convert_binding_constraints_to_component(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH

        output_path = local_path / "reference.yaml"
        with open(output_path) as system_file:
            expected_data = parse_yaml_components(system_file)

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        # This test is on the inner function _convert_model_to_component_list, no need to pass a model_list to the converter constructor
        converter = self._init_converter_from_path(
            input_path, output_path, "full", model_list=[]
        )
        path_cc = RESOURCES_FOLDER / "battery.yaml"

        with path_cc.open() as template:
            bc_data = parse_conversion_template(template)
        (
            binding_components,
            binding_connections,
            area_connections,
        ) = converter._convert_model_to_component_list(
            bc_data, bc_data.get_excluded_objects_ids()
        )  # Bad design, either the test should call a higher level function, or virtual objects should be deduced from single model

        assert area_connections == []
        assert binding_connections == [
            c for c in expected_data.connections if c.component1 == "fr_battery"
        ]
        assert binding_components == [
            c for c in expected_data.components if c.id == "fr_battery"
        ]
        # TODO enrich

    def test_hybrid_data_series_presence(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH
        lib_paths: list = LIB_PATHS
        model_list: list = "battery"

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        converter = self._init_converter_from_path(
            input_path, output_path, "hybrid", lib_paths, model_list
        )
        path_cc = (
            Path(__file__).parent.parent.parent
            / "src"
            / "antares_gems_converter"
            / "input_converter"
            / "data"
            / "model_configuration"
            / "battery.yaml"
        )

        with path_cc.open() as template:
            bc_data = parse_conversion_template(template)

        (
            _,
            _,
            area_connections,
        ) = converter._convert_model_to_component_list(bc_data)

        output_path = converter.output_folder
        path1 = output_path / "input" / "data-series" / "marginal_cost_fr_battery.tsv"
        path2 = (
            output_path
            / "input"
            / "data-series"
            / "p_max_injection_modulation_fr_battery.tsv"
        )
        path3 = (
            output_path
            / "input"
            / "data-series"
            / "p_max_withdrawal_modulation_fr_battery.tsv"
        )
        path4 = (
            output_path / "input" / "data-series" / "upper_rule_curve_fr_battery.tsv"
        )
        assert check_file_exists(path1)
        assert check_file_exists(path2)
        assert check_file_exists(path3)
        assert check_file_exists(path4)
        ### Compare area connections
        expected_area_connections = [
            AreaConnectionsSchema(
                component="fr_battery", port="injection_port", area="fr"
            )
        ]
        assert area_connections == expected_area_connections

    def test_hybrid_convert_study_path_to_input_study(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH

        output_path = local_path / "reference_hybrid.yaml"
        with open(output_path) as system_file:
            expected_data = parse_yaml_components(system_file)

        model_list: list = ["battery"]

        input_path = tmp_path / "input" / LOCAL_PATH
        output_path = tmp_path / "output" / LOCAL_PATH
        shutil.copytree(local_path, input_path)

        converter = self._init_converter_from_path(
            input_path, output_path, "hybrid", MODEL_LIST_WITH_BASE, model_list
        )
        thermal_cluster_filepath = (
            converter.output_folder
            / "input"
            / "thermal"
            / "clusters"
            / "z_batteries"
            / "list.ini"
        )
        bc_filepath = (
            converter.output_folder
            / "input"
            / "bindingconstraints"
            / "bindingconstraints.ini"
        )
        links_filepath = (
            converter.output_folder / "input" / "links" / "fr" / "properties.ini"
        )
        assert thermal_cluster_filepath.stat().st_size > 0
        assert bc_filepath.stat().st_size > 0
        assert links_filepath.stat().st_size > 0
        obtained_data = converter.convert_study_to_input_system()

        # Check files have been correctly deleted
        thermal_cluster_filepath = (
            converter.output_folder
            / "input"
            / "thermal"
            / "clusters"
            / "z_batteries"
            / "list.ini"
        )
        bc_filepath = (
            converter.output_folder
            / "input"
            / "bindingconstraints"
            / "bindingconstraints.ini"
        )
        links_filepath = (
            converter.output_folder / "input" / "links" / "fr" / "properties.ini"
        )
        assert thermal_cluster_filepath.stat().st_size == 0
        assert bc_filepath.stat().st_size == 0
        assert links_filepath.stat().st_size == 0
        # TODO check folder data-models is present

        assert obtained_data.components == expected_data.components

    def test_convert_study_path_to_input_study(self, tmp_path: Path):
        local_path = Path(__file__).parent / "resources" / LOCAL_PATH
        ref_path = local_path / "reference.yaml"
        input_path = tmp_path / "input"
        output_path = tmp_path / "output"
        shutil.copytree(local_path, input_path)
        converter = self._init_converter_from_path(
            input_path,
            output_path,
            "full",
            MODEL_LIST_WITH_BASE,
            model_list=[
                model for model in MODEL_NAME_TO_FILE_NAME.keys() if model != "hydro"
            ],
        )
        obtained_sys = converter.convert_study_to_input_system()
        with open(ref_path) as system_file:
            expected_sys = parse_yaml_components(system_file)
        assert obtained_sys.components == expected_sys.components

    def test_multiply_operation(self):
        operation = Operation(multiply_by=2)
        assert operation.execute(10) == 20

        operation = Operation(multiply_by="factor")
        preprocessed_values = {"factor": 5}
        assert operation.execute(10, preprocessed_values) == 50

        operation = Operation(multiply_by=2)
        df = pd.Series([1, 2, 3, 4, 5, 6])
        assert operation.execute(df).all() == pd.Series([2, 4, 6, 8, 10, 12]).all()

    def test_divide_operation(self):
        operation = Operation(divide_by=2)
        assert operation.execute(10) == 5

        operation = Operation(divide_by="divisor")
        preprocessed_values = {"divisor": 2}
        assert operation.execute(10, preprocessed_values) == 5

        operation = Operation(divide_by=2)
        df = pd.Series([1, 2, 3, 4, 5, 6])
        assert operation.execute(df).all() == pd.Series([0.5, 1, 1.5, 2, 2.5, 3]).all()

    def test_max_operation(self):
        operation = Operation(type="max")
        assert operation.execute([1, 2, 3, 4, 5]) == 5.0

        df = pd.Series([1, 2, 3, 4, 5, 6])
        assert operation.execute(df) == 6.0

    def test_missing_preprocessed_value(self):
        operation = Operation(multiply_by="missing_key")
        with pytest.raises(ValueError):
            operation.execute(10, {})

    def test_missing_operation(self):
        operation = Operation()
        with pytest.raises(ValueError):
            operation.execute(10)
