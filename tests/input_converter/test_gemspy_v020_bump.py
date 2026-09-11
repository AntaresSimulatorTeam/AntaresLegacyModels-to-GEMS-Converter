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
"""
Regression tests for the gemspy v0.2.0 bump (converter v0.4.0).

What changed in v0.2.0:
  - ComponentSchema gained a top-level `scenario_group` field.
  - Templates now declare `scenario-group` per component (not at template level).
  - The converter reads the legacy Antares ScenarioBuilder and propagates per-component
    scenario groups to the output SystemSchema.
  - A `modeler-scenariobuilder.dat` file is generated when SB entries exist.
  - gems_craft_hybrid provides HybridSystemSchema for hybrid-mode output.

These tests exercise the gemspy v0.2.0 public API end-to-end:
  - parse_yaml_system / parse_yaml_library / resolve_library / resolve_system
  - ComponentSchema.scenario_group round-trips through YAML
  - HybridSystemSchema is accepted by parse_yaml_system
"""
import os
from pathlib import Path

import pytest
from antares.craft.model.area import AreaProperties
from antares.craft.model.hydro import HydroProperties
from antares.craft.model.settings.general import GeneralParametersUpdate
from antares.craft.model.settings.study_settings import StudySettingsUpdate
from antares.craft.model.study import Study, create_study_local
from antares.craft.model.thermal import ThermalClusterProperties
from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study.parsing import (
    ComponentParameterSchema,
    ComponentPropertySchema,
    ComponentSchema,
    SystemSchema,
    parse_yaml_system,
)
from gems_craft.study.resolve_components import resolve_system
from gems_craft_hybrid.study.parsing import AreaConnectionsSchema, HybridSystemSchema

from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter
from antares_gems_converter.input_converter.src.logger import Logger
from antares_gems_converter.input_converter.src.utils import dump_to_yaml
from tests.input_converter.conftest import create_dataframe_from_constant

LIB_PATHS = [
    "src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml",
    "src/antares_gems_converter/libs/reference_models/andromede_v1_models.yml",
]
LIB_PATHS_ABS = [str(Path(os.getcwd()) / p) for p in LIB_PATHS]


def _load_libraries() -> dict:
    """Load both GEMS model libraries via the gemspy v0.2.0 API."""
    lib_schemas = []
    for path in LIB_PATHS:
        with open(path) as f:
            lib_schemas.append(parse_yaml_library(f))
    return resolve_library(lib_schemas)


@pytest.fixture
def two_area_study_with_data(tmp_path) -> Study:
    """
    Study with 2 areas (fr, de), 1 link, thermal cluster, hydro reservoir,
    load/wind timeseries — covers the main component types converted by the
    legacy→GEMS pipeline.
    """
    study = create_study_local("bump_test_study", "930", tmp_path)
    for area_id in ["fr", "de"]:
        study.create_area(
            area_id,
            properties=AreaProperties(
                energy_cost_spilled="1", energy_cost_unsupplied="0.5"
            ),
        )

    # Link fr↔de (sorted alphabetically → "de / fr")
    study.create_link(area_from="de", area_to="fr")

    # nb_years = 3 so SB indices 0, 1, 2 are valid
    study.update_settings(
        StudySettingsUpdate(general_parameters=GeneralParametersUpdate(nb_years=3))
    )

    # Thermal cluster on fr
    study.get_areas()["fr"].create_thermal_cluster(
        "gas_fr", ThermalClusterProperties(unit_count=1, nominal_capacity=100.0)
    )
    # Thermal series (8760 rows, 1 column) + modulation (8760 rows, 4 columns)
    study.get_areas()["fr"].get_thermals()["gas_fr"].set_series(
        create_dataframe_from_constant(lines=8760)
    )
    study.get_areas()["fr"].get_thermals()["gas_fr"].set_prepro_modulation(
        create_dataframe_from_constant(lines=8760, columns=4)
    )

    # Hydro reservoir on both areas
    hydro_props = HydroProperties(
        reservoir=True,
        reservoir_capacity=500,
        pumping_efficiency=0.75,
        overflow_spilled_cost_difference=0,
    )
    for area_id in ["fr", "de"]:
        study.get_areas()[area_id].hydro.update_properties(hydro_props)

    # Load timeseries on fr (required for the load component to be emitted)
    study.get_areas()["fr"].set_load(create_dataframe_from_constant(lines=8760))

    # Wind timeseries on fr
    study.get_areas()["fr"].set_wind(create_dataframe_from_constant(lines=8760))

    return study


def _make_converter(
    study: Study, model_list: list[str], mode: str = "full"
) -> AntaresStudyConverter:
    logger = Logger(__name__, study.path)
    return AntaresStudyConverter(
        study_input=study,
        logger=logger,
        mode=mode,
        lib_paths=LIB_PATHS,
        models_to_convert=model_list,
        output_folder=study.path.parent / "converter_output",
    )


# ---------------------------------------------------------------------------
# 1. gemspy v0.2.0 library loading — API smoke test
# ---------------------------------------------------------------------------


class TestGemsPyV020LibraryAPI:
    """Verify that the gemspy v0.2.0 parse/resolve library API accepts our YAML libs."""

    def test_parse_yaml_library_both_libs(self):
        for path in LIB_PATHS:
            with open(path) as f:
                lib_schema = parse_yaml_library(f)
            assert lib_schema is not None, f"parse_yaml_library returned None for {path}"

    def test_resolve_library_returns_dict_keyed_by_id(self):
        libs = _load_libraries()
        # Library IDs come from the 'library.id' field in each YAML file
        assert "antares_legacy_models" in libs
        # andromede library uses hyphens in its id
        assert "andromede-v1-models" in libs

    def test_antares_legacy_models_has_expected_model_ids(self):
        libs = _load_libraries()
        legacy = libs["antares_legacy_models"]
        # Model keys in the resolved library are fully qualified: "<lib_id>.<model_id>"
        expected_models = {
            "antares_legacy_models.area",
            "antares_legacy_models.thermal",
            "antares_legacy_models.renewable",
            "antares_legacy_models.link",
            "antares_legacy_models.short_term_storage",
        }
        for model_id in expected_models:
            assert model_id in legacy.models, f"Model '{model_id}' missing from library"


# ---------------------------------------------------------------------------
# 2. ComponentSchema.scenario_group — new field in gemspy v0.2.0
# ---------------------------------------------------------------------------


class TestComponentSchemaScenarioGroup:
    """
    scenario_group is the key new field on ComponentSchema in gemspy v0.2.0.
    When the legacy Antares ScenarioBuilder has entries for a component type, the
    converted component must carry a non-None scenario_group; when the SB is empty
    it must be None.
    """

    def test_no_sb_all_scenario_groups_none(self, two_area_study_with_data: Study):
        converter = _make_converter(
            two_area_study_with_data, model_list=["wind", "thermal", "link"]
        )
        system = converter.convert_study_to_input_system()

        for comp in system.components:
            assert comp.scenario_group is None, (
                f"Component '{comp.id}' must have scenario_group=None when legacy SB "
                f"is empty, got '{comp.scenario_group}'"
            )

    def test_sb_wind_sets_scenario_group_on_wind_components(
        self, two_area_study_with_data: Study
    ):
        sb = two_area_study_with_data.get_scenario_builder()
        sb.wind.get_area("fr").set_new_scenario([2])
        two_area_study_with_data.set_scenario_builder(sb)

        converter = _make_converter(two_area_study_with_data, model_list=["wind"])
        system = converter.convert_study_to_input_system()

        wind_fr = [c for c in system.components if c.id == "fr_wind"]
        assert wind_fr, "Expected a fr_wind component"
        assert wind_fr[0].scenario_group == "wind_fr_group"

    def test_sb_thermal_sets_scenario_group_on_thermal_components(
        self, two_area_study_with_data: Study
    ):
        sb = two_area_study_with_data.get_scenario_builder()
        sb.thermal.get_cluster("fr", "gas_fr").set_new_scenario([1])
        two_area_study_with_data.set_scenario_builder(sb)

        converter = _make_converter(two_area_study_with_data, model_list=["thermal"])
        system = converter.convert_study_to_input_system()

        thermal_fr = [c for c in system.components if c.id == "fr_thermal_gas_fr"]
        assert thermal_fr, "Expected a fr_thermal_gas_fr component"
        assert thermal_fr[0].scenario_group == "thermal_fr_gas_fr_group"

    def test_sb_hydro_sets_scenario_group_only_on_matching_area(
        self, two_area_study_with_data: Study
    ):
        # Only fr gets a hydro SB entry; de must remain None
        sb = two_area_study_with_data.get_scenario_builder()
        sb.hydro.get_area("fr").set_new_scenario([3])
        two_area_study_with_data.set_scenario_builder(sb)

        converter = _make_converter(two_area_study_with_data, model_list=["hydro"])
        system = converter.convert_study_to_input_system()

        comp_by_id = {c.id: c for c in system.components}
        assert comp_by_id["fr_hydro_storage"].scenario_group == "hydro_inflows_fr_group"
        assert comp_by_id["de_hydro_storage"].scenario_group is None

    def test_sb_link_normalized_slash_to_underscore(
        self, two_area_study_with_data: Study
    ):
        # Link ID sorted alphabetically: create_link(de, fr) → "de / fr"
        # The converter must normalize " / " → "_" in the group name
        sb = two_area_study_with_data.get_scenario_builder()
        sb.link.get_link("de / fr").set_new_scenario([5])
        two_area_study_with_data.set_scenario_builder(sb)

        converter = _make_converter(two_area_study_with_data, model_list=["link"])
        system = converter.convert_study_to_input_system()

        link_comp = [c for c in system.components if c.id == "de_fr_link"]
        assert link_comp, "Expected a de_fr_link component"
        assert link_comp[0].scenario_group == "de_fr_ntc_group"


# ---------------------------------------------------------------------------
# 3. YAML round-trip with scenario_group — gemspy v0.2.0 parse_yaml_system
# ---------------------------------------------------------------------------


class TestScenarioGroupYAMLRoundtrip:
    """
    Verify that scenario_group values survive a dump_to_yaml → parse_yaml_system round-trip.
    This exercises the gemspy v0.2.0 YAML serialisation/deserialisation contract.

    SystemSchema objects are built directly (not via convert_study_to_input_system) to
    isolate the gemspy v0.2.0 API from the converter's internal serialisation path.
    """

    def test_scenario_group_survives_yaml_roundtrip(self, tmp_path: Path):
        # Build a minimal SystemSchema by hand so only the gemspy API is under test
        components = [
            ComponentSchema(
                id="fr_wind",
                model="antares_legacy_models.renewable",
                scenario_group="wind_fr_group",
                parameters=[
                    ComponentParameterSchema(
                        id="nominal_capacity",
                        time_dependent=False,
                        scenario_dependent=False,
                        value=100.0,
                    )
                ],
                properties=[ComponentPropertySchema(id="carrier", value="electricity")],
            ),
            ComponentSchema(
                id="fr_thermal_gas_fr",
                model="antares_legacy_models.thermal",
                scenario_group="thermal_fr_gas_fr_group",
                properties=[ComponentPropertySchema(id="carrier", value="electricity")],
            ),
        ]
        system = SystemSchema(id="bump_test", components=components)

        yaml_path = tmp_path / "system_roundtrip.yml"
        dump_to_yaml(model=system, output_path=yaml_path)

        with open(yaml_path) as f:
            reloaded = parse_yaml_system(f)

        wind_fr = next((c for c in reloaded.components if c.id == "fr_wind"), None)
        assert wind_fr is not None
        assert wind_fr.scenario_group == "wind_fr_group"

        thermal_fr = next(
            (c for c in reloaded.components if c.id == "fr_thermal_gas_fr"), None
        )
        assert thermal_fr is not None
        assert thermal_fr.scenario_group == "thermal_fr_gas_fr_group"

    def test_none_scenario_group_not_written_to_yaml(self, tmp_path: Path):
        # scenario_group=None must not appear in the YAML output
        components = [
            ComponentSchema(
                id="fr_wind",
                model="antares_legacy_models.renewable",
                # scenario_group deliberately omitted → defaults to None
                properties=[ComponentPropertySchema(id="carrier", value="electricity")],
            )
        ]
        system = SystemSchema(id="bump_test", components=components)

        yaml_path = tmp_path / "no_sb_system.yml"
        dump_to_yaml(model=system, output_path=yaml_path)

        content = yaml_path.read_text()
        assert "scenario-group" not in content, (
            "scenario-group must not appear in YAML when scenario_group is None"
        )


# ---------------------------------------------------------------------------
# 4. resolve_system — gemspy v0.2.0 end-to-end model resolution
# ---------------------------------------------------------------------------


class TestResolveSystemV020:
    """
    resolve_system validates that each ComponentSchema references a model that exists
    in the loaded libraries. These tests confirm that the converted output is accepted
    by gemspy v0.2.0's resolver, catching any model-id / parameter-id mismatches.
    """

    def test_full_conversion_resolves_against_library(
        self, two_area_study_with_data: Study
    ):
        converter = _make_converter(
            two_area_study_with_data,
            model_list=["thermal", "link", "hydro", "wind", "load"],
        )
        system = converter.convert_study_to_input_system()

        libs = _load_libraries()
        # resolve_system raises on unknown model ids or parameter mismatches
        resolved = resolve_system(system, libs)
        assert resolved is not None

    def test_area_only_conversion_resolves(self, two_area_study_with_data: Study):
        converter = _make_converter(two_area_study_with_data, model_list=["area"])
        system = converter.convert_study_to_input_system()

        libs = _load_libraries()
        resolved = resolve_system(system, libs)
        assert resolved is not None

    def test_system_with_scenario_groups_resolves(
        self, two_area_study_with_data: Study
    ):
        # scenario_group is metadata — resolve_system must accept it without error
        sb = two_area_study_with_data.get_scenario_builder()
        sb.wind.get_area("fr").set_new_scenario([2])
        sb.hydro.get_area("fr").set_new_scenario([3])
        sb.link.get_link("de / fr").set_new_scenario([1])
        two_area_study_with_data.set_scenario_builder(sb)

        converter = _make_converter(
            two_area_study_with_data, model_list=["wind", "hydro", "link"]
        )
        system = converter.convert_study_to_input_system()

        libs = _load_libraries()
        resolved = resolve_system(system, libs)
        assert resolved is not None


# ---------------------------------------------------------------------------
# 5. HybridSystemSchema — gems_craft_hybrid API introduced in gemspy v0.2.0
# ---------------------------------------------------------------------------


class TestHybridSystemSchemaV020:
    """
    HybridSystemSchema lives in gems_craft_hybrid (split from gems_craft in v0.2.0).
    Verify that a hybrid-mode conversion produces output accepted by parse_yaml_system
    with the HybridSystemSchema class.
    """

    def test_hybrid_conversion_produces_hybrid_system(
        self, two_area_study_with_data: Study, tmp_path: Path
    ):
        converter = _make_converter(
            two_area_study_with_data, model_list=["wind"], mode="hybrid"
        )
        hybrid_system = converter.convert_study_to_input_system()

        # HybridSystemSchema is a subclass/variant; the system must be an instance of it
        assert isinstance(hybrid_system, HybridSystemSchema), (
            f"Hybrid-mode conversion must return HybridSystemSchema, "
            f"got {type(hybrid_system).__name__}"
        )

    def test_hybrid_yaml_parses_as_hybrid_system_schema(self, tmp_path: Path):
        # Build a minimal HybridSystemSchema by hand to isolate the gemspy v0.2.0 API
        components = [
            ComponentSchema(
                id="fr_wind",
                model="antares_legacy_models.renewable",
                scenario_group="wind_fr_group",
                properties=[ComponentPropertySchema(id="carrier", value="electricity")],
            )
        ]
        area_connections = [
            AreaConnectionsSchema(
                component="fr_wind", port="balance_port", area="fr"
            )
        ]
        system = HybridSystemSchema(
            id="bump_test",
            components=components,
            area_connections=area_connections,
        )

        yaml_path = tmp_path / "hybrid_system.yml"
        dump_to_yaml(model=system, output_path=yaml_path)

        with open(yaml_path) as f:
            reloaded = parse_yaml_system(f, HybridSystemSchema)

        assert reloaded is not None
        assert isinstance(reloaded, HybridSystemSchema)

        wind = next((c for c in reloaded.components if c.id == "fr_wind"), None)
        assert wind is not None
        assert wind.scenario_group == "wind_fr_group"

    def test_hybrid_wind_with_sb_produces_scenario_group_and_dat_file(
        self, two_area_study_with_data: Study
    ):
        sb = two_area_study_with_data.get_scenario_builder()
        sb.wind.get_area("fr").set_new_scenario([2])
        two_area_study_with_data.set_scenario_builder(sb)

        converter = _make_converter(
            two_area_study_with_data, model_list=["wind"], mode="hybrid"
        )
        system = converter.convert_study_to_input_system()
        converter.process_all()

        # The GEMS side: wind component must carry its scenario group
        wind_comps = [c for c in system.components if "wind" in c.id]
        assert wind_comps, "Expected at least one wind component in hybrid output"
        assert all(c.scenario_group == "wind_fr_group" for c in wind_comps)

        # The modeler-scenariobuilder.dat must be present
        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert sb_files, "modeler-scenariobuilder.dat must be generated in hybrid mode"
        content = sb_files[0].read_text()
        assert "wind_fr_group, 0 = 2" in content

        # The legacy study must have its wind SB entry cleared
        hybrid_sb = converter.study.get_scenario_builder()
        fr_wind_scenario = hybrid_sb.wind.get_area("fr").get_scenario()
        assert all(ts is None for ts in fr_wind_scenario), (
            "Legacy SB wind entry for 'fr' must be cleared after hybrid conversion"
        )
