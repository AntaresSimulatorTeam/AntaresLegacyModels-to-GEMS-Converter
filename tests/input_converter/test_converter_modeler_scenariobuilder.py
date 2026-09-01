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
import pytest
from antares.craft.model.area import AreaProperties
from antares.craft.model.hydro import HydroProperties
from antares.craft.model.settings.general import GeneralParametersUpdate
from antares.craft.model.settings.study_settings import StudySettingsUpdate
from antares.craft.model.study import Study, create_study_local

from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter
from antares_gems_converter.input_converter.src.logger import Logger
from tests.input_converter.conftest import create_dataframe_from_constant

LIB_PATHS = [
    "src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml",
    "src/antares_gems_converter/libs/reference_models/andromede_v1_models.yml",
]


class TestConverterScenarioBuilder:
    def _init_converter_from_study(
        self,
        local_study: Study,
        model_list: list,
        mode: str = "full",
    ) -> AntaresStudyConverter:
        logger = Logger(__name__, local_study.path)
        return AntaresStudyConverter(
            study_input=local_study,
            logger=logger,
            mode=mode,
            lib_paths=LIB_PATHS,
            models_to_convert=model_list,
            output_folder=local_study.path.parent / "converter_output",
        )

    # -------------------------------------------------------------------------
    # Case 1 — legacy SB empty: no scenario_group on components, no SB file
    # -------------------------------------------------------------------------

    def test_empty_legacy_sb_no_scenario_builder(self, fr_wind: Study):
        converter = self._init_converter_from_study(fr_wind, model_list=["wind"])
        system = converter.convert_study_to_input_system()

        assert system.components, "Expected at least one component"
        assert all(c.scenario_group is None for c in system.components), (
            "No component must have a scenario_group when the legacy SB is empty"
        )

        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert not sb_files, (
            "No modeler-scenariobuilder.dat should be generated when the legacy SB is empty"
        )

    # -------------------------------------------------------------------------
    # Case 2 — legacy SB full: scenario_group set on components, SB file generated
    # -------------------------------------------------------------------------

    def test_full_legacy_wind_and_thermal_sb(self, fr_wind: Study):
        sb = fr_wind.get_scenario_builder()
        sb.wind.get_area("fr").set_new_scenario([2])
        sb.thermal.get_cluster("fr", "gaz").set_new_scenario([3])
        fr_wind.set_scenario_builder(sb)

        converter = self._init_converter_from_study(fr_wind, model_list=["wind", "thermal"])
        system = converter.convert_study_to_input_system()

        wind_components = [c for c in system.components if "wind" in c.id]
        assert wind_components, "Expected at least one wind component"
        assert all(c.scenario_group == "wind_fr_group" for c in wind_components), (
            "Wind components must carry scenario_group='wind_fr_group'"
        )

        thermal_components = [c for c in system.components if "gaz" in c.id]
        assert thermal_components, "Expected at least one thermal component"
        assert all(c.scenario_group == "thermal_fr_gaz_group" for c in thermal_components), (
            "Thermal components must carry scenario_group='thermal_fr_gaz_group'"
        )

        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert sb_files, "modeler-scenariobuilder.dat must be generated"

        content = sb_files[0].read_text()
        assert "wind_fr_group, 0 = 2" in content, (
            f"Expected 'wind_fr_group, 0 = 2' in generated SB file, got:\n{content}"
        )
        assert "thermal_fr_gaz_group, 0 = 3" in content, (
            f"Expected 'thermal_fr_gaz_group, 0 = 3' in generated SB file, got:\n{content}"
        )

    # -------------------------------------------------------------------------
    # Case 3 — hybrid mode: legacy SB cleared for converted areas, modeler SB
    # still generated for the GEMS system
    # -------------------------------------------------------------------------

    def test_hybrid_legacy_sb_cleared_after_conversion(self, fr_wind: Study):
        sb = fr_wind.get_scenario_builder()
        sb.wind.get_area("fr").set_new_scenario([2])
        sb.thermal.get_cluster("fr", "gaz").set_new_scenario([3])
        fr_wind.set_scenario_builder(sb)

        converter = self._init_converter_from_study(
            fr_wind, model_list=["wind", "thermal"], mode="hybrid"
        )
        converter.process_all()

        hybrid_sb = converter.study.get_scenario_builder()
        fr_scenario = hybrid_sb.wind.get_area("fr").get_scenario()
        assert all(ts is None for ts in fr_scenario), (
            "Legacy SB wind entries for 'fr' must be cleared after hybrid conversion"
        )

        assert "gaz" not in converter.study.get_areas()["fr"].get_thermals(), (
            "Thermal cluster 'gaz' must be deleted from the study after hybrid conversion"
        )

    def test_hybrid__sb_file_generated(self, fr_wind: Study):
        sb = fr_wind.get_scenario_builder()
        sb.wind.get_area("fr").set_new_scenario([2])
        fr_wind.set_scenario_builder(sb)

        converter = self._init_converter_from_study(fr_wind, model_list=["wind"], mode="hybrid")
        system = converter.convert_study_to_input_system()
        converter.process_all()

        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert sb_files, "modeler-scenariobuilder.dat must still be generated in hybrid mode"
        content = sb_files[0].read_text()
        assert "wind_fr_group, 0 = 2" in content, (
            f"Expected 'wind_fr_group, 0 = 2' in generated SB file, got:\n{content}"
        )

        wind_components = [c for c in system.components if "wind" in c.id]
        assert wind_components, "Expected at least one wind component in hybrid mode"
        assert all(c.scenario_group == "wind_fr_group" for c in wind_components), (
            "Wind components must carry scenario_group in hybrid mode when legacy SB has entries"
        )

    # -------------------------------------------------------------------------
    # Case 5 — link SB: NTC entries produce a scenario group per link
    # -------------------------------------------------------------------------

    def test_full_legacy_link_sb(self, fr_wind: Study):
        sb = fr_wind.get_scenario_builder()
        # Link IDs are sorted alphabetically: create_link(fr, at) → "at / fr"
        sb.link.get_link("at / fr").set_new_scenario([5])
        fr_wind.set_scenario_builder(sb)

        converter = self._init_converter_from_study(fr_wind, model_list=["link"])
        system = converter.convert_study_to_input_system()

        link_components = [c for c in system.components if "at_fr" in c.id]
        assert link_components, "Expected at least one at/fr link component"
        assert all(c.scenario_group == "at_fr_ntc_group" for c in link_components), (
            "Link components must carry scenario_group='at_fr_ntc_group'"
        )

        other_link_components = [c for c in system.components if "at_it" in c.id or "fr_it" in c.id]
        assert all(c.scenario_group is None for c in other_link_components), (
            "Links without SB entries must have scenario_group=None"
        )

        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert sb_files, "modeler-scenariobuilder.dat must be generated"
        content = sb_files[0].read_text()
        assert "at_fr_ntc_group, 0 = 5" in content, (
            f"Expected 'at_fr_ntc_group, 0 = 5' in generated SB file, got:\n{content}"
        )

    # -------------------------------------------------------------------------
    # Case 6 — hydro inflows SB: entries produce a scenario group for hydro
    # -------------------------------------------------------------------------

    def test_full_legacy_hydro_sb(self, fr_wind: Study):
        sb = fr_wind.get_scenario_builder()
        sb.hydro.get_area("fr").set_new_scenario([7])
        fr_wind.set_scenario_builder(sb)

        converter = self._init_converter_from_study(fr_wind, model_list=["hydro"])
        system = converter.convert_study_to_input_system()

        fr_hydro = [c for c in system.components if c.id == "fr_hydro_storage"]
        assert fr_hydro, "Expected a fr_hydro_storage component"
        assert fr_hydro[0].scenario_group == "hydro_inflows_fr_group", (
            "fr_hydro_storage must carry scenario_group='hydro_inflows_fr_group'"
        )

        other_hydro = [c for c in system.components if "hydro_storage" in c.id and c.id != "fr_hydro_storage"]
        assert all(c.scenario_group is None for c in other_hydro), (
            "Hydro components for areas without SB entries must have scenario_group=None"
        )

        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert sb_files, "modeler-scenariobuilder.dat must be generated"
        content = sb_files[0].read_text()
        assert "hydro_inflows_fr_group, 0 = 7" in content, (
            f"Expected 'hydro_inflows_fr_group, 0 = 7' in generated SB file, got:\n{content}"
        )

    # -------------------------------------------------------------------------
    # Case 7 — ROR SB: hydro entries produce a separate group for run-of-river
    # -------------------------------------------------------------------------

    def test_full_legacy_ror_sb(self, fr_wind: Study):
        fr_wind.get_areas()["fr"].hydro.set_ror_series(create_dataframe_from_constant(lines=8760))
        sb = fr_wind.get_scenario_builder()
        sb.hydro.get_area("fr").set_new_scenario([7])
        fr_wind.set_scenario_builder(sb)

        converter = self._init_converter_from_study(fr_wind, model_list=["ror"])
        system = converter.convert_study_to_input_system()

        fr_ror = [c for c in system.components if c.id == "fr_run_of_river"]
        assert fr_ror, "Expected a fr_run_of_river component"
        assert fr_ror[0].scenario_group == "ror_fr_group", (
            "fr_run_of_river must carry scenario_group='ror_fr_group'"
        )

        other_ror = [c for c in system.components if "run_of_river" in c.id and c.id != "fr_run_of_river"]
        assert all(c.scenario_group is None for c in other_ror), (
            "ROR components for areas without SB entries must have scenario_group=None"
        )

        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert sb_files, "modeler-scenariobuilder.dat must be generated"
        content = sb_files[0].read_text()
        assert "ror_fr_group, 0 = 7" in content, (
            f"Expected 'ror_fr_group, 0 = 7' in generated SB file, got:\n{content}"
        )


# =============================================================================
# Two-area scenario builder combinations (ba00 / hr00)
# =============================================================================

class TestScenarioBuilderTwoAreas:
    """Tests covering different combinations of legacy SB entries across 2 areas (ba00, hr00)
    and their link, to verify that each component receives the correct scenario_group and
    that modeler-scenariobuilder.dat is generated with the right content.
    """

    @pytest.fixture
    def two_area_study(self, tmp_path):
        """Study with 2 areas (ba00, hr00), hydro reservoir enabled, 1 link, nb_years=3."""
        study = create_study_local("test_two_areas", "930", tmp_path)
        for area_id in ["ba00", "hr00"]:
            study.create_area(
                area_id,
                properties=AreaProperties(energy_cost_spilled="1", energy_cost_unsupplied="0.5"),
            )
        hydro_props = HydroProperties(
            reservoir=True,
            reservoir_capacity=1000,
            pumping_efficiency=0.75,
            overflow_spilled_cost_difference=0,
        )
        for area_id in ["ba00", "hr00"]:
            study.get_areas()[area_id].hydro.update_properties(hydro_props)
        # Link ID sorted alphabetically: create_link(ba00, hr00) → "ba00 / hr00"
        study.create_link(area_from="ba00", area_to="hr00")
        # hr00 uses year indices 0, 1, 2 in some tests → nb_years must cover that
        study.update_settings(
            StudySettingsUpdate(general_parameters=GeneralParametersUpdate(nb_years=3))
        )
        return study

    def _init_converter(self, study: Study, models: list[str]) -> AntaresStudyConverter:
        logger = Logger(__name__, study.path)
        return AntaresStudyConverter(
            study_input=study,
            logger=logger,
            mode="full",
            lib_paths=LIB_PATHS,
            models_to_convert=models,
            output_folder=study.path.parent / "converter_output",
        )

    def _sb_content(self, converter: AntaresStudyConverter) -> str:
        sb_file = converter.output_folder / "input" / "data-series" / "modeler-scenariobuilder.dat"
        return sb_file.read_text() if sb_file.exists() else ""

    # -------------------------------------------------------------------------
    # Case A — only ba00 has a hydro SB entry; hr00 must get scenario_group=None
    # -------------------------------------------------------------------------

    def test_hydro_sb_one_area_only(self, two_area_study):
        sb = two_area_study.get_scenario_builder()
        sb.hydro.get_area("ba00").set_new_scenario([1, 2])
        two_area_study.set_scenario_builder(sb)

        converter = self._init_converter(two_area_study, ["hydro"])
        system = converter.convert_study_to_input_system()
        comp = {c.id: c for c in system.components}

        assert comp["ba00_hydro_storage"].scenario_group == "hydro_inflows_ba00_group"
        assert comp["hr00_hydro_storage"].scenario_group is None

        content = self._sb_content(converter)
        assert "hydro_inflows_ba00_group, 0 = 1" in content
        assert "hydro_inflows_ba00_group, 1 = 2" in content
        assert "hr00" not in content

    # -------------------------------------------------------------------------
    # Case B — both areas have hydro SB entries with different year counts
    # -------------------------------------------------------------------------

    def test_hydro_sb_both_areas(self, two_area_study):
        sb = two_area_study.get_scenario_builder()
        sb.hydro.get_area("ba00").set_new_scenario([1, 2])       # years 0→1, 1→2
        sb.hydro.get_area("hr00").set_new_scenario([2, 1, 3])    # years 0→2, 1→1, 2→3
        two_area_study.set_scenario_builder(sb)

        converter = self._init_converter(two_area_study, ["hydro"])
        system = converter.convert_study_to_input_system()
        comp = {c.id: c for c in system.components}

        assert comp["ba00_hydro_storage"].scenario_group == "hydro_inflows_ba00_group"
        assert comp["hr00_hydro_storage"].scenario_group == "hydro_inflows_hr00_group"

        content = self._sb_content(converter)
        assert "hydro_inflows_ba00_group, 0 = 1" in content
        assert "hydro_inflows_ba00_group, 1 = 2" in content
        assert "hydro_inflows_hr00_group, 0 = 2" in content
        assert "hydro_inflows_hr00_group, 1 = 1" in content
        assert "hydro_inflows_hr00_group, 2 = 3" in content

    # -------------------------------------------------------------------------
    # Case C — only the NTC link has a SB entry; hydro components get None
    # -------------------------------------------------------------------------

    def test_ntc_sb_only(self, two_area_study):
        sb = two_area_study.get_scenario_builder()
        sb.link.get_link("ba00 / hr00").set_new_scenario([1])
        two_area_study.set_scenario_builder(sb)

        converter = self._init_converter(two_area_study, ["link"])
        system = converter.convert_study_to_input_system()
        comp = {c.id: c for c in system.components}

        assert comp["ba00_hr00_link"].scenario_group == "ba00_hr00_ntc_group"

        content = self._sb_content(converter)
        assert "ba00_hr00_ntc_group, 0 = 1" in content

    # -------------------------------------------------------------------------
    # Case D — all combined: hydro for both areas + NTC (mirrors example script)
    # -------------------------------------------------------------------------

    def test_all_combined(self, two_area_study):
        sb = two_area_study.get_scenario_builder()
        sb.hydro.get_area("ba00").set_new_scenario([1, 2])
        sb.hydro.get_area("hr00").set_new_scenario([2, 1, 3])
        sb.link.get_link("ba00 / hr00").set_new_scenario([1])
        two_area_study.set_scenario_builder(sb)

        converter = self._init_converter(two_area_study, ["hydro", "link"])
        system = converter.convert_study_to_input_system()
        comp = {c.id: c for c in system.components}

        assert comp["ba00_hydro_storage"].scenario_group == "hydro_inflows_ba00_group"
        assert comp["hr00_hydro_storage"].scenario_group == "hydro_inflows_hr00_group"
        assert comp["ba00_hr00_link"].scenario_group == "ba00_hr00_ntc_group"

        content = self._sb_content(converter)
        for line in [
            "ba00_hr00_ntc_group, 0 = 1",
            "hydro_inflows_ba00_group, 0 = 1",
            "hydro_inflows_ba00_group, 1 = 2",
            "hydro_inflows_hr00_group, 0 = 2",
            "hydro_inflows_hr00_group, 1 = 1",
            "hydro_inflows_hr00_group, 2 = 3",
        ]:
            assert line in content, f"Missing: '{line}'\nActual:\n{content}"

    # -------------------------------------------------------------------------
    # Case E — no SB at all: all scenario_groups must be None, no file generated
    # -------------------------------------------------------------------------

    def test_no_sb(self, two_area_study):
        converter = self._init_converter(two_area_study, ["hydro", "link"])
        system = converter.convert_study_to_input_system()

        assert all(c.scenario_group is None for c in system.components), (
            "All components must have scenario_group=None when the legacy SB is empty"
        )
        sb_file = converter.output_folder / "input" / "data-series" / "modeler-scenariobuilder.dat"
        assert not sb_file.exists(), "No SB file must be generated when the legacy SB is empty"
