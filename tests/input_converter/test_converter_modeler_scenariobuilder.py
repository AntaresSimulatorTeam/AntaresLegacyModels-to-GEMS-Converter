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
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter
from antares_gems_converter.input_converter.src.logger import Logger

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
        assert all(c.scenario_group == "wind_group" for c in wind_components), (
            "Wind components must carry scenario_group='wind_group'"
        )

        thermal_components = [c for c in system.components if "gaz" in c.id]
        assert thermal_components, "Expected at least one thermal component"
        assert all(c.scenario_group == "thermal_group" for c in thermal_components), (
            "Thermal components must carry scenario_group='thermal_group'"
        )

        sb_files = list(converter.output_folder.glob("**/modeler-scenariobuilder.dat"))
        assert sb_files, "modeler-scenariobuilder.dat must be generated"

        content = sb_files[0].read_text()
        assert "wind_group, 0 = 2" in content, (
            f"Expected 'wind_group, 0 = 2' in generated SB file, got:\n{content}"
        )
        assert "thermal_group, 0 = 3" in content, (
            f"Expected 'thermal_group, 0 = 3' in generated SB file, got:\n{content}"
        )

    # -------------------------------------------------------------------------
    # Case 3 — hybrid mode: legacy SB cleared for converted areas, modeler SB
    # still generated for the GEMS system
    # -------------------------------------------------------------------------

    def test_hybrid_legacy_sb_cleared_after_conversion(self, fr_wind: Study):
        sb = fr_wind.get_scenario_builder()
        sb.wind.get_area("fr").set_new_scenario([2])
        fr_wind.set_scenario_builder(sb)

        converter = self._init_converter_from_study(fr_wind, model_list=["wind"], mode="hybrid")
        converter.process_all()

        hybrid_sb = converter.study.get_scenario_builder()
        fr_scenario = hybrid_sb.wind.get_area("fr").get_scenario()
        assert all(ts is None for ts in fr_scenario), (
            "Legacy SB wind entries for 'fr' must be cleared after hybrid conversion"
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
        assert "wind_group, 0 = 2" in content, (
            f"Expected 'wind_group, 0 = 2' in generated SB file, got:\n{content}"
        )

        wind_components = [c for c in system.components if "wind" in c.id]
        assert wind_components, "Expected at least one wind component in hybrid mode"
        assert all(c.scenario_group == "wind_group" for c in wind_components), (
            "Wind components must carry scenario_group in hybrid mode when legacy SB has entries"
        )
