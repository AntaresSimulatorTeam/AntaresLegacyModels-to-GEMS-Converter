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
from pathlib import Path

import pytest
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter

LIB_PATHS = [
    "src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml",
    "src/antares_gems_converter/libs/reference_models/andromede_v1_models.yml",
]


@pytest.mark.parametrize("mode", ["full", "hybrid"])
def test_default_models_on_study_without_battery(
    local_study_w_areas: Study, tmp_path: Path, mode: str
) -> None:
    # battery is converted by default, but the study has no battery artefact
    output_folder = tmp_path / "output"
    output_folder.mkdir()
    converter = AntaresStudyConverter(
        study_input=local_study_w_areas,
        logger=logging.getLogger(__name__),
        mode=mode,
        output_folder=output_folder,
        lib_paths=LIB_PATHS,
    )
    converter.process_all()
    assert converter.output_system_path.is_file()
