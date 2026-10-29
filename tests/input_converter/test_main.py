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
import shutil
import sys
from pathlib import Path

import pytest

from antares_gems_converter.input_converter.src.data_preprocessing.data_classes import (
    ConversionMode,
)
from antares_gems_converter.input_converter.src.data_preprocessing.preprocessing import (
    ModelConversionPreprocessor,
)
from antares_gems_converter.input_converter.src.main import main

RESOURCES = Path(__file__).parent / "resources"
STUDY_NAME = "mini_test_batterie_BP23"
LIB_PATHS = [
    Path("src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml"),
    Path("src/antares_gems_converter/libs/reference_models/andromede_v1_models.yml"),
]


def _run_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str) -> Path:
    """Write `config` (with `{study}`, `{output}` and `{libs}` placeholders) to an
    INI file, run the CLI on it and return the converted study folder."""
    study_path = tmp_path / "input" / STUDY_NAME
    shutil.copytree(RESOURCES / STUDY_NAME, study_path)
    output_folder = tmp_path / "output"
    output_folder.mkdir()
    # Spaces after the commas: users write lists this way.
    libs = ", ".join(str(lib.absolute()) for lib in LIB_PATHS)

    conf_path = tmp_path / "config.ini"
    conf_path.write_text(
        config.format(study=study_path, output=output_folder, libs=libs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "antares-to-gems",
            "--conf",
            str(conf_path),
            "--logging",
            str(tmp_path / "log"),
        ],
    )
    main()
    return output_folder / STUDY_NAME


def test_main_hybrid_mode_upper_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _run_main(
        tmp_path,
        monkeypatch,
        "[study]\nstudy_path = {study}\noutput_folder = {output}\n"
        "lib_paths = {libs}\nmode = HYBRID\n\n[hybrid]\nmodel_list = battery\n",
    )
    system = (output / "input" / "system.yml").read_text()
    assert "fr_battery" in system
    assert "area-connections" in system


def test_main_full_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = _run_main(
        tmp_path,
        monkeypatch,
        "[study]\nstudy_path = {study}\noutput_folder = {output}\n"
        "lib_paths = {libs}\nmode = FULL\n",
    )
    assert (output / "input" / "system.yml").is_file()


@pytest.mark.parametrize(
    "config, message",
    [
        (
            "[study]\nstudy_path = {study}\noutput_folder = {output}\n"
            "lib_paths = {libs}\nmode = partial\n",
            "Invalid conversion mode 'partial'",
        ),
        (
            "[study]\nstudy_path = {study}\noutput_folder = {output}\n"
            "lib_paths = {libs}\nmode = hybrid\n",
            "no 'model_list'",
        ),
        (
            "[study]\nstudy_path = {study}\noutput_folder = {output}\n",
            "No option 'lib_paths'",
        ),
    ],
)
def test_main_invalid_config_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: str, message: str
) -> None:
    with pytest.raises(SystemExit, match=message):
        _run_main(tmp_path, monkeypatch, config)


def test_preprocessed_values_not_shared_between_instances(tmp_path: Path) -> None:
    first = ModelConversionPreprocessor(None, ConversionMode.FULL, tmp_path)  # type: ignore
    second = ModelConversionPreprocessor(None, ConversionMode.FULL, tmp_path)  # type: ignore
    first.preprocessed_values["reservoir_capacity"] = 10.0
    assert second.preprocessed_values == {}
