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
"""Fixtures shared by all the test folders."""
import json
import os
from pathlib import Path

import pytest

_dependencies = json.loads(
    (Path(__file__).resolve().parents[1] / "dependencies.json").read_text()
)
ANTARES_SIMULATOR_VERSION = _dependencies["antares_simulator_version"]

ANTARES_VERSION_POSIX = f"antares-{ANTARES_SIMULATOR_VERSION}-Ubuntu-22.04"
ANTARES_VERSION_WINDOWS = f"rte-antares-{ANTARES_SIMULATOR_VERSION}-installer-64bits"

current_dir = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def antares_exec_folder() -> Path:
    if os.name == "nt":
        window_path = current_dir.parent / ANTARES_VERSION_WINDOWS / "bin"
        return window_path
    if os.name == "posix":
        posix_path = current_dir / ANTARES_VERSION_POSIX / "bin"
        return posix_path
    else:
        raise RuntimeError("Unsupported OS")


def assert_antares_exec_exists(antares_exec_folder: Path) -> None:
    antares_exec = antares_exec_folder / "antares-modeler"
    assert antares_exec.exists(), f"Antares executable not found at {antares_exec}"
