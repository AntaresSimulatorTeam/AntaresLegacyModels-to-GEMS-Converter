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
import json
import os
import shutil
from pathlib import Path

import pytest

from tests.views_e2e.pipeline import SimulationResults, simulate
from tests.views_e2e.studies import STUDIES

REPO_ROOT = Path(__file__).resolve().parents[2]


def antares_bin_dir() -> Path:
    """`ANTARES_BIN_DIR`, else the bin/ of the release of dependencies.json extracted at the repo root."""
    if "ANTARES_BIN_DIR" in os.environ:
        return Path(os.environ["ANTARES_BIN_DIR"])
    version = json.loads((REPO_ROOT / "dependencies.json").read_text())[
        "antares_simulator_version"
    ]
    return REPO_ROOT / f"antares-{version}-Ubuntu-22.04" / "bin"


def report_dir() -> Path | None:
    """Folder where the views, system and comparison report are kept (`VIEWS_E2E_REPORT_DIR`)."""
    path = os.environ.get("VIEWS_E2E_REPORT_DIR")
    return Path(path) if path else None


@pytest.fixture(scope="session")
def simulations(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, SimulationResults]:
    """Each study of STUDIES, simulated once for the whole session (lazily)."""
    bin_dir = antares_bin_dir()
    if not (bin_dir / "antares-solver").exists():
        pytest.fail(f"antares-solver not found in {bin_dir} (set ANTARES_BIN_DIR)")

    class Simulations(dict):  # type: ignore[type-arg]
        def __missing__(self, study_name: str) -> SimulationResults:
            base = tmp_path_factory.mktemp(study_name)
            study = STUDIES[study_name](base / "study")
            results = simulate(study, bin_dir, base / "views")
            if report := report_dir():
                _keep(results, report / study_name)
            self[study_name] = results
            return results

    return Simulations()


def _keep(results: SimulationResults, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results.system_file, target / "system-for-views.yml")
    shutil.copy2(
        results.output_dir / "system-for-views.yml",
        target / "system-for-views.antares.yml",
    )
    for (view_id, pattern_id), view in results.views.items():
        view.write_parquet(target / f"{view_id}--{pattern_id}.parquet")
