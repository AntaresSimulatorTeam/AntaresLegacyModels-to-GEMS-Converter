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
"""E2E: the hybrid study built from the Xpansion direct generator output.

- antares-problem-generator (Antares Simulator) writes the Benders problems;
- the Benders resolution needs Antares Xpansion: set ANTARES_XPANSION_DIR to
  its install folder (the one holding antares-xpansion-launcher) to run it.
"""

import json
import os
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from tests.antares_historic.conftest import assert_antares_exec_exists
from tests.xpansion_generator.utils import (
    CCGT_COST_PER_UNIT,
    HOURS_SIMULATED,
    OCGT_COST_PER_UNIT,
    PEAK_FR,
    SEMIBASE_DE,
    SEMIBASE_FR,
    UNSUPPLIED_COST,
    assemble_hybrid_study,
    run_generator,
)

NB_YEARS = 2
# fr CCGT availability per year: 90 % then 80 % (legacy SB: random -> columns 1, 2)
FR_CCGT_AVAILABILITY = [90.0, 80.0]
CANDIDATES = {
    # component: (cost per unit, max units)
    PEAK_FR: (OCGT_COST_PER_UNIT, 10),
    SEMIBASE_FR: (CCGT_COST_PER_UNIT, 3),
    SEMIBASE_DE: (CCGT_COST_PER_UNIT, 3),
}


@pytest.fixture
def hybrid_study(tmp_path: Path) -> Path:
    _, generated = run_generator(
        tmp_path, nb_years=NB_YEARS, fr_ccgt_availability=FR_CCGT_AVAILABILITY
    )
    return assemble_hybrid_study(tmp_path / "base", generated, tmp_path / "hybrid")


def _master_columns(master_mps: Path) -> dict[str, tuple[float, float]]:
    """{variable: (objective coefficient, upper bound)} of the master problem."""
    cost: dict[str, float] = {}
    upper: dict[str, float] = {}
    section = ""
    for line in master_mps.read_text().splitlines():
        fields = line.split()
        if not line.startswith(" "):
            section = fields[0] if fields else ""
        elif section == "COLUMNS" and fields[1] == "OBJ":
            cost[fields[0]] = float(fields[2])
        elif section == "BOUNDS" and fields[0] == "UP":
            upper[fields[2]] = float(fields[3])
    return {name: (cost[name], upper.get(name, float("inf"))) for name in cost}


def test_problem_generator_writes_benders_problems(
    hybrid_study: Path, antares_exec_folder: Path
) -> None:
    assert_antares_exec_exists(antares_exec_folder)
    result = subprocess.run(
        [str(antares_exec_folder / "antares-problem-generator"), str(hybrid_study)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    (output,) = (hybrid_study / "output").iterdir()
    assert _master_columns(output / "master.mps") == {
        f"{component}.units_to_invest": (cost, max_units)
        for component, (cost, max_units) in CANDIDATES.items()
    }

    structure = [
        line.split("\t") for line in (output / "structure.txt").read_text().split("\n")
    ]
    problems = Counter(problem for problem, *_ in structure if problem)
    assert problems.pop("master") == len(CANDIDATES)
    # one subproblem per week and MC year, each sharing every candidate
    assert len(problems) == 52 * NB_YEARS
    assert set(problems.values()) == {len(CANDIDATES)}


@pytest.mark.skipif(
    "ANTARES_XPANSION_DIR" not in os.environ,
    reason="needs Antares Xpansion: set ANTARES_XPANSION_DIR",
)
def test_xpansion_finds_the_analytic_optimum(hybrid_study: Path) -> None:
    xpansion_dir = Path(os.environ["ANTARES_XPANSION_DIR"])
    result = subprocess.run(
        [
            str(xpansion_dir / "antares-xpansion-launcher"),
            "--step",
            "gems",
            "-i",
            str(hybrid_study),
            "--installDir",
            str(xpansion_dir / "bin"),
            "--allow-run-as-root",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    (out_json,) = (hybrid_study / "output").glob("*/expansion/out.json")
    solution = json.loads(out_json.read_text())["solution"]

    # fr needs 1000 - 600 (base) - 3 x 100 x availability from OCGT, at worst
    # 160 MW (80 % year): 3.2 units; de: 3 CCGT units, 150 MW unsupplied
    units = {PEAK_FR: 3.2, SEMIBASE_FR: 3, SEMIBASE_DE: 3}
    assert solution["values"] == {
        f"{component}.units_to_invest": pytest.approx(value, rel=1e-6)
        for component, value in units.items()
    }
    investment = sum(units[c] * cost for c, (cost, _) in CANDIDATES.items())
    fr_ccgt = [300 * a / 100 for a in FR_CCGT_AVAILABILITY]
    fr = [600 * 20 + ccgt * 50 + (400 - ccgt) * 100 for ccgt in fr_ccgt]
    de = 300 * 50 + 150 * UNSUPPLIED_COST
    operation = HOURS_SIMULATED * (sum(fr) / NB_YEARS + de)
    assert solution["investment_cost"] == pytest.approx(investment, rel=1e-6)
    assert solution["overall_cost"] == pytest.approx(investment + operation, rel=1e-6)
