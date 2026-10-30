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
    CCGT_UNIT_SIZE,
    HOURS_SIMULATED,
    OCGT_COST_PER_UNIT,
    OCGT_MAX_INVESTMENT,
    OCGT_UNIT_SIZE,
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
OCGT_COST_PER_MW = OCGT_COST_PER_UNIT / OCGT_UNIT_SIZE
CCGT_COST_PER_MW = CCGT_COST_PER_UNIT / CCGT_UNIT_SIZE
SEMIBASE = {
    # component: (master variable, cost coefficient, upper bound, integer)
    SEMIBASE_FR: ("invested_units", CCGT_COST_PER_UNIT, 3, True),
    SEMIBASE_DE: ("invested_units", CCGT_COST_PER_UNIT, 3, True),
}
CANDIDATES = {
    # OCGT candidate: continuous (MW) or discrete (units of 50 MW)
    "continuous_peak": {
        PEAK_FR: (
            "invested_capacity",
            OCGT_COST_PER_MW,
            OCGT_MAX_INVESTMENT,
            False,
        ),
        **SEMIBASE,
    },
    "discrete_peak": {
        PEAK_FR: ("invested_units", OCGT_COST_PER_UNIT, 10, True),
        **SEMIBASE,
    },
}
OCGT_UNITS = {"continuous_peak": None, "discrete_peak": 10}


def _hybrid_study(tmp_path: Path, case: str) -> Path:
    _, generated = run_generator(
        tmp_path,
        nb_years=NB_YEARS,
        fr_ccgt_availability=FR_CCGT_AVAILABILITY,
        ocgt_units=OCGT_UNITS[case],
    )
    return assemble_hybrid_study(tmp_path / "base", generated, tmp_path / "hybrid")


def _master_columns(master_mps: Path) -> dict[str, tuple[float, float, bool]]:
    """{variable: (objective coefficient, upper bound, integer)} of the master problem."""
    cost: dict[str, float] = {}
    upper: dict[str, float] = {}
    integer: set[str] = set()
    section = ""
    in_integer_block = False
    for line in master_mps.read_text().splitlines():
        fields = line.split()
        if not line.startswith(" "):
            section = fields[0] if fields else ""
        elif section == "COLUMNS" and "'MARKER'" in fields:
            in_integer_block = "'INTORG'" in fields
        elif section == "COLUMNS" and fields[1] == "OBJ":
            cost[fields[0]] = float(fields[2])
            if in_integer_block:
                integer.add(fields[0])
        elif section == "BOUNDS" and fields[0] in ("UP", "UI"):
            upper[fields[2]] = float(fields[3])
    return {
        name: (cost[name], upper.get(name, float("inf")), name in integer)
        for name in cost
    }


@pytest.mark.parametrize("case", list(CANDIDATES))
def test_problem_generator_writes_benders_problems(
    tmp_path: Path, antares_exec_folder: Path, case: str
) -> None:
    assert_antares_exec_exists(antares_exec_folder)
    hybrid_study = _hybrid_study(tmp_path, case)
    result = subprocess.run(
        [str(antares_exec_folder / "antares-problem-generator"), str(hybrid_study)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    (output,) = (hybrid_study / "output").iterdir()
    candidates = CANDIDATES[case]
    columns = _master_columns(output / "master.mps")
    assert columns == {
        f"{component}.{variable}": (pytest.approx(cost), max_value, integer)
        for component, (variable, cost, max_value, integer) in candidates.items()
    }

    structure = [
        line.split("\t") for line in (output / "structure.txt").read_text().split("\n")
    ]
    problems = Counter(problem for problem, *_ in structure if problem)
    assert problems.pop("master") == len(candidates)
    # one subproblem per week and MC year, each sharing every candidate
    assert len(problems) == 52 * NB_YEARS
    assert set(problems.values()) == {len(candidates)}


# Each candidate keeps its cluster's installed capacity (existing_capacity):
# fr CCGT 100 MW and OCGT 50 MW, de CCGT 100 MW. Every CCGT unit is built. In
# the 80 % year, fr then lacks 1000 - 600 - 0.8 x 400 - 50 = 30 MW: 30 MW of
# continuous OCGT, or 1 unit of 50 MW (0.6 if the integrality were lost).
INVESTMENTS = {
    # case: {component: (master variable, value, investment cost)}
    "continuous_peak": {
        PEAK_FR: ("invested_capacity", 30, 30 * OCGT_COST_PER_MW),
    },
    "discrete_peak": {
        PEAK_FR: ("invested_units", 1, OCGT_COST_PER_UNIT),
    },
}


@pytest.mark.skipif(
    "ANTARES_XPANSION_DIR" not in os.environ,
    reason="needs Antares Xpansion: set ANTARES_XPANSION_DIR",
)
@pytest.mark.parametrize("case", list(INVESTMENTS))
def test_xpansion_finds_the_analytic_optimum(tmp_path: Path, case: str) -> None:
    hybrid_study = _hybrid_study(tmp_path, case)
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

    investments = {
        **INVESTMENTS[case],
        SEMIBASE_FR: ("invested_units", 3, 3 * CCGT_COST_PER_UNIT),
        SEMIBASE_DE: ("invested_units", 3, 3 * CCGT_COST_PER_UNIT),
    }
    assert solution["values"] == {
        f"{component}.{variable}": pytest.approx(value, rel=1e-6)
        for component, (variable, value, _) in investments.items()
    }
    investment = sum(cost for _, _, cost in investments.values())
    # fr: CCGT 400 MW x availability, OCGT the rest; de: 400 MW CCGT, 50 MW unsupplied
    fr_ccgt = [400 * a / 100 for a in FR_CCGT_AVAILABILITY]
    fr = [600 * 20 + ccgt * 50 + (400 - ccgt) * 100 for ccgt in fr_ccgt]
    de = 400 * 50 + 50 * UNSUPPLIED_COST
    operation = HOURS_SIMULATED * (sum(fr) / NB_YEARS + de)
    assert solution["investment_cost"] == pytest.approx(investment, rel=1e-6)
    assert solution["overall_cost"] == pytest.approx(investment + operation, rel=1e-6)
