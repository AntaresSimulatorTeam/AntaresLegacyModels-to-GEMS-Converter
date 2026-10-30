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
"""E2E: a hybrid study using the antares_legacy_investment_candidates library
runs in Antares (antares-solver, antares-problem-generator) and in Antares
Xpansion. The Xpansion test runs only when ANTARES_XPANSION_DIR is set to its
install folder (the one holding antares-xpansion-launcher).

Study: two independent areas (one candidate model each), load 400 MW, base
cluster 300 MW at 10 EUR/MWh, candidate cluster 20 MW at 50 EUR/MWh (the
candidate's existing_capacity). Unsupplied energy costs 1000 EUR/MWh, so each
candidate covers the 80 MW missing, within its bounds.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from antares.craft import AreaProperties, ThermalClusterProperties, create_study_local

from tests.antares_historic.conftest import assert_antares_exec_exists
from tests.antares_historic.utils import ANTARES_VERSION_CREATED_STUDIES

LIBRARY_ID = "antares_legacy_investment_candidates"
LIBRARY_PATH = Path(f"src/antares_gems_converter/libs/{LIBRARY_ID}/{LIBRARY_ID}.yml")
CONTINUOUS = f"{LIBRARY_ID}.continuous_investment_candidate"
DISCRETE = f"{LIBRARY_ID}.discrete_investment_candidate"
VARIABLES = {CONTINUOUS: "invested_capacity", DISCRETE: "invested_units"}
COST_PER_MW = {CONTINUOUS: 30000, DISCRETE: 20000}
UNIT_SIZE = 30

HOURS = 8760
HOURS_SIMULATED = 8736  # 52 weeks
LOAD, BASE, EXISTING = 400, 300, 20
BASE_COST, CANDIDATE_COST, UNSUPPLIED_COST = 10, 50, 1000

# case: {area: (model, min_investment, max_investment, optimal variable value)}
CASES = {
    # 80 MW; 80 MW rounded up to 3 units of 30 MW
    "free": {"continuous": (CONTINUOUS, 0, 500, 80), "discrete": (DISCRETE, 0, 400, 3)},
    # min_investment binds; max_investment binds, rounded down to 1 unit (45 / 30)
    "bounds": {
        "continuous": (CONTINUOUS, 150, 500, 150),
        "discrete": (DISCRETE, 0, 45, 1),
    },
}


def _variable(area: str, model: str) -> str:
    return f"{area}_candidate.{VARIABLES[model]}"


def _parameter(param_id: str, value: float | str, series: bool = False) -> dict:
    return {
        "id": param_id,
        "value": value,
        "time-dependent": series,
        "scenario-dependent": series,
    }


def _create_study(parent_dir: Path, case: str) -> Path:
    study = create_study_local(case, ANTARES_VERSION_CREATED_STUDIES, parent_dir)
    input_dir = parent_dir / case / "input"
    (input_dir / "data-series").mkdir(parents=True)
    components, connections = [], []
    for area_name, (model, min_investment, max_investment, _) in CASES[case].items():
        area = study.create_area(
            area_name,
            properties=AreaProperties(
                energy_cost_unsupplied=UNSUPPLIED_COST, energy_cost_spilled=0
            ),
        )
        area.set_load(pd.DataFrame(np.full((HOURS, 1), LOAD)))
        for name, capacity, cost in [
            ("base", BASE, BASE_COST),
            ("candidate_cluster", EXISTING, CANDIDATE_COST),
        ]:
            area.create_thermal_cluster(
                name,
                ThermalClusterProperties(
                    nominal_capacity=capacity, marginal_cost=cost, market_bid_cost=cost
                ),
            ).set_series(pd.DataFrame(np.full((HOURS, 1), capacity)))

        component = f"{area_name}_candidate"
        series = f"availability_rate_{component}"
        pd.DataFrame(np.ones((HOURS, 1))).to_csv(
            input_dir / "data-series" / f"{series}.tsv",
            sep="\t",
            header=False,
            index=False,
        )
        parameters = [
            _parameter("existing_capacity", EXISTING),
            _parameter("min_investment", min_investment),
            _parameter("max_investment", max_investment),
            _parameter("annual_investment_cost", COST_PER_MW[model]),
            _parameter("availability_rate", series, series=True),
        ]
        if model == DISCRETE:
            parameters.append(_parameter("unit_size", UNIT_SIZE))
        components.append({"id": component, "model": model, "parameters": parameters})
        connections.append(
            {
                "component": component,
                "port": "capacity_port",
                "thermal-component": {
                    "area": area_name,
                    "cluster-id": "candidate_cluster",
                },
            }
        )

    (input_dir / "model-libraries").mkdir()
    shutil.copy2(LIBRARY_PATH, input_dir / "model-libraries")
    system = {
        "id": case,
        "components": components,
        "thermal-capacity-connections": connections,
    }
    (input_dir / "system.yml").write_text(yaml.safe_dump({"system": system}))
    # Scenario-independent variables: Antares requires a Benders decomposition
    models = [
        {
            "id": model,
            "model-decomposition": {
                "variables": [{"id": variable, "location": "master-and-subproblems"}],
                "objective-contributions": [{"id": "objective", "location": "master"}],
            },
        }
        for model, variable in VARIABLES.items()
    ]
    (input_dir / "optim-config.yml").write_text(
        yaml.safe_dump({"resolution-mode": "benders-decomposition", "models": models})
    )
    settings = parent_dir / case / "user" / "expansion" / "settings.ini"
    settings.parent.mkdir(parents=True)
    settings.write_text("optimality_gap = 0\nrelative_gap = 1e-8\nsolver = Cbc\n")
    return parent_dir / case


def _run(command: list) -> None:
    result = subprocess.run(
        [str(arg) for arg in command], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


@pytest.mark.parametrize("case", CASES)
def test_antares_solver_runs_the_study(
    case: str, tmp_path: Path, antares_exec_folder: Path
) -> None:
    assert_antares_exec_exists(antares_exec_folder)
    study_path = _create_study(tmp_path, case)
    # the discrete model needs a MIP solver: highs, not sirius (the default)
    _run(
        [antares_exec_folder / "antares-solver", study_path, "--linear-solver", "highs"]
    )


def _master_columns(master_mps: Path) -> dict[str, list]:
    """{variable: [objective coefficient, lower bound, upper bound, integer]}."""
    columns: dict[str, list] = {}
    section, integer = "", False
    for line in master_mps.read_text().splitlines():
        fields = line.split()
        if not line.startswith(" "):
            section = fields[0] if fields else ""
        elif section == "COLUMNS" and "'MARKER'" in fields:
            integer = "'INTORG'" in fields
        elif section == "COLUMNS" and fields[1] == "OBJ":
            columns[fields[0]] = [float(fields[2]), 0.0, float("inf"), integer]
        elif section == "BOUNDS" and fields[0] in ("LO", "LI", "UP", "UI"):
            columns[fields[2]][1 if fields[0][0] == "L" else 2] = float(fields[3])
    return columns


@pytest.mark.parametrize("case", CASES)
def test_problem_generator_writes_the_master_problem(
    case: str, tmp_path: Path, antares_exec_folder: Path
) -> None:
    assert_antares_exec_exists(antares_exec_folder)
    study_path = _create_study(tmp_path, case)
    _run([antares_exec_folder / "antares-problem-generator", study_path])

    (output,) = (study_path / "output").iterdir()
    # discrete: cost per unit, bounds in units, integer variable
    scale = {CONTINUOUS: 1, DISCRETE: UNIT_SIZE}
    assert _master_columns(output / "master.mps") == {
        _variable(area, model): [
            pytest.approx(COST_PER_MW[model] * scale[model]),
            pytest.approx(min_investment / scale[model]),
            pytest.approx(max_investment / scale[model]),
            model == DISCRETE,
        ]
        for area, (model, min_investment, max_investment, _) in CASES[case].items()
    }


@pytest.mark.skipif(
    "ANTARES_XPANSION_DIR" not in os.environ,
    reason="needs Antares Xpansion: set ANTARES_XPANSION_DIR",
)
@pytest.mark.parametrize("case", CASES)
def test_xpansion_finds_the_analytic_optimum(case: str, tmp_path: Path) -> None:
    study_path = _create_study(tmp_path, case)
    xpansion_dir = Path(os.environ["ANTARES_XPANSION_DIR"])
    _run(
        [
            xpansion_dir / "antares-xpansion-launcher",
            "--step",
            "gems",
            "-i",
            study_path,
            "--installDir",
            xpansion_dir / "bin",
            "--allow-run-as-root",
        ]
    )
    (out_json,) = (study_path / "output").glob("*/expansion/out.json")
    solution = json.loads(out_json.read_text())["solution"]

    candidates = CASES[case]
    assert solution["values"] == {
        _variable(area, model): pytest.approx(value, abs=1e-6)
        for area, (model, _, _, value) in candidates.items()
    }
    investment = operation = 0.0
    for model, _, _, value in candidates.values():
        invested = value * (UNIT_SIZE if model == DISCRETE else 1)
        generation = min(LOAD - BASE, EXISTING + invested)
        investment += COST_PER_MW[model] * invested
        operation += HOURS_SIMULATED * (
            BASE * BASE_COST
            + generation * CANDIDATE_COST
            + (LOAD - BASE - generation) * UNSUPPLIED_COST
        )
    assert solution["investment_cost"] == pytest.approx(investment, rel=1e-6)
    # Antares adds a small random noise to the thermal costs (~1e-6)
    assert solution["overall_cost"] == pytest.approx(investment + operation, rel=1e-5)
