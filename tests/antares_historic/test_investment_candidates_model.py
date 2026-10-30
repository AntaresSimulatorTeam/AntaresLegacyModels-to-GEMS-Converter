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
"""E2E tests of the antares_legacy_investment_candidates library.

- Equivalence with legacy (antares-solver, as for the other libraries): a
  candidate with a fixed investment (min_investment = max_investment) gives the
  same weekly objectives as the legacy study whose cluster has the capacity
  existing_capacity + invested capacity.
- A study using both models runs in Antares (antares-solver,
  antares-problem-generator) and in Antares Xpansion. The Xpansion test runs
  only when ANTARES_XPANSION_DIR is set to its install folder (the one holding
  antares-xpansion-launcher).

Benders study: two independent areas (one candidate model each), load 400 MW, base
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
from tests.antares_historic.utils import (
    ANTARES_VERSION_CREATED_STUDIES,
    _create_area_with_base_clusters,
    addHybridBehavior,
    first_optim_relgap,
    random_availability_ratio,
)

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


def _candidate(
    area: str,
    model: str,
    existing_capacity: float,
    min_investment: float,
    max_investment: float,
    availability_rate: np.ndarray,
    input_dir: Path,
) -> tuple[dict, dict]:
    """Component and thermal-capacity-connection of a candidate on the cluster
    candidate_cluster of area; writes its availability series."""
    component = f"{area}_candidate"
    series = f"availability_rate_{component}"
    (input_dir / "data-series").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(availability_rate).to_csv(
        input_dir / "data-series" / f"{series}.tsv", sep="\t", header=False, index=False
    )
    parameters = [
        _parameter("existing_capacity", existing_capacity),
        _parameter("min_investment", min_investment),
        _parameter("max_investment", max_investment),
        _parameter("annual_investment_cost", COST_PER_MW[model]),
        _parameter("availability_rate", series, series=True),
    ]
    if model == DISCRETE:
        parameters.append(_parameter("unit_size", UNIT_SIZE))
    connection = {
        "component": component,
        "port": "capacity_port",
        "thermal-component": {"area": area, "cluster-id": "candidate_cluster"},
    }
    return {"id": component, "model": model, "parameters": parameters}, connection


def _write_gems_input(
    input_dir: Path, components: list[dict], connections: list[dict]
) -> None:
    """Adds the candidates to the system.yml (created if absent), with the
    library and the optim-config.yml."""
    (input_dir / "model-libraries").mkdir(exist_ok=True)
    shutil.copy2(LIBRARY_PATH, input_dir / "model-libraries")
    system_file = input_dir / "system.yml"
    system = (
        yaml.safe_load(system_file.read_text())["system"]
        if system_file.exists()
        else {"id": "investment_candidates"}
    )
    system.setdefault("components", []).extend(components)
    system["thermal-capacity-connections"] = connections
    system_file.write_text(yaml.safe_dump({"system": system}))
    # Scenario-independent variables: Antares requires a Benders decomposition,
    # also when antares-solver runs the study
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


def _create_study(parent_dir: Path, case: str) -> Path:
    study = create_study_local(case, ANTARES_VERSION_CREATED_STUDIES, parent_dir)
    input_dir = parent_dir / case / "input"
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
        component, connection = _candidate(
            area_name,
            model,
            EXISTING,
            min_investment,
            max_investment,
            np.ones((HOURS, 1)),
            input_dir,
        )
        components.append(component)
        connections.append(connection)
    _write_gems_input(input_dir, components, connections)
    settings = parent_dir / case / "user" / "expansion" / "settings.ini"
    settings.parent.mkdir(parents=True)
    settings.write_text("optimality_gap = 0\nrelative_gap = 1e-8\nsolver = Cbc\n")
    return parent_dir / case


def _run(command: list) -> None:
    result = subprocess.run(
        [str(arg) for arg in command], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]


# Equivalence: the single-area study of the thermal tests, plus a candidate
# cluster at 15 EUR/MWh with existing capacity E = 50 MW and an hourly
# availability rate r. Legacy: the cluster has capacity E + I and series
# (E + I) x r. Hybrid: same cluster, but series E x r only, and a candidate with
# a fixed investment I (in the master only: no investment cost in the weekly
# problems). The connection replaces the cluster's availability by (E + I) x r.
# In accurate unit commitment the cluster's nominal capacity also bounds the
# generation, so it is E + I in both studies.
EQUIVALENCE_EXISTING = 50
EQUIVALENCE_RATE = random_availability_ratio(seed=167)
EQUIVALENCE_REL_ACCURACY = 5 * 1e-5
# model: invested capacity I
FIXED_INVESTMENTS = {CONTINUOUS: 75.3, DISCRETE: 2 * UNIT_SIZE}


def _equivalence_study(
    parent_dir: Path, name: str, capacity: float, series_capacity: float
) -> Path:
    study = create_study_local(name, ANTARES_VERSION_CREATED_STUDIES, parent_dir)
    area = _create_area_with_base_clusters(
        study,
        pd.read_csv("tests/antares_historic/data/load_matrix_1.txt", header=None),
    )
    area.create_thermal_cluster(
        "candidate_cluster",
        ThermalClusterProperties(
            nominal_capacity=capacity, marginal_cost=15, market_bid_cost=15
        ),
    ).set_series(pd.DataFrame(series_capacity * EQUIVALENCE_RATE))
    addHybridBehavior(parent_dir / name)
    return parent_dir / name


@pytest.mark.parametrize("model", FIXED_INVESTMENTS)
def test_fixed_investment_matches_legacy_cluster(
    model: str, tmp_path: Path, antares_exec_folder: Path
) -> None:
    assert_antares_exec_exists(antares_exec_folder)
    invested = FIXED_INVESTMENTS[model]
    total = EQUIVALENCE_EXISTING + invested
    legacy = _equivalence_study(tmp_path, "legacy", total, total)
    hybrid = _equivalence_study(tmp_path, "hybrid", total, EQUIVALENCE_EXISTING)
    component, connection = _candidate(
        "unique",
        model,
        EQUIVALENCE_EXISTING,
        invested,
        invested,
        EQUIVALENCE_RATE,
        hybrid / "input",
    )
    _write_gems_input(hybrid / "input", [component], [connection])

    gap = first_optim_relgap(antares_exec_folder, legacy, hybrid, "highs")
    assert gap < EQUIVALENCE_REL_ACCURACY
    # sensitivity: 10 % less capacity in the legacy study
    perturbated = _equivalence_study(tmp_path, "perturbated", total, 0.9 * total)
    gap = first_optim_relgap(antares_exec_folder, perturbated, hybrid, "highs")
    assert gap > 10 * EQUIVALENCE_REL_ACCURACY


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
