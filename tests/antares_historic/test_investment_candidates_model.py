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

- Equivalence (antares-solver, as for the other libraries): a candidate whose
  investment is fixed (min_investment = max_investment) connected to a legacy
  thermal cluster gives the same weekly objectives as the legacy study whose
  cluster has the capacity existing_capacity + invested capacity. The weekly
  problems hold no investment cost: it is in the Benders master.
- Benders: antares-problem-generator writes the master problem (costs, bounds,
  integer variables), and Antares Xpansion finds the analytic optimum. The
  Xpansion test runs only when ANTARES_XPANSION_DIR is set to its install
  folder (the one holding antares-xpansion-launcher).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
import yaml
from antares.craft import (
    AreaProperties,
    ThermalClusterGroup,
    ThermalClusterProperties,
    create_study_local,
)

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
CONTINUOUS_MODEL = f"{LIBRARY_ID}.continuous_investment_candidate"
DISCRETE_MODEL = f"{LIBRARY_ID}.discrete_investment_candidate"
INVESTMENT_VARIABLES = {
    CONTINUOUS_MODEL: "invested_capacity",
    DISCRETE_MODEL: "invested_units",
}

# The investment variables are scenario-independent: Antares only accepts them
# with a Benders decomposition (variable in master and subproblems, investment
# cost in the master), also when antares-solver solves the weekly problems.
BENDERS_OPTIM_CONFIG = {
    "resolution-mode": "benders-decomposition",
    "models": [
        {
            "id": model,
            "model-decomposition": {
                "variables": [{"id": variable, "location": "master-and-subproblems"}],
                "objective-contributions": [{"id": "objective", "location": "master"}],
            },
        }
        for model, variable in INVESTMENT_VARIABLES.items()
    ],
}

LOAD_FILE = Path("tests/antares_historic/data/load_matrix_1.txt")
TEST_REL_ACCURACY = 5 * 1e-5
TEST_SOLVER = "highs"
HOURS = 8760
HOURS_SIMULATED = 8736  # 52 weeks
CANDIDATE_CLUSTER = "candidate_cluster"
CANDIDATE = "candidate"


def _parameter(param_id: str, value: Any, series: bool = False) -> dict:
    return {
        "id": param_id,
        "value": value,
        "time-dependent": series,
        "scenario-dependent": series,
    }


def _candidate(
    component_id: str,
    model: str,
    existing_capacity: float,
    min_investment: float,
    max_investment: float,
    annual_investment_cost: float,
    unit_size: Optional[float] = None,
) -> dict:
    parameters = [
        _parameter("existing_capacity", existing_capacity),
        _parameter("min_investment", min_investment),
        _parameter("max_investment", max_investment),
        _parameter("annual_investment_cost", annual_investment_cost),
        _parameter("availability_rate", f"availability_rate_{component_id}", True),
    ]
    if unit_size is not None:
        parameters.insert(0, _parameter("unit_size", unit_size))
    return {"id": component_id, "model": model, "parameters": parameters}


def _connection(component_id: str, area: str, cluster_id: str) -> dict:
    return {
        "component": component_id,
        "port": "capacity_port",
        "thermal-component": {"area": area, "cluster-id": cluster_id},
    }


def _write_candidates(
    study_path: Path,
    candidates: list[dict],
    connections: list[dict],
    availability_rates: dict[str, np.ndarray],
    optim_config: dict,
) -> None:
    """Add the candidates to the system.yml of the study (created if absent),
    with their library, availability series and optim-config.yml."""
    input_dir = study_path / "input"
    system_file = input_dir / "system.yml"
    system: dict = {"system": {"id": "investment_candidates"}}
    if system_file.exists():
        system = yaml.safe_load(system_file.read_text())
    system["system"].setdefault("components", []).extend(candidates)
    system["system"]["thermal-capacity-connections"] = connections
    system_file.write_text(yaml.safe_dump(system, sort_keys=False))

    (input_dir / "model-libraries").mkdir(parents=True, exist_ok=True)
    shutil.copy2(LIBRARY_PATH, input_dir / "model-libraries" / LIBRARY_PATH.name)
    series_dir = input_dir / "data-series"
    series_dir.mkdir(parents=True, exist_ok=True)
    for component_id, rate in availability_rates.items():
        pd.DataFrame(rate).to_csv(
            series_dir / f"availability_rate_{component_id}.tsv",
            sep="\t",
            header=False,
            index=False,
        )
    (input_dir / "optim-config.yml").write_text(yaml.safe_dump(optim_config))


# =============================================================================
# Equivalence with the legacy study (antares-solver)
# =============================================================================

AVAILABILITY_RATE = random_availability_ratio(seed=167)
CANDIDATE_MARGINAL_COST = 15
EXISTING_CAPACITY = 50

# model: (invested capacity, unit size)
FIXED_INVESTMENTS = {
    "continuous": (CONTINUOUS_MODEL, 75.3, None),
    "discrete": (DISCRETE_MODEL, 100, 50),  # 2 units
}


def _create_study(
    study_name: str,
    parent_dir: Path,
    cluster_capacity: float,
    cluster_series: np.ndarray,
) -> Path:
    """The single-area study of the thermal tests, plus the candidate cluster."""
    study = create_study_local(
        study_name=study_name,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir,
    )
    area = _create_area_with_base_clusters(study, pd.read_csv(LOAD_FILE, header=None))
    cluster = area.create_thermal_cluster(
        CANDIDATE_CLUSTER,
        ThermalClusterProperties(
            unit_count=1,
            nominal_capacity=cluster_capacity,
            marginal_cost=CANDIDATE_MARGINAL_COST,
            market_bid_cost=CANDIDATE_MARGINAL_COST,
            group=ThermalClusterGroup.GAS,
        ),
    )
    cluster.set_series(pd.DataFrame(cluster_series))
    study_path = parent_dir / study_name
    addHybridBehavior(study_path)
    return study_path


def _legacy_study(name: str, parent_dir: Path, total_capacity: float) -> Path:
    return _create_study(
        name, parent_dir, total_capacity, total_capacity * AVAILABILITY_RATE
    )


def _hybrid_study(
    name: str,
    parent_dir: Path,
    model: str,
    invested_capacity: float,
    unit_size: Optional[float],
) -> Path:
    """The legacy cluster keeps its existing capacity (and series): the
    connection replaces its availability by the candidate's capacity."""
    total_capacity = EXISTING_CAPACITY + invested_capacity
    study_path = _create_study(
        name, parent_dir, total_capacity, EXISTING_CAPACITY * AVAILABILITY_RATE
    )
    _write_candidates(
        study_path,
        [
            _candidate(
                CANDIDATE,
                model,
                EXISTING_CAPACITY,
                min_investment=invested_capacity,
                max_investment=invested_capacity,
                # in the master only: not in the weekly objectives
                annual_investment_cost=1000,
                unit_size=unit_size,
            )
        ],
        [_connection(CANDIDATE, "unique", CANDIDATE_CLUSTER)],
        {CANDIDATE: AVAILABILITY_RATE},
        BENDERS_OPTIM_CONFIG,
    )
    return study_path


@pytest.mark.parametrize("case", list(FIXED_INVESTMENTS))
def test_fixed_investment_matches_legacy_cluster(
    case: str, tmp_path: Path, antares_exec_folder: Path
) -> None:
    assert_antares_exec_exists(antares_exec_folder)
    model, invested_capacity, unit_size = FIXED_INVESTMENTS[case]
    total_capacity = EXISTING_CAPACITY + invested_capacity
    legacy = _legacy_study(f"legacy_{case}", tmp_path, total_capacity)
    hybrid = _hybrid_study(
        f"hybrid_{case}", tmp_path, model, invested_capacity, unit_size
    )
    rel_gap = first_optim_relgap(antares_exec_folder, legacy, hybrid, TEST_SOLVER)
    assert rel_gap < TEST_REL_ACCURACY

    # the test is sensitive to the capacity: 10 % less in the legacy study
    perturbated = _legacy_study(
        f"legacy_{case}_perturbated", tmp_path, 0.9 * total_capacity
    )
    rel_gap_perturbated = first_optim_relgap(
        antares_exec_folder, perturbated, hybrid, TEST_SOLVER
    )
    assert rel_gap_perturbated > 10 * TEST_REL_ACCURACY


# =============================================================================
# Benders decomposition (antares-problem-generator, Antares Xpansion)
# =============================================================================

# Two independent areas, load 400 MW, base cluster 300 MW at 10 EUR/MWh, and a
# candidate cluster at 50 EUR/MWh with 20 MW installed (existing_capacity),
# always available. Unsupplied energy costs 1000 EUR/MWh, so each candidate
# covers the 80 MW missing, within its bounds.
LOAD = 400
BASE_CAPACITY = 300
UNSUPPLIED_COST = 1000
CANDIDATE_COST = 50
CONTINUOUS_COST_PER_MW = 30000
DISCRETE_COST_PER_MW = 20000
DISCRETE_UNIT_SIZE = 30

# case: {area: (model, min_investment, max_investment, optimal value)}
BENDERS_CASES = {
    # continuous: 80 MW; discrete: 80 MW rounded up to 3 units of 30 MW
    "free": {
        "continuous": (CONTINUOUS_MODEL, 0, 500, 80),
        "discrete": (DISCRETE_MODEL, 0, 400, 3),
    },
    # continuous: min_investment binds (150 MW); discrete: max_investment binds,
    # rounded down to 1 unit (45 / 30 = 1.5), 50 MW unsupplied
    "bounds": {
        "continuous": (CONTINUOUS_MODEL, 150, 500, 150),
        "discrete": (DISCRETE_MODEL, 0, 45, 1),
    },
}


def _invested_capacity(model: str, value: float) -> float:
    return value * DISCRETE_UNIT_SIZE if model == DISCRETE_MODEL else value


def _benders_study(parent_dir: Path, case: str) -> Path:
    study = create_study_local(
        study_name=case,
        version=ANTARES_VERSION_CREATED_STUDIES,
        parent_directory=parent_dir,
    )
    candidates, connections, rates = [], [], {}
    for area_name, (model, min_investment, max_investment, _) in BENDERS_CASES[
        case
    ].items():
        area = study.create_area(
            area_name,
            properties=AreaProperties(
                energy_cost_unsupplied=UNSUPPLIED_COST, energy_cost_spilled=0
            ),
        )
        area.set_load(pd.DataFrame(np.full((HOURS, 1), LOAD)))
        for cluster_name, capacity, cost in [
            ("base", BASE_CAPACITY, 10),
            (CANDIDATE_CLUSTER, 20, CANDIDATE_COST),
        ]:
            cluster = area.create_thermal_cluster(
                cluster_name,
                ThermalClusterProperties(
                    unit_count=1,
                    nominal_capacity=capacity,
                    marginal_cost=cost,
                    market_bid_cost=cost,
                ),
            )
            cluster.set_series(pd.DataFrame(np.full((HOURS, 1), capacity)))
        component_id = f"{area_name}_candidate"
        discrete = model == DISCRETE_MODEL
        candidates.append(
            _candidate(
                component_id,
                model,
                existing_capacity=20,
                min_investment=min_investment,
                max_investment=max_investment,
                annual_investment_cost=(
                    DISCRETE_COST_PER_MW if discrete else CONTINUOUS_COST_PER_MW
                ),
                unit_size=DISCRETE_UNIT_SIZE if discrete else None,
            )
        )
        connections.append(_connection(component_id, area_name, CANDIDATE_CLUSTER))
        rates[component_id] = np.ones((HOURS, 1))

    study_path = parent_dir / case
    _write_candidates(
        study_path,
        candidates,
        connections,
        rates,
        BENDERS_OPTIM_CONFIG,
    )
    settings = study_path / "user" / "expansion" / "settings.ini"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        "optimality_gap = 0\nrelative_gap = 1e-8\nmax_iteration = 200\nsolver = Cbc\n"
    )
    return study_path


def _master_columns(
    master_mps: Path,
) -> dict[str, tuple[float, float, float, bool]]:
    """{variable: (objective coefficient, lower bound, upper bound, integer)}."""
    cost: dict[str, float] = {}
    lower: dict[str, float] = {}
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
        elif section == "BOUNDS" and fields[0] in ("LO", "LI"):
            lower[fields[2]] = float(fields[3])
        elif section == "BOUNDS" and fields[0] in ("UP", "UI"):
            upper[fields[2]] = float(fields[3])
    return {
        name: (
            cost[name],
            lower.get(name, 0.0),
            upper.get(name, float("inf")),
            name in integer,
        )
        for name in cost
    }


@pytest.mark.parametrize("case", list(BENDERS_CASES))
def test_problem_generator_writes_master_problem(
    case: str, tmp_path: Path, antares_exec_folder: Path
) -> None:
    assert_antares_exec_exists(antares_exec_folder)
    study_path = _benders_study(tmp_path, case)
    result = subprocess.run(
        [str(antares_exec_folder / "antares-problem-generator"), str(study_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    (output,) = (study_path / "output").iterdir()
    expected = {}
    for area_name, (model, min_investment, max_investment, _) in BENDERS_CASES[
        case
    ].items():
        variable = f"{area_name}_candidate.{INVESTMENT_VARIABLES[model]}"
        if model == DISCRETE_MODEL:
            # cost per unit; bounds in units, the master keeps them integer
            expected[variable] = (
                pytest.approx(DISCRETE_COST_PER_MW * DISCRETE_UNIT_SIZE),
                pytest.approx(min_investment / DISCRETE_UNIT_SIZE),
                pytest.approx(max_investment / DISCRETE_UNIT_SIZE),
                True,
            )
        else:
            expected[variable] = (
                pytest.approx(CONTINUOUS_COST_PER_MW),
                pytest.approx(min_investment),
                pytest.approx(max_investment),
                False,
            )
    assert _master_columns(output / "master.mps") == expected


@pytest.mark.skipif(
    "ANTARES_XPANSION_DIR" not in os.environ,
    reason="needs Antares Xpansion: set ANTARES_XPANSION_DIR",
)
@pytest.mark.parametrize("case", list(BENDERS_CASES))
def test_xpansion_finds_the_analytic_optimum(case: str, tmp_path: Path) -> None:
    study_path = _benders_study(tmp_path, case)
    xpansion_dir = Path(os.environ["ANTARES_XPANSION_DIR"])
    result = subprocess.run(
        [
            str(xpansion_dir / "antares-xpansion-launcher"),
            "--step",
            "gems",
            "-i",
            str(study_path),
            "--installDir",
            str(xpansion_dir / "bin"),
            "--allow-run-as-root",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]

    (out_json,) = (study_path / "output").glob("*/expansion/out.json")
    solution = json.loads(out_json.read_text())["solution"]

    candidates = BENDERS_CASES[case]
    assert solution["values"] == {
        f"{area}_candidate.{INVESTMENT_VARIABLES[model]}": pytest.approx(
            value, rel=1e-6, abs=1e-6
        )
        for area, (model, _, _, value) in candidates.items()
    }
    investment = 0.0
    operation = 0.0
    for model, _, _, value in candidates.values():
        invested = _invested_capacity(model, value)
        cost_per_mw = (
            DISCRETE_COST_PER_MW if model == DISCRETE_MODEL else CONTINUOUS_COST_PER_MW
        )
        investment += cost_per_mw * invested
        candidate_generation = min(LOAD - BASE_CAPACITY, 20 + invested)
        unsupplied = LOAD - BASE_CAPACITY - candidate_generation
        operation += HOURS_SIMULATED * (
            BASE_CAPACITY * 10
            + candidate_generation * CANDIDATE_COST
            + unsupplied * UNSUPPLIED_COST
        )
    assert solution["investment_cost"] == pytest.approx(investment, rel=1e-6)
    # Antares adds a small random noise to the thermal costs (~1e-6)
    assert solution["overall_cost"] == pytest.approx(investment + operation, rel=1e-5)
