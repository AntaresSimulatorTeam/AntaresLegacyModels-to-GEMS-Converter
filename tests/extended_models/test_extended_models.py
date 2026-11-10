# Copyright (c) 2025, RTE (https://www.rte-france.com)
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
"""E2E tests of extended_models.yml: small studies, with the area and load of
antares_legacy_models.yml, solved with antares-modeler."""
from pathlib import Path

import numpy as np

from tests.extended_models.gems_study import (
    GemsResult,
    GemsStudy,
    Value,
    component,
    connection,
)

LIBS_DIR = Path("src/antares_gems_converter/libs")
LEGACY_LIBRARY = LIBS_DIR / "antares_historic" / "antares_legacy_models.yml"
EXTENDED_LIBRARY = LIBS_DIR / "extended_models" / "extended_models.yml"
LEGACY = "antares_legacy_models"
EXTENDED = "extended_models"

REL_TOL = 1e-6
NUM_TIME_STEPS = 168
HOURS = np.arange(NUM_TIME_STEPS)
DAILY_SHAPE = np.sin(2 * np.pi * (HOURS - 6) / 24)
SOLAR = 1100 * np.maximum(0, DAILY_SHAPE)
LOAD = 900 + 400 * DAILY_SHAPE + 100 * np.sin(2 * np.pi * HOURS / NUM_TIME_STEPS)
# Unused by thermal_fast, but parameters of the model
UNUSED_THERMAL_FAST_PARAMETERS: dict[str, Value] = dict(
    startup_cost=0, fixed_cost=0, d_min_up=1, d_min_down=1
)
# Models with a category of antares_legacy_taxonomy, whose components set the
# properties technology and name
TAXONOMY_MODELS = {"thermal_milp", "thermal_fast", "short-term-storage"}


def study(load: np.ndarray = LOAD, spillage_cost: float = 100) -> GemsStudy:
    """A study with one area and its load."""
    result = GemsStudy(
        libraries=[LEGACY_LIBRARY, EXTENDED_LIBRARY],
        components=[
            component(
                "area",
                f"{LEGACY}.area",
                spillage_cost=spillage_cost,
                unsupplied_energy_cost=3000,
            ),
            component("load", f"{LEGACY}.load", load="load"),
        ],
        data_series={"load": load},
    )
    result.connections.append(
        connection("area", "balance_port", "load", "balance_port")
    )
    return result


def extended_component(component_id: str, model: str, **parameters: Value) -> dict:
    """A component of a model of extended_models.yml."""
    result = component(component_id, f"{EXTENDED}.{model}", **parameters)
    if model in TAXONOMY_MODELS:
        result["properties"] = [
            {"id": "technology", "value": model},
            {"id": "name", "value": component_id},
        ]
    return result


def connect_to_area(gems_study: GemsStudy, comp: dict) -> None:
    gems_study.components.append(comp)
    gems_study.connections.append(
        connection("area", "balance_port", comp["id"], "balance_port")
    )


def solve(
    gems_study: GemsStudy, tmp_path: Path, name: str, exec_folder: Path
) -> GemsResult:
    return gems_study.solve(tmp_path.absolute() / name, exec_folder)


def thermal_fast(component_id: str, capacity: float, cost: float) -> dict:
    return extended_component(
        component_id,
        "thermal_fast",
        minimum_generation_modulation=0,
        p_max_cluster=capacity,
        p_min_unit=0,
        p_max_unit=100,
        generation_cost=cost,
        unit_count=capacity / 100,
        spinning=0,
        **UNUSED_THERMAL_FAST_PARAMETERS,
    )


# ---------------------------------------------------------------- storage


def storage_study(mccormick: int, mccormick_grand_maison: int) -> GemsStudy:
    result = study(load=400 + 100 * DAILY_SHAPE - SOLAR, spillage_cost=50)
    connect_to_area(result, thermal_fast("gas", capacity=1000, cost=80))
    result.data_series["injection_modulation"] = np.where(HOURS % 24 < 18, 1.0, 0.8)
    connect_to_area(
        result,
        extended_component(
            "storage",
            "short-term-storage",
            reservoir_capacity=1000,
            injection_nominal_capacity=300,
            withdrawal_nominal_capacity=300,
            efficiency_injection=0.8,
            efficiency_withdrawal=1,
            lower_rule_curve=0.1,
            upper_rule_curve=0.9,
            p_max_injection_modulation="injection_modulation",
            p_max_withdrawal_modulation=1,
            inflows=0,
            initial_level=0.5,
            cost_injection=0,
            cost_withdrawal=0,
            cost_level=0.01,
            p_t_cost=0.5,
            nb_cycles_max=0.5,
            mccormick=mccormick,
            mccormick_grand_maison=mccormick_grand_maison,
            p_withdrawal_when_injection_max=100,
            cycle_constraint=0,
        ),
    )
    return result


def test_grand_maison_mccormick_is_looser(
    tmp_path: Path, antares_exec_folder: Path
) -> None:
    usual = solve(storage_study(1, 0), tmp_path, "usual", antares_exec_folder)
    grand_maison = solve(
        storage_study(0, 1), tmp_path, "grand_maison", antares_exec_folder
    )
    assert grand_maison.objective < usual.objective * (1 - 100 * REL_TOL)


# ---------------------------------------------------------------- binding constraints

# Port -> (component, cost, capacity, alpha); the other ports are not connected.
NUCLEAR_CLUSTERS = {
    "flow_fr_nuclear_cp0_cp1_cp2": ("cp0", 10, 500, 1),
    "flow_fr_nuclear_epr": ("epr", 12, 400, 1),
    "flow_fr_nuclear_n4": ("n4", 14, 300, 0.8),
    "flow_y_nuc_modulation_nuclear_p4": ("modulation_p4", 16, 200, 1.5),
}
ALPHAS = [
    "alpha_fr_nuclear_cp0_cp1_cp2",
    "alpha_fr_nuclear_epr",
    "alpha_fr_nuclear_n4",
    "alpha_fr_nuclear_p4",
    "alpha_fr_nuclear_peak1",
    "alpha_fr_nuclear_peak2",
    "alpha_fr_nuclear_peak3",
    "alpha_y_nuc_modulation_nuclear_cp0_cp1_cp2",
    "alpha_y_nuc_modulation_nuclear_epr",
    "alpha_y_nuc_modulation_nuclear_n4",
    "alpha_y_nuc_modulation_nuclear_p4",
]
DAILY_RIGHT_HAND_SIDE = 600 + 80 * (HOURS // 24)


def nuclear_study(model: str) -> GemsStudy:
    """The nuclear clusters under the binding constraint model, with a right hand
    side that changes from day to day."""
    result = study(spillage_cost=10)
    for component_id, cost, capacity, _ in NUCLEAR_CLUSTERS.values():
        connect_to_area(result, thermal_fast(component_id, capacity, cost))
    connect_to_area(result, thermal_fast("gas", capacity=2000, cost=70))
    alphas = {
        port.replace("flow_", "alpha_"): alpha
        for port, (*_, alpha) in NUCLEAR_CLUSTERS.items()
    }
    parameters: dict[str, Value] = {alpha: alphas.get(alpha, 1) for alpha in ALPHAS}
    parameters["rhs"] = "rhs"
    result.data_series["rhs"] = DAILY_RIGHT_HAND_SIDE
    result.components.append(extended_component("constraint", model, **parameters))
    for port, (component_id, *_) in NUCLEAR_CLUSTERS.items():
        result.connections.append(
            connection(component_id, "balance_port", "constraint", port)
        )
    return result


def test_daily_binding_constraint_is_tighter_than_weekly(
    tmp_path: Path, antares_exec_folder: Path
) -> None:
    """The daily constraint is not the weekly one: the days are constrained one by one."""
    daily = solve(
        nuclear_study("bc_nuc_daily_less"), tmp_path, "daily", antares_exec_folder
    )
    weekly = solve(
        nuclear_study("bc_nuc_weekly_less"), tmp_path, "weekly", antares_exec_folder
    )
    assert weekly.objective < daily.objective * (1 - 100 * REL_TOL)
