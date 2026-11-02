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
"""The models of extended_models.yml behave as the models they come from.

Each model was written from a model of a former external library, kept as test data
(REFERENCE_LIBRARY, whose id is antares_legacy_models), with the same ids of models,
parameters, variables and ports. For each model, the same study is built twice and
solved with antares-modeler:
- the reference study only uses REFERENCE_LIBRARY;
- the extended study uses antares_legacy_models.yml (area, load) and extended_models.yml.
The components of the tested models are the same in both studies, but for the
library id of their model. Both studies must have the same optimal objective. Each
test also checks that the modelled feature is active in its study, i.e. that removing
it changes the objective.
"""
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import yaml

from tests.antares_historic.gems_study import (
    GemsResult,
    GemsStudy,
    Value,
    component,
    connection,
)

LIBS_DIR = Path("src/antares_gems_converter/libs")
LEGACY_LIBRARY = LIBS_DIR / "antares_historic" / "antares_legacy_models.yml"
EXTENDED_LIBRARY = LIBS_DIR / "extended_models" / "extended_models.yml"
REFERENCE_LIBRARY = Path(
    "tests/antares_historic/data/reference_libraries/antares_legacy_models_external.yml"
)
LEGACY = "antares_legacy_models"
EXTENDED = "extended_models"

REL_TOL = 1e-6
NUM_TIME_STEPS = 168
HOURS = np.arange(NUM_TIME_STEPS)
DAILY_SHAPE = np.sin(2 * np.pi * (HOURS - 6) / 24)
SOLAR_SHAPE = np.maximum(0, DAILY_SHAPE)
LOAD = 900 + 400 * DAILY_SHAPE + 100 * np.sin(2 * np.pi * HOURS / NUM_TIME_STEPS)
# Unused by thermal_fast, but parameters of the model
UNUSED_THERMAL_FAST_PARAMETERS: dict[str, Value] = dict(
    startup_cost=0, fixed_cost=0, d_min_up=1, d_min_down=1
)


class Flavour:
    """How to write a study with the reference library or with the new libraries."""

    def __init__(self, is_reference: bool) -> None:
        self.is_reference = is_reference
        self.libraries = (
            [REFERENCE_LIBRARY] if is_reference else [LEGACY_LIBRARY, EXTENDED_LIBRARY]
        )

    def study(
        self,
        load: np.ndarray = LOAD,
        spillage_cost: float = 100,
        unsupplied_energy_cost: float = 3000,
    ) -> GemsStudy:
        """A study with one area and its load, which are not the tested models."""
        area_parameters: dict[str, Value] = {"spillage_cost": spillage_cost}
        if self.is_reference:
            area_parameters["ens_cost"] = unsupplied_energy_cost
        else:
            area_parameters["unsupplied_energy_cost"] = unsupplied_energy_cost
        study = GemsStudy(
            libraries=self.libraries,
            components=[
                component("area", f"{LEGACY}.area", **area_parameters),
                component("load", f"{LEGACY}.load", load="load"),
            ],
            data_series={"load": load},
        )
        study.connections.append(
            connection("area", "balance_port", "load", "balance_port")
        )
        return study

    def component(self, component_id: str, model: str, **parameters: Value) -> dict:
        """A component of a tested model, the same with both libraries."""
        library = LEGACY if self.is_reference else EXTENDED
        return component(component_id, f"{library}.{model}", **parameters)


REFERENCE = Flavour(is_reference=True)
EXTENDED_FLAVOUR = Flavour(is_reference=False)


def solve(study: GemsStudy, tmp_path: Path, name: str, exec_folder: Path) -> GemsResult:
    return study.solve(tmp_path.absolute() / name, exec_folder)


def connect_to_area(study: GemsStudy, comp: dict, port: str = "balance_port") -> None:
    study.components.append(comp)
    study.connections.append(connection("area", "balance_port", comp["id"], port))


def solve_both(
    build: Callable[[Flavour], GemsStudy], tmp_path: Path, exec_folder: Path
) -> tuple[GemsResult, GemsResult]:
    reference = solve(build(REFERENCE), tmp_path, "reference", exec_folder)
    extended = solve(build(EXTENDED_FLAVOUR), tmp_path, "extended", exec_folder)
    return reference, extended


def gas(flavour: Flavour, capacity: float = 1000, cost: float = 80) -> dict:
    return flavour.component(
        "gas",
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


# ---------------------------------------------------------------- thermal


def thermal_milp_study(flavour: Flavour, base_d_min_down: int = 8) -> GemsStudy:
    study = flavour.study()
    base_availability = np.where((HOURS >= 60) & (HOURS < 90), 600.0, 800.0)
    study.data_series["base_availability"] = base_availability
    connect_to_area(
        study,
        flavour.component(
            "base",
            "thermal_milp",
            minimum_generation_modulation=0.2,
            p_max_cluster="base_availability",
            p_min_unit=60,
            p_max_unit=100,
            generation_cost=20,
            startup_cost=100,
            fixed_cost=200,
            d_min_up=6,
            d_min_down=base_d_min_down,
            unit_count=8,
            spinning=0,
        ),
    )
    connect_to_area(
        study,
        flavour.component(
            "peak",
            "thermal_milp",
            minimum_generation_modulation=0,
            p_max_cluster=600,
            p_min_unit=40,
            p_max_unit=100,
            generation_cost=60,
            startup_cost=50,
            fixed_cost=100,
            d_min_up=3,
            d_min_down=2,
            unit_count=6,
            spinning=5,
        ),
    )
    return study


def test_thermal_milp(tmp_path: Path, antares_exec_folder: Path) -> None:
    reference, extended = solve_both(thermal_milp_study, tmp_path, antares_exec_folder)
    shorter_min_down = solve(
        thermal_milp_study(EXTENDED_FLAVOUR, base_d_min_down=7),
        tmp_path,
        "shorter_min_down",
        antares_exec_folder,
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert shorter_min_down.objective < extended.objective * (1 - 100 * REL_TOL)
    # The units are integer, and the units of the peak cluster start and stop
    units_on = extended.series("peak", "nb_units_on")
    assert units_on == pytest.approx(np.round(units_on), abs=1e-6)
    assert len(set(np.round(units_on))) > 2


def thermal_fast_study(flavour: Flavour, spinning: float = 10) -> GemsStudy:
    study = flavour.study()
    study.data_series["must_run_modulation"] = np.where(HOURS % 24 < 8, 0.5, 0.1)
    connect_to_area(
        study,
        flavour.component(
            "cheap",
            "thermal_fast",
            minimum_generation_modulation=0,
            p_max_cluster=700,
            p_min_unit=0,
            p_max_unit=100,
            generation_cost=20,
            unit_count=7,
            spinning=spinning,
            **UNUSED_THERMAL_FAST_PARAMETERS,
        ),
    )
    connect_to_area(
        study,
        flavour.component(
            "must_run",
            "thermal_fast",
            minimum_generation_modulation="must_run_modulation",
            p_max_cluster=900,
            p_min_unit=50,
            p_max_unit=100,
            generation_cost=60,
            unit_count=9,
            spinning=0,
            **UNUSED_THERMAL_FAST_PARAMETERS,
        ),
    )
    return study


def test_thermal_fast(tmp_path: Path, antares_exec_folder: Path) -> None:
    reference, extended = solve_both(thermal_fast_study, tmp_path, antares_exec_folder)
    without_spinning = solve(
        thermal_fast_study(EXTENDED_FLAVOUR, spinning=0),
        tmp_path,
        "without_spinning",
        antares_exec_folder,
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert extended.series("cheap", "generation") == pytest.approx(
        reference.series("cheap", "generation"), abs=1e-6
    )
    assert without_spinning.objective < extended.objective * (1 - 100 * REL_TOL)


# ---------------------------------------------------------------- storage

SOLAR = 1100 * SOLAR_SHAPE
STORAGE_LOAD = 400 + 100 * DAILY_SHAPE - SOLAR


def storage_study(
    flavour: Flavour,
    mccormick: int,
    mccormick_grand_maison: int,
    cycle_constraint: int,
    shift: int = 0,
) -> GemsStudy:
    """shift: in hours, of the load. The storage injects around noon and withdraws
    around midnight, the boundaries of the days of the daily cycle limits, when
    shift is 0, and conversely when shift is 12."""
    study = flavour.study(load=np.roll(STORAGE_LOAD, shift), spillage_cost=50)
    connect_to_area(study, gas(flavour))
    study.data_series["injection_modulation"] = np.roll(
        np.where(HOURS % 24 < 18, 1.0, 0.8), shift
    )
    connect_to_area(
        study,
        flavour.component(
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
            cycle_constraint=cycle_constraint,
        ),
        port="injection_port",
    )
    return study


# (mccormick, mccormick_grand_maison, cycle_constraint, shift)
STORAGE_CASES = {
    "no_cut": (0, 0, 0, 0),
    "mccormick": (1, 0, 0, 0),
    "grand_maison_mccormick": (0, 1, 0, 0),
    "both_mccormick": (1, 1, 0, 0),
    "daily_cycles": (0, 0, 1, 0),
    "daily_cycles_shifted": (0, 0, 1, 12),
    "mccormick_and_daily_cycles": (1, 0, 1, 0),
}


@pytest.mark.parametrize(
    "mccormick, mccormick_grand_maison, cycle_constraint, shift",
    STORAGE_CASES.values(),
    ids=STORAGE_CASES.keys(),
)
def test_short_term_storage(
    tmp_path: Path,
    antares_exec_folder: Path,
    mccormick: int,
    mccormick_grand_maison: int,
    cycle_constraint: int,
    shift: int,
) -> None:
    flags = (mccormick, mccormick_grand_maison, cycle_constraint)
    reference, extended = solve_both(
        lambda flavour: storage_study(flavour, *flags, shift=shift),
        tmp_path,
        antares_exec_folder,
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    if any(flags):
        no_cut = solve(
            storage_study(EXTENDED_FLAVOUR, 0, 0, 0, shift=shift),
            tmp_path,
            "no_cut",
            antares_exec_folder,
        )
        assert no_cut.objective < extended.objective * (1 - 100 * REL_TOL)


def test_grand_maison_mccormick_is_looser(
    tmp_path: Path, antares_exec_folder: Path
) -> None:
    usual = solve(
        storage_study(EXTENDED_FLAVOUR, 1, 0, 0), tmp_path, "usual", antares_exec_folder
    )
    grand_maison = solve(
        storage_study(EXTENDED_FLAVOUR, 0, 1, 0),
        tmp_path,
        "grand_maison",
        antares_exec_folder,
    )
    assert grand_maison.objective < usual.objective * (1 - 100 * REL_TOL)


# ---------------------------------------------------------------- curtailment

CURTAILMENT_LOAD = 400 + 100 * DAILY_SHAPE - SOLAR
NON_CURTAILABLE_SHARE = 0.2 + 0.4 * (HOURS % 24 >= 12)


def curtailment_study(flavour: Flavour, taux_non_ecretable: Value | None) -> GemsStudy:
    """taux_non_ecretable: None for no enr component."""
    study = flavour.study(load=CURTAILMENT_LOAD, spillage_cost=200)
    study.data_series["solar_load_factor"] = SOLAR_SHAPE
    study.data_series["non_curtailable_share"] = NON_CURTAILABLE_SHARE
    connect_to_area(study, gas(flavour))
    if taux_non_ecretable is not None:
        connect_to_area(
            study,
            flavour.component(
                "curtailment",
                "enr",
                generation_cost=-30,
                enr_total="solar_load_factor",
                enr_capacity=1100,
                taux_non_ecretable=taux_non_ecretable,
            ),
        )
    return study


def test_enr(tmp_path: Path, antares_exec_folder: Path) -> None:
    reference, extended = solve_both(
        lambda flavour: curtailment_study(flavour, "non_curtailable_share"),
        tmp_path,
        antares_exec_folder,
    )
    fully_curtailable = solve(
        curtailment_study(EXTENDED_FLAVOUR, 0), tmp_path, "fully", antares_exec_folder
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert extended.series("curtailment", "enr_ecrete") == pytest.approx(
        reference.series("curtailment", "enr_ecrete"), abs=1e-6
    )
    assert fully_curtailable.objective < extended.objective * (1 - 100 * REL_TOL)


BAND_COSTS = [-5, -10, -20, -40, -250]
BAND_SHARES = [0.1, 0.1, 0.15, 0.15, 0.3]
BAND_3_SHARE = BAND_SHARES[2] * (1 + DAILY_SHAPE) / 2


def multi_band_curtailment_study(flavour: Flavour) -> GemsStudy:
    study = curtailment_study(flavour, None)
    study.data_series["band_share_3"] = BAND_3_SHARE
    connect_to_area(
        study,
        flavour.component(
            "curtailment",
            "enr_multi_bandes",
            enr_fc_total="solar_load_factor",
            enr_capacity=1100,
            **{f"generation_cost_{i + 1}": cost for i, cost in enumerate(BAND_COSTS)},
            **{
                f"taux_ecretable_{i + 1}": "band_share_3" if i == 2 else share
                for i, share in enumerate(BAND_SHARES)
            },
        ),
    )
    return study


def test_enr_multi_bandes(tmp_path: Path, antares_exec_folder: Path) -> None:
    reference, extended = solve_both(
        multi_band_curtailment_study, tmp_path, antares_exec_folder
    )
    # The bands are independent: 5 enr components are equivalent.
    single_bands = curtailment_study(EXTENDED_FLAVOUR, None)
    for i, (cost, share) in enumerate(zip(BAND_COSTS, BAND_SHARES)):
        share_series = f"taux_non_ecretable_{i + 1}"
        band_share = BAND_3_SHARE if i == 2 else share * np.ones(NUM_TIME_STEPS)
        single_bands.data_series[share_series] = 1 - band_share
        connect_to_area(
            single_bands,
            EXTENDED_FLAVOUR.component(
                f"band_{i + 1}",
                "enr",
                generation_cost=cost,
                enr_total="solar_load_factor",
                enr_capacity=1100,
                taux_non_ecretable=share_series,
            ),
        )
    single_bands_result = solve(
        single_bands, tmp_path, "single_bands", antares_exec_folder
    )
    no_curtailment = solve(
        curtailment_study(EXTENDED_FLAVOUR, None),
        tmp_path,
        "no_curtailment",
        antares_exec_folder,
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert single_bands_result.objective == pytest.approx(
        extended.objective, rel=REL_TOL
    )
    # Bands 1 to 4 are cheaper than spillage, band 5 is not
    for band in range(1, 5):
        assert extended.series("curtailment", f"enr_ecrete_{band}").max() > 0
    assert extended.series("curtailment", "enr_ecrete_5").max() == pytest.approx(0)
    assert extended.objective < no_curtailment.objective * (1 - 100 * REL_TOL)


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
DAY = HOURS // 24
RIGHT_HAND_SIDES = {
    "hourly_less": 700 + 200 * DAILY_SHAPE,
    "hourly_greater": 800 + 0 * HOURS,
    "daily_less": 600 + 80 * DAY + 0 * HOURS,
    "weekly_less": 700 + 0 * HOURS,
    "weekly_greater": 1100 + 0 * HOURS,
}
# Model -> right hand side
BINDING_CONSTRAINT_CASES = {
    "bc_nuc_hourly_less": "hourly_less",
    "bc_nuc_hourly_greater": "hourly_greater",
    "bc_nuc_daily_less": "daily_less",
    "bc_nuc_weekly_less": "weekly_less",
    "bc_nuc_weekly_greater": "weekly_greater",
}


def nuclear_study(
    flavour: Flavour, model: str | None = None, rhs: str | None = None
) -> GemsStudy:
    """model: None for no binding constraint."""
    study = flavour.study(spillage_cost=10)
    for component_id, cost, capacity, _ in NUCLEAR_CLUSTERS.values():
        connect_to_area(
            study,
            flavour.component(
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
            ),
        )
    connect_to_area(study, gas(flavour, capacity=2000, cost=70))
    if model is None or rhs is None:
        return study
    alphas = {
        port.replace("flow_", "alpha_"): alpha
        for port, (*_, alpha) in NUCLEAR_CLUSTERS.items()
    }
    parameters: dict[str, Value] = {alpha: alphas.get(alpha, 1) for alpha in ALPHAS}
    parameters["rhs"] = rhs
    study.data_series[rhs] = RIGHT_HAND_SIDES[rhs]
    study.components.append(flavour.component("constraint", model, **parameters))
    for port, (component_id, *_) in NUCLEAR_CLUSTERS.items():
        study.connections.append(
            connection(component_id, "balance_port", "constraint", port)
        )
    return study


@pytest.mark.parametrize(
    "model, rhs",
    BINDING_CONSTRAINT_CASES.items(),
    ids=BINDING_CONSTRAINT_CASES.keys(),
)
def test_bc_nuc(
    tmp_path: Path, antares_exec_folder: Path, model: str, rhs: str
) -> None:
    reference, extended = solve_both(
        lambda flavour: nuclear_study(flavour, model, rhs),
        tmp_path,
        antares_exec_folder,
    )
    unconstrained = solve(
        nuclear_study(EXTENDED_FLAVOUR), tmp_path, "unconstrained", antares_exec_folder
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert unconstrained.objective < extended.objective * (1 - 100 * REL_TOL)


def test_daily_binding_constraint_is_tighter_than_weekly(
    tmp_path: Path, antares_exec_folder: Path
) -> None:
    """The daily constraint is not the weekly one: the days are constrained one by one."""
    daily = solve(
        nuclear_study(EXTENDED_FLAVOUR, "bc_nuc_daily_less", "daily_less"),
        tmp_path,
        "daily",
        antares_exec_folder,
    )
    weekly = solve(
        nuclear_study(EXTENDED_FLAVOUR, "bc_nuc_weekly_less", "daily_less"),
        tmp_path,
        "weekly",
        antares_exec_folder,
    )
    assert weekly.objective < daily.objective * (1 - 100 * REL_TOL)


# ---------------------------------------------------------------- same interface


def _models(library: Path) -> dict[str, dict]:
    with open(library, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return {model["id"]: model for model in content["library"]["models"]}


def _declarations(model: dict, key: str) -> list[tuple]:
    return [
        (item["id"], item.get("time-dependent"), item.get("scenario-dependent"))
        for item in model.get(key, [])
    ]


@pytest.mark.parametrize("model_id", sorted(_models(EXTENDED_LIBRARY)))
def test_same_ids_as_the_reference_library(model_id: str) -> None:
    """A system written for the reference library can use extended_models.yml, but
    for the library id: same models, parameters (and their dependencies), variables,
    ports and extra outputs (this does not need the Antares binary)."""
    model = _models(EXTENDED_LIBRARY)[model_id]
    reference = _models(REFERENCE_LIBRARY)[model_id]
    for key in ["parameters", "variables", "ports", "extra-outputs"]:
        assert _declarations(model, key) == _declarations(reference, key), key


# ---------------------------------------------------------------- daily windows


def _expressions(model: dict, prefix: str) -> list[str]:
    constraints = model.get("constraints", []) + model.get("binding-constraints", [])
    return [c["expression"] for c in constraints if c["id"].startswith(prefix)]


def _explicit_indices(expression: str, term: str) -> list[int]:
    """The time steps of the explicit sum term[i] + term[i+1] + ..."""
    return sorted(
        {int(i) for i in re.findall(rf"{re.escape(term)}\[(\d+)\]", expression)}
    )


def _window(expression: str, term: str) -> list[int]:
    """The time steps of the time sums of term: sum(i .. j, ...term...)."""
    windows = {
        (int(start), int(end))
        for start, end, operand in re.findall(
            r"sum\((\d+) \.\. (\d+), ([^<>]*?)\) *(?:<=|$)", expression
        )
        if term in operand
    }
    assert len(windows) == 1, expression
    start, end = windows.pop()
    return list(range(start, end + 1))


@pytest.mark.parametrize(
    "model, prefix, reference_term, term",
    [
        ("short-term-storage", "daily_pump_max", "p_injection", "p_injection"),
        ("short-term-storage", "daily_turb_max", "p_withdrawal", "p_withdrawal"),
        ("bc_nuc_daily_less", "bc_daily", "rhs", "rhs"),
        (
            "bc_nuc_daily_less",
            "bc_daily",
            "(sum_connections(flow_fr_nuclear_epr.flow))",
            "sum_connections(flow_fr_nuclear_epr.flow)",
        ),
    ],
)
def test_daily_windows_are_the_reference_ones(
    model: str, prefix: str, reference_term: str, term: str
) -> None:
    """The time sums cover the time steps of the explicit sums of the reference
    library, day by day (this does not need the Antares binary)."""
    reference = _expressions(_models(REFERENCE_LIBRARY)[model], prefix)
    expressions = _expressions(_models(EXTENDED_LIBRARY)[model], prefix)

    assert len(expressions) == len(reference) == 7
    for day, (reference_expression, expression) in enumerate(
        zip(reference, expressions)
    ):
        expected = list(range(24 * day, 24 * day + 24))
        assert _explicit_indices(reference_expression, reference_term) == expected
        assert _window(expression, term) == expected
