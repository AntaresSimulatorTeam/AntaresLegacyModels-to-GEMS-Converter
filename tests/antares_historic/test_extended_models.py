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
(REFERENCE_LIBRARY, whose id is also antares_legacy_models). For each model, the same
study is built twice and solved with antares-modeler:
- the reference study only uses REFERENCE_LIBRARY;
- the extended study uses antares_legacy_models.yml (area, load) and
  extended_models.yml, with the renamed parameters.
Both must have the same optimal objective. Each test also checks that the modelled
feature is active in its study, i.e. that removing it changes the objective.
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

# New parameter id -> parameter id in REFERENCE_LIBRARY
THERMAL_RENAMES = {
    "cluster_min_gen_modulation": "minimum_generation_modulation",
    "cluster_max_generation": "p_max_cluster",
    "min_power_per_unit": "p_min_unit",
    "max_power_per_unit": "p_max_unit",
    "generation_cost": "generation_cost",
    "startup_cost": "startup_cost",
    "fixed_cost": "fixed_cost",
    "min_up_duration": "d_min_up",
    "min_down_duration": "d_min_down",
    "num_units": "unit_count",
    "spinning": "spinning",
}
STS_RENAMES = {
    "reservoir_capacity": "reservoir_capacity",
    "nominal_injection_capacity": "injection_nominal_capacity",
    "nominal_withdrawal_capacity": "withdrawal_nominal_capacity",
    "injection_efficiency": "efficiency_injection",
    "withdrawal_efficiency": "efficiency_withdrawal",
    "lower_rule_curve": "lower_rule_curve",
    "upper_rule_curve": "upper_rule_curve",
    "max_injection_modulation": "p_max_injection_modulation",
    "max_withdrawal_modulation": "p_max_withdrawal_modulation",
    "inflows": "inflows",
    "initial_level": "initial_level",
    "injection_cost": "cost_injection",
    "withdrawal_cost": "cost_withdrawal",
    "level_cost": "cost_level",
    "hurdle_cost": "p_t_cost",
    "max_daily_cycles": "nb_cycles_max",
    "is_cycle_limited": "cycle_constraint",
}
CURTAILMENT_RENAMES = {
    "generation_cost": "generation_cost",
    "load_factor": "enr_total",
    "installed_capacity": "enr_capacity",
    "non_curtailable_share": "taux_non_ecretable",
}
MULTI_BAND_CURTAILMENT_RENAMES = {
    "load_factor": "enr_fc_total",
    "installed_capacity": "enr_capacity",
    **{f"generation_cost_{i}": f"generation_cost_{i}" for i in range(1, 6)},
    **{f"curtailable_share_{i}": f"taux_ecretable_{i}" for i in range(1, 6)},
}


def renamed(parameters: dict[str, Value], renames: dict[str, str]) -> dict[str, Value]:
    return {renames[key]: value for key, value in parameters.items()}


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
        """A study with one area and its load."""
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

    def model(self, reference_model: str, model: str) -> str:
        return (
            f"{LEGACY}.{reference_model}"
            if self.is_reference
            else f"{EXTENDED}.{model}"
        )

    def thermal(self, component_id: str, variant: str, **parameters: Value) -> dict:
        """variant: milp or fast; parameters with their new ids."""
        if self.is_reference:
            parameters = renamed(parameters, THERMAL_RENAMES)
            if variant == "fast":  # Unused parameters, removed from thermal_fast
                parameters.update(
                    startup_cost=0, fixed_cost=0, d_min_up=1, d_min_down=1
                )
        model = self.model(f"thermal_{variant}", f"thermal_{variant}")
        return component(component_id, model, **parameters)

    def short_term_storage(
        self,
        component_id: str,
        is_mccormick_enabled: int,
        is_grand_maison_mccormick_enabled: int,
        withdrawal_power_at_max_injection: float,
        **parameters: Value,
    ) -> dict:
        """The McCormick cuts take the arguments of the reference model."""
        if self.is_reference:
            parameters = renamed(parameters, STS_RENAMES)
            parameters.update(
                mccormick=is_mccormick_enabled,
                mccormick_grand_maison=is_grand_maison_mccormick_enabled,
                p_withdrawal_when_injection_max=withdrawal_power_at_max_injection,
            )
        else:
            # Both reference cuts in one: the usual cut (0) is the tighter one.
            parameters["is_mccormick_enabled"] = max(
                is_mccormick_enabled, is_grand_maison_mccormick_enabled
            )
            parameters["withdrawal_power_at_max_injection"] = (
                0 if is_mccormick_enabled else withdrawal_power_at_max_injection
            )
        model = self.model("short-term-storage", "short_term_storage_extended")
        return component(component_id, model, **parameters)

    def curtailable_renewable(self, component_id: str, **parameters: Value) -> dict:
        if self.is_reference:
            parameters = renamed(parameters, CURTAILMENT_RENAMES)
        model = self.model("enr", "curtailable_renewable")
        return component(component_id, model, **parameters)

    def multi_band_curtailable_renewable(
        self, component_id: str, **parameters: Value
    ) -> dict:
        if self.is_reference:
            parameters = renamed(parameters, MULTI_BAND_CURTAILMENT_RENAMES)
        model = self.model("enr_multi_bandes", "multi_band_curtailable_renewable")
        return component(component_id, model, **parameters)


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


# ---------------------------------------------------------------- thermal


def thermal_milp_study(flavour: Flavour, base_min_down_duration: int = 8) -> GemsStudy:
    study = flavour.study()
    base_availability = np.where((HOURS >= 60) & (HOURS < 90), 600.0, 800.0)
    study.data_series["base_availability"] = base_availability
    connect_to_area(
        study,
        flavour.thermal(
            "base",
            "milp",
            cluster_min_gen_modulation=0.2,
            cluster_max_generation="base_availability",
            min_power_per_unit=60,
            max_power_per_unit=100,
            generation_cost=20,
            startup_cost=100,
            fixed_cost=200,
            min_up_duration=6,
            min_down_duration=base_min_down_duration,
            num_units=8,
            spinning=0,
        ),
    )
    connect_to_area(
        study,
        flavour.thermal(
            "peak",
            "milp",
            cluster_min_gen_modulation=0,
            cluster_max_generation=600,
            min_power_per_unit=40,
            max_power_per_unit=100,
            generation_cost=60,
            startup_cost=50,
            fixed_cost=100,
            min_up_duration=3,
            min_down_duration=2,
            num_units=6,
            spinning=5,
        ),
    )
    return study


def test_thermal_milp(tmp_path: Path, antares_exec_folder: Path) -> None:
    reference, extended = solve_both(thermal_milp_study, tmp_path, antares_exec_folder)
    shorter_min_down = solve(
        thermal_milp_study(EXTENDED_FLAVOUR, base_min_down_duration=7),
        tmp_path,
        "shorter_min_down",
        antares_exec_folder,
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert shorter_min_down.objective < extended.objective * (1 - 100 * REL_TOL)
    # The units are integer, and the units of the peak cluster start and stop
    units_on = extended.series("peak", "num_units_on")
    assert units_on == pytest.approx(np.round(units_on), abs=1e-6)
    assert len(set(np.round(units_on))) > 2


def thermal_fast_study(flavour: Flavour, spinning: float = 10) -> GemsStudy:
    study = flavour.study()
    study.data_series["must_run_modulation"] = np.where(HOURS % 24 < 8, 0.5, 0.1)
    connect_to_area(
        study,
        flavour.thermal(
            "cheap",
            "fast",
            cluster_min_gen_modulation=0,
            cluster_max_generation=700,
            min_power_per_unit=0,
            max_power_per_unit=100,
            generation_cost=20,
            num_units=7,
            spinning=spinning,
        ),
    )
    connect_to_area(
        study,
        flavour.thermal(
            "must_run",
            "fast",
            cluster_min_gen_modulation="must_run_modulation",
            cluster_max_generation=900,
            min_power_per_unit=50,
            max_power_per_unit=100,
            generation_cost=60,
            num_units=9,
            spinning=0,
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
    assert extended.series("cheap", "generation_power") == pytest.approx(
        reference.series("cheap", "generation"), abs=1e-6
    )
    assert without_spinning.objective < extended.objective * (1 - 100 * REL_TOL)


# ---------------------------------------------------------------- storage

SOLAR = 1100 * SOLAR_SHAPE
STORAGE_LOAD = 400 + 100 * DAILY_SHAPE - SOLAR


def storage_study(
    flavour: Flavour,
    is_mccormick_enabled: int,
    is_grand_maison_mccormick_enabled: int,
    is_cycle_limited: int,
    shift: int = 0,
) -> GemsStudy:
    """shift: in hours, of the load. The storage injects around noon and withdraws
    around midnight, the boundaries of the days of the daily cycle limits, when
    shift is 0, and conversely when shift is 12."""
    study = flavour.study(load=np.roll(STORAGE_LOAD, shift), spillage_cost=50)
    connect_to_area(
        study,
        flavour.thermal(
            "gas",
            "fast",
            cluster_min_gen_modulation=0,
            cluster_max_generation=1000,
            min_power_per_unit=0,
            max_power_per_unit=100,
            generation_cost=80,
            num_units=10,
            spinning=0,
        ),
    )
    study.data_series["injection_modulation"] = np.roll(
        np.where(HOURS % 24 < 18, 1.0, 0.8), shift
    )
    connect_to_area(
        study,
        flavour.short_term_storage(
            "storage",
            is_mccormick_enabled=is_mccormick_enabled,
            is_grand_maison_mccormick_enabled=is_grand_maison_mccormick_enabled,
            withdrawal_power_at_max_injection=100,
            reservoir_capacity=1000,
            nominal_injection_capacity=300,
            nominal_withdrawal_capacity=300,
            injection_efficiency=0.8,
            withdrawal_efficiency=1,
            lower_rule_curve=0.1,
            upper_rule_curve=0.9,
            max_injection_modulation="injection_modulation",
            max_withdrawal_modulation=1,
            inflows=0,
            initial_level=0.5,
            injection_cost=0,
            withdrawal_cost=0,
            level_cost=0.01,
            hurdle_cost=0.5,
            max_daily_cycles=0.5,
            is_cycle_limited=is_cycle_limited,
        ),
        port="injection_port" if flavour.is_reference else "balance_port",
    )
    return study


# (McCormick cut, Grand'Maison McCormick cut, daily cycle limits, shift)
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
    "is_mccormick_enabled, is_grand_maison_mccormick_enabled, is_cycle_limited, shift",
    STORAGE_CASES.values(),
    ids=STORAGE_CASES.keys(),
)
def test_short_term_storage_extended(
    tmp_path: Path,
    antares_exec_folder: Path,
    is_mccormick_enabled: int,
    is_grand_maison_mccormick_enabled: int,
    is_cycle_limited: int,
    shift: int,
) -> None:
    flags = (is_mccormick_enabled, is_grand_maison_mccormick_enabled, is_cycle_limited)
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


def curtailment_study(flavour: Flavour, non_curtailable_share: Value) -> GemsStudy:
    study = flavour.study(load=CURTAILMENT_LOAD, spillage_cost=200)
    study.data_series["solar_load_factor"] = SOLAR_SHAPE
    study.data_series["non_curtailable_share"] = NON_CURTAILABLE_SHARE
    connect_to_area(
        study,
        flavour.thermal(
            "gas",
            "fast",
            cluster_min_gen_modulation=0,
            cluster_max_generation=1000,
            min_power_per_unit=0,
            max_power_per_unit=100,
            generation_cost=80,
            num_units=10,
            spinning=0,
        ),
    )
    connect_to_area(
        study,
        flavour.curtailable_renewable(
            "curtailment",
            generation_cost=-30,
            load_factor="solar_load_factor",
            installed_capacity=1100,
            non_curtailable_share=non_curtailable_share,
        ),
    )
    return study


def test_curtailable_renewable(tmp_path: Path, antares_exec_folder: Path) -> None:
    reference, extended = solve_both(
        lambda flavour: curtailment_study(flavour, "non_curtailable_share"),
        tmp_path,
        antares_exec_folder,
    )
    fully_curtailable = solve(
        curtailment_study(EXTENDED_FLAVOUR, 0), tmp_path, "fully", antares_exec_folder
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert extended.series("curtailment", "curtailed_power") == pytest.approx(
        reference.series("curtailment", "enr_ecrete"), abs=1e-6
    )
    assert fully_curtailable.objective < extended.objective * (1 - 100 * REL_TOL)


BAND_COSTS = [-5, -10, -20, -40, -250]
BAND_SHARES = [0.1, 0.1, 0.15, 0.15, 0.3]


def multi_band_curtailment_study(flavour: Flavour) -> GemsStudy:
    study = curtailment_study(flavour, 0)
    study.components.pop()  # The single band curtailment
    study.connections.pop()
    study.data_series["band_share_3"] = BAND_SHARES[2] * (1 + DAILY_SHAPE) / 2
    connect_to_area(
        study,
        flavour.multi_band_curtailable_renewable(
            "curtailment",
            load_factor="solar_load_factor",
            installed_capacity=1100,
            **{f"generation_cost_{i + 1}": cost for i, cost in enumerate(BAND_COSTS)},
            **{
                f"curtailable_share_{i + 1}": "band_share_3" if i == 2 else share
                for i, share in enumerate(BAND_SHARES)
            },
        ),
    )
    return study


def test_multi_band_curtailable_renewable(
    tmp_path: Path, antares_exec_folder: Path
) -> None:
    reference, extended = solve_both(
        multi_band_curtailment_study, tmp_path, antares_exec_folder
    )
    # The bands are independent: 5 single band curtailments are equivalent.
    single_bands = curtailment_study(EXTENDED_FLAVOUR, 0)
    single_bands.components.pop()
    single_bands.connections.pop()
    for i, (cost, share) in enumerate(zip(BAND_COSTS, BAND_SHARES)):
        share_series = f"non_curtailable_share_{i + 1}"
        share_values = BAND_SHARES[2] * (1 + DAILY_SHAPE) / 2 if i == 2 else share
        single_bands.data_series[share_series] = 1 - share_values * np.ones(
            NUM_TIME_STEPS
        )
        connect_to_area(
            single_bands,
            EXTENDED_FLAVOUR.curtailable_renewable(
                f"band_{i + 1}",
                generation_cost=cost,
                load_factor="solar_load_factor",
                installed_capacity=1100,
                non_curtailable_share=share_series,
            ),
        )
    single_bands_result = solve(
        single_bands, tmp_path, "single_bands", antares_exec_folder
    )
    no_curtailment = curtailment_study(EXTENDED_FLAVOUR, 1)
    no_curtailment_result = solve(
        no_curtailment, tmp_path, "no_curtailment", antares_exec_folder
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert single_bands_result.objective == pytest.approx(
        extended.objective, rel=REL_TOL
    )
    # Bands 1 to 4 are cheaper than spillage, band 5 is not
    for band in range(1, 5):
        assert extended.series("curtailment", f"curtailed_power_{band}").max() > 0
    assert extended.series("curtailment", "curtailed_power_5").max() == pytest.approx(0)
    assert extended.objective < no_curtailment_result.objective * (1 - 100 * REL_TOL)


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
# Reference model -> (model, direction, right hand side)
BINDING_CONSTRAINT_CASES = {
    "bc_nuc_hourly_less": ("nuclear_hourly_binding_constraint", 1, "hourly_less"),
    "bc_nuc_hourly_greater": (
        "nuclear_hourly_binding_constraint",
        -1,
        "hourly_greater",
    ),
    "bc_nuc_daily_less": ("nuclear_daily_binding_constraint", 1, "daily_less"),
    "bc_nuc_weekly_less": ("nuclear_weekly_binding_constraint", 1, "weekly_less"),
    "bc_nuc_weekly_greater": (
        "nuclear_weekly_binding_constraint",
        -1,
        "weekly_greater",
    ),
}


def nuclear_study(
    flavour: Flavour,
    binding_constraint: tuple[str, str, int, str] | None,
) -> GemsStudy:
    """binding_constraint: (reference model, model, direction, right hand side)."""
    study = flavour.study(spillage_cost=10)
    for port, (component_id, cost, capacity, _) in NUCLEAR_CLUSTERS.items():
        connect_to_area(
            study,
            flavour.thermal(
                component_id,
                "fast",
                cluster_min_gen_modulation=0,
                cluster_max_generation=capacity,
                min_power_per_unit=0,
                max_power_per_unit=100,
                generation_cost=cost,
                num_units=capacity / 100,
                spinning=0,
            ),
        )
    connect_to_area(
        study,
        flavour.thermal(
            "gas",
            "fast",
            cluster_min_gen_modulation=0,
            cluster_max_generation=2000,
            min_power_per_unit=0,
            max_power_per_unit=100,
            generation_cost=70,
            num_units=20,
            spinning=0,
        ),
    )
    if binding_constraint is None:
        return study
    reference_model, model, direction, rhs = binding_constraint
    alphas = {
        port.replace("flow_", "alpha_"): alpha
        for port, (*_, alpha) in NUCLEAR_CLUSTERS.items()
    }
    parameters: dict[str, Value] = {alpha: alphas.get(alpha, 1) for alpha in ALPHAS}
    parameters["rhs"] = rhs
    if not flavour.is_reference:
        parameters["direction"] = direction
    study.data_series[rhs] = RIGHT_HAND_SIDES[rhs]
    study.components.append(
        component("constraint", flavour.model(reference_model, model), **parameters)
    )
    for port, (component_id, *_) in NUCLEAR_CLUSTERS.items():
        study.connections.append(
            connection(component_id, "balance_port", "constraint", port)
        )
    return study


@pytest.mark.parametrize(
    "reference_model, model, direction, rhs",
    [(key, *value) for key, value in BINDING_CONSTRAINT_CASES.items()],
    ids=BINDING_CONSTRAINT_CASES.keys(),
)
def test_nuclear_binding_constraint(
    tmp_path: Path,
    antares_exec_folder: Path,
    reference_model: str,
    model: str,
    direction: int,
    rhs: str,
) -> None:
    binding_constraint = (reference_model, model, direction, rhs)
    reference, extended = solve_both(
        lambda flavour: nuclear_study(flavour, binding_constraint),
        tmp_path,
        antares_exec_folder,
    )
    unconstrained = solve(
        nuclear_study(EXTENDED_FLAVOUR, None),
        tmp_path,
        "unconstrained",
        antares_exec_folder,
    )

    assert extended.objective == pytest.approx(reference.objective, rel=REL_TOL)
    assert unconstrained.objective < extended.objective * (1 - 100 * REL_TOL)


def test_daily_binding_constraint_is_tighter_than_weekly(
    tmp_path: Path, antares_exec_folder: Path
) -> None:
    """The daily constraint is not the weekly one: the days are constrained one by one."""
    daily = solve(
        nuclear_study(
            EXTENDED_FLAVOUR,
            ("bc_nuc_daily_less", "nuclear_daily_binding_constraint", 1, "daily_less"),
        ),
        tmp_path,
        "daily",
        antares_exec_folder,
    )
    weekly = solve(
        nuclear_study(
            EXTENDED_FLAVOUR,
            (
                "bc_nuc_weekly_less",
                "nuclear_weekly_binding_constraint",
                1,
                "daily_less",
            ),
        ),
        tmp_path,
        "weekly",
        antares_exec_folder,
    )
    assert weekly.objective < daily.objective * (1 - 100 * REL_TOL)


# ---------------------------------------------------------------- daily windows


def _models(library: Path) -> dict[str, dict]:
    with open(library, encoding="utf-8") as f:
        content = yaml.safe_load(f)
    return {model["id"]: model for model in content["library"]["models"]}


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
    "reference_model, reference_prefix, reference_term, model, prefix, term",
    [
        (
            "short-term-storage",
            "daily_pump_max",
            "p_injection",
            "short_term_storage_extended",
            "max_daily_injection",
            "injection_power",
        ),
        (
            "short-term-storage",
            "daily_turb_max",
            "p_withdrawal",
            "short_term_storage_extended",
            "max_daily_withdrawal",
            "withdrawal_power",
        ),
        (
            "bc_nuc_daily_less",
            "bc_daily",
            "rhs",
            "nuclear_daily_binding_constraint",
            "daily",
            "rhs",
        ),
        (
            "bc_nuc_daily_less",
            "bc_daily",
            "(sum_connections(flow_fr_nuclear_epr.flow))",
            "nuclear_daily_binding_constraint",
            "daily",
            "sum_connections(flow_fr_nuclear_epr.flow)",
        ),
    ],
)
def test_daily_windows_are_the_reference_ones(
    reference_model: str,
    reference_prefix: str,
    reference_term: str,
    model: str,
    prefix: str,
    term: str,
) -> None:
    """The time sums cover the time steps of the explicit sums of the reference
    library, day by day (this does not need the Antares binary)."""
    reference = _expressions(
        _models(REFERENCE_LIBRARY)[reference_model], reference_prefix
    )
    expressions = _expressions(_models(EXTENDED_LIBRARY)[model], prefix)

    assert len(expressions) == len(reference) == 7
    for day, (reference_expression, expression) in enumerate(
        zip(reference, expressions)
    ):
        expected = list(range(24 * day, 24 * day + 24))
        assert _explicit_indices(reference_expression, reference_term) == expected
        assert _window(expression, term) == expected
