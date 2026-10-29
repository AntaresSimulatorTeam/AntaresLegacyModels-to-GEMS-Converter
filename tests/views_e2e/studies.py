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
"""Legacy studies the views are compared on, built with antares-craft."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from antares.craft import (
    AreaProperties,
    GeneralParametersUpdate,
    LinkPropertiesUpdate,
    OptimizationParametersUpdate,
    RenewableClusterGroup,
    RenewableClusterProperties,
    STStorageGroup,
    STStorageProperties,
    Study,
    StudySettingsUpdate,
    ThermalClusterGroup,
    ThermalClusterProperties,
    create_study_local,
)

STUDY_VERSION = "9.3"
HOURS = 8760
NB_YEARS = 2


def _daily_profile(base: float, amplitude: float, phase: float = 0) -> np.ndarray:
    hours = np.arange(HOURS)
    return base + amplitude * np.sin(2 * np.pi * (hours - phase) / 24)


def two_areas(parent_dir: Path) -> Study:
    """Two areas linked by a congested link with hurdle costs, two MC years.

    fr: thermal clusters of three groups (one with emissions, one with
    unit-commitment costs), a battery and must-run misc generation; its load
    peaks above its capacity plus imports, so that some hours have unsupplied
    energy. de: thermal and renewable clusters, the latter exceeding the load at
    night in year 2, so that some hours have spilled energy.
    """
    study = create_study_local("two_areas", STUDY_VERSION, parent_dir)
    study.update_settings(
        StudySettingsUpdate(
            general_parameters=GeneralParametersUpdate(
                nb_years=NB_YEARS, year_by_year=True
            ),
            optimization_parameters=OptimizationParametersUpdate(
                include_hurdlecosts=True
            ),
        )
    )

    fr = study.create_area(
        "fr",
        properties=AreaProperties(energy_cost_unsupplied=3000, energy_cost_spilled=1),
    )
    fr_load = _daily_profile(550, 150)
    fr.set_load(pd.DataFrame(np.stack([fr_load, fr_load * 1.1], axis=1)))
    fr.set_misc_gen(pd.DataFrame(np.tile([20, 10, 0, 0, 0, 0, 5, -10], (HOURS, 1))))
    for name, group, units, unit_capacity, cost, extra in [
        ("nuke", ThermalClusterGroup.NUCLEAR, 2, 150, 10, {}),
        ("gas", ThermalClusterGroup.GAS, 3, 50, 60, {"co2": 0.4, "nox": 0.1}),
        (
            "peaker",
            ThermalClusterGroup.OIL,
            2,
            50,
            150,
            {"startup_cost": 500, "fixed_cost": 100, "min_stable_power": 20},
        ),
    ]:
        cluster = fr.create_thermal_cluster(
            name,
            ThermalClusterProperties(
                group=group,
                unit_count=units,
                nominal_capacity=unit_capacity,
                marginal_cost=cost,
                market_bid_cost=cost,
                **extra,
            ),
        )
        cluster.set_series(pd.DataFrame(units * unit_capacity * np.ones((HOURS, 1))))
    fr.create_st_storage(
        "battery",
        STStorageProperties(
            group=STStorageGroup.BATTERY.value,
            injection_nominal_capacity=50,
            withdrawal_nominal_capacity=50,
            reservoir_capacity=200,
            efficiency=0.9,
            initial_level=0.5,
        ),
    )

    de = study.create_area(
        "de",
        properties=AreaProperties(energy_cost_unsupplied=3000, energy_cost_spilled=1),
    )
    de_load = _daily_profile(300, 100, phase=6)
    de.set_load(pd.DataFrame(np.stack([de_load, de_load * 0.8], axis=1)))
    for name, group, capacity, cost in [
        ("lignite", ThermalClusterGroup.LIGNITE, 300, 20),
        ("ccgt", ThermalClusterGroup.GAS, 200, 45),
    ]:
        cluster = de.create_thermal_cluster(
            name,
            ThermalClusterProperties(
                group=group,
                unit_count=1,
                nominal_capacity=capacity,
                marginal_cost=cost,
                market_bid_cost=cost,
            ),
        )
        cluster.set_series(pd.DataFrame(capacity * np.ones((HOURS, 1))))
    for name, group, profile in [
        ("onshore", RenewableClusterGroup.WIND_ON_SHORE, _daily_profile(120, 60, 3)),
        (
            "pv",
            RenewableClusterGroup.PV_SOLAR,
            np.maximum(_daily_profile(0, 150, 6), 0),
        ),
    ]:
        cluster = de.create_renewable_cluster(
            name,
            RenewableClusterProperties(
                group=group.value, unit_count=1, nominal_capacity=300
            ),
        )
        cluster.set_series(pd.DataFrame(np.stack([profile, profile * 1.5], axis=1)))

    link = study.create_link(area_from="de", area_to="fr")
    link.update_properties(LinkPropertiesUpdate(hurdles_cost=True))
    link.set_capacity_direct(pd.DataFrame(150 * np.ones((HOURS, 1))))
    link.set_capacity_indirect(pd.DataFrame(100 * np.ones((HOURS, 1))))
    parameters = np.zeros((HOURS, 6))
    parameters[:, 0] = 0.5  # direct hurdle cost
    parameters[:, 1] = 0.5  # indirect hurdle cost
    link.set_parameters(pd.DataFrame(parameters))
    return study


STUDIES: dict[str, Callable[[Path], Study]] = {"two_areas": two_areas}
