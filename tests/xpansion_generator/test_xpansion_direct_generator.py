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
"""Files written by the Xpansion direct generator (no Antares binary needed)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml
from gems_craft_hybrid.study.parsing import parse_yaml_hybrid_system

from antares_gems_converter.xpansion_generator.xpansion_direct_generator import (
    generate,
)
from antares_gems_converter.xpansion_generator.xpansion_direct_generator_lib import (
    BATTERY_NOT_SUPPORTED,
)
from tests.xpansion_generator.utils import (
    BE_OCGT,
    CCGT_COST_PER_UNIT,
    CCGT_UNIT_SIZE,
    DE_CCGT,
    FR_CCGT,
    FR_OCGT,
    HOURS,
    OCGT_COST_PER_UNIT,
    OCGT_MAX_INVESTMENT,
    OCGT_UNIT_SIZE,
    PEAK_FR,
    SEMIBASE_DE,
    SEMIBASE_FR,
    create_base_study,
    run_generator,
    write_excel_inputs,
)


LIBRARY = "antares_legacy_investment_candidates"
CONTINUOUS_MODEL = f"{LIBRARY}.continuous_investment_candidate"
DISCRETE_MODEL = f"{LIBRARY}.discrete_investment_candidate"


@pytest.fixture
def generated(tmp_path: Path) -> Path:
    return run_generator(tmp_path)[1]


def _scenario_builder(output_dir: Path) -> dict[str, dict[int, int]]:
    groups: dict[str, dict[int, int]] = {}
    lines = (output_dir / "data-series" / "modeler-scenariobuilder.dat").read_text()
    for line in lines.splitlines():
        group_year, ts = line.split("=")
        group, year = group_year.split(",")
        groups.setdefault(group.strip(), {})[int(year)] = int(ts)
    return groups


def test_system_yml_is_a_gemspy_hybrid_system(generated: Path) -> None:
    with (generated / "system.yml").open() as file:
        system = parse_yaml_hybrid_system(file)

    components = {component.id: component for component in system.components}
    assert list(components) == [PEAK_FR, SEMIBASE_FR, SEMIBASE_DE]
    for component_id, component in components.items():
        assert component.model == DISCRETE_MODEL
        # one scenario group per component: series differ in column count
        assert component.scenario_group == component_id

    def parameters(component_id: str) -> dict[str, float | str]:
        return {p.id: p.value for p in components[component_id].parameters or []}

    assert parameters(PEAK_FR) == {
        "unit_size": OCGT_UNIT_SIZE,
        # installed capacity of fr_ocgt ch4 pcomp
        "existing_capacity": 50,
        "min_investment": 0,
        # 10 units of 50 MW
        "max_investment": 500,
        # per MW: the inputs give it per unit
        "annual_investment_cost": pytest.approx(OCGT_COST_PER_UNIT / OCGT_UNIT_SIZE),
        "availability_rate": f"availability_rate_{PEAK_FR}",
    }
    assert parameters(SEMIBASE_DE)["max_investment"] == 300
    assert parameters(SEMIBASE_DE)["annual_investment_cost"] == pytest.approx(
        CCGT_COST_PER_UNIT / CCGT_UNIT_SIZE
    )

    assert [
        (c.component, c.port, c.thermal_component.area, c.thermal_component.cluster_id)
        for c in system.thermal_capacity_connections or []
    ] == [
        (PEAK_FR, "capacity_port", "fr", FR_OCGT),
        (SEMIBASE_FR, "capacity_port", "fr", FR_CCGT),
        (SEMIBASE_DE, "capacity_port", "de", DE_CCGT),
    ]


def test_availability_series_are_tsv_files(generated: Path) -> None:
    series_dir = generated / "data-series"
    assert sorted(path.name for path in series_dir.iterdir()) == sorted(
        [f"availability_rate_{c}.tsv" for c in [PEAK_FR, SEMIBASE_FR, SEMIBASE_DE]]
        + ["modeler-scenariobuilder.dat"]
    )
    coef = pd.read_csv(
        series_dir / f"availability_rate_{SEMIBASE_FR}.tsv", sep="\t", header=None
    )
    assert coef.shape == (HOURS, 1)
    assert coef[0].tolist() == pytest.approx([0.9] * HOURS)


def _decomposition(model: str, variable: str) -> dict:
    return {
        "id": model,
        "model-decomposition": {
            "variables": [{"id": variable, "location": "master-and-subproblems"}],
            "objective-contributions": [{"id": "objective", "location": "master"}],
        },
    }


def test_optim_config_puts_investment_in_master(generated: Path) -> None:
    optim_config = yaml.safe_load((generated / "optim-config.yml").read_text())
    # only the models used: every candidate is discrete
    assert optim_config == {
        "resolution-mode": "benders-decomposition",
        "models": [_decomposition(DISCRETE_MODEL, "invested_units")],
    }


def test_candidate_without_unit_count_is_continuous(tmp_path: Path) -> None:
    _, output_dir = run_generator(tmp_path, ocgt_units=None)

    with (output_dir / "system.yml").open() as file:
        system = parse_yaml_hybrid_system(file)
    components = {component.id: component for component in system.components}
    assert {c: components[c].model for c in components} == {
        PEAK_FR: CONTINUOUS_MODEL,
        SEMIBASE_FR: DISCRETE_MODEL,
        SEMIBASE_DE: DISCRETE_MODEL,
    }
    assert {p.id: p.value for p in components[PEAK_FR].parameters or []} == {
        "existing_capacity": 50,
        "min_investment": 0,
        # gisement_ini, in MW
        "max_investment": OCGT_MAX_INVESTMENT,
        "annual_investment_cost": pytest.approx(OCGT_COST_PER_UNIT / OCGT_UNIT_SIZE),
        "availability_rate": f"availability_rate_{PEAK_FR}",
    }

    optim_config = yaml.safe_load((output_dir / "optim-config.yml").read_text())
    assert optim_config["models"] == [
        _decomposition(CONTINUOUS_MODEL, "invested_capacity"),
        _decomposition(DISCRETE_MODEL, "invested_units"),
    ]


def test_scenario_builder_follows_legacy_thermal_scenario_builder(
    tmp_path: Path,
) -> None:
    write_excel_inputs(tmp_path / "xlsx", ["fr", "de"])
    study = create_base_study(tmp_path, nb_years=2, fr_ccgt_availability=[90, 80])
    legacy_sb = study.get_scenario_builder()
    legacy_sb.thermal.get_cluster("fr", FR_CCGT).set_new_scenario([2, None])
    study.set_scenario_builder(legacy_sb)

    output_dir = tmp_path / "generated"
    generate(tmp_path / "xlsx", tmp_path / "base", output_dir)

    assert _scenario_builder(output_dir) == {
        # year 0 from the legacy SB, year 1 (random in Antares) cycles the columns
        SEMIBASE_FR: {0: 2, 1: 2},
        PEAK_FR: {0: 1, 1: 1},
        SEMIBASE_DE: {0: 1, 1: 1},
    }
    coef = pd.read_csv(
        output_dir / "data-series" / f"availability_rate_{SEMIBASE_FR}.tsv",
        sep="\t",
        header=None,
    )
    assert coef.iloc[0].tolist() == pytest.approx([0.9, 0.8])


def test_skipped_candidates_are_logged(tmp_path: Path) -> None:
    _, output_dir = run_generator(
        tmp_path, areas=["fr", "de", "it", "be"], with_disabled_cluster=True
    )

    system = yaml.safe_load((output_dir / "system.yml").read_text())["system"]
    assert set(system) == {"components", "id", "thermal-capacity-connections"}
    assert [c["id"] for c in system["components"]] == [
        PEAK_FR,
        SEMIBASE_FR,
        SEMIBASE_DE,
    ]

    log = (output_dir / "log_composants_system_2031.txt").read_text()
    for component, reason in [
        ("candidat_invest_peak_be", "cluster désactivé (enabled = false)"),
        ("candidat_invest_peak_de", "cluster vide (capacité installée nulle)"),
        ("candidat_invest_peak_it", "cluster absent du list.ini"),
        ("candidat_invest_battery_fr", BATTERY_NOT_SUPPORTED),
        ("candidat_invest_battery_de", "short-term storage absent du list.ini"),
    ]:
        line = next(line for line in log.splitlines() if f"- {component} " in line)
        assert reason in line
    assert BE_OCGT in log
