import os
from dataclasses import replace
from pathlib import Path
from time import time
from typing import Optional

import numpy as np
import pandas as pd
import pytest
from antares.craft import STStorageProperties

from antares_gems_converter.input_converter.src.logger import Logger
from tests.antares_historic.utils import (
    STS_TIMESERIES_SETTER_MAP,
    convert_study,
    createSTSTestAntaresStudy,
    first_optim_relgap,
)

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "SHORT TERM STORAGE"  : CONSTANT DATA  ##

LOAD_FILES_DIR = Path("tests/antares_historic/data")
STS_TEST_REL_ACCURACY = 1e-6
STS_TEST_SOLVER = "highs"
MODIFICATION_RATIO = 1.2
LOAD_TIME_SERIE_FILES_STS = [
    "load_matrix_1.txt",
    # "load_matrix_2.txt", #uncomment to test with different load profile
    # "load_matrix_original.txt",
]

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "SHORT TERM STORAGE"  : TESTS  ##

# General tests [OK:sts_list_general_test]

# Testing Boolean/discrete parameters
## initial_level_optim [TODO]
## enabled [TODO]
## penalize_variation_injection [TODO]
## penalize_variation_withdrawal [TODO]

# Testing Float parameters
## injection_nominal_capacity [OK : test_injection_nominal_capacity]
## withdrawal_nominal_capacity [OK: test_withdrawal_nominal_capacity]
## reservoir_capacity [OK: test_reservoir_capacity]
## efficiency [OK : test_efficiency]
## efficiency_withdrawal [OK : test_efficiency_withdrawal]
## initial_level [OK : test_initial_level]

# Testing Timeseries parameters
## p_max_injection [OK : test_p_max_injection]
## p_max_withdrawal [OK : test_p_max_withdrawal]
## lower_rule_curve [OK : test_lower_rule_curve]
## upper_rule_curve [OK : test_upper_rule_curve]
## storage_inflows [OK : test_storage_inflows]
## cost_injection [OK : test_cost_injection]
## cost_withdrawal [OK : test_cost_withdrawal]
## cost_level [OK : test_cost_level]
## cost_variation_injection [TODO]
## cost_variation_withdrawal [TODO]


def sts_test_procedure(
    study_name: str,
    study_path: Path,
    sts_properties: STStorageProperties,
    load_time_serie_file: Path,
    exec_folder: Path,
) -> None:
    createSTSTestAntaresStudy(
        study_name,
        study_path,
        load_time_serie_file,
        sts_properties,
    )
    original_study_path, converted_study_path = convert_study(
        study_path, study_name, ["short-term-storage"]
    )
    rel_gap = first_optim_relgap(
        exec_folder, original_study_path, converted_study_path, STS_TEST_SOLVER
    )
    assert rel_gap < STS_TEST_REL_ACCURACY


def sts_test_procedure_float_param(
    sts_properties: STStorageProperties,
    tested_param: str,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    # Run base test
    study_name_base = f"e2e_{str(int(100*time()))}"

    createSTSTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        sts_properties,
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["short-term-storage"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, STS_TEST_SOLVER
    )
    assert rel_gap < STS_TEST_REL_ACCURACY

    ref_value_param = getattr(sts_properties, tested_param)

    for modification in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        sts_properties_perturbated = replace(
            sts_properties, **{tested_param: ref_value_param * modification}
        )

        study_name_perturbated = f"e2e_{str(int(100*time()))}"

        createSTSTestAntaresStudy(
            study_name_perturbated,
            auto_generated_studies_path,
            LOAD_FILES_DIR / load_time_serie_file,
            sts_properties_perturbated,
        )
        rel_gap = first_optim_relgap(
            antares_exec_folder,
            auto_generated_studies_path / study_name_perturbated,
            converted_study_path,
            STS_TEST_SOLVER,
        )
        assert rel_gap > 100 * STS_TEST_REL_ACCURACY


@pytest.fixture(scope="session")
def sts_list_general_test() -> list[STStorageProperties]:
    return [
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=300,
            efficiency=1,
            initial_level=0.5,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=300,
            efficiency=0.8,
            initial_level=0.5,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=100,
            efficiency=0.8,
            initial_level=0.5,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=100,
            efficiency=0.8,
            initial_level=0.2,
        ),
        STStorageProperties(
            group="battery",
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=300,
            efficiency_withdrawal=0.9,
            efficiency=0.85,
            initial_level=0.5,
        ),
    ]  # ,


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_general_sts(
    sts_list_general_test: list[STStorageProperties],
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    for sts_properties in sts_list_general_test:
        study_name = f"e2e_general_test_{str(int(100*time()))}"
        sts_test_procedure(
            study_name,
            auto_generated_studies_path,
            sts_properties,
            LOAD_FILES_DIR / load_time_serie_file,
            antares_exec_folder,
        )


@pytest.mark.parametrize("injection_nominal_capacity_base", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_injection_nominal_capacity(
    injection_nominal_capacity_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=injection_nominal_capacity_base,
        withdrawal_nominal_capacity=200,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "injection_nominal_capacity",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("withdrawal_nominal_capacity_base", [50.0, 100.0, 200.0])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_withdrawal_nominal_capacity(
    withdrawal_nominal_capacity_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=200,
        withdrawal_nominal_capacity=withdrawal_nominal_capacity_base,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "withdrawal_nominal_capacity",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("reservoir_capacity_base", [100.0, 200.0, 300.0])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_reservoir_capacity(
    reservoir_capacity_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=reservoir_capacity_base,
        efficiency=1,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "reservoir_capacity",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("efficiency_base", [0.5, 0.6, 0.7])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_efficiency(
    efficiency_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=200,
        efficiency=efficiency_base,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "efficiency",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("efficiency_withdrawal_base", [0.5, 0.6, 0.7])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_efficiency_withdrawal(
    efficiency_withdrawal_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=200,
        efficiency_withdrawal=efficiency_withdrawal_base,
        efficiency=0.5 / MODIFICATION_RATIO,
        initial_level=0.5,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "efficiency_withdrawal",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("initial_level_base", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_initial_level(
    initial_level_base: float,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=80,
        withdrawal_nominal_capacity=80,
        reservoir_capacity=250,
        initial_level=initial_level_base,
    )

    sts_test_procedure_float_param(
        sts_properties,
        "initial_level",
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


## ---- Timeseries parameter test helpers ---- ##


def sts_test_procedure_timeseries_param(
    sts_properties: STStorageProperties,
    tested_param_key: str,
    base_timeseries: pd.DataFrame,
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
    sts_timeseries_extra: Optional[dict[str, pd.DataFrame]] = None,
) -> None:
    """Test that a timeseries parameter is correctly handled by the GEMS model.

    Creates a base study with the given timeseries, converts it, verifies near-zero
    relative gap, then creates a perturbed study with MODIFICATION_RATIO * base_timeseries
    and verifies the gap with the original converted study is significantly larger.
    """
    base_sts_timeseries = {tested_param_key: base_timeseries}
    if sts_timeseries_extra:
        base_sts_timeseries.update(sts_timeseries_extra)

    study_name_base = f"e2e_{str(int(100*time()))}"
    createSTSTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        sts_properties,
        sts_timeseries=base_sts_timeseries,
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["short-term-storage"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, STS_TEST_SOLVER
    )
    assert rel_gap < STS_TEST_REL_ACCURACY

    perturbed_timeseries = base_timeseries * MODIFICATION_RATIO
    perturbed_sts_timeseries = {tested_param_key: perturbed_timeseries}
    if sts_timeseries_extra:
        perturbed_sts_timeseries.update(sts_timeseries_extra)

    study_name_perturbed = f"e2e_{str(int(100*time()))}"
    createSTSTestAntaresStudy(
        study_name_perturbed,
        auto_generated_studies_path,
        LOAD_FILES_DIR / load_time_serie_file,
        sts_properties,
        sts_timeseries=perturbed_sts_timeseries,
    )
    rel_gap_perturbed = first_optim_relgap(
        antares_exec_folder,
        auto_generated_studies_path / study_name_perturbed,
        converted_study_path,
        STS_TEST_SOLVER,
    )
    assert rel_gap_perturbed > 100 * STS_TEST_REL_ACCURACY


## ---- Tests for timeseries parameters ---- ##


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_p_max_injection(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(0.5 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "pmax_injection",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_p_max_withdrawal(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(0.5 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "pmax_withdrawal",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_lower_rule_curve(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(0.1 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "lower_rule_curve",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_upper_rule_curve(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(0.8 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "upper_rule_curve",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_storage_inflows(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(5.0 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "storage_inflows",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


## ---- Tests for cost timeseries parameters ---- ##


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_cost_injection(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(5.0 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "cost_injection",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_cost_withdrawal(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(5.0 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "cost_withdrawal",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )


@pytest.mark.parametrize("load_time_serie_file", LOAD_TIME_SERIE_FILES_STS)
def test_cost_level(
    load_time_serie_file: str,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    sts_properties = STStorageProperties(
        group="battery",
        injection_nominal_capacity=100,
        withdrawal_nominal_capacity=100,
        reservoir_capacity=300,
        efficiency=1,
        initial_level=0.5,
    )
    base_timeseries = pd.DataFrame(1.0 * np.ones((8760, 1)))
    sts_test_procedure_timeseries_param(
        sts_properties,
        "cost_level",
        base_timeseries,
        load_time_serie_file,
        auto_generated_studies_path,
        antares_exec_folder,
    )
