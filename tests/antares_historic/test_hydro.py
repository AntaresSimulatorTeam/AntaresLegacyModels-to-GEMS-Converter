from pathlib import Path
from time import time

import numpy as np
import pytest
import pandas as pd

from tests.antares_historic.utils import (
    convert_study,
    createHydroTestAntaresStudy,
    first_optim_relgap,
)
from antares.craft import HydroPropertiesUpdate

## TESTING PROCEDURE FOR GEMS MODEL REPRESENTING ANTARES v9.3 "HYDRO" : CONSTANT DATA  ##

LOAD_FILES_DIR = Path("tests/antares_historic/data")
TEST_REL_ACCURACY = 5e-6 # this is high beacause there is noise on pumping and turbining variables in Antares that we can't reproduce
TEST_SOLVER = "highs"
MODIFICATION_RATIO = 1.2

## TESTING PROCEDURE FOR GEMS MODEL REPRESENTING ANTARES v9.3 "HYDRO" : TESTS  ##

# Testing Float properties (HydroProperties via update_properties)
## inter_daily_breakdown: [TODO]
## intra_daily_modulation: [TODO]
## inter_monthly_breakdown: [TODO]
## reservoir_capacity: [OK : test_reservoir_capacity]
## leeway_low: [TODO]
## leeway_up: [TODO]
## pumping_efficiency: [OK : test_pumping_efficiency]
## overflow_spilled_cost_difference: [OK : test_overflow_cost]

# Testing Boolean properties (HydroProperties via update_properties)
## reservoir: [TODO]
## follow_load: [TODO]
## use_water: [TODO]
## hard_bounds: [TODO]
## use_heuristic: [TODO]
## power_to_level: [TODO]
## use_leeway: [TODO]

# Testing Integer properties (HydroProperties via update_properties)
## initialize_reservoir_date: [TODO]

# Testing Timeseries parameters
## ror_series (set_ror_series, 8760 rows x N cols): [OK : test_ror]
## maxpower col 0 - generating max power (set_maxpower, 365 rows x 4 cols): [OK : test_nominal_generation_capacity]
## maxpower col 1 - generating max power per unit (set_maxpower, 365 rows x 4 cols): [TODO]
## maxpower col 2 - pumping max power (set_maxpower, 365 rows x 4 cols): [OK : test_nominal_pumping_capacity]
## maxpower col 3 - pumping max power per unit (set_maxpower, 365 rows x 4 cols): [TODO]
## mingen - minimum generation (set_mingen, 365 rows): [OK : test_minimum_generation]
## reservoir col 0 - lower rule curve (set_reservoir, 365 rows x 3 cols): [TODO]
## reservoir col 1 - initial and final level (set_reservoir, 365 rows x 3 cols): [TODO]
## reservoir col 2 - upper rule curve (set_reservoir, 365 rows x 3 cols): [TODO]
## mod_series - inflows (set_mod_series, 365 rows): [OK : test_inflows]
## credit_modulations (set_credits_modulation): [TODO]
## energy (set_energy): [TODO]
## water_values (set_water_values): [TODO]


@pytest.mark.parametrize("ror", [100])
def test_ror(
    ror: float, auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    study_name = f"ror_test_study_{str(int(100*time()))}"
    load_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"
    np.random.seed(1)  # for reproducibility
    ror_timeseries = pd.DataFrame(ror * np.random.random((8760, 1)))

    createHydroTestAntaresStudy(
        study_name,
        auto_generated_studies_path,
        load_time_serie_file,
        ror_timeseries,
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name, ["ror"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for modification in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"ror_test_study_{str(int(100*time()))}"
        ror_perturbated = ror_timeseries * modification
        createHydroTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            ror_perturbated,
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY


@pytest.mark.parametrize("base_reservoir_capacity", [500, 1000, 5000])
def test_reservoir_capacity(
    base_reservoir_capacity: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    load_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"

    study_name_base = f"hydro_reservoir_base_{str(int(100*time()))}"
    createHydroTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        load_time_serie_file,
        hydro_properties=HydroPropertiesUpdate(reservoir_capacity=base_reservoir_capacity),
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["hydro"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"hydro_reservoir_{str(int(100*time()))}"
        createHydroTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            hydro_properties=HydroPropertiesUpdate(reservoir_capacity=base_reservoir_capacity * perturbation),
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY


@pytest.mark.parametrize("base_pumping_efficiency", [0.75])
def test_pumping_efficiency(
    base_pumping_efficiency: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    load_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"

    study_name_base = f"hydro_pumpeff_base_{str(int(100*time()))}"
    createHydroTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        load_time_serie_file,
        hydro_properties=HydroPropertiesUpdate(pumping_efficiency=base_pumping_efficiency),
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["hydro"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_efficiency = base_pumping_efficiency * perturbation
        perturbed_study_name = f"hydro_pumpeff_{str(int(100*time()))}"
        createHydroTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            hydro_properties=HydroPropertiesUpdate(pumping_efficiency=perturbed_efficiency),
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY


@pytest.mark.parametrize("base_overflow_cost", [1])
def test_overflow_cost(
    base_overflow_cost: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    load_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"

    study_name_base = f"hydro_overflow_base_{str(int(100*time()))}"
    createHydroTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        load_time_serie_file,
        hydro_properties=HydroPropertiesUpdate(overflow_spilled_cost_difference=base_overflow_cost),
        inflows=pd.DataFrame(np.ones((365,1))*100000)
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["hydro"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"hydro_overflow_{str(int(100*time()))}"
        createHydroTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            hydro_properties=HydroPropertiesUpdate(overflow_spilled_cost_difference=base_overflow_cost * perturbation),
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY


@pytest.mark.parametrize("base_capacity", [100])
def test_nominal_generation_and_pumping_capacity(
    base_capacity: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    load_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"
    np.random.seed(42)
    maxpower = pd.DataFrame(np.zeros((365,4)))
    base_ts = base_capacity * (0.5 + 0.5 * np.random.random((365)))
    maxpower.loc[:,0] = base_ts
    maxpower.loc[:,1] = 24
    maxpower.loc[:,2] = base_ts
    maxpower.loc[:,3] = 24
    

    study_name_base = f"hydro_nomgen_base_{str(int(100*time()))}"
    createHydroTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        load_time_serie_file,
        maxpower=maxpower,
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["hydro"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"hydro_nomgen_{str(int(100*time()))}"
        maxpower.loc[:,0] = base_ts * perturbation
        maxpower.loc[:,2] = base_ts * perturbation
        createHydroTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            maxpower=maxpower,
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY


@pytest.mark.parametrize("base_mingen", [50])
def test_minimum_generation(
    base_mingen: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    load_time_serie_file = LOAD_FILES_DIR / "constant_load.txt" # to force the result of hydro heuristic
    np.random.seed(42)
    inflow_ts = pd.DataFrame(1500 * np.ones((365,1)))
    inflow_ts.loc[364,0] = 0 # to force the result of hydro heuristic
    daily_profile = 0.5 + 0.5*np.random.random((24,1))
    mingen_ts = pd.DataFrame(np.tile(daily_profile,(365,1))/sum(daily_profile) * base_mingen * 24)
    mingen_ts.loc[8736:8760,0] = 0 # to force the result of hydro heuristic

    study_name_base = f"hydro_mingen_base_{str(int(100*time()))}"
    createHydroTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        load_time_serie_file,
        minimum_generation=mingen_ts,
        inflows=inflow_ts
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["hydro"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"hydro_mingen_{str(int(100*time()))}"
        createHydroTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            minimum_generation=mingen_ts * perturbation,
            inflows=inflow_ts
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY

@pytest.mark.parametrize("base_inflow_level", [200])
def test_inflows(
    base_inflow_level: float,
    auto_generated_studies_path: Path,
    antares_exec_folder: Path,
) -> None:
    load_time_serie_file = LOAD_FILES_DIR / "constant_load.txt" # to force the result of hydro heuristic
    base_ts = pd.DataFrame(base_inflow_level * np.ones((365, 1)))
    base_ts.loc[364,0] = 0 # to force the result of hydro heuristic

    study_name_base = f"hydro_inflows_base_{str(int(100*time()))}"
    createHydroTestAntaresStudy(
        study_name_base,
        auto_generated_studies_path,
        load_time_serie_file,
        inflows=base_ts,
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name_base, ["hydro"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for perturbation in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"hydro_inflows_{str(int(100*time()))}"
        createHydroTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            inflows=base_ts * perturbation,
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY