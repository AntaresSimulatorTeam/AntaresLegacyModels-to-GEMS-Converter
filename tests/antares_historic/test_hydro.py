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

## TESTING PROCEDURE FOR GEMS MODEL REPRESENTING ANTARES v9.3 "ROR" : CONSTANT DATA  ##

LOAD_FILES_DIR = Path("tests/antares_historic/data")
TEST_REL_ACCURACY = 1e-6
TEST_SOLVER = "highs"
MODIFICATION_RATIO = 1.2

## TESTING PROCEDURE FOR GEMS MODEL REPRESENTING ANTARES v9.3 "ROR" : TESTS  ##

# General tests [OK : test_ror]

# Testing Timeseries parameters
## generation [OK : test_ror]


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
