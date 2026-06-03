from pathlib import Path
from time import time

import numpy as np
import pytest
import pandas as pd

from tests.antares_historic.utils import (
    convert_study,
    createMiscGenTestAntaresStudy,
    first_optim_relgap,
)

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "MISC GEN"  : CONSTANT DATA  ##

LOAD_FILES_DIR = Path("tests/antares_historic/data")
TEST_REL_ACCURACY = 1e-6
TEST_SOLVER = "highs"
MODIFICATION_RATIO = 1.2

## TESTING PROCEDURE FOR GEMS MODEL REPRENSENTING ANTARES v9.3 "MISC GEN"  : TESTS  ##

# General tests [OK : test_misc_gen]

# Testing Timeseries parameters
## generation [OK : test_misc_gen]


@pytest.mark.parametrize("misc_gen", [100])
def test_misc_gen(
    misc_gen: float, auto_generated_studies_path: Path, antares_exec_folder: Path
) -> None:
    study_name = f"misc_gen_test_study_{str(int(100*time()))}"
    load_time_serie_file = LOAD_FILES_DIR / "load_matrix_1.txt"
    np.random.seed(1)  # for reproducibility
    misc_gen_timeseries = pd.DataFrame(misc_gen * np.random.random((8760, 8)))

    createMiscGenTestAntaresStudy(
        study_name,
        auto_generated_studies_path,
        load_time_serie_file,
        misc_gen_timeseries,
    )
    original_study_path, converted_study_path = convert_study(
        auto_generated_studies_path, study_name, ["misc_gen"]
    )
    rel_gap = first_optim_relgap(
        antares_exec_folder, original_study_path, converted_study_path, TEST_SOLVER
    )
    assert rel_gap < TEST_REL_ACCURACY

    for modification in [MODIFICATION_RATIO, 1 / MODIFICATION_RATIO]:
        perturbed_study_name = f"misc_gen_test_study_{str(int(100*time()))}"
        misc_gen_perturbated = misc_gen_timeseries * modification
        createMiscGenTestAntaresStudy(
            perturbed_study_name,
            auto_generated_studies_path,
            load_time_serie_file,
            misc_gen_perturbated,
        )
        perturbed_study_path = auto_generated_studies_path / perturbed_study_name
        rel_gap = first_optim_relgap(
            antares_exec_folder, perturbed_study_path, converted_study_path, TEST_SOLVER
        )
        assert rel_gap > 10 * TEST_REL_ACCURACY
