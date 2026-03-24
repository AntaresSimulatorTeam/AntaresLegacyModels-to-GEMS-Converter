from pathlib import Path
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)


from tests.antares_historic.utils import (
    convert_study,
    first_optim_relgap,
    addHybridBehavior
)
from convert_bc import convert_bc_nuc

THERMAL_TEST_SOLVER = "xpress"
THERMAL_TEST_REL_ACCURACY = 5 * 1e-5

antares_exec_folder = "D:/AppliRTE/antares-9.3.5-win64/bin"
# antares_exec_folder = "D:/AppliRTE/rte-antares-9.3.6-installer-64bits/bin"

study_path = Path("C:/Users/gerbauxjul/Documents/6-Etudes_Antares/4-hyb_nuc")
# study_name = "OneNodeBase_93"
study_name = "BP25_Aref_Liv3_EUeq_FReq31_2031"
# addHybridBehavior(study_path / study_name)
original_study_path, converted_study_path = convert_study(
    study_path, study_name, ["thermal"]
)
print(original_study_path)
print(converted_study_path)

convert_bc_nuc(study_path, study_name+"_converted")

rel_gap = first_optim_relgap(
    antares_exec_folder, original_study_path, converted_study_path, THERMAL_TEST_SOLVER
)
print(rel_gap)
assert rel_gap < THERMAL_TEST_REL_ACCURACY