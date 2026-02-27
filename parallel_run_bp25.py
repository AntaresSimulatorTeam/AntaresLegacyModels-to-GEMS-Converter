import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tests.antares_historic.utils import convert_study_adapted
study_dir = Path("C:/Users/jeannecor/Documents/1-PROJECTS/OPEN SOURCE")
study_name = "BP25"
model_list = ["short-term-storage"]
#model_list = ["p2g_asservi", "p2g_base"]

study_path, out_folder = convert_study_adapted(study_dir, study_name, model_list)
print("Study:", study_path)
print("Converted output folder:", out_folder)

