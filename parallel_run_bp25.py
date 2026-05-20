import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tests.antares_historic.utils import convert_study_adapted
study_dir = Path("C:/Users/jeannecor/Documents/1-PROJECTS/OPEN SOURCE")
study_name = "BP25"
# model_list = ["dsr_industrie"]
# model_list = ["battery"]
# model_list = ["short-term-storage"]
# model_list = ["p2g_asservi"]
# model_list = ["effacement_residentiel_report"]
# model_list = ["electric_vehicle_eu_charging_constraint"]
# model_list = ["electric_vehicle_eu"]
# model_list = ["electric_vehicle_fr"]
# model_list = ["electric_vehicle_eu_charging_constraint","electric_vehicle_eu"]
# model_list = ["psp_closed_daily"]
model_list = ["p2g_base"]


study_path, out_folder = convert_study_adapted(study_dir, study_name, model_list)
print("Study:", study_path)
print("Converted output folder:", out_folder)

