from pathlib import Path
import sys

sys.path.insert(0, "src")
from gems.input_converter.src.logger import Logger
from gems.input_converter.src.data_preprocessing.data_classes import ConversionMode
from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter

study_path = Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25")
logger = Logger(__name__, "")
converter = AntaresStudyConverter(
    study_input=study_path,
    logger=logger,
    mode=ConversionMode.HYBRID.value,
    output_folder=Path(r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted"),
    lib_paths=[
        Path("src/antares_gems_converter/libs/antares_historic/antares_historic.yml"),
        Path("src/antares_gems_converter/libs/reference_models/andromede_v1_models.yml"),
    ],
    models_to_convert=[
        "battery",
        "dsr_industrie",
        "short-term-storage",
        "effacement_residentiel_report",
        "p2g_asservi",
        "psp_closed_daily",
        "p2g_base",
        "electric_vehicle_fr",
        "electric_vehicle_eu_charging_constraint",
        "electric_vehicle_eu",
    ],
)
model_conversion_templates = converter._build_model_conversion_templates()
tmpl = model_conversion_templates["p2g_base"]
vo = tmpl.get_excluded_objects_ids()
converter._convert_single_model(tmpl, vo, [], [], [])
p2g_obj = [o for o in converter.legacy_objects if o.binding_constraint_id and o.binding_constraint_id.startswith("p2g_fatalband")]
print("p2g legacy objs", len(p2g_obj))
print(p2g_obj[:10])
