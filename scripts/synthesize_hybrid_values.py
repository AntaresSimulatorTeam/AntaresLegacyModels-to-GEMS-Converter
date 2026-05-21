import pandas as pd
import os

# ============================================================
# PARAMETRES
# ============================================================

BASE_PATH = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\output"
SIMULATION_ID = "20260521-1154eco"  # <-- paramètre modifiable
FILE_NAME = "simulation_table--optim-nb-1.csv"

# Composants et outputs à extraire
FILTERS = [
    {"component": "electric_vehicle_fr", "output": "charging"},
    {"component": "electric_vehicle_fr", "output": "charging_mobilite_lourde"},
    {"component": "electric_vehicle_fr", "output": "v2g"},
    {"component": "electric_vehicle_fr", "output": "suivi_stock_energie_ve"},
    {"component": "electric_vehicle_eu_be", "output": "charging"},
    {"component": "electric_vehicle_eu_be", "output": "v2g"},
    {"component": "electric_vehicle_eu_be", "output": "suivi_stock_energie_ve"},
    {"component": "electric_vehicle_eu_de", "output": "charging"},
    {"component": "electric_vehicle_eu_de", "output": "v2g"},
    {"component": "electric_vehicle_eu_de", "output": "suivi_stock_energie_ve"},
    {"component": "battery_de", "output": "p_withdrawal"},
    {"component": "battery_de", "output": "p_injection"},
    {"component": "battery_de", "output": "level"},
    {"component": "battery_be", "output": "p_withdrawal"},
    {"component": "battery_be", "output": "p_injection"},
    {"component": "battery_be", "output": "level"},
    {"component": "psp_closed_ch", "output": "p_withdrawal"},
    {"component": "psp_closed_ch", "output": "p_injection"},
    {"component": "psp_closed_es", "output": "p_withdrawal"},
    {"component": "psp_closed_es", "output": "p_injection"},
    {"component": "psp_closed_at", "output": "p_withdrawal"},
    {"component": "psp_closed_at", "output": "p_injection"},
    {"component": "psp_closed_itn", "output": "p_withdrawal"},
    {"component": "psp_closed_itn", "output": "p_injection"},
    {"component": "dsr_industrie_fr", "output": "curtailment"},
    {"component": "fr_fr_psp_open", "output": "p_injection"},
    {"component": "fr_fr_psp_open", "output": "p_withdrawal"},
    {"component": "fr_fr_psp_open", "output": "level"},
    {"component": "fr_fr_psp_closed", "output": "p_injection"},
    {"component": "fr_fr_psp_closed", "output": "p_withdrawal"},
    {"component": "fr_fr_psp_closed", "output": "level"},
    {"component": "fr_fr_battery", "output": "p_injection"},
    {"component": "fr_fr_battery", "output": "p_withdrawal"},
    {"component": "fr_fr_battery", "output": "level"},
    {"component": "p2g_asservi_se1", "output": "load"},
    {"component": "p2g_base_se1", "output": "load"},

]

# ============================================================
# CONSTRUCTION DU CHEMIN
# ============================================================

file_path = os.path.join(BASE_PATH, SIMULATION_ID, FILE_NAME)

# ============================================================
# LECTURE DU CSV
# ============================================================

df = pd.read_csv(file_path, sep=",", decimal=".")

# ============================================================
# FILTRAGE
# ============================================================

# Construction d'un masque de filtre combiné
mask = pd.Series(False, index=df.index)
for f in FILTERS:
    mask |= (df["component"] == f["component"]) & (df["output"] == f["output"])

df_filtered = df[mask].copy()

# ============================================================
# AGGREGATION : somme des valeurs par (component, output, block)
# ============================================================

df_agg = (
    df_filtered
    .groupby(["component", "output", "block"], as_index=False)["value"]
    .sum()
)

# ============================================================
# PIVOT : une colonne par semaine
# ============================================================

df_pivot = df_agg.pivot_table(
    index=["component", "output"],
    columns="block",
    values="value",
    aggfunc="sum"
)

# Renommage des colonnes en "value_week1", "value_week2", etc.
df_pivot.columns = [f"value_week{int(col)}" for col in df_pivot.columns]
df_pivot = df_pivot.reset_index()

# ============================================================
# TRI : ordonnancement spécifique des lignes
# ============================================================
expected_order = [
    ("electric_vehicle_fr", "charging"),
    ("electric_vehicle_fr", "charging_mobilite_lourde"),
    ("electric_vehicle_fr", "suivi_stock_energie_ve"),
    ("electric_vehicle_fr", "v2g"),
    ("electric_vehicle_eu_be", "charging"),
    ("electric_vehicle_eu_be", "suivi_stock_energie_ve"),
    ("electric_vehicle_eu_be", "v2g"),
    ("electric_vehicle_eu_de", "charging"),
    ("electric_vehicle_eu_de", "suivi_stock_energie_ve"),
    ("electric_vehicle_eu_de", "v2g"),
    ("battery_de", "p_withdrawal"),
    ("battery_de", "p_injection"),
    ("battery_de", "level"),
    ("battery_be", "p_withdrawal"),
    ("battery_be", "p_injection"),
    ("battery_be", "level"),
    ("psp_closed_ch", "p_withdrawal"),
    ("psp_closed_ch", "p_injection"),
    ("psp_closed_es", "p_withdrawal"),
    ("psp_closed_es", "p_injection"),
    ("psp_closed_at", "p_withdrawal"),
    ("psp_closed_at", "p_injection"),
    ("psp_closed_itn", "p_withdrawal"),
    ("psp_closed_itn", "p_injection"),
    ("effacement_report_chauffage", "effacement (somme hebdo)"),
    ("effacement_report_chauffage", "report (somme hebdo)"),
    ("dsr_industrie_fr", "curtailment"),
    ("fr_fr_battery", "p_withdrawal"),
    ("fr_fr_battery", "p_injection"),
    ("fr_fr_battery", "level"),
    ("fr_fr_psp_closed", "p_withdrawal"),
    ("fr_fr_psp_closed", "p_injection"),
    ("fr_fr_psp_closed", "level"),
    ("fr_fr_psp_open", "p_withdrawal"),
    ("fr_fr_psp_open", "p_injection"),
    ("fr_fr_psp_open", "level"),
    ("p2g_asservi_se1", "load"),
    ("p2g_base_se1", "load"),
]
order_map = {pair: idx for idx, pair in enumerate(expected_order)}

df_pivot["_sort_index"] = df_pivot.apply(
    lambda row: order_map.get((row["component"], row["output"]), len(order_map)),
    axis=1,
)
df_pivot = df_pivot.sort_values(["_sort_index", "component", "output"]).drop(columns=["_sort_index"])

# ============================================================
# EXPORT EXCEL
# ============================================================

output_dir = os.path.dirname(file_path)
output_file = os.path.join(output_dir, f"synthesis_{SIMULATION_ID}.xlsx")

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_pivot.to_excel(writer, index=False, sheet_name="Synthesis")

    # Formatage : séparateur décimal "," et largeur des colonnes auto
    workbook = writer.book
    worksheet = writer.sheets["Synthesis"]

    # Format nombre avec virgule comme séparateur décimal
    from openpyxl.styles import numbers
    decimal_format = '#,##0.00'  # format Excel FR : virgule comme séparateur décimal

    for row in worksheet.iter_rows(min_row=2, min_col=3):  # à partir de la 3e colonne (value_week*)
        for cell in row:
            if cell.value is not None:
                cell.number_format = decimal_format

    # Ajustement automatique de la largeur des colonnes
    for col in worksheet.columns:
        max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
        worksheet.column_dimensions[col[0].column_letter].width = max_length + 4

print(f"Fichier exporté : {output_file}")
print(f"\nAperçu du résultat :")
print(df_pivot.to_string())
