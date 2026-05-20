import pandas as pd
import os

# ============================================================
# PARAMETRES
# ============================================================

BASE_PATH = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\output"
SIMULATION_ID = "20260519-1816eco"  # <-- paramètre modifiable
FILE_NAME = "simulation_table--optim-nb-1.csv"

# Composants et outputs à extraire
FILTERS = [
    # {"component": "electric_vehicle_fr", "output": "charging"},
    # {"component": "electric_vehicle_fr", "output": "charging_mobilite_lourde"},
    # {"component": "electric_vehicle_fr", "output": "v2g"},
    # {"component": "electric_vehicle_fr", "output": "suivi_stock_energie_ve"},
    # {"component": "electric_vehicle_eu_be", "output": "charging"},
    # {"component": "electric_vehicle_eu_be", "output": "v2g"},
    # {"component": "electric_vehicle_eu_be", "output": "suivi_stock_energie_ve"},
    # {"component": "electric_vehicle_eu_de", "output": "charging"},
    # {"component": "electric_vehicle_eu_de", "output": "v2g"},
    # {"component": "electric_vehicle_eu_de", "output": "suivi_stock_energie_ve"},
    # {"component": "battery_de", "output": "p_withdrawal"},
    # {"component": "battery_de", "output": "p_injection"},
    # {"component": "battery_de", "output": "level"},
    # {"component": "battery_be", "output": "p_withdrawal"},
    # {"component": "battery_be", "output": "p_injection"},
    # {"component": "battery_be", "output": "level"},
    # {"component": "psp_closed_ch", "output": "p_withdrawal"},
    # {"component": "psp_closed_ch", "output": "p_injection"},
    # {"component": "psp_closed_es", "output": "p_withdrawal"},
    # {"component": "psp_closed_es", "output": "p_injection"},
    # {"component": "psp_closed_at", "output": "p_withdrawal"},
    # {"component": "psp_closed_at", "output": "p_injection"},
    # {"component": "psp_closed_itn", "output": "p_withdrawal"},
    # {"component": "psp_closed_itn", "output": "p_injection"},
    # {"component": "dsr_industrie_fr", "output": "curtailment"},
    # {"component": "fr_fr_psp_open", "output": "p_injection"},
    # {"component": "fr_fr_psp_open", "output": "p_withdrawal"},
    # {"component": "fr_fr_psp_open", "output": "level"},
    # {"component": "fr_fr_psp_closed", "output": "p_injection"},
    # {"component": "fr_fr_psp_closed", "output": "p_withdrawal"},
    # {"component": "fr_fr_psp_closed", "output": "level"},
    # {"component": "fr_fr_battery", "output": "p_injection"},
    # {"component": "fr_fr_battery", "output": "p_withdrawal"},
    # {"component": "fr_fr_battery", "output": "level"},
    {"component": "p2g_asservi_se1", "output": "load"},

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
