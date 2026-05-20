"""
Script de copie et renommage des fichiers binding constraints.
Basé sur le tableau de correspondance du fichier tableau_correspondance.docx.

Dossier source : C:\\Users\\jeannecor\\Documents\\1-PROJECTS\\OPEN SOURCE\\BP25\\input\\bindingconstraints
Dossier cible  : C:\\Users\\jeannecor\\Documents\\1-PROJECTS\\OPEN SOURCE\\antares-studies-converted\\BP25\\input\\data-series
"""

import os
import re
import glob
import shutil

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SOURCE_DIR = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25\input\bindingconstraints"
TARGET_DIR = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\input\data-series"


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 1 : Détection automatique des zones (valeurs de X)
# ─────────────────────────────────────────────────────────────────────────────

def detect_zones(source_dir):
    """
    Détecte toutes les zones existantes (valeurs de X) en scannant
    les fichiers ve_stock_X_eq.txt dans le dossier source.
    Ce pattern est utilisé comme référence pour identifier les zones disponibles.
    """
    zones = []
    pattern = os.path.join(source_dir, "ve_stock_*_eq.txt")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        # Extrait X depuis "ve_stock_X_eq.txt"
        match = re.match(r"ve_stock_(.+)_eq\.txt", filename)
        if match:
            zone = match.group(1)
            zones.append(zone)
    zones.sort()
    return zones


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 2 : Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def duplicate_lines_24x(content):
    lines = content.splitlines()
    result_lines = []
    for line in lines:
        try:
            value = float(line.strip())
            new_value = value / 24
            # Conserve le format entier si la valeur originale était entière
            if value == int(value):
                formatted = str(int(new_value)) if new_value == int(new_value) else f"{new_value}"
            else:
                formatted = str(new_value)
            for _ in range(24):
                result_lines.append(formatted)
        except ValueError:
            # Ligne non numérique (entête, ligne vide, etc.) : copie brute sans division
            for _ in range(24):
                result_lines.append(line)
    return "\n".join(result_lines)

def copy_file(src_path, dst_path, transform=None):
    """
    Copie un fichier source vers la destination.
    Si transform est fourni, applique la transformation au contenu avant écriture.
    """
    if not os.path.exists(src_path):
        print(f"  [ABSENT]  Fichier source introuvable : {src_path}")
        return False

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if transform:
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        content = transform(content)
        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        shutil.copy2(src_path, dst_path)

    print(f"  [OK]      {os.path.basename(src_path)}  -->  {os.path.basename(dst_path)}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# ETAPE 3 : Traitement des règles du tableau
# ─────────────────────────────────────────────────────────────────────────────

def process_all(source_dir, target_dir, zones):
    """
    Applique toutes les règles de copie/renommage issues du tableau Word.
    """
    os.makedirs(target_dir, exist_ok=True)
    errors = 0
    success = 0

    # Zones hors "fr" (utilisées pour ve_eu_load_min_gt.txt)
    zones_sans_fr = [z for z in zones if z != "fr"]

    print(f"\nZones détectées : {zones}")
    print(f"Zones hors 'fr' : {zones_sans_fr}")
    print(f"\n{'─'*70}")

    # ── Règle 1 : ve_stock_X_eq.txt → stock_mobile_X_electric_vehicle_Y.tsv ──
    # Si X = fr → Y = fr ; sinon Y = eu
    print("\n[Règle 1] ve_stock_X_eq.txt → stock_mobile_X_electric_vehicle_Y.tsv")
    for zone in zones:
        src = os.path.join(source_dir, f"ve_stock_{zone}_eq.txt")
        y = "fr" if zone == "fr" else "eu"
        dst = os.path.join(target_dir, f"stock_mobile_{zone}_electric_vehicle_{y}.tsv")
        ok = copy_file(src, dst)
        success += ok
        errors += not ok

    # ── Règle 2 : ve_level_min_X_gt.txt → stock_minimal_X_electric_vehicle_Y.tsv ──
    # Si X = fr → Y = fr ; sinon Y = eu
    print("\n[Règle 2] ve_level_min_X_gt.txt → stock_minimal_X_electric_vehicle_Y.tsv")
    for zone in zones:
        src = os.path.join(source_dir, f"ve_level_min_{zone}_gt.txt")
        y = "fr" if zone == "fr" else "eu"
        dst = os.path.join(target_dir, f"stock_minimal_{zone}_electric_vehicle_{y}.tsv")
        ok = copy_file(src, dst)
        success += ok
        errors += not ok

    # ── Règle 3 : ve_eu_load_min_gt.txt → ve_eu_agregee_min_24h_charging_at_day_at_electric_vehicle_eu_charging_constraint.tsv ──
    # Pour cette règle, on ne génère qu’un seul fichier pour la zone "at"
    # Chaque ligne dupliquée 24 fois
    print("\n[Règle 3] ve_eu_load_min_gt.txt → ve_eu_agregee_min_24h_charging_at_day_at_electric_vehicle_eu_charging_constraint.tsv")
    print("          (chaque ligne dupliquée 24 fois, zone 'at' uniquement)")
    src = os.path.join(source_dir, "ve_eu_load_min_gt.txt")
    zone = "at"
    dst = os.path.join(target_dir, f"ve_eu_agregee_min_24h_charging_at_day_{zone}_electric_vehicle_eu_charging_constraint.tsv")
    ok = copy_file(src, dst, transform=duplicate_lines_24x)
    success += ok
    errors += not ok

    # ── Règle 4 : ve_v2g_limit_lt.txt → ve_fr_max_v2g_24h_fr_electric_vehicle_fr.tsv ──
    # Chaque ligne dupliquée 24 fois
    print("\n[Règle 4] ve_v2g_limit_lt.txt → ve_fr_max_v2g_24h_fr_electric_vehicle_fr.tsv")
    print("          (chaque ligne dupliquée 24 fois)")
    src = os.path.join(source_dir, "ve_v2g_limit_lt.txt")
    dst = os.path.join(target_dir, "ve_fr_max_v2g_24h_fr_electric_vehicle_fr.tsv")
    ok = copy_file(src, dst, transform=duplicate_lines_24x)
    success += ok
    errors += not ok

    # ── Règle 5 : ve_fr_load_min_gt.txt → ve_fr_min_24h_charging_fr_electric_vehicle_fr.tsv ──
    # Chaque ligne dupliquée 24 fois
    print("\n[Règle 5] ve_fr_load_min_gt.txt → ve_fr_min_24h_charging_fr_electric_vehicle_fr.tsv")
    print("          (chaque ligne dupliquée 24 fois)")
    src = os.path.join(source_dir, "ve_fr_load_min_gt.txt")
    dst = os.path.join(target_dir, "ve_fr_min_24h_charging_fr_electric_vehicle_fr.tsv")
    ok = copy_file(src, dst, transform=duplicate_lines_24x)
    success += ok
    errors += not ok

    # ── Règle 6 : ve_fr_night_load_min_gt.txt → ve_fr_min_24h_charging_at_night_fr_electric_vehicle_fr.tsv ──
    # Chaque ligne dupliquée 24 fois
    print("\n[Règle 6] ve_fr_night_load_min_gt.txt → ve_fr_min_24h_charging_at_night_fr_electric_vehicle_fr.tsv")
    print("          (chaque ligne dupliquée 24 fois)")
    src = os.path.join(source_dir, "ve_fr_night_load_min_gt.txt")
    dst = os.path.join(target_dir, "ve_fr_min_24h_charging_at_night_fr_electric_vehicle_fr.tsv")
    ok = copy_file(src, dst, transform=duplicate_lines_24x)
    success += ok
    errors += not ok

    # ── Règle 7 : ve_fr_mobilite_lourde_eq.txt → ve_fr_max_mobilite_lourde_24h_fr_electric_vehicle_fr.tsv ──
    # Chaque ligne dupliquée 24 fois
    print("\n[Règle 7] ve_fr_mobilite_lourde_eq.txt → ve_fr_max_mobilite_lourde_24h_fr_electric_vehicle_fr.tsv")
    print("          (chaque ligne dupliquée 24 fois)")
    src = os.path.join(source_dir, "ve_fr_mobilite_lourde_eq.txt")
    dst = os.path.join(target_dir, "ve_fr_max_mobilite_lourde_24h_fr_electric_vehicle_fr.tsv")
    ok = copy_file(src, dst, transform=duplicate_lines_24x)
    success += ok
    errors += not ok

    return success, errors


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  COPIE ET RENOMMAGE - BINDING CONSTRAINTS")
    print("=" * 70)
    print(f"\nSource : {SOURCE_DIR}")
    print(f"Cible  : {TARGET_DIR}")

    # Vérification du dossier source
    if not os.path.isdir(SOURCE_DIR):
        print(f"\n[ERREUR] Dossier source introuvable : {SOURCE_DIR}")
        exit(1)

    # Détection des zones
    zones = detect_zones(SOURCE_DIR)
    if not zones:
        print("\n[ERREUR] Aucune zone détectée. Vérifier que le dossier source contient des fichiers ve_stock_X_eq.txt")
        exit(1)

    # Traitement
    success, errors = process_all(SOURCE_DIR, TARGET_DIR, zones)

    # Bilan
    print(f"\n{'─'*70}")
    print(f"BILAN : {success} fichier(s) copié(s) avec succès, {errors} erreur(s)")
    print("=" * 70)
