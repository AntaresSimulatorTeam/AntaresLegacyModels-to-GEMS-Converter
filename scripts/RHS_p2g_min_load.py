r"""
Script de copie des fichiers p2g_fatalband_X_gt.txt vers des fichiers
min_load_X_z_p2g_base.tsv et mise à jour de system.yml.

Dossier source : C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25\input\bindingconstraints
Dossier cible  : C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\input\data-series
"""

import glob
import os
import re
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

SOURCE_DIR = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\BP25\input\bindingconstraints"
TARGET_DIR = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\input\data-series"
SYSTEM_YML_PATH = r"C:\Users\jeannecor\Documents\1-PROJECTS\OPEN SOURCE\antares-studies-converted\BP25\input\system.yml"


def detect_areas(source_dir):
    """Détecte les areas X dans les fichiers p2g_fatalband_X_gt.txt."""
    areas = []
    pattern = os.path.join(source_dir, "p2g_fatalband_*_gt.txt")
    for filepath in glob.glob(pattern):
        filename = os.path.basename(filepath)
        match = re.match(r"p2g_fatalband_(.+)_gt\.txt", filename)
        if match:
            areas.append(match.group(1))
    areas.sort()
    return areas


def copy_file(src_path, dst_path):
    if not os.path.exists(src_path):
        print(f"  [ABSENT] Fichier source introuvable : {src_path}")
        return False

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(src_path, "r", encoding="utf-8", errors="replace") as src_file:
        content = src_file.read()
    with open(dst_path, "w", encoding="utf-8") as dst_file:
        dst_file.write(content)

    print(f"  [OK]      {os.path.basename(src_path)}  -->  {os.path.basename(dst_path)}")
    return True


def load_system_yml(path):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f)
    return yaml, data


def write_system_yml(yaml, data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)


def build_area_component_map(system_data):
    area_to_component = {}
    for entry in system_data.get("system", {}).get("area-connections", []):
        area = entry.get("area")
        component = entry.get("component")
        if area and component:
            area_to_component[area] = component
    return area_to_component


def add_min_load_parameters(system_data, area_to_component, areas):
    added = []
    components = system_data.get("system", {}).get("components", [])
    for area in areas:
        component_id = area_to_component.get(area)
        if component_id is None:
            print(f"  [WARN] Aucun component trouvé pour l'area '{area}'")
            continue

        component = next((comp for comp in components if comp.get("id") == component_id), None)
        if component is None:
            print(f"  [WARN] Component '{component_id}' introuvable dans system.yml")
            continue

        parameters = component.get("parameters")
        if parameters is None:
            parameters = []
            component["parameters"] = parameters

        if any(p.get("id") == "min_load" for p in parameters):
            print(f"  [SKIP] min_load déjà présent dans component {component_id}")
            continue

        value = f"min_load_{area}_z_p2g_base"
        new_param = CommentedMap()
        new_param["id"] = "min_load"
        new_param["scenario-dependent"] = False
        new_param["time-dependent"] = True
        new_param["value"] = value
        new_param.yaml_add_eol_comment("Added by p2g min_load script", key="id")
        parameters.append(new_param)
        added.append((component_id, value))
        print(f"  [ADD]     min_load ajouté pour component {component_id} -> {value}")

    return added


def main():
    print("=" * 70)
    print("  COPIE P2G FATALBAND ET MISE A JOUR DE system.yml")
    print("=" * 70)
    print(f"Source : {SOURCE_DIR}")
    print(f"Cible  : {TARGET_DIR}")
    print(f"system.yml : {SYSTEM_YML_PATH}\n")

    if not os.path.isdir(SOURCE_DIR):
        print(f"[ERREUR] Dossier source introuvable : {SOURCE_DIR}")
        return 1

    if not os.path.exists(SYSTEM_YML_PATH):
        print(f"[ERREUR] system.yml introuvable : {SYSTEM_YML_PATH}")
        return 1

    areas = detect_areas(SOURCE_DIR)
    if not areas:
        print("[ERREUR] Aucune area détectée. Vérifiez la présence de fichiers p2g_fatalband_X_gt.txt")
        return 1

    print(f"Areas détectées : {areas}\n")

    success = 0
    errors = 0
    for area in areas:
        src = os.path.join(SOURCE_DIR, f"p2g_fatalband_{area}_gt.txt")
        dst = os.path.join(TARGET_DIR, f"min_load_{area}_z_p2g_base.tsv")
        if copy_file(src, dst):
            success += 1
        else:
            errors += 1

    print(f"\nFichiers copiés : {success}, erreurs : {errors}\n")

    yaml, system_data = load_system_yml(SYSTEM_YML_PATH)
    area_to_component = build_area_component_map(system_data)
    if not area_to_component:
        print("[ERREUR] Aucune area-connections trouvée dans system.yml")
        return 1

    added = add_min_load_parameters(system_data, area_to_component, areas)
    write_system_yml(yaml, system_data, SYSTEM_YML_PATH)

    print("\nMise à jour de system.yml terminée.")
    if added:
        for component_id, value in added:
            print(f"  - composant {component_id} : {value}")
    else:
        print("Aucun nouveau paramètre ajouté dans system.yml.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
