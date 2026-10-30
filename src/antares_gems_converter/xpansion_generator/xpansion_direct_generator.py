# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
"""Génère directement une étude GemsPy (hybride) à partir des données Xpansion.

Utilisation :

    python -m antares_gems_converter.xpansion_generator.xpansion_direct_generator \\
        --input-dir <dossier Excel Xpansion> \\
        --study <étude Antares de base> \\
        --output-dir <dossier de sortie>

La lecture/écriture des fichiers Excel nécessite openpyxl (pip install openpyxl).

Entrées :
- les fichiers Excel Xpansion (zones, paramètres candidats, correspondance
  type candidat -> cluster Antares) ;
- l'étude Antares de base, lue avec AntaresCraft ;
- la librairie de modèles antares_legacy_investment_candidates.yml
  (libs/antares_legacy_investment_candidates), vérifiée avec GemsPy.

Sorties, dans le dossier de sortie (--output-dir) :
- CandidatsX_generated_<année>.xlsx : tableau maître des candidats ;
- model-libraries/antares_legacy_investment_candidates.yml ;
- data-series/availability_rate_<composant>.tsv ;
- data-series/modeler-scenariobuilder.dat : un scenario group par composant ;
- system.yml ;
- optim-config.yml : résolution par décomposition de Benders ;
- log_composants_system_<année>.txt : composants créés / non créés.

system.yml, optim-config.yml, model-libraries et data-series vont dans le
dossier input/ de l'étude hybride, résolue avec Antares Xpansion
(antares-xpansion-launcher --step gems ; user/expansion/settings.ini requis).

Deux familles de candidats, traitées séparément :
- thermiques (TAC/CCG) : modèle discrete_investment_candidate (nombre
  d'unités entier) si nbr_unit_initial est renseigné, continuous_investment_candidate
  (gisement_ini en MW) sinon ; série de disponibilité et capacité installée
  (existing_capacity) du cluster Antares, thermal-capacity-connections ;
- batteries : journalisées seulement, Antares ne sachant pas encore connecter
  un port de capacité à un short-term storage.

Seul le cas YEAR == ANNEE_REF_EQUILIBRAGE est géré pour le moment.
Le détail des traitements est dans xpansion_direct_generator_lib.py.
"""

import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from gems_craft_hybrid.study.parsing import HybridSystemSchema

from antares_gems_converter.xpansion_generator import (
    xpansion_direct_generator_lib as lib,
)

# =============================================================================
# PARAMETRES DE CONFIGURATION (À adapter selon votre environnement)
# =============================================================================

# Les chemins (dossier des fichiers Excel d'entrée Xpansion, étude Antares de
# base, dossier de sortie) sont passés en ligne de commande : voir parse_args.
FICHIER_CARACT_XP = "X_caract_generateur.xlsx"
FICHIER_LISTE_PAYS_EU = "List_countries.xlsx"
FICHIER_LISTE_PAYS_FR = "List_countries_fr.xlsx"
FICHIER_PARAMETRES_CANDIDATS = "X_parameters_candidates.xlsx"
# Correspondance type candidat (tac_ch4, ccg_ch4...) -> suffixe du cluster Antares
FICHIER_CORRESPONDANCE_CLUSTERS = "correspondance_cluster_antares.xlsx"

# Suffixe du short-term storage Antares auquel sont connectés les candidats
# batteries : <zone><suffixe>, ex. fr_battery_pcomp
SUFFIXE_STORAGE_BATTERIE = "_battery_pcomp"

# Librairie de modèles copiée dans model-libraries par défaut : celle livrée
# avec le convertisseur (une autre peut être passée avec --model-library)
CHEMIN_MODELE_SOURCE = (
    Path(__file__).parents[1]
    / "libs"
    / "antares_legacy_investment_candidates"
    / "antares_legacy_investment_candidates.yml"
)

# Noms des fichiers produits
FICHIER_CANDIDATS_SORTIE = "CandidatsX_generated_{year}.xlsx"
FICHIER_LOG_SORTIE = "log_composants_system_{year}.txt"
PREFIXE_DISPONIBILITE = "availability_rate_"

# Paramètres de génération
YEAR = 2031
ANNEE_REF_EQUILIBRAGE = 2031
CALAGE_EU = True  # True = candidats EU, False = candidats FR uniquement

# Cas 2 (année différente de l'année de référence) désactivé pour le moment.
# Il nécessitera les fichiers suivants et les calculs de gisement associés :
# FICHIER_CANDIDATS_ENTREE = "CandidatsX_generated_{annee_ref_equilibrage}.xlsx"
# FICHIER_SOLUTIONS_CAPA_X = "Solutions_capa_X_<scenario>.xlsx"

# Configuration du system.yml
ID_SYSTEME = "generation_etude_xpansion"
LIBRAIRIE = "antares_legacy_investment_candidates"
NOM_MODELE_CONTINU = f"{LIBRAIRIE}.continuous_investment_candidate"
NOM_MODELE_DISCRET = f"{LIBRAIRIE}.discrete_investment_candidate"
PREFIXE_COMPOSANT = "candidat_invest_"
PORT_CAPACITE = "capacity_port"


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def generate(
    input_dir: Path,
    study_path: Path,
    output_dir: Path,
    model_source: Path = CHEMIN_MODELE_SOURCE,
) -> None:
    """Enchaîne les étapes de génération."""
    print("=" * 80)
    print("XPANSION DIRECT GENERATOR")
    print("=" * 80)

    if YEAR != ANNEE_REF_EQUILIBRAGE:
        raise ValueError("Cette version ne gère que YEAR == ANNEE_REF_EQUILIBRAGE.")

    inputs = lib.XpansionInputFiles(
        caract_xp=input_dir / FICHIER_CARACT_XP,
        countries=input_dir
        / (FICHIER_LISTE_PAYS_EU if CALAGE_EU else FICHIER_LISTE_PAYS_FR),
        candidate_parameters=input_dir / FICHIER_PARAMETRES_CANDIDATS,
        cluster_correspondence=input_dir / FICHIER_CORRESPONDANCE_CLUSTERS,
    )
    inputs.check_exist()

    print("\n[1/6] Librairie de modèles (GemsPy)...")
    library_file = lib.copy_model_library(model_source, output_dir / "model-libraries")
    lib.check_models_in_library(library_file, [NOM_MODELE_CONTINU, NOM_MODELE_DISCRET])
    print(f"  ✓ Modèles vérifiés et copiés vers: {library_file}")

    print("\n[2/6] Lecture des fichiers Excel Xpansion...")
    lib.check_caract_xp(inputs.caract_xp)
    areas = lib.read_areas(inputs.countries)
    mapping = lib.read_cluster_correspondence(inputs.cluster_correspondence)
    print(f"  ✓ {len(areas)} zone(s), {len(mapping)} type(s) de candidat")

    print("\n[3/6] Tableau maître des candidats...")
    battery_candidates = lib.build_battery_candidates(
        inputs.candidate_parameters, areas, mapping, ANNEE_REF_EQUILIBRAGE, first_id=1
    )
    thermal_candidates = lib.build_thermal_candidates(
        inputs.candidate_parameters,
        areas,
        mapping,
        ANNEE_REF_EQUILIBRAGE,
        first_id=1 + 2 * len(battery_candidates),
    )
    candidates_file = output_dir / FICHIER_CANDIDATS_SORTIE.format(year=YEAR)
    lib.write_candidates_excel(
        pd.concat([battery_candidates, thermal_candidates], ignore_index=True),
        candidates_file,
    )
    print(f"  ✓ Candidats écrits: {candidates_file}")

    print("\n[4/6] Lecture de l'étude Antares (AntaresCraft)...")
    study = lib.read_antares_study(study_path)
    print(f"  ✓ Étude lue: {study_path}")

    print("\n[5/6] Construction des composants...")
    print("  Candidats thermiques :")
    settings = lib.ComponentSettings(
        NOM_MODELE_CONTINU, NOM_MODELE_DISCRET, PREFIXE_COMPOSANT, PORT_CAPACITE
    )
    thermal = lib.build_thermal_components(
        thermal_candidates,
        study,
        mapping,
        series_dir=output_dir / "data-series",
        series_prefix=PREFIXE_DISPONIBILITE,
        settings=settings,
    )
    lib.print_family_summary("thermique", thermal.log)
    print("  Candidats batteries :")
    battery = lib.build_battery_components(
        battery_candidates, study, SUFFIXE_STORAGE_BATTERIE, PREFIXE_COMPOSANT
    )
    lib.print_family_summary("batterie", battery.log)

    print("\n[6/6] Écriture du system.yml, de l'optim-config.yml et du journal...")
    system = HybridSystemSchema(
        id=ID_SYSTEME,
        components=thermal.components,
        thermal_capacity_connections=thermal.connections,
    )
    system_file = output_dir / "system.yml"
    lib.write_system_yml(system, system_file)
    print(f"  ✓ Fichier écrit: {system_file}")
    optim_config_file = output_dir / "optim-config.yml"
    used_models = {component.model for component in thermal.components}
    lib.write_optim_config(
        {
            model: variable
            for model, variable in settings.investment_variables().items()
            if model in used_models
        },
        optim_config_file,
    )
    print(f"  ✓ Fichier écrit: {optim_config_file}")
    lib.write_modeler_scenario_builder(
        thermal.scenario_groups, output_dir / "data-series"
    )
    print("  ✓ Scenario builder écrit: data-series/modeler-scenariobuilder.dat")

    log_file = output_dir / FICHIER_LOG_SORTIE.format(year=YEAR)
    lib.write_generation_log(
        log_file, YEAR, study_path, system_file, thermal.log + battery.log
    )
    print(f"  ✓ Journal des composants écrit: {log_file}")

    print("\n" + "=" * 80)
    print(f"✓ Génération terminée. Fichiers créés dans: {output_dir}")
    print("=" * 80)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Générateur direct Xpansion -> GemsPy")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Dossier contenant les fichiers Excel d'entrée Xpansion",
    )
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="Étude Antares de base (clusters, séries, short-term storages)",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Dossier de sortie"
    )
    parser.add_argument(
        "--model-library",
        type=Path,
        default=CHEMIN_MODELE_SOURCE,
        help="Librairie de modèles (par défaut : antares_legacy_investment_candidates.yml)",
    )
    return parser.parse_args()


def main() -> int:
    """Point d'entrée du script."""
    args = parse_args()
    try:
        generate(args.input_dir, args.study, args.output_dir, args.model_library)
    except Exception as e:
        print(f"\n✗ ERREUR: {e}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
