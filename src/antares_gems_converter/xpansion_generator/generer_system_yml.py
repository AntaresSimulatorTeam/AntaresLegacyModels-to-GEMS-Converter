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
"""
Script de generation du fichier system.yml a partir des fichiers list.ini
des clusters candidats a l'investissement (dossiers "00_xp...") d'une etude
Antares.

Utilisation :

    python -m antares_gems_converter.xpansion_generator.generer_system_yml \\
        --study <etude Antares>

Fonctionnement :
1. Cree les dossiers input/model-libraries et input/data-series.
2. Parcourt input/thermal/clusters/ a la recherche des dossiers dont le nom
   commence par le prefixe "00_xp".
3. Pour chaque dossier trouve, lit le fichier list.ini et construit un
   composant "candidat_invest_NOM_CLUSTER" dont les parametres reprennent
   chaque ligne cle = valeur du list.ini.
4. Ecrit le resultat dans input/system.yml.
"""

import configparser
import os
import shutil
from argparse import ArgumentParser

import yaml

# ---------------------------------------------------------------------------
# PARAMETRES A ADAPTER
# ---------------------------------------------------------------------------

# Le chemin racine de l'etude Antares est passe en ligne de commande (--study)

# Prefixe identifiant les dossiers de clusters candidats a l'investissement
PREFIXE_DOSSIER_CANDIDAT = "00_xp"

# Nom du modele Andromede utilise pour les composants d'investissement
NOM_MODELE = "xpansion_models.thermal_candidate_to_invest"

# Nom du groupe de scenarios associe aux composants generes
SCENARIO_GROUP = "scenario_thermal_xpansion"

# Identifiant du systeme (facultatif, a completer/adapter si necessaire)
ID_SYSTEME = "generation_etude_xpansion"

# Liste des area-connections a inserer dans le system.yml
# (vide par defaut, a completer manuellement si besoin)
AREA_CONNECTIONS: list = []

# Chemin du fichier modele source a copier dans input/model-libraries
# (celui livre avec ce script)
CHEMIN_MODELE_SOURCE = os.path.join(os.path.dirname(__file__), "xpansion_models.yml")


# ---------------------------------------------------------------------------
# FONCTIONS
# ---------------------------------------------------------------------------


class IndentDumper(yaml.Dumper):
    """Dumper YAML qui indente les items de liste sous leur cle parente,
    pour respecter le format attendu du system.yml Antares/Andromede."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def convertir_valeur(valeur_brute: str):
    """Convertit une valeur texte issue d'un list.ini vers son type Python
    natif le plus probable (bool, int, float ou str)."""
    valeur = valeur_brute.strip()

    if valeur.lower() in ("true", "false"):
        return valeur.lower() == "true"

    try:
        return int(valeur)
    except ValueError:
        pass

    try:
        return float(valeur)
    except ValueError:
        pass

    return valeur


def extraire_nom_cluster(nom_dossier: str) -> str:
    """Extrait le NOM_CLUSTER a partir du nom du dossier candidat
    (tout ce qui suit le prefixe '00_xp')."""
    suffixe = nom_dossier[len(PREFIXE_DOSSIER_CANDIDAT) :]
    return suffixe.lstrip("_-")


def trouver_list_ini(dossier_candidat: str):
    """Recherche le ou les fichiers list.ini dans le dossier candidat
    (recherche directe a la racine, puis recursive en secours)."""
    chemin_direct = os.path.join(dossier_candidat, "list.ini")
    if os.path.isfile(chemin_direct):
        return [chemin_direct]

    fichiers_trouves = []
    for racine, _, fichiers in os.walk(dossier_candidat):
        for fichier in fichiers:
            if fichier.lower() == "list.ini":
                fichiers_trouves.append(os.path.join(racine, fichier))
    return fichiers_trouves


def construire_composant(nom_cluster: str, section: configparser.SectionProxy) -> dict:
    """Construit le dictionnaire d'un composant a partir d'une section
    du fichier list.ini (une ligne du list.ini = un parametre)."""
    parametres = []
    for cle, valeur_brute in section.items():
        parametres.append(
            {
                "id": cle,
                "scenario-dependent": False,
                "time-dependent": False,
                "value": convertir_valeur(valeur_brute),
            }
        )

    return {
        "id": f"candidat_invest_{nom_cluster}",
        "model": NOM_MODELE,
        "parameters": parametres,
        "scenario-group": SCENARIO_GROUP,
    }


def collecter_composants(chemin_clusters: str) -> list:
    """Parcourt le dossier thermal/clusters et construit la liste des
    composants a partir de chaque list.ini trouve dans les dossiers
    candidats (prefixe 00_xp)."""
    composants = []

    if not os.path.isdir(chemin_clusters):
        raise FileNotFoundError(
            f"Le dossier des clusters thermiques est introuvable : {chemin_clusters}"
        )

    for entree in sorted(os.scandir(chemin_clusters), key=lambda e: e.name):
        if not entree.is_dir():
            continue
        if not entree.name.startswith(PREFIXE_DOSSIER_CANDIDAT):
            continue

        nom_cluster = extraire_nom_cluster(entree.name)
        fichiers_list_ini = trouver_list_ini(entree.path)

        if not fichiers_list_ini:
            print(f"  [Attention] Aucun list.ini trouve dans : {entree.name}")
            continue

        for chemin_list_ini in fichiers_list_ini:
            parser = configparser.ConfigParser(interpolation=None)
            parser.optionxform = str  # type: ignore[assignment, method-assign]  # conserve la casse exacte des cles
            parser.read(chemin_list_ini, encoding="utf-8")

            nb_sections = len(parser.sections())
            for section_nom in parser.sections():
                # Si plusieurs sections/clusters dans un meme list.ini,
                # on suffixe le nom pour eviter les collisions d'id.
                nom_final = (
                    nom_cluster if nb_sections == 1 else f"{nom_cluster}_{section_nom}"
                )
                composant = construire_composant(nom_final, parser[section_nom])
                composants.append(composant)
                print(
                    f"  Composant construit : candidat_invest_{nom_final} "
                    f"({len(composant['parameters'])} parametres)"
                )

    return composants


def creer_arborescence(chemin_input: str):
    """Cree les dossiers model-libraries et data-series s'ils n'existent pas."""
    for nom_dossier in ("model-libraries", "data-series"):
        chemin_dossier = os.path.join(chemin_input, nom_dossier)
        os.makedirs(chemin_dossier, exist_ok=True)
        print(f"  Dossier verifie/cree : {chemin_dossier}")


def copier_modele_libraries(chemin_input: str):
    """Copie le fichier modele source (CHEMIN_MODELE_SOURCE) dans le
    dossier input/model-libraries de l'etude."""
    if not os.path.isfile(CHEMIN_MODELE_SOURCE):
        print(
            f"  [Attention] Fichier modele introuvable, copie ignoree : "
            f"{CHEMIN_MODELE_SOURCE}"
        )
        return

    chemin_destination = os.path.join(
        chemin_input, "model-libraries", os.path.basename(CHEMIN_MODELE_SOURCE)
    )
    shutil.copy2(CHEMIN_MODELE_SOURCE, chemin_destination)
    print(f"  Fichier modele copie vers : {chemin_destination}")


def ecrire_system_yml(chemin_input: str, composants: list):
    """Ecrit le fichier system.yml dans le dossier input de l'etude."""
    contenu = {
        "system": {
            "area-connections": AREA_CONNECTIONS,
            "components": composants,
            "id": ID_SYSTEME,
            "nodes": [],
        }
    }

    chemin_system_yml = os.path.join(chemin_input, "system.yml")
    with open(chemin_system_yml, "w", encoding="utf-8") as f:
        yaml.dump(
            contenu,
            f,
            Dumper=IndentDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    print(f"\nFichier system.yml ecrit ici : {chemin_system_yml}")


def main():
    parser = ArgumentParser(description="Generation du system.yml (dossiers 00_xp)")
    parser.add_argument(
        "--study", required=True, help="Chemin racine de l'etude Antares"
    )
    chemin_etude_antares = parser.parse_args().study

    chemin_input = os.path.join(chemin_etude_antares, "input")
    chemin_clusters = os.path.join(chemin_input, "thermal", "clusters")

    if not os.path.isdir(chemin_input):
        raise FileNotFoundError(
            f"Le dossier input de l'etude est introuvable : {chemin_input}"
        )

    print("Etape 1 - Creation de l'arborescence (model-libraries, data-series)")
    creer_arborescence(chemin_input)
    copier_modele_libraries(chemin_input)

    print("\nEtape 2 - Collecte des composants candidats (dossiers 00_xp...)")
    composants = collecter_composants(chemin_clusters)

    if not composants:
        print(
            "\n[Attention] Aucun composant n'a ete trouve. "
            "Verifie le prefixe des dossiers et la presence de list.ini."
        )

    print("\nEtape 3 - Ecriture du fichier system.yml")
    ecrire_system_yml(chemin_input, composants)

    print(f"\nTermine. {len(composants)} composant(s) ecrit(s) dans system.yml.")


if __name__ == "__main__":
    main()
