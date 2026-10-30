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
"""Fonctions du générateur direct Xpansion -> GemsPy.

Ce module contient le détail des traitements appelés, dans l'ordre, par le
script principal ``xpansion_direct_generator.py``. Il s'appuie autant que
possible sur les projets existants :

- **AntaresCraft** (``antares.craft``) pour lire l'étude Antares de base :
  zones, clusters thermiques (capacité, état ``enabled``, séries de
  disponibilité) et short-term storages ;
- **GemsPy** (``gems_craft`` / ``gems_craft_hybrid``) pour lire et vérifier la
  librairie de modèles, et pour décrire le ``system.yml`` avec les mêmes
  schémas pydantic que ceux qu'utilise GemsPy pour le relire.

Organisation :

1. Entrées Excel Xpansion
2. Tableau maître des candidats (``CandidatsX_generated_<année>.xlsx``)
3. Librairie de modèles
4. Composants thermiques (TAC/CCG)
5. Candidats batteries (journalisés, pas encore de composant)
6. Écriture du ``system.yml``, de l'``optim-config.yml`` et du scenario builder
7. Journal des composants créés / non créés
"""

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from antares.craft import Study, read_study_local
from antares.craft.exceptions.exceptions import InvalidRequestForScenarioBuilder
from antares.craft.model.scenario_builder import ScenarioBuilder
from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.resolve_library import resolve_library
from gems_craft.study.parsing import ComponentParameterSchema, ComponentSchema
from gems_craft_hybrid.study.parsing import (
    HybridSystemSchema,
    ThermalCapacityConnectionSchema,
    ThermalComponentSchema,
)

from antares_gems_converter.input_converter.src.scenario_builder import (
    get_legacy_sb,
    write_scenario_builder_file,
)

logger = logging.getLogger(__name__)

# =============================================================================
# 1. ENTRÉES EXCEL XPANSION
# =============================================================================


@dataclass(frozen=True)
class XpansionInputFiles:
    """Fichiers Excel d'entrée du générateur."""

    caract_xp: Path
    countries: Path
    candidate_parameters: Path
    cluster_correspondence: Path

    def check_exist(self) -> None:
        for input_file in [
            self.caract_xp,
            self.countries,
            self.candidate_parameters,
            self.cluster_correspondence,
        ]:
            if not input_file.exists():
                raise FileNotFoundError(f"Fichier d'entrée introuvable: {input_file}")


@dataclass(frozen=True)
class CandidateTypeMapping:
    """Ligne du fichier de correspondance pour un type candidat (ex: ccg_ch4)."""

    component_name: str  # nom_composant_system_yml, ex: semibase
    cluster_suffix: Optional[str]  # nom_cluster_etude_antares, ex: _ccgt ch4 pcomp


def read_first_sheet(file_path: Path) -> pd.DataFrame:
    """Retourne le premier onglet d'un classeur, comme le script R."""
    data = pd.read_excel(file_path, sheet_name=0)
    data.columns = [str(column).strip() for column in data.columns]
    return data


def find_column(table: Any, *names: str) -> str:
    """Premier nom de colonne disponible parmi ses alias (DataFrame ou ligne)."""
    for name in names:
        if name in table:
            return name
    raise ValueError(f"Colonne absente du fichier de paramètres: {names}")


def check_caract_xp(file_path: Path) -> None:
    """Vérifie que le fichier de caractéristiques contient les colonnes attendues."""
    missing = {"xp_nodes", "mod_generateur"} - set(read_first_sheet(file_path).columns)
    if missing:
        raise ValueError(f"Colonnes absentes de {file_path}: {sorted(missing)}")


def read_areas(countries_file: Path) -> List[str]:
    """Zones (en minuscules, comme les identifiants Antares) des candidats."""
    areas = read_first_sheet(countries_file)
    if "area" not in areas.columns:
        raise ValueError(f"La colonne 'area' est absente de {countries_file}")
    return [str(value).strip().lower() for value in areas["area"].dropna()]


def read_candidate_parameters(
    parameters_file: Path, pattern: str, year: int
) -> pd.Series:
    """Ligne de X_parameters_candidates pour un type candidat et une année."""
    parameters = read_first_sheet(parameters_file)
    type_column = find_column(parameters, "type")
    year_column = find_column(parameters, "year")
    mask = (
        parameters[type_column].astype(str).str.contains(pattern, regex=False, na=False)
    )
    mask &= pd.to_numeric(parameters[year_column], errors="coerce") == year
    result = parameters.loc[mask]
    if result.empty:
        raise ValueError(f"Aucun paramètre trouvé pour '{pattern}' en {year}")
    return result.iloc[0]


def read_cluster_correspondence(
    correspondence_file: Path,
) -> Dict[str, CandidateTypeMapping]:
    """Type candidat -> nom de composant et suffixe du cluster Antares.

    Le suffixe est absent pour les types sans cluster thermique (batterie).
    """
    table = read_first_sheet(correspondence_file)
    required = {
        "nom_cluster_etude_antares",
        "type_dans_fichiers_entree",
        "nom_composant_system_yml",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(
            f"Colonnes absentes de {correspondence_file}: {sorted(missing)}"
        )

    table = table.dropna(
        subset=["type_dans_fichiers_entree", "nom_composant_system_yml"]
    )
    return {
        str(row["type_dans_fichiers_entree"]).strip(): CandidateTypeMapping(
            component_name=str(row["nom_composant_system_yml"]).strip(),
            cluster_suffix=(
                None
                if pd.isna(row["nom_cluster_etude_antares"])
                else str(row["nom_cluster_etude_antares"]).strip().lower()
            ),
        )
        for _, row in table.iterrows()
    }


# =============================================================================
# 2. TABLEAU MAÎTRE DES CANDIDATS
# =============================================================================

CANDIDATE_COLUMNS = [
    "type",
    "zone",
    "start_node",
    "end_node",
    "cost_per_year",
    "unit_size",
    "max_unit",
    "name",
    "id",
    "gisement_total",
    "Pmax",
]


def candidate_name(
    mapping: Dict[str, CandidateTypeMapping], candidate_type: str, area: str
) -> str:
    """Nom du candidat : <nom_composant_system_yml>_<zone>."""
    if candidate_type not in mapping:
        raise ValueError(
            f"Type '{candidate_type}' absent de la colonne 'type_dans_fichiers_entree' "
            f"du fichier de correspondance"
        )
    return f"{mapping[candidate_type].component_name}_{area}"


def build_battery_candidates(
    parameters_file: Path,
    areas: List[str],
    mapping: Dict[str, CandidateTypeMapping],
    year: int,
    first_id: int,
) -> pd.DataFrame:
    """Candidats batteries : une ligne par zone.

    Le gisement vient directement de gisement_ini ; pas d'unité ni de taille
    d'unité (laissées à 0 comme dans le script R). Les identifiants suivent le
    script R : first_id, first_id + 2, first_id + 4...
    """
    battery = read_candidate_parameters(parameters_file, "batterie", year)
    cost = battery[find_column(battery, "cost_per_year")]
    pmax = battery[find_column(battery, "Pmax")]
    gisement = battery[find_column(battery, "gisement_ini", "gisement_ini_2050")]

    candidates = [
        {
            "type": "batterie",
            "zone": area,
            "start_node": None,
            "end_node": None,
            "cost_per_year": cost,
            "unit_size": 0,
            "max_unit": 0,
            "name": candidate_name(mapping, "batterie", area),
            "id": first_id + 2 * index,
            "gisement_total": gisement,
            "Pmax": pmax,
        }
        for index, area in enumerate(areas)
    ]
    return pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)


def thermal_candidate_types(year: int) -> List[str]:
    """Types thermiques candidats : ch4 jusqu'en 2031, h2 ensuite (règle du script R)."""
    fuel = "ch4" if year <= 2031 else "h2"
    return [f"{technology}_{fuel}" for technology in ["tac", "ccg"]]


def build_thermal_candidates(
    parameters_file: Path,
    areas: List[str],
    mapping: Dict[str, CandidateTypeMapping],
    year: int,
    first_id: int,
) -> pd.DataFrame:
    """Candidats TAC et CCG : une ligne par technologie et par zone.

    Candidat discret si nbr_unit_initial est renseigné : gisement_total =
    max_unit x unit_size. Candidat continu sinon : max_unit vide et
    gisement_total = gisement_ini (MW). Identifiants consécutifs à partir de
    first_id.
    """
    candidates: List[Dict[str, Any]] = []
    for fuel_type in thermal_candidate_types(year):
        thermal = read_candidate_parameters(parameters_file, fuel_type, year)
        unit_size = thermal[find_column(thermal, "Pmax")]
        max_unit = thermal[
            find_column(thermal, "nbr_unit_initial", "nbr_unit_initial_2050")
        ]
        if pd.isna(max_unit):
            # Candidat continu : gisement donné directement en MW
            gisement = thermal[
                find_column(thermal, "gisement_ini", "gisement_ini_2050")
            ]
        else:
            gisement = pd.to_numeric(max_unit, errors="coerce") * pd.to_numeric(
                unit_size, errors="coerce"
            )
        for area in areas:
            candidates.append(
                {
                    "type": fuel_type,
                    "zone": area,
                    "start_node": None,
                    "end_node": None,
                    "cost_per_year": thermal[find_column(thermal, "cost_per_year")],
                    "unit_size": unit_size,
                    "max_unit": max_unit,
                    "name": candidate_name(mapping, fuel_type, area),
                    "id": first_id + len(candidates),
                    "gisement_total": gisement,
                    "Pmax": unit_size,
                }
            )
    return pd.DataFrame(candidates, columns=CANDIDATE_COLUMNS)


def write_candidates_excel(candidates: pd.DataFrame, output_file: Path) -> None:
    candidates.to_excel(output_file, index=False)


# =============================================================================
# 3. LIBRAIRIE DE MODÈLES
# =============================================================================


def copy_model_library(model_source: Path, library_dir: Path) -> Path:
    """Copie la librairie de modèles dans model-libraries/."""
    if not model_source.exists():
        raise FileNotFoundError(f"Fichier modèle introuvable: {model_source}")
    library_dir.mkdir(parents=True, exist_ok=True)
    destination = library_dir / model_source.name
    shutil.copy2(model_source, destination)
    return destination


def check_models_in_library(library_file: Path, model_ids: List[str]) -> None:
    """Charge la librairie avec GemsPy et vérifie qu'elle définit les modèles utilisés.

    model_ids sont au format du system.yml : "<id librairie>.<id modèle>".
    """
    with library_file.open(encoding="utf-8") as file:
        libraries = resolve_library([parse_yaml_library(file)])
    available = {
        model_id for library in libraries.values() for model_id in library.models
    }
    missing = [model_id for model_id in model_ids if model_id not in available]
    if missing:
        raise ValueError(f"Modèles absents de {library_file}: {missing}")


# =============================================================================
# 4 & 5. COMPOSANTS (schémas GemsPy)
# =============================================================================

# Antares (10.2) ne sait connecter un port de capacité qu'à un cluster
# thermique (thermal-capacity-connections) : un candidat batterie n'aurait
# aucun effet sur le short-term storage, et sa variable, absente des
# sous-problèmes, fait échouer le Benders d'Xpansion.
BATTERY_NOT_SUPPORTED = (
    "connexion aux short-term storages non supportée par Antares "
    "(st-storage-capacity-connections)"
)


# Variables d'investissement des modèles de la librairie
# antares_legacy_investment_candidates
CONTINUOUS_INVESTMENT_VARIABLE = "invested_capacity"
DISCRETE_INVESTMENT_VARIABLE = "invested_units"


@dataclass(frozen=True)
class ComponentSettings:
    """Réglages communs à tous les composants d'une famille.

    continuous_model / discrete_model : modèles des candidats sans / avec
    nombre d'unités (max_unit).
    """

    continuous_model: str
    discrete_model: str
    component_prefix: str
    port: str

    def investment_variables(self) -> Dict[str, str]:
        """{modèle: variable d'investissement}."""
        return {
            self.continuous_model: CONTINUOUS_INVESTMENT_VARIABLE,
            self.discrete_model: DISCRETE_INVESTMENT_VARIABLE,
        }


@dataclass(frozen=True)
class ComponentLogEntry:
    """Un candidat et son devenir : créé dans le system.yml, ou non et pourquoi."""

    family: str
    component: str
    type: str
    zone: str
    antares_target: str
    reason: Optional[str] = None  # None = composant créé

    @property
    def created(self) -> bool:
        return self.reason is None


@dataclass
class FamilyComponents:
    """Composants, connexions, scenario groups et journal d'une famille de candidats.

    scenario_groups : {scenario group: {année MC (0-based): colonne de la série (1-based)}}
    """

    components: List[ComponentSchema] = field(default_factory=list)
    connections: List[ThermalCapacityConnectionSchema] = field(default_factory=list)
    scenario_groups: Dict[str, Dict[int, int]] = field(default_factory=dict)
    log: List[ComponentLogEntry] = field(default_factory=list)


def constant_parameter(param_id: str, value: Any) -> ComponentParameterSchema:
    """Paramètre constant (ni temporel ni scénario) d'un composant."""
    number = float(value)
    if pd.isna(number):
        raise ValueError(f"Valeur manquante pour le paramètre '{param_id}'")
    return ComponentParameterSchema(id=param_id, value=number)


def investment_model_and_parameters(
    candidate: pd.Series,
    existing_capacity: float,
    series_name: str,
    settings: ComponentSettings,
) -> Tuple[str, List[ComponentParameterSchema]]:
    """Modèle et paramètres d'un candidat (librairie antares_legacy_investment_candidates).

    - modèle discret si max_unit est renseigné, continu sinon ;
    - existing_capacity : capacité installée du cluster connecté, que la
      thermal-capacity-connection remplacerait sinon ;
    - max_investment : gisement_total (MW), min_investment : 0 ;
    - annual_investment_cost (par MW et par an) : cost_per_year, donné par
      unité de Pmax (unit_size) dans les fichiers d'entrée, divisé par unit_size.
    """
    unit_size = float(candidate["unit_size"])
    if pd.isna(unit_size) or unit_size <= 0:
        raise ValueError(
            f"Taille d'unité (Pmax) absente ou nulle pour le type '{candidate['type']}'"
        )
    parameters = [
        constant_parameter("existing_capacity", existing_capacity),
        constant_parameter("min_investment", 0),
        constant_parameter("max_investment", candidate["gisement_total"]),
        constant_parameter(
            "annual_investment_cost", float(candidate["cost_per_year"]) / unit_size
        ),
        ComponentParameterSchema(
            id="availability_rate",
            value=series_name,
            time_dependent=True,
            scenario_dependent=True,
        ),
    ]
    if pd.isna(candidate["max_unit"]):
        return settings.continuous_model, parameters
    return (
        settings.discrete_model,
        [constant_parameter("unit_size", unit_size)] + parameters,
    )


def build_thermal_component_without_cluster(
    candidate: pd.Series,
    component_id: str,
    cluster_id: str,
    settings: ComponentSettings,
) -> Optional[Tuple[ComponentSchema, Optional[ThermalCapacityConnectionSchema]]]:
    """À compléter : candidat thermique dont le cluster n'existe pas dans l'étude legacy.

    Appelée uniquement quand le cluster <zone><suffixe> est absent de l'étude
    Antares (pas quand il existe mais est vide). Pour créer quand même le
    composant d'investissement, retourner (composant, connexion) :

    - composant : ComponentSchema construit avec
      investment_model_and_parameters ; il faudra décider d'où viennent
      existing_capacity et availability_rate, puisqu'il n'y a ni cluster ni
      série ;
    - connexion : ThermalCapacityConnectionSchema, ou None si le composant ne
      doit pas être connecté (le cluster n'existant pas, une connexion vers
      cluster_id n'aurait de sens que s'il est créé côté Antares).

    Retourner None : le candidat n'est pas créé et le journal indique
    « cluster absent » (comportement actuel).
    """
    return None


def read_antares_study(study_path: Path) -> Study:
    """Lit l'étude Antares de base avec AntaresCraft."""
    return read_study_local(study_path)


def _thermal_cluster_exists(study: Study, area: str, cluster_id: str) -> bool:
    areas = study.get_areas()
    return area in areas and cluster_id in areas[area].get_thermals()


def _unusable_cluster_reason(study: Study, area: str, cluster_id: str) -> Optional[str]:
    """Raison pour laquelle un cluster existant est inutilisable, None sinon.

    Antares ignore un cluster désactivé : une thermal-capacity-connection vers
    lui arrête la simulation (« Area has no thermal cluster named ... »).
    """
    properties = study.get_areas()[area].get_thermals()[cluster_id].properties
    if not properties.enabled:
        return "cluster désactivé (enabled = false)"
    if properties.nominal_capacity * properties.unit_count <= 0:
        return "cluster vide (capacité installée nulle)"
    return None


def thermal_year_ts(
    legacy_sb: Optional[ScenarioBuilder],
    area: str,
    cluster_id: str,
    nb_years: int,
    nb_columns: int,
) -> Dict[int, int]:
    """{année MC (0-based): colonne de la série (1-based)} d'un cluster thermique.

    Reprend le scenario builder legacy du cluster. Une année sans entrée (tirage
    aléatoire côté Antares) prend la colonne (année modulo nb_columns) + 1.
    """
    scenario: List[Optional[int]] = []
    if legacy_sb is not None:
        try:
            scenario = legacy_sb.thermal.get_cluster(area, cluster_id).get_scenario()
        except InvalidRequestForScenarioBuilder:
            pass
    year_ts = {}
    for year in range(nb_years):
        ts = scenario[year] if year < len(scenario) else None
        year_ts[year] = ts if ts is not None else year % nb_columns + 1
    return year_ts


def build_thermal_components(
    candidates: pd.DataFrame,
    study: Study,
    mapping: Dict[str, CandidateTypeMapping],
    series_dir: Path,
    series_prefix: str,
    settings: ComponentSettings,
) -> FamilyComponents:
    """Composants TAC/CCG avec leur série de disponibilité.

    Pour chaque candidat, le cluster <zone><suffixe> de l'étude Antares fournit
    la série de disponibilité, divisée par sa capacité installée
    (nominal_capacity x unit_count) et écrite dans
    data-series/<series_prefix><composant>.tsv ; la capacité installée devient
    l'existing_capacity du candidat. Chaque composant a son propre
    scenario group (son id), les séries n'ayant pas toutes le même nombre de
    colonnes. Les candidats dont le cluster est absent, désactivé, vide ou sans
    série ne sont pas créés.
    """
    result = FamilyComponents()
    series_dir.mkdir(parents=True, exist_ok=True)
    legacy_sb = get_legacy_sb(study, logger)
    nb_years = study.get_settings().general_parameters.nb_years

    for _, row in candidates.iterrows():
        area = row["zone"]
        component_id = f"{settings.component_prefix}{row['name']}"
        suffix = mapping[row["type"]].cluster_suffix
        if suffix is None:
            raise ValueError(
                f"Type thermique '{row['type']}' sans 'nom_cluster_etude_antares' "
                f"dans le fichier de correspondance"
            )
        cluster_id = f"{area}{suffix}"

        def log(reason: Optional[str] = None) -> None:
            result.log.append(
                ComponentLogEntry(
                    "thermique", component_id, row["type"], area, cluster_id, reason
                )
            )

        if not _thermal_cluster_exists(study, area, cluster_id):
            fallback = build_thermal_component_without_cluster(
                row, component_id, cluster_id, settings
            )
            if fallback is None:
                log("cluster absent du list.ini de l'étude Antares")
                continue
            component, connection = fallback
            result.components.append(component)
            if connection is not None:
                result.connections.append(connection)
            log()
            continue

        reason = _unusable_cluster_reason(study, area, cluster_id)
        if reason:
            log(reason)
            continue

        cluster = study.get_areas()[area].get_thermals()[cluster_id]
        try:
            series = cluster.get_series_matrix()
        except FileNotFoundError:
            log("série de disponibilité absente (series.txt introuvable)")
            continue
        # AntaresCraft renvoie une série nulle quand series.txt est vide
        if (series == 0).all().all():
            log("série de disponibilité vide ou nulle (series.txt)")
            continue

        installed_capacity = (
            cluster.properties.nominal_capacity * cluster.properties.unit_count
        )
        series_name = f"{series_prefix}{component_id}"
        # Antares ne lit que les data-series .tsv ou .csv
        (series / installed_capacity).to_csv(
            series_dir / f"{series_name}.tsv",
            sep="\t",
            header=False,
            index=False,
            float_format="%.6g",
        )
        result.scenario_groups[component_id] = thermal_year_ts(
            legacy_sb, area, cluster_id, nb_years, series.shape[1]
        )

        model, parameters = investment_model_and_parameters(
            row, installed_capacity, series_name, settings
        )
        result.components.append(
            ComponentSchema(
                id=component_id,
                model=model,
                scenario_group=component_id,
                parameters=parameters,
            )
        )
        result.connections.append(
            ThermalCapacityConnectionSchema(
                component=component_id,
                port=settings.port,
                thermal_component=ThermalComponentSchema(
                    area=area, cluster_id=cluster_id
                ),
            )
        )
        log()

    return result


def build_battery_components(
    candidates: pd.DataFrame,
    study: Study,
    storage_suffix: str,
    component_prefix: str,
) -> FamilyComponents:
    """Candidats batteries : journalisés, aucun composant créé pour le moment.

    Le short-term storage <zone><storage_suffix> visé est cherché dans l'étude
    Antares pour que le journal distingue les zones sans storage de celles où
    seule la connexion manque (voir BATTERY_NOT_SUPPORTED).
    """
    result = FamilyComponents()
    areas = study.get_areas()

    for _, row in candidates.iterrows():
        area = row["zone"]
        storage_id = f"{area}{storage_suffix}"
        if area not in areas or storage_id not in areas[area].get_st_storages():
            reason = "short-term storage absent du list.ini de l'étude Antares"
        else:
            reason = BATTERY_NOT_SUPPORTED
        result.log.append(
            ComponentLogEntry(
                "batterie",
                f"{component_prefix}{row['name']}",
                row["type"],
                area,
                storage_id,
                reason,
            )
        )

    return result


# =============================================================================
# 6. ÉCRITURE DU SYSTEM.YML
# =============================================================================


class _IndentDumper(yaml.Dumper):
    """Indente les listes sous leur clé parente."""

    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def _parameter_to_yaml(parameter: ComponentParameterSchema) -> Dict[str, Any]:
    value = parameter.value
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return {
        "id": parameter.id,
        "value": value,
        "scenario-dependent": parameter.scenario_dependent,
        "time-dependent": parameter.time_dependent,
    }


def _component_to_yaml(component: ComponentSchema) -> Dict[str, Any]:
    return {
        "id": component.id,
        "model": component.model,
        "parameters": [
            _parameter_to_yaml(parameter) for parameter in component.parameters or []
        ],
        "scenario-group": component.scenario_group,
    }


def _dump_yaml(content: Dict[str, Any], output_file: Path) -> None:
    with open(output_file, "w", encoding="utf-8") as file:
        yaml.dump(
            content,
            file,
            Dumper=_IndentDumper,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )


def write_system_yml(system: HybridSystemSchema, output_file: Path) -> None:
    """Écrit le system.yml.

    Le contenu est celui du schéma GemsPy ; la mise en forme (ordre des clés,
    entiers sans décimale, listes indentées, pas de integer-strategy par
    défaut) reprend celle des versions précédentes du générateur, ce que
    gems_craft.study.parsing.write_yaml_system ne permet pas.
    """
    _dump_yaml(
        {
            "system": {
                "components": [
                    _component_to_yaml(component) for component in system.components
                ],
                "id": system.id,
                "thermal-capacity-connections": [
                    connection.model_dump(by_alias=True)
                    for connection in system.thermal_capacity_connections or []
                ],
            }
        },
        output_file,
    )


def write_optim_config(investment_variables: Dict[str, str], output_file: Path) -> None:
    """Écrit l'optim-config.yml, exigé par Antares dès qu'un system.yml existe.

    investment_variables : {modèle utilisé dans le system.yml: sa variable
    d'investissement}. Les variables d'investissement ne dépendant pas du
    scénario, l'étude doit être résolue par décomposition de Benders (Xpansion,
    étape « gems »). Sans model-decomposition, le problème maître est vide : la
    variable va dans le maître et les sous-problèmes, le coût d'investissement
    dans le maître.
    """
    _dump_yaml(
        {
            "resolution-mode": "benders-decomposition",
            "models": [
                {
                    "id": model,
                    "model-decomposition": {
                        "variables": [
                            {"id": variable, "location": "master-and-subproblems"}
                        ],
                        "objective-contributions": [
                            {"id": "objective", "location": "master"}
                        ],
                    },
                }
                for model, variable in investment_variables.items()
            ],
        },
        output_file,
    )


def write_modeler_scenario_builder(
    scenario_groups: Dict[str, Dict[int, int]], series_dir: Path
) -> None:
    """Écrit data-series/modeler-scenariobuilder.dat (une ligne par groupe et année)."""
    write_scenario_builder_file(scenario_groups, series_dir, logger)


# =============================================================================
# 7. JOURNAL DES COMPOSANTS CRÉÉS / NON CRÉÉS
# =============================================================================


def _reason_counts(entries: List[ComponentLogEntry]) -> Dict[str, int]:
    """Nombre de candidats non créés par raison, dans l'ordre d'apparition."""
    counts: Dict[str, int] = {}
    for entry in entries:
        if entry.reason is not None:
            counts[entry.reason] = counts.get(entry.reason, 0) + 1
    return counts


def print_family_summary(family: str, entries: List[ComponentLogEntry]) -> None:
    created = sum(entry.created for entry in entries)
    print(
        f"  ✓ {created}/{len(entries)} composant(s) {family}(s) créé(s) dans le system.yml"
    )
    for reason, count in _reason_counts(entries).items():
        print(f"  ⚠ {count} non créé(s) : {reason}")


def write_generation_log(
    output_file: Path,
    year: int,
    antares_study_path: Path,
    system_file: Path,
    entries: List[ComponentLogEntry],
) -> None:
    """Écrit le résumé des composants créés / non créés, famille par famille."""
    lines = [
        "RÉSUMÉ DE LA GÉNÉRATION DU SYSTEM.YML",
        "=" * 80,
        f"Date              : {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"Année             : {year}",
        f"Étude Antares     : {antares_study_path}",
        f"Fichier system.yml: {system_file}",
        "",
    ]
    for family in ["thermique", "batterie"]:
        family_entries = [entry for entry in entries if entry.family == family]
        created = [entry for entry in family_entries if entry.created]
        not_created = [entry for entry in family_entries if not entry.created]
        lines += [
            "-" * 80,
            f"{family.upper()}S : {len(created)} créé(s) / {len(family_entries)} candidat(s)",
            "-" * 80,
        ]
        if not_created:
            lines.append("Non créés, par raison :")
            for reason, count in _reason_counts(family_entries).items():
                lines.append(f"  - {reason} : {count}")
        lines += ["", f"Créés ({len(created)}) :"]
        lines += [
            f"  + {entry.component:<40} -> {entry.antares_target}" for entry in created
        ] or ["  (aucun)"]
        lines += ["", f"Non créés ({len(not_created)}) :"]
        lines += [
            f"  - {entry.component:<40} -> {entry.antares_target:<25} : {entry.reason}"
            for entry in not_created
        ] or ["  (aucun)"]
        lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")
