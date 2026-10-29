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
"""Xpansion inputs (Excel files + base Antares study) whose optimum is known.

- fr: load 1000 MW, base cluster 600 MW at 20 EUR/MWh, CCGT and OCGT "pcomp"
  clusters (availability 90 % and 100 %), battery short-term storage;
- de: load 450 MW, CCGT "pcomp" cluster, empty OCGT "pcomp" cluster;
- it: in the Excel zones, not in the study;
- be (optional): OCGT "pcomp" cluster, disabled.

Candidates: OCGT ("peak") 10 units of 50 MW at 1.5 MEUR/unit/year, CCGT
("semibase") 3 units of 100 MW at 6 MEUR/unit/year. Unsupplied energy costs
1000 EUR/MWh, so every candidate is built up to the load: fr 3 CCGT + 2.6 OCGT
units (OCGT covers 1000 - 600 - 0.9 x 300 = 130 MW), de 3 CCGT units.
"""

import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from antares.craft import (
    AreaProperties,
    GeneralParametersUpdate,
    STStorageProperties,
    Study,
    StudySettingsUpdate,
    ThermalClusterProperties,
    create_study_local,
)
from antares.craft.model.area import Area

from antares_gems_converter.xpansion_generator.xpansion_direct_generator import (
    generate,
)

HOURS = 8760
HOURS_SIMULATED = 8736  # 52 weeks
UNSUPPLIED_COST = 1000

CCGT_COST_PER_UNIT = 6.0e6
OCGT_COST_PER_UNIT = 1.5e6

FR_CCGT = "fr_ccgt ch4 pcomp"
FR_OCGT = "fr_ocgt ch4 pcomp"
DE_CCGT = "de_ccgt ch4 pcomp"
DE_OCGT = "de_ocgt ch4 pcomp"
BE_OCGT = "be_ocgt ch4 pcomp"

PEAK_FR = "candidat_invest_peak_fr"
SEMIBASE_FR = "candidat_invest_semibase_fr"
SEMIBASE_DE = "candidat_invest_semibase_de"


def write_excel_inputs(input_dir: Path, areas: List[str]) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"xp_nodes": ["FR"], "mod_generateur": ["x"]}).to_excel(
        input_dir / "X_caract_generateur.xlsx", index=False
    )
    pd.DataFrame({"area": [area.upper() for area in areas]}).to_excel(
        input_dir / "List_countries.xlsx", index=False
    )
    pd.DataFrame({"area": ["FR"]}).to_excel(
        input_dir / "List_countries_fr.xlsx", index=False
    )
    pd.DataFrame(
        [
            ["batterie", 2031, 4.0e6, 100, None, 500],
            ["tac_ch4", 2031, OCGT_COST_PER_UNIT, 50, 10, None],
            ["ccg_ch4", 2031, CCGT_COST_PER_UNIT, 100, 3, None],
            # another year: must be ignored
            ["tac_ch4", 2035, 9.9e9, 50, 10, None],
        ],
        columns=[
            "type",
            "year",
            "cost_per_year",
            "Pmax",
            "nbr_unit_initial",
            "gisement_ini",
        ],
    ).to_excel(input_dir / "X_parameters_candidates.xlsx", index=False)
    pd.DataFrame(
        [
            ["_ocgt ch4 pcomp", "tac_ch4", "peak"],
            ["_ccgt ch4 pcomp", "ccg_ch4", "semibase"],
            [None, "batterie", "battery"],
        ],
        columns=[
            "nom_cluster_etude_antares",
            "type_dans_fichiers_entree",
            "nom_composant_system_yml",
        ],
    ).to_excel(input_dir / "correspondance_cluster_antares.xlsx", index=False)


def _create_area(study: Study, name: str, load: float) -> Area:
    area = study.create_area(
        name,
        properties=AreaProperties(
            energy_cost_unsupplied=UNSUPPLIED_COST, energy_cost_spilled=0
        ),
    )
    area.set_load(pd.DataFrame(np.full((HOURS, 1), load)))
    return area


def _create_cluster(
    area: Area,
    name: str,
    nominal_capacity: float,
    marginal_cost: float,
    availability: np.ndarray,
    enabled: bool = True,
) -> None:
    cluster = area.create_thermal_cluster(
        name,
        ThermalClusterProperties(
            unit_count=1,
            nominal_capacity=nominal_capacity,
            marginal_cost=marginal_cost,
            market_bid_cost=marginal_cost,
            enabled=enabled,
        ),
    )
    cluster.set_series(pd.DataFrame(availability))


def create_base_study(
    root: Path,
    nb_years: int = 1,
    fr_ccgt_availability: Optional[List[float]] = None,
    with_disabled_cluster: bool = False,
) -> Study:
    """fr_ccgt_availability: one value (MW, out of 100) per series column."""
    study = create_study_local("base", "9.2", root)
    study.update_settings(
        StudySettingsUpdate(
            general_parameters=GeneralParametersUpdate(nb_years=nb_years)
        )
    )

    fr = _create_area(study, "fr", 1000)
    _create_cluster(fr, "fr_base", 600, 20, np.full((HOURS, 1), 600.0))
    columns = fr_ccgt_availability or [90.0]
    _create_cluster(
        fr, FR_CCGT, 100, 50, np.column_stack([np.full(HOURS, v) for v in columns])
    )
    _create_cluster(fr, FR_OCGT, 50, 100, np.full((HOURS, 1), 50.0))
    fr.create_st_storage(
        "fr_battery_pcomp",
        STStorageProperties(
            injection_nominal_capacity=100,
            withdrawal_nominal_capacity=100,
            reservoir_capacity=200,
        ),
    )

    de = _create_area(study, "de", 450)
    _create_cluster(de, DE_CCGT, 100, 50, np.full((HOURS, 1), 100.0))
    _create_cluster(de, DE_OCGT, 0, 100, np.full((HOURS, 1), 0.0))

    if with_disabled_cluster:
        be = _create_area(study, "be", 100)
        _create_cluster(be, BE_OCGT, 50, 100, np.full((HOURS, 1), 50.0), enabled=False)
    return study


def assemble_hybrid_study(base_study: Path, generated: Path, hybrid: Path) -> Path:
    """Copy of the base study with the generated files in input/, plus the
    Xpansion settings needed by antares-xpansion-launcher."""
    shutil.copytree(base_study, hybrid)
    for name in ["system.yml", "optim-config.yml"]:
        shutil.copy2(generated / name, hybrid / "input" / name)
    for name in ["model-libraries", "data-series"]:
        shutil.copytree(generated / name, hybrid / "input" / name)
    settings = hybrid / "user" / "expansion" / "settings.ini"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text(
        "optimality_gap = 0\nrelative_gap = 1e-8\nmax_iteration = 200\nsolver = Cbc\n"
    )
    return hybrid


def run_generator(
    root: Path,
    areas: Optional[List[str]] = None,
    nb_years: int = 1,
    fr_ccgt_availability: Optional[List[float]] = None,
    with_disabled_cluster: bool = False,
) -> Tuple[Study, Path]:
    """Build the inputs under root and run the generator; return the base study
    and the output folder."""
    write_excel_inputs(root / "xlsx", areas or ["fr", "de", "it"])
    study = create_base_study(
        root, nb_years, fr_ccgt_availability, with_disabled_cluster
    )
    output_dir = root / "generated"
    generate(root / "xlsx", root / "base", output_dir)
    return study, output_dir
