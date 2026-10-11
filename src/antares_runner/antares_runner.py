# Copyright (c) 2025, RTE (https://www.rte-france.com)
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


import os
import subprocess
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def read_simulation_tables(simulation_dir: Path, pattern: str) -> pd.DataFrame | None:
    # Since Antares 10.2, there is one simulation table per MC year, named
    # simulation-table-<year>[-<stage>].csv: concatenate them by year.
    files = sorted(
        simulation_dir.glob(pattern), key=lambda f: int(f.stem.split("-")[2])
    )
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


class AntaresHybridRunner:
    def __init__(
        self, exec_dir: Path, study_dir: Path, solver: None | str = None
    ) -> None:
        self.exec_dir, self.study_dir, self.solver = exec_dir, study_dir, solver
        self.exec_path = self.exec_dir / Path("antares-solver")
        self.OUTPUT_FILE_DIR_NAME = "output"
        self.SIMULATION_TABLE_1_STAGE = "optim-nb-1"
        self.SIMULATION_TABLE_2_STAGE = "optim-nb-2"

    def run(self) -> None:
        # Build the command to run antares-solver.exe
        # Since Antares 10.2, simulation tables are only written on request, and
        # there is one per stage: write only the stages read below.
        command = [
            str(self.exec_path.absolute()),
            str(self.study_dir.absolute()),
            "--output",
            "simulation-tables",
            "--simulation-table-stages",
            f"{self.SIMULATION_TABLE_1_STAGE},{self.SIMULATION_TABLE_2_STAGE}",
        ]
        if self.solver:
            command.append("--linear-solver")
            command.append(self.solver)
        # Run the command and wait for it to finish
        try:
            start = time.time()
            subprocess.run(
                command,
                check=True,  # raise CalledProcessError on failure
                capture_output=True,  # capture stdout/stderr if needed
                text=True,  # decode bytes to string
            )
            # Log or inspect result.stdout / result.stderr here if desired
            end = time.time()
            self.total_execution_time = end - start
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Antares execution failed with code {e.returncode}: {e.stderr}"
            )
        # Read the objective value from the solution file
        path = self.study_dir / Path(self.OUTPUT_FILE_DIR_NAME)
        folders = [
            name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
        ]
        self.simulation_id = sorted(folders)[-1]
        simulation_dir = path / self.simulation_id
        simulation_table_1 = read_simulation_tables(
            simulation_dir, f"simulation-table-*-{self.SIMULATION_TABLE_1_STAGE}.csv"
        )
        if simulation_table_1 is None:
            raise FileNotFoundError(
                f"No optim-nb-1 simulation table in {simulation_dir}"
            )
        self.simulation_table_1 = simulation_table_1
        # Since Antares 10.2, there is no optim-nb-2 table when the second
        # optimization does not run (e.g. unit-commitment-mode = milp)
        simulation_table_2 = read_simulation_tables(
            simulation_dir, f"simulation-table-*-{self.SIMULATION_TABLE_2_STAGE}.csv"
        )
        self.simulation_table_2 = (
            simulation_table_2
            if simulation_table_2 is not None
            else simulation_table_1.iloc[0:0]
        )


class AntaresHybridStudyBenchmarker:
    def __init__(
        self,
        exec_path: Path,
        study_path_1: Path,
        study_path_2: Path,
        solver: None | str = None,
    ) -> None:
        self.exec_path, self.study_path_1, self.study_path_2, self.solver = (
            exec_path,
            study_path_1,
            study_path_2,
            solver,
        )
        self.weekly_objectives_1: list = []
        self.weekly_objectives_2: list = []
        self.simulation_tables_1: list = []
        self.simulation_tables_2: list = []

    def run(self) -> None:
        for study in [self.study_path_1, self.study_path_2]:
            r = AntaresHybridRunner(self.exec_path, study, self.solver)
            r.run()
            self.weekly_objectives_1.append(
                r.simulation_table_1.query('output == "OBJECTIVE_VALUE"')[
                    "value"
                ].values
            )
            self.weekly_objectives_2.append(
                r.simulation_table_2.query('output == "OBJECTIVE_VALUE"')[
                    "value"
                ].values
            )
            self.simulation_tables_1.append(r.simulation_table_1)
            self.simulation_tables_2.append(r.simulation_table_2)

    def weekly_rel_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.abs(
            (self.weekly_objectives_1[0] - self.weekly_objectives_1[1])
            / self.weekly_objectives_1[0]
        ), np.abs(
            (self.weekly_objectives_2[0] - self.weekly_objectives_2[1])
            / self.weekly_objectives_2[0]
        )

    def weekly_abs_gaps(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.abs(
            self.weekly_objectives_1[0] - self.weekly_objectives_1[1]
        ), np.abs(self.weekly_objectives_2[0] - self.weekly_objectives_2[1])


class AntaresModelerRunner:
    def __init__(self, exec_dir: str, study_dir: str) -> None:
        self.exec_dir, self.study_dir = exec_dir, study_dir
        self.exec_path = self.exec_dir / Path("antares-modeler")
        self.OUTPUT_FILE_DIR_NAME = "output"

    def run(self) -> None:
        # Build the command to run antares-modeler.exe
        command = [str(self.exec_path), str(self.study_dir)]
        # Run the command and wait for it to finish
        try:
            start = time.time()
            subprocess.run(
                command,
                check=True,  # raise CalledProcessError on failure
                capture_output=True,  # capture stdout/stderr if needed
                text=True,  # decode bytes to string
            )
            # Log or inspect result.stdout / result.stderr here if desired
            end = time.time()
            self.total_execution_time = end - start
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Antares execution failed with code {e.returncode}: {e.stderr}"
            )
        # Read the objective value from the solution file
        # Since Antares 10.2, tables are written in a folder per simulation
        path = self.study_dir / Path(self.OUTPUT_FILE_DIR_NAME)
        self.simulation_id = sorted(os.listdir(path))[-1]
        simulation_dir = path / self.simulation_id
        simulation_table = read_simulation_tables(
            simulation_dir, "simulation-table-*.csv"
        )
        if simulation_table is None:
            raise FileNotFoundError(f"No simulation table in {simulation_dir}")
        self.simulation_table = simulation_table
