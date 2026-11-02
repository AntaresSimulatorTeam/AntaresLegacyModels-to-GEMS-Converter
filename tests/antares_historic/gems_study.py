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
"""Build small GEMS studies in memory and solve them with antares-modeler."""
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

from antares_runner.antares_runner import AntaresModelerRunner

Value = Union[float, str]


def component(component_id: str, model: str, **parameters: Value) -> dict:
    """A component whose parameters are constants (numbers) or data series ids (strings)."""
    return {
        "id": component_id,
        "model": model,
        "parameters": [
            {
                "id": parameter_id,
                "time-dependent": isinstance(value, str),
                "scenario-dependent": isinstance(value, str),
                "value": value,
            }
            for parameter_id, value in parameters.items()
        ],
    }


def connection(component1: str, port1: str, component2: str, port2: str) -> dict:
    return {
        "component1": component1,
        "port1": port1,
        "component2": component2,
        "port2": port2,
    }


@dataclass
class GemsStudy:
    libraries: list[Path]
    components: list[dict] = field(default_factory=list)
    connections: list[dict] = field(default_factory=list)
    data_series: dict[str, np.ndarray] = field(default_factory=dict)
    last_time_step: int = 167

    def write(self, study_dir: Path) -> None:
        if study_dir.exists():
            shutil.rmtree(study_dir)
        input_dir = study_dir / "input"
        (input_dir / "model-libraries").mkdir(parents=True)
        (input_dir / "data-series").mkdir()
        for library in self.libraries:
            shutil.copy(library, input_dir / "model-libraries" / library.name)
        for series_id, values in self.data_series.items():
            np.savetxt(
                input_dir / "data-series" / f"{series_id}.csv",
                np.reshape(values, (len(values), -1)),
                delimiter="\t",
            )
        system = {
            "system": {
                "id": "system",
                "components": self.components,
                "connections": self.connections,
            }
        }
        (input_dir / "system.yml").write_text(yaml.safe_dump(system, sort_keys=False))
        (input_dir / "optim-config.yml").write_text(yaml.safe_dump({"models": []}))
        parameters = {
            "solver": "highs",
            "solver-logs": False,
            "solver-parameters": "",
            "no-output": False,
            "first-time-step": 0,
            "last-time-step": self.last_time_step,
        }
        (study_dir / "parameters.yml").write_text(yaml.safe_dump(parameters))

    def solve(self, study_dir: Path, exec_folder: Path) -> "GemsResult":
        self.write(study_dir)
        runner = AntaresModelerRunner(exec_folder, study_dir)  # type: ignore[arg-type]
        runner.run()
        return GemsResult(runner.simulation_table)


@dataclass
class GemsResult:
    table: pd.DataFrame

    @property
    def objective(self) -> float:
        values = self.table.query("output == 'OBJECTIVE_VALUE'")["value"]
        return float(values.iloc[0])

    def series(self, component_id: str, output: str) -> np.ndarray:
        rows = self.table[
            (self.table["component"] == component_id) & (self.table["output"] == output)
        ].sort_values("absolute_time_index")
        return rows["value"].to_numpy()
