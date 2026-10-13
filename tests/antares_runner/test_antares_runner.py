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
from pathlib import Path

from antares_runner.antares_runner import read_simulation_tables


def _write_table(simulation_dir: Path, name: str, scenario: int) -> None:
    (simulation_dir / name).write_text(f"scenario_index,value\n{scenario},1.0\n")


def test_read_simulation_tables_orders_years(tmp_path: Path):
    # Year 10 is sorted after year 2, not before
    for year in [10, 2, 0]:
        _write_table(tmp_path, f"simulation-table-{year}-optim-nb-1.csv", year)
    _write_table(tmp_path, "simulation-table-0-optim-nb-2.csv", -1)

    table = read_simulation_tables(tmp_path, "simulation-table-*-optim-nb-1.csv")

    assert table is not None
    assert table["scenario_index"].tolist() == [0, 2, 10]


def test_read_simulation_tables_no_table(tmp_path: Path):
    _write_table(tmp_path, "simulation-table-0-optim-nb-1.csv", 0)

    assert read_simulation_tables(tmp_path, "simulation-table-*-optim-nb-2.csv") is None
