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
import logging
from pathlib import Path

# Antares >= 10.2 requires this file next to system.yml. Default content: no
# per-model setting, so the solver defaults apply (sequential-subproblems,
# cyclic out-of-bounds processing). Also valid for GemsPy.
OPTIM_CONFIG_FILE = "optim-config.yml"
DEFAULT_OPTIM_CONFIG = "models: []\n"


def write_optim_config(path: Path, logger: logging.Logger) -> None:
    # Keep an existing optim-config.yml (e.g. from a hybrid input study)
    if path.exists():
        return
    path.write_text(DEFAULT_OPTIM_CONFIG)
    logger.info(f"Generated optim-config file at {path}")
