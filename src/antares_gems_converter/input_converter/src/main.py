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
import os
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from configparser import ConfigParser, NoOptionError
from pathlib import Path
from typing import Optional

from antares_gems_converter.input_converter.src.data_preprocessing.data_classes import (
    ConversionMode,
)

from .converter import AntaresStudyConverter
from .logger import Logger

DEFAULT: dict = {}
LOGGER_PATH: str = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data/logging.log"
)


class CreateFileIfMissing:
    """
    Argparse type that accepts a file path and creates the file if it doesn't exist.
    Fails if the path points to an existing directory or if the parent directory is missing.
    """

    def __call__(self, string: str) -> Path:
        path = Path(string).expanduser().resolve()

        if path.exists() and not path.is_file():
            raise ArgumentTypeError(f"Path exists and is not a file: '{path}'")

        if not path.parent.exists():
            raise ArgumentTypeError(f"Parent directory does not exist: '{path.parent}'")

        if not path.exists():
            try:
                path.touch()
            except Exception as e:
                raise ArgumentTypeError(f"Failed to create file: {e}")

        return path


class PathType:
    def __init__(
        self,
        exists: bool = False,
        file_ok: bool = False,
        dir_ok: bool = False,
    ) -> None:
        if not (file_ok or dir_ok):
            msg = "Either `file_ok` or `dir_ok` must be set at a minimum."
            raise ValueError(msg)
        self.exists = exists
        self.file_ok = file_ok
        self.dir_ok = dir_ok

    def __call__(self, string: str) -> Path:
        file_path = Path(string).expanduser()
        if not self.exists:
            return file_path
        if self.file_ok and self.dir_ok:
            if file_path.exists():
                return file_path
            msg = f"The file or directory path does not exist: '{file_path}'"
            raise ArgumentTypeError(msg)
        elif self.file_ok:
            if file_path.is_file():
                return file_path
            elif file_path.exists():
                msg = f"The path is not a regular file: '{file_path}'"
            else:
                msg = f"The file path does not exist: '{file_path}'"
            raise ArgumentTypeError(msg)
        elif self.dir_ok:
            if file_path.is_dir():
                return file_path
            elif file_path.exists():
                msg = f"The path is not a directory: '{file_path}'"
            else:
                msg = f"The directory path does not exist: '{file_path}'"
            raise ArgumentTypeError(msg)
        else:  # pragma: no cover
            raise NotImplementedError((self.file_ok, self.dir_ok))


def parse_commandline() -> Namespace:
    """Parse command-line arguments using argparse to specify configuration file paths,
    logging options, and version display. Returns the parsed arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--conf",
        type=PathType(exists=True, file_ok=True),
        help="Give the path of the configuration file",
        default="data/config.ini",
    )
    parser.add_argument(
        "-l",
        "--logging",
        type=CreateFileIfMissing(),
        help="Give the path of the logger file",
        default=LOGGER_PATH,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_commandline()
    log_path = args.logging
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger: logging.Logger = Logger(__name__, log_path)

    config_parser = ConfigParser()

    # Load the default configuration dictionary into the config parser.
    config_parser.read_dict(DEFAULT)

    if args.conf:
        config_parser.read(args.conf)

    if not config_parser.has_section("study"):
        sys.exit(f"Aborting: missing 'study' section in configuration")

    try:
        path_validator = PathType(exists=True, dir_ok=True)
        study_path = path_validator(config_parser.get("study", "study_path"))
        output_folder = path_validator(config_parser.get("study", "output_folder"))
        lib_paths = [
            lib_path.strip()
            for lib_path in config_parser.get("study", "lib_paths").split(",")
            if lib_path.strip()
        ]
        try:
            for path in lib_paths:
                PathType(exists=True, file_ok=True)(path)
        except ArgumentTypeError as e:
            sys.exit(f"Aborting: Invalid lib_paths in configuration file: {e}")
    except ArgumentTypeError as e:
        sys.exit(f"Aborting: Invalid path in configuration file: {e}")
    except NoOptionError as e:
        sys.exit(f"Aborting: {e.message} in configuration file")

    raw_mode = config_parser.get("study", "mode", fallback=ConversionMode.FULL.value)
    try:
        mode = ConversionMode(raw_mode.strip().lower())
    except ValueError:
        sys.exit(
            f"Aborting: Invalid conversion mode '{raw_mode}', possible values are "
            f"{[conv_mode.value for conv_mode in ConversionMode]}"
        )

    models_to_convert: Optional[list[str]] = None
    if mode == ConversionMode.HYBRID:
        raw_model_list = config_parser.get("hybrid", "model_list", fallback="")
        models_to_convert = [
            model.strip() for model in raw_model_list.split(",") if model.strip()
        ]
        if not models_to_convert:
            sys.exit(
                "Aborting: 'hybrid' mode set but no 'model_list' in 'hybrid' section"
            )

    converter = AntaresStudyConverter(
        study_input=study_path,
        logger=logger,
        mode=mode.value,
        output_folder=output_folder,
        lib_paths=lib_paths,
        models_to_convert=models_to_convert,
    )
    converter.process_all()


if __name__ == "__main__":
    main()
