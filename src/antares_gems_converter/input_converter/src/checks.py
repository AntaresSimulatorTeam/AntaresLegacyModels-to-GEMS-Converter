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
"""Check the conversion templates against the GEMS library models."""
from pathlib import Path

from antares_gems_converter.input_converter.src.parsing import ConversionTemplate
from gems_craft.model.parsing import LibrarySchema, ModelSchema, parse_yaml_library


def parse_library(path: str) -> LibrarySchema:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return parse_yaml_library(file)
    except Exception as e:
        raise ValueError(f"Failed to parse library file {path}: {e}") from e


def check_converted_models_are_in_libs(
    model_conversion_templates: dict[str, ConversionTemplate], lib_paths: list[str]
) -> None:
    lib_to_models: dict[str, dict[str, ModelSchema]] = {}
    for lib_path in lib_paths:
        lib_data = parse_library(lib_path)
        lib_to_models[lib_data.id] = {model.id: model for model in lib_data.models}

    mismatches: list[str] = []
    for conversion_template in model_conversion_templates.values():
        lib_id, model_id = conversion_template.model.split(".")
        if lib_id not in lib_to_models:
            raise ValueError(
                f"Library {lib_id} has not been found in provided libraries"
            )
        if model_id not in lib_to_models[lib_id]:
            raise ValueError(f"Model {model_id} has not been found in library {lib_id}")
        mismatches += template_mismatches_with_model(
            conversion_template, lib_to_models[lib_id][model_id]
        )

    # A mismatch would otherwise only surface later, in GEMS or the solver,
    # far from the template, or go unnoticed
    if mismatches:
        raise ValueError(
            "Conversion templates do not match the parameters and properties"
            " of their library models:\n" + "\n".join(mismatches)
        )


def template_mismatches_with_model(
    conversion_template: ConversionTemplate, model: ModelSchema
) -> list[str]:
    model_ids = {
        "parameter": {param.id for param in model.parameters},
        "property": {prop.id for prop in model.properties},
    }
    mismatches = []
    for comp in conversion_template.components:
        template_ids = {
            "parameter": {param.id for param in comp.parameters},
            "property": {prop.id for prop in comp.properties},
        }
        for kind in ("parameter", "property"):
            for issue, ids in (
                ("unknown", template_ids[kind] - model_ids[kind]),
                ("missing", model_ids[kind] - template_ids[kind]),
            ):
                if ids:
                    mismatches.append(
                        f"- template '{conversion_template.name}', component"
                        f" '{comp.id}', model '{conversion_template.model}':"
                        f" {issue} {kind}(s) {sorted(ids)}"
                    )
    return mismatches
