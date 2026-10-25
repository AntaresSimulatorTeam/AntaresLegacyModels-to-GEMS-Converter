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

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Mapping, Optional, TextIO, Union

import pandas as pd
from gems_craft.model.parsing import LibrarySchema, parse_yaml_library
from pydantic import BaseModel, Field, model_validator
from yaml import safe_load


class ObjectKind(StrEnum):
    """Type of the study objects that a template is instantiated for."""

    AREA = "area"
    LINK = "link"
    THERMAL = "thermal"
    RENEWABLE = "renewable"
    ST_STORAGE = "st_storage"


CLUSTER_KINDS = (ObjectKind.THERMAL, ObjectKind.RENEWABLE, ObjectKind.ST_STORAGE)
# Attributes of a link that templates can reference as ${link}.<attribute>
LINK_ATTRIBUTES = ("area_from_id", "area_to_id")
LINK_ATTRIBUTE_PATTERN = re.compile(r"\$\{link\}\.(\w*)")
PLACEHOLDER_PATTERN = re.compile(r"\$\{[^}]*\}")


def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


class ModifiedBaseModel(BaseModel):
    class Config:
        alias_generator = _to_kebab
        extra = "forbid"
        populate_by_name = True


class Operation(ModifiedBaseModel):
    type: Optional[str] = None
    multiply_by: Optional[Union[float, str]] = None
    divide_by: Optional[Union[float, str]] = None

    def execute(
        self,
        initial_value: Union[pd.DataFrame, pd.Series, float],
        preprocessed_values: Optional[Union[dict[str, float], float]] = None,
    ) -> Union[float, pd.Series, pd.DataFrame]:
        if self.type == "max":
            return float(max(initial_value))  # type: ignore

        operand = self.multiply_by if self.multiply_by is not None else self.divide_by
        if operand is None:
            raise ValueError(
                "Operation must have at least one of 'multiply_by', 'divide_by', or 'type'"
            )
        if isinstance(operand, str):
            # The id of another parameter, whose value is in preprocessed_values
            if (
                not isinstance(preprocessed_values, dict)
                or operand not in preprocessed_values
            ):
                raise ValueError(
                    f"Missing value for key '{operand}' in preprocessed_values"
                )
            operand = preprocessed_values[operand]
        if self.multiply_by is not None:
            return initial_value * operand
        return initial_value / operand


class ObjectProperties(ModifiedBaseModel):
    type: str
    area: Optional[str] = None
    link: Optional[str] = None
    cluster: Optional[str] = None
    binding_constraint_id: Optional[str] = None
    field: Optional[str] = None


ALLOWED_TYPES: list = [
    "area",
    "binding_constraint",
    "thermal",
    "renewable",
    "link",
    "st_storage",
    "load",
    "solar",
    "wind",
    "misc_gen",
    "hydro",
]


class ConversionValue(ModifiedBaseModel):
    object_properties: Optional[ObjectProperties] = None
    column: Optional[int] = None
    operation: Optional[Operation] = None
    constant: Optional[float] = None

    def check_validity(self) -> bool:
        if not self.object_properties:
            raise ValueError(f"Object properties from {self} must not be None")
        if self.object_properties.type not in ALLOWED_TYPES:
            raise ValueError(f"Unknown value type: {self.object_properties.type}")
        return True


class PropertyConversionValue(ConversionValue):
    constant: Optional[Union[float, str]] = None  # type: ignore[assignment]


class ParameterConversionConfig(ModifiedBaseModel):
    id: str
    time_dependent: bool = False
    scenario_dependent: bool = False
    value: ConversionValue


class PropertyConversionConfig(ModifiedBaseModel):
    id: str
    value: PropertyConversionValue


class ComponentConversionConfig(ModifiedBaseModel):
    id: str
    scenario_group: Optional[str] = None
    parameters: list[ParameterConversionConfig]
    properties: list[PropertyConversionConfig] = Field(default_factory=list)


class ReferencedLegacyObjects(ModifiedBaseModel):
    id: str
    object_properties: ObjectProperties


class AreaConnectionConversionConfig(ModifiedBaseModel):
    component: str
    port: str
    area: str


class PortConnectionConversionConfig(ModifiedBaseModel):
    component1: str
    port1: str
    component2: str
    port2: str


class TemplateParameter(ModifiedBaseModel):
    name: str
    description: Optional[str] = None
    cluster_type: Optional[str] = None
    exclude: Optional[list[ReferencedLegacyObjects]] = None


class TemplateBody(ModifiedBaseModel):
    """What a template converts a study object to."""

    name: str
    model: str
    components: list[ComponentConversionConfig]
    connections: list[PortConnectionConversionConfig] = Field(default_factory=list)
    area_connections: list[AreaConnectionConversionConfig] = Field(default_factory=list)
    legacy_objects_to_delete: list[ReferencedLegacyObjects] = Field(
        default_factory=list
    )


class TemplateInstance(TemplateBody):
    """A template instantiated for one study object: no placeholder left."""


class ConversionTemplate(TemplateBody):
    """A conversion template, as read from YAML, with placeholders."""

    generator_version_compatibility: Optional[str] = None
    template_parameters: list[TemplateParameter] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_template_parameters(self) -> "ConversionTemplate":
        # A template is instantiated for each link, area or cluster of a given
        # type, which its single parameter defines (see `for_each`)
        if len(self.template_parameters) != 1:
            raise ValueError(
                f"Template '{self.name}' must have exactly one template "
                f"parameter, got {len(self.template_parameters)}"
            )
        parameter = self.template_parameters[0]
        if parameter.name not in (ObjectKind.AREA, ObjectKind.LINK):
            raise ValueError(
                f"Template '{self.name}': the template parameter must be named "
                f"'area' or 'link', got '{parameter.name}'"
            )
        if parameter.cluster_type is not None and (
            parameter.name != ObjectKind.AREA
            or parameter.cluster_type not in CLUSTER_KINDS
        ):
            raise ValueError(
                f"Template '{self.name}': cluster-type must be one of "
                f"{[str(kind) for kind in CLUSTER_KINDS]} on an 'area' "
                f"parameter, got '{parameter.cluster_type}'"
            )
        for match in LINK_ATTRIBUTE_PATTERN.finditer(self.model_dump_json()):
            if match.group(1) not in LINK_ATTRIBUTES:
                raise ValueError(
                    f"Template '{self.name}': unknown link attribute "
                    f"'{match.group()}', expected one of {list(LINK_ATTRIBUTES)}"
                )
        return self

    @property
    def for_each(self) -> ObjectKind:
        """The type of the study objects this template is instantiated for."""
        parameter = self.template_parameters[0]
        if parameter.name == ObjectKind.LINK:
            return ObjectKind.LINK
        if parameter.cluster_type:
            return ObjectKind(parameter.cluster_type)
        return ObjectKind.AREA

    def instantiate(self, substitutions: Mapping[str, str]) -> TemplateInstance:
        """Return the template for one study object: each placeholder of
        `substitutions` replaced by its value, in order, in every string."""
        # Replaced in the JSON text of the template, the values escaped as JSON
        text = self.model_dump_json(include=set(TemplateBody.model_fields))
        for placeholder, value in substitutions.items():
            text = text.replace(placeholder, json.dumps(value)[1:-1])
        if leftover := PLACEHOLDER_PATTERN.search(text):
            raise ValueError(
                f"Template '{self.name}': unresolved placeholder '{leftover.group()}'"
            )
        return TemplateInstance.model_validate_json(text)

    @property
    def excluded_objects(self) -> list[ObjectProperties]:
        """The objects this template is not instantiated for (see
        `templates.is_excluded`)."""
        return [
            excluded.object_properties
            for parameter in self.template_parameters
            for excluded in parameter.exclude or []
        ]


def parse_conversion_template(input_template: TextIO) -> ConversionTemplate:
    tree = safe_load(input_template)
    return ConversionTemplate.model_validate(tree["template"])


def parse_library(path: str) -> LibrarySchema:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return parse_yaml_library(file)
    except Exception as e:
        raise ValueError(f"Failed to parse library file {path}: {e}") from e
