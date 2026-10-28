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
AD_HOC_NAME_PATTERN = re.compile(r"\w+")
# Fields an exclusion must set, by type (area and cluster for a cluster type)
EXCLUSION_FIELDS: dict[str, tuple[str, ...]] = {
    ObjectKind.AREA: ("area",),
    ObjectKind.LINK: ("link",),
}


def _to_kebab(snake: str) -> str:
    return snake.replace("_", "-")


def _substitute(text: str, substitutions: Mapping[str, str]) -> str:
    """Replace in `text`, the JSON text of a model, each placeholder of
    `substitutions` by its value, escaped as JSON, in one pass: the longest
    placeholder matches first ("${link}.area_from_id." before "${link}"), and
    no substituted value is substituted again."""
    if not substitutions:
        return text
    pattern = re.compile(
        "|".join(map(re.escape, sorted(substitutions, key=len, reverse=True)))
    )
    return pattern.sub(
        lambda match: json.dumps(substitutions[match.group()])[1:-1], text
    )


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

        multiply = self.multiply_by is not None
        operand = self.multiply_by if multiply else self.divide_by
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
        return initial_value * operand if multiply else initial_value / operand


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
    """A parameter of a template, which yields the values the template is
    instantiated for: the study objects (areas, links or clusters), restricted
    to `values` / `cluster-values` when given, or, for an ad-hoc parameter,
    its `values`."""

    name: str
    description: Optional[str] = None
    ad_hoc: bool = False
    values: Optional[list[str]] = None
    cluster_type: Optional[str] = None
    cluster_values: Optional[list[str]] = None
    exclude: Optional[list[ReferencedLegacyObjects]] = None

    @property
    def kind(self) -> Optional[ObjectKind]:
        """The type of the study objects this parameter yields, None for an
        ad-hoc parameter."""
        if self.ad_hoc:
            return None
        return ObjectKind(self.cluster_type or self.name)

    @property
    def placeholders(self) -> tuple[str, ...]:
        """The placeholders this parameter gives a value to."""
        if self.cluster_type is not None:
            return (f"${{{self.name}}}", f"${{{self.cluster_type}}}")
        return (f"${{{self.name}}}",)

    @property
    def excluded_objects(self) -> list[ObjectProperties]:
        """The objects this parameter does not yield (see
        `templates.is_excluded`)."""
        return [excluded.object_properties for excluded in self.exclude or []]

    def resolve(self, substitutions: Mapping[str, str]) -> "TemplateParameter":
        """Return the parameter with the placeholders of `substitutions`, those
        of the outer parameters, replaced in its lists and exclusions."""
        if not substitutions:
            return self
        text = _substitute(json.dumps(self.model_dump()), substitutions)
        return TemplateParameter.model_validate_json(text)


class TemplateBody(ModifiedBaseModel):
    """The fields shared by a conversion template and its instances: what a
    study object is converted to (the GEMS model, components and connections)
    and, in hybrid mode, the legacy objects to delete.

    Not used on its own: a ConversionTemplate holds these fields with
    placeholders such as ${area}, a TemplateInstance with the ids of one study
    object.
    """

    name: str
    model: str
    components: list[ComponentConversionConfig]
    connections: list[PortConnectionConversionConfig] = Field(default_factory=list)
    area_connections: list[AreaConnectionConversionConfig] = Field(default_factory=list)
    legacy_objects_to_delete: list[ReferencedLegacyObjects] = Field(
        default_factory=list
    )


class TemplateInstance(TemplateBody):
    """A ConversionTemplate instantiated for one area, link or cluster (see
    `ConversionTemplate.instantiate`): the same body, every placeholder replaced
    by the ids of that object. This is what the converter turns into GEMS
    components."""


class ConversionTemplate(TemplateBody):
    """A conversion template as read from YAML: the body, with placeholders
    such as ${area}, and the template parameters, whose values the template is
    instantiated for, nested in declaration order (outer first). At most one
    parameter yields study objects (`for_each`), and it skips the objects it
    excludes (`excluded_objects`); the others are ad-hoc."""

    generator_version_compatibility: Optional[str] = None
    template_parameters: list[TemplateParameter] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_template(self) -> "ConversionTemplate":
        if not self.template_parameters:
            raise ValueError(
                f"Template '{self.name}' must have at least one template parameter"
            )
        names = [parameter.name for parameter in self.template_parameters]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Template '{self.name}': template parameter names must be "
                f"unique, got {names}"
            )
        # An instance has at most one study object, hence one legacy SB entry
        legacy_names = [p.name for p in self.template_parameters if not p.ad_hoc]
        if len(legacy_names) > 1:
            raise ValueError(
                f"Template '{self.name}': at most one template parameter can "
                f"yield study objects, got {legacy_names}; mark the others ad-hoc"
            )
        # Ad-hoc components do not replace a legacy object
        if self.ad_hoc_parameters and self.legacy_objects_to_delete:
            raise ValueError(
                f"Template '{self.name}': legacy-objects-to-delete is not allowed "
                f"with an ad-hoc template parameter"
            )
        outer_placeholders: set[str] = set()
        # Those of the parameters declared before the one that yields study objects
        exclusion_placeholders: set[str] = set()
        for parameter in self.template_parameters:
            self._check_parameter(parameter, outer_placeholders)
            if not parameter.ad_hoc:
                exclusion_placeholders = set(outer_placeholders)
            outer_placeholders.update(parameter.placeholders)
        # A typo such as ${link}.area_frm_id would be substituted as ${link}
        # and leave no placeholder for `instantiate` to report (in the body
        # and in the lists of the parameters)
        for match in LINK_ATTRIBUTE_PATTERN.finditer(json.dumps(self.model_dump())):
            if match.group(1) not in LINK_ATTRIBUTES:
                raise ValueError(
                    f"Template '{self.name}': unknown link attribute "
                    f"'{match.group()}', expected one of {list(LINK_ATTRIBUTES)}"
                )
        # An ill-formed exclusion would silently exclude nothing, or more than
        # meant: {type: area, area: fr, cluster: c} excludes the whole area fr
        for excluded in self.excluded_objects:
            fields = EXCLUSION_FIELDS.get(excluded.type, ("area", "cluster"))
            set_fields = {name for name, value in excluded if value is not None}
            if excluded.type not in list(ObjectKind) or set_fields != {"type", *fields}:
                raise ValueError(
                    f"Template '{self.name}': invalid exclusion {excluded!r}, its "
                    f"type must be one of {', '.join(ObjectKind)} and it must set "
                    f"only {', '.join(fields)}"
                )
            # Only an area exclusion or one of the kind the template is
            # instantiated for can match (see `templates.is_excluded`); an
            # ad-hoc parameter has no exclusions, so for_each is not None
            assert self.for_each is not None
            allowed = dict.fromkeys((ObjectKind.AREA, self.for_each))
            if excluded.type not in allowed:
                raise ValueError(
                    f"Template '{self.name}': exclusion {excluded!r} never "
                    f"matches, the template is instantiated for each "
                    f"{self.for_each}: only {' and '.join(allowed)} exclusions apply"
                )
            # The placeholders substituted in an exclusion are those of the
            # outer parameters and, in a cluster exclusion only, ${area}:
            # {type: area, area: ${area}} matches no area
            substituted = exclusion_placeholders | (
                {"${area}"} if excluded.type in CLUSTER_KINDS else set()
            )
            for value in (excluded.area, excluded.link, excluded.cluster):
                for match in PLACEHOLDER_PATTERN.finditer(value or ""):
                    if match.group() not in substituted:
                        raise ValueError(
                            f"Template '{self.name}': exclusion {excluded!r} "
                            f"never matches, '{match.group()}' is not substituted "
                            f"in an exclusion of type {excluded.type}"
                        )
        return self

    def _check_parameter(
        self, parameter: TemplateParameter, outer_placeholders: set[str]
    ) -> None:
        """Check one template parameter; `outer_placeholders` are those of the
        parameters declared before it."""
        prefix = f"Template '{self.name}', parameter '{parameter.name}'"
        if parameter.ad_hoc:
            if not parameter.values:
                raise ValueError(f"{prefix}: an ad-hoc parameter requires values")
            if any(
                key is not None
                for key in (
                    parameter.cluster_type,
                    parameter.cluster_values,
                    parameter.exclude,
                )
            ):
                raise ValueError(
                    f"{prefix}: cluster-type, cluster-values and exclude are not "
                    f"allowed on an ad-hoc parameter"
                )
            # Its placeholder must not be mistaken for that of a study object
            if not AD_HOC_NAME_PATTERN.fullmatch(parameter.name) or (
                parameter.name in list(ObjectKind)
            ):
                raise ValueError(
                    f"{prefix}: the name of an ad-hoc parameter must be made of "
                    f"letters, digits and underscores, and not be one of "
                    f"{', '.join(ObjectKind)}"
                )
        elif parameter.name not in (ObjectKind.AREA, ObjectKind.LINK):
            raise ValueError(
                f"{prefix}: a template parameter must be named 'area' or 'link', "
                f"unless it is ad-hoc"
            )
        if parameter.cluster_type is not None and (
            parameter.name != ObjectKind.AREA
            or parameter.cluster_type not in CLUSTER_KINDS
        ):
            raise ValueError(
                f"Template '{self.name}': cluster-type must be one of "
                f"{', '.join(CLUSTER_KINDS)} on an 'area' parameter, "
                f"got '{parameter.cluster_type}'"
            )
        if parameter.cluster_values is not None and parameter.cluster_type is None:
            raise ValueError(f"{prefix}: cluster-values requires a cluster-type")
        for key, entries in (
            ("values", parameter.values),
            ("cluster-values", parameter.cluster_values),
        ):
            if entries is not None and (
                not entries or len(set(entries)) != len(entries)
            ):
                raise ValueError(
                    f"{prefix}: {key} must be a non-empty list without duplicates, "
                    f"got {entries}"
                )
        # The entries of the lists may use the placeholders of the outer
        # parameters, those of cluster-values the area of the cluster (the
        # exclusions are checked with the template's)
        lists = {
            "values": (parameter.values or [], outer_placeholders),
            "cluster-values": (
                parameter.cluster_values or [],
                outer_placeholders | {"${area}"},
            ),
        }
        for key, (entries, allowed) in lists.items():
            for entry in entries:
                for match in PLACEHOLDER_PATTERN.finditer(entry):
                    if match.group() not in allowed:
                        raise ValueError(
                            f"{prefix}: unknown placeholder '{match.group()}' in "
                            f"{key}, expected one of the outer parameters' "
                            f"{sorted(allowed)}"
                        )

    def _body_json(self) -> str:
        # json.dumps, unlike model_dump_json, keeps inf and nan constants
        return json.dumps(self.model_dump(include=set(TemplateBody.model_fields)))

    @property
    def legacy_parameter(self) -> Optional[TemplateParameter]:
        """The parameter that yields study objects, if any."""
        return next((p for p in self.template_parameters if not p.ad_hoc), None)

    @property
    def ad_hoc_parameters(self) -> list[TemplateParameter]:
        return [p for p in self.template_parameters if p.ad_hoc]

    @property
    def for_each(self) -> Optional[ObjectKind]:
        """The type of the study objects this template is instantiated for,
        None when all its parameters are ad-hoc."""
        parameter = self.legacy_parameter
        return parameter.kind if parameter else None

    def instantiate(self, substitutions: Mapping[str, str]) -> TemplateInstance:
        """Return the template for one combination of the values of its
        parameters: each placeholder of `substitutions` replaced by its value
        in every string."""
        # Replaced in the JSON text of the template (see `_substitute`)
        text = _substitute(self._body_json(), substitutions)
        if leftover := PLACEHOLDER_PATTERN.search(text):
            raise ValueError(
                f"Template '{self.name}': unresolved placeholder '{leftover.group()}'"
            )
        return TemplateInstance.model_validate_json(text)

    @property
    def excluded_objects(self) -> list[ObjectProperties]:
        """The objects this template is not instantiated for (see
        `templates.is_excluded`)."""
        parameter = self.legacy_parameter
        return parameter.excluded_objects if parameter else []


def parse_conversion_template(input_template: TextIO) -> ConversionTemplate:
    tree = safe_load(input_template)
    return ConversionTemplate.model_validate(tree["template"])


def parse_library(path: str) -> LibrarySchema:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            return parse_yaml_library(file)
    except Exception as e:
        raise ValueError(f"Failed to parse library file {path}: {e}") from e
