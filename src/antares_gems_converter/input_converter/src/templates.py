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
"""Load the conversion templates and list what they are instantiated for: the
arguments of their parameters (`template_arguments`), study objects or ad-hoc
values."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence

from antares.craft.model.area import Area
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import (
    MODEL_NAME_TO_FILE_NAME,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
)
from antares_gems_converter.input_converter.src.parsing import (
    CLUSTER_KINDS,
    ConversionTemplate,
    ObjectKind,
    ObjectProperties,
    TemplateParameter,
    parse_conversion_template,
)

MODEL_TEMPLATE_FOLDER = Path(__file__).parents[1] / "data" / "model_configuration"


@dataclass(frozen=True)
class StudyObject:
    """A link, area or cluster of the study that a template is instantiated for."""

    kind: ObjectKind
    area: Optional[str] = None
    cluster: Optional[str] = None
    link: Optional[str] = None
    area_from: Optional[str] = None
    area_to: Optional[str] = None

    def substitutions(self) -> dict[str, str]:
        """{placeholder: value} to instantiate a template for this object."""
        if self.kind == ObjectKind.LINK:
            assert self.link and self.area_from and self.area_to
            # The attributes of parsing.LINK_ATTRIBUTES, a following "." dropped:
            # "${link}.area_from_id._node" -> "<area from>_node"
            return {
                "${link}.area_from_id.": self.area_from,
                "${link}.area_to_id.": self.area_to,
                "${link}.area_from_id": self.area_from,
                "${link}.area_to_id": self.area_to,
                "${link}": self.link,
            }
        assert self.area
        substitutions = {"${area}": self.area}
        if self.kind in CLUSTER_KINDS:
            assert self.cluster
            # The cluster placeholder is named after the cluster type
            substitutions[f"${{{self.kind}}}"] = self.cluster
        return substitutions


def load_conversion_template(model: str) -> ConversionTemplate:
    model_conversion_template_file = (
        MODEL_TEMPLATE_FOLDER / MODEL_NAME_TO_FILE_NAME[model]
    )
    if not model_conversion_template_file.exists():
        raise FileNotFoundError(
            f"The model configuration file for {model} has not been found at the location {model_conversion_template_file}"
        )
    with model_conversion_template_file.open() as template:
        return parse_conversion_template(template)


def load_conversion_templates(models: list[str]) -> dict[str, ConversionTemplate]:
    return {model: load_conversion_template(model) for model in models}


def is_excluded(obj: StudyObject, exclusions: Sequence[ObjectProperties]) -> bool:
    """Whether the study object `obj` matches one of `exclusions`.

    An excluded area also excludes its clusters and the links it is an end
    of. In a cluster exclusion, ${area} stands for the area of the cluster.
    """
    for excluded in exclusions:
        if excluded.type == ObjectKind.AREA:
            matches = excluded.area in (obj.area, obj.area_from, obj.area_to)
        elif excluded.type == ObjectKind.LINK:
            matches = excluded.link == obj.link
        else:
            area = obj.area or ""
            matches = (
                excluded.type == obj.kind
                and (excluded.area or "").replace("${area}", area) == obj.area
                and (excluded.cluster or "").replace("${area}", area) == obj.cluster
            )
        if matches:
            return True
    return False


def _check_listed(
    template_name: str,
    parameter: TemplateParameter,
    listed: Iterable[str],
    found: Iterable[str],
    what: str,
) -> None:
    if missing := sorted(set(listed) - set(found)):
        raise ValueError(
            f"Template '{template_name}', parameter '{parameter.name}': "
            f"{what} {missing} not found in the study"
        )


def study_objects(
    template_name: str,
    parameter: TemplateParameter,
    study: Study,
    areas: Mapping[str, Area],
    logger: logging.Logger,
) -> list[StudyObject]:
    """Return, in study order, the links, areas or clusters that `parameter`,
    a parameter of the template named `template_name` with no ad-hoc flag,
    yields: those listed in its values and cluster-values, if any, except those
    it excludes and the disabled renewable clusters.

    A listed link or area that is not in the study, or a listed cluster that is
    in none of the listed areas, raises a ValueError.
    """
    kind = parameter.kind
    assert kind is not None
    exclusions = parameter.excluded_objects
    objects = []
    if kind == ObjectKind.LINK:
        links = study.get_links()
        if parameter.values is not None:
            _check_listed(template_name, parameter, parameter.values, links, "links")
        for link in links.values():
            link_object = StudyObject(
                kind,
                link=link.id,
                area_from=link.area_from_id,
                area_to=link.area_to_id,
            )
            if parameter.values is not None and link.id not in parameter.values:
                continue
            if not is_excluded(link_object, exclusions):
                objects.append(link_object)
        return objects

    if parameter.values is not None:
        _check_listed(template_name, parameter, parameter.values, areas, "areas")
    # The cluster-values entries found in at least one area
    found_clusters: set[str] = set()
    for area in areas.values():
        if parameter.values is not None and area.id not in parameter.values:
            continue
        area_object = StudyObject(ObjectKind.AREA, area=area.id)
        # Its clusters are excluded too (see is_excluded): skip them, unless
        # they must be listed to check the cluster-values
        excluded = is_excluded(area_object, exclusions)
        if kind == ObjectKind.AREA:
            if not excluded:
                objects.append(area_object)
            continue
        if excluded and parameter.cluster_values is None:
            continue

        clusters = getattr(area, TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[kind])()
        if parameter.cluster_values is not None:
            # In the entries, ${area} stands for the area of the cluster
            listed: dict[str, list[str]] = {}
            for entry in parameter.cluster_values:
                listed.setdefault(entry.replace("${area}", area.id), []).append(entry)
            found_clusters.update(e for c in clusters for e in listed.get(c, []))
            clusters = {c: clusters[c] for c in clusters if c in listed}
        if excluded:
            continue
        for cluster_id, cluster in clusters.items():
            cluster_object = StudyObject(kind, area=area.id, cluster=cluster_id)
            if is_excluded(cluster_object, exclusions):
                continue
            if kind == ObjectKind.RENEWABLE and not cluster.properties.enabled:
                logger.warning(
                    f"Renewable cluster '{cluster_id}' in area "
                    f"'{area.id}' is disabled; skipping conversion."
                )
                continue
            objects.append(cluster_object)
    if parameter.cluster_values is not None:
        _check_listed(
            template_name,
            parameter,
            parameter.cluster_values,
            found_clusters,
            f"{kind} clusters",
        )
    return objects


@dataclass(frozen=True)
class TemplateArguments:
    """The values of the parameters of a template for one of its instances:
    the substitutions to instantiate it, and the study object of its parameter
    with no ad-hoc flag, if any."""

    substitutions: dict[str, str]
    study_object: Optional[StudyObject] = None


def template_arguments(
    conversion_template: ConversionTemplate,
    study: Study,
    areas: Mapping[str, Area],
    logger: logging.Logger,
) -> list[TemplateArguments]:
    """Return the arguments `conversion_template` is instantiated for: every
    combination of the values of its parameters, nested in declaration order
    (outer first), each parameter's lists resolved with the values of the
    outer parameters."""

    def walk(
        parameters: Sequence[TemplateParameter],
        substitutions: dict[str, str],
        study_object: Optional[StudyObject],
    ) -> Iterator[TemplateArguments]:
        if not parameters:
            yield TemplateArguments(substitutions, study_object)
            return
        parameter = parameters[0].resolve(substitutions)
        if parameter.ad_hoc:
            assert parameter.values
            for value in parameter.values:
                yield from walk(
                    parameters[1:],
                    {**substitutions, f"${{{parameter.name}}}": value},
                    study_object,
                )
            return
        for obj in study_objects(
            conversion_template.name, parameter, study, areas, logger
        ):
            yield from walk(
                parameters[1:], {**substitutions, **obj.substitutions()}, obj
            )

    return list(walk(conversion_template.template_parameters, {}, None))
