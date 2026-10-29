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
"""Load the conversion templates and list the study objects they are
instantiated for."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

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


def study_objects(
    conversion_template: ConversionTemplate,
    study: Study,
    areas: Mapping[str, Area],
    logger: logging.Logger,
) -> Iterator[StudyObject]:
    """Yield, in study order, the links, areas or clusters that
    `conversion_template` is instantiated for, except those it excludes and the
    disabled clusters."""
    kind = conversion_template.for_each
    exclusions = conversion_template.excluded_objects
    if kind == ObjectKind.LINK:
        for link in study.get_links().values():
            link_object = StudyObject(
                kind,
                link=link.id,
                area_from=link.area_from_id,
                area_to=link.area_to_id,
            )
            if not is_excluded(link_object, exclusions):
                yield link_object
        return

    for area in areas.values():
        area_object = StudyObject(ObjectKind.AREA, area=area.id)
        # Its clusters are excluded too (see is_excluded): skip without listing them
        if is_excluded(area_object, exclusions):
            continue
        if kind == ObjectKind.AREA:
            yield area_object
            continue

        clusters = getattr(area, TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[kind])()
        for cluster_id, cluster in clusters.items():
            cluster_object = StudyObject(kind, area=area.id, cluster=cluster_id)
            if is_excluded(cluster_object, exclusions):
                continue
            # A disabled cluster contributes nothing in Antares: no component, and
            # in hybrid mode it is not deleted, so it stays, disabled, in the study
            if not cluster.properties.enabled:
                logger.warning(
                    f"{kind} cluster '{cluster_id}' in area "
                    f"'{area.id}' is disabled; skipping conversion."
                )
                continue
            yield cluster_object
