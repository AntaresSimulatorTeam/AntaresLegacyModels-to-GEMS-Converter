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
"""Load the conversion templates and resolve them for each study object."""
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

    @property
    def areas(self) -> tuple[str, ...]:
        """The areas of the object: both ends of a link, else its area."""
        return tuple(
            area
            for area in (self.area, self.area_from, self.area_to)
            if area is not None
        )

    def substitutions(self) -> dict[str, str]:
        """{placeholder: value} to instantiate a template for this object,
        in the order they must be applied."""
        if self.link is not None:
            assert self.area_from is not None and self.area_to is not None
            return {
                # "${link}.area_from_id._node" -> "<area from>_node"
                "${link}.area_from_id.": self.area_from,
                "${link}.area_to_id.": self.area_to,
                "${link}.area_from_id": self.area_from,
                "${link}.area_to_id": self.area_to,
                "${link}": self.link,
            }
        assert self.area is not None
        substitutions = {"${area}": self.area}
        if self.cluster is not None:
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


def _resolves_to(pattern: Optional[str], value: Optional[str], area: str) -> bool:
    return pattern is not None and pattern.replace("${area}", area) == value


def is_excluded(
    study_object: StudyObject, exclusions: Sequence[ObjectProperties]
) -> bool:
    """Whether `study_object` matches one of `exclusions`.

    An excluded area also excludes its clusters and the links it is an end
    of. In a cluster exclusion, ${area} stands for the area of the cluster.
    """
    for excluded in exclusions:
        if excluded.type == ObjectKind.AREA:
            if excluded.area in study_object.areas:
                return True
        elif excluded.type == ObjectKind.LINK:
            if excluded.link is not None and excluded.link == study_object.link:
                return True
        elif excluded.type == study_object.kind and study_object.area is not None:
            area = study_object.area
            if _resolves_to(excluded.area, area, area) and _resolves_to(
                excluded.cluster, study_object.cluster, area
            ):
                return True
    return False


def study_objects(
    conversion_template: ConversionTemplate,
    study: Study,
    areas: Mapping[str, Area],
    exclusions: Sequence[ObjectProperties],
    logger: logging.Logger,
) -> Iterator[StudyObject]:
    """Yield, in study order, the links, areas or clusters that
    `conversion_template` is instantiated for, except the excluded ones."""
    kind = conversion_template.for_each
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
        # Skipped before listing its clusters, so no warning is logged for them
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
            if kind == ObjectKind.RENEWABLE and not cluster.properties.enabled:
                logger.warning(
                    f"Renewable cluster '{cluster_id}' in area "
                    f"'{area.id}' is disabled; skipping conversion."
                )
                continue
            yield cluster_object
