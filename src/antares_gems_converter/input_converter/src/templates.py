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
from typing import Iterator, Mapping, Optional

from antares.craft.model.area import Area
from antares.craft.model.link import Link
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import (
    LINK_TYPES,
    MODEL_NAME_TO_FILE_NAME,
    TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD,
)
from antares_gems_converter.input_converter.src.parsing import (
    ConversionTemplate,
    VirtualObjectsRepository,
    parse_conversion_template,
)

MODEL_TEMPLATE_FOLDER = Path(__file__).parents[1] / "data" / "model_configuration"


@dataclass(frozen=True)
class StudyObject:
    """The link, area or cluster that a template was resolved for."""

    link: Optional[str] = None
    area: Optional[str] = None
    cluster: Optional[str] = None


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


def build_virtual_objects_repo(
    model_conversion_templates: dict[str, ConversionTemplate],
) -> VirtualObjectsRepository:
    virtual_objects = VirtualObjectsRepository()
    for conversion_template in model_conversion_templates.values():
        virtual_objects.add(conversion_template.get_excluded_objects_ids())
    return virtual_objects


def is_virtual_link(link: Link, virtual_objects: VirtualObjectsRepository) -> bool:
    return (
        link.id in virtual_objects.links
        or link.area_from_id in virtual_objects.areas
        or link.area_to_id in virtual_objects.areas
    )


def resolved_templates(
    conversion_template: ConversionTemplate,
    study: Study,
    areas: Mapping[str, Area],
    virtual_objects: VirtualObjectsRepository,
    logger: logging.Logger,
) -> Iterator[tuple[ConversionTemplate, StudyObject]]:
    """Yield the template resolved for each link, area or cluster it applies
    to, with that object."""
    model_name = conversion_template.name
    template_parameter = conversion_template.template_parameters[0]
    model_area_pattern = f"${{{template_parameter.name}}}"

    if model_name in LINK_TYPES:
        for link in study.get_links().values():
            if is_virtual_link(link, virtual_objects):
                continue
            yield conversion_template.resolve_template(
                model_area_pattern, link.id
            ), StudyObject(link=link.id)
        return

    cluster_type = template_parameter.cluster_type
    for area in areas.values():
        if area.id in virtual_objects.areas:
            continue
        area_resolved_template = conversion_template.resolve_template(
            model_area_pattern, area.id
        )
        if not cluster_type:
            yield area_resolved_template, StudyObject(area=area.id)
            continue

        virtual_clusters = virtual_objects.resolve_template("${area}", area.id).thermals
        clusters = getattr(area, TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD[cluster_type])()
        for cluster_id, cluster in clusters.items():
            if cluster_id in virtual_clusters:
                continue
            if cluster_type == "renewable" and not cluster.properties.enabled:
                logger.warning(
                    f"Renewable cluster '{cluster_id}' in area "
                    f"'{area.id}' is disabled; skipping conversion."
                )
                continue
            yield area_resolved_template.resolve_template(
                f"${{{cluster_type}}}", cluster_id
            ), StudyObject(area=area.id, cluster=cluster_id)
