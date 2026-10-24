# Copyright (c) 2026, RTE (https://www.rte-france.com)
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
"""Template loop: what a template is instantiated for, and how."""
import io

import pytest
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import MODEL_NAME_TO_FILE_NAME
from antares_gems_converter.input_converter.src.parsing import (
    ObjectKind,
    parse_conversion_template,
)
from antares_gems_converter.input_converter.src.scenario_builder import read_year_ts
from antares_gems_converter.input_converter.src.templates import (
    StudyObject,
    load_conversion_template,
)

EXPECTED_FOR_EACH = {
    "area": ObjectKind.AREA,
    "battery": ObjectKind.AREA,
    "link": ObjectKind.LINK,
    "load": ObjectKind.AREA,
    "solar": ObjectKind.AREA,
    "short-term-storage": ObjectKind.ST_STORAGE,
    "thermal": ObjectKind.THERMAL,
    "renewable": ObjectKind.RENEWABLE,
    "wind": ObjectKind.AREA,
    "misc_gen": ObjectKind.AREA,
    "hydro": ObjectKind.AREA,
    "ror": ObjectKind.AREA,
}


def test_for_each_covers_every_template():
    assert set(EXPECTED_FOR_EACH) == set(MODEL_NAME_TO_FILE_NAME)


@pytest.mark.parametrize("model, kind", EXPECTED_FOR_EACH.items())
def test_for_each(model: str, kind: ObjectKind):
    assert load_conversion_template(model).for_each == kind


def _parse(template_parameters: str, connections: str = "") -> None:
    parse_conversion_template(
        io.StringIO(
            "template:\n"
            "  name: test\n"
            "  model: lib.model\n"
            f"{template_parameters}"
            "  components:\n"
            "    - id: c\n"
            "      parameters: []\n"
            f"{connections}"
        )
    )


@pytest.mark.parametrize(
    "template_parameters, error",
    [
        ("", "exactly one template parameter, got 0"),
        (
            "  template-parameters:\n    - name: area\n    - name: link\n",
            "exactly one template parameter, got 2",
        ),
        ("  template-parameters:\n    - name: zone\n", "named 'area' or 'link'"),
        (
            "  template-parameters:\n    - name: area\n      cluster-type: hydro\n",
            "cluster-type must be one of",
        ),
        (
            "  template-parameters:\n    - name: link\n      cluster-type: thermal\n",
            "cluster-type must be one of",
        ),
    ],
)
def test_invalid_template_parameters(template_parameters: str, error: str):
    with pytest.raises(ValueError, match=error):
        _parse(template_parameters)


def test_unknown_link_attribute():
    with pytest.raises(ValueError, match="unknown link attribute"):
        _parse(
            "  template-parameters:\n    - name: link\n",
            "  connections:\n"
            "    - component1: c\n"
            "      port1: p\n"
            "      component2: ${link}.area_frm_id._node\n"
            "      port2: p\n",
        )


def test_study_object_substitutions():
    thermal = StudyObject(ObjectKind.THERMAL, area="fr", cluster="gaz")
    assert thermal.substitutions() == {"${area}": "fr", "${thermal}": "gaz"}
    assert thermal.areas == ("fr",)
    area = StudyObject(ObjectKind.AREA, area="fr")
    assert area.substitutions() == {"${area}": "fr"}
    link = StudyObject(ObjectKind.LINK, link="at / fr", area_from="at", area_to="fr")
    assert link.substitutions()["${link}"] == "at / fr"
    assert link.areas == ("at", "fr")


def test_read_year_ts_by_kind(fr_wind: Study):
    sb = fr_wind.get_scenario_builder()
    sb.wind.get_area("fr").set_new_scenario([2])
    sb.thermal.get_cluster("fr", "gaz").set_new_scenario([3])
    sb.renewable.get_cluster("fr", "generation").set_new_scenario([4])
    sb.link.get_link("at / fr").set_new_scenario([5])

    fr = StudyObject(ObjectKind.AREA, area="fr")
    assert read_year_ts(sb, "wind", fr) == {0: 2}
    # Templates without a legacy SB attribute
    assert read_year_ts(sb, "area", fr) == {}
    assert read_year_ts(sb, "misc_gen", fr) == {}
    assert read_year_ts(sb, "battery", fr) == {}
    gaz = StudyObject(ObjectKind.THERMAL, area="fr", cluster="gaz")
    assert read_year_ts(sb, "thermal", gaz) == {0: 3}
    generation = StudyObject(ObjectKind.RENEWABLE, area="fr", cluster="generation")
    assert read_year_ts(sb, "renewable", generation) == {0: 4}
    storage = StudyObject(ObjectKind.ST_STORAGE, area="fr", cluster="storage_1")
    assert read_year_ts(sb, "st_storage", storage) == {}
    link = StudyObject(ObjectKind.LINK, link="at / fr", area_from="at", area_to="fr")
    assert read_year_ts(sb, "link", link) == {0: 5}
    assert read_year_ts(None, "wind", fr) == {}
