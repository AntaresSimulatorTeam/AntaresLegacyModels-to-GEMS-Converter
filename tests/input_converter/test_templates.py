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
import shutil
from pathlib import Path

import pytest
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import MODEL_NAME_TO_FILE_NAME
from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter
from antares_gems_converter.input_converter.src.logger import Logger
from antares_gems_converter.input_converter.src.parsing import (
    ObjectKind,
    ObjectProperties,
    parse_conversion_template,
)
from antares_gems_converter.input_converter.src.scenario_builder import read_year_ts
from antares_gems_converter.input_converter.src.templates import (
    StudyObject,
    is_excluded,
    load_conversion_template,
)

REPO_ROOT = Path(__file__).parents[2]
LIB_PATHS = [
    str(REPO_ROOT / "src/antares_gems_converter/libs" / lib)
    for lib in [
        "antares_historic/antares_legacy_models.yml",
        "reference_models/andromede_v1_models.yml",
    ]
]
BATTERY_STUDY = Path(__file__).parent / "resources" / "mini_test_batterie_BP23"

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
    area = StudyObject(ObjectKind.AREA, area="fr")
    assert area.substitutions() == {"${area}": "fr"}
    link = StudyObject(ObjectKind.LINK, link="at / fr", area_from="at", area_to="fr")
    assert link.substitutions()["${link}"] == "at / fr"


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


FR = StudyObject(ObjectKind.AREA, area="fr")
FR_GAZ = StudyObject(ObjectKind.THERMAL, area="fr", cluster="gaz")
Z_BATTERIES = StudyObject(ObjectKind.AREA, area="z_batteries")
FR_Z_LINK = StudyObject(
    ObjectKind.LINK, link="fr / z_batteries", area_from="fr", area_to="z_batteries"
)
FR_INJ = StudyObject(ObjectKind.THERMAL, area="fr", cluster="fr_batteries_inj")
Z_CLUSTER = StudyObject(
    ObjectKind.THERMAL, area="z_batteries", cluster="z_batteries_batteries_fr_1"
)


def test_battery_exclusions():
    exclusions = load_conversion_template("battery").excluded_objects
    # An excluded area also excludes its clusters and the links it is an end of
    assert is_excluded(Z_BATTERIES, exclusions)
    assert is_excluded(Z_CLUSTER, exclusions)
    assert is_excluded(FR_Z_LINK, exclusions)
    # ${area} stands for the area of the cluster
    assert is_excluded(FR_INJ, exclusions)
    assert not is_excluded(
        StudyObject(ObjectKind.THERMAL, area="it", cluster="fr_batteries_inj"),
        exclusions,
    )
    assert not is_excluded(FR, exclusions)
    assert not is_excluded(
        StudyObject(ObjectKind.THERMAL, area="fr", cluster="gaz"), exclusions
    )


def test_cluster_exclusion_checks_type():
    exclusions = [
        ObjectProperties(type="thermal", area="${area}", cluster="${area}_inj")
    ]
    assert is_excluded(
        StudyObject(ObjectKind.THERMAL, area="fr", cluster="fr_inj"), exclusions
    )
    assert not is_excluded(
        StudyObject(ObjectKind.RENEWABLE, area="fr", cluster="fr_inj"), exclusions
    )


def test_link_exclusion():
    exclusions = [ObjectProperties(type="link", link="fr / z_batteries")]
    assert is_excluded(FR_Z_LINK, exclusions)
    assert not is_excluded(FR, exclusions)
    assert not is_excluded(Z_BATTERIES, exclusions)


BATTERY_ARTEFACTS = {
    "fr_thermal_fr_batteries_inj",
    "fr_z_batteries_link",
    "z_batteries_thermal_z_batteries_batteries_fr_1",
    "z_batteries_thermal_z_batteries_batteries_fr_2",
}


def _component_ids(tmp_path: Path, mode: str, models: list[str]) -> set[str]:
    input_path = tmp_path / "input" / BATTERY_STUDY.name
    shutil.copytree(BATTERY_STUDY, input_path)
    converter = AntaresStudyConverter(
        study_input=input_path,
        logger=Logger(__name__, str(input_path)),
        mode=mode,
        output_folder=tmp_path / "output",
        lib_paths=LIB_PATHS,
        models_to_convert=models,
    )
    system, _ = converter.system_and_scenario_groups(
        *converter.load_and_check_templates()
    )
    return {component.id for component in system.components}


@pytest.mark.parametrize("mode", ["full", "hybrid"])
def test_battery_exclusions_apply_to_all_templates(tmp_path: Path, mode: str):
    # Without them, the battery legacy artefacts would be converted too, and
    # the battery counted twice
    ids = _component_ids(tmp_path, mode, ["battery", "thermal", "link"])
    assert "fr_battery" in ids
    assert "fr_thermal_fr_nuclear_epr" in ids
    assert not ids & (BATTERY_ARTEFACTS | {"z_batteries_node"})


@pytest.mark.parametrize("mode", ["full", "hybrid"])
def test_battery_exclusions_only_when_battery_is_converted(tmp_path: Path, mode: str):
    ids = _component_ids(tmp_path, mode, ["thermal", "link"])
    assert BATTERY_ARTEFACTS <= ids
    assert ("z_batteries_node" in ids) == (mode == "full")


def test_instantiate_thermal():
    template = load_conversion_template("thermal")
    instance = template.instantiate(FR_GAZ.substitutions())
    component = instance.components[0]
    assert component.id == "fr_thermal_gaz"
    assert component.scenario_group == "thermal_fr_gaz_group"
    cluster = component.parameters[0].value.object_properties
    assert cluster is not None
    assert (cluster.area, cluster.cluster) == ("fr", "gaz")
    name = next(prop for prop in component.properties if prop.id == "name")
    assert name.value.constant == "gaz"
    deleted = instance.legacy_objects_to_delete[0].object_properties
    assert (deleted.type, deleted.area, deleted.cluster) == ("thermal", "fr", "gaz")
    # The template itself keeps its placeholders
    assert template.components[0].id == "${area}_thermal_${thermal}"


def test_instantiate_link_attributes():
    link = StudyObject(ObjectKind.LINK, link="at / fr", area_from="at", area_to="fr")
    instance = load_conversion_template("link").instantiate(link.substitutions())
    assert instance.components[0].id == "at / fr_link"
    assert [c.component2 for c in instance.connections] == ["at_node", "fr_node"]
    assert [c.area for c in instance.area_connections] == ["at", "fr"]
    assert instance.legacy_objects_to_delete[0].object_properties.link == "at / fr"


def test_instantiate_battery_term_ids():
    # "%" and "." in binding constraint term ids are not interpreted
    instance = load_conversion_template("battery").instantiate(FR.substitutions())
    fields = {
        param.id: param.value.object_properties.field
        for param in instance.components[0].parameters
        if param.value.object_properties is not None
    }
    assert fields["efficiency_injection"] == "fr%z_batteries"
    assert fields["efficiency_withdrawal"] == "fr.fr_batteries_inj"


def test_instantiate_rejects_unresolved_placeholder():
    template = load_conversion_template("thermal")
    with pytest.raises(ValueError, match="unresolved placeholder"):
        template.instantiate(FR.substitutions())
