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
import logging
import shutil
from pathlib import Path

import pytest
from antares.craft.model.renewable import RenewableClusterProperties
from antares.craft.model.study import Study

from antares_gems_converter.input_converter.src.config import MODEL_NAME_TO_FILE_NAME
from antares_gems_converter.input_converter.src.converter import AntaresStudyConverter
from antares_gems_converter.input_converter.src.deletion import delete_legacy_objects
from antares_gems_converter.input_converter.src.logger import Logger
from antares_gems_converter.input_converter.src.parsing import (
    ConversionTemplate,
    ObjectKind,
    ObjectProperties,
)
from antares_gems_converter.input_converter.src.scenario_builder import read_year_ts
from antares_gems_converter.input_converter.src.templates import (
    StudyObject,
    is_excluded,
    load_conversion_template,
    study_objects,
)
from tests.input_converter.conftest import (
    LIB_PATHS,
    exclude_battery_artefacts,
    with_exclude,
)

BATTERY_STUDY = Path(__file__).parent / "resources" / "mini_test_batterie_BP23"

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
AT_FR = StudyObject(ObjectKind.LINK, link="at / fr", area_from="at", area_to="fr")

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


def _template(template_parameters: list[dict], **body: list) -> ConversionTemplate:
    return ConversionTemplate.model_validate(
        {
            "name": "test",
            "model": "lib.model",
            "template-parameters": template_parameters,
            "components": [{"id": "c", "parameters": []}],
            **body,
        }
    )


@pytest.mark.parametrize(
    "template_parameters, error",
    [
        ([], "exactly one template parameter, got 0"),
        ([{"name": "area"}, {"name": "link"}], "exactly one template parameter, got 2"),
        ([{"name": "zone"}], "named 'area' or 'link'"),
        ([{"name": "area", "cluster-type": "hydro"}], "cluster-type must be one of"),
        ([{"name": "link", "cluster-type": "thermal"}], "cluster-type must be one of"),
    ],
)
def test_invalid_template_parameters(template_parameters: list[dict], error: str):
    with pytest.raises(ValueError, match=error):
        _template(template_parameters)


@pytest.mark.parametrize(
    "object_properties",
    [
        {"type": "areas", "area": "z_batteries"},
        {"type": "link", "area": "z_batteries"},
        {"type": "thermal", "cluster": "inj"},
    ],
)
def test_invalid_exclusion(object_properties: dict):
    exclude = [{"id": "e", "object-properties": object_properties}]
    with pytest.raises(ValueError, match="invalid exclusion"):
        _template([{"name": "area", "exclude": exclude}])


def test_unknown_link_attribute():
    component2 = "${link}.area_frm_id._node"
    connection = {
        "component1": "c",
        "port1": "p",
        "component2": component2,
        "port2": "p",
    }
    with pytest.raises(ValueError, match="unknown link attribute"):
        _template([{"name": "link"}], connections=[connection])


def test_study_object_substitutions():
    assert FR_GAZ.substitutions() == {"${area}": "fr", "${thermal}": "gaz"}
    assert FR.substitutions() == {"${area}": "fr"}
    assert AT_FR.substitutions()["${link}"] == "at / fr"


def test_read_year_ts_by_kind(fr_wind: Study):
    sb = fr_wind.get_scenario_builder()
    sb.wind.get_area("fr").set_new_scenario([2])
    sb.thermal.get_cluster("fr", "gaz").set_new_scenario([3])
    sb.renewable.get_cluster("fr", "generation").set_new_scenario([4])
    sb.link.get_link("at / fr").set_new_scenario([5])

    assert read_year_ts(sb, "wind", FR) == {0: 2}
    # Templates without a legacy SB attribute
    assert read_year_ts(sb, "area", FR) == {}
    assert read_year_ts(sb, "misc_gen", FR) == {}
    assert read_year_ts(sb, "battery", FR) == {}
    assert read_year_ts(sb, "thermal", FR_GAZ) == {0: 3}
    generation = StudyObject(ObjectKind.RENEWABLE, area="fr", cluster="generation")
    assert read_year_ts(sb, "renewable", generation) == {0: 4}
    storage = StudyObject(ObjectKind.ST_STORAGE, area="fr", cluster="storage_1")
    assert read_year_ts(sb, "st_storage", storage) == {}
    assert read_year_ts(sb, "link", AT_FR) == {0: 5}
    assert read_year_ts(None, "wind", FR) == {}


def test_battery_exclusions():
    exclusions = load_conversion_template("battery").excluded_objects
    # An excluded area also excludes its clusters and the links it is an end of
    assert is_excluded(Z_BATTERIES, exclusions)
    assert is_excluded(Z_CLUSTER, exclusions)
    assert is_excluded(FR_Z_LINK, exclusions)
    assert not is_excluded(FR, exclusions)
    assert not is_excluded(FR_INJ, exclusions)


def test_cluster_exclusion_with_area_placeholder():
    exclusions = [
        ObjectProperties(
            type="thermal", area="${area}", cluster="${area}_batteries_inj"
        )
    ]
    # ${area} stands for the area of the cluster, of the excluded type only
    assert is_excluded(FR_INJ, exclusions)
    assert not is_excluded(
        StudyObject(ObjectKind.THERMAL, area="it", cluster="fr_batteries_inj"),
        exclusions,
    )
    assert not is_excluded(
        StudyObject(ObjectKind.RENEWABLE, area="fr", cluster="fr_batteries_inj"),
        exclusions,
    )
    assert not is_excluded(FR_GAZ, exclusions)


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


def _converter(tmp_path: Path, mode: str, models: list[str]) -> AntaresStudyConverter:
    input_path = tmp_path / "input" / BATTERY_STUDY.name
    shutil.copytree(BATTERY_STUDY, input_path)
    return AntaresStudyConverter(
        study_input=input_path,
        logger=Logger(__name__, str(input_path)),
        mode=mode,
        output_folder=tmp_path / "output",
        lib_paths=LIB_PATHS,
        models_to_convert=models,
    )


@pytest.mark.parametrize("mode", ["full", "hybrid"])
@pytest.mark.parametrize("excluding", [False, True])
def test_exclusions_are_local(tmp_path: Path, mode: str, excluding: bool):
    converter = _converter(tmp_path, mode, ["battery", "thermal", "link"])
    templates = converter.load_and_check_templates()
    if excluding:
        templates = exclude_battery_artefacts(templates)
    system, _ = converter.system_and_scenario_groups(templates)
    ids = {component.id for component in system.components}
    assert {"fr_battery", "fr_thermal_fr_nuclear_epr"} <= ids
    # What battery excludes only applies to battery: unless the thermal and
    # link templates exclude them too, the battery artefacts are converted
    assert ids & BATTERY_ARTEFACTS == (set() if excluding else BATTERY_ARTEFACTS)


def test_excluded_objects_stay_legacy_in_hybrid_mode(tmp_path: Path):
    converter = _converter(tmp_path, "hybrid", ["thermal", "link"])
    system, _ = converter.system_and_scenario_groups(
        exclude_battery_artefacts(converter.load_and_check_templates())
    )
    assert not {component.id for component in system.components} & BATTERY_ARTEFACTS
    # Not converted, so not deleted: the binding constraints referencing them
    # do not make the deletion fail
    delete_legacy_objects(
        converter.study, converter.areas, converter.legacy_objects, converter.logger
    )
    assert "fr_batteries_inj" in converter.study.get_areas()["fr"].get_thermals()
    assert "fr / z_batteries" in converter.study.get_links()


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
    instance = load_conversion_template("link").instantiate(AT_FR.substitutions())
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


def test_instantiate_keeps_non_finite_constants():
    parameter = {"id": "max", "value": {"constant": float("inf")}}
    component = {"id": "${area}_c", "parameters": [parameter]}
    template = _template([{"name": "area"}], components=[component])
    instance = template.instantiate(FR.substitutions())
    assert instance.components[0].parameters[0].value.constant == float("inf")


def test_instantiate_rejects_unresolved_placeholder():
    template = load_conversion_template("thermal")
    with pytest.raises(ValueError, match="unresolved placeholder"):
        template.instantiate(FR.substitutions())


def test_study_objects_skip_excluded_and_disabled(
    local_study_w_links: Study, caplog: pytest.LogCaptureFixture
):
    study = local_study_w_links
    areas = study.get_areas()
    areas["fr"].create_renewable_cluster("pv", RenewableClusterProperties())
    areas["fr"].create_renewable_cluster(
        "pv_off", RenewableClusterProperties(enabled=False)
    )
    areas["it"].create_renewable_cluster(
        "it_off", RenewableClusterProperties(enabled=False)
    )
    exclude_it = [{"id": "it", "object-properties": {"type": "area", "area": "it"}}]
    logger = logging.getLogger(__name__)

    renewable = with_exclude(load_conversion_template("renewable"), exclude_it)
    with caplog.at_level(logging.WARNING):
        objects = list(study_objects(renewable, study, areas, logger))
    assert objects == [StudyObject(ObjectKind.RENEWABLE, area="fr", cluster="pv")]
    # A disabled cluster is skipped with a warning, unless its area is excluded
    assert "'pv_off'" in caplog.text and "'it_off'" not in caplog.text

    # A link is excluded with either of its areas
    link = with_exclude(load_conversion_template("link"), exclude_it)
    assert list(study_objects(link, study, areas, logger)) == [AT_FR]
