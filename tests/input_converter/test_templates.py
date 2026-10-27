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
import json
import logging
import shutil
from pathlib import Path

import pytest
from antares.craft.model.renewable import RenewableClusterProperties
from antares.craft.model.st_storage import STStorageProperties
from antares.craft.model.study import Study
from antares.craft.model.thermal import ThermalClusterProperties
from gems_craft_hybrid.study.parsing import HybridSystemSchema

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
    Binding,
    StudyObject,
    bindings,
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


def _exclusion(type_: str, **fields: str) -> dict:
    return {"id": "e", "object-properties": {"type": type_, **fields}}


@pytest.mark.parametrize(
    "template_parameters, error",
    [
        ([], "at least one template parameter"),
        ([{"name": "area"}, {"name": "link"}], "at most one template parameter"),
        ([{"name": "area"}, {"name": "area"}], "names must be unique"),
        ([{"name": "zone"}], "named 'area' or 'link', unless it is ad-hoc"),
        ([{"name": "area", "cluster-type": "hydro"}], "cluster-type must be one of"),
        ([{"name": "link", "cluster-type": "thermal"}], "cluster-type must be one of"),
        ([{"name": "area", "cluster-values": ["c"]}], "requires a cluster-type"),
        ([{"name": "area", "values": []}], "non-empty list without duplicates"),
        ([{"name": "area", "values": ["fr", "fr"]}], "without duplicates"),
        ([{"name": "site", "ad-hoc": True}], "ad-hoc parameter requires values"),
        (
            [
                {
                    "name": "site",
                    "ad-hoc": True,
                    "values": ["a"],
                    "cluster-type": "thermal",
                }
            ],
            "not allowed on an ad-hoc parameter",
        ),
        (
            [{"name": "site", "ad-hoc": True, "values": ["a"], "exclude": []}],
            "not allowed on an ad-hoc parameter",
        ),
        ([{"name": "area", "ad-hoc": True, "values": ["a"]}], "name of an ad-hoc"),
        ([{"name": "thermal", "ad-hoc": True, "values": ["a"]}], "name of an ad-hoc"),
        ([{"name": "my-site", "ad-hoc": True, "values": ["a"]}], "name of an ad-hoc"),
        # A list may only use the placeholders of the outer parameters
        (
            [{"name": "area", "values": ["${area}"]}],
            "unknown placeholder '\\$\\{area\\}'",
        ),
        (
            [
                {"name": "area", "values": ["${site}"]},
                {"name": "site", "ad-hoc": True, "values": ["a"]},
            ],
            "unknown placeholder '\\$\\{site\\}'",
        ),
        (
            [{"name": "area", "cluster-type": "thermal", "cluster-values": ["${x}"]}],
            "unknown placeholder '\\$\\{x\\}' in cluster-values",
        ),
        (
            [{"name": "site", "ad-hoc": True, "values": ["a"], "cluster-values": []}],
            "not allowed on an ad-hoc parameter",
        ),
        (
            [{"name": "area", "cluster-type": "thermal", "cluster-values": ["c", "c"]}],
            "cluster-values must be a non-empty list without duplicates",
        ),
        (
            [
                {"name": "link"},
                {"name": "x", "ad-hoc": True, "values": ["${link}.area_frm_id"]},
            ],
            "unknown link attribute",
        ),
        (
            [{"name": "area", "exclude": [_exclusion("area", area="${site}")]}],
            "unknown placeholder '\\$\\{site\\}' in exclude",
        ),
        # Values are strings: numbers must be quoted in YAML
        ([{"name": "year", "ad-hoc": True, "values": [2030]}], "valid string"),
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
        {"type": "area", "area": "fr", "cluster": "fr_batteries_inj"},
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


def _study_objects(template: ConversionTemplate, study: Study) -> list[StudyObject]:
    assert template.legacy_parameter is not None
    return study_objects(
        template.name,
        template.legacy_parameter,
        study,
        study.get_areas(),
        logging.getLogger(__name__),
    )


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
        objects = _study_objects(renewable, study)
    assert objects == [StudyObject(ObjectKind.RENEWABLE, area="fr", cluster="pv")]
    # A disabled cluster is skipped with a warning, unless its area is excluded
    assert "'pv_off'" in caplog.text and "'it_off'" not in caplog.text

    # A link is excluded with either of its areas: "it" ends "at / it" and
    # "fr / it", "at" starts "at / fr" and "at / it"
    link = with_exclude(load_conversion_template("link"), exclude_it)
    assert _study_objects(link, study) == [AT_FR]
    exclude_at = [{"id": "at", "object-properties": {"type": "area", "area": "at"}}]
    link = with_exclude(load_conversion_template("link"), exclude_at)
    assert [obj.link for obj in _study_objects(link, study)] == ["fr / it"]


def test_ad_hoc_parameter_forbids_legacy_objects_to_delete():
    deleted = {"id": "d", "object-properties": {"type": "area", "area": "${site}"}}
    with pytest.raises(ValueError, match="legacy-objects-to-delete is not allowed"):
        _template(
            [{"name": "site", "ad-hoc": True, "values": ["a"]}],
            **{"legacy-objects-to-delete": [deleted]},
        )


def test_valid_template_parameters():
    template = _template(
        [
            {"name": "site", "ad-hoc": True, "values": ["fr", "it"]},
            {"name": "area", "values": ["${site}"], "cluster-type": "thermal"},
            {"name": "unit", "ad-hoc": True, "values": ["${site}_${thermal}_1"]},
        ]
    )
    assert template.for_each == ObjectKind.THERMAL
    assert [p.name for p in template.ad_hoc_parameters] == ["site", "unit"]
    ad_hoc_only = _template([{"name": "site", "ad-hoc": True, "values": ["a"]}])
    assert ad_hoc_only.for_each is None
    assert ad_hoc_only.excluded_objects == []


def _with_parameter(model: str, **keys: object) -> ConversionTemplate:
    """The reference template of `model`, its parameter given `keys`."""
    data = load_conversion_template(model).model_dump()
    data["template_parameters"][0].update(keys)
    return ConversionTemplate.model_validate(data)


@pytest.fixture
def study_w_clusters(local_study_with_st_storage: Study) -> Study:
    """fr: thermal gaz, coal, it_gaz; renewable generation; storage storage_1
    it: thermal it_gaz, coal; renewable it_pv; storage it_storage"""
    study = local_study_with_st_storage
    areas = study.get_areas()
    areas["fr"].create_thermal_cluster("coal", ThermalClusterProperties())
    areas["fr"].create_thermal_cluster("it_gaz", ThermalClusterProperties())
    areas["it"].create_thermal_cluster("it_gaz", ThermalClusterProperties())
    areas["it"].create_thermal_cluster("coal", ThermalClusterProperties())
    areas["it"].create_renewable_cluster("it_pv", RenewableClusterProperties())
    areas["it"].create_st_storage("it_storage", STStorageProperties())
    return study


def test_values_restrict_areas_and_links(study_w_clusters: Study):
    area = _with_parameter("load", values=["it", "fr"])
    # In study order, whatever the order of the list
    assert [obj.area for obj in _study_objects(area, study_w_clusters)] == [
        "fr",
        "it",
    ]
    link = _with_parameter("link", values=["at / fr"])
    assert _study_objects(link, study_w_clusters) == [AT_FR]


@pytest.mark.parametrize(
    "model, cluster_values, expected",
    [
        ("thermal", ["coal"], [("fr", "coal"), ("it", "coal")]),
        ("thermal", ["${area}_gaz"], [("it", "it_gaz")]),
        (
            "thermal",
            ["gaz", "it_gaz"],
            [("fr", "gaz"), ("fr", "it_gaz"), ("it", "it_gaz")],
        ),
        ("renewable", ["it_pv"], [("it", "it_pv")]),
        ("short-term-storage", ["storage_1"], [("fr", "storage_1")]),
    ],
)
def test_cluster_values_restrict_clusters(
    study_w_clusters: Study,
    model: str,
    cluster_values: list[str],
    expected: list[tuple[str, str]],
):
    template = _with_parameter(model, cluster_values=cluster_values)
    objects = _study_objects(template, study_w_clusters)
    assert [(obj.area, obj.cluster) for obj in objects] == expected


def test_cluster_values_resolving_to_the_same_cluster(study_w_clusters: Study):
    # In it, both entries are it_gaz: each is found
    template = _with_parameter(
        "thermal", values=["it"], cluster_values=["it_gaz", "${area}_gaz"]
    )
    assert _study_objects(template, study_w_clusters) == [
        StudyObject(ObjectKind.THERMAL, area="it", cluster="it_gaz")
    ]


def test_exclude_applies_after_lists(study_w_clusters: Study):
    exclude = [
        {"id": "it", "object-properties": {"type": "area", "area": "it"}},
        {
            "id": "coal",
            "object-properties": {"type": "thermal", "area": "fr", "cluster": "coal"},
        },
    ]
    template = _with_parameter(
        "thermal", values=["fr", "it"], cluster_values=["coal", "gaz"]
    )
    template = with_exclude(template, exclude)
    objects = _study_objects(template, study_w_clusters)
    assert objects == [FR_GAZ]
    # A listed cluster only in an excluded area exists: it is not an error
    template = with_exclude(
        _with_parameter("renewable", cluster_values=["it_pv"]), exclude
    )
    assert _study_objects(template, study_w_clusters) == []


@pytest.mark.parametrize(
    "model, keys, error",
    [
        ("load", {"values": ["fr", "de"]}, "areas \\['de'\\] not found"),
        ("link", {"values": ["fr / at"]}, "links \\['fr / at'\\] not found"),
        (
            "thermal",
            {"cluster-values": ["gaz", "nuclear"]},
            "thermal clusters \\['nuclear'\\] not found",
        ),
        # gaz is in fr, which is not listed
        (
            "thermal",
            {"values": ["it"], "cluster-values": ["gaz"]},
            "thermal clusters \\['gaz'\\] not found",
        ),
        (
            "short-term-storage",
            {"cluster-values": ["${area}_battery"]},
            "st_storage clusters \\['\\$\\{area\\}_battery'\\] not found",
        ),
    ],
)
def test_unknown_listed_ids(
    study_w_clusters: Study, model: str, keys: dict, error: str
):
    data = load_conversion_template(model).model_dump(by_alias=True)
    data["template-parameters"][0].update(keys)
    template = ConversionTemplate.model_validate(data)
    with pytest.raises(ValueError, match=f"Template '{template.name}', parameter "):
        _study_objects(template, study_w_clusters)
    with pytest.raises(ValueError, match=error):
        _study_objects(template, study_w_clusters)


def _bindings(template: ConversionTemplate, study: Study) -> list[Binding]:
    return bindings(template, study, study.get_areas(), logging.getLogger(__name__))


def test_bindings_nest_parameters_in_declaration_order(study_w_clusters: Study):
    area_site = _template(
        [
            {"name": "area", "values": ["fr", "it"]},
            {"name": "site", "ad-hoc": True, "values": ["n", "${area}_s"]},
        ]
    )
    assert [b.substitutions for b in _bindings(area_site, study_w_clusters)] == [
        {"${area}": "fr", "${site}": "n"},
        {"${area}": "fr", "${site}": "fr_s"},
        {"${area}": "it", "${site}": "n"},
        {"${area}": "it", "${site}": "it_s"},
    ]
    site_area = _template(
        [
            {"name": "site", "ad-hoc": True, "values": ["fr", "it"]},
            {"name": "area", "values": ["${site}"], "cluster-type": "thermal"},
        ]
    )
    result = _bindings(site_area, study_w_clusters)
    assert [b.substitutions["${thermal}"] for b in result] == [
        "gaz",
        "coal",
        "it_gaz",
        "it_gaz",
        "coal",
    ]
    assert result[0] == Binding(
        {"${site}": "fr", "${area}": "fr", "${thermal}": "gaz"}, FR_GAZ
    )
    ad_hoc_only = _template([{"name": "site", "ad-hoc": True, "values": ["a", "b"]}])
    assert _bindings(ad_hoc_only, study_w_clusters) == [
        Binding({"${site}": "a"}),
        Binding({"${site}": "b"}),
    ]


def test_legacy_bindings_are_study_objects(study_w_clusters: Study):
    template = load_conversion_template("thermal")
    result = _bindings(template, study_w_clusters)
    assert [b.study_object for b in result] == _study_objects(
        template, study_w_clusters
    )
    assert all(b.substitutions == b.study_object.substitutions() for b in result)


ELECTROLYSER = {
    "name": "electrolyser",
    "model": "lib.electrolyser",
    "template-parameters": [
        {"name": "area", "values": ["fr", "it"]},
        {"name": "site", "ad-hoc": True, "values": ["north", "south"]},
    ],
    "components": [
        {
            "id": "${area}_electrolyser_${site}",
            "scenario-group": "electrolyser_${area}_${site}",
            "parameters": [
                {"id": "efficiency", "value": {"constant": 0.7}},
            ],
            "properties": [{"id": "site", "value": {"constant": "${site}"}}],
        }
    ],
    "connections": [
        {
            "component1": "${area}_electrolyser_${site}",
            "port1": "power_port",
            "component2": "${area}_node",
            "port2": "balance_port",
        }
    ],
    "area-connections": [
        {
            "component": "${area}_electrolyser_${site}",
            "port": "power_port",
            "area": "${area}",
        }
    ],
}
ELECTROLYSER_IDS = [
    "fr_electrolyser_north",
    "fr_electrolyser_south",
    "it_electrolyser_north",
    "it_electrolyser_south",
]


def _study_converter(study: Study, mode: str) -> AntaresStudyConverter:
    return AntaresStudyConverter(
        study_input=study,
        logger=Logger(__name__, study.path),
        mode=mode,
        lib_paths=LIB_PATHS,
        models_to_convert=[],
        output_folder=study.path.parent / "converter_output",
    )


@pytest.mark.parametrize("mode", ["full", "hybrid"])
@pytest.mark.parametrize("site_outside", [False, True])
def test_convert_ad_hoc_template(
    study_w_clusters: Study, mode: str, site_outside: bool
):
    converter = _study_converter(study_w_clusters, mode)
    template = ConversionTemplate.model_validate(ELECTROLYSER)
    if site_outside:
        template.template_parameters.reverse()
    system, _ = converter.system_and_scenario_groups({"electrolyser": template})
    ids = [c.id for c in system.components]
    # The outer parameter varies slowest
    order = [0, 2, 1, 3] if site_outside else [0, 1, 2, 3]
    assert ids == [ELECTROLYSER_IDS[i] for i in order]
    component = system.components[0]
    assert component.parameters[0].value == 0.7
    assert component.properties[0].value == "north"
    if mode == "full":
        assert [c.component1 for c in system.connections] == ids
        assert [c.component2 for c in system.connections] == [
            f"{id.split('_')[0]}_node" for id in ids
        ]
    else:
        assert isinstance(system, HybridSystemSchema)
        assert [(c.component, c.area) for c in system.area_connections] == [
            (id, id.split("_")[0]) for id in ids
        ]
        assert converter.legacy_objects == []


def test_convert_ad_hoc_only_template(study_w_clusters: Study):
    converter = _study_converter(study_w_clusters, "hybrid")
    data = {
        **ELECTROLYSER,
        "template-parameters": [
            {"name": "site", "ad-hoc": True, "values": ["north", "south"]}
        ],
    }
    data = json.loads(json.dumps(data).replace("${area}", "fr"))
    system, _ = converter.system_and_scenario_groups(
        {"electrolyser": ConversionTemplate.model_validate(data)}
    )
    assert isinstance(system, HybridSystemSchema)
    assert [(c.component, c.area) for c in system.area_connections] == [
        ("fr_electrolyser_north", "fr"),
        ("fr_electrolyser_south", "fr"),
    ]


def test_ad_hoc_parameter_drops_scenario_groups(study_w_clusters: Study):
    sb = study_w_clusters.get_scenario_builder()
    sb.thermal.get_cluster("fr", "gaz").set_new_scenario([2])
    study_w_clusters.set_scenario_builder(sb)
    converter = _study_converter(study_w_clusters, "full")
    thermal = _with_parameter("thermal", values=["fr"], cluster_values=["gaz"])
    _, scenario_groups = converter.system_and_scenario_groups({"thermal": thermal})
    assert scenario_groups == {"thermal_fr_gaz_group": {0: 2}}

    data = thermal.model_dump()
    data["template_parameters"].append(
        {"name": "unit", "ad_hoc": True, "values": ["1"]}
    )
    data["legacy_objects_to_delete"] = []
    data["components"][0]["id"] += "_${unit}"
    ad_hoc = ConversionTemplate.model_validate(data)
    system, scenario_groups = converter.system_and_scenario_groups({"thermal": ad_hoc})
    assert [c.id for c in system.components] == ["fr_thermal_gaz_1"]
    assert system.components[0].scenario_group is None and scenario_groups == {}


@pytest.mark.parametrize("mode", ["full", "hybrid"])
def test_convert_restricted_clusters(study_w_clusters: Study, mode: str):
    converter = _study_converter(study_w_clusters, mode)
    thermal = _with_parameter("thermal", values=["fr"], cluster_values=["coal"])
    templates = {"thermal": thermal, "link": load_conversion_template("link")}
    system, _ = converter.system_and_scenario_groups(templates)
    # A list only restricts its own template
    assert [c.id for c in system.components] == [
        "fr_thermal_coal",
        "at_fr_link",
        "at_it_link",
        "fr_it_link",
    ]
    if mode == "hybrid":
        # The clusters not listed are not deleted: they stay legacy
        delete_legacy_objects(
            converter.study, converter.areas, converter.legacy_objects, converter.logger
        )
        areas = converter.study.get_areas()
        assert set(areas["fr"].get_thermals()) == {"gaz", "it_gaz"}
        assert set(areas["it"].get_thermals()) == {"it_gaz", "coal"}


@pytest.mark.parametrize(
    "model, keys, converted, kept",
    [
        ("link", {"values": ["at / fr"]}, ["at_fr_link"], {"at / it", "fr / it"}),
        (
            "renewable",
            {"cluster_values": ["it_pv"]},
            ["it_renewable_it_pv"],
            {("fr", "generation")},
        ),
        (
            "short-term-storage",
            {"values": ["it"]},
            ["it_short_term_storage_it_storage"],
            {("fr", "storage_1")},
        ),
    ],
)
def test_objects_not_listed_stay_legacy(
    study_w_clusters: Study, model: str, keys: dict, converted: list, kept: set
):
    converter = _study_converter(study_w_clusters, "hybrid")
    template = _with_parameter(model, **keys)
    system, _ = converter.system_and_scenario_groups({model: template})
    assert [c.id for c in system.components] == converted
    delete_legacy_objects(
        converter.study, converter.areas, converter.legacy_objects, converter.logger
    )
    areas = converter.study.get_areas()
    if model == "link":
        remaining = set(converter.study.get_links())
    elif model == "renewable":
        remaining = {(a, c) for a in areas for c in areas[a].get_renewables()}
    else:
        remaining = {(a, c) for a in areas for c in areas[a].get_st_storages()}
    assert remaining == kept


def test_unknown_listed_id_fails_before_converting(
    study_w_clusters: Study, monkeypatch: pytest.MonkeyPatch
):
    converter = _study_converter(study_w_clusters, "full")
    converted: list[str] = []
    monkeypatch.setattr(
        converter,
        "apply_conversion_template",
        lambda template, *args: converted.append(template.name),
    )
    templates = {
        "electrolyser": ConversionTemplate.model_validate(ELECTROLYSER),
        "load": _with_parameter("load", values=["de"]),
    }
    with pytest.raises(ValueError, match="areas \\['de'\\] not found"):
        converter.system_and_scenario_groups(templates)
    # Not even the templates before the faulty one
    assert converted == []


def test_duplicate_component_ids(study_w_clusters: Study):
    converter = _study_converter(study_w_clusters, "full")
    template = ConversionTemplate.model_validate(ELECTROLYSER)
    other = template.model_copy(update={"name": "other"})
    with pytest.raises(
        ValueError, match="Duplicate component ids.*fr_electrolyser_north"
    ):
        converter.system_and_scenario_groups({"electrolyser": template, "other": other})
