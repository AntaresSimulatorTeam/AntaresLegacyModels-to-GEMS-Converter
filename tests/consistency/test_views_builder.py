"""GEMS-ViewsBuilder accepts the libraries, catalogs, taxonomies and view configs of the
package.
"""

from pathlib import Path

import pytest
import yaml
from gems_craft.model.taxonomy import load_taxonomy
from gems_views_builder.input.catalog import load_catalogs
from gems_views_builder.input.library import create_lib_from_yml, load_yml_libs
from gems_views_builder.input.view_config import (
    load_view_configs,
    populate_view_configs_with_metrics,
)
from gems_views_builder.validation.catalogs import validate_catalogs
from gems_views_builder.validation.catalogs_taxonomy_validator import (
    CatalogsTaxonomyValidator,
)
from gems_views_builder.validation.catalogs_view_config_validator import (
    ViewConfigCatalogsValidator,
)
from gems_views_builder.validation.input_consistency_validator import (
    associated_catalogs,
)
from gems_views_builder.validation.validate_library_taxonomy import (
    validate_libraries_against_taxonomy,
)
from gems_views_builder.validation.view_config import validate_view_configs
from gems_views_builder.validation.view_config_metrics_against_catalogs_validator import (
    ViewConfigMetricsAgainstCatalogsValidator,
)
from gems_views_builder.validation.view_config_taxonomy import (
    ViewConfigTaxonomyValidator,
)

from tests.consistency.yaml_artifacts import (
    CATALOG_FILES,
    LIBRARY_FILES,
    MAIN_TAXONOMY_FILES,
    VIEW_CONFIG_FILES,
    artifact_id,
    declared_taxonomy,
    ids,
    views_builder_taxonomy_file,
)

TAXONOMY_IDS = sorted(artifact_id(path) for path in MAIN_TAXONOMY_FILES)


def files_declaring(taxonomy_id: str, files: list[Path]) -> list[Path]:
    return [path for path in files if declared_taxonomy(path) == taxonomy_id]


# Fields left "to be filled in" in the view configs, filled case by case for each study.
# A view config that sets its own aggregations-patterns (e.g. the digest) keeps them.
AGGREGATION_PATTERNS = [
    {"id": "annual_all_scenario", "time-granularity": "year", "scenario": True},
    {"id": "hourly_per_scenario", "time-granularity": "hour", "scenario": False},
]


def write_filled_view_config(view_file: Path, output_dir: Path) -> Path:
    """Copy a view config with its aggregations-patterns filled, if left empty."""
    view = yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"]
    if not view.get("aggregations-patterns"):
        view["aggregations-patterns"] = AGGREGATION_PATTERNS

    filled_view_file = output_dir / view_file.name
    filled_view_file.write_text(
        yaml.safe_dump({"view": view}, sort_keys=False), encoding="utf-8"
    )
    return filled_view_file


# Each taxonomy is validated with the libraries, catalogs and view configs that
# declare it. The view builder doesn't resolve parent categories, so it is given
# the flattened copy of the taxonomy, if any.
# This is the only check of the libraries against their taxonomy: against the
# flattened copy, it requires all inherited items (test_taxonomies). When the
# flattened copy is removed, the libraries are checked against the main taxonomy,
# without the inherited items (GemsPy `check_library_against_taxonomy` does not
# resolve parents): a check resolving them must then be restored, unless the view
# builder does it by then.
@pytest.mark.parametrize("taxonomy_id", TAXONOMY_IDS)
def test_view_configs_are_compatible_with_views_builder(taxonomy_id, tmp_path):
    """For each taxonomy, GEMS-ViewsBuilder validates the libraries, catalogs and view
    configs that declare it, against the flattened copy of the taxonomy if any, with the
    checks of its `validate_input_data` (simulation tables aside), and every view config
    gets at least one metric. The view configs without aggregation patterns are given
    some, as for a study.

    Limits: a model without a taxonomy category is not checked against the taxonomy
    (e.g. the models of `andromede_v1_models.yml`).
    """
    view_files = files_declaring(taxonomy_id, VIEW_CONFIG_FILES)
    catalog_files = files_declaring(taxonomy_id, CATALOG_FILES)
    library_files = files_declaring(taxonomy_id, LIBRARY_FILES)

    view_configs = load_view_configs(
        [write_filled_view_config(view_file, tmp_path) for view_file in view_files]
    )
    taxonomy = load_taxonomy(views_builder_taxonomy_file(taxonomy_id))
    libraries = [create_lib_from_yml(lib) for lib in load_yml_libs(library_files)]
    catalogs = load_catalogs(catalog_files)

    # Same checks, in the same order, as the view builder validate_input_data
    # (simulation tables aside).
    validate_view_configs(view_configs)
    validate_catalogs(catalogs)
    validate_libraries_against_taxonomy(libraries, taxonomy)
    CatalogsTaxonomyValidator(catalogs, taxonomy).validate()
    catalogs_by_ids = {catalog.id: catalog for catalog in catalogs}
    for view_config in view_configs:
        view_config_catalogs = associated_catalogs(view_config, catalogs_by_ids)
        ViewConfigTaxonomyValidator(taxonomy, view_config).validate()
        ViewConfigCatalogsValidator(view_config_catalogs, view_config).validate()
        ViewConfigMetricsAgainstCatalogsValidator(
            view_config_catalogs, view_config
        ).validate()

    populate_view_configs_with_metrics(view_configs, catalogs)
    for view_config in view_configs:
        assert len(view_config.get_metrics()) == len(view_config.full_metric_ids) > 0


def metric_ids_of(view_file: Path) -> list[str]:
    return [
        metric["id"]
        for metric in yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"][
            "metrics"
        ]
    ]


def catalog_metric_ids(catalog_ids: set[str]) -> list[str]:
    """The metrics of these catalogs, as <catalog_id>.<metric_id>, in catalog order."""
    metric_ids = []
    for catalog_file in CATALOG_FILES:
        catalog = yaml.safe_load(catalog_file.read_text(encoding="utf-8"))["catalog"]
        if catalog["id"] in catalog_ids:
            metric_ids += [
                f"{catalog['id']}.{metric['id']}"
                for metric in catalog["metrics-definition"]
            ]
    return metric_ids


# The view configs that mirror the legacy digest (mc-all/grid/digest.txt): every area
# metric (those that differ from the digest are commented in the file), and the only
# link metric of the digest.
DIGEST_VIEW_CONFIG_METRICS = {
    "view_digest_area": catalog_metric_ids({"antares_legacy_area_catalog"}),
    "view_digest_link": ["antares_legacy_link_catalog.FLOW_LIN"],
}

DIGEST_VIEW_CONFIG_FILES = [
    path
    for path in VIEW_CONFIG_FILES
    if artifact_id(path) in DIGEST_VIEW_CONFIG_METRICS
]
TEMPLATE_VIEW_CONFIG_FILES = [
    path for path in VIEW_CONFIG_FILES if path not in DIGEST_VIEW_CONFIG_FILES
]


@pytest.mark.parametrize(
    "view_file", TEMPLATE_VIEW_CONFIG_FILES, ids=ids(TEMPLATE_VIEW_CONFIG_FILES)
)
def test_view_config_lists_all_metrics_of_its_catalogs(view_file):
    """Each view config, digest aside, lists every metric of the catalogs it uses, in
    the catalog order, and nothing else, to be trimmed for each study.
    """
    metric_ids = metric_ids_of(view_file)
    catalog_ids = {metric_id.split(".")[0] for metric_id in metric_ids}

    assert metric_ids == catalog_metric_ids(catalog_ids)


def test_digest_view_configs_exist():
    assert sorted(artifact_id(path) for path in DIGEST_VIEW_CONFIG_FILES) == sorted(
        DIGEST_VIEW_CONFIG_METRICS
    )


@pytest.mark.parametrize(
    "view_file", DIGEST_VIEW_CONFIG_FILES, ids=ids(DIGEST_VIEW_CONFIG_FILES)
)
def test_digest_view_config(view_file):
    """A digest view config lists its metrics in the catalog order, with a single
    pattern giving the expected value over the whole period, and
    no location filter (the digest ignores the geographic trimming).
    """
    view = yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"]

    assert metric_ids_of(view_file) == DIGEST_VIEW_CONFIG_METRICS[view["id"]]
    assert view["aggregations-patterns"] == [
        {"id": "year_all", "time-granularity": "year", "scenario": True}
    ]
    assert "location-filter" not in view["scope"]
