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

SRC_DIR = Path("src/antares_gems_converter")
CATALOG_FILES = sorted((SRC_DIR / "catalogs").glob("*.yml"))
LIBRARY_FILES = sorted((SRC_DIR / "libs").rglob("*.yml"))
VIEW_CONFIG_FILES = sorted((SRC_DIR / "views_config").glob("*.yml"))
# The view builder doesn't resolve parent categories, so it is given the flattened taxonomy.
VIEWS_BUILDER_TAXONOMY_FILE = (
    SRC_DIR / "taxonomies" / "antares_legacy_taxonomy_flattened.yml"
)

# Fields left "to be filled in" in the view configs, filled case by case for each study.
AGGREGATION_PATTERNS = [
    {"id": "annual_all_scenario", "time-granularity": "year", "scenario": True},
    {"id": "hourly_per_scenario", "time-granularity": "hour", "scenario": False},
]


def write_filled_view_config(view_file: Path, output_dir: Path) -> Path:
    """Copy a view config with its aggregations-patterns filled."""
    view = yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"]
    view["aggregations-patterns"] = AGGREGATION_PATTERNS

    filled_view_file = output_dir / view_file.name
    filled_view_file.write_text(
        yaml.safe_dump({"view": view}, sort_keys=False), encoding="utf-8"
    )
    return filled_view_file


def test_view_configs_are_compatible_with_views_builder(tmp_path):
    view_configs = load_view_configs(
        [
            write_filled_view_config(view_file, tmp_path)
            for view_file in VIEW_CONFIG_FILES
        ]
    )
    taxonomy = load_taxonomy(VIEWS_BUILDER_TAXONOMY_FILE)
    libraries = [create_lib_from_yml(lib) for lib in load_yml_libs(LIBRARY_FILES)]
    catalogs = load_catalogs(CATALOG_FILES)

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


@pytest.mark.parametrize("view_file", VIEW_CONFIG_FILES, ids=lambda f: f.stem)
def test_view_config_lists_all_metrics_of_its_catalogs(view_file):
    metric_ids = [
        metric["id"]
        for metric in yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"][
            "metrics"
        ]
    ]
    catalog_ids = {metric_id.split(".")[0] for metric_id in metric_ids}

    catalog_metric_ids = []
    for catalog_file in CATALOG_FILES:
        catalog = yaml.safe_load(catalog_file.read_text(encoding="utf-8"))["catalog"]
        if catalog["id"] in catalog_ids:
            catalog_metric_ids += [
                f"{catalog['id']}.{metric['id']}"
                for metric in catalog["metrics-definition"]
            ]

    assert metric_ids == catalog_metric_ids
