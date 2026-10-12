from pathlib import Path

import pytest
import yaml
from gems_views_builder.input.catalog import load_catalogs
from gems_views_builder.input.taxonomy import load_taxonomy
from gems_views_builder.input.view_config import load_view_config
from gems_views_builder.validation.catalogs_taxonomy_validator import (
    CatalogsTaxonomyValidator,
)
from gems_views_builder.validation.catalogs_view_config_validator import (
    ViewConfigCatalogsValidator,
)
from gems_views_builder.validation.view_config_metrics_against_catalogs_validator import (
    ViewConfigMetricsAgainstCatalogsValidator,
)
from gems_views_builder.validation.view_config_taxonomy import (
    ViewConfigTaxonomyValidator,
)

SRC_DIR = Path("src/antares_gems_converter")
CATALOGS_DIR = SRC_DIR / "catalogs"
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
    """Copy a view config with its aggregations-patterns and all the metrics of its catalogs filled."""
    view = yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"]
    view["aggregations-patterns"] = AGGREGATION_PATTERNS
    view["metrics"] = []
    for catalog_ref in view["catalogs"]:
        catalog_id = catalog_ref["id"]
        catalog_file = CATALOGS_DIR / f"{catalog_id}.yml"
        catalog = yaml.safe_load(catalog_file.read_text(encoding="utf-8"))["catalog"]
        view["metrics"] += [
            {"id": f"{catalog_id}.{metric['id']}"}
            for metric in catalog["metrics-definition"]
        ]
    view["scope"].pop("extra-locations", None)

    filled_view_file = output_dir / view_file.name
    filled_view_file.write_text(
        yaml.safe_dump({"view": view}, sort_keys=False), encoding="utf-8"
    )
    return filled_view_file


@pytest.mark.parametrize("view_file", VIEW_CONFIG_FILES, ids=lambda f: f.stem)
def test_view_config_is_compatible_with_views_builder(view_file, tmp_path):
    filled_view_file = write_filled_view_config(view_file, tmp_path)

    view_config = load_view_config(filled_view_file)
    taxonomy = load_taxonomy(VIEWS_BUILDER_TAXONOMY_FILE)
    catalogs = load_catalogs(CATALOGS_DIR, view_config.catalog_ids)

    # Same checks, in the same order, as the view builder InputConsistencyValidator.
    ViewConfigTaxonomyValidator(taxonomy, view_config).validate()
    CatalogsTaxonomyValidator(catalogs, taxonomy).validate()
    ViewConfigCatalogsValidator(catalogs, view_config).validate()
    ViewConfigMetricsAgainstCatalogsValidator(catalogs, view_config).validate()

    view_config.fetch_metrics(catalogs)
    assert len(view_config.get_metrics()) == len(view_config.metric_ids) > 0
