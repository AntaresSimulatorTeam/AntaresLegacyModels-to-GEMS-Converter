from pathlib import Path

import pytest
from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.taxonomy import (
    Taxonomy,
    check_library_against_taxonomy,
    load_taxonomy,
)

TAXONOMIES_DIR = Path("src/antares_gems_converter/taxonomies")
MODEL_LIBRARIES_DIR = Path("src/antares_gems_converter/libs")


def load_taxonomy_by_id(taxonomy_id: str) -> Taxonomy:
    for path in TAXONOMIES_DIR.glob("*.yml"):
        # The flattened taxonomy (for the view builder) shares its id with the main one.
        if path.stem.endswith("_flattened"):
            continue
        taxonomy = load_taxonomy(path)
        if taxonomy.id == taxonomy_id:
            return taxonomy

    raise ValueError(f"Taxonomy '{taxonomy_id}' not found")


MODEL_LIBRARY_FILES = list(MODEL_LIBRARIES_DIR.rglob("*.yml"))


@pytest.mark.parametrize("library_file", MODEL_LIBRARY_FILES)
def test_models_implement_taxonomy_contract(library_file):
    with open(library_file, encoding="utf-8") as f:
        library = parse_yaml_library(f)

    taxonomy = load_taxonomy_by_id(library.taxonomy)

    check_library_against_taxonomy(library, taxonomy)
