from pathlib import Path

import pytest
from gems_craft.model.parsing import LibrarySchema, parse_yaml_library
from gems_craft.model.taxonomy import load_taxonomy
from gems_craft.model.validation import check_library_against_taxonomy

TAXONOMIES_DIR = Path("src/antares_gems_converter/taxonomies")
MODEL_LIBRARIES_DIR = Path("src/antares_gems_converter/libs")

MODEL_LIBRARY_FILES = sorted(MODEL_LIBRARIES_DIR.rglob("*.yml"))
TAXONOMY_FILES = sorted(TAXONOMIES_DIR.glob("*.yml"))


def parse_library_file(library_file: Path) -> LibrarySchema:
    with open(library_file, encoding="utf-8") as f:
        return parse_yaml_library(f)


@pytest.mark.parametrize(
    "library_file", MODEL_LIBRARY_FILES, ids=[p.stem for p in MODEL_LIBRARY_FILES]
)
def test_library_taxonomy_exists(library_file):
    taxonomy_id = parse_library_file(library_file).taxonomy
    taxonomy_ids = {load_taxonomy(path).id for path in TAXONOMY_FILES}

    assert taxonomy_id in taxonomy_ids, f"Taxonomy '{taxonomy_id}' not found"


# The flattened taxonomy (for the view builder) shares its id with the main one,
# so each library is checked against every taxonomy file with its declared id.
@pytest.mark.parametrize(
    "taxonomy_file", TAXONOMY_FILES, ids=[p.stem for p in TAXONOMY_FILES]
)
@pytest.mark.parametrize(
    "library_file", MODEL_LIBRARY_FILES, ids=[p.stem for p in MODEL_LIBRARY_FILES]
)
def test_models_implement_taxonomy_contract(library_file, taxonomy_file):
    library = parse_library_file(library_file)
    taxonomy = load_taxonomy(taxonomy_file)
    if taxonomy.id != library.taxonomy:
        pytest.skip(f"{library_file.name} uses taxonomy '{library.taxonomy}'")

    check_library_against_taxonomy(library, taxonomy)
