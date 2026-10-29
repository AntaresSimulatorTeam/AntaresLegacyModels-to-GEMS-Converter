from pathlib import Path

import pytest
from gems_craft.model.parsing import parse_yaml_library
from gems_craft.model.taxonomy import check_library_against_taxonomy, load_taxonomy

TAXONOMIES_DIR = Path("src/antares_gems_converter/taxonomies")
MODEL_LIBRARIES_DIR = Path("src/antares_gems_converter/libs")


def parse_library_file(library_file: Path):
    with open(library_file, encoding="utf-8") as f:
        return parse_yaml_library(f)


MODEL_LIBRARY_FILES = sorted(MODEL_LIBRARIES_DIR.rglob("*.yml"))
TAXONOMY_FILES = sorted(TAXONOMIES_DIR.glob("*.yml"))

LIBRARY_TAXONOMY_IDS = {
    path: parse_library_file(path).taxonomy for path in MODEL_LIBRARY_FILES
}
TAXONOMY_IDS = {path: load_taxonomy(path).id for path in TAXONOMY_FILES}

# The flattened taxonomy (for the view builder) shares its id with the main one,
# so each library is checked against every taxonomy file with its declared id.
LIBRARY_TAXONOMY_PAIRS = [
    pytest.param(
        library_file, taxonomy_file, id=f"{library_file.stem}-{taxonomy_file.stem}"
    )
    for library_file in MODEL_LIBRARY_FILES
    for taxonomy_file in TAXONOMY_FILES
    if LIBRARY_TAXONOMY_IDS[library_file] == TAXONOMY_IDS[taxonomy_file]
]


@pytest.mark.parametrize(
    "library_file", MODEL_LIBRARY_FILES, ids=[p.stem for p in MODEL_LIBRARY_FILES]
)
def test_library_taxonomy_exists(library_file):
    assert (
        LIBRARY_TAXONOMY_IDS[library_file] in TAXONOMY_IDS.values()
    ), f"Taxonomy '{LIBRARY_TAXONOMY_IDS[library_file]}' of {library_file} not found"


@pytest.mark.parametrize("library_file, taxonomy_file", LIBRARY_TAXONOMY_PAIRS)
def test_models_implement_taxonomy_contract(library_file, taxonomy_file):
    check_library_against_taxonomy(
        parse_library_file(library_file), load_taxonomy(taxonomy_file)
    )
