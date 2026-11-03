from pathlib import Path

import pytest
from gems_craft.model.parsing import LibrarySchema, parse_yaml_library
from gems_craft.model.taxonomy import load_taxonomy
from gems_craft.model.validation import check_library_against_taxonomy

from tests.antares_historic.yaml_artifacts import (
    LIBRARY_FILES,
    TAXONOMY_FILES,
    ids,
)

# That the taxonomy a library declares exists is checked, for every kind of
# artifact, by test_declared_taxonomy_exists (test_yaml_artifacts.py).


def parse_library_file(library_file: Path) -> LibrarySchema:
    with open(library_file, encoding="utf-8") as f:
        return parse_yaml_library(f)


# A flattened taxonomy (for the view builder) shares its id with its main one,
# so each library is checked against every taxonomy file with its declared id.
@pytest.mark.parametrize("taxonomy_file", TAXONOMY_FILES, ids=ids(TAXONOMY_FILES))
@pytest.mark.parametrize("library_file", LIBRARY_FILES, ids=ids(LIBRARY_FILES))
def test_models_implement_taxonomy_contract(library_file, taxonomy_file):
    library = parse_library_file(library_file)
    taxonomy = load_taxonomy(taxonomy_file)
    if taxonomy.id != library.taxonomy:
        pytest.skip(f"{library_file.name} uses taxonomy '{library.taxonomy}'")

    check_library_against_taxonomy(library, taxonomy)
