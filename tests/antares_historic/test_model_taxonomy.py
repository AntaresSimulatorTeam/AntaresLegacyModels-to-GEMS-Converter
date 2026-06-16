from pathlib import Path

import pytest
import yaml
from gems.model.parsing import LibrarySchema
from gems.model.taxonomy import Taxonomy, check_library_against_taxonomy, load_taxonomy

TAXONOMIES_DIR = Path("src/antares_gems_converter/taxonomies")
MODEL_LIBRARIES_DIR = Path("src/antares_gems_converter/libs")


def load_library(path: Path) -> tuple[LibrarySchema, str]:
    """Load a library YAML file, returning the parsed LibrarySchema and its taxonomy id.

    Fields not yet supported by LibrarySchema (taxonomy, version, area-connection)
    are stripped before validation.
    """
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    lib_dict = raw["library"]
    taxonomy_id = lib_dict.pop("taxonomy", None)
    lib_dict.pop("version", None)

    for port_type in lib_dict.get("port-types", []):
        port_type.pop("area-connection", None)

    return LibrarySchema.model_validate(lib_dict), taxonomy_id


def load_taxonomy_by_id(taxonomy_id: str) -> Taxonomy:
    for path in TAXONOMIES_DIR.glob("*.yml"):
        taxonomy = load_taxonomy(path)
        if taxonomy.id == taxonomy_id:
            return taxonomy

    raise ValueError(f"Taxonomy '{taxonomy_id}' not found")


MODEL_LIBRARY_FILES = list(MODEL_LIBRARIES_DIR.rglob("*.yml"))


@pytest.mark.parametrize("library_file", MODEL_LIBRARY_FILES)
def test_models_implement_taxonomy_contract(library_file):
    library, taxonomy_id = load_library(library_file)

    taxonomy = load_taxonomy_by_id(taxonomy_id)

    check_library_against_taxonomy(library, taxonomy)
