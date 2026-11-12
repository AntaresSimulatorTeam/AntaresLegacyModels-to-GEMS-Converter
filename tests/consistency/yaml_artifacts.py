"""Discovery of the YAML artifacts checked by the consistency tests.

Every YAML file under ``src/antares_gems_converter`` is classified by its root
key, not by its location, so a library, catalog, taxonomy or view config added
anywhere is picked up by the tests. ``test_yaml_artifacts.py`` then checks that
each file is recognised and stored in the directory of its kind.
"""

from functools import cache
from pathlib import Path
from typing import Any

import yaml

SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "antares_gems_converter"

LIBRARY = "library"
CATALOG = "catalog"
TAXONOMY = "taxonomy"
VIEW_CONFIG = "view"
CONVERSION_TEMPLATE = "template"

# Directory each kind of artifact must be stored in (subdirectories allowed).
KIND_DIRS = {
    LIBRARY: SRC_DIR / "libs",
    CATALOG: SRC_DIR / "catalogs",
    TAXONOMY: SRC_DIR / "taxonomies",
    VIEW_CONFIG: SRC_DIR / "views_config",
    CONVERSION_TEMPLATE: SRC_DIR / "input_converter" / "data" / "model_configuration",
}

# File name suffix of the flattened copy of a taxonomy (no parent categories),
# given to the view builder until it supports parent categories.
FLATTENED_SUFFIX = "_flattened"

YAML_FILES = sorted(
    path for path in SRC_DIR.rglob("*") if path.suffix in {".yml", ".yaml"}
)


@cache
def load_yaml(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def kind_of(path: Path) -> str | None:
    """The artifact kind of a YAML file: its single root key, if known."""
    content = load_yaml(path)
    if isinstance(content, dict) and len(content) == 1:
        (root_key,) = content
        if root_key in KIND_DIRS:
            return root_key
    return None


def files_of_kind(kind: str) -> list[Path]:
    return [path for path in YAML_FILES if kind_of(path) == kind]


LIBRARY_FILES = files_of_kind(LIBRARY)
CATALOG_FILES = files_of_kind(CATALOG)
TAXONOMY_FILES = files_of_kind(TAXONOMY)
VIEW_CONFIG_FILES = files_of_kind(VIEW_CONFIG)


def is_flattened(taxonomy_file: Path) -> bool:
    return taxonomy_file.stem.endswith(FLATTENED_SUFFIX)


MAIN_TAXONOMY_FILES = [path for path in TAXONOMY_FILES if not is_flattened(path)]
FLATTENED_TAXONOMY_FILES = [path for path in TAXONOMY_FILES if is_flattened(path)]


def artifact_id(path: Path) -> str:
    return load_yaml(path)[kind_of(path)]["id"]


def declared_taxonomy(path: Path) -> str:
    """The taxonomy id a library, catalog or view config declares."""
    return load_yaml(path)[kind_of(path)]["taxonomy"]


def main_taxonomy_file(taxonomy_id: str) -> Path:
    matches = [path for path in MAIN_TAXONOMY_FILES if artifact_id(path) == taxonomy_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one taxonomy file with id '{taxonomy_id}', found {matches}"
        )
    return matches[0]


def flattened_taxonomy_file(taxonomy_file: Path) -> Path | None:
    """The flattened copy of a main taxonomy file, if any."""
    path = taxonomy_file.with_name(
        f"{taxonomy_file.stem}{FLATTENED_SUFFIX}{taxonomy_file.suffix}"
    )
    return path if path in FLATTENED_TAXONOMY_FILES else None


def views_builder_taxonomy_file(taxonomy_id: str) -> Path:
    """The taxonomy given to the view builder: the flattened copy if any."""
    main_file = main_taxonomy_file(taxonomy_id)
    return flattened_taxonomy_file(main_file) or main_file


def ids(paths: list[Path]) -> list[str]:
    """pytest ids for files: the path relative to the source directory."""
    return [str(path.relative_to(SRC_DIR)) for path in paths]
