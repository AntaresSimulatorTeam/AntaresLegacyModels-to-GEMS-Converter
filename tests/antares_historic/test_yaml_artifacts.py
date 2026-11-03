"""Guards that every YAML artifact of the package is covered by the consistency tests.

The consistency tests (test_model_taxonomy, test_catalogs, test_views_builder)
run on the files found by ``yaml_artifacts``. These tests make sure no file
escapes them: an unrecognised YAML file, or one stored outside the directory
of its kind, fails here.
"""

from collections import Counter
from pathlib import Path

import pytest
from gems_craft.model.taxonomy import load_taxonomy

from tests.antares_historic.yaml_artifacts import (
    CATALOG,
    CATALOG_FILES,
    FLATTENED_SUFFIX,
    FLATTENED_TAXONOMY_FILES,
    KIND_DIRS,
    LIBRARY,
    LIBRARY_FILES,
    MAIN_TAXONOMY_FILES,
    TAXONOMY,
    TAXONOMY_FILES,
    VIEW_CONFIG,
    VIEW_CONFIG_FILES,
    YAML_FILES,
    artifact_id,
    declared_taxonomy,
    ids,
    kind_of,
)

FILES_BY_KIND = {
    LIBRARY: LIBRARY_FILES,
    CATALOG: CATALOG_FILES,
    TAXONOMY: TAXONOMY_FILES,
    VIEW_CONFIG: VIEW_CONFIG_FILES,
}

# Kinds the consistency tests and tooling only look for as *.yml
# (the update-library-checksums workflow only checksums libs/**/*.yml).
YML_ONLY_KINDS = set(FILES_BY_KIND)

FILES_DECLARING_A_TAXONOMY = LIBRARY_FILES + CATALOG_FILES + VIEW_CONFIG_FILES


@pytest.mark.parametrize("yaml_file", YAML_FILES, ids=ids(YAML_FILES))
def test_yaml_file_is_a_known_artifact(yaml_file: Path) -> None:
    assert kind_of(yaml_file) is not None, (
        f"{yaml_file} is not a recognised artifact: its single root key must be "
        f"one of {sorted(KIND_DIRS)}. If it is a new kind of file, declare it in "
        "tests/antares_historic/yaml_artifacts.py and cover it with a test."
    )


@pytest.mark.parametrize("yaml_file", YAML_FILES, ids=ids(YAML_FILES))
def test_artifact_is_stored_in_the_directory_of_its_kind(yaml_file: Path) -> None:
    kind = kind_of(yaml_file)
    if kind is None:
        pytest.skip("unrecognised file, see test_yaml_file_is_a_known_artifact")

    assert yaml_file.is_relative_to(
        KIND_DIRS[kind]
    ), f"{yaml_file} is a {kind}: store it under {KIND_DIRS[kind]}"
    if kind in YML_ONLY_KINDS:
        assert yaml_file.suffix == ".yml", f"{yaml_file}: use the .yml extension"


@pytest.mark.parametrize("kind", FILES_BY_KIND)
def test_artifacts_of_each_kind_are_found(kind: str) -> None:
    # Guards against a renamed root key or directory silently emptying the tests.
    assert FILES_BY_KIND[kind], f"No {kind} file found under {KIND_DIRS[kind]}"


@pytest.mark.parametrize("kind", FILES_BY_KIND)
def test_artifact_ids_are_unique(kind: str) -> None:
    # A taxonomy and its flattened copy share their id.
    files = MAIN_TAXONOMY_FILES if kind == TAXONOMY else FILES_BY_KIND[kind]
    counts = Counter(artifact_id(path) for path in files)
    duplicates = sorted(id_ for id_, count in counts.items() if count > 1)

    assert not duplicates, f"Duplicate {kind} ids: {duplicates}"


@pytest.mark.parametrize("taxonomy_file", TAXONOMY_FILES, ids=ids(TAXONOMY_FILES))
def test_taxonomy_file_is_valid(taxonomy_file: Path) -> None:
    load_taxonomy(taxonomy_file)


@pytest.mark.parametrize(
    "flattened_file", FLATTENED_TAXONOMY_FILES, ids=ids(FLATTENED_TAXONOMY_FILES)
)
def test_flattened_taxonomy_has_a_main_taxonomy(flattened_file: Path) -> None:
    main_file = flattened_file.with_name(
        flattened_file.name.replace(f"{FLATTENED_SUFFIX}.", ".")
    )

    assert main_file in MAIN_TAXONOMY_FILES, f"{flattened_file}: {main_file} missing"
    assert artifact_id(flattened_file) == artifact_id(main_file)


@pytest.mark.parametrize(
    "artifact_file", FILES_DECLARING_A_TAXONOMY, ids=ids(FILES_DECLARING_A_TAXONOMY)
)
def test_declared_taxonomy_exists(artifact_file: Path) -> None:
    taxonomy_ids = {artifact_id(path) for path in MAIN_TAXONOMY_FILES}

    assert (
        declared_taxonomy(artifact_file) in taxonomy_ids
    ), f"{artifact_file}: taxonomy '{declared_taxonomy(artifact_file)}' not found"
