from pathlib import Path

import pytest
import yaml


TAXONOMIES_DIR = Path("src/antares_gems_converter/taxonomies")
MODEL_LIBRARIES_DIR = Path("src/antares_gems_converter/libs")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

# TODO : remove this test once it will be implemented in GEMSPy
def load_taxonomy_by_id(taxonomy_id):
    for path in TAXONOMIES_DIR.glob("*.yml"):
        content = load_yaml(path)

        if content["taxonomy"]["id"] == taxonomy_id:
            return content

    raise ValueError(f"Taxonomy '{taxonomy_id}' not found")


def build_taxonomy_index(taxonomy):
    categories = {}

    for category in taxonomy["taxonomy"]["categories"]:
        categories[category["id"]] = {
            "parent": category.get("parent-category"),
            "variables": {item["id"] for item in category.get("variables", [])},
            "ports": {item["id"] for item in category.get("ports", [])},
            "constraints": {item["id"] for item in category.get("constraints", [])},
            "extra_outputs": {item["id"] for item in category.get("extra-outputs", [])},
            "properties": {item["id"] for item in category.get("properties", [])},
            "parameters": {item["id"] for item in category.get("parameters", [])},
        }

    return categories


def get_expected_items(category_id, taxonomy_index, field):
    """
    Returns all expected items for a taxonomy category,
    including inherited items from parent categories.
    """
    expected = set()

    current = category_id

    while current:
        category = taxonomy_index[current]
        expected.update(category[field])
        current = category["parent"]

    return expected


MODEL_LIBRARY_FILES = list(MODEL_LIBRARIES_DIR.rglob("*.yml"))


@pytest.mark.parametrize("library_file", MODEL_LIBRARY_FILES)
def test_models_implement_taxonomy_contract(library_file):
    library = load_yaml(library_file)

    taxonomy_id = library["library"]["taxonomy"]

    taxonomy = load_taxonomy_by_id(taxonomy_id)
    taxonomy_index = build_taxonomy_index(taxonomy)

    errors = []

    for model in library["library"]["models"]:
        model_id = model["id"]

        category_id = model.get("taxonomy-category")

        # Models without taxonomy category are ignored
        if not category_id:
            continue

        if category_id not in taxonomy_index:
            errors.append(
                f"{library_file.name}: model '{model_id}' "
                f"references unknown taxonomy category "
                f"'{category_id}'"
            )
            continue

        actual = {
            "variables": {item["id"] for item in model.get("variables", [])},
            "ports": {item["id"] for item in model.get("ports", [])},
            "constraints": {
                item["id"] for item in model.get("binding-constraints", [])
            },
            "extra_outputs": {item["id"] for item in model.get("extra-outputs", [])},
            "properties": {item["id"] for item in model.get("properties", [])},
            "parameters": {item["id"] for item in model.get("parameters", [])},
        }

        for field in actual:
            expected = get_expected_items(
                category_id,
                taxonomy_index,
                field,
            )

            missing = expected - actual[field]

            if missing:
                errors.append(
                    f"{library_file.name}: model '{model_id}' "
                    f"is missing {field}: {sorted(missing)}"
                )

    assert not errors, "\n".join(errors)
