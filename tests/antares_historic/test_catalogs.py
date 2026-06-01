from pathlib import Path

import pytest
import yaml


CATALOGS_DIR = Path("src/antares_gems_converter/catalogs")
TAXONOMIES_DIR = Path("src/antares_gems_converter/taxonomies")


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_taxonomy_index(taxonomy):
    categories = {}

    for category in taxonomy["taxonomy"]["categories"]:
        ids = {
            variable["id"]
            for variable in category.get("variables", [])
        }

        ids.update(
            output["id"]
            for output in category.get("extra-outputs", [])
        )

        properties = {
            prop["id"]
            for prop in category.get("properties", [])
        }

        categories[category["id"]] = {
            "ids": ids,
            "properties": properties,
            "parent": category.get("parent-category"),
        }

    return categories


def is_declared_output(category_id, output_id, taxonomy_index):
    current = category_id

    while current:
        category = taxonomy_index.get(current)

        if category is None:
            return False

        if output_id in category["ids"]:
            return True

        current = category["parent"]

    return False

def has_property(category_id, property_id, taxonomy_index):
    current = category_id

    while current:
        category = taxonomy_index.get(current)

        if category is None:
            return False

        if property_id in category["properties"]:
            return True

        current = category["parent"]

    return False

def load_taxonomy_by_id(taxonomy_id):
    for path in TAXONOMIES_DIR.glob("*.yml"):
        taxonomy = load_yaml(path)

        if taxonomy["taxonomy"]["id"] == taxonomy_id:
            return taxonomy

    raise ValueError(f"Taxonomy '{taxonomy_id}' not found")


CATALOG_FILES = list(CATALOGS_DIR.glob("*.yml"))


@pytest.mark.parametrize("catalog_file", CATALOG_FILES)
def test_metrics_output_ids_and_properties_are_declared(catalog_file):
    catalog = load_yaml(catalog_file)

    taxonomy_id = catalog["catalog"]["taxonomy"]
    taxonomy = load_taxonomy_by_id(taxonomy_id)

    taxonomy_index = build_taxonomy_index(taxonomy)

    errors = []

    for metric in catalog["catalog"]["metrics-definition"]:
        metric_id = metric["id"]

        for term in metric.get("terms", []):
            category = term["taxonomy-category"]
            output_id = term["output-id"]

            if category not in taxonomy_index:
                errors.append(
                    f"{catalog_file.name}: metric '{metric_id}' "
                    f"references unknown category '{category}'"
                )
                continue

            if not is_declared_output(
                category,
                output_id,
                taxonomy_index,
            ):
                errors.append(
                    f"{catalog_file.name}: metric '{metric_id}' "
                    f"references output-id '{output_id}' "
                    f"not declared in category '{category}' "
                )
        
        breakdowns = metric.get("breakdown", [])

        for breakdown in breakdowns:
            property_id = breakdown["key"]

            for term in metric.get("terms", []):
                category = term["taxonomy-category"]

                if not has_property(
                    category,
                    property_id,
                    taxonomy_index,
                ):
                    errors.append(
                        f"{catalog_file.name}: metric '{metric_id}' "
                        f"uses breakdown key '{property_id}' "
                        f"but category '{category}' does not declare "
                        f"this property (nor any parent category)"
                    )

    assert not errors, "\n".join(errors)