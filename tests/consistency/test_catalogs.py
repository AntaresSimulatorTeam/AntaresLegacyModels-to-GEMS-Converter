"""The breakdown keys of the catalog metrics are properties of the taxonomy categories
they apply to.

The rest of the catalog-taxonomy consistency (categories, output ids, location ports)
is checked by GEMS-ViewsBuilder `CatalogsTaxonomyValidator`, in test_views_builder.
"""

import pytest

from tests.consistency.yaml_artifacts import (
    CATALOG_FILES,
    declared_taxonomy,
    ids,
    load_yaml,
    main_taxonomy_file,
)


# GEMS-ViewsBuilder does not check breakdown keys against the taxonomy (an unknown
# key gives a `None` breakdown value): this check stays here until it does.
def build_taxonomy_index(taxonomy):
    """The properties and the parent of each category."""
    return {
        category["id"]: {
            "properties": {prop["id"] for prop in category.get("properties", [])},
            "parent": category.get("parent-category"),
        }
        for category in taxonomy["taxonomy"]["categories"]
    }


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


@pytest.mark.parametrize("catalog_file", CATALOG_FILES, ids=ids(CATALOG_FILES))
def test_metric_breakdown_keys_are_category_properties(catalog_file):
    """In each catalog, every breakdown key of a metric is a property of the taxonomy
    category of each of its terms (or of a parent category), in the declared taxonomy.

    Limits: checked against the main taxonomy only; the flattened copy given to
    GEMS-ViewsBuilder has the same properties (test_flattened_taxonomy_matches_main_taxonomy).
    """
    catalog = load_yaml(catalog_file)
    taxonomy_index = build_taxonomy_index(
        load_yaml(main_taxonomy_file(declared_taxonomy(catalog_file)))
    )

    errors = [
        f"{catalog_file.name}: metric '{metric['id']}' "
        f"uses breakdown key '{breakdown['key']}' "
        f"but category '{term['taxonomy-category']}' does not declare "
        f"this property (nor any parent category)"
        for metric in catalog["catalog"]["metrics-definition"]
        for breakdown in metric.get("breakdown", [])
        for term in metric.get("terms", [])
        if not has_property(term["taxonomy-category"], breakdown["key"], taxonomy_index)
    ]

    assert not errors, "\n".join(errors)
