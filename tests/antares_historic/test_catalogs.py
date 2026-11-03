import pytest

from tests.antares_historic.yaml_artifacts import (
    CATALOG_FILES,
    FLATTENED_TAXONOMY_FILES,
    declared_taxonomy,
    flattened_taxonomy_file,
    ids,
    load_yaml,
    main_taxonomy_file,
)


# This test will be in GEMS view builder but we keep it here for now for ease of development when working on metrics definition.
def build_taxonomy_index(taxonomy):
    categories = {}

    for category in taxonomy["taxonomy"]["categories"]:
        ids = {variable["id"] for variable in category.get("variables", [])}

        ids.update(output["id"] for output in category.get("extra-outputs", []))

        properties = {prop["id"] for prop in category.get("properties", [])}

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


@pytest.mark.parametrize("catalog_file", CATALOG_FILES, ids=ids(CATALOG_FILES))
def test_metrics_output_ids_and_properties_are_declared(catalog_file):
    catalog = load_yaml(catalog_file)

    taxonomy = load_yaml(main_taxonomy_file(declared_taxonomy(catalog_file)))

    taxonomy_index = build_taxonomy_index(taxonomy)

    errors = find_taxonomy_errors(catalog, catalog_file, taxonomy_index)

    assert not errors, "\n".join(errors)


# The flattened taxonomy is provisional: it exists only until the view builder
# supports parent categories, at which point it can be removed.
@pytest.mark.parametrize("catalog_file", CATALOG_FILES, ids=ids(CATALOG_FILES))
def test_metrics_output_ids_and_properties_are_declared_in_flattened_taxonomy(
    catalog_file,
):
    catalog = load_yaml(catalog_file)

    flattened_file = flattened_taxonomy_file(
        main_taxonomy_file(declared_taxonomy(catalog_file))
    )
    if flattened_file is None:
        pytest.skip(f"taxonomy '{declared_taxonomy(catalog_file)}' is not flattened")
    taxonomy = load_yaml(flattened_file)
    taxonomy_index = build_taxonomy_index(taxonomy)

    errors = find_taxonomy_errors(catalog, catalog_file, taxonomy_index)

    assert not errors, "\n".join(errors)


def resolve_category(category_id, taxonomy_index):
    """Return the ids and properties of a category, parents included."""
    ids, properties = set(), set()
    current = category_id

    while current:
        category = taxonomy_index[current]
        ids |= category["ids"]
        properties |= category["properties"]
        current = category["parent"]

    return ids, properties


# The flattened taxonomies are maintained by hand: each must stay equivalent to
# its main taxonomy once parent categories are resolved.
@pytest.mark.parametrize(
    "flattened_file", FLATTENED_TAXONOMY_FILES, ids=ids(FLATTENED_TAXONOMY_FILES)
)
def test_flattened_taxonomy_matches_main_taxonomy(flattened_file):
    flattened_taxonomy = load_yaml(flattened_file)
    main_index = build_taxonomy_index(
        load_yaml(main_taxonomy_file(flattened_taxonomy["taxonomy"]["id"]))
    )
    flattened_index = build_taxonomy_index(flattened_taxonomy)

    assert flattened_index.keys() == main_index.keys()

    for category_id, category in flattened_index.items():
        assert category["parent"] is None, f"'{category_id}' still has a parent"
        assert (category["ids"], category["properties"]) == resolve_category(
            category_id, main_index
        ), f"'{category_id}' differs between main and flattened taxonomies"


def find_taxonomy_errors(catalog, catalog_file, taxonomy_index):
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

    return errors
