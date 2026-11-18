"""Each flattened taxonomy is equivalent to its main taxonomy."""

import pytest
from gems_craft.model.taxonomy import TaxonomyCategory, load_taxonomy

from tests.consistency.yaml_artifacts import (
    FLATTENED_TAXONOMY_FILES,
    artifact_id,
    ids,
    main_taxonomy_file,
)

# The items a taxonomy category lists, each one required from the models of the
# category (GemsPy `check_library_against_taxonomy`).
CATEGORY_FIELDS = [
    "variables",
    "parameters",
    "ports",
    "port_field_definitions",
    "constraints",
    "binding_constraints",
    "extra_outputs",
    "properties",
]


def inherited_items(
    category_id: str, field: str, categories: dict[str, TaxonomyCategory]
) -> set[str]:
    """The ids listed in one field of a category and of its parent categories."""
    items: set[str] = set()
    current: str | None = category_id

    while current:
        items |= {item.id for item in getattr(categories[current], field)}
        current = categories[current].parent_category

    return items


# The flattened taxonomies are maintained by hand: each must stay equivalent to
# its main taxonomy once parent categories are resolved. The libraries are checked
# against the flattened copy only (test_views_builder), which is enough because
# GemsPy `check_library_against_taxonomy` does not resolve parent categories: the
# flattened copy requires at least what the main taxonomy does.
@pytest.mark.parametrize(
    "flattened_file", FLATTENED_TAXONOMY_FILES, ids=ids(FLATTENED_TAXONOMY_FILES)
)
def test_flattened_taxonomy_matches_main_taxonomy(flattened_file):
    """Each flattened taxonomy, maintained by hand, has the categories of its main
    taxonomy, none with a parent, each with the variables, parameters, ports,
    port-field definitions, constraints, binding constraints, extra outputs and
    properties it has in the main taxonomy, inherited ones included.
    """
    main_categories = {
        category.id: category
        for category in load_taxonomy(
            main_taxonomy_file(artifact_id(flattened_file))
        ).categories
    }
    flattened_categories = {
        category.id: category for category in load_taxonomy(flattened_file).categories
    }

    assert flattened_categories.keys() == main_categories.keys()

    errors = []
    for category_id, category in flattened_categories.items():
        if category.parent_category is not None:
            errors.append(f"'{category_id}' still has a parent")
        for field in CATEGORY_FIELDS:
            flattened_items = {item.id for item in getattr(category, field)}
            main_items = inherited_items(category_id, field, main_categories)
            if flattened_items != main_items:
                errors.append(
                    f"'{category_id}' {field}: missing from the flattened taxonomy "
                    f"{sorted(main_items - flattened_items)}, not in the main "
                    f"taxonomy {sorted(flattened_items - main_items)}"
                )

    assert not errors, "\n".join(errors)
