"""The list of tests in CHECKS.md is complete and up to date.

It is generated from the docstrings of the tests by tests/document_checks.py.
"""

from pathlib import Path

import pytest

from tests.document_checks import (
    CHECKS_DOC,
    DOCUMENTED_DIRS,
    REGENERATE_COMMAND,
    TESTS_DIR,
    missing_docstrings,
    modules_of,
    relative,
    updated_checks_doc,
)

DOCUMENTED_MODULES = [
    path for directory in DOCUMENTED_DIRS for path in modules_of(directory)
]


@pytest.mark.parametrize(
    "test_module", DOCUMENTED_MODULES, ids=[relative(p) for p in DOCUMENTED_MODULES]
)
def test_every_test_has_a_docstring(test_module: Path) -> None:
    """Every test module and test of a directory listed in CHECKS.md has a docstring
    stating what it guarantees."""
    missing = missing_docstrings(test_module)

    assert not missing, (
        f"{relative(test_module)}: no docstring for {missing}. State what is "
        f"guaranteed, then run `{REGENERATE_COMMAND}`."
    )


@pytest.mark.parametrize("directory", DOCUMENTED_DIRS)
def test_documented_directory_has_a_title(directory: str) -> None:
    """Every test directory listed in CHECKS.md has a title, in the docstring of its
    `__init__.py`."""
    missing = missing_docstrings(TESTS_DIR / directory / "__init__.py")

    assert not missing, f"tests/{directory}/__init__.py: add a docstring (its title)"


def test_checks_doc_is_up_to_date() -> None:
    """CHECKS.md lists the current tests of its directories, with their current
    docstrings."""
    current = CHECKS_DOC.read_text(encoding="utf-8")

    assert current == updated_checks_doc(
        current
    ), f"{relative(CHECKS_DOC)} is out of date: run `{REGENERATE_COMMAND}`"
