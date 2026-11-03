"""Consistency of the YAML artifacts and of this documentation

The libraries, catalogs, taxonomies and view configs of the package, wherever they
are stored: they are found by their root key (see `tests/consistency/yaml_artifacts.py`).
No Antares binary is needed: CI runs these tests first, in the "Consistency checks"
step, so they fail fast.
"""
