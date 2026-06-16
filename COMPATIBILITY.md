# Antares-Legacy-to-GEMS Converter — Compatibility Matrix

This table maps converter versions to the tool versions they are compatible with.

| Converter | Antares-Simulator | antares-craft | GemsPy | Notes |
|-----------|-------------------|---------------|--------|-------|
| 0.2.0     | 10.1.0            | 0.14.0        | 0.1.2  | GemsPy 0.1.2; Antares Legacy Models Library 2.1.1|
| 0.1.3     | 10.1.0            | 0.14.0        | 0.1.0  | Antares Legacy Models Library 2.1.0: market bid cost in thermal, STS overflow and variation penalties |
| 0.1.2     | 10.1.0            | 0.14.0        | 0.1.0  | New templates for hydro and misc gen, changes on template parsing |
| 0.1.1     | 10.1.0            | 0.14.0        | 0.1.0  | Added new models in Antares Legacy Models |
| 0.1.0     | 10.1.0            | 0.14.0        | 0.1.0  | antares-craft 0.14.0 (numpy 2.x) GemsPy 0.1.0 (massive refactoring: rename Input* classes to *Schema) |
| 0.0.1     | 10.0.0            | 0.3.0         | 0.0.2  | Initial release |

## Versioning Policy

- **Converter** — version in `pyproject.toml` (`[project] version`). Follows semantic versioning:
  - **Major** — Antares-Simulator major version bump
  - **Minor** — Bug fix, new feature, or antares-craft/GemsPy version update
  - **Patch** — Dependency updates, code optimisation, or library-only change

- **Antares Legacy Models Library** — version tracked in `src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml` (`library.version`). Independent versioning:
  - **Major** — New legacy model added to `antares_legacy_models.yml`
  - **Minor** — Bug fix or improvement to an existing model
  - **Patch** — Non-functional change (rename variable/parameter, internal refactor)

- **Antares-Simulator** — tracked version in `dependencies.json` (`antares_simulator_version`). The version downloaded by CI and used for all tests.

- **antares-craft** — pinned version in `pyproject.toml`. The library used to read Antares studies.

- **GemsPy** — pinned version in `pyproject.toml`. The GEMS interpreter used for study generation and validation.

## Python library dependencies

Runtime and development Python packages are pinned to exact versions in `pyproject.toml`. Resolved versions are recorded in `uv.lock` (regenerate with `uv lock` after changing pins).

| Package | Pinned version |
|---------|----------------|
| antares-craft | 0.14.0 |
| antares-study-version | 1.0.20 |
| antares-timeseries-generation | 0.1.9 |
| gemspy | 0.1.2 |
| numpy | 2.2.6 |
| pandas | 2.2.3 |
| pydantic | 2.11.10 |
| PyYAML | 6.0.3 |
| pytest | 9.0.3 |
| pytest-cov | 7.1.0 |
| mypy | 2.1.0 |
| types-pyyaml | 6.0.12 |
| black | 23.7.0 |

## Compatibility Rules

- Patch versions are always backward-compatible within the same Major.Minor.
- Upgrading Antares-Simulator, antares-craft, or GemsPy may require a converter Minor or Major bump — see git history for details.

## Version Files

| Component | Current Version | Version File |
|-----------|----------------|--------------|
| Converter | 0.2.0 | `pyproject.toml` |
| Antares Legacy Models Library | 2.1.1 | `src/antares_gems_converter/libs/antares_historic/antares_legacy_models.yml` |
| Antares-Simulator | 10.1.0 | `dependencies.json` |
| antares-craft | 0.14.0 | `pyproject.toml` |
| GemsPy | 0.1.2 | `pyproject.toml` |
