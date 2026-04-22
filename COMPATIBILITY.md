# Antares-Legacy-to-GEMS Converter — Compatibility Matrix

This table maps converter versions to the tool versions they are compatible with.

| Converter | Antares-Simulator | antares-craft | GemsPy | Notes |
|-----------|-------------------|---------------|--------|-------|
| 0.0.1     | 10.0.0            | 0.3.0         | 0.0.2  | Initial release |

## Versioning Policy

- **Converter** — version in `pyproject.toml` (`[project] version`). Follows semantic versioning:
  - **Major** — Antares-Simulator major version bump
  - **Minor** — Bug fix, new feature, or antares-craft/GemsPy version update
  - **Patch** — Dependency updates, code optimisation, or library-only change

- **Antares Legacy Models Library** — version tracked in `src/antares_gems_converter/libs/antares_historic/CHANGELOG-antares_legacy_models_library.md` (latest entry header). Independent versioning:
  - **Major** — New legacy model added to `antares_legacy_models.yml`
  - **Minor** — Bug fix or improvement to an existing model
  - **Patch** — Non-functional change (rename variable/parameter, internal refactor)

- **Antares-Simulator** — tracked version in `dependencies.json` (`antares_simulator_version`). The version downloaded by CI and used for all tests.

- **antares-craft** — pinned version in `requirements.txt`. The library used to read Antares studies.

- **GemsPy** — pinned version in `requirements.txt`. The GEMS interpreter used for study generation and validation.

## Compatibility Rules

- Patch versions are always backward-compatible within the same Major.Minor.
- Upgrading Antares-Simulator, antares-craft, or GemsPy may require a converter Minor or Major bump — see git history for details.

## Version Files

| Component | Current Version | Version File |
|-----------|----------------|--------------|
| Converter | 0.0.1 | `pyproject.toml` |
| Antares Legacy Models Library | 1.0.0 | `src/antares_gems_converter/libs/antares_historic/CHANGELOG-antares_legacy_models_library.md` |
| Antares-Simulator | 10.0.0 | `dependencies.json` |
| antares-craft | 0.3.0 | `requirements.txt` |
| GemsPy | 0.0.2 | `requirements.txt` |
