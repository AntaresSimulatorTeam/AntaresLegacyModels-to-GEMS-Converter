# AGENTS.md

This file provides guidance to AI coding agents (LLMs, copilots, code assistants) when working with this repository. It is tool-agnostic and should be read by any AI agent before making changes.

---

## Project Overview

**AntaresLegacyModels-to-GEMS-Converter** converts [Antares Simulator](https://antares-simulator.readthedocs.io/) legacy studies into the [GEMS](https://gems-energy.readthedocs.io/) format. It takes mathematical models that were historically embedded in the Antares solver code (thermal clusters, links, short-term storage, batteries, renewables, load/solar/wind) and produces explicit GEMS YAML system files and timeseries data.

Repository: `AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter` — License: MPL 2.0

---

## Directory Layout

```
src/antares_gems_converter/
  input_converter/
    src/
      main.py                  # Command line interface entry point 
      converter.py             # AntaresStudyConverter orchestrator:
- Reads an Antares study (full or hybrid mode)
- Loads YAML model templates and validates them against provided lib files
- Iterates areas/links/clusters → builds InputComponent + InputPortConnections lists
- Hybrid: deletes converted Antares objects from the study copy in-place
- Full: also converts areas themselves into GEMS components
- process_all() dumps the final InputSystem to output/input/system.yml


      parsing.py - Pydantic models for YAML conversion templates
- ConversionTemplate - top-level: model id, component, connections, legacy objects to delete
- ParameterConversionConfig - parameter source: constant, matrix column, or object + optional operation
- Operation - max / multiply_by / divide_by on a value or timeseries
- VirtualObjectsRepository - areas/links/thermals to skip during iteration
- All models have resolve_template() to substitute ${area} / ${thermal} placeholders
- parse_conversion_template() - YAML file → ConversionTemplate
      config.py                # Central registry of type→method mappings
      utils.py                 # YAML I/O, path resolution, dataframe validation
      logger.py                # Dual logging (file + stdout)
      data_preprocessing/
        preprocessing.py       # ModelConversionPreprocessor: data extraction
        data_classes.py        # ConversionMode enum (FULL, HYBRID)
    data/
      config.ini               # Default INI config template
      model_configuration/     # YAML conversion templates (one per model type)
  libs/
    antares_historic/
      antares_legacy_models.yml   # GEMS model definitions for Antares legacy components
    reference_models/
      andromede_v1_models.yml     # Andromede reference models (battery, DSR, etc.)
  antares_runner/
    antares_runner.py             # AntaresHybridRunner, benchmarker, modeler runner
tests/
  input_converter/               # Unit/integration tests (in-memory studies via antares-craft)
  antares_historic/              # E2E tests (require Antares Simulator binary)
.github/workflows/ci.yml        # CI pipeline
pyproject.toml                   # Package metadata, tool config (Black, isort, mypy)
dependencies.json                # Antares Simulator version for CI
```

---

## Conversion Pipeline

### Overview

1. **CLI** (`main.py`) reads an INI config specifying study path, output folder, library paths, and conversion mode.
2. **AntaresStudyConverter** (`converter.py`) loads the Antares study via `antares-craft`, loads YAML conversion templates, and iterates through each model type.
3. **Template resolution** (`parsing.py`) replaces `${area}`, `${thermal}`, `${link}`, etc. placeholders with actual study object IDs.
4. **Data extraction** (`preprocessing.py`) reads timeseries and scalar parameters from the study, writes timeseries to TSV files in `data-series/`.
5. **Output**: a GEMS `InputSystem` dumped to `input/system.yml`, with libraries copied to `model-libraries/` and timeseries in `data-series/`.

### Two Conversion Modes

- **FULL** — generates a standalone GEMS study from scratch. Components are connected via port-to-port connections.
- **HYBRID** — copies the original Antares study, replaces selected components with GEMS equivalents using area-connections, and deletes the replaced legacy objects from the copy.

### Config Registry (`config.py`)

This module is a central lookup table that maps component types to `antares-craft` API methods. All dictionaries (`TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD`, `MATRIX_TYPES_TO_SET_METHOD`, `TIMESERIES_NAME_TO_METHOD`, `MODEL_NAME_TO_FILE_NAME`, etc.) are authoritative — adding a new model type requires entries here.

---

## YAML Conversion Templates

Templates in `input_converter/data/model_configuration/` define how each Antares component type maps to a GEMS model. Every template follows this structure:

```yaml
template:
  name: <model-name>                    # e.g., thermal, link, battery
  model: <library_id>.<model_id>        # GEMS model reference
  template-parameters:
    - name: <variable>                  # e.g., area
      cluster-type: <type>              # optional: thermal, renewable, st_storage
      exclude:                          # optional: virtual objects to skip
        - id: <identifier>
          object-properties: { type: ..., area: ... }

  component:
    id: <id-with-${variable}-placeholders>
    parameters:
      - id: <param_id>
        time-dependent: true|false
        scenario-dependent: true|false
        value:
          constant: <number>            # OR object-properties reference below
          object-properties:
            type: thermal|link|load|solar|wind|st_storage|binding_constraint
            area: ${area}
            cluster: ${thermal}         # for cluster types
            field: <antares_craft_field>
          column: <int>                 # optional: extract specific column from matrix
          operation:                    # optional: transform after extraction
            type: max                   # OR multiply_by: <value> OR divide_by: <param_ref>

  connections: [...]          # FULL mode: port-to-port
  area-connections: [...]     # HYBRID mode: port-to-area
  legacy-objects-to-delete: [...]  # HYBRID mode: objects to remove from copied study
```

### Key template conventions

- **Placeholder syntax**: `${area}`, `${thermal}`, `${link}`, etc. — resolved per-object iteration.
- **Link attribute access**: `${link}.area_from_id`, `${link}.area_to_id` resolved via `getattr()` on Link objects.
- **Binding constraint fields**: use `%` and `.` separators (e.g., `${area}%z_batteries`, `${area}.${area}_batteries_inj`).
- **Column indexing**: 0-based, references actual columns in Antares data matrices.
- **Operations**: `divide_by` can reference another parameter by name (e.g., `divide_by: reservoir_capacity`).

---

## GEMS Model Libraries

- **`antares_legacy_models.yml`** — Defines models for: `area`, `load`, `link`, `renewable`, `thermal`, `short-term-storage`. These mirror the implicit models in the Antares solver.
- **`andromede_v1_models.yml`** — Defines models for: `battery`, `dsr`, `electrolyser`. These represent new modelling constructs not in the legacy solver.

Both use the `flow` port-type with a single `flow` field for power balance connections.

---

## Running Tests

### Prerequisites

- Python >= 3.10 (CI uses 3.12)
- E2E tests in `tests/antares_historic/` require the Antares Simulator binary (version in `dependencies.json`, currently 10.0.0). CI downloads it automatically.

### Commands

```bash
pip install -e .
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run a specific test
pytest tests/input_converter/test_converter.py::TestConverter::test_thermal_model_conversion

# Run one test file
pytest tests/antares_historic/test_thermal_model.py

# Type checking
mypy src/

# Formatting check
black --check --diff .

# Coverage
pytest --cov=antares_gems_converter --cov=antares_runner --cov-report=html
```

### Test Architecture

| Directory | What it tests | Dependencies |
|-----------|---------------|--------------|
| `tests/input_converter/` | Converter logic: template resolution, YAML output, component generation | `antares-craft` in-memory studies (no binary needed) |
| `tests/antares_historic/` | Converted models produce correct optimization results | Antares Simulator binary |

**Fixture pattern** (`tests/input_converter/conftest.py`): Fixtures build up progressively — `local_study` → `local_study_w_areas` → `local_study_w_thermal` etc. Each adds one layer of study objects. Tests use these to create `AntaresStudyConverter` instances.

**E2E test pattern** (`tests/antares_historic/`): Create an Antares study programmatically, convert it to GEMS format, run both original and converted through the solver, compare objective values with `pytest.approx()`.

---

## CI Pipeline (`.github/workflows/ci.yml`)

Runs on every push to any branch:

1. Install deps (`pip install -e . && pip install -r requirements-dev.txt`)
2. Type checking: `mypy src/`
3. Format checking: `black --check --diff` (Black 23.7.x)
4. Download and extract Antares Simulator binary
5. Run tests with coverage
6. Upload coverage artifact

---

## Git & Branching Model

- **`main`** — stable releases
- Feature branches from `main`, PRs back to `main`
- No direct pushes to `main`
- CI runs on every push

---

## Coding Conventions

### Python

- Type hints required; `pathlib.Path` for file paths
- Pydantic `BaseModel` for data structures (with `_to_kebab` alias generator for YAML key mapping)
- Logging via `logging.getLogger(__name__)` — no bare `print()`
- Black formatter (line-length 88), isort with Black profile
- mypy enforced on `src/` only (`ignore_missing_imports = true`)
- `pythonpath = src` in `pytest.ini` — imports use `antares_gems_converter.*`

### YAML

- Conversion template files use kebab-case keys: `template-parameters`, `cluster-type`, `object-properties`
- Library model IDs: lowercase with underscores or hyphens
- Indentation: 2 spaces

---

## Critical Rules for AI Agents

1. **Template ↔ model parameter ids must match.** Every parameter in a conversion template's `component.parameters` must correspond to a parameter in the referenced GEMS model. A mismatch causes silent failures at runtime.

2. **`config.py` is the single source of truth for mappings.** Adding a new model type requires entries in `MODEL_NAME_TO_FILE_NAME`, and possibly `TEMPLATE_CLUSTER_TYPE_TO_GET_METHOD`, `TIMESERIES_NAME_TO_METHOD`, and other mapping dictionaries. Missing entries cause `KeyError` at runtime.

3. **Library YAML files are shared across studies.** `libs/antares_historic/antares_legacy_models.yml` and `libs/reference_models/andromede_v1_models.yml` are referenced by all conversion templates. Edits to model parameters, port definitions, or constraints affect every conversion.

4. **FULL vs HYBRID mode produce different outputs.** FULL mode uses `connections` (port-to-port). HYBRID mode uses `area-connections` and `legacy-objects-to-delete`. Both sections must be kept consistent in templates.

5. **Battery conversion uses virtual objects.** The battery template excludes virtual areas (`z_batteries`, `z_batteries_pcomp`) and proxy thermal clusters. The `VirtualObjectsRepository` tracks these so they are skipped during area/cluster iteration. Breaking this exclusion logic causes duplicate or invalid components.

6. **Timeseries validity check.** `check_timeseries_validity()` rejects empty or all-zero dataframes. If a load/solar/wind area has no data, no component is generated for it. This is intentional — do not "fix" it by removing the check.

7. **Space handling in IDs.** Component IDs have spaces replaced with underscores (`id.replace(" ", "_")`). This is applied during iteration, not in templates.

8. **Floating-point comparisons.** Always use `pytest.approx()` in tests. E2E tests use relative tolerance (`rel=0.01`) to account for LP relaxation vs MIP differences.

9. **Do not commit** `venv/`, `site/`, extracted Antares binaries, or `.pyc` files.

10. **Version tracking.** The Antares Simulator version is tracked in `dependencies.json`. When updating, also check the CI workflow and test conftest for hardcoded version references.

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `antares-craft` | Read/write Antares studies programmatically |
| `gemspy` | GEMS interpreter — provides `InputSystem`, `InputComponent`, etc. |
| `pydantic` | Data validation for conversion templates and structured data |
| `pandas` / `numpy` | Timeseries data handling and numerical operations |
| `PyYAML` | YAML parsing and generation |

---

## Related Projects

| Project | Relationship |
|---------|-------------|
| [GEMS](https://github.com/AntaresSimulatorTeam/GEMS) | Defines the target YAML modelling language |
| [Antares Simulator](https://github.com/AntaresSimulatorTeam/Antares_Simulator) | Source format; solver binary used for E2E validation |
| [GemsPy](https://github.com/AntaresSimulatorTeam/GemsPy) | Python interpreter for GEMS models |
| [PyPSA-to-GEMS-Converter](https://github.com/AntaresSimulatorTeam/PyPSA-to-GEMS-Converter) | Sibling converter for PyPSA models |
