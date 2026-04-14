# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

For full project context, conventions, template format details, and critical rules, see [AGENTS.md](AGENTS.md) — read it before making any changes.

## Quick Reference

```bash
# Install
pip install -e . && pip install -r requirements-dev.txt

# Run all tests
pytest

# Run a specific test
pytest tests/input_converter/test_converter.py::TestConverter::test_thermal_model_conversion

# Type checking & formatting
mypy src/
black --check --diff .

# Run the converter
python -m antares_gems_converter.input_converter.src.main --conf path/to/config.ini
```

## Key Reminders

- Antares binary version is tracked in `dependencies.json` (currently 10.0.0); CI downloads it automatically
- `config.py` is the single source of truth for model type mappings — new model types require entries there
- Template parameter counts must exactly match the referenced GEMS model's parameters
- `libs/` YAML model files are shared across all conversions — edits affect everything
- FULL mode uses port-to-port `connections`; HYBRID mode uses `area-connections` + `legacy-objects-to-delete`
- Battery conversion depends on virtual object exclusion (`z_batteries`, proxy thermals) — handle with care
- Git workflow: feature branches from `main`, PRs back to `main`
