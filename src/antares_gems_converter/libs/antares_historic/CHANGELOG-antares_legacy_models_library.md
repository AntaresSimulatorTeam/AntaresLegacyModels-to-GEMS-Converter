# Antares Legacy Models Library — Changelog

All notable changes to the Antares Legacy models library (`src/antares_gems_converter/libs/`) are documented here.

Versioning follows the rules defined in `COMPATIBILITY.md`:

- **Major** — New legacy model added
- **Minor** — Bug fix or improvement to an existing model
- **Patch** — Non-functional change (rename variable/parameter, internal refactor)

---

## [1.1.0]

Renaming to match naming conventions. 

New extra-outputs to be able to reproduce lecagy output metrics.

New parameter on link model : loop flow. 

---

## [1.0.0] — 2026-04-19

Initial baseline release.

Supported model types: area, thermal, link, short-term storage (STS), load, renewables.
HYBRID and FULL conversion modes supported.

Validated against Antares-Simulator 10.0.0, antares-craft 0.3.0, GemsPy 0.0.2.
