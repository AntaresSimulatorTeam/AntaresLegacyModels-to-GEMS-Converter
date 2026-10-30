# Antares Legacy Investment Candidates Library — Changelog

All notable changes to the Antares legacy investment candidates library (`src/antares_gems_converter/libs/antares_legacy_investment_candidates/`) are documented here.

Versioning follows the rules defined in `COMPATIBILITY.md`:

- **Major** — New model added
- **Minor** — Bug fix or improvement to an existing model
- **Patch** — Non-functional change (rename variable/parameter, internal refactor)

---

## [1.0.0]

- Initial release, replacing `xpansion_models.yml` of the Xpansion direct generator (`xpansion_generator/`).
- `continuous_investment_candidate`: continuous invested capacity, bounded by `max_capacity`.
- `discrete_investment_candidate`: integer number of units of `unit_size`, bounded by `max_units`.
- Both have a `capacity_port`, connected to an Antares thermal cluster through `thermal-capacity-connections` (hybrid mode only).
- Changes from `xpansion_models.yml`:
  - Identifiers in English: `gisement_total` → `max_capacity`, `coef_disponibilite` → `availability_rate`, `max_unit` → `max_units`, `cost_per_year` → `annual_investment_cost`, `units_to_invest` → `invested_units`.
  - `annual_investment_cost` is per MW and per year in both models, as Xpansion's `annual-cost-per-mw`; the old `cost_per_year` was per unit.
  - The `capa_max` constraint (`units_to_invest * unit_size <= gisement_total`) is removed: the `max_units` bound gives the same limit when the capacity limit is a multiple of `unit_size`.
  - The battery model and the `storage_capacity_port` port type are removed: Antares cannot connect a capacity port to a short-term storage yet.
