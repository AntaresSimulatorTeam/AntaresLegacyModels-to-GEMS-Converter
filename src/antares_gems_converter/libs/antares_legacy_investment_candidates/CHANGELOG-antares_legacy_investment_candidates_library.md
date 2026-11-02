# Antares Legacy Investment Candidates Library — Changelog

All notable changes to the Antares legacy investment candidates library (`src/antares_gems_converter/libs/antares_legacy_investment_candidates/`) are documented here.

Versioning follows the rules defined in `COMPATIBILITY.md`:

- **Major** — New model added
- **Minor** — Bug fix or improvement to an existing model
- **Patch** — Non-functional change (rename variable/parameter, internal refactor)

---

## [1.0.0]

- Initial release, replacing `xpansion_models.yml` of the Xpansion direct generator (`xpansion_generator/`, on the `feature/xpansion-direct-generator` branch).
- `continuous_investment_candidate`: continuous invested capacity, between `min_investment` and `max_investment` (MW).
- `discrete_investment_candidate`: integer number of units of `unit_size`, whose invested capacity is between `min_investment` and `max_investment` (MW).
- Both models, inspired by Antares Xpansion link candidates:
  - `existing_capacity` (MW): capacity already installed, without investment cost, with the same `availability_rate` as the invested capacity (Xpansion's `already-installed-capacity`, without a separate profile).
  - `min_investment` / `max_investment` (MW): limits on the invested capacity, existing capacity excluded (Xpansion's `max-investment`; Xpansion has no minimum).
  - A `capacity_port`, connected to an Antares thermal cluster through `thermal-capacity-connections` (hybrid mode only). Its capacity replaces the cluster's availability: `existing_capacity` keeps the cluster's installed capacity.
- Studies using these models need `resolution-mode: benders-decomposition` in `optim-config.yml` (Antares 10.2 rejects scenario-independent variables otherwise), and a solver supporting integer variables for the discrete model. The connected cluster's nominal capacity must cover `existing_capacity + max_investment` (it still bounds generation in accurate unit commitment). In the discrete model, `unit_size` must be positive and a multiple of it must lie between `min_investment` and `max_investment`.
- E2E tests: `tests/antares_historic/test_investment_candidates_model.py`.
- Changes from `xpansion_models.yml`:
  - Identifiers in English: `gisement_total` → `max_investment`, `coef_disponibilite` → `availability_rate`, `cost_per_year` → `annual_investment_cost`, `units_to_invest` → `invested_units`.
  - `annual_investment_cost` is per MW and per year in both models, as Xpansion's `annual-cost-per-mw`; the old `cost_per_year` was per unit.
  - `max_unit` (units) and the `capa_max` constraint (`units_to_invest * unit_size <= gisement_total`) are replaced by the `max_investment` bound (MW): `invested_units <= max_investment / unit_size`.
  - The battery model and the `storage_capacity_port` port type are removed: Antares cannot connect a capacity port to a short-term storage yet.
