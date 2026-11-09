# Antares Legacy Investment Candidates Library — Changelog

All notable changes to the Antares legacy investment candidates library (`src/antares_gems_converter/libs/antares_legacy_investment_candidates/`) are documented here.

Versioning follows the rules defined in `COMPATIBILITY.md`:

- **Major** — New model added
- **Minor** — Bug fix or improvement to an existing model
- **Patch** — Non-functional change (rename variable/parameter, internal refactor)

---

## [1.0.0]

- Initial release.
- `continuous_investment_candidate`: continuous invested capacity, between `min_investment` and `max_investment` (MW).
- `discrete_investment_candidate`: integer number of units of `unit_size` (MW), between `min_units` and `max_units` (Xpansion's `max-units`).
- Both models, inspired by Antares Xpansion link candidates:
  - `existing_capacity` (MW): capacity already installed, without investment cost, with the same `availability_rate` as the invested capacity (Xpansion's `already-installed-capacity`, without a separate profile).
  - The bounds apply to the investment, existing capacity excluded (Xpansion's `max-investment` and `max-units`; Xpansion has no minimum).
  - `annual_investment_cost` is per MW and per year, also for the discrete model (Xpansion's `annual-cost-per-mw`).
  - A `capacity_port`, connected to an Antares thermal cluster through `thermal-capacity-connections` (hybrid mode only). Its capacity replaces the cluster's availability: `existing_capacity` keeps the cluster's installed capacity.
- Studies using these models need `resolution-mode: benders-decomposition` in `optim-config.yml` (Antares 10.2 rejects scenario-independent variables otherwise), and a solver supporting integer variables for the discrete model. The connected cluster's nominal capacity must cover the existing capacity plus the largest investment (it still bounds generation in accurate unit commitment).
- E2E tests: `tests/antares_historic/test_investment_candidates_model.py`.
