# Extended Models Library — Changelog

All notable changes to `extended_models.yml` are documented here. Its versioning follows
the rules of the Antares Legacy Models Library (see `COMPATIBILITY.md`).

---

## [1.0.0]

First version, from an external library also named `antares_legacy_models` (kept as test
data: `tests/antares_historic/data/reference_libraries/antares_legacy_models_external.yml`).

- **Scope**: only the models that are not in `antares_legacy_models.yml`: `thermal_milp`,
  `thermal_fast`, `short-term-storage`, `enr`, `enr_multi_bandes`, `bc_nuc_hourly_less`,
  `bc_nuc_hourly_greater`, `bc_nuc_daily_less`, `bc_nuc_weekly_less` and
  `bc_nuc_weekly_greater`. Dropped, since `antares_legacy_models.yml` has them: `area`,
  `load`, `link`, `renewable` and `thermal_accurate` (`antares_legacy_models.thermal`; the
  external `nb_units_on` bounds divided by `p_max_unit*(1-spinning/100)` without
  parentheses).
- **Same interface**: the ids of the models, parameters (and their time and scenario
  dependencies), variables, ports and extra outputs are those of the external library, so
  that a system written for it only changes the library id of these models
  (`extended_models.<model>`). The models of the dropped list come from
  `antares_legacy_models.yml`, with its own ids.
- **Library id**: `extended_models`, to be loaded next to `antares_legacy_models` (its
  components connect to the `area` of that library).
- **Port type** `flow`: the one of `antares_legacy_models.yml` (fields `price`,
  `spillage_bound`, `unsupplied_energy_bound` and their `area-connection`). Every model
  defines the bounds like its counterpart in `antares_legacy_models.yml`, and the binding
  constraint models define `price` (`0`): Antares Modeler requires each field of a connection
  to be defined on exactly one side.
- **Compacted expressions**, with the same optimal solutions (checked by
  `tests/antares_historic/test_extended_models.py`):
    - Daily sums written with time sums on a range, `sum(0 .. 23, x)`, instead of
      `x[0] + ... + x[23]` (`daily_pump_max_<d>_w1`, `daily_turb_max_<d>_w1`, `bc_daily_<d>`).
    - Parameters, ports and port field definitions of the `bc_nuc_*` models shared with YAML
      anchors.
    - The two McCormick cuts of `short-term-storage` merged into `mccormick constraint`
      (`mccormick constraint_grand_maison` removed), exact for flags 0 or 1: the Grand'Maison
      cut is the usual one when `p_withdrawal_when_injection_max` is 0, and the usual one is
      tighter.
    - The curtailment limits of `enr` and `enr_multi_bandes` written as upper bounds of
      `enr_ecrete` and `enr_ecrete_<i>` (constraints `taux_ecretement` and
      `taux_ecretement_<i>` removed), exact for shares between 0 and 1.
- The parameters `startup_cost`, `fixed_cost`, `d_min_up` and `d_min_down` of `thermal_fast`
  are kept for the interface, though unused.
