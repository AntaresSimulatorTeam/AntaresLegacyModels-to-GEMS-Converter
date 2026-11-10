# Extended Models Library — Changelog

All notable changes to `extended_models.yml` are documented here. Its versioning follows
the rules of the Antares Legacy Models Library (see `COMPATIBILITY.md`).

---

## [1.0.0]

First version, from an external library also named `antares_legacy_models`.

- **Scope**: only the models that are not in `antares_legacy_models.yml`: `thermal_milp`,
  `thermal_fast`, `short-term-storage`, `enr`, `enr_multi_bandes`, `bc_nuc_hourly_less`,
  `bc_nuc_hourly_greater`, `bc_nuc_daily_less`, `bc_nuc_weekly_less` and
  `bc_nuc_weekly_greater`. Dropped, since `antares_legacy_models.yml` has them: `area`,
  `load`, `link`, `renewable` and `thermal_accurate` (`antares_legacy_models.thermal`; the
  external `nb_units_on` bounds divided by `p_max_unit*(1-spinning/100)` without
  parentheses).
- **Same interface**: the ids of the models, parameters (and their time and scenario
  dependencies, but for the constant parameters below), variables, ports and extra outputs
  are those of the external library, but for the renames of the taxonomy below. A system
  written for it changes the library id of these models (`extended_models.<model>`) and,
  for the models aligned with the taxonomy, the storage port and the properties. The models
  of the dropped list come from `antares_legacy_models.yml`, with its own ids.
- **Taxonomy**: three models implement a category of `antares_legacy_taxonomy` (and its
  flattened copy), so that GEMS-ViewsBuilder includes them in the area metrics:
    - `thermal_milp`, `thermal_fast`: `dispatchable_generation`. Variable `generation` renamed
      `generation_power`. Extra outputs as in `antares_legacy_models.thermal` (`prop_cost`,
      `non_prop_cost`, `actual_num_units_on`, `cluster_availability`, `min_gen_power`,
      `up_margin`, `profit`), and the 13 `<pollutant>_emissions` at 0, the models having no
      emission rate. In `thermal_fast`, which has no unit commitment, `non_prop_cost` is 0
      and `actual_num_units_on` is `ceil(generation_power / (p_max_unit * (1 - spinning / 100)))`.
    - `short-term-storage`: `short_term_storage`. Variables `p_injection` and `p_withdrawal`
      renamed `injection_power` and `withdrawal_power`, port `injection_port` renamed
      `balance_port` (connections must be updated), extra output `profit`. Its `level` is the
      level before the flows of the time step, unlike the legacy short-term storage: the
      `STS_LEVEL` metric adds both conventions.
    - Properties `technology` and `name`, which every component of these models must set
      (GemsPy requires them, antares-modeler does not check them).
    - The other models have no matching category and have none: a curtailment category for
      `enr` and `enr_multi_bandes` is to be discussed.
- **Library id**: `extended_models`, to be loaded next to `antares_legacy_models` (its
  components connect to the `area` of that library).
- **Port type** `flow`: the one of `antares_legacy_models.yml` (fields `price`,
  `spillage_bound`, `unsupplied_energy_bound` and their `area-connection`). Every model
  defines the bounds like its counterpart in `antares_legacy_models.yml`, and the binding
  constraint models define `price` (`0`): Antares Modeler requires each field of a connection
  to be defined on exactly one side.
- **Compacted expressions**, with the same optimal solutions (checked against the external
  library during the development of this version):
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
- `d_min_up`, `d_min_down` (`thermal_milp`, `thermal_fast`) and the `alpha_<cluster>` of the
  `bc_nuc_*` models are declared `time-dependent: false` and `scenario-dependent: false`
  (the external library left the defaults, time and scenario dependent). GemsPy rejects a
  time-dependent bound of a time sum (`d_min_up`, `d_min_down`), and with a time-dependent
  `alpha` it built each `bc_daily_<d>` once per time step (`alpha` is outside the time index
  of `(sum_connections(...))[h]`). Systems give these parameters as constants, so they are
  not affected.
