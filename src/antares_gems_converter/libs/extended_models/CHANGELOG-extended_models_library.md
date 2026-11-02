# Extended Models Library — Changelog

All notable changes to `extended_models.yml` are documented here. Its
versioning follows the rules of the Antares Legacy Models Library (see `COMPATIBILITY.md`).

---

## [1.0.0]

First version, from an external library also named `antares_legacy_models` (kept as test
data: `tests/antares_historic/data/reference_libraries/antares_legacy_models_external.yml`).

- **Scope**: only the models that are not in `antares_legacy_models.yml`. Dropped, since
  `antares_legacy_models.yml` has them: `area`, `load`, `link`, `renewable` and
  `thermal_accurate` (`antares_legacy_models.thermal`; the external `nb_units_on` bounds
  divided by `p_max_unit*(1-spinning/100)` without parentheses).
- **Library id**: `extended_models`, to be loaded next to
  `antares_legacy_models` (its components connect to the `area` of that library).
- **Port type** `flow`: the one of `antares_legacy_models.yml` (fields `price`,
  `spillage_bound`, `unsupplied_energy_bound` and their `area-connection`). Every model
  defines the bounds like its counterpart in `antares_legacy_models.yml`, and the binding
  constraint models define `price` (`0`): Antares Modeler requires each field of a connection
  to be defined on exactly one side.
- **Ids** renamed after `antares_legacy_models.yml` and the naming rules of `AGENTS.md`
  (table below).
- **Compacted expressions**, with the same optimal solutions (checked by
  `tests/antares_historic/test_extended_models.py`):
    - Daily sums written with time sums on a range, `sum(0 .. 23, x)`, instead of
      `x[0] + ... + x[23]` (cycle limits of the storage, daily binding constraint). With the
      merges below, the library goes from about 2,580 to 520 lines.
    - `bc_nuc_<time step>_less` and `bc_nuc_<time step>_greater` merged into
      `nuclear_<time step>_binding_constraint`, with the parameter `direction`
      (`1`: less, `-1`: greater); parameters and ports shared with YAML anchors.
    - The two McCormick cuts of the storage merged into one: the usual cut is the Grand'Maison
      cut with `withdrawal_power_at_max_injection = 0`.
    - Curtailment limits written as variable upper bounds instead of constraints (they were
      tighter than the bounds, the shares being between 0 and 1).
    - `thermal_fast`: unused parameters removed (`startup_cost`, `fixed_cost`, `d_min_up`,
      `d_min_down`).

### Renamed models and parameters

| External library | `extended_models` |
|---|---|
| `thermal_milp`, `thermal_fast` | `thermal_milp`, `thermal_fast` |
| `minimum_generation_modulation`, `p_max_cluster` | `cluster_min_gen_modulation`, `cluster_max_generation` |
| `p_min_unit`, `p_max_unit`, `unit_count` | `min_power_per_unit`, `max_power_per_unit`, `num_units` |
| `d_min_up`, `d_min_down` | `min_up_duration`, `min_down_duration` |
| `generation`, `nb_units_on`, `nb_starting`, `nb_stopping`, `nb_failing` | `generation_power`, `num_units_on`, `num_units_starting`, `num_units_stopping`, `num_units_failing` |
| `short-term-storage` | `short_term_storage_extended` (port `injection_port` → `balance_port`) |
| `injection_nominal_capacity`, `withdrawal_nominal_capacity` | `nominal_injection_capacity`, `nominal_withdrawal_capacity` |
| `efficiency_injection`, `efficiency_withdrawal` | `injection_efficiency`, `withdrawal_efficiency` |
| `p_max_injection_modulation`, `p_max_withdrawal_modulation` | `max_injection_modulation`, `max_withdrawal_modulation` |
| `cost_injection`, `cost_withdrawal`, `cost_level` | `injection_cost`, `withdrawal_cost`, `level_cost` |
| `p_t_cost` | `hurdle_cost` |
| `mccormick`, `mccormick_grand_maison`, `p_withdrawal_when_injection_max` | `is_mccormick_enabled` = max(`mccormick`, `mccormick_grand_maison`), `withdrawal_power_at_max_injection` = 0 if `mccormick` else `p_withdrawal_when_injection_max` |
| `cycle_constraint`, `nb_cycles_max` | `is_cycle_limited`, `max_daily_cycles` |
| `p_injection`, `p_withdrawal` | `injection_power`, `withdrawal_power` |
| `enr` | `curtailable_renewable` |
| `enr_total`, `enr_capacity`, `taux_non_ecretable` | `load_factor`, `installed_capacity`, `non_curtailable_share` |
| `enr_ecrete`; `enr_non_ecretable`, `enr_non_ecrete` | `curtailed_power`; `non_curtailable_power`, `non_curtailed_power` |
| `enr_multi_bandes` | `multi_band_curtailable_renewable` |
| `enr_fc_total`, `taux_ecretable_<i>`, `enr_ecrete_<i>` | `load_factor`, `curtailable_share_<i>`, `curtailed_power_<i>` |
| `enr_ecrete_total`, `enr_prod_total` | `total_curtailed_power`, `generation_power` |
| `bc_nuc_hourly_less`, `bc_nuc_hourly_greater` | `nuclear_hourly_binding_constraint`, `direction` 1 and -1 |
| `bc_nuc_daily_less` | `nuclear_daily_binding_constraint`, `direction` 1 |
| `bc_nuc_weekly_less`, `bc_nuc_weekly_greater` | `nuclear_weekly_binding_constraint`, `direction` 1 and -1 |

The other ids are unchanged (`generation_cost`, `startup_cost`, `fixed_cost`, `spinning`,
`reservoir_capacity`, rule curves, `inflows`, `initial_level`, `level`, the `alpha_<cluster>`
and `rhs` parameters and the `flow_<cluster>` ports of the binding constraints).
