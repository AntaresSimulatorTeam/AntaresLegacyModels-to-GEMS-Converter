# Spec draft: Xpansion candidates in the converter

Draft, 2026-10-29; updated 2026-10-30 for the `antares_legacy_investment_candidates` library (#167).

The converter can cover the Xpansion direct generator once #144 (PR #153) and #129 land, plus two new pieces: a `thermal-capacity-connections` section in templates and id sanitisation of placeholders. `optim-config.yml` stays out of scope: the converter already keeps the one of the input study. Evidence: the E2E runs of the direct generator (Antares 10.2.0, Xpansion 1.9.0).

## Requirements from the experiment

Each line was a hard failure or a measured behaviour in the E2E runs, so the spec should state it as a rule.

| # | Requirement | Evidence |
| --- | --- | --- |
| R1 | A candidate is a GEMS component in HYBRID mode, connected by `thermal-capacity-connections` to an existing, enabled legacy cluster | Antares 10.2 has no other capacity connection; a disabled cluster stops the run (*Area has no thermal cluster named …*) |
| R2 | The candidate's port capacity **replaces** the cluster's own availability, it does not add to it. To keep the legacy capacity, the candidate sets `existing_capacity` to the cluster's installed capacity | fr CCGT produced 270 MW (invest only), not 270 + 90. With `existing_capacity`, the Xpansion optimum counts the legacy fr CCGT and OCGT capacity |
| R3 | The legacy cluster is kept: no `legacy-objects-to-delete` | Its dispatch variable carries the invested capacity |
| R4 | Data series are `.tsv` / `.csv` | `.txt` series are not loaded |
| R5 | One scenario group per component, every MC year mapped | A group shared by series of different column counts fails (*TS number 2 exceeds 1 column*) |
| R6 | Without a scenario group, Antares hybrid uses **column 1 for every MC year**, silently | 2-year run gave the 1-year optimum (2.6 units instead of 3.2). GemsPy maps year *i* to column *i* instead |
| R7 | Component ids, scenario groups and series ids: lowercase letters, digits, `_` | Antares id rule; legacy cluster ids such as `fr_ccgt ch4 pcomp` hold spaces |
| R8 | Investment variables are scenario-independent, so the study is solved with `benders-decomposition` (Xpansion, step `gems`) | `antares-solver` alone re-invests every week at the full annual cost |
| R9 | Batteries cannot be candidates yet | No st-storage capacity connection in Antares 10.2 or GemsPy 0.2 |
| R10 | Integer investment variables stay integer in the Benders master; bounds written as parameter expressions (`max_investment / unit_size`) are accepted | `master.mps` holds `INTORG` markers and `UI` bounds; Xpansion builds 1 OCGT unit where the continuous optimum is 0.6 |

## From script steps to converter features

Most steps map onto features that already exist or are specified; three are gaps.

| Script step | Converter feature | Status |
| --- | --- | --- |
| Zones from `List_countries.xlsx` | `cluster-values: ["${area}_ccgt ch4 pcomp"]` on the cluster parameter: areas without that cluster are simply not iterated. `values` on `area` if the list must be restricted further | #144, PR #153 open |
| Candidate type → cluster suffix (`correspondance_cluster_antares.xlsx`) | One template per technology (peak, semibase), the suffix written in its `cluster-values` | #144 |
| Scalars from `X_parameters_candidates.xlsx` (cost, unit size, units) | `external-value` (`table`, `key`, `field`) | #129. Gaps: xlsx tables, and a key on two columns (type, year) |
| `max_investment` = units × unit size (discrete) or `gisement_ini` (continuous) | `multiply_by: unit_size` on the unit count (parameter reference already exists) | Needs #129's "apply `operation` without `column`" |
| `annual_investment_cost` = cost per unit ÷ unit size (the library is per MW) | `divide_by: unit_size` | Same as above |
| Discrete model when the unit count is given, continuous otherwise | One template per technology and model | The choice moves to the template author |
| `availability_rate` = series ÷ (nominal capacity × unit count); `existing_capacity` = nominal capacity × unit count | No product of two fields today | **Gap**, see below |
| `thermal-capacity-connections` | Template section next to `area-connections` | **Gap**, new |
| Scenario group per candidate, from the legacy thermal SB | `read_year_ts()` already reads the thermal SB of the iterated cluster; the template declares `scenario-group: ..._${area}_${thermal}` | Exists; #143 for partial SBs; R6 changes the "no group" case |
| Skip empty, missing and disabled clusters; log them | Disabled clusters are skipped (#128); missing ones are not iterated | Exists. Gap: no report of the candidates not created |
| Library `antares_legacy_investment_candidates` | In `libs/antares_legacy_investment_candidates/` (#167), checked both ways against the templates (`checks.py`) | Done (#167) |
| 2031 only; ch4 until 2031, h2 after | Preparing the input tables | Out of the converter |

The library takes `availability_rate` as a fraction and `existing_capacity` in MW, so the converter must compute both from the cluster: a product of two fields (`nominal_capacity` × `unit_count`), and a series divided by it. The proposal is a derived `installed_capacity` field in `preprocessing.py`, usable as `object-properties` field and as `divide_by` operand. The alternative, the arithmetic in the model (raw series, `nominal_capacity` and `unit_count` as parameters; checked with Antares 10.2), would change the library's parameters.

## Template sketch

One template per technology, HYBRID mode only. Ids are shown after sanitisation (W2 below).

```yaml
template:
  name: thermal_candidate_semibase
  model: antares_legacy_investment_candidates.discrete_investment_candidate
  template-parameters:
    - name: area
      cluster-type: thermal
      cluster-values: ["${area}_ccgt ch4 pcomp"]      # #144
  components:
    - id: ${area}_candidate_semibase
      scenario-group: candidate_${area}_${thermal}_group   # legacy thermal SB of the cluster
      parameters:
        - id: unit_size
          time-dependent: false
          scenario-dependent: false
          value:
            external-value: {table: candidates_2031.csv, key: ccg_ch4, field: Pmax}   # #129
        - id: max_investment          # same, field nbr_unit_initial, multiply_by: unit_size
        - id: annual_investment_cost  # same, field cost_per_year, divide_by: unit_size
        - id: min_investment
          time-dependent: false
          scenario-dependent: false
          value: {constant: 0}
        - id: existing_capacity
          time-dependent: false
          scenario-dependent: false
          value:
            object-properties: {type: thermal, area: ${area}, cluster: ${thermal}, field: installed_capacity}   # new derived field
        - id: availability_rate
          time-dependent: true
          scenario-dependent: true
          value:
            object-properties: {type: thermal, area: ${area}, cluster: ${thermal}, field: thermal_series}
            operation: {divide_by: existing_capacity}
  thermal-capacity-connections:          # new (W1)
    - component: ${area}_candidate_semibase
      port: capacity_port
      thermal-component: {area: ${area}, cluster-id: ${thermal}}
  # no legacy-objects-to-delete: the cluster stays (R3)
```

The correspondence Excel file disappears (its content is the `cluster-values` of each template), and the zone list mostly does too: areas without the `pcomp` cluster are not iterated.

## Work items

New issues:

1. **W1 — `thermal-capacity-connections` in templates.** Same shape as `system.yml` (`component`, `port`, `thermal-component: {area, cluster-id}`), placeholders allowed. HYBRID only: rejected at load in FULL mode, where there is no legacy cluster. The connected cluster must be iterated and enabled (R1). `checks.py` checks that the port's type declares a `thermal-capacity-connection` field.
2. **W2 — Sanitise ids built from placeholders.** Generalise the `" / "` → `_` rule of link ids (AGENTS.md rule 7): lowercase, and anything outside `[a-z0-9_]` becomes `_`, for component ids, scenario groups and series ids (R7). `cluster-id` in a connection keeps the raw antares-craft id. Duplicate ids after sanitisation are an error.
3. **W3 — Report the candidates not created.** For each listed `cluster-values` entry, log the areas where it is absent, disabled or empty (the direct generator's log file is what users read).
4. **W4 — Derived `installed_capacity` field** (`nominal_capacity` × `unit_count`) for thermal clusters in `preprocessing.py`, for `existing_capacity` and the `availability_rate` division. (The library itself is #167.)
5. **W5 — Xpansion in CI.** Add `antares_xpansion_version` to `dependencies.json` and download the release in CI, so the Benders E2E test (now opt-in through `ANTARES_XPANSION_DIR`) runs on every push. It needs OpenMPI on the runner (`libopenmpi3`, `openmpi-bin`; as root, `OMPI_ALLOW_RUN_AS_ROOT=1` and `OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`).

Amendments to open issues:

- **#143** (partial SB): the "no explicit year → clear the scenario group" rule relies on GemsPy's identity mapping. Antares hybrid uses column 1 for every year instead (R6), so the same study gives different results in the two engines. Proposal: never clear the group; write the identity mapping explicitly (year *i* → column *i* + 1, error if the series has fewer columns than MC years).
- **#129** (external data): decide the *table formats* open question for xlsx, and allow a key on several columns (`key: {type: ccg_ch4, year: 2031}`), or keep both out and require a prepared csv.
- **#158** (ad-hoc parameters drop the SB): a pure configuration value such as `year: [2031]` would cost the candidates their scenario groups. Either let single-valued ad-hoc parameters keep the SB lookup, or add study-level variables in `config.ini` that templates read as placeholders.

## Out of scope and open questions

Out of scope:

- **`optim-config.yml`**: the user provides it in the input study; `optim_config.py` already keeps an existing file. Without it, the default (`models: []`) leaves the Benders master empty, so the docs should say it is required for candidate templates.
- **Batteries** (R9), until Antares supports a capacity connection to short-term storages.
- **Input preparation**: year selection, the ch4 → h2 rule, the candidate master table.

Open questions:

- [x] Is `cost_per_year` per unit or per MW? Per unit of `Pmax` in the inputs; the library takes `annual_investment_cost` per MW (as Xpansion's `annual-cost-per-mw`), so the generator divides it by `unit_size`.
- [x] Integer or continuous investment? Both: `discrete_investment_candidate` (integer `invested_units`) when the unit count is given, `continuous_investment_candidate` (`invested_capacity` in MW, up to `gisement_ini`) otherwise.
- [ ] `min_investment` is 0 for every candidate: should the inputs give it?
- [ ] Should W2 also apply to existing templates (renewable and storage ids can hold spaces too), at the cost of changing their output ids?
- [ ] The direct generator stays a separate script until W1–W4 land: keep it in `src/`, or move it to an `examples/` folder?
