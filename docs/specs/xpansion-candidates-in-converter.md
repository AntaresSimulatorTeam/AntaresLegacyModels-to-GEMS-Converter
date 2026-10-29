# Spec draft: Xpansion candidates in the converter

Draft, 2026-10-29.

The converter can cover the Xpansion direct generator once #144 (PR #153) and #129 land, plus two new pieces: a `thermal-capacity-connections` section in templates and id sanitisation of placeholders. `optim-config.yml` stays out of scope: the converter already keeps the one of the input study. Evidence: the E2E runs of the direct generator (Antares 10.2.0, Xpansion 1.9.0).

## Requirements from the experiment

Each line was a hard failure or a measured behaviour in the E2E runs, so the spec should state it as a rule.

| # | Requirement | Evidence |
| --- | --- | --- |
| R1 | A candidate is a GEMS component in HYBRID mode, connected by `thermal-capacity-connections` to an existing, enabled legacy cluster | Antares 10.2 has no other capacity connection; a disabled cluster stops the run (*Area has no thermal cluster named …*) |
| R2 | The invested capacity **replaces** the cluster's own availability, it does not add to it | fr CCGT produced 270 MW (invest only), not 270 + 90. The base study must hold placeholder `pcomp` clusters |
| R3 | The legacy cluster is kept: no `legacy-objects-to-delete` | Its dispatch variable carries the invested capacity |
| R4 | Data series are `.tsv` / `.csv` | `.txt` series are not loaded |
| R5 | One scenario group per component, every MC year mapped | A group shared by series of different column counts fails (*TS number 2 exceeds 1 column*) |
| R6 | Without a scenario group, Antares hybrid uses **column 1 for every MC year**, silently | 2-year run gave the 1-year optimum (2.6 units instead of 3.2). GemsPy maps year *i* to column *i* instead |
| R7 | Component ids, scenario groups and series ids: lowercase letters, digits, `_` | Antares id rule; legacy cluster ids such as `fr_ccgt ch4 pcomp` hold spaces |
| R8 | Investment variables are scenario-independent, so the study is solved with `benders-decomposition` (Xpansion, step `gems`) | `antares-solver` alone re-invests every week at the full annual cost |
| R9 | Batteries cannot be candidates yet | No st-storage capacity connection in Antares 10.2 or GemsPy 0.2 |

## From script steps to converter features

Most steps map onto features that already exist or are specified; three are gaps.

| Script step | Converter feature | Status |
| --- | --- | --- |
| Zones from `List_countries.xlsx` | `cluster-values: ["${area}_ccgt ch4 pcomp"]` on the cluster parameter: areas without that cluster are simply not iterated. `values` on `area` if the list must be restricted further | #144, PR #153 open |
| Candidate type → cluster suffix (`correspondance_cluster_antares.xlsx`) | One template per technology (peak, semibase), the suffix written in its `cluster-values` | #144 |
| Scalars from `X_parameters_candidates.xlsx` (cost, unit size, units) | `external-value` (`table`, `key`, `field`) | #129. Gaps: xlsx tables, and a key on two columns (type, year) |
| `gisement_total` = units × unit size | `multiply_by: max_unit` (parameter reference already exists) | Needs #129's "apply `operation` without `column`". Simpler: drop `gisement_total`, it repeats the `max_unit` bound |
| Availability = series ÷ (nominal capacity × unit count) | No product of two fields today | **Gap**, see below |
| `thermal-capacity-connections` | Template section next to `area-connections` | **Gap**, new |
| Scenario group per candidate, from the legacy thermal SB | `read_year_ts()` already reads the thermal SB of the iterated cluster; the template declares `scenario-group: ..._${area}_${thermal}` | Exists; #143 for partial SBs; R6 changes the "no group" case |
| Skip empty, missing and disabled clusters; log them | Disabled clusters are skipped (#128); missing ones are not iterated | Exists. Gap: no report of the candidates not created |
| `xpansion_models.yml` | A library in `libs/`, checked both ways against the templates (`checks.py`) | Move it; rename `Pmax` (R7) |
| 2031 only; ch4 until 2031, h2 after | Preparing the input tables | Out of the converter |

For the availability coefficient, the simplest fix is to put the arithmetic in the model rather than in the converter. The model takes the raw cluster series and the cluster's `nominal_capacity` and `unit_count` as parameters (all plain `object-properties`), and its port field becomes `units_to_invest * unit_size * cluster_availability / (cluster_nominal_capacity * cluster_unit_count)`. Checked with Antares 10.2: such a port field is accepted, and the capacity constraints get the expected coefficients. The alternative is a derived `installed_capacity` field in `preprocessing.py`.

## Template sketch

One template per technology, HYBRID mode only. Ids are shown after sanitisation (W2 below).

```yaml
template:
  name: thermal_candidate_semibase
  model: xpansion_models.thermal_candidate_to_invest
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
        - id: max_unit            # same, field nbr_unit_initial
        - id: cost_per_year       # same, field cost_per_year
        - id: cluster_availability
          time-dependent: true
          scenario-dependent: true
          value:
            object-properties: {type: thermal, area: ${area}, cluster: ${thermal}, field: thermal_series}
        - id: cluster_nominal_capacity   # object-properties, field nominal_capacity
        - id: cluster_unit_count         # object-properties, field unit_count
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
4. **W4 — Library `xpansion_models.yml` in `libs/`**, with the availability arithmetic moved into the model, `gisement_total` dropped and `Pmax` renamed.
5. **W5 — Xpansion in CI.** Add `antares_xpansion_version` to `dependencies.json` and download the release in CI, so the Benders E2E test (now opt-in through `ANTARES_XPANSION_DIR`) runs on every push. It needs OpenMPI on the runner.

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

- [ ] Is `cost_per_year` per unit or per MW? It decides whether the model multiplies it by `unit_size`.
- [ ] Integer `units_to_invest`, or continuous as today?
- [ ] Should W2 also apply to existing templates (renewable and storage ids can hold spaces too), at the cost of changing their output ids?
- [ ] The direct generator stays a separate script until W1–W4 land: keep it in `src/`, or move it to an `examples/` folder?
