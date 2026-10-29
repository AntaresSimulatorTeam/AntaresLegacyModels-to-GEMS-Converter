# Spec: Xpansion thermal candidates in the converter

Draft v2, 2026-10-29. Compacted from the v1 draft of #164. Choices marked **(Q*n*)** wait on the open questions at the end.

## Goal

The converter should produce, in HYBRID mode, the same `system.yml` as the Xpansion direct generator, with no Xpansion-specific code in `src/`. Only three things are Xpansion-specific: a model library, the user's candidate templates and an `optim-config.yml`. Everything else is a generic converter feature.

## Constraints measured in the E2E runs (Antares 10.2.0, Xpansion 1.9.0)

| # | Rule | Converter today |
| --- | --- | --- |
| C1 | A candidate is a GEMS component linked by `thermal-capacity-connections` to an **enabled legacy** thermal cluster. The invested capacity **replaces** the cluster's availability; the cluster stays in the study | No such section in templates |
| C2 | Without a scenario group, Antares hybrid uses **column 1 for every MC year**; GemsPy uses year *i* → column *i* | Group cleared when the legacy SB is empty or partial (#143) |
| C3 | GEMS ids (components, groups, series): `[a-z0-9_]` only. Legacy cluster ids can hold spaces | Only `" / "` → `_` |
| C4 | Investment variables are scenario-independent: `benders-decomposition` with a model decomposition in `optim-config.yml`, or the master is empty | The input study's file is kept |
| C5 | Series in `.tsv`, one scenario group per component, every MC year mapped | Met, except "every year" (C2) |

Out of scope: batteries (no st-storage capacity connection in Antares 10.2), Excel inputs, year selection and the ch4 → h2 rule.

## Target template

One user template per technology, e.g. `thermal_candidate_semibase.yaml`:

```yaml
template:
  name: thermal_candidate_semibase
  model: xpansion_models.thermal_candidate
  template-parameters:
    - name: area
      cluster-type: thermal
      cluster-values: ["${area}_ccgt ch4 pcomp"]   # #153
  components:
    - id: ${area}_candidate_semibase
      scenario-group: thermal_${area}_${thermal}_group   # same mapping as the cluster (#158)
      parameters:
        - {id: unit_size, time-dependent: false, scenario-dependent: false, value: {constant: 300}}      # (Q3)
        - {id: max_unit, time-dependent: false, scenario-dependent: false, value: {constant: 10}}
        - {id: cost_per_year, time-dependent: false, scenario-dependent: false, value: {constant: 60000}}
        - id: cluster_availability
          time-dependent: true
          scenario-dependent: true
          value: {object-properties: {type: thermal, area: "${area}", cluster: "${thermal}", field: thermal_series}}
        - id: cluster_nominal_capacity   # object-properties, field nominal_capacity
        - id: cluster_unit_count         # object-properties, field unit_count
  thermal-capacity-connections:
    - component: ${area}_candidate_semibase
      port: capacity_port
      thermal-component: {area: "${area}", cluster-id: "${thermal}"}
```

Model (`xpansion_models.thermal_candidate`): port field `units_to_invest * unit_size * cluster_availability / (cluster_nominal_capacity * cluster_unit_count)`; bound `units_to_invest <= max_unit` (no `gisement_total`); cost `units_to_invest * cost_per_year` **(Q4)**. Antares 10.2 accepts this port field. The battery model is dropped.

## Work items

Generic converter changes, each its own PR:

1. **Complete scenario groups (#143 amended, C2).** A converted object always gets a fully mapped group: years with an explicit legacy entry keep it; years without one take the rule chosen in **(Q1)**. The group is never cleared, so Antares hybrid and GemsPy agree. Changes the results of existing hybrid conversions whose objects have a random legacy SB.
2. **Sanitise generated ids (C3).** Lowercase; anything outside `[a-z0-9_]` → `_`; applied to component ids, scenario groups and series ids, where `" / "` → `_` is applied today. Raw ids are kept in `object-properties` and connections. Collisions are already an error since #153. Applies to every template **(Q2)**.
3. **User template folder** (open question of #144 / #129). `[study] template_folder`: every `*.yaml` in it is a template, its `name` used in `models_to_convert`. Required here: candidate templates are study-specific.
4. **`thermal-capacity-connections` in templates (C1).** Same shape as `system.yml`, placeholders allowed, HYBRID only (load error in FULL). Checks:
   - at load (`checks.py`): the port type declares `thermal-capacity-connection`;
   - at merge: the target cluster is not in any template's `legacy-objects-to-delete` (e.g. `thermal.yaml` run on the same study);
   - at conversion: a cluster with `nominal_capacity × unit_count = 0` is skipped with a warning, like a disabled one.

   Also: when `values` lists the areas, a `cluster-values` entry missing in a listed area is a warning (today it is silent).

Xpansion deliverables, no converter code:

5. **Library** `libs/xpansion/xpansion_models.yml` (thermal model above), checked by `checks.py` like the others.
6. **User guide** `docs/xpansion-candidates.md`: example templates, the required `optim-config.yml` (C4), the `pcomp` placeholder clusters (C1).
7. **E2E test**: convert the test study of #164 with the converter, then compare it with the direct generator's output, and run `antares-problem-generator` (`structure.txt`, master bounds and costs). The Benders test stays opt-in (`ANTARES_XPANSION_DIR`) **(Q5)**.
8. **Retire the direct generator (Q6).**

Not needed: #129 (scalars are the same for every area of a technology, so they fit as constants **(Q3)**), and the #158 amendment of v1 (revised #158 infers the group from the `thermal_series` read).

## Open questions

- **Q1. Years with no legacy SB entry.** (a) column `(year mod N) + 1`, as the direct generator, with a warning if N > 1; (b) identity, error if N < number of MC years; (c) error unless N = 1. Note that a 1-column series makes all three agree.
- **Q2.** Sanitise ids for every template (changes the output ids of clusters with spaces, which Antares rejects today anyway), or only for new templates?
- **Q3.** Scalars as constants in the template (regenerated per year), or #129 `external-value` with a csv?
- **Q4.** `cost_per_year` per unit or per MW? Integer `units_to_invest`?
- **Q5.** Xpansion in CI (OpenMPI, release download), on every push, nightly, or never?
- **Q6.** Delete the direct generator, or turn it into a script that writes the candidate templates from the Excel inputs?
- **Q7.** `optim-config.yml`: user-supplied (docs only), or shipped next to the library and copied when a template uses `xpansion_models`?
