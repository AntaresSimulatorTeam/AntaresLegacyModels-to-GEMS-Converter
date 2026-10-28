# Template parameters: what a template is instantiated for

The `template-parameters` of a conversion template define what the template is instantiated for. The template is instantiated once for each combination of the values of its parameters. In each instance, every placeholder (`${area}`, `${thermal}`, `${link}`, `${<ad-hoc parameter>}`…) is replaced by the corresponding value.

| Key | On | Effect |
|---|---|---|
| `name: area` / `name: link` | — | iterate over the areas / links of the study |
| `cluster-type` | `area` | iterate over the clusters of this type (`thermal`, `renewable`, `st_storage`) in each area; placeholder `${<type>}` |
| `values` | `area`, `link` | only these areas / links |
| `cluster-values` | `area` with `cluster-type` | only these clusters; `${area}` in an entry stands for the area of the cluster |
| `exclude` | `area`, `link` | skip these objects; applied after `values` / `cluster-values` |
| `ad-hoc: true` + `values` | any other name | iterate over `values`, with no lookup in the study |

## Where the templates are

The converter reads the templates from the package: `src/antares_gems_converter/input_converter/data/model_configuration/*.yaml`. To restrict or extend what a template converts, edit its file there (in a source checkout, or in the installed package). Only the models listed in `MODEL_NAME_TO_FILE_NAME` (`src/antares_gems_converter/input_converter/src/config.py`) can be converted, and only those named in `model_list` of the converter configuration are. A new template, such as the `electrolyser` of example 4, therefore needs a new YAML file and an entry in `MODEL_NAME_TO_FILE_NAME`; its `model` must exist in one of the libraries given to the converter.

## Rules

- Each template's lists and exclusions apply to that template only.
- Values are strings. Quote in YAML anything YAML would read as another type: numbers (`values: ["2030"]`), and `yes`, `no`, `on`, `off`, `true`, `false`, `null` (the area `no`, Norway: `values: [fr, "no"]`).
- Lists are non-empty and have no duplicates.
- Objects are iterated in study order, whatever the order of the list. Ad-hoc values are iterated in the order of the list.
- Link ids are the antares-craft ids: `"<area 1> / <area 2>"`, the areas sorted alphabetically (`"de / fr"`, not `"fr / de"`).
- A listed area or link that is not in the study, or a listed cluster that is in none of the listed areas, is an error. The conversion fails before anything is converted.
- An exclusion naming an object that is not in the study is not an error: it excludes nothing (a template can exclude `z_batteries` in a study without batteries). Check the spelling of the exclusions.
- Objects not converted (not listed, or excluded) are handled differently in the two modes:
  - **hybrid mode**: they stay legacy, simulated as before;
  - **full mode**: they are absent from the converted system. Leaving out an area also leaves out what its other templates would connect to it: restrict the templates consistently (for example, exclude the area in the link, thermal, load… templates too).
- Every placeholder of the template must be given by a parameter (`${stie}` instead of `${site}` is an error when the templates are loaded).

The examples below assume a study with:

| Area | Thermal clusters |
|---|---|
| `de` | `nuclear_1`, `de_ccgt` |
| `fr` | `nuclear_1`, `fr_ccgt`, `coal` |
| `it` | `it_ccgt` |

and the links `de / fr` and `fr / it`.

## 1. `exclude`: skip some objects

```yaml
template:
  name: thermal
  model: antares_legacy_models.thermal
  template-parameters:
    - name: area
      cluster-type: thermal
      exclude:
        - id: no_italy                 # an area, with its clusters and links
          object-properties: {type: area, area: it}
        - id: no_french_coal           # one cluster
          object-properties: {type: thermal, area: fr, cluster: coal}
        - id: no_ccgt                  # ${area}: the area of the cluster
          object-properties: {type: thermal, area: "${area}", cluster: "${area}_ccgt"}
  # components, connections, ... unchanged
```

Instances: `de` / `nuclear_1`, `fr` / `nuclear_1`.

An exclusion sets `type`, and only `area` (area), `link` (link), or `area` and `cluster` (cluster types). Its `type` is `area` or what the parameter iterates over (`link`, or the `cluster-type`): any other exclusion could never match and is rejected. It may use the placeholders of the outer parameters and, in a cluster exclusion only, `${area}`. In hybrid mode, the excluded clusters are not converted, so they are not deleted: they stay legacy.

## 2. `values` and `cluster-values`: convert only some objects

```yaml
template:
  name: thermal
  model: antares_legacy_models.thermal
  template-parameters:
    - name: area
      values: [fr, de]                              # only these areas
      cluster-type: thermal
      cluster-values: [nuclear_1, "${area}_ccgt"]   # only these clusters
      exclude:                                      # applied after the lists
        - id: no_german_nuclear
          object-properties: {type: thermal, area: de, cluster: nuclear_1}
```

Instances: `de` / `de_ccgt`, `fr` / `nuclear_1`, `fr` / `fr_ccgt`.

This works the same way for `renewable` and `st_storage`. In hybrid mode, `it_ccgt`, `coal` and `nuclear_1` of `de` are not converted and stay legacy.

## 3. `values` on links

```yaml
template:
  name: link
  model: antares_legacy_models.link
  template-parameters:
    - name: link
      values: ["de / fr"]
```

Instance: `de / fr` only. `fr / it` stays legacy in hybrid mode.

## 4. Ad-hoc parameter: iterate over a list unrelated to the study

```yaml
template:
  name: electrolyser
  model: andromede-v1-models.electrolyser
  template-parameters:
    - name: area                        # outer loop
      values: [fr, de]
    - name: site                        # inner loop, ${site}
      ad-hoc: true
      values: [north, "${area}_harbour"]
  components:
    - id: ${area}_electrolyser_${site}
      parameters: [...]                 # constants (external sources: see #129)
  area-connections:
    - component: ${area}_electrolyser_${site}
      port: power_port
      area: ${area}
```

Instances (nested loops, the first parameter outermost):

| `${area}` | `${site}` | component |
|---|---|---|
| `de` | `north` | `de_electrolyser_north` |
| `de` | `de_harbour` | `de_electrolyser_de_harbour` |
| `fr` | `north` | `fr_electrolyser_north` |
| `fr` | `fr_harbour` | `fr_electrolyser_fr_harbour` |

- **Placeholders in lists:** the entries of a list may use the placeholders of the parameters declared before it (`${area}` above).
- **Declaration order:** parameters can be declared in any order; the first one is the outermost loop. At most one parameter is not ad-hoc (`area` or `link`). A template may have only ad-hoc parameters; its `area-connections` then name the areas explicitly.
- **Ad-hoc parameter keys:** an ad-hoc parameter has no `cluster-type`, `cluster-values` or `exclude`. Its name is made of ASCII letters, digits and underscores (`region`, not `région`), and is not `area`, `link`, `thermal`, `renewable` or `st_storage`.
- **Used placeholder:** the placeholder of an ad-hoc parameter must be used in the template or in the lists of the parameters declared after it. Otherwise every value would give the same components, and the template is rejected.
- **No legacy object:** ad-hoc components replace no legacy object. The template cannot have `legacy-objects-to-delete`, and its components get no scenario group from the legacy scenario builder: a `scenario-group` of the template is ignored.
- **Not for legacy objects:** an ad-hoc parameter is meant for new components, not to split a legacy object. Adding one to e.g. the `thermal` template (`unit: ["1", "2"]`) drops the scenario groups of the clusters, and, since `legacy-objects-to-delete` is not allowed, in hybrid mode the legacy clusters stay in the study next to their converted components: they would be counted twice.
- **Unique ids:** component ids must be unique in the converted system, otherwise the conversion fails. In hybrid mode, they must not reuse the id of a legacy object still in the study.
