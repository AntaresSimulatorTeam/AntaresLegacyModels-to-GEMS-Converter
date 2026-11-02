# Scenario groups: which time series column each Monte Carlo year uses

In GEMS, a component whose parameters have several time series columns (`scenario-dependent: true`) belongs to a **scenario group**. The file `input/data-series/modeler-scenariobuilder.dat` gives, for each group and Monte Carlo year, the column to use: one line `<group>, <year> = <column>` per year. All the components of a group use the same column in a given year.

In Antares, the same choice is made by the **legacy scenario builder** of the study: for each object (a thermal cluster, the load of an area, a link…) and each year, the series to use, or nothing for a random draw.

The converter turns the legacy scenario builder into scenario groups. This page explains how it decides which group a component gets.

## The rule: the group follows the data the component reads

A template declares a group with `scenario-group` on a component, e.g. `thermal_${area}_${thermal}_group`. For each such component, after the placeholders are replaced, the converter looks at the parameters marked `scenario-dependent: true` and at the legacy objects they read (`object-properties`). Only the legacy objects that have an entry in the legacy scenario builder count:

| `object-properties` `type` | Entry of the legacy scenario builder |
|---|---|
| `thermal` | the thermal cluster (`area`, `cluster`) |
| `renewable` | the renewable cluster (`area`, `cluster`) |
| `load`, `solar`, `wind` | the load, solar or wind of the `area` |
| `hydro` (any field: inflows, ror, max power…) | the hydro of the `area` |
| `link` | the `link` |

Other types (`st_storage`, `misc_gen`, `area`, `binding_constraint`) and constants have no entry: they don't count. Parameters that are not scenario-dependent have a single column, so they don't count either.

Several fields of the same object count as one object: the `hydro` template reads `maxpower`, `mingen`, `mod_inflows`… of the same area, so its source is the hydro of that area.

Then, according to the number of legacy objects found:

| Legacy objects read | Scenario group of the component |
|---|---|
| **one** | that object's entries in the legacy scenario builder. The years with a random draw are left out. If every year is random, the component gets no group. |
| **several** | ambiguous, so the conversion fails with a `ValueError` naming the template, the component and the objects. |
| **none**, e.g. a new component with constant parameters | there is nothing to convert. The group is kept only if you give your own `modeler-scenariobuilder.dat` (`modeler_scenario_builder_file` of `AntaresStudyConverter`) and it defines the group; if the file doesn't define it, the conversion fails with a `ValueError`. Without such a file, the group is dropped with a warning. |

The rule is the same whatever the template parameters: with or without ad-hoc parameters, and whether or not the template deletes legacy objects in hybrid mode.

### Shared groups

Several components get the same group when their `scenario-group`, once the placeholders are replaced, gives the same name. This happens when the name does not use every placeholder of the loop. With `thermal_group`, every thermal component of the template shares one group. With `thermal_${area}_group`, the clusters of an area share one. It also happens across templates, e.g. a thermal and a renewable template both declaring `res_group`.

The components of a group use the same column in every year, so they must get the same entries from the legacy scenario builder. Each component's entries are read from its own legacy object, then compared with those of the components already in the group:

| Entries of the component's legacy object | Result |
|---|---|
| the same explicit years, with the same series, as the group | the component joins the group |
| different explicit years or series | the conversion fails with a `ValueError` naming the group, its components so far and this component. It stops at the first such component, before `system.yml` is written. |
| every year random | the component gets no group (as for a single object) and is not compared |

Only the explicit years are compared. Two objects with the same explicit years that are both random in another year are accepted in one group, but they then use the same column in that year, where Antares draws each one separately. To keep Antares' behaviour, give such components their own group, i.e. use every placeholder of the loop in the name.

### Group names

The name only labels the group: the entries always come from the legacy object the component reads, whatever the name. A thermal component named `foo_${area}_group` that reads `fr/gaz` gets the group `foo_fr_group`, with the entries of `fr/gaz`. The name only matters when it is the same for several components (see above) and when you give your own `modeler-scenariobuilder.dat` (see below).

The `" / "` of link ids (`at / fr`) is replaced by `_` in group names, as in component ids: the group `${link}_ntc_group` of link `at / fr` is `at_fr_ntc_group`.

## Output file

- Without a user file, the converter writes `modeler-scenariobuilder.dat` from the groups above, if there is at least one.
- With a user file, it is copied as is, and nothing is generated. The converter only checks it for the groups with no legacy source (see above). A group that does have a source is set from the legacy scenario builder, as without a file: make sure your file defines it, or GemsPy fails at run time.

In hybrid mode, the legacy scenario builder entries of the load, solar and wind of an area are cleared when that object is converted (blanked) in the study.

## Examples

**Reference templates.** Each component with a `scenario-group` reads a single legacy object, so each gets that object's entries:

| Template | Group | Legacy object read |
|---|---|---|
| thermal | `thermal_${area}_${thermal}_group` | the thermal cluster |
| renewable | `renewable_${area}_${renewable}_group` | the renewable cluster |
| load, solar, wind | `load_${area}_group`, … | the load, solar or wind of the area |
| hydro | `hydro_inflows_${area}_group` | the hydro of the area |
| ror | `run_of_river_${area}_group` | the hydro of the area |
| link | `${link}_ntc_group` | the link |

The other reference templates (short-term storage, battery, misc gen, area) declare no group.

**A cluster split by an ad-hoc parameter.** With an ad-hoc `unit: ["1", "2"]` added to the thermal template, the components `fr_thermal_gaz_1` and `fr_thermal_gaz_2` both read the series of `fr/gaz`. Both get the group `thermal_fr_gaz_group`, with the entries of `fr/gaz`. See [template-parameters.md](template-parameters.md) for what else a split implies in hybrid mode.

**A group name that does not depend on the cluster.** With `scenario-group: thermal_${area}_group` in the thermal template, the clusters `gaz` and `coal` of `fr` are both in `thermal_fr_group`, and the result depends on their series in the legacy scenario builder, here for 3 Monte Carlo years:

| `fr/gaz` | `fr/coal` | Result |
|---|---|---|
| 2, 1, random | 2, 1, random | both in `thermal_fr_group`, with series 2 and 1 in the first two years. The third year is random for both, and both use the same column in it. |
| 2, 1, random | 3, 1, random | `ValueError`: the first year differs |
| 2, 1, random | 2, 1, 4 | `ValueError`: the third year is explicit for `coal` only |
| 2, 1, random | random, random, random | `fr_thermal_gaz` in `thermal_fr_group`, `fr_thermal_coal` with no group |

**A new component.** The electrolyser of [example 4](template-parameters.md#4-ad-hoc-parameter-iterate-over-a-list-unrelated-to-the-study) only has constant parameters. If it declared a `scenario-group`, the group would be dropped with a warning, unless your own `modeler-scenariobuilder.dat` defines it.
