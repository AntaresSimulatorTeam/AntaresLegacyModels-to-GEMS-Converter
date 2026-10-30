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

Several components can declare the same group: they then share its columns. This is only allowed if they get the same entries from the legacy scenario builder. Otherwise the conversion fails with a `ValueError` naming the group and its components. For example, a template giving every thermal cluster of an area the group `thermal_${area}_group` fails as soon as two clusters of that area have different entries.

### Group names

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

**A new component.** The electrolyser of [example 4](template-parameters.md#4-ad-hoc-parameter-iterate-over-a-list-unrelated-to-the-study) only has constant parameters. If it declared a `scenario-group`, the group would be dropped with a warning, unless your own `modeler-scenariobuilder.dat` defines it.
