# Legacy variables → metrics correspondence table

As of 2026-11-18

## Purpose and reading rules

For each legacy output variable that can be selected by the thematic trimming, this table gives the metrics to add to each view config. It may be hard-coded in Antares or stored in a YAML file: the choice is still open.

- The key is the variable name as it appears in `select_var +` and `select_var -` in `generaldata.ini`, compared case-insensitively.
- A variable can give several metrics, in one or several views.
- A metric is referenced by its id in the view's catalog; the view config writes it as `<catalog>.<metric>`.
- A variable with no metric is not in the tables; it is listed separately, at the end of this document.
- The unit is the one of the legacy variable. It comes from the [Antares outputs reference](https://antares-doc.readthedocs.io/en/latest/reference/outputs/), which is sometimes wrong: corrected units are flagged in the Remark column.

The catalog of each metric is the one of the view config template: [`antares_legacy_area_catalog`](../catalogs/antares_legacy_area_catalog.yml) for [`view_area`](view_config_area.yml), [`antares_legacy_link_catalog`](../catalogs/antares_legacy_link_catalog.yml) for [`view_link`](view_config_link.yml), [`antares_legacy_thermal_cluster_catalog`](../catalogs/antares_legacy_thermal_cluster_catalog.yml) for [`view_thermal`](view_config_thermal.yml), [`antares_legacy_short_term_storage_catalog`](../catalogs/antares_legacy_short_term_storage_catalog.yml) for [`view_storage`](view_config_sts.yml), [`antares_legacy_renewable_cluster_catalog`](../catalogs/antares_legacy_renewable_cluster_catalog.yml) for [`view_renewable`](view_config_renewable.yml).

## Area view (`view_area`)

44 legacy variables feed the 43 metrics of the area catalog; four variables share `RES_GENERATION`.

| Legacy variable | Metric(s) | Unit | Remark |
| --- | --- | --- | --- |
| `OV. COST` | `OV_COST` | Euro |  |
| `OP. COST` | `OP_COST` | Euro |  |
| `MRG. PRICE` | `MRG_PRICE` | Euro/MWh | The doc wrongly gives Euro |
| `CO2 EMIS.`, `NH3 EMIS.`, `SO2 EMIS.`, `NOX EMIS.`, `PM2_5 EMIS.`, `PM5 EMIS.`, `PM10 EMIS.`, `NMVOC EMIS.`, `OP1 EMIS.` … `OP5 EMIS.` | `<POLLUTANT>_EMIS` (`CO2_EMIS` … `OP5_EMIS`) | Tons | One entry per pollutant |
| `BALANCE` | `BALANCE` | MWh |  |
| `ROW BAL.` | `ROW_BAL` | MWh |  |
| `PSP` | `PSP` | MWh |  |
| `MISC. NDG` | `MISC_NDG` | MWh |  |
| `LOAD` | `LOAD` | MWh |  |
| `RES LOAD` | `RES_LOAD` | MWh |  |
| `Dispatch. Gen.` | `DTG_BY_TECHNOLOGY` | MWh | Detailed by thermal group (`technology` property) |
| `Renewable Gen.` | `RES_GENERATION` | MWh | Renewable mode "clusters" |
| `WIND`, `SOLAR` | `RES_GENERATION` | MWh | Mode "aggregated"; the metric outputs all technologies, not only the selected one |
| `H. ROR` | `RES_GENERATION` | MWh | Technology `run_of_river`; same caveat |
| `H. STOR` | `H_STOR` | MWh |  |
| `H. PUMP` | `H_PUMP` | MWh |  |
| `H. LEV` | `H_LEV` | % | Percentage of the reservoir size; the doc wrongly gives MWh |
| `H. INFL` | `H_INFL` | MWh |  |
| `H. OVFL` | `H_OVFL` | MWh | The doc wrongly gives % |
| `H. VAL` | `H_VAL` | Euro/MWh | Flagged "different from the one in Antares" in the catalog |
| `H. COST` | `H_COST` | Euro | Same |
| `STS by group` | `STS_INJECTION`, `STS_WITHDRAWAL`, `STS_LEVEL` | MWh | Detailed by storage group; unit not given in the doc |
| `UNSP. ENRG` | `UNSP_ENRG` | MWh |  |
| `SPIL. ENRG` | `SPIL_ENRG` | MWh | Not given in the doc |
| `LOLD` | `LOLD` | h |  |
| `LOLP` | `LOLP` | % | The catalog averages a 0/1 indicator: it gives a fraction, not a percentage |
| `NPCAP HOURS` | `NPCAP_HOURS` | h |  |
| `AVL DTG` | `AVL_DTG` | MWh |  |
| `DTG MRG` | `DTG_MRG` | MWh |  |
| `NP COST` | `NP_COST` | Euro |  |
| `NODU` | `NODU` | None | Number of thermal units on |

## Link view (`view_link`)

The 9 metrics of the link catalog cover 9 of the 10 legacy link variables; only `FLOW QUAD.` has no metric (deprecated).

| Legacy variable | Metric | Unit | Remark |
| --- | --- | --- | --- |
| `FLOW LIN.` | `FLOW_LIN` | MWh |  |
| `UCAP LIN.` | `UCAP_LIN` | MWh |  |
| `LOOP FLOW` | `LOOP_FLOW` | MWh |  |
| `CONG. FEE (ALG.)` | `CONG_FEE_ALG` | Euro |  |
| `CONG. FEE (ABS.)` | `CONG_FEE_ABS` | Euro |  |
| `MARG. COST` | `MARG_COST` | Euro/MW |  |
| `CONG. PROB +` | `CONG_PROB_DIR` | % | The catalog sums a 0/1 indicator over time: it gives a number of hours, not a probability |
| `CONG. PROB -` | `CONG_PROB_INDIR` | % | Same |
| `HURDLE COST` | `HURDLE_COST` | Euro |  |

## Per-cluster views: thermal, storage, renewable

These views use the area as location and detail each metric by cluster (`name` property).

The detail by `name` requires the system for views to write this property on thermal clusters and storages, which is not the case yet: see [AntaresSimulatorTeam/Antares_Simulator#4001](https://github.com/AntaresSimulatorTeam/Antares_Simulator/issues/4001).

| View | Legacy variable | Metric | Unit | Remark |
| --- | --- | --- | --- | --- |
| `view_thermal` | `DTG by plant` | `GENERATION` | MWh |  |
| `view_thermal` | `MIN DTG by plant` | `MIN_GEN` | MWh |  |
| `view_thermal` | `Profit by plant` | `PROFIT` | Euro | No unit column in the doc; the description says "in euros" |
| `view_thermal` | `NODU by plant` | `NODU` | None | Number of units on |
| `view_thermal` | `NP Cost by plant` | `NP_COST` | Euro |  |
| `view_storage` | `STS inj by plant` | `P_INJECTION` | MWh | Not given in the doc |
| `view_storage` | `STS withdrawal by plant` | `P_WITHDRAWAL` | MWh | Not given in the doc |
| `view_storage` | `STS lvl by plant` | `LEVELS` | MWh | Not given in the doc |
| `view_storage` | `STS Cashflow By Cluster` | `CASHFLOW` | Euro | Antares sums it over time; the catalog averaged it, fixed in [AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter#136](https://github.com/AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter/pull/136) |
| `view_renewable` | `RES generation by plant` | `GENERATION` | MWh | Run of river gives a `run_of_river` row next to the renewable clusters; `name` property of renewables: see [AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter#134](https://github.com/AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter/issues/134) |

## Legacy variables without a metric

These variables have no metric in the catalogs yet: for each one, the metric must be created or the variable excluded from V1.

| Legacy variable | Target view | Unit | Nature |
| --- | --- | --- | --- |
| `OV. COST CSR`, `MRG. PRICE CSR`, `UNSP. ENRG CSR`, `LOLD CSR`, `LOLP CSR`, `DTG MRG CSR`, `MAX MRG CSR` | `view_area` | Euro, Euro/MWh, MWh, h, %, MWh, MWh (in this order) | Adequacy patch (CSR) |
| `DENS`, `LMR VIOL.` | `view_area` | MWh, None | Adequacy patch |
| `MAX MRG` | `view_area` | MWh | Antares-specific computation, to be handled as post-processing (TODO already noted in the catalog) |
| `RAMP COST` | `view_area` | Euro | Thermal ramping |
| `RAMP COST BY PLANT` | `view_thermal` | Euro | Thermal ramping |
| `FLOW QUAD.` | `view_link` | MWh | Deprecated: excluded from V1 |
| `RESERVE PARTICIPATION COST` and the other reserve variables | To be defined | Not in the doc | Reserves |
| `BC. MARG. COST` | None | Euro/Mwh | Binding constraints, out of scope |

## Open points

- [ ] Hard-code the table in Antares or store it in a YAML file (in that case, next to the catalogs, in the GEMS repo or the converter).
- [ ] Create, or exclude from V1, each variable without a metric.
- [ ] Fix the `LOLP`, `CONG_PROB_DIR` and `CONG_PROB_INDIR` metrics so they give a percentage, like Antares.
