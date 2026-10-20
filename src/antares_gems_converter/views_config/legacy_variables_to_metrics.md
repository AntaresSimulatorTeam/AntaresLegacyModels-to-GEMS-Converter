# Legacy variables → metrics correspondence table

As of 2026-10-19

## Purpose and reading rules

For each legacy output variable that can be selected by the thematic trimming, this table gives the metrics to add to each view config. It may be hard-coded in Antares or stored in a YAML file: the choice is still open.

- The key is the variable name as it appears in `select_var +` and `select_var -` in `generaldata.ini`, compared case-insensitively.
- A variable can give several metrics, in one or several views.
- A metric is referenced by its id in the view's catalog; the view config writes it as `<catalog>.<metric>`.
- A variable with no metric is not in the tables; it is listed separately, at the end of this document.

The catalog of each metric is the one of the view config template: [`antares_legacy_area_catalog`](../catalogs/antares_legacy_area_catalog.yml) for [`view_area`](view_config_area.yml), [`antares_legacy_link_catalog`](../catalogs/antares_legacy_link_catalog.yml) for [`view_link`](view_config_link.yml), [`antares_legacy_thermal_cluster_catalog`](../catalogs/antares_legacy_thermal_cluster_catalog.yml) for [`view_thermal`](view_config_thermal.yml), [`antares_legacy_short_term_storage_catalog`](../catalogs/antares_legacy_short_term_storage_catalog.yml) for [`view_storage`](view_config_sts.yml).

## Area view (`view_area`)

44 legacy variables feed the 43 metrics of the area catalog; four variables share `RES_GENERATION`.

| Legacy variable | Metric(s) | Remark |
| --- | --- | --- |
| `OV. COST` | `OV_COST` |  |
| `OP. COST` | `OP_COST` |  |
| `MRG. PRICE` | `MRG_PRICE` |  |
| `CO2 EMIS.`, `NH3 EMIS.`, `SO2 EMIS.`, `NOX EMIS.`, `PM2_5 EMIS.`, `PM5 EMIS.`, `PM10 EMIS.`, `NMVOC EMIS.`, `OP1 EMIS.` … `OP5 EMIS.` | `<POLLUTANT>_EMIS` (`CO2_EMIS` … `OP5_EMIS`) | One entry per pollutant |
| `BALANCE` | `BALANCE` |  |
| `ROW BAL.` | `ROW_BAL` | Same term as `MISC_NDG` and `PSP` in the catalog: to be checked |
| `PSP` | `PSP` | Same |
| `MISC. NDG` | `MISC_NDG` | Same |
| `LOAD` | `LOAD` |  |
| `RES LOAD` | `RES_LOAD` |  |
| `Dispatch. Gen.` | `DTG_BY_TECHNOLOGY` | Detailed by thermal group (`technology` property) |
| `Renewable Gen.` | `RES_GENERATION` | Renewable mode "clusters" |
| `WIND`, `SOLAR` | `RES_GENERATION` | Mode "aggregated"; the metric outputs all technologies, not only the selected one |
| `H. ROR` | `RES_GENERATION` | Technology `run_of_river`; same caveat |
| `H. STOR` | `H_STOR` |  |
| `H. PUMP` | `H_PUMP` |  |
| `H. LEV` | `H_LEV` |  |
| `H. INFL` | `H_INFL` |  |
| `H. OVFL` | `H_OVFL` |  |
| `H. VAL` | `H_VAL` | Flagged "different from the one in Antares" in the catalog |
| `H. COST` | `H_COST` | Same |
| `STS by group` | `STS_INJECTION`, `STS_WITHDRAWAL`, `STS_LEVEL` | Detailed by storage group |
| `UNSP. ENRG` | `UNSP_ENRG` |  |
| `SPIL. ENRG` | `SPIL_ENRG` |  |
| `LOLD` | `LOLD` |  |
| `LOLP` | `LOLP` |  |
| `NPCAP HOURS` | `NPCAP_HOURS` |  |
| `AVL DTG` | `AVL_DTG` |  |
| `DTG MRG` | `DTG_MRG` |  |
| `NP COST` | `NP_COST` |  |
| `NODU` | `NODU` |  |

## Link view (`view_link`)

The 9 metrics of the link catalog cover 9 of the 10 legacy link variables; only `FLOW QUAD.` has no metric (deprecated).

| Legacy variable | Metric |
| --- | --- |
| `FLOW LIN.` | `FLOW_LIN` |
| `UCAP LIN.` | `UCAP_LIN` |
| `LOOP FLOW` | `LOOP_FLOW` |
| `CONG. FEE (ALG.)` | `CONG_FEE_ALG` |
| `CONG. FEE (ABS.)` | `CONG_FEE_ABS` |
| `MARG. COST` | `MARG_COST` |
| `CONG. PROB +` | `CONG_PROB_DIR` |
| `CONG. PROB -` | `CONG_PROB_INDIR` |
| `HURDLE COST` | `HURDLE_COST` |

## Per-cluster views: thermal, storage, renewable

These views use the area as location and detail each metric by cluster (`name` property). The renewable view and its catalog are still to be created.

The detail by `name` requires the system for views to write this property on thermal clusters and storages, which is not the case yet: see [AntaresSimulatorTeam/Antares_Simulator#4001](https://github.com/AntaresSimulatorTeam/Antares_Simulator/issues/4001).

| View | Legacy variable | Metric | Remark |
| --- | --- | --- | --- |
| `view_thermal` | `DTG by plant` | `GENERATION` |  |
| `view_thermal` | `MIN DTG by plant` | `MIN_GEN` |  |
| `view_thermal` | `Profit by plant` | `PROFIT` |  |
| `view_thermal` | `NODU by plant` | `NODU` |  |
| `view_thermal` | `NP Cost by plant` | `NP_COST` |  |
| `view_storage` | `STS inj by plant` | `P_INJECTION` |  |
| `view_storage` | `STS withdrawal by plant` | `P_WITHDRAWAL` |  |
| `view_storage` | `STS lvl by plant` | `LEVELS` |  |
| `view_storage` | `STS Cashflow By Cluster` | `CASHFLOW` | Averaged over time by the catalog: to be checked |
| `view_renewable` (to be created) | `RES generation by plant` | `GENERATION` (to be created) | New renewable catalog, detailed by `name`: see [AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter#134](https://github.com/AntaresSimulatorTeam/AntaresLegacyModels-to-GEMS-Converter/issues/134), which also covers the `name` property of renewables |

## Legacy variables without a metric

These variables have no metric in the catalogs yet: for each one, the metric must be created or the variable excluded from V1.

| Legacy variable | Target view | Nature |
| --- | --- | --- |
| `OV. COST CSR`, `MRG. PRICE CSR`, `UNSP. ENRG CSR`, `LOLD CSR`, `LOLP CSR`, `DTG MRG CSR`, `MAX MRG CSR` | `view_area` | Adequacy patch (CSR) |
| `DENS`, `LMR VIOL.` | `view_area` | Adequacy patch |
| `MAX MRG` | `view_area` | Antares-specific computation, to be handled as post-processing (TODO already noted in the catalog) |
| `RAMP COST` | `view_area` | Thermal ramping |
| `RAMP COST BY PLANT` | `view_thermal` | Thermal ramping |
| `FLOW QUAD.` | `view_link` | Deprecated: excluded from V1 |
| `RESERVE PARTICIPATION COST` and the other reserve variables | To be defined | Reserves |
| `BC. MARG. COST` | None | Binding constraints, out of scope |

## Open points

- [ ] Hard-code the table in Antares or store it in a YAML file (in that case, next to the catalogs, in the GEMS repo or the converter).
- [ ] Create, or exclude from V1, each variable without a metric.
- [ ] Create the renewable view and catalog (`view_renewable`, `GENERATION`).
