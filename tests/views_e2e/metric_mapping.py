# Copyright (c) 2024, RTE (https://www.rte-france.com)
#
# See AUTHORS.txt
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# SPDX-License-Identifier: MPL-2.0
#
# This file is part of the Antares project.
"""Legacy counterpart of each catalog metric: the mc-ind / mc-all column it must equal."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class Source(Enum):
    """Legacy file a metric is read from: `<areas|links>/<object>/<data type>-<frequency>.txt`."""

    AREA_VALUES = ("areas", "values")
    AREA_DETAILS = ("areas", "details")
    AREA_ST_STORAGE = ("areas", "details-STstorage")
    LINK_VALUES = ("links", "values")

    @property
    def object_kind(self) -> str:
        return self.value[0]

    @property
    def data_type(self) -> str:
        return self.value[1]


# Legacy captions of the thermal groups and renewable technologies, by the
# lower-case group the views break DTG_BY_TECHNOLOGY / RES_GENERATION down by.
THERMAL_GROUP_CAPTIONS = {
    "nuclear": "NUCLEAR",
    "lignite": "LIGNITE",
    "hard coal": "COAL",
    "gas": "GAS",
    "oil": "OIL",
    "mixed fuel": "MIX. FUEL",
    "other 1": "MISC. DTG",
    "other 2": "MISC. DTG 2",
    "other 3": "MISC. DTG 3",
    "other 4": "MISC. DTG 4",
}
RENEWABLE_TECHNOLOGY_CAPTIONS = {
    "wind onshore": "WIND ONSHORE",
    "wind offshore": "WIND OFFSHORE",
    "solar pv": "SOLAR PV",
    "solar thermal": "SOLAR CONCRT.",
    "solar rooftop": "SOLAR ROOFT",
    "other res 1": "RENW. 1",
    "other res 2": "RENW. 2",
    "other res 3": "RENW. 3",
    "other res 4": "RENW. 4",
    "wind": "WIND",
    "solar": "SOLAR",
    "run_of_river": "H. ROR",
}


@dataclass(frozen=True)
class LegacyCounterpart:
    """The legacy column a metric must equal, as `legacy = scale * view`.

    `column` is the legacy caption; for a metric broken down by a property,
    `caption` turns the breakdown value into the caption (`column` is then
    ignored). For the details files, `unit` is the second header row, which
    tells the variable apart (the first one holds the cluster name).
    """

    source: Source
    column: str = ""
    unit: Optional[str] = None
    caption: Optional[Callable[[str], str]] = None
    scale: float = 1.0
    atol: float = 1.0
    rtol: float = 1e-3

    def legacy_caption(self, breakdown_value: Optional[str]) -> str:
        if breakdown_value is None:
            return self.column
        assert self.caption is not None, "metric broken down without a caption rule"
        return self.caption(breakdown_value)


def _area(column: str, **kwargs) -> LegacyCounterpart:  # type: ignore[no-untyped-def]
    return LegacyCounterpart(Source.AREA_VALUES, column, **kwargs)


def _link(column: str, **kwargs) -> LegacyCounterpart:  # type: ignore[no-untyped-def]
    return LegacyCounterpart(Source.LINK_VALUES, column, **kwargs)


def _cluster(source: Source, unit: str, **kwargs) -> LegacyCounterpart:  # type: ignore[no-untyped-def]
    # The breakdown value is the cluster id, the details files caption it with its
    # name: the comparison resolves one into the other (see comparison.py).
    return LegacyCounterpart(source, unit=unit, caption=lambda name: name, **kwargs)


def _sts_group(suffix: str) -> LegacyCounterpart:
    return LegacyCounterpart(
        Source.AREA_VALUES, caption=lambda group: f"{group.upper()}_{suffix}"
    )


# `<catalog id>.<metric id>` -> its legacy counterpart.
METRIC_TO_LEGACY: dict[str, LegacyCounterpart] = {
    # ---- areas --------------------------------------------------------------
    "antares_legacy_area_catalog.OV_COST": _area("OV. COST"),
    "antares_legacy_area_catalog.OP_COST": _area("OP. COST"),
    "antares_legacy_area_catalog.MRG_PRICE": _area("MRG. PRICE", atol=1e-2),
    **{
        f"antares_legacy_area_catalog.{pollutant}_EMIS": _area(f"{pollutant} EMIS.")
        for pollutant in [
            "CO2",
            "NH3",
            "SO2",
            "NOX",
            "PM2_5",
            "PM5",
            "PM10",
            "NMVOC",
            "OP1",
            "OP2",
            "OP3",
            "OP4",
            "OP5",
        ]
    },
    "antares_legacy_area_catalog.LOAD": _area("LOAD"),
    "antares_legacy_area_catalog.RES_LOAD": _area("RES LOAD"),
    "antares_legacy_area_catalog.DTG_BY_TECHNOLOGY": LegacyCounterpart(
        Source.AREA_VALUES,
        caption=lambda group: THERMAL_GROUP_CAPTIONS.get(group, group.upper()),
    ),
    "antares_legacy_area_catalog.BALANCE": _area("BALANCE"),
    "antares_legacy_area_catalog.RES_GENERATION": LegacyCounterpart(
        Source.AREA_VALUES,
        caption=lambda tech: RENEWABLE_TECHNOLOGY_CAPTIONS.get(tech, tech.upper()),
    ),
    "antares_legacy_area_catalog.STS_INJECTION": _sts_group("INJECTION"),
    "antares_legacy_area_catalog.STS_WITHDRAWAL": _sts_group("WITHDRAWAL"),
    "antares_legacy_area_catalog.STS_LEVEL": _sts_group("LEVEL"),
    "antares_legacy_area_catalog.UNSP_ENRG": _area("UNSP. ENRG"),
    "antares_legacy_area_catalog.SPIL_ENRG": _area("SPIL. ENRG"),
    "antares_legacy_area_catalog.NODU": _area("NODU"),
    "antares_legacy_area_catalog.NP_COST": _area("NP COST"),
    "antares_legacy_area_catalog.AVL_DTG": _area("AVL DTG"),
    "antares_legacy_area_catalog.DTG_MRG": _area("DTG MRG"),
    "antares_legacy_area_catalog.LOLD": _area("LOLD"),
    "antares_legacy_area_catalog.LOLP": _area("LOLP", scale=100),
    "antares_legacy_area_catalog.NPCAP_HOURS": _area("NPCAP HOURS"),
    "antares_legacy_area_catalog.ROW_BAL": _area("ROW BAL."),
    "antares_legacy_area_catalog.MISC_NDG": _area("MISC. NDG"),
    "antares_legacy_area_catalog.PSP": _area("PSP"),
    "antares_legacy_area_catalog.H_STOR": _area("H. STOR"),
    "antares_legacy_area_catalog.H_PUMP": _area("H. PUMP"),
    "antares_legacy_area_catalog.H_OVFL": _area("H. OVFL"),
    "antares_legacy_area_catalog.H_INFL": _area("H. INFL"),
    "antares_legacy_area_catalog.H_LEV": _area("H. LEV"),
    "antares_legacy_area_catalog.H_VAL": _area("H. VAL"),
    "antares_legacy_area_catalog.H_COST": _area("H. COST"),
    # ---- links --------------------------------------------------------------
    "antares_legacy_link_catalog.FLOW_LIN": _link("FLOW LIN."),
    "antares_legacy_link_catalog.UCAP_LIN": _link("UCAP LIN."),
    "antares_legacy_link_catalog.CONG_FEE_ABS": _link("CONG. FEE (ABS.)"),
    "antares_legacy_link_catalog.CONG_FEE_ALG": _link("CONG. FEE (ALG.)"),
    "antares_legacy_link_catalog.MARG_COST": _link("MARG. COST", atol=1e-2),
    "antares_legacy_link_catalog.CONG_PROB_DIR": _link("CONG. PROB +", scale=100),
    "antares_legacy_link_catalog.CONG_PROB_INDIR": _link("CONG. PROB -", scale=100),
    "antares_legacy_link_catalog.HURDLE_COST": _link("HURDLE COST"),
    "antares_legacy_link_catalog.LOOP_FLOW": _link("LOOP FLOW"),
    # ---- short-term storages ------------------------------------------------
    "antares_legacy_short_term_storage_catalog.P_INJECTION": _cluster(
        Source.AREA_ST_STORAGE, "P-injection - MW"
    ),
    "antares_legacy_short_term_storage_catalog.P_WITHDRAWAL": _cluster(
        Source.AREA_ST_STORAGE, "P-withdrawal - MW"
    ),
    "antares_legacy_short_term_storage_catalog.LEVELS": _cluster(
        Source.AREA_ST_STORAGE, "Levels - MWh"
    ),
    "antares_legacy_short_term_storage_catalog.CASHFLOW": _cluster(
        Source.AREA_ST_STORAGE, "CashFlow - Euro"
    ),
    # ---- thermal clusters ---------------------------------------------------
    "antares_legacy_thermal_cluster_catalog.GENERATION": _cluster(
        Source.AREA_DETAILS, "MWh"
    ),
    "antares_legacy_thermal_cluster_catalog.MIN_GEN": _cluster(
        Source.AREA_DETAILS, "MIN GEN - MWh"
    ),
    "antares_legacy_thermal_cluster_catalog.PROFIT": _cluster(
        Source.AREA_DETAILS, "Profit - Euro"
    ),
    "antares_legacy_thermal_cluster_catalog.NODU": _cluster(
        Source.AREA_DETAILS, "NODU"
    ),
    "antares_legacy_thermal_cluster_catalog.NP_COST": _cluster(
        Source.AREA_DETAILS, "NP Cost - Euro"
    ),
}
