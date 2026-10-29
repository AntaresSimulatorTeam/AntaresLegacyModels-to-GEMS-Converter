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
"""Compare the rows of a view metric to their legacy counterpart, read with antares-craft."""

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from antares.craft.model.output import (
    Frequency,
    MCAllAreasDataType,
    MCAllLinksDataType,
    MCIndAreasDataType,
    MCIndLinksDataType,
)

from tests.views_e2e.metric_mapping import LegacyCounterpart, Source
from tests.views_e2e.pipeline import CALENDAR_START, SimulationResults

# Aggregation pattern id -> (legacy frequency, per MC year)
PATTERN_TO_LEGACY = {
    "hourly_per_scenario": (Frequency.HOURLY, True),
    "annual_per_scenario": (Frequency.ANNUAL, True),
    "annual_all_scenario": (Frequency.ANNUAL, False),
}
# Statistic of a view aggregated over scenarios -> the mc-all one
SCENARIO_STAT_TO_LEGACY = {"exp": "EXP", "std": "std", "min": "min", "max": "max"}

_BREAKDOWN = re.compile(r"\(([^,()]+),([^()]*)\)")


@dataclass
class Comparison:
    """One series of a view metric (location, breakdown, scenario or statistic) vs legacy."""

    location: str
    breakdown: Optional[str]
    selector: str  # "year <n>" or the scenario statistic
    legacy_column: str
    size: int = 0
    max_abs_diff: float = 0.0
    worst: str = ""
    error: Optional[str] = None
    # Why the legacy outputs have nothing to compare to (e.g. a statistic mc-all doesn't print)
    not_comparable: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        where = f"{self.location}" + (f" [{self.breakdown}]" if self.breakdown else "")
        what = f"{where}, {self.selector}, legacy {self.legacy_column!r}"
        if self.error:
            return f"{what}: {self.error}"
        if self.not_comparable:
            return f"{what}: not comparable, {self.not_comparable}"
        return f"{what}: OK on {self.size} values"


def compare_metric(
    results: SimulationResults,
    view: pl.DataFrame,
    metric_id: str,
    pattern_id: str,
    counterpart: LegacyCounterpart,
) -> list[Comparison]:
    rows = view.filter(pl.col("metric_id") == metric_id).to_pandas()
    frequency, per_year = PATTERN_TO_LEGACY[pattern_id]
    legacy_objects = _legacy_objects(results, counterpart.source)
    comparisons = []
    # A view row is dated (and, per scenario, has a scenario): one without is spurious.
    undated = rows["view_date"].isna() | (per_year & rows["scenario_id"].isna())
    for (location, breakdown), group in rows[undated].groupby(
        ["metric_location", "breakdown_properties"], dropna=False
    ):
        comparison = Comparison(location, _breakdown_value(breakdown), "-", "-")
        comparison.error = (
            f"{len(group)} view rows without date or scenario, of values "
            f"{sorted(group['metric_value'].unique())}"
        )
        comparisons.append(comparison)
    rows = rows[~undated]
    for (location, breakdown), group in rows.groupby(
        ["metric_location", "breakdown_properties"], dropna=False
    ):
        breakdown_value = _breakdown_value(breakdown)
        caption = counterpart.legacy_caption(breakdown_value)
        legacy_object = legacy_objects.get(location)
        selectors = (
            group.groupby("scenario_id") if per_year else group.groupby("scenario_stat")
        )
        for selector, series in selectors:
            if per_year:
                label, year = f"year {int(selector) + 1}", int(selector) + 1
                stat = ""
            else:
                label, year = str(selector), None
                stat = SCENARIO_STAT_TO_LEGACY[str(selector)]
            comparison = Comparison(location, breakdown_value, label, caption)
            comparisons.append(comparison)
            if legacy_object is None:
                comparison.error = (
                    f"no legacy {counterpart.source.object_kind} for this location"
                )
                continue
            legacy = _legacy_file(
                results, counterpart.source, legacy_object, frequency, year
            )
            column = _column(legacy, caption, counterpart.unit, stat)
            if column is None and _column(legacy, caption, counterpart.unit, None):
                comparison.not_comparable = (
                    f"the legacy output has no {stat!r} statistic"
                )
                continue
            if column is None:
                comparison.error = f"no such legacy column (columns: {sorted({c[0] for c in legacy.columns})})"
                continue
            _compare_values(comparison, series, legacy[column], frequency, counterpart)
    return comparisons


def _legacy_objects(
    results: SimulationResults, source: Source
) -> dict[str, tuple[str, ...]]:
    """View location -> the legacy area (or link areas) it stands for."""
    if source.object_kind == "links":
        return {
            f"{link.area_from_id}_{link.area_to_id}_link": (
                link.area_from_id,
                link.area_to_id,
            )
            for link in results.study.get_links().values()
        }
    return {f"{area_id}_node": (area_id,) for area_id in results.study.get_areas()}


def _legacy_file(
    results: SimulationResults,
    source: Source,
    legacy_object: tuple[str, ...],
    frequency: Frequency,
    year: Optional[int],
) -> pd.DataFrame:
    output = results.output
    if source.object_kind == "links":
        if year is None:
            return output.get_mc_all_link(
                frequency, MCAllLinksDataType(source.data_type), *legacy_object
            )
        return output.get_mc_ind_link(
            year, frequency, MCIndLinksDataType(source.data_type), *legacy_object
        )
    if year is None:
        return output.get_mc_all_area(
            frequency, MCAllAreasDataType(source.data_type), *legacy_object
        )
    return output.get_mc_ind_area(
        year, frequency, MCIndAreasDataType(source.data_type), *legacy_object
    )


def _column(
    legacy: pd.DataFrame, caption: str, unit: Optional[str], stat: Optional[str]
) -> Optional[tuple[str, str, str]]:
    """The (caption, unit, statistic) column, captions and units compared case-insensitively.

    With `stat` None, any column of this caption and unit.

    In mc-all, a variable printed once for all years (e.g. LOLP) has the
    statistic `values` instead of EXP/std/min/max: its EXP is that column.
    """
    candidates = [
        column
        for column in legacy.columns
        if column[0].strip().lower() == caption.lower()
        and (unit is None or column[1].strip().lower() == unit.lower())
    ]
    for column in candidates:
        if stat is None or column[2] == stat:
            return column
    if stat == "EXP":
        for column in candidates:
            if column[2] == "values":
                return column
    return None


def _breakdown_value(breakdown: object) -> Optional[str]:
    """`{(name,gas)}` -> `gas`; None without breakdown."""
    matches = _BREAKDOWN.findall(str(breakdown or ""))
    assert len(matches) <= 1, f"several breakdown properties: {breakdown}"
    return matches[0][1].strip() if matches else None


def _compare_values(
    comparison: Comparison,
    series: pd.DataFrame,
    legacy_column: pd.Series,
    frequency: Frequency,
    counterpart: LegacyCounterpart,
) -> None:
    if frequency == Frequency.HOURLY:
        index = (
            (pd.to_datetime(series["view_date"]) - CALENDAR_START) / timedelta(hours=1)
        ).astype(int)
    else:
        index = pd.Series(0, index=series.index)
    if index.max() >= len(legacy_column):
        comparison.error = (
            f"view has time step {index.max()}, legacy only {len(legacy_column)}"
        )
        return
    view_values = counterpart.scale * series["metric_value"].to_numpy(dtype=float)
    legacy_values = legacy_column.to_numpy(dtype=float)[index.to_numpy()]
    comparison.size = len(view_values)
    if np.isnan(legacy_values).all():
        comparison.not_comparable = "the legacy output prints N/A"
        return
    diff = np.abs(view_values - legacy_values)
    tolerance = counterpart.atol + counterpart.rtol * np.abs(legacy_values)
    # N/A on both sides (e.g. hydro values without hydro) is a match
    both_nan = np.isnan(view_values) & np.isnan(legacy_values)
    bad = ~both_nan & ~(diff <= tolerance)
    if len(diff):
        comparison.max_abs_diff = float(np.nanmax(np.where(both_nan, 0, diff)))
    if bad.any():
        worst = int(np.nanargmax(np.where(bad, np.nan_to_num(diff, nan=np.inf), -1)))
        comparison.worst = (
            f"time step {int(index.iloc[worst])}: view {view_values[worst]:.6g}"
            f" vs legacy {legacy_values[worst]:.6g}"
        )
        comparison.error = (
            f"{int(bad.sum())}/{len(bad)} values differ, worst at {comparison.worst}"
        )
