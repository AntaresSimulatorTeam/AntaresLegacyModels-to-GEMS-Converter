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
"""The views of the converter view configs equal the legacy outputs of the same simulation.

For each study, view config, metric and aggregation pattern: every series of the
view (location, breakdown, MC year or scenario statistic) is compared to its
legacy counterpart (metric_mapping.py), mc-ind for the per-scenario patterns,
mc-all for the one aggregated over scenarios.
"""

import json
import os
from typing import Optional

import pytest
import yaml

from tests.views_e2e.comparison import compare_metric
from tests.views_e2e.conftest import report_dir
from tests.views_e2e.metric_mapping import METRIC_TO_LEGACY
from tests.views_e2e.pipeline import (
    AGGREGATION_PATTERNS,
    VIEW_CONFIGS_DIR,
    SimulationResults,
)
from tests.views_e2e.studies import STUDIES

pytestmark = pytest.mark.views_e2e

# (`<catalog id>.<metric id>`, aggregation pattern id or None for all of them)
# -> why these views don't match the legacy outputs yet. Their tests are strict
# xfails: fixing a gap makes its tests fail, to remove it from here.
_AREA = "antares_legacy_area_catalog"
_LINK = "antares_legacy_link_catalog"
_THERMAL = "antares_legacy_thermal_cluster_catalog"
_START_UP_COST = (
    "Antares non_prop_cost misses the start-up cost on the first hour of each week"
)
_UNDATED_ROWS = (
    "Antares 10.2 writes in system-for-views.yml components without simulation table "
    "rows (misc gen, hydro... without data), for which GEMS-ViewsBuilder gives a view "
    "row without date nor scenario, of value 0 or NaN"
)
# Gaps that depend on the Antares version: not strict xfails
VERSION_DEPENDENT_GAPS = {_UNDATED_ROWS}
KNOWN_GAPS: dict[tuple[str, Optional[str]], str] = {
    (f"{_AREA}.OV_COST", "hourly_per_scenario"): _START_UP_COST,
    (f"{_AREA}.OP_COST", "hourly_per_scenario"): _START_UP_COST,
    (f"{_AREA}.NP_COST", None): _START_UP_COST,
    (f"{_THERMAL}.NP_COST", None): _START_UP_COST,
    (f"{_THERMAL}.MIN_GEN", None): "Antares min_gen_power differs from MIN GEN",
    (f"{_THERMAL}.PROFIT", None): "Antares thermal profit differs from Profit",
    (
        f"{_LINK}.MARG_COST",
        None,
    ): "Antares capacity_shadow_price differs from MARG. COST",
    (f"{_LINK}.CONG_PROB_DIR", "annual_per_scenario"): (
        "legacy annual CONG. PROB is 100 if congested at least one hour: needs a max time operator (#159)"
    ),
    (f"{_LINK}.CONG_PROB_DIR", "annual_all_scenario"): (
        "legacy annual CONG. PROB is 100 if congested at least one hour: needs a max time operator (#159)"
    ),
    **{
        (f"{_AREA}.{metric}", None): _UNDATED_ROWS
        for metric in [
            "RES_LOAD",
            "RES_GENERATION",
            "ROW_BAL",
            "MISC_NDG",
            "PSP",
            "H_STOR",
            "H_PUMP",
            "H_OVFL",
            "H_INFL",
            "H_LEV",
        ]
    },
}


def known_gap(metric: str, pattern_id: str) -> Optional[str]:
    return KNOWN_GAPS.get((metric, pattern_id), KNOWN_GAPS.get((metric, None)))


def _view_metrics() -> list[tuple[str, str]]:
    """(view id, `<catalog id>.<metric id>`) of every metric of the converter view configs."""
    metrics = []
    for view_file in sorted(VIEW_CONFIGS_DIR.glob("*.yml")):
        view = yaml.safe_load(view_file.read_text(encoding="utf-8"))["view"]
        metrics += [(view["id"], metric["id"]) for metric in view["metrics"]]
    return metrics


def _params() -> list:  # type: ignore[type-arg]
    params = []
    for study_name in STUDIES:
        for view_id, metric in _view_metrics():
            for pattern in AGGREGATION_PATTERNS:
                marks = []
                if gap := known_gap(metric, pattern["id"]):
                    strict = gap not in VERSION_DEPENDENT_GAPS
                    marks.append(pytest.mark.xfail(reason=gap, strict=strict))
                params.append(
                    pytest.param(
                        study_name,
                        view_id,
                        metric,
                        pattern["id"],
                        marks=marks,
                        id=f"{study_name}-{metric.split('.')[1]}-{view_id}-{pattern['id']}",
                    )
                )
    return params


def test_every_view_metric_has_a_legacy_counterpart() -> None:
    assert (
        sorted({metric for _, metric in _view_metrics()} - set(METRIC_TO_LEGACY)) == []
    )


@pytest.mark.xfail(
    reason="Antares 10.2 writes no `name` property for thermal clusters and short-term storages",
    strict=False,
)
@pytest.mark.parametrize("study_name", list(STUDIES))
def test_system_for_views_has_the_model_properties(
    simulations: dict[str, SimulationResults], study_name: str
) -> None:
    """Antares writes every property the converter libraries declare.

    The views tests complete the missing ones (see complete_system_for_views) so
    as to compare the metrics anyway.
    """
    assert simulations[study_name].missing_properties == {}


@pytest.mark.parametrize("study_name,view_id,metric,pattern_id", _params())
def test_view_metric_matches_legacy_output(
    simulations: dict[str, SimulationResults],
    study_name: str,
    view_id: str,
    metric: str,
    pattern_id: str,
) -> None:
    results = simulations[study_name]
    view = results.views[(view_id, pattern_id)]
    metric_id = metric.split(".")[1]
    comparisons = compare_metric(
        results, view, metric_id, pattern_id, METRIC_TO_LEGACY[metric]
    )
    _report(study_name, view_id, metric, pattern_id, comparisons)
    if not comparisons:
        pytest.skip(
            f"no row of {metric} in {view_id}: no component of the study contributes"
        )
    if all(c.ok and c.not_comparable for c in comparisons):
        pytest.skip(f"not comparable: {comparisons[0].not_comparable}")
    failures = [str(c) for c in comparisons if not c.ok]
    assert (
        not failures
    ), f"{len(failures)}/{len(comparisons)} series differ:\n" + "\n".join(failures)


def _report(study_name, view_id, metric, pattern_id, comparisons) -> None:  # type: ignore[no-untyped-def]
    """Append the comparisons to `<report dir>/comparisons.jsonl`, for the CI summary."""
    report = report_dir()
    if report is None:
        return
    report.mkdir(parents=True, exist_ok=True)
    # One file per pytest-xdist worker, so that workers don't interleave lines
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    with open(report / f"comparisons-{worker}.jsonl", "a", encoding="utf-8") as f:
        for c in comparisons:
            f.write(
                json.dumps(
                    {
                        "study": study_name,
                        "view": view_id,
                        "metric": metric,
                        "pattern": pattern_id,
                        "location": c.location,
                        "breakdown": c.breakdown,
                        "selector": c.selector,
                        "legacy_column": c.legacy_column,
                        "size": c.size,
                        "max_abs_diff": c.max_abs_diff,
                        "ok": c.ok,
                        "error": c.error,
                        "not_comparable": c.not_comparable,
                        "known_gap": known_gap(metric, pattern_id),
                    }
                )
                + "\n"
            )
