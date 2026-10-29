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
"""Markdown summary of the comparisons written by the views tests.

    python -m tests.views_e2e.report <report dir> [title]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def summary(report_dir: Path, title: str = "Views vs legacy outputs") -> str:
    rows = [
        json.loads(line)
        for jsonl in sorted(report_dir.glob("comparisons-*.jsonl"))
        for line in jsonl.read_text(encoding="utf-8").splitlines()
    ]
    by_metric: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    for row in rows:
        by_metric[(row["study"], row["view"], row["metric"], row["pattern"])].append(
            row
        )

    counts: dict[str, int] = defaultdict(int)
    lines = [
        "| study | view | metric | pattern | status | series | max abs diff | detail |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for (study, view, metric, pattern), comparisons in sorted(by_metric.items()):
        failed = [c for c in comparisons if not c["ok"]]
        compared = [c for c in comparisons if c["ok"] and not c["not_comparable"]]
        known_gap = comparisons[0]["known_gap"]
        if failed:
            status = "known gap" if known_gap else "MISMATCH"
            detail = known_gap or f"{failed[0]['location']}: {failed[0]['error']}"
        elif compared:
            status, detail = "ok", ""
        else:
            status, detail = "not comparable", comparisons[0]["not_comparable"]
        counts[status] += 1
        max_diff = max(c["max_abs_diff"] for c in comparisons)
        icon = {"ok": "✅", "MISMATCH": "❌", "known gap": "⚠️", "not comparable": "➖"}
        lines.append(
            f"| {study} | {view} | {metric.split('.')[1]} | {pattern} | "
            f"{icon[status]} {status} | {len(comparisons)} | {max_diff:.4g} | {detail} |"
        )
    totals = ", ".join(f"{n} {status}" for status, n in sorted(counts.items()))
    return "\n".join(
        [f"## {title}", "", f"Metrics × patterns: {totals}", "", *lines, ""]
    )


if __name__ == "__main__":
    print(summary(Path(sys.argv[1]), *sys.argv[2:3]))
