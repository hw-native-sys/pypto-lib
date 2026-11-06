# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Pick the fast card of a multi-card swimlane run from chip_swimlane_records.json.

Usage: fast_card.py <dfx_outputs-dir>

Scans <dfx_outputs>/rank{r}/d{k}/chip_swimlane_records.json, computes each
dispatch's span from the joined task rows, sums the spans per rank, and prints
the per-rank table followed by the fast rank's record files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    from simpler_setup.tools.swimlane_converter import read_perf_data
except ImportError as exc:
    raise SystemExit("Could not import simpler_setup. Activate the pypto-lib environment first.") from exc

RECORDS = "chip_swimlane_records.json"
RANK_RE = re.compile(r"rank(\d+)$")


def span_us(records: Path) -> float:
    rows = read_perf_data(records).get("tasks")
    if not rows:
        raise ValueError(f"{records}: no joined task rows")
    if all(r.get("dispatch_time_us", -1) >= 0 and r.get("finish_time_us", 0) > 0 for r in rows):
        return max(r["finish_time_us"] for r in rows) - min(r["dispatch_time_us"] for r in rows)
    return max(r["end_time_us"] for r in rows) - min(r["start_time_us"] for r in rows)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    per_rank: dict[int, list[tuple[Path, float]]] = {}
    for rank_dir in sorted(root.glob("rank*")):
        match = RANK_RE.fullmatch(rank_dir.name)
        if not match:
            continue
        files = sorted(rank_dir.glob(f"d*/{RECORDS}"), key=lambda p: int(p.parent.name[1:]))
        if not files:
            print(f"error: {rank_dir} has no {RECORDS}", file=sys.stderr)
            return 1
        per_rank[int(match.group(1))] = [(f, span_us(f)) for f in files]
    if not per_rank:
        print(f"error: no rank*/d*/{RECORDS} under {root}", file=sys.stderr)
        return 1
    if len({len(v) for v in per_rank.values()}) != 1:
        counts = ", ".join(f"rank{r}={len(v)}" for r, v in sorted(per_rank.items()))
        print(f"error: dispatch counts differ across ranks ({counts}); refusing to pick", file=sys.stderr)
        return 1

    totals = {rank: sum(s for _, s in items) for rank, items in per_rank.items()}
    fast = min(totals, key=lambda r: (totals[r], r))
    print("rank  dispatches  span_us")
    for rank in sorted(totals):
        mark = "  <- fast" if rank == fast else ""
        print(f"{rank:<5} {len(per_rank[rank]):<11} {totals[rank]:.3f}{mark}")
    for path, _ in per_rank[fast]:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
