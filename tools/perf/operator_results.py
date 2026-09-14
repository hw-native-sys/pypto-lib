# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Validate raw benchmark dispatches and compare only compatible results."""

from __future__ import annotations

import statistics

from tools.perf.operator_contract import ContractError, SAMPLING, check_specs, positive


def benchmark_payload(stats):
    """Read the public RunResult.bench data without guessing a log layout."""
    if stats is None:
        raise ContractError("benchmark unavailable or skipped")
    def invocation(item):
        return {"inv": item.inv, "task": item.task, "effective_us": item.effective_us,
                "task_name": getattr(item, "task_name", item.task)}

    grid = stats.rounds_dispatches
    if grid:
        rounds = [{str(pid): [invocation(item) for item in items] for pid, items in row.items()}
                  for row in grid]
    else:
        # Only the single-card consumer may accept this representation.
        rounds = [{str(item.pid): [invocation(item)]} for item in stats.invocations]
    return {"rounds": stats.rounds, "warmup": stats.warmup,
            "fallback_flattened": stats.fallback_flattened,
            "unstable_dispatch_slots": getattr(stats, "unstable_dispatch_slots", False),
            "all_zero_device": stats.all_zero_device, "distributed_grid": bool(grid),
            "samples": rounds}


def validate_result(payload, case, *, process_rc=0):
    if process_rc != 0 or payload.get("passed") is not True:
        raise ContractError(f"operator did not pass correctness (exit {process_rc}): {payload.get('error')}")
    if payload.get("case_id") != case["case_id"] or payload.get("sampling") != SAMPLING:
        raise ContractError("case identity or sampling differs from the request")
    if payload.get("run_config") != case["run_config"]:
        raise ContractError("runtime settings differ from the case contract")
    if not payload.get("fixture_sha256") or not payload.get("golden_sha256"):
        raise ContractError("missing actual input or reference fingerprint")
    check_specs(case, payload["specs"], require_outputs=True)
    bench = payload["benchmark"]
    if (bench["rounds"], bench["warmup"]) != (100, 5):
        raise ContractError("expected 100 measured rounds after 5 warmup rounds")
    if any(bench.get(key) is not False for key in (
        "fallback_flattened", "unstable_dispatch_slots", "all_zero_device",
    )):
        raise ContractError("benchmark has unavailable, flattened or unstable timing")
    count = case["device_count"]
    if count > 1 and bench.get("distributed_grid") is not True:
        raise ContractError("distributed round boundaries are missing")
    rows = bench["samples"]
    if len(rows) != 100:
        raise ContractError("incomplete benchmark rounds")
    pids = set(rows[0])
    if len(pids) != count:
        raise ContractError("benchmark rank count differs from allocated devices")
    samples = {pid: [] for pid in pids}
    slots = {}
    identities = {}
    seen = set()
    for row in rows:
        if set(row) != pids:
            raise ContractError("missing or unexpected rank in a measured round")
        for pid, dispatches in row.items():
            if not dispatches or len(dispatches) != len(rows[0][pid]):
                raise ContractError("dispatch count changed between rounds")
            total = 0.0
            for slot, dispatch in enumerate(dispatches):
                key = (pid, slot)
                identity = dispatch["task"]
                if not identity or identities.setdefault(key, identity) != identity:
                    raise ContractError("dispatch slot changed callable between rounds")
                inv = dispatch["inv"]
                if type(inv) is not int or inv < 0 or (pid, inv) in seen:
                    raise ContractError("duplicate or invalid rank invocation")
                seen.add((pid, inv))
                value = positive(dispatch["effective_us"], "effective_us")
                slots.setdefault(key, []).append(value)
                total += value
            samples[pid].append(total)
    per_rank = [
        {"trace_pid": pid, "median_us": statistics.median(values), "samples_us": values,
         "dispatches": [{"slot": slot, "task": identities[(pid, slot)],
                         "samples_us": values, "median_us": statistics.median(values)}
                        for (rank, slot), values in sorted(slots.items()) if rank == pid]}
        for pid, values in sorted(samples.items())
    ]
    medians = [rank["median_us"] for rank in per_rank]
    return {"metric_us": min(medians), "max_rank_median_us": max(medians),
            "rank_spread_us": max(medians) - min(medians), "ranks": per_rank}


def comparison(current, baseline, *, mode="history"):
    """A positive percentage means slower; unavailable evidence is never zero."""
    if not baseline or current.get("status") != "pass" or baseline.get("status") != "pass":
        return {"delta_pct": None, "reason": "no passing baseline/current result"}
    keys = ["case_contract", "device_identity", "fixture_sha256", "golden_sha256"]
    keys.append("source_sha" if mode == "toolchain" else "toolchain")
    for key in keys:
        if not current.get(key) or current[key] != baseline.get(key):
            return {"delta_pct": None, "reason": f"incomparable {key}"}
    now = positive(current["metric_us"], "current metric")
    before = positive(baseline["metric_us"], "baseline metric")
    return {"delta_pct": (now / before - 1) * 100, "baseline_run_id": baseline.get("run_id"),
            "baseline_date": baseline.get("logical_date")}
