# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Summarize per-rank device execution windows from resident STRACE records."""

import argparse
import json
import re
import statistics
from pathlib import Path


def route_distribution(runtime, ranks, owner, rows=8192):
    """Read the production prefill routes after token delivery has completed."""
    import torch

    counts = torch.zeros((40, 384), dtype=torch.int64)
    for rank in range(owner, owner + 4):
        source = ranks[0][rank]["moe_route_indices"]
        indices = torch.empty(source.shape, dtype=torch.int32)
        runtime.copy_from(indices.data_ptr(), source.data_ptr,
                          indices.numel() * indices.element_size(), worker_id=rank)
        indices = indices[:, :(rows + 3 - (rank - owner)) // 4]
        if bool(((indices < 0) | (indices >= 384)).any()):
            raise ValueError("production prefill contains an invalid expert ID")
        for layer in range(40):
            counts[layer] += torch.bincount(indices[layer].long().flatten(), minlength=384)
    totals = counts.sum(1)
    if not bool((totals == rows * 6).all()):
        raise ValueError("production prefill route counts do not conserve all six branches")
    return dict(total_routes_per_layer=totals.tolist(), expert_counts_per_layer=counts.tolist(),
                receiver_rank_counts_per_layer=counts.reshape(40, 16, 24).sum(2).tolist())


def rank_timings(path, devices=tuple(range(16)), *, dp_rank=0):
    """Read orch/sched unions in device nanoseconds and retain every dispatch."""
    physical_to_rank = {device: rank for rank, device in enumerate(devices)}
    identities, windows = {}, {}
    active = {}
    expected = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("A3_ENGINE_BEGIN "):
            active = json.loads(line.split(" ", 1)[1])
            if active.get("label", "").startswith("measured_"):
                expected.append(active.copy())
        elif line.startswith("A3_ENGINE_END "):
            active = {}
        identity = re.search(r"\[chip_process pid=(\d+) dev=(\d+)\]", line)
        if identity:
            identities[int(identity[1])] = int(identity[2])
        if "[STRACE]" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line.split("[STRACE]", 1)[1]))
        if not all(name in fields for name in ("pid", "hid", "name")):
            continue
        pid = int(fields["pid"])
        if "dev_id" in fields:
            identities[pid] = int(fields["dev_id"])
        name = fields["name"].rsplit(".", 1)[-1]
        if name not in ("orch", "sched") or fields.get("clk") != "dev":
            continue
        key = pid, fields["hid"], fields.get("inv", "")
        window = windows.setdefault(key, dict(marker=active.copy(), spans={}, pid=pid))
        window["spans"][name] = int(fields["ts"]), int(fields["dur"])
    dispatches = []
    incomplete = 0
    for window in windows.values():
        pid, spans = window["pid"], window["spans"]
        if pid not in identities or set(spans) != {"orch", "sched"}:
            incomplete += 1
            continue
        device = identities[pid]
        if device not in physical_to_rank:
            continue
        start = min(value[0] for value in spans.values())
        end = max(value[0] + value[1] for value in spans.values())
        dispatches.append(dict(**window["marker"], rank=physical_to_rank[device], physical_device=device,
                               effective_us=(end - start) / 1000, orch_us=spans["orch"][1] / 1000,
                               sched_us=spans["sched"][1] / 1000))
    stages = {}
    for dispatch in dispatches:
        if not dispatch.get("label", "").startswith("measured_"):
            continue
        key = dispatch["mode"], dispatch["step"], dispatch.get("profiled", False)
        stage = stages.setdefault(key, {})
        stage.setdefault(dispatch["rank"], []).append(dispatch["effective_us"])
    summaries = []
    active_ranks = list(range(dp_rank * 4, dp_rank * 4 + 4))
    for (mode, step, profiled), samples in sorted(stages.items(), key=lambda item: item[0][1]):
        medians = {rank: statistics.median(values) for rank, values in samples.items()}
        comparable = {rank: medians[rank] for rank in active_ranks if rank in medians}
        fastest = min(comparable, key=comparable.get) if len(comparable) == 4 else None
        slowest = max(medians, key=medians.get)
        summaries.append(dict(mode=mode, step=step, profiled=profiled, median_us_per_rank=medians,
                              samples_us_per_rank=samples, fastest_rank=fastest,
                              fastest_rank_median_us=medians.get(fastest), slowest_rank=slowest,
                              slowest_rank_median_us=medians[slowest]))
    missing = []
    for marker in expected:
        seen = [dispatch["rank"] for dispatch in dispatches
                if all(dispatch.get(key) == value for key, value in marker.items())]
        absent = [rank for rank in physical_to_rank.values() if seen.count(rank) != 1]
        if absent:
            missing.append(dict(**marker, missing_or_duplicate_ranks=absent))
    complete = bool(expected) and not missing and not incomplete
    return dict(method="union of device-domain orch and sched STRACE spans", units="microseconds",
                excludes_host_preparation_and_copy=True, incomplete_windows=incomplete,
                status="complete" if complete else "incomplete_or_no_measured_requests",
                fastest_rank_selection="minimum per-rank median within the active TP group",
                active_tp_ranks=active_ranks, missing_dispatches=missing,
                dispatches=dispatches, measured_stages=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    report["device_timings"] = rank_timings(args.execution_log, dp_rank=report.get("dp_rank", 0))
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
