# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute one model CLI, retaining its golden result and raw benchmark grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys

# Running as a file must work without an editable install of pypto-lib.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_operator_suite import (
    ContractError, ROOT, SAMPLING, case_arguments, check_specs, devices,
    load_manifest, write_json,
)


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


def tensor_fingerprint(values, specs, direction):
    """Hash actual contents without serializing multi-GB weight snapshots."""
    import torch

    result = hashlib.sha256()
    for spec in sorted(specs, key=lambda item: item.name):
        is_tensor = hasattr(spec, "shape")
        include = spec.is_input if is_tensor and direction == "input" else (
            spec.is_output if is_tensor else direction == "input")
        if not include:
            continue
        value = values[spec.name]
        result.update(spec.name.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            result.update(json.dumps([list(value.shape), str(value.dtype)]).encode())
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            result.update(memoryview(raw))
        else:
            result.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return result.hexdigest()


def describe_specs(specs):
    return [{"name": spec.name, "shape": list(spec.shape), "dtype": str(spec.dtype),
             "direction": spec.direction, "resident": spec.resident}
            for spec in specs if hasattr(spec, "shape")]


def make_capture(original_run, case, output, selected):
    """Install a process-local observer; model code keeps its own golden/tolerances."""
    calls = 0

    def capture(**kwargs):
        nonlocal calls
        calls += 1
        if calls != 1:
            output.unlink(missing_ok=True)
            raise ContractError("a case must call golden.run exactly once")
        specs = kwargs["specs"]
        check_specs(case, describe_specs(specs))
        cfg = kwargs["config"]
        distributed = cfg.get("distributed_config")
        actual_devices = list(distributed.device_ids) if distributed else [cfg["device_id"]]
        if actual_devices != selected or cfg.get("platform") != "a2a3":
            raise ContractError("the model's RunConfig differs from its allocated devices/platform")
        if cfg.get("enable_chip_swimlane", 0) or kwargs.get("compile_only") or kwargs.get("runtime_dir"):
            raise ContractError("official measurement requires a fresh compile with swimlane disabled")
        golden_fn = kwargs.get("golden_fn")
        if golden_fn is None or kwargs.get("golden_data"):
            raise ContractError("this capture requires the model's in-memory golden")
        payload = {"case_id": case["case_id"], "sampling": SAMPLING, "passed": False}
        overrides = case.get("run_config", {})
        kwargs["config"] = {**cfg, **overrides}
        payload["run_config"] = overrides

        def golden(values):
            payload["fixture_sha256"] = tensor_fingerprint(values, specs, "input")
            golden_fn(values)
            payload["golden_sha256"] = tensor_fingerprint(values, specs, "output")

        kwargs["golden_fn"] = golden
        kwargs["save_data"] = False
        result = original_run(**kwargs)
        payload.update(passed=result.passed, error=result.error, specs=describe_specs(specs),
                       work_dir=str(result.work_dir) if result.work_dir else None)
        if result.passed:
            payload["benchmark"] = benchmark_payload(result.bench)
        write_json(output, payload)
        return result

    return capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest = load_manifest()
    case = next(case for case in manifest["cases"] if case["case_id"] == args.case)
    selected = devices(args.devices)
    if len(selected) != case["device_count"]:
        raise ContractError("device count differs from the case")
    if os.environ.get("PYTHONHASHSEED") != str(SAMPLING["seed"]):
        raise ContractError("PYTHONHASHSEED must be set before Python starts")
    import numpy as np
    import torch
    import golden

    random.seed(SAMPLING["seed"])
    np.random.seed(SAMPLING["seed"])
    torch.manual_seed(SAMPLING["seed"])
    golden.run = make_capture(golden.run, case, args.output, selected)
    entry = ROOT / case["entrypoint"]
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry), *case_arguments(case, selected)]
    runpy.run_path(str(entry), run_name="__main__")
    if not args.output.is_file():
        raise ContractError("model produced no golden result")


if __name__ == "__main__":
    main()
