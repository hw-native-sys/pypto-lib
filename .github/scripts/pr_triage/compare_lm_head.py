# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compare two LM-head implementations with one frozen golden and process."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys


def source_revision(repo):
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def benchmark_samples(bench, rounds, warmup, world_size):
    if bench is None:
        raise ValueError("hardware benchmark results are unavailable")
    if (bench.rounds, bench.warmup) != (rounds, warmup):
        raise ValueError("benchmark sampling differs from the requested contract")
    if bench.fallback_flattened or bench.all_zero_device or getattr(bench, "unstable_dispatch_slots", False):
        raise ValueError("flattened, zero or unstable benchmark dispatches")
    samples = bench.per_rank("effective")
    if len(samples) != world_size:
        raise ValueError("benchmark rank count differs from the allocation")
    for values in samples.values():
        if len(values) != rounds or any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("invalid or incomplete effective_us samples")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-repo", required=True, type=Path)
    parser.add_argument("--candidate-repo", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--devices", default="0")
    parser.add_argument("--head-tp", type=int, default=1)
    parser.add_argument("--attention-tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--entry", choices=("projection", "sample"), default="sample")
    parser.add_argument("--active", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("--cpu-runtime", type=Path)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--vocab", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--row-boundaries", action="store_true")
    parser.add_argument("--candidate-only", action="store_true")
    parser.add_argument("--golden-data", type=Path)
    args = parser.parse_args()
    baseline_repo = args.baseline_repo.resolve()
    candidate_repo = args.candidate_repo.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.candidate_only and args.golden_data is None:
        parser.error("--candidate-only requires --golden-data from a passing baseline")
    if not args.sim and not os.environ.get("TASK_DEVICE"):
        parser.error("Device execution requires a task-submit allocation")
    devices = [int(x) for x in args.devices.split(",")]
    if len(devices) != args.head_tp * args.dp or len(devices) != len(set(devices)):
        parser.error("device count must equal --head-tp times --dp")
    if not args.sim:
        allocated = [int(item) for item in os.environ["TASK_DEVICE"].split(",")]
        if devices != allocated:
            parser.error("device IDs must exactly match TASK_DEVICE")
        if any(value is not None for value in (args.hidden, args.vocab, args.rows)):
            parser.error("bounded fixture overrides require --sim")
    for name in ("config.py", "utils.py"):
        relative = Path("models/deepseek_v4_flash_dspark") / name
        if (baseline_repo / relative).read_bytes() != (candidate_repo / relative).read_bytes():
            raise RuntimeError(f"Baseline and candidate must share {name}")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["PYPTO_PROG_BUILD_DIR"] = str(out / "builds")
    os.environ["PYPTO_BENCH"] = "0" if args.sim else "1"
    os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
    os.environ["PYPTO_BENCH_WARMUP"] = str(args.warmup)
    os.environ.setdefault("SIMPLER_SCHEDULER_TIMEOUT_MS", "120000")
    model = Path("models/deepseek_v4_flash_dspark")
    sys.path.insert(0, str(candidate_repo))
    sys.path.insert(0, str(candidate_repo / model))
    sys.argv = [
        "lm_head.py", "--tp", str(args.attention_tp), "--lm-head-tp", str(args.head_tp), "--dp", str(args.dp),
    ]
    if args.sim:
        if args.cpu_runtime is None:
            parser.error("--sim requires --cpu-runtime with an isolated configure()")
        runtime_spec = importlib.util.spec_from_file_location("lm_head_cpu_runtime", args.cpu_runtime)
        runtime = importlib.util.module_from_spec(runtime_spec)
        runtime_spec.loader.exec_module(runtime)
        runtime.configure()

    import config
    changes = {}
    if args.hidden is not None:
        changes["hidden_size"] = args.hidden
    if args.vocab is not None:
        changes["vocab_size"] = args.vocab
    if changes:
        config.FLASH = dataclasses.replace(config.FLASH, **changes)
    if args.rows is not None:
        config.MOE_TOKENS = args.rows

    import torch
    from golden import TensorSpec, run
    from pypto.ir import DistributedConfig

    torch.manual_seed(0)

    def load(label, repo):
        path = repo / model / "lm_head.py"
        name = f"pr1361_{label}_lm_head"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def prepare(module):
        specs = module.build_tensor_specs(args.active)
        if args.row_boundaries:
            def indices():
                value = torch.full((module.WORLD_SIZE, module.MAX_LOGIT_ROWS), -1, dtype=torch.int32)
                row_values = [0, module.TEST_TOKENS - 1, module.TEST_TOKENS + 17, -2, 5, 1]
                for rank in range(module.WORLD_SIZE):
                    for i, index in enumerate(row_values):
                        value[rank, i] = index
                if module.MAX_LOGIT_ROWS > module.MM_ROW_TILE:
                    value[:, module.MM_ROW_TILE] = 7
                    value[:, -1] = module.TEST_TOKENS + 3
                return value
            specs[-1].init_value = indices
        if args.entry == "projection":
            return module.l3_lm_head, specs, module.golden_lm_head

        def grammar():
            value = torch.full(
                (module.WORLD_SIZE, module.MAX_LOGIT_ROWS, module.GREEDY_GRID_ROWS, module.GRAMMAR_SEGMENT_WORDS),
                -1, dtype=torch.int16,
            )
            value[0, 0] = 0
            value[0, 0, 0, 0] = -32768
            if module.WORLD_SIZE > 1:
                value[1, 1] = 0
                value[1, 1, -1, 0] = 1
            return value

        specs.append(TensorSpec(
            "grammar_mask",
            [module.WORLD_SIZE, module.MAX_LOGIT_ROWS, module.GREEDY_GRID_ROWS, module.GRAMMAR_SEGMENT_WORDS],
            torch.int16, init_value=grammar,
        ))
        specs.append(TensorSpec(
            "sampled_ids", [module.WORLD_SIZE, module.MAX_LOGIT_ROWS, module.SAMPLED_IDS_PAD], torch.int32,
        ))
        return module.l3_lm_head_sample, specs, module.golden_lm_head_sample

    def execute(label, repo, golden_data=None):
        module = load(label, repo)
        fn, specs, golden = prepare(module)
        print(f"COMPARE {label}: head_tp={module.TP_SIZE} world={module.WORLD_SIZE} D={module.D} "
              f"vocab={module.VOCAB} rows={module.MAX_LOGIT_ROWS}", flush=True)
        result = run(
            fn=fn, specs=specs, golden_fn=golden, golden_data=str(golden_data) if golden_data else None,
            save_data=golden_data is None,
            config=dict(
                distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0),
                backend_type="a2a3", execution_mode=args.sim, ring_heap=module.LM_HEAD_RING_HEAP,
                enable_chip_swimlane=0, log_level="error",
            ),
            rtol=1e-3, atol=1e-3,
        )
        source = repo / model / "lm_head.py"
        record = dict(
            passed=result.passed, error=result.error, source_revision=source_revision(repo),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), work_dir=str(result.work_dir),
            head_tp=module.TP_SIZE, world=module.WORLD_SIZE, hidden=module.D, vocab=module.VOCAB,
            rows=module.MAX_LOGIT_ROWS, active=args.active, rounds=args.rounds, warmup=args.warmup,
            entry=args.entry, sim=args.sim, replayed=golden_data is not None, row_boundaries=args.row_boundaries,
        )
        if not args.sim:
            try:
                samples = benchmark_samples(result.bench, args.rounds, args.warmup, len(devices))
            except ValueError as error:
                record["benchmark_error"] = str(error)
                (out / f"{label}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
                raise
            record["effective_us_samples_by_rank"] = {str(k): list(v) for k, v in samples.items()}
            record["effective_us_median_by_rank"] = {str(k): statistics.median(v) for k, v in samples.items()}
            record["fastest_rank_median_us"] = min(record["effective_us_median_by_rank"].values())
        (out / f"{label}.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        if not result.passed:
            raise RuntimeError(f"{label} validation failed: {result.error}")
        return Path(result.work_dir) / "data"

    snapshot = args.golden_data
    if not args.candidate_only:
        snapshot = execute("baseline", baseline_repo)
    if not (snapshot / "in").is_dir() or not (snapshot / "out").is_dir():
        raise RuntimeError("A complete passing golden snapshot is required")
    execute("candidate", candidate_repo, snapshot)
    print(f"PASS: frozen golden replay {snapshot}", flush=True)


if __name__ == "__main__":
    main()
