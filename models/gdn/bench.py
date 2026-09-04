# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Latency benchmark for the six GDN stages.

Times each stage on device over a sweep of sequence lengths and head counts, in
one process: one compile and one timed loop per (stage, shape), no re-runs for
extra samples. The reported number is the mean of the per-round Effective window
-- the same field as the harness's `[RUN] effective_us ... mean=` line.

Shapes follow the reference benchmark (`megagdn-pto/benchmarks/kernel/
bench_gdn_kernels.py`): D = 128, chunk = 128, and its `--l-seg` list. That
benchmark packs `N_seq` sequences into one call, which these kernels do not
support yet, so the sweep varies the single-sequence length instead. Only
chunk_h is sensitive to the difference -- its recurrence is sequential within a
sequence, so N sequences of L tokens have N times the parallelism of one
sequence of N*L. The other five stages do identical work either way.

Inputs are random by default: the stages have no data-dependent control flow and
no denormal path, so values do not move the time. Pass `--data reference` to
time the pipeline's own data instead; that builds a float64 reference chain on
the host and is only affordable at small T.

    python models/gdn/bench.py -p a2a3 -d 0
    python models/gdn/bench.py -p a2a3 -d 0 --seq-len 4096,8192 --json out.json
"""
import argparse
import dataclasses
import importlib
import json
import os
import statistics
import time

STAGES = ("chunk_cumsum", "scaled_dot_kkt", "solve_tril", "wy_fast",
          "chunk_h", "chunk_o")

# The reference benchmark's fixed dimensions and its --l-seg list.
D = 128
CHUNK = 128
SEQ_LENS = (4096, 8192, 16384)
HEADS = (16,)


def _random_specs(specs):
    """Same shapes and dtypes, values drawn at random instead of from the chain."""
    import torch

    out = []
    for spec in specs:
        if getattr(spec, "is_output", False) or spec.name in ("mask", "tril", "neg_eye"):
            out.append(spec)
            continue
        dtype = spec.dtype
        shape = list(spec.shape)
        if dtype in (torch.float16, torch.float32):
            init = (lambda s=shape, dt=dtype: torch.randn(s, dtype=dt) * 0.1)
        else:
            init = (lambda s=shape, dt=dtype: torch.zeros(s, dtype=dt))
        out.append(dataclasses.replace(spec, init_value=init))
    return out


def bench_stage(stage: str, t: int, h: int, platform: str, device: int,
                rounds: int, warmup: int, data: str) -> dict:
    """One compile and one timed loop. Returns the record for this (stage, shape)."""
    from golden import run_jit

    mod = importlib.import_module(f"models.gdn.{stage}")
    fn = mod.build_kernel(t=t, h=h, d=D, chunk=CHUNK)
    specs = mod.build_tensor_specs(t=t, h=h, d=D, chunk=CHUNK)
    if data == "random":
        specs = _random_specs(specs)

    started = time.time()
    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=None,                       # timing only; correctness is test_gdn_stages
        runtime_cfg=dict(platform=platform, device_id=device),
    )
    rec = dict(stage=stage, t=t, h=h, d=D, chunk=CHUNK, rounds=rounds,
               warmup=warmup, data=data, ok=bool(result.passed),
               compile_and_run_s=round(time.time() - started, 2))
    samples = []
    if result.bench is not None:
        try:
            samples = [s for s in result.bench.per_round("effective") if s > 0]
        except Exception:                     # noqa: BLE001 -- no span tree on *sim
            samples = []
    if samples:
        rec.update(mean_us=round(statistics.fmean(samples), 1),
                   median_us=round(statistics.median(samples), 1),
                   min_us=round(min(samples), 1), max_us=round(max(samples), 1))
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=str,
                        default=",".join(str(x) for x in SEQ_LENS),
                        help="comma-separated single-sequence token counts")
    parser.add_argument("--heads", type=str,
                        default=",".join(str(x) for x in HEADS),
                        help="comma-separated head counts (H = Hg; no GQA yet)")
    parser.add_argument("--stages", type=str, default=",".join(STAGES))
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--data", choices=["random", "reference"], default="random")
    parser.add_argument("--json", type=str, default=None,
                        help="write the records here as JSON")
    args = parser.parse_args()

    os.environ["PYPTO_BENCH"] = "1"
    os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
    os.environ["PYPTO_BENCH_WARMUP"] = str(args.warmup)

    seq_lens = [int(x) for x in args.seq_len.split(",") if x]
    heads = [int(x) for x in args.heads.split(",") if x]
    stages = [s for s in args.stages.split(",") if s]

    records = []
    for t in seq_lens:
        for h in heads:
            for stage in stages:
                rec = bench_stage(stage, t, h, args.platform, args.device,
                                  args.rounds, args.warmup, args.data)
                records.append(rec)
                print(f"[bench] {stage:<16} T={t:<7} H={h:<3} "
                      f"mean={rec.get('mean_us', float('nan'))} us "
                      f"(median={rec.get('median_us')}, min={rec.get('min_us')}, "
                      f"max={rec.get('max_us')})", flush=True)

    print()
    print(format_table(records))
    if args.json:
        payload = dict(platform=args.platform, device=args.device, d=D, chunk=CHUNK,
                       rounds=args.rounds, warmup=args.warmup, data=args.data,
                       metric="effective_us", records=records)
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"[bench] wrote {args.json}")
    return 0 if all(r["ok"] for r in records) else 1


def format_table(records: list[dict]) -> str:
    """Stages down the rows, shapes across the columns, mean Effective µs."""
    shapes = sorted({(r["t"], r["h"]) for r in records})
    stages = [s for s in STAGES if any(r["stage"] == s for r in records)]
    by = {(r["stage"], r["t"], r["h"]): r for r in records}
    head = f"| {'stage':<16} |" + "".join(f" T={t} H={h} |" for t, h in shapes)
    rule = f"|{'-' * 18}|" + "".join("-" * (len(f" T={t} H={h} |") - 1) + "|"
                                     for t, h in shapes)
    lines = [head, rule]
    for stage in stages:
        cells = []
        for t, h in shapes:
            rec = by.get((stage, t, h))
            width = len(f" T={t} H={h} ")
            cells.append(f"{rec.get('mean_us', 'n/a') if rec else '-':>{width}}|")
        lines.append(f"| {stage:<16} |" + "".join(cells))
    total = []
    for t, h in shapes:
        vals = [by[(s, t, h)].get("mean_us") for s in stages if (s, t, h) in by]
        width = len(f" T={t} H={h} ")
        total.append(f"{round(sum(v for v in vals if v), 1) if all(vals) else '-':>{width}}|")
    lines.append(f"| {'TOTAL':<16} |" + "".join(total))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
