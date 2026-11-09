# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Calculate forward FA MFU from theoretical useful QK/PV work; omit FA MBU."""

import argparse
import json

from calculate import nonnegative_int, positive_float, positive_int


def causal_pairs(q_len, kv_len, alignment):
    """Count diagonal-inclusive valid pairs for one batch and query head."""
    offset = kv_len - q_len if alignment == "bottom-right" else 0
    # Each query i (one-based) contributes clamp(i + offset, 0, kv_len).
    first = max(1, 1 + offset)
    last = min(kv_len, q_len + offset)
    ramp_count = max(0, last - first + 1)
    saturated_count = max(0, q_len + offset - kv_len)
    return (first + last) * ramp_count // 2 + saturated_count * kv_len


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "query-heads", "q-len", "kv-len", "qk-dim", "v-dim"):
        parser.add_argument(f"--{name}", type=positive_int, required=True, help="Semantic, unpadded dimension")
    parser.add_argument("--pair-mode", choices=("dense", "causal", "masked", "sparse"), required=True)
    parser.add_argument("--causal-alignment", choices=("top-left", "bottom-right"))
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument(
        "--effective-pairs", type=nonnegative_int,
        help="Aggregate semantic pair count across batch, query heads and queries; masked or sparse mode",
    )
    counts.add_argument(
        "--selected-k", type=nonnegative_int,
        help="Semantic selected entries per query in sparse mode; excludes capacity padding",
    )
    parser.add_argument("--time-us", type=positive_float, required=True)
    parser.add_argument(
        "--time-kind", required=True,
        choices=("kernel-span", "visible-span", "profiler-duration", "mean-core-hypothesis"),
    )
    parser.add_argument(
        "--peak-tflops", type=positive_float, required=True,
        help="Device Cube peak matching the compute mode of both QK and PV",
    )
    args = parser.parse_args()

    if args.causal_alignment is not None and args.pair_mode != "causal":
        parser.error("--causal-alignment applies only to causal mode")
    if args.selected_k is not None and args.pair_mode != "sparse":
        parser.error("--selected-k applies only to sparse mode")
    if args.effective_pairs is not None and args.pair_mode not in ("masked", "sparse"):
        parser.error("--effective-pairs applies only to masked or sparse mode")

    batch_heads = args.batch * args.query_heads
    max_pairs = batch_heads * args.q_len * args.kv_len
    alignment = None
    if args.pair_mode == "dense":
        pairs = max_pairs
    elif args.pair_mode == "causal":
        if args.q_len != args.kv_len and args.causal_alignment is None:
            parser.error("unequal query/KV lengths require an explicit --causal-alignment")
        alignment = args.causal_alignment or "top-left"
        pairs = batch_heads * causal_pairs(args.q_len, args.kv_len, alignment)
    elif args.pair_mode == "masked":
        if args.effective_pairs is None:
            parser.error("masked mode requires --effective-pairs from the semantic mask")
        pairs = args.effective_pairs
    else:
        if args.selected_k is None and args.effective_pairs is None:
            parser.error("sparse mode requires --selected-k or --effective-pairs")
        pairs = args.effective_pairs
        if args.selected_k is not None:
            pairs = batch_heads * args.q_len * args.selected_k
    if pairs > max_pairs:
        parser.error("semantic pair count exceeds the unpadded query/KV domain")

    qk_flops = 2 * pairs * args.qk_dim
    pv_flops = 2 * pairs * args.v_dim
    flops = qk_flops + pv_flops
    tflops = flops / args.time_us / 1e6
    result = {
        "operator": "fa-forward",
        "batch": args.batch,
        "query_heads": args.query_heads,
        "q_len": args.q_len,
        "kv_len": args.kv_len,
        "qk_dim": args.qk_dim,
        "v_dim": args.v_dim,
        "pair_mode": args.pair_mode,
        "causal_alignment": alignment,
        "selected_k": args.selected_k,
        "theoretical_pairs": pairs,
        "theoretical_qk_flops": qk_flops,
        "theoretical_pv_flops": pv_flops,
        "theoretical_flops": flops,
        "flops_basis": "Semantic QK/PV only; excludes masked dense pairs, padding and hardware-added work",
        "time_us": args.time_us,
        "time_kind": args.time_kind,
        "effective_tflops": tflops,
        "peak_tflops": args.peak_tflops,
        "mfu_pct": 100 * tflops / args.peak_tflops,
    }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
