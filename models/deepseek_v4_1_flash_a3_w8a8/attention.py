# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MLA attention core over per-token packed key sets with a per-head sink.

Each token owns its selected key rows (the sliding window plus the indexer's
Top-K), because a multi-request batch addresses different physical cache rows
per request.  One token's local heads form the matmul rows; keys are scanned in
16-row tiles with online softmax.
"""

import math

import pypto.language as pl
import torch

from config import FLASH as M, LOCAL_HEADS, SELECTED_KEYS, T_DYN, WINDOW_KEYS


# model config
HEAD_DIM = M.head_dim
HEADS = LOCAL_HEADS
HEAD_TILE = max(16, HEADS)
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)
MASK_FLOOR = -1.0e30

# tiling
KEY_TILE = 16
KEYS_TEST = 16
ATTN_WORKERS = 16


@pl.jit.inline(auto_scope=False)
def attention_core(
    q: pl.Tensor,
    selected: pl.Tensor,
    combined: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    keys: pl.constexpr,
    ready: pl.Scalar[pl.TASK_ID],
):
    """Online-softmax attention of every token over its packed key set.

    One token's local heads are the matmul rows; its packed keys are shared by
    those heads. The per-lane visibility mask is materialized into a full tile
    (the backend does not broadcast a row across heads).
    """
    head_rows = pl.tensor.dim(q, 0)
    tokens = head_rows // HEADS
    q_flat = q
    out_flat = pl.reshape(output, [tokens * HEADS, HEAD_DIM])
    key_flat = pl.reshape(selected, [tokens * keys, HEAD_DIM])
    sink_logical = pl.reshape(sink, [1, HEADS])
    for token in pl.spmd(tokens, name_hint="attention_core", deps=[ready]):
        token_idx = pl.tile.get_block_idx()
        sink_row = pl.slice(sink_logical, [1, HEAD_TILE], [0, 0], valid_shape=[1, HEADS])
        sink_row = pl.set_validshape(pl.fillpad(sink_row, pad_value=pl.PadValue.zero), 1, HEAD_TILE)
        q_tile = pl.slice(q_flat, [HEAD_TILE, HEAD_DIM], [token_idx * HEADS, 0], valid_shape=[HEADS, HEAD_DIM])
        q_tile = pl.set_validshape(pl.fillpad(q_tile, pad_value=pl.PadValue.zero), HEAD_TILE, HEAD_DIM)
        running_max = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=MASK_FLOOR)
        denominator = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
        accumulator = pl.full([HEAD_TILE, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for k0 in pl.range(0, keys, KEY_TILE):
            key = pl.slice(key_flat, [KEY_TILE, HEAD_DIM], [token_idx * keys + k0, 0])
            scores = pl.matmul(q_tile, key, b_trans=True, out_dtype=pl.FP32)
            scores = pl.mul(scores, ATTN_SCALE)
            mask_row = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for lane in pl.range(KEY_TILE):
                if pl.read(combined, [token_idx, k0 + lane]) >= 0:
                    pl.write(mask_row, [0, lane], 1.0)
            mask = pl.full([HEAD_TILE, KEY_TILE], dtype=pl.FP32, value=0.0)
            for head in pl.range(HEAD_TILE):
                mask[head : head + 1, 0:KEY_TILE] = mask_row
            scores = pl.add(scores, pl.mul(pl.sub(mask, 1.0), -MASK_FLOOR))
            tile_max = pl.reshape(pl.row_max(scores), [1, HEAD_TILE])
            next_max = pl.maximum(running_max, pl.maximum(tile_max, sink_row))
            correction = pl.exp(pl.sub(running_max, next_max))
            next_max_col = pl.reshape(next_max, [HEAD_TILE, 1])
            probability = pl.mul(pl.exp(pl.sub(scores, next_max_col)), mask)
            probability_bf16 = pl.cast(probability, target_type=pl.BF16, mode="rint")
            numerator = pl.matmul(probability_bf16, key, out_dtype=pl.FP32)
            accumulator = pl.add(
                pl.row_expand_mul(accumulator, pl.reshape(correction, [HEAD_TILE, 1])), numerator
            )
            denominator = pl.add(
                pl.mul(denominator, correction),
                pl.reshape(pl.row_sum(probability), [1, HEAD_TILE]),
            )
            running_max = next_max
        denominator = pl.add(denominator, pl.exp(pl.neg(pl.sub(running_max, sink_row))))
        attended_rows = pl.row_expand_mul(accumulator, pl.reshape(pl.recip(denominator), [HEAD_TILE, 1]))
        rounded = pl.cast(attended_rows, target_type=pl.BF16, mode="rint")
        live_heads = pl.slice(rounded, [HEADS, HEAD_DIM], [0, 0])
        out_flat[token_idx * HEADS : token_idx * HEADS + HEADS, 0:HEAD_DIM] = live_heads
    return output


@pl.jit
def attention_core_test(
    q: pl.Tensor[[T_DYN, HEADS, HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, KEYS_TEST, HEAD_DIM], pl.BF16],
    combined: pl.Tensor[[T_DYN, KEYS_TEST], pl.INT32],
    sink: pl.Tensor[[HEADS], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HEADS, HEAD_DIM], pl.BF16]],
):
    """Bounded packed-core entry (16-key set)."""
    q.bind_dynamic(0, T_DYN)
    selected.bind_dynamic(0, T_DYN)
    combined.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    start = pl.system.task_dummy(deps=[])
    q_flat = pl.reshape(q, [pl.tensor.dim(q, 0) * HEADS, HEAD_DIM])
    out_flat = pl.reshape(output, [pl.tensor.dim(q, 0) * HEADS, HEAD_DIM])
    attention_core(q_flat, selected, combined, sink, out_flat, keys=KEYS_TEST, ready=start)
    return output


@pl.jit
def packed_attention_test(
    q: pl.Tensor[[T_DYN, HEADS, HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, SELECTED_KEYS, HEAD_DIM], pl.BF16],
    combined: pl.Tensor[[T_DYN, SELECTED_KEYS], pl.INT32],
    sink: pl.Tensor[[HEADS], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HEADS, HEAD_DIM], pl.BF16]],
):
    """Bounded packed-core entry (window + Top-K set)."""
    q.bind_dynamic(0, T_DYN)
    selected.bind_dynamic(0, T_DYN)
    combined.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    start = pl.system.task_dummy(deps=[])
    q_flat = pl.reshape(q, [pl.tensor.dim(q, 0) * HEADS, HEAD_DIM])
    out_flat = pl.reshape(output, [pl.tensor.dim(q, 0) * HEADS, HEAD_DIM])
    attention_core(q_flat, selected, combined, sink, out_flat, keys=SELECTED_KEYS, ready=start)
    return output


@pl.jit
def window_attention_test(
    q: pl.Tensor[[T_DYN, HEADS, HEAD_DIM], pl.BF16],
    selected: pl.Tensor[[T_DYN, WINDOW_KEYS, HEAD_DIM], pl.BF16],
    combined: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    sink: pl.Tensor[[HEADS], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HEADS, HEAD_DIM], pl.BF16]],
):
    """Bounded packed-core entry (SWA window set)."""
    q.bind_dynamic(0, T_DYN)
    selected.bind_dynamic(0, T_DYN)
    combined.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    start = pl.system.task_dummy(deps=[])
    q_flat = pl.reshape(q, [pl.tensor.dim(q, 0) * HEADS, HEAD_DIM])
    out_flat = pl.reshape(output, [pl.tensor.dim(q, 0) * HEADS, HEAD_DIM])
    attention_core(q_flat, selected, combined, sink, out_flat, keys=WINDOW_KEYS, ready=start)
    return output


def golden_attention_core(tensors):
    """Torch reference: sink-augmented softmax over the visible packed rows."""
    q = tensors["q"].float()
    selected = tensors["selected"].float()
    combined = tensors["combined"]
    sink = tensors["sink"].float()
    out = torch.zeros_like(q)
    for token in range(q.shape[0]):
        visible = torch.nonzero(combined[token] >= 0, as_tuple=False).flatten()
        for head in range(q.shape[1]):
            if visible.numel() == 0:
                scores = torch.zeros(0)
            else:
                scores = (q[token, head] @ selected[token, visible].T) * ATTN_SCALE
            logits = torch.cat([scores, sink[head].reshape(1)])
            probability = torch.softmax(logits, dim=-1)
            if visible.numel() == 0:
                out[token, head] = 0.0
            else:
                out[token, head] = probability[:-1] @ selected[token, visible]
    tensors["output"][:] = out.to(torch.bfloat16)


def build_packed_specs(tokens: int = 8, keys: int = KEYS_TEST, seed: int = 0):
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(seed)
    selected = torch.randn(tokens, keys, HEAD_DIM, generator=generator)
    ramp = torch.arange(1, keys + 1, dtype=torch.int32).reshape(1, -1)
    limit = (torch.arange(tokens, dtype=torch.int32).reshape(-1, 1) + 1) * (keys // tokens + 1)
    combined = torch.where(ramp <= limit, ramp - 1, torch.full_like(ramp, -1))
    return [
        TensorSpec("q", [tokens, HEADS, HEAD_DIM], torch.bfloat16, init_value=torch.randn),
        TensorSpec("selected", [tokens, keys, HEAD_DIM], torch.bfloat16, init_value=selected),
        TensorSpec("combined", [tokens, keys], torch.int32, init_value=combined),
        TensorSpec("sink", [HEADS], torch.float32, init_value=lambda: torch.randn(HEADS) * 0.5),
        TensorSpec("output", [tokens, HEADS, HEAD_DIM], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    compare = {"output": ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=5e-3)}
    cases = [
        ("core16", attention_core_test, build_packed_specs(args.tokens, KEYS_TEST)),
        ("packed640", packed_attention_test, build_packed_specs(args.tokens, SELECTED_KEYS)),
        ("window128", window_attention_test, build_packed_specs(args.tokens, WINDOW_KEYS)),
    ]
    failures = []
    for name, fn, specs in cases:
        print(f"[ATTN] {name}", flush=True)
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_attention_core,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=2e-2,
            atol=2e-2,
            compare_fn=compare,
            compile_only=args.compile_only,
        )
        if not result.passed:
            failures.append(f"{name}: {result.error or 'validation failed'}")
    if failures:
        for failure in failures:
            print(failure)
        raise SystemExit(1)
