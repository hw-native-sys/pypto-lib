# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
# Native A5 VF, VPTO and Cube/Vector transfers require device execution.
"""PyPTO sparse MLA with packed-cache dequantization and sink-aware Cube attention."""
"""Gather sparse attention over packed FP8/FP4 caches with an attention sink."""

from pathlib import Path

import pypto.language as pl
import torch
from pypto.runtime import pto_isa_include_dir

from config import (
    FLASH as M, CMP_BLOCKS_DYN, ORI_BLOCKS_DYN, T_DYN, WINDOW_CACHE_BYTES, COMPRESSED_CACHE_BYTES,
)
from quant import (
    dequantize_mxfp4_cache, dequantize_mxfp8_cache, quantize_mxfp4_cache, quantize_mxfp8_cache,
    pack_quantized_cache, unpack_quantized_cache,
)

# model config
HEAD_DIM = M.head_dim
LOCAL_H = M.num_attention_heads
WINDOW_KEYS = M.sliding_window
INDEX_TOPK = M.index_topk

# tiling
WORKERS = 32


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "sparse_gather_vf.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "sparse_gather_vf.cpp",
               include_dirs=(pto_isa_include_dir(),), dual_aiv_dispatch=True)
def _gather_sparse_vf_rows(
    output: pl.Out[pl.Tensor],
    query: pl.Tensor,
    sink: pl.Tensor,
    window: pl.Tensor,
    compressed: pl.Tensor,
    window_indices: pl.Tensor,
    compressed_indices: pl.Tensor,
    partials: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    heads: pl.Scalar[pl.INT32],
    window_keys: pl.Scalar[pl.INT32],
    compressed_keys: pl.Scalar[pl.INT32],
    balanced: pl.Scalar[pl.INT32],
    cache_mode: pl.constexpr,
) -> tuple[pl.Tensor, pl.Tensor]:
    """Stream cache rows through L1 with sink-aware sparse attention.

    Scalar gather indices use refreshed GM cache lines; vector masks use
    the asynchronous index copy in UB.
    """
    return output, partials



@pl.jit.inline(auto_scope=False)
def gather_sparse_attention(
    query: pl.Tensor,
    window_indices: pl.Tensor,
    window: pl.Tensor,
    compressed_indices: pl.Tensor,
    compressed: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
    compressed_keys: pl.constexpr,
):
    """Attend H=32/64, D=512 packed caches with K=128 or 640 and both Vector lanes."""
    query = query[:num_tokens, :]
    active_output = output[:num_tokens, :]
    window_indices = window_indices[:num_tokens, :]
    compressed_indices = compressed_indices[:num_tokens, :]
    if compressed_keys == 0:
        # Use fewer workers without increasing the maximum queries per worker.
        query_waves = pl.max(1, (num_tokens + WORKERS - 1) // WORKERS)
        workers = pl.max(1, pl.min(WORKERS, (num_tokens + query_waves - 1) // query_waves))
    else:
        workers = pl.max(1, pl.min(num_tokens, WORKERS))
    workers = pl.min(workers, pl.system.available_cluster_count())
    parts = (WINDOW_KEYS + compressed_keys) // 128
    whole_tiles = ((num_tokens + workers - 1) // workers) * parts
    balanced_tiles = (num_tokens * parts + workers - 1) // workers
    balanced = pl.cast(0, pl.INT32)
    if num_tokens > workers:
        if whole_tiles - balanced_tiles >= 3:
            balanced = pl.cast(1, pl.INT32)
    slots = pl.max(1, 2 * workers * balanced)
    partials = pl.create_tensor([slots * 64 * (HEAD_DIM + 2) + 2 * workers], dtype=pl.FP32)
    if balanced != 0:
        ready_offset = 2 * workers * 64 * (HEAD_DIM + 2)
        for ready_index in pl.range(2 * workers):
            pl.write(partials, [ready_offset + ready_index], pl.cast(0, pl.FP32))
    with pl.spmd(workers, name_hint="sparse_gather_vf", sync_start=True, deps=[source_ready]) as attended:
        computed_output, computed_partials = _gather_sparse_vf_rows(
            active_output, query, sink, window, compressed,
            window_indices, compressed_indices, partials, num_tokens, LOCAL_H, WINDOW_KEYS, compressed_keys,
            balanced, cache_mode=compressed_keys,
        )
    return attended



@pl.jit(auto_scope=False)
def l2_gather_sparse_attention(
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_indices: pl.Tensor[[T_DYN, WINDOW_KEYS], pl.INT32],
    window: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, WINDOW_CACHE_BYTES], pl.UINT8],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    compressed: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, COMPRESSED_CACHE_BYTES], pl.UINT8],
    sink: pl.Tensor[[LOCAL_H], pl.FP32],
    output: pl.InOut[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
    start: pl.Scalar[pl.INT32],
    compressed_keys: pl.Scalar[pl.INT32],
):
    if num_tokens > 0:
        q = query[start:start + num_tokens, :]
        wi = window_indices[start:start + num_tokens, :]
        ci = compressed_indices[start:start + num_tokens, :]
        ready = pl.system.task_dummy(deps=[])
        if compressed_keys == 0:
            gather_sparse_attention(q, wi, window, wi, window, sink, output, num_tokens, ready, 0)
        else:
            gather_sparse_attention(q, wi, window, ci, compressed, sink, output, num_tokens, ready, INDEX_TOPK)
    return output


def golden_gather_sparse_attention(
    query: torch.Tensor,
    window_indices: torch.Tensor,
    window: torch.Tensor,
    compressed_indices: torch.Tensor,
    compressed: torch.Tensor,
    sink: torch.Tensor,
    *,
    compressed_keys: int,
    num_tokens: int | None = None,
    key_tile: int = 128,
) -> torch.Tensor:
    """Attend packed FP8/FP4 cache rows with online BF16 PV and one final sink."""
    tokens = query.shape[0] if num_tokens is None else num_tokens
    heads = sink.numel()
    head_dim = query.shape[-1] if query.ndim == 3 else query.shape[-1] // heads
    q = query[:tokens].reshape(tokens, heads, head_dim)
    output = torch.empty_like(q, dtype=torch.bfloat16)
    sources = [(window, window_indices, False)]
    if compressed_keys:
        sources.append((compressed, compressed_indices[:, :compressed_keys], True))
    for token_begin in range(0, tokens, 128):
        token_end = min(token_begin + 128, tokens)
        chunk = q[token_begin:token_end].float()
        maximum = chunk.new_full(chunk.shape[:-1], -1e30)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(chunk)
        for cache, indices, fp4 in sources:
            flat_cache = cache.reshape(-1, cache.shape[-1])
            for begin in range(0, indices.shape[-1], key_tile):
                rows = indices[token_begin:token_end, begin:begin + key_tile].long()
                valid = rows >= 0
                packed = flat_cache[rows.clamp_min(0)]
                payload, scales = unpack_quantized_cache(packed, fp4, head_dim=head_dim)
                if fp4:
                    values = dequantize_mxfp4_cache(payload, scales, 16, "bf16")
                else:
                    values = dequantize_mxfp8_cache(payload, scales)
                values = values.bfloat16().float().masked_fill(~valid.unsqueeze(-1), 0)
                scores = torch.einsum("thd,tkd->thk", chunk, values) * head_dim**-0.5
                scores = scores.masked_fill(~valid.unsqueeze(1), -1e30)
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1))
                correction = torch.exp(maximum - next_maximum)
                probabilities = torch.exp(scores - next_maximum.unsqueeze(-1))
                probabilities = probabilities.masked_fill(~valid.unsqueeze(1), 0)
                denominator = denominator * correction + probabilities.sum(dim=-1)
                weighted = torch.einsum("thk,tkd->thd", probabilities.bfloat16().float(), values)
                numerator = numerator * correction.unsqueeze(-1) + weighted
                maximum = next_maximum
        final_maximum = torch.maximum(maximum, sink.float())
        correction = torch.exp(maximum - final_maximum)
        denominator = denominator * correction + torch.exp(sink.float() - final_maximum)
        output[token_begin:token_end] = (numerator * (correction / denominator).unsqueeze(-1)).bfloat16()
    return output.reshape(query[:tokens].shape)



def build_sparse_attention_specs(rows, active, *, compressed_keys, start=0, case="normal"):
    """Build packed caches, irregular indices, poisoned query padding and output sentinels."""
    from golden import ScalarSpec, TensorSpec

    if not 0 <= start <= rows or not 0 <= active <= rows - start:
        raise ValueError("active query range must fit in rows")
    if compressed_keys not in (0, INDEX_TOPK) or case not in ("normal", "codes"):
        raise ValueError("unsupported compressed key count or fixture case")
    window_pages = 1 if case == "codes" else 2
    compressed_pages = 1 if not compressed_keys or case == "codes" else 4
    values = {}

    def initialize(name):
        if not values:
            generator = torch.Generator().manual_seed(4102026 + active)
            query = torch.randn(rows, LOCAL_H * HEAD_DIM, generator=generator).mul_(0.35).bfloat16()
            window = torch.randn(window_pages, 128, 1, HEAD_DIM, generator=generator).mul_(0.6).bfloat16()
            wp, ws = quantize_mxfp8_cache(window)
            wi = torch.full((rows, WINDOW_KEYS), -1, dtype=torch.int32)
            for row in range(rows):
                kind = row % 6
                if kind == 1:
                    wi[row, :13] = torch.arange(13, dtype=torch.int32) + 127
                elif kind == 2:
                    wi[row] = (torch.arange(WINDOW_KEYS, dtype=torch.int32) + 221) % 256
                elif kind == 3:
                    wi[row] = torch.arange(WINDOW_KEYS, dtype=torch.int32) + 96
                elif kind == 4:
                    wi[row] = torch.randint(0, 256, (WINDOW_KEYS,), generator=generator, dtype=torch.int32)
                    wi[row, ::7] = -1
                    wi[row, 5:9] = 17
                elif kind == 5:
                    wi[row, :65] = torch.arange(65, dtype=torch.int32) + 64
            compressed = torch.randn(compressed_pages, 128, 1, HEAD_DIM, generator=generator).mul_(0.6).bfloat16()
            cp, cs = quantize_mxfp4_cache(compressed, 16, "bf16")
            ci = torch.randint(0, compressed_pages * 128, (rows, INDEX_TOPK), generator=generator, dtype=torch.int32)
            ci[:, ::19] = torch.iinfo(torch.int32).min
            ci[:, 5:9] = 17
            if case == "codes":
                wp = torch.linspace(-448, 448, 128 * HEAD_DIM).reshape(1, 128, 1, HEAD_DIM)
                wp = wp.to(torch.float8_e4m3fn)
                physical_rows = torch.arange(128).reshape(128, 1)
                packed = torch.arange(128 * (HEAD_DIM // 2)).reshape(128, HEAD_DIM // 2)
                cp = ((packed * 37 + physical_rows * 13) % 256).to(torch.uint8)
                cp = cp.reshape(1, 128, 1, HEAD_DIM // 2)
                scales = torch.tensor([0.0, 0.125, 1.0, 131072.0], dtype=torch.bfloat16)
                row_scale = (2.0 ** (physical_rows.remainder(5) - 2)).bfloat16()
                ws = scales.repeat(128 * (HEAD_DIM // 32) // 4).reshape(128, HEAD_DIM // 32) * row_scale
                cs = scales.repeat(128 * (HEAD_DIM // 16) // 4).reshape(128, HEAD_DIM // 16) * row_scale
                ws = ws.reshape(1, 128, 1, HEAD_DIM // 32)
                cs = cs.reshape(1, 128, 1, HEAD_DIM // 16)
                wi.fill_(-2)
                ci.fill_(torch.iinfo(torch.int32).min)
                for row in range(rows):
                    if compressed_keys and row % 2:
                        ci[row, 0] = row % 128
                    else:
                        wi[row, 0] = row % 128
                query.zero_()
            else:
                ci[(wi < 0).all(dim=-1)] = -1
            if active:
                wi[start] = -1
                ci[start] = -1
            query[:start] = float("nan")
            query[start + active:] = float("nan")
            values.update(query=query, window_indices=wi, window=pack_quantized_cache(wp, ws))
            values.update(compressed_indices=ci, compressed=pack_quantized_cache(cp, cs))
            if not compressed_keys:
                values["compressed_indices"].fill_(-1)
                values["compressed"].zero_()
            values["sink"] = torch.full((LOCAL_H,), -100.0) if case == "codes" else torch.linspace(-6.0, -1.0, LOCAL_H)
            values["output"] = torch.full_like(query, 7.25)
        return values[name].clone()

    shapes = [
        ("query", [rows, LOCAL_H * HEAD_DIM], torch.bfloat16),
        ("window_indices", [rows, WINDOW_KEYS], torch.int32),
        ("window", [window_pages, 128, 1, WINDOW_CACHE_BYTES], torch.uint8),
        ("compressed_indices", [rows, INDEX_TOPK], torch.int32),
        ("compressed", [compressed_pages, 128, 1, COMPRESSED_CACHE_BYTES], torch.uint8),
    ]
    shapes += [("sink", [LOCAL_H], torch.float32), ("output", [rows, LOCAL_H * HEAD_DIM], torch.bfloat16)]
    specs = []
    for name, shape, dtype in shapes:
        specs.append(TensorSpec(name, shape, dtype, init_value=lambda name=name: initialize(name)))
    specs += [ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True)]
    specs += [ScalarSpec("start", torch.int32, start, compile_runtime=True)]
    specs += [ScalarSpec("compressed_keys", torch.int32, compressed_keys)]
    return specs


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="Validate packed gather sparse attention on Ascend950")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, help="active row count; defaults to boundary cases")
    parser.add_argument("--compressed-keys", type=int, choices=[0, INDEX_TOPK], help="defaults to both key counts")
    parser.add_argument("--case", choices=["normal", "codes", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--runtime-dir", type=str)
    args = parser.parse_args()
    if args.rows is not None and args.rows < 0:
        parser.error("--rows must be nonnegative")
    if args.runtime_dir and (args.rows is None or args.compressed_keys is None or args.case == "all"):
        parser.error("--runtime-dir requires --rows, --compressed-keys and one --case")
    torch.set_num_threads(8)
    keys = (0, INDEX_TOPK) if args.compressed_keys is None else (args.compressed_keys,)
    cases = []
    if args.case in ("normal", "all"):
        cases = [(n + 1, n, 0, "normal") for n in (0, 1, 3, 33, 64, 96, 128, 129)]
        cases += [(13, 7, 3, "normal"), (73, 72, 1, "normal")]
    if args.case in ("codes", "all"):
        cases.append((13, 7, 3, "codes"))
    if args.rows is not None:
        cases = [(args.rows + 1, args.rows, 1, case) for case in ("normal", "codes") if args.case in (case, "all")]
    records = []
    for compressed_keys in keys:
        for rows, active, start, case in cases:
            specs = build_sparse_attention_specs(rows, active, compressed_keys=compressed_keys, start=start, case=case)
            query_rows = slice(start, start + active)
            exact = case == "codes" or active <= 1
            compare = ratio_allclose(atol=0.0 if exact else 0.003, rtol=0.0 if exact else 0.03, max_error_ratio=0.0)
            result = run(
                l2_gather_sparse_attention,
                specs,
                golden_fn=lambda tensors: tensors["output"][:active].copy_(golden_gather_sparse_attention(
                    tensors["query"][query_rows], tensors["window_indices"][query_rows], tensors["window"],
                    tensors["compressed_indices"][query_rows], tensors["compressed"], tensors["sink"],
                    compressed_keys=compressed_keys,
                )),
                compare_fn={"output": compare},
                config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
                compile_only=args.compile_only,
                save_data=args.save_data,
                runtime_dir=args.runtime_dir,
            )
            if not result.passed:
                raise RuntimeError(result.error)
            records.append(dict(compressed_keys=compressed_keys, active=active, start=start, case=case, passed=True))
            status = "SIM_VALIDATED" if args.execution_mode else "DEVICE_VALIDATED"
            if args.compile_only:
                status = "COMPILE_ONLY"
            detail = f"compressed_keys={compressed_keys} active={active} start={start} case={case}"
            print(status, detail, result.work_dir)
    _main_results = records
