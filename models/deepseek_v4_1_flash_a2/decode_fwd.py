# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4.1 Flash A2/A3 single-step SWA Decode entry."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl

from models.deepseek_v4_1_flash_a2.config import FLASH
from models.deepseek_v4_1_flash_a2.gemm import make_int8_linear_direct
from models.deepseek_v4_1_flash_a2.norm_rope import make_rope, make_rmsnorm
from models.deepseek_v4_1_flash_a2.runtime_config import DECODE_PARALLEL, DecodeParallelConfig
from models.deepseek_v4_1_flash_a2.staged import make_grouped_wo_a, make_swa


# Model configuration.
MODEL = asdict(FLASH)
CAPACITY = 16
HIDDEN = FLASH.hidden_size
LATENT = FLASH.q_lora_rank
WIDTH = FLASH.num_attention_heads * FLASH.head_dim
HEAD_DIM = FLASH.head_dim
ROPE_PAIRS = FLASH.qk_rope_head_dim // 2
GROUP_INPUT = WIDTH // FLASH.o_groups
GROUPED_WIDTH = FLASH.o_groups * FLASH.o_lora_rank

# Decode state layout.
BLOCK_SIZE = 128
CACHE_BLOCKS = 2
WINDOW = FLASH.sliding_window


def validate_decode_contract(parallel: DecodeParallelConfig = DECODE_PARALLEL) -> DecodeParallelConfig:
    """Validate the supported topology and the fixed A2/A3 cache ABI."""

    parallel.validate()
    if parallel.ep not in (2, 16):
        raise ValueError("this entry supports only the EP2 gate and the DP8/TP2/EP16 contract")
    if parallel.ep == 2 and (parallel.dp, parallel.tp) != (2, 1):
        raise ValueError("EP2 requires DP2/TP1")
    if parallel.ep == 16 and (parallel.dp, parallel.tp) not in ((16, 1), (8, 2), (4, 4)):
        raise ValueError("EP16 requires DP16/TP1, DP8/TP2, or DP4/TP4")
    return parallel


def make_decode_step(capacity: int = CAPACITY):
    """Build one PyPTO Decode step with paged SWA cache publication."""

    if capacity != CAPACITY:
        raise ValueError(f"A2/A3 decode capacity is fixed at {CAPACITY} for this entry")
    q_a = pl.jit.inline(make_int8_linear_direct(capacity, HIDDEN, LATENT)._func)
    q_norm = make_rmsnorm(capacity, LATENT, FLASH.rms_norm_eps)
    q_b = pl.jit.inline(make_int8_linear_direct(capacity, LATENT, WIDTH)._func)
    q_rope = make_rope(capacity, FLASH.num_attention_heads, HEAD_DIM, FLASH.qk_rope_head_dim)
    kv = pl.jit.inline(make_int8_linear_direct(capacity, HIDDEN, HEAD_DIM)._func)
    kv_norm = make_rmsnorm(capacity, HEAD_DIM, FLASH.rms_norm_eps)
    kv_rope = make_rope(capacity, 1, HEAD_DIM, FLASH.qk_rope_head_dim)
    attend = pl.jit.inline(make_swa(capacity, MODEL)._func)
    inverse_rope = make_rope(capacity, FLASH.num_attention_heads, HEAD_DIM, FLASH.qk_rope_head_dim)
    grouped = pl.jit.inline(make_grouped_wo_a(capacity, FLASH.num_attention_heads, HEAD_DIM, FLASH.o_groups, FLASH.o_lora_rank)._func)
    output_linear = pl.jit.inline(make_int8_linear_direct(capacity, GROUPED_WIDTH, HIDDEN)._func)

    @pl.jit
    def decode_step(
        value: pl.Tensor[[capacity, HIDDEN], pl.BF16],
        wq_a: pl.Tensor[[HIDDEN, LATENT], pl.INT8],
        wq_a_scale: pl.Tensor[[LATENT], pl.FP32],
        q_norm_weight: pl.Tensor[[LATENT], pl.BF16],
        wq_b: pl.Tensor[[LATENT, WIDTH], pl.INT8],
        wq_b_scale: pl.Tensor[[WIDTH], pl.FP32],
        wkv: pl.Tensor[[HIDDEN, HEAD_DIM], pl.INT8],
        wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[FLASH.o_groups, GROUP_INPUT, FLASH.o_lora_rank], pl.BF16],
        wo_b: pl.Tensor[[FLASH.o_groups * FLASH.o_lora_rank, HIDDEN], pl.INT8],
        wo_b_scale: pl.Tensor[[HIDDEN], pl.FP32],
        cosine: pl.Tensor[[capacity, ROPE_PAIRS], pl.FP32],
        sine: pl.Tensor[[capacity, ROPE_PAIRS], pl.FP32],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        indices: pl.Tensor[[capacity, WINDOW], pl.INT32],
        sink: pl.Tensor[[FLASH.num_attention_heads], pl.FP32],
        positions: pl.Tensor[[capacity], pl.INT64],
        request_ids: pl.Tensor[[capacity], pl.INT32],
        cache_source_ids: pl.Tensor[[capacity], pl.INT32],
        finished: pl.Tensor[[capacity], pl.INT8],
        query_latent: pl.InOut[pl.Tensor[[capacity, LATENT], pl.BF16]],
        query: pl.InOut[pl.Tensor[[capacity, WIDTH], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[capacity, HEAD_DIM], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        attended: pl.InOut[pl.Tensor[[capacity, WIDTH], pl.BF16]],
        output: pl.InOut[pl.Tensor[[capacity, HIDDEN], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        q_a_output = pl.create_tensor([capacity, LATENT], dtype=pl.BF16)
        q_b_output = pl.create_tensor([capacity, WIDTH], dtype=pl.BF16)
        kv_output = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.BF16)
        kv_normed = pl.create_tensor([capacity, HEAD_DIM], dtype=pl.BF16)
        rotated = pl.create_tensor([capacity, WIDTH], dtype=pl.BF16)
        grouped_output = pl.create_tensor([capacity, GROUPED_WIDTH], dtype=pl.BF16)
        inverse_sine = pl.create_tensor([capacity, ROPE_PAIRS], dtype=pl.FP32)
        q_a(value, wq_a, wq_a_scale, q_a_output, active)
        q_norm(q_a_output, q_norm_weight, query_latent, active)
        q_b(query_latent, wq_b, wq_b_scale, q_b_output, active)
        q_rope(q_b_output, cosine, sine, query, active)
        kv(value, wkv, wkv_scale, kv_output, active)
        kv_norm(kv_output, kv_norm_weight, kv_normed, active)
        kv_rope(kv_normed, cosine, sine, window_kv, active)
        attend(
            query,
            window_kv,
            slots,
            indices,
            sink,
            positions,
            request_ids,
            cache_source_ids,
            finished,
            cache,
            attended,
            active,
        )
        for block in pl.spmd(active, name_hint="dsv41_a2a3_decode_inverse_rope"):
            inverse_sine[block:block + 1, :] = pl.mul(sine[block:block + 1, :], -1.0)
        inverse_rope(attended, cosine, inverse_sine, rotated, active)
        grouped(rotated, wo_a, grouped_output, active)
        output_linear(grouped_output, wo_b, wo_b_scale, output, active)
        current_length = pl.read(cache_lens, [0])
        pl.write(cache_lens, [0], pl.cast(current_length + active, pl.INT32))
        return query_latent, query, window_kv, cache, cache_lens, attended, output

    return decode_step


def validate_decode_state(values: dict[str, torch.Tensor | int]) -> None:
    """Validate request, page, slot, and cache controls for one step."""

    import torch

    active = int(values["active"])
    if not 0 <= active <= CAPACITY:
        raise ValueError(f"active must be in [0, {CAPACITY}], got {active}")
    cache = values["cache"]
    cache_lens = values["cache_lens"]
    if cache.shape != (CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM):
        raise ValueError("cache must use the [2, 128, 1, 512] A2/A3 layout")
    if cache_lens.shape != (CACHE_BLOCKS,):
        raise ValueError(f"cache_lens must have shape {(CACHE_BLOCKS,)}, got {tuple(cache_lens.shape)}")
    if torch.any(cache_lens < 0) or torch.any(cache_lens > CACHE_BLOCKS * BLOCK_SIZE):
        raise ValueError("cache_lens contains an invalid physical row count")
    if values["slots"].shape != (CAPACITY, 2) or values["indices"].shape != (CAPACITY, WINDOW):
        raise ValueError("slots and indices must use the fixed Decode capacity and window")
    if values["positions"].shape != (CAPACITY,) or values["request_ids"].shape != (CAPACITY,):
        raise ValueError("position_ids and request_ids must be one row per Decode capacity slot")
    if values["cache_source_ids"].shape != (CAPACITY,) or values["finished"].shape != (CAPACITY,):
        raise ValueError("cache source IDs and finished flags must be one row per Decode capacity slot")
    pages = values["slots"][:, 0]
    offsets = values["slots"][:, 1]
    if torch.any((pages < -1) | (pages >= CACHE_BLOCKS)) or torch.any((offsets < 0) | (offsets >= BLOCK_SIZE)):
        raise ValueError("slot mapping contains an invalid page or offset")
    if torch.any(pages[:active] < 0):
        raise ValueError("active Decode rows must map to a writable cache page")
    if torch.any(values["positions"][:active] < 0):
        raise ValueError("active Decode positions must be non-negative")
    if torch.any(values["request_ids"][:active] < 0) or torch.any(values["cache_source_ids"][:active] < 0):
        raise ValueError("active request and cache source IDs must be non-negative")
    physical = values["indices"]
    if torch.any((physical < -1) | (physical >= CACHE_BLOCKS * BLOCK_SIZE)):
        raise ValueError("page table contains an invalid physical cache row")
    if torch.any(values["finished"][:active].ne(0)):
        raise ValueError("finished requests cannot be dispatched as active Decode rows")


def decode_step_reference(values: dict[str, torch.Tensor | int]) -> None:
    """Reference one SWA Decode step and update only the declared state."""

    from models.deepseek_v4_1_flash_a2.output_proj import output_projection_reference
    from models.deepseek_v4_1_flash_a2.qkv import qkv_reference
    from models.deepseek_v4_1_flash_a2.swa_attention import swa_attention_reference
    from models.deepseek_v4_1_flash_a2.swa_cache import gather_swa_cache_reference, write_swa_cache_reference

    validate_decode_state(values)
    active = int(values["active"])
    qkv_weights = {
        name: values[name]
        for name in ("wq_a", "wq_a_scale", "q_norm_weight", "wq_b", "wq_b_scale", "wkv", "wkv_scale", "kv_norm_weight")
    }
    qkv = qkv_reference(values["value"], qkv_weights, values["cosine"], values["sine"], MODEL)
    cache_after = write_swa_cache_reference(values["cache"], qkv["window_kv"][:active], values["slots"][:active])
    selected = gather_swa_cache_reference(cache_after, values["indices"])
    attended = swa_attention_reference(
        qkv["query"], selected, values["indices"], values["sink"], HEAD_DIM, valid_tokens=active
    )
    projected = output_projection_reference(
        attended,
        values["wo_a"],
        values["wo_b"],
        values["wo_b_scale"],
        values["cosine"],
        values["sine"],
        FLASH.num_attention_heads,
        HEAD_DIM,
        FLASH.qk_rope_head_dim,
    )
    values["query_latent"][:active] = qkv["query_latent"][:active]
    values["query"][:active] = qkv["query"][:active]
    values["window_kv"][:active] = qkv["window_kv"][:active]
    values["attended"][:active] = attended[:active]
    values["output"][:active] = projected["output"][:active]
    values["cache"][:] = cache_after
    values["cache_lens"][0] += active


def _zero_specs() -> list[Any]:
    """Build a deterministic zero fixture for compile and ABI checks."""

    from golden import ScalarSpec, TensorSpec
    import torch

    def spec(name: str, shape: list[int], dtype: torch.dtype, init_value=None):
        value = torch.zeros(shape, dtype=dtype) if init_value is None else init_value
        return TensorSpec(name, shape, dtype, init_value=value)

    slots = torch.zeros((CAPACITY, 2), dtype=torch.int64)
    slots[:, 1] = torch.arange(CAPACITY, dtype=torch.int64)
    indices = torch.full((CAPACITY, WINDOW), -1, dtype=torch.int32)
    indices[:, -1] = torch.arange(CAPACITY, dtype=torch.int32)

    specs = [
        spec("value", [CAPACITY, HIDDEN], torch.bfloat16),
        spec("wq_a", [HIDDEN, LATENT], torch.int8),
        spec("wq_a_scale", [LATENT], torch.float32),
        spec("q_norm_weight", [LATENT], torch.bfloat16),
        spec("wq_b", [LATENT, WIDTH], torch.int8),
        spec("wq_b_scale", [WIDTH], torch.float32),
        spec("wkv", [HIDDEN, HEAD_DIM], torch.int8),
        spec("wkv_scale", [HEAD_DIM], torch.float32),
        spec("kv_norm_weight", [HEAD_DIM], torch.bfloat16),
        spec("wo_a", [FLASH.o_groups, GROUP_INPUT, FLASH.o_lora_rank], torch.bfloat16),
        spec("wo_b", [FLASH.o_groups * FLASH.o_lora_rank, HIDDEN], torch.int8),
        spec("wo_b_scale", [HIDDEN], torch.float32),
        spec("cosine", [CAPACITY, ROPE_PAIRS], torch.float32),
        spec("sine", [CAPACITY, ROPE_PAIRS], torch.float32),
        spec("slots", [CAPACITY, 2], torch.int64, slots),
        spec("indices", [CAPACITY, WINDOW], torch.int32, indices),
        spec("sink", [FLASH.num_attention_heads], torch.float32),
        spec("positions", [CAPACITY], torch.int64),
        spec("request_ids", [CAPACITY], torch.int32),
        spec("cache_source_ids", [CAPACITY], torch.int32),
        spec("finished", [CAPACITY], torch.int8),
        spec("query_latent", [CAPACITY, LATENT], torch.bfloat16),
        spec("query", [CAPACITY, WIDTH], torch.bfloat16),
        spec("window_kv", [CAPACITY, HEAD_DIM], torch.bfloat16),
        spec("cache", [CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16),
        spec("cache_lens", [2], torch.int32),
        spec("attended", [CAPACITY, WIDTH], torch.bfloat16),
        spec("output", [CAPACITY, HIDDEN], torch.bfloat16),
        ScalarSpec("active", torch.int32, 1),
    ]
    return specs


def _golden(values: dict[str, torch.Tensor | int]) -> None:
    """Fill the golden outputs for the deterministic ABI fixture."""

    decode_step_reference(values)


def main(argv: list[str] | None = None) -> int:
    """Compile or execute the bounded single-step Decode entry."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--ep", type=int, default=DECODE_PARALLEL.ep)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--pcp", type=int, default=1)
    parser.add_argument("--dcp", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    default_dp = 2 if args.ep == 2 else 8
    default_tp = 1 if args.ep == 2 else 2
    parallel = DecodeParallelConfig(
        dp=default_dp if args.dp is None else args.dp,
        tp=default_tp if args.tp is None else args.tp,
        ep=args.ep,
        pp=args.pp,
        pcp=args.pcp,
        dcp=args.dcp,
    )
    validate_decode_contract(parallel)
    from golden import run

    result = run(
        fn=make_decode_step(),
        specs=_zero_specs(),
        golden_fn=None if args.compile_only else _golden,
        config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
        rtol=0.05,
        atol=0.05,
        compile_only=args.compile_only,
    )
    if not result.passed:
        print(result.error or result, file=sys.stderr)
        return 1
    print(f"[decode] single-step SWA ABI PASS; topology EP{args.ep}; result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
