# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Continuous A2/A3 SWA Decode state replay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl

from models.deepseek_v4_1_flash_a2.decode_fwd import (
    BLOCK_SIZE,
    CACHE_BLOCKS,
    CAPACITY,
    GROUPED_WIDTH,
    HEAD_DIM,
    HIDDEN,
    LATENT,
    ROPE_PAIRS,
    WIDTH,
    WINDOW,
    decode_step_reference,
    make_decode_step,
    validate_decode_state,
)
from models.deepseek_v4_1_flash_a2.config import FLASH


def make_decode_replay(steps: int, capacity: int = CAPACITY):
    """Build a fixed-step kernel that carries one cache through all calls."""

    if steps not in (1, 2, 8):
        raise ValueError("continuous replay supports exactly 1, 2, or 8 steps")
    step = pl.jit.inline(make_decode_step(capacity)._func)

    @pl.jit
    def decode_replay(
        value: pl.Tensor[[steps, capacity, HIDDEN], pl.BF16],
        wq_a: pl.Tensor[[HIDDEN, LATENT], pl.INT8],
        wq_a_scale: pl.Tensor[[LATENT], pl.FP32],
        q_norm_weight: pl.Tensor[[LATENT], pl.BF16],
        wq_b: pl.Tensor[[LATENT, WIDTH], pl.INT8],
        wq_b_scale: pl.Tensor[[WIDTH], pl.FP32],
        wkv: pl.Tensor[[HIDDEN, HEAD_DIM], pl.INT8],
        wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        wo_a: pl.Tensor[[FLASH.o_groups, WIDTH // FLASH.o_groups, FLASH.o_lora_rank], pl.BF16],
        wo_b: pl.Tensor[[GROUPED_WIDTH, HIDDEN], pl.INT8],
        wo_b_scale: pl.Tensor[[HIDDEN], pl.FP32],
        cosine: pl.Tensor[[steps, capacity, ROPE_PAIRS], pl.FP32],
        sine: pl.Tensor[[steps, capacity, ROPE_PAIRS], pl.FP32],
        slots: pl.Tensor[[steps, capacity, 2], pl.INT64],
        indices: pl.Tensor[[steps, capacity, WINDOW], pl.INT32],
        sink: pl.Tensor[[FLASH.num_attention_heads], pl.FP32],
        positions: pl.Tensor[[steps, capacity], pl.INT64],
        request_ids: pl.Tensor[[steps, capacity], pl.INT32],
        cache_source_ids: pl.Tensor[[steps, capacity], pl.INT32],
        finished: pl.Tensor[[steps, capacity], pl.INT8],
        query_latent: pl.InOut[pl.Tensor[[steps, capacity, LATENT], pl.BF16]],
        query: pl.InOut[pl.Tensor[[steps, capacity, WIDTH], pl.BF16]],
        window_kv: pl.InOut[pl.Tensor[[steps, capacity, HEAD_DIM], pl.BF16]],
        cache: pl.InOut[pl.Tensor[[CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
        cache_lens: pl.InOut[pl.Tensor[[2], pl.INT32]],
        attended: pl.InOut[pl.Tensor[[steps, capacity, WIDTH], pl.BF16]],
        output: pl.InOut[pl.Tensor[[steps, capacity, HIDDEN], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        for step_index in pl.unroll(steps):
            step(
                value[step_index, :, :],
                wq_a,
                wq_a_scale,
                q_norm_weight,
                wq_b,
                wq_b_scale,
                wkv,
                wkv_scale,
                kv_norm_weight,
                wo_a,
                wo_b,
                wo_b_scale,
                cosine[step_index, :, :],
                sine[step_index, :, :],
                slots[step_index, :, :],
                indices[step_index, :, :],
                sink,
                positions[step_index, :],
                request_ids[step_index, :],
                cache_source_ids[step_index, :],
                finished[step_index, :],
                query_latent[step_index, :, :],
                query[step_index, :, :],
                window_kv[step_index, :, :],
                cache,
                cache_lens,
                attended[step_index, :, :],
                output[step_index, :, :],
                active,
            )
        return query_latent, query, window_kv, cache, cache_lens, attended, output

    return decode_replay


def _replay_inputs(steps: int) -> dict[str, torch.Tensor | int]:
    """Create a deterministic one-request state sequence for ABI replay."""

    import torch

    slots = torch.full((steps, CAPACITY, 2), -1, dtype=torch.int64)
    slots[:, :, 1] = 0
    indices = torch.full((steps, CAPACITY, WINDOW), -1, dtype=torch.int32)
    for step in range(steps):
        slots[step, 0] = torch.tensor([0, step], dtype=torch.int64)
        # Read the complete history available at this point, including the
        # row published by the current step.  This keeps the replay bounded
        # while exercising paged cache gather across more than one token.
        history = torch.arange(step + 1, dtype=torch.int32)
        indices[step, 0, -(step + 1):] = history
    values: dict[str, torch.Tensor | int] = {
        "value": torch.ones((steps, CAPACITY, HIDDEN), dtype=torch.bfloat16),
        "wq_a": torch.zeros((HIDDEN, LATENT), dtype=torch.int8),
        "wq_a_scale": torch.zeros(LATENT, dtype=torch.float32),
        "q_norm_weight": torch.zeros(LATENT, dtype=torch.bfloat16),
        "wq_b": torch.zeros((LATENT, WIDTH), dtype=torch.int8),
        "wq_b_scale": torch.zeros(WIDTH, dtype=torch.float32),
        "wkv": torch.ones((HIDDEN, HEAD_DIM), dtype=torch.int8),
        "wkv_scale": torch.full((HEAD_DIM,), 1.0 / HIDDEN, dtype=torch.float32),
        "kv_norm_weight": torch.ones(HEAD_DIM, dtype=torch.bfloat16),
        "wo_a": torch.zeros((FLASH.o_groups, WIDTH // FLASH.o_groups, FLASH.o_lora_rank), dtype=torch.bfloat16),
        "wo_b": torch.zeros((GROUPED_WIDTH, HIDDEN), dtype=torch.int8),
        "wo_b_scale": torch.zeros(HIDDEN, dtype=torch.float32),
        "cosine": torch.ones((steps, CAPACITY, ROPE_PAIRS), dtype=torch.float32),
        "sine": torch.zeros((steps, CAPACITY, ROPE_PAIRS), dtype=torch.float32),
        "slots": slots,
        "indices": indices,
        "sink": torch.zeros(FLASH.num_attention_heads, dtype=torch.float32),
        "positions": torch.arange(steps, dtype=torch.int64).reshape(steps, 1).expand(steps, CAPACITY).contiguous(),
        "request_ids": torch.zeros((steps, CAPACITY), dtype=torch.int32),
        "cache_source_ids": torch.zeros((steps, CAPACITY), dtype=torch.int32),
        "finished": torch.zeros((steps, CAPACITY), dtype=torch.int8),
        "query_latent": torch.zeros((steps, CAPACITY, LATENT), dtype=torch.bfloat16),
        "query": torch.zeros((steps, CAPACITY, WIDTH), dtype=torch.bfloat16),
        "window_kv": torch.zeros((steps, CAPACITY, HEAD_DIM), dtype=torch.bfloat16),
        "cache": torch.zeros((CACHE_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM), dtype=torch.bfloat16),
        "cache_lens": torch.zeros(2, dtype=torch.int32),
        "attended": torch.zeros((steps, CAPACITY, WIDTH), dtype=torch.bfloat16),
        "output": torch.zeros((steps, CAPACITY, HIDDEN), dtype=torch.bfloat16),
        "active": 1,
    }
    return values


def _specs(steps: int) -> list[Any]:
    from golden import ScalarSpec, TensorSpec

    import torch

    specs: list[Any] = []
    for name, value in _replay_inputs(steps).items():
        if name == "active":
            specs.append(ScalarSpec(name, torch.int32, int(value)))
        else:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"replay tensor {name!r} has type {type(value).__name__}")
            specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    return specs


def _golden(values: dict[str, torch.Tensor | int]) -> None:
    steps = int(values["value"].shape[0])
    import torch

    validate_decode_state({
        key: value[0] if isinstance(value, torch.Tensor) and value.ndim > 0 and key in {
            "value", "cosine", "sine", "slots", "indices", "positions", "request_ids",
            "cache_source_ids", "finished", "query_latent", "query", "window_kv", "attended", "output",
        } else value
        for key, value in values.items()
    })
    for step in range(steps):
        one = dict(values)
        for name in (
            "value", "cosine", "sine", "slots", "indices", "positions", "request_ids",
            "cache_source_ids", "finished", "query_latent", "query", "window_kv", "attended", "output",
        ):
            tensor = values[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"replay tensor {name!r} has type {type(tensor).__name__}")
            one[name] = tensor[step]
        decode_step_reference(one)
    cache_lens = values["cache_lens"]
    if not isinstance(cache_lens, torch.Tensor):
        raise TypeError("cache_lens must be a tensor")
    expected = torch.tensor([steps, 0], dtype=torch.int32)
    if not torch.equal(cache_lens, expected):
        raise ValueError(f"cache_lens state mismatch: got {cache_lens.tolist()}, expected {expected.tolist()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, choices=(1, 2, 8), required=True)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a5"), default="a2a3")
    parser.add_argument("--sim", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    from golden import run

    result = run(
        fn=make_decode_replay(args.steps),
        specs=_specs(args.steps),
        golden_fn=None if args.compile_only else _golden,
        config={"backend_type": args.platform, "execution_mode": args.sim, "device_id": args.device},
        rtol=0.05,
        atol=0.05,
        compile_only=args.compile_only,
    )
    if not result.passed:
        print(result.error or result, file=sys.stderr)
        return 1
    print(f"[decode-replay] {args.steps}-step state/ABI PASS; checkpoint accuracy not claimed; result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
