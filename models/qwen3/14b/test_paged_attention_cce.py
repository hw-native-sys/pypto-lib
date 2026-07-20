# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""On-device correctness and performance driver for Qwen decode attention."""

import argparse
import math

import torch

from golden import TensorSpec, run_jit
from paged_attention_cce import (
    BATCH,
    BLOCK_SIZE,
    CAUSAL_MASK_SIZE,
    HEAD_DIM,
    KV_HIDDEN,
    NUM_HEADS,
    NUM_KV_HEADS,
    SUPPORTED_PLATFORMS,
    qwen_decode_attention_cache_offset_test,
    qwen_decode_attention_cce,
    qwen_prefill_attention_cce,
)


def build_specs(
    batch: int,
    context_len: int,
    capacity: int,
    *,
    random_data: bool,
    seq_lens: torch.Tensor | None = None,
    cache_layers: int = 1,
) -> list[TensorSpec]:
    max_blocks = math.ceil(capacity / BLOCK_SIZE)
    layer_num_blocks = batch * max_blocks
    num_blocks = cache_layers * layer_num_blocks
    init = torch.randn if random_data else torch.zeros
    seq_lens = seq_lens if seq_lens is not None else torch.full([batch], context_len, dtype=torch.int32)
    return [
        TensorSpec(
            "query",
            [batch, NUM_HEADS, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: init([batch, NUM_HEADS, HEAD_DIM]),
        ),
        TensorSpec(
            "key_cache",
            [num_blocks, BLOCK_SIZE, KV_HIDDEN],
            torch.bfloat16,
            init_value=lambda: init([num_blocks, BLOCK_SIZE, KV_HIDDEN]),
        ),
        TensorSpec(
            "value_cache",
            [num_blocks, BLOCK_SIZE, KV_HIDDEN],
            torch.bfloat16,
            init_value=lambda: init([num_blocks, BLOCK_SIZE, KV_HIDDEN]),
        ),
        TensorSpec(
            "block_table",
            [batch, max_blocks],
            torch.int32,
            init_value=lambda: torch.arange(layer_num_blocks, dtype=torch.int32).reshape(batch, max_blocks),
        ),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens),
        TensorSpec(
            "out",
            [batch, NUM_HEADS, HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def build_prefill_specs(
    batch: int,
    context_len: int,
    query_len: int,
    capacity: int,
    *,
    random_data: bool,
) -> list[TensorSpec]:
    max_blocks = math.ceil(capacity / BLOCK_SIZE)
    layer_num_blocks = batch * max_blocks
    init = torch.randn if random_data else torch.zeros
    q_lens = torch.full([batch], query_len, dtype=torch.int32)
    kv_lens = torch.full([batch], context_len, dtype=torch.int32)
    total_q = batch * query_len
    causal_mask = torch.triu(
        torch.ones([CAUSAL_MASK_SIZE, CAUSAL_MASK_SIZE], dtype=torch.int8),
        diagonal=1,
    )
    return [
        TensorSpec(
            "query",
            [total_q, NUM_HEADS, HEAD_DIM],
            torch.bfloat16,
            init_value=lambda: init([total_q, NUM_HEADS, HEAD_DIM]),
        ),
        TensorSpec(
            "key_cache",
            [layer_num_blocks, BLOCK_SIZE, KV_HIDDEN],
            torch.bfloat16,
            init_value=lambda: init([layer_num_blocks, BLOCK_SIZE, KV_HIDDEN]),
        ),
        TensorSpec(
            "value_cache",
            [layer_num_blocks, BLOCK_SIZE, KV_HIDDEN],
            torch.bfloat16,
            init_value=lambda: init([layer_num_blocks, BLOCK_SIZE, KV_HIDDEN]),
        ),
        TensorSpec(
            "block_table",
            [batch, max_blocks],
            torch.int32,
            init_value=lambda: torch.arange(layer_num_blocks, dtype=torch.int32).reshape(batch, max_blocks),
        ),
        TensorSpec("q_lens", [batch], torch.int32, init_value=q_lens),
        TensorSpec("kv_lens", [batch], torch.int32, init_value=kv_lens),
        TensorSpec(
            "causal_mask",
            [CAUSAL_MASK_SIZE, CAUSAL_MASK_SIZE],
            torch.int8,
            init_value=lambda: causal_mask,
        ),
        TensorSpec(
            "out",
            [total_q, NUM_HEADS, HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def golden_attention(
    values: dict[str, torch.Tensor],
    *,
    cache_block_offset: int = 0,
) -> None:
    query = values["query"].float()
    key_cache = values["key_cache"].reshape(-1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)[cache_block_offset:]
    value_cache = values["value_cache"].reshape(-1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)[cache_block_offset:]
    block_table = values["block_table"]
    seq_lens = values["seq_lens"]
    out = torch.empty_like(query)
    heads_per_kv = NUM_HEADS // NUM_KV_HEADS
    scale = 1.0 / math.sqrt(HEAD_DIM)

    for batch_idx in range(query.shape[0]):
        seq_len = int(seq_lens[batch_idx].item())
        block_count = math.ceil(seq_len / BLOCK_SIZE)
        block_ids = block_table[batch_idx, :block_count].long()
        keys = key_cache[block_ids].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:seq_len]
        vals = value_cache[block_ids].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:seq_len]
        keys = keys.repeat_interleave(heads_per_kv, dim=1).float()
        vals = vals.repeat_interleave(heads_per_kv, dim=1).float()
        scores = torch.einsum("hd,shd->hs", query[batch_idx], keys) * scale
        probs = torch.softmax(scores, dim=-1)
        out[batch_idx] = torch.einsum("hs,shd->hd", probs, vals)
    values["out"][:] = out.to(torch.bfloat16)


def golden_prefill_attention(values: dict[str, torch.Tensor]) -> None:
    query = values["query"].float()
    key_cache = values["key_cache"].reshape(-1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    value_cache = values["value_cache"].reshape(-1, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM)
    block_table = values["block_table"]
    q_lens = values["q_lens"]
    kv_lens = values["kv_lens"]
    out = torch.empty_like(query)
    heads_per_kv = NUM_HEADS // NUM_KV_HEADS
    scale = 1.0 / math.sqrt(HEAD_DIM)
    q_base = 0

    for batch_idx in range(q_lens.shape[0]):
        q_len = int(q_lens[batch_idx].item())
        kv_len = int(kv_lens[batch_idx].item())
        block_count = math.ceil(kv_len / BLOCK_SIZE)
        block_ids = block_table[batch_idx, :block_count].long()
        keys = key_cache[block_ids].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:kv_len]
        vals = value_cache[block_ids].reshape(-1, NUM_KV_HEADS, HEAD_DIM)[:kv_len]
        keys = keys.repeat_interleave(heads_per_kv, dim=1).float()
        vals = vals.repeat_interleave(heads_per_kv, dim=1).float()
        q_tile = query[q_base : q_base + q_len]
        q_start = kv_len - q_len
        for rel_t in range(q_len):
            ctx_len = q_start + rel_t + 1
            scores = torch.einsum("hd,shd->hs", q_tile[rel_t], keys[:ctx_len]) * scale
            probs = torch.softmax(scores, dim=-1)
            out[q_base + rel_t] = torch.einsum("hs,shd->hd", probs, vals[:ctx_len])
        q_base += q_len
    values["out"][:] = out.to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=SUPPORTED_PLATFORMS,
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1, choices=[1, BATCH])
    parser.add_argument("--context-len", type=int, default=3338)
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument(
        "--check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare device output with the golden result",
    )
    parser.add_argument("--ragged", action="store_true")
    parser.add_argument("--cache-offset-test", action="store_true")
    parser.add_argument("--prefill", action="store_true")
    parser.add_argument("--query-len", type=int, default=64)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--enable-l2-swimlane", action="store_true")
    args = parser.parse_args()
    if not 0 < args.context_len <= args.capacity:
        raise ValueError("context length must be in (0, capacity]")
    if not 0 < args.query_len <= args.context_len:
        raise ValueError("query length must be in (0, context_len]")
    if args.ragged and args.batch == 1:
        raise ValueError("ragged lengths require batch 16")

    torch.manual_seed(1234)
    if args.prefill:
        specs = build_prefill_specs(
            args.batch,
            args.context_len,
            args.query_len,
            args.capacity,
            random_data=args.check,
        )
        result = run_jit(
            fn=qwen_prefill_attention_cce,
            specs=specs,
            golden_fn=golden_prefill_attention if args.check else None,
            runtime_cfg={
                "platform": args.platform,
                "device_id": args.device,
                "enable_l2_swimlane": args.enable_l2_swimlane,
            },
            compile_only=args.compile_only or args.platform.endswith("sim"),
            rtol=5e-3,
            atol=2e-2,
            save_data=False,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
        return

    seq_lens = None
    if args.ragged:
        step = max(1, args.context_len // (args.batch * 2))
        seq_lens = torch.tensor(
            [max(1, args.context_len - idx * step) for idx in range(args.batch)],
            dtype=torch.int32,
        )
    cache_layers = 2 if args.cache_offset_test else 1
    specs = build_specs(
        args.batch,
        args.context_len,
        args.capacity,
        random_data=args.check,
        seq_lens=seq_lens,
        cache_layers=cache_layers,
    )
    fn = qwen_decode_attention_cache_offset_test if args.cache_offset_test else qwen_decode_attention_cce
    if args.check and args.cache_offset_test:
        layer_num_blocks = args.batch * math.ceil(args.capacity / BLOCK_SIZE)
        golden_fn = lambda values: golden_attention(
            values,
            cache_block_offset=layer_num_blocks,
        )
    else:
        golden_fn = golden_attention if args.check else None

    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        runtime_cfg={
            "platform": args.platform,
            "device_id": args.device,
            "enable_l2_swimlane": args.enable_l2_swimlane,
        },
        compile_only=args.compile_only or args.platform.endswith("sim"),
        rtol=5e-3,
        atol=2e-2,
        save_data=False,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
