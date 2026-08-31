# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 DSpark drafter context KV: project target hidden states into the paged sliding-window cache.

One drafter stage's share of ``precompute_and_store_context_kv``. It runs on every
proposal over the target model's whole token stream for that step, so the token
count is the decode verify width on a decode step and the prompt chunk on a
prefill step. The rows go through this stage's own ``wkv`` / ``kv_norm`` / RoPE and
land in its paged SWA cache at the host-lowered ``slot_mapping`` rows.
"""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FLASH as M,
    KV_ORI_BLOCK_NUM,
    PREFILL_BATCH,
    PREFILL_SEQ,
    TP,
)
from qkv_proj_rope import (
    kv_proj_rope,
    materialize_rope_rows,
    materialize_rope_rows_dynamic,
    rope_prepare,
)


# Dynamic shape variables.
T_DYN = pl.dynamic("DSPARK_CONTEXT_KV_T_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("DSPARK_CONTEXT_KV_ORI_BLOCK_NUM_DYN")

# The public DSpark program supports at most 16 requests with 7 draft rows.
DSPARK_QUERY_TOKENS = 16 * 7
DSPARK_CONTEXT_LAYERS = 3

# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
WIN = M.sliding_window
MAX_SEQ_LEN = M.max_position_embeddings
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
EPS = M.rms_norm_eps


@pl.jit.inline
def dspark_context_kv(
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[DSPARK_CONTEXT_LAYERS, T_DYN], pl.INT64],
    layer_index: pl.Scalar[pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)

    rope_cos_t = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows_dynamic(
        freqs_cos, freqs_sin, position_ids, rope_cos_t, rope_sin_t
    )

    rope_cos_il = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([t_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_t, rope_sin_t, rope_cos_il, rope_sin_signed, rope_swap_idx)

    # This no-work source task seeds the explicit kv_proj dependency chain.
    late_dep = pl.system.task_dummy(deps=[])
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(main_x, wkv, gamma_ckv, rope_cos_il, rope_sin_signed, rope_swap_idx, kv, late_dep)

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_context_kv_scatter"):
        for write_t in pl.range(t_dim):
            write_row_i64 = pl.read(slot_mapping, [layer_index, write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

    return kv_cache


@pl.jit.inline
def dspark_context_kv_query(
    main_x: pl.Tensor[[DSPARK_QUERY_TOKENS, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT32],
    slot_mapping: pl.Tensor[[DSPARK_QUERY_TOKENS], pl.INT64],
    kv_cache: pl.Tensor[[KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
):
    """Project a padded DSpark query-width context without dynamic shape variables."""
    rope_cos_t = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        DSPARK_QUERY_TOKENS,
        rope_cos_t,
        rope_sin_t,
    )

    rope_cos_il = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([DSPARK_QUERY_TOKENS, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_t, rope_sin_t, rope_cos_il, rope_sin_signed, rope_swap_idx)

    kv = pl.create_tensor([DSPARK_QUERY_TOKENS, HEAD_DIM], dtype=pl.BF16)
    late_dep = pl.system.task_dummy(deps=[])
    kv_proj_rope(
        main_x,
        wkv,
        gamma_ckv,
        rope_cos_il,
        rope_sin_signed,
        rope_swap_idx,
        kv,
        late_dep,
    )

    kv_cache_flat = pl.reshape(kv_cache, [KV_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_context_kv_query_scatter"):
        for write_t in pl.range(DSPARK_QUERY_TOKENS):
            write_row_i64 = pl.read(slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]


@pl.jit
def dspark_context_kv_test(
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    slot_mapping: pl.Tensor[[DSPARK_CONTEXT_LAYERS, T_DYN], pl.INT64],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
):
    main_x.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    slot_mapping.bind_dynamic(1, T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    return dspark_context_kv(
        main_x,
        wkv,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        position_ids,
        slot_mapping,
        pl.const(0, pl.INT32),
        kv_cache,
    )


def golden_dspark_context_kv(tensors):
    import torch

    main_x = tensors["main_x"].to(torch.bfloat16).float()
    wkv = tensors["wkv"].to(torch.bfloat16).float()
    kv_proj = torch.matmul(main_x, wkv)
    inv_rms = torch.rsqrt(kv_proj.square().mean(-1, keepdim=True) + EPS)
    kv_full = kv_proj * inv_rms * tensors["gamma_ckv"].float()

    positions = tensors["position_ids"].long()
    rope_cos = tensors["freqs_cos"].index_select(0, positions).float()[:, :ROPE_HALF]
    rope_sin = tensors["freqs_sin"].index_select(0, positions).float()[:, :ROPE_HALF]
    rope_pairs = kv_full[:, NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pairs[..., 0]
    rope_odd = rope_pairs[..., 1]
    rotated_even = (rope_even * rope_cos - rope_odd * rope_sin).to(torch.bfloat16)
    rotated_odd = (rope_even * rope_sin + rope_odd * rope_cos).to(torch.bfloat16)
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1).flatten(-2)

    kv = torch.cat([kv_full[:, :NOPE_DIM], rotated.float()], dim=-1).to(torch.bfloat16)
    kv_cache_flat = tensors["kv_cache"].view(-1, HEAD_DIM)
    slots = tensors["slot_mapping"][0]
    for token_idx in range(kv.shape[0]):
        cache_row = int(slots[token_idx].item())
        if cache_row >= 0:
            kv_cache_flat[cache_row] = kv[token_idx]


def build_tensor_specs(batch, seq):
    import torch
    from golden import TensorSpec
    from utils import (
        block_table,
        build_rope_tables,
        paged_slot_mapping,
        position_ids_from_starts,
        swa_decode_start_set,
    )

    t = batch * seq
    freqs_cos, freqs_sin = build_rope_tables(M, 0, dtype=torch.bfloat16)

    def init_start_pos():
        if seq == DECODE_SEQ:
            return swa_decode_start_set(batch=batch, window=WIN)
        return torch.zeros(batch, dtype=torch.int32)

    def init_block_table():
        return block_table(batch=batch, table_blocks=ORI_MAX_BLOCKS, physical_blocks=KV_ORI_BLOCK_NUM)

    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=seq).reshape(-1).contiguous()

    def init_slot_mapping():
        slots = paged_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=seq), init_block_table(), block_size=BLOCK_SIZE
        ).reshape(-1).contiguous()
        return slots.unsqueeze(0).expand(DSPARK_CONTEXT_LAYERS, -1).contiguous()

    def init_main_x():
        return torch.randn(t, D, dtype=torch.bfloat16) * 0.05

    def init_wkv():
        return (torch.randn(D, HEAD_DIM) / D ** 0.5).to(torch.bfloat16)

    def init_kv_cache():
        cache = torch.randn(KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        rms = cache.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / rms).to(torch.bfloat16)

    return [
        TensorSpec("main_x", [t, D], torch.bfloat16, init_value=init_main_x),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.ones(HEAD_DIM)),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=lambda: freqs_cos.clone()),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=lambda: freqs_sin.clone()),
        TensorSpec("position_ids", [t], torch.int32, init_value=init_position_ids),
        TensorSpec(
            "slot_mapping",
            [DSPARK_CONTEXT_LAYERS, t],
            torch.int64,
            init_value=init_slot_mapping,
        ),
        TensorSpec(
            "kv_cache",
            [KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
            is_output=True,
        ),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="DeepSeek-V4 DSpark drafter context-KV validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes = {
        "decode": (DECODE_BATCH // TP, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }
    for mode in (modes if args.mode == "all" else [args.mode]):
        batch, seq = modes[mode]
        print(f"--- dspark_context_kv_test {mode}: T={batch * seq} ---")
        result = run_jit(
            fn=dspark_context_kv_test,
            specs=build_tensor_specs(batch, seq),
            golden_fn=golden_dspark_context_kv,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            },
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
