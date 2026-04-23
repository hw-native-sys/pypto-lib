# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B decode Scope 2 — q/k norm + RoPE + paged KV update + decode attention.

Implements the PagedAttention contract described in ``pa_impl.md``:
- ``block_table`` maps logical attention blocks to physical cache blocks.
- ``slot_mapping`` identifies the write slot for the current decode token.

Due to current backend scalar-load lowering, ``block_table`` is passed as a
flattened 1D tensor of length ``batch * max_blocks_per_seq`` instead of a 2D
``[batch, max_blocks_per_seq]`` view. The logical mapping is unchanged.

The device path stores KV cache as a 2D tensor with logical row order
``[physical_block, kv_head, token_in_block]`` flattened to
``[num_blocks * num_kv_heads * BLOCK_SIZE, head_dim]``. That lets each paged
K/V block for one KV head be read back as a single contiguous
``[BLOCK_SIZE, head_dim]`` slice without extra staging buffers.

Input projections are FP32; q_norm/k_norm weights are FP32; KV caches are BF16.
Output attention is BF16.
"""

from __future__ import annotations

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM

EPS = 1e-6
HEAD_DIM_INV = 1.0 / HEAD_DIM
Q_HEAD_BATCH = 5
Q_HEAD_PAD = 16
SEQ_TILE = 64
BATCH_TILE = 16
SB_BATCH = 64
BLOCK_SIZE = SEQ_TILE


def build_qwen3_scope2_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    q_per_kv = num_heads // num_kv_heads
    half_dim = head_dim // 2
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_scale = 1.0 / (head_dim ** 0.5)
    max_ctx_blocks = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks_per_seq = max_ctx_blocks
    num_blocks = batch * max_blocks_per_seq

    @pl.program
    class Qwen3Scope2:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_scope2(
            self,
            q_proj: pl.Tensor[[batch, hidden], pl.FP32],
            k_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            v_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            q_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            k_norm_weight: pl.Tensor[[1, head_dim], pl.FP32],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            block_table: pl.Tensor[[batch * max_blocks_per_seq], pl.INT32],
            slot_mapping: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[num_blocks * num_kv_heads * BLOCK_SIZE, head_dim], pl.BF16],
            v_cache: pl.Tensor[[num_blocks * num_kv_heads * BLOCK_SIZE, head_dim], pl.BF16],
            attn_out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            q_proj_norm = pl.create_tensor([batch, hidden], dtype=pl.FP32)
            k_proj_norm = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)

            for b0 in pl.range(0, batch, BATCH_TILE):
                with pl.at(level=pl.Level.CORE_GROUP):
                    for h in pl.range(num_heads):
                        q0 = h * head_dim
                        q_chunk = pl.slice(q_proj, [BATCH_TILE, head_dim], [b0, q0])
                        q_sq_sum = pl.row_sum(pl.mul(q_chunk, q_chunk))
                        q_inv_rms = pl.rsqrt(pl.add(pl.mul(q_sq_sum, HEAD_DIM_INV), EPS))
                        q_chunk_norm = pl.col_expand_mul(
                            pl.row_expand_mul(q_chunk, q_inv_rms),
                            q_norm_weight,
                        )
                        q_proj_norm = pl.assemble(q_proj_norm, q_chunk_norm, [b0, q0])

                    for h in pl.range(num_kv_heads):
                        k0 = h * head_dim
                        k_chunk = pl.slice(k_proj, [BATCH_TILE, head_dim], [b0, k0])
                        k_sq_sum = pl.row_sum(pl.mul(k_chunk, k_chunk))
                        k_inv_rms = pl.rsqrt(pl.add(pl.mul(k_sq_sum, HEAD_DIM_INV), EPS))
                        k_chunk_norm = pl.col_expand_mul(
                            pl.row_expand_mul(k_chunk, k_inv_rms),
                            k_norm_weight,
                        )
                        k_proj_norm = pl.assemble(k_proj_norm, k_chunk_norm, [b0, k0])

            all_q_padded = pl.create_tensor([batch * total_q_groups * Q_HEAD_PAD, head_dim], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP):
                for idx in pl.range(batch * total_q_groups):
                    all_q_padded = pl.assemble(
                        all_q_padded,
                        pl.cast(
                            pl.full([Q_HEAD_PAD - Q_HEAD_BATCH, head_dim], dtype=pl.FP32, value=0.0),
                            target_type=pl.BF16,
                        ),
                        [idx * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                    )

            for b in pl.range(batch):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE
                block_table_base = b * max_blocks_per_seq
                slot = pl.tensor.read(slot_mapping, [b])
                slot_block = slot // BLOCK_SIZE
                slot_offset = slot - slot_block * BLOCK_SIZE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ki in pl.parallel(0, num_kv_heads, chunk=8):
                        kv_col = ki * head_dim
                        cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
                        k_lo = pl.slice(k_proj_norm, [1, half_dim], [b, kv_col])
                        k_hi = pl.slice(k_proj_norm, [1, half_dim], [b, kv_col + half_dim])
                        rot_lo = pl.sub(
                            pl.col_expand_mul(k_lo, cos_lo),
                            pl.col_expand_mul(k_hi, sin_lo),
                        )
                        rot_hi = pl.add(
                            pl.col_expand_mul(k_hi, cos_hi),
                            pl.col_expand_mul(k_lo, sin_hi),
                        )
                        k_cache = pl.assemble(
                            k_cache,
                            pl.cast(rot_lo, target_type=pl.BF16),
                            [cache_row, 0],
                        )
                        k_cache = pl.assemble(
                            k_cache,
                            pl.cast(rot_hi, target_type=pl.BF16),
                            [cache_row, half_dim],
                        )
                        v_cache = pl.assemble(
                            v_cache,
                            pl.cast(pl.slice(v_proj, [1, head_dim], [b, kv_col]), target_type=pl.BF16),
                            [cache_row, 0],
                        )
                        q_base = ki * q_per_kv
                        for qi in pl.range(Q_HEAD_BATCH):
                            q_col = (q_base + qi) * head_dim
                            q_lo = pl.slice(q_proj_norm, [1, half_dim], [b, q_col])
                            q_hi = pl.slice(q_proj_norm, [1, half_dim], [b, q_col + half_dim])
                            rot_lo_bf16 = pl.cast(
                                pl.sub(
                                    pl.col_expand_mul(q_lo, cos_lo),
                                    pl.col_expand_mul(q_hi, sin_lo),
                                ),
                                target_type=pl.BF16,
                            )
                            rot_hi_bf16 = pl.cast(
                                pl.add(
                                    pl.col_expand_mul(q_hi, cos_hi),
                                    pl.col_expand_mul(q_lo, sin_hi),
                                ),
                                target_type=pl.BF16,
                            )
                            all_q_padded = pl.assemble(
                                all_q_padded,
                                rot_lo_bf16,
                                [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, 0],
                            )
                            all_q_padded = pl.assemble(
                                all_q_padded,
                                rot_hi_bf16,
                                [b * total_q_groups * Q_HEAD_PAD + ki * Q_HEAD_PAD + qi, half_dim],
                            )

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                attn_row_padded = pl.create_tensor(
                    [1, total_q_groups * Q_HEAD_PAD * head_dim],
                    dtype=pl.BF16,
                )
                for gi in pl.range(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(
                        all_q_padded,
                        [Q_HEAD_PAD, head_dim],
                        [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0],
                    )
                    all_raw_scores = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, BLOCK_SIZE], dtype=pl.BF16)
                    all_oi_tmp = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, head_dim], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([max_ctx_blocks * Q_HEAD_PAD, 1], dtype=pl.FP32)

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            block_table_idx = block_table_base + sb
                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                            k_tile = pl.slice(k_cache, [BLOCK_SIZE, head_dim], [cache_row0, 0])
                            raw_scores = pl.matmul(q_padded, k_tile, b_trans=True, out_dtype=pl.FP32)
                            all_raw_scores = pl.assemble(all_raw_scores, raw_scores, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            s0 = sb * BLOCK_SIZE
                            valid_len = pl.min(BLOCK_SIZE, ctx_len - s0)
                            scores_valid = pl.slice(
                                all_raw_scores,
                                [Q_HEAD_PAD, BLOCK_SIZE],
                                [sb * Q_HEAD_PAD, 0],
                                valid_shape=[Q_HEAD_PAD, valid_len],
                            )
                            scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                            scores = pl.mul(scores_padded, attn_scale)
                            cur_mi = pl.row_max(scores)
                            exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                            exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                            exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                            cur_li = pl.row_sum(exp_scores_fp32)
                            all_exp_padded = pl.assemble(all_exp_padded, exp_scores_bf16, [sb * Q_HEAD_PAD, 0])
                            all_cur_mi = pl.assemble(all_cur_mi, cur_mi, [sb * Q_HEAD_PAD, 0])
                            all_cur_li = pl.assemble(all_cur_li, cur_li, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                        for sb in pl.parallel(ctx_blocks, chunk=SB_BATCH):
                            block_table_idx = block_table_base + sb
                            pbid = pl.cast(pl.tensor.read(block_table, [block_table_idx]), pl.INDEX)
                            cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                            exp_tile = pl.slice(
                                all_exp_padded,
                                [Q_HEAD_PAD, BLOCK_SIZE],
                                [sb * Q_HEAD_PAD, 0],
                            )
                            v_tile = pl.slice(v_cache, [BLOCK_SIZE, head_dim], [cache_row0, 0])
                            oi_tmp = pl.matmul(exp_tile, v_tile, out_dtype=pl.FP32)
                            all_oi_tmp = pl.assemble(all_oi_tmp, oi_tmp, [sb * Q_HEAD_PAD, 0])

                    with pl.at(level=pl.Level.CORE_GROUP):
                        oi = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [0, 0])
                        mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [0, 0])
                        li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [0, 0])
                        for sb in pl.range(1, ctx_blocks):
                            oi_tmp_valid = pl.slice(all_oi_tmp, [Q_HEAD_PAD, head_dim], [sb * Q_HEAD_PAD, 0])
                            cur_mi = pl.slice(all_cur_mi, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                            cur_li = pl.slice(all_cur_li, [Q_HEAD_PAD, 1], [sb * Q_HEAD_PAD, 0])
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                            oi = pl.add(
                                pl.row_expand_mul(oi, alpha),
                                pl.row_expand_mul(oi_tmp_valid, beta),
                            )
                            mi = mi_new
                        ctx = pl.row_expand_div(oi, li)
                        ctx_flat_padded = pl.reshape(ctx, [1, Q_HEAD_PAD * head_dim])
                        ctx_flat_padded_bf16 = pl.cast(ctx_flat_padded, target_type=pl.BF16)
                        attn_row_padded = pl.assemble(
                            attn_row_padded,
                            ctx_flat_padded_bf16,
                            [0, gi * Q_HEAD_PAD * head_dim],
                        )

                for gi in pl.range(total_q_groups):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    with pl.at(level=pl.Level.CORE_GROUP):
                        ctx_flat_bf16 = pl.slice(
                            attn_row_padded,
                            [1, Q_HEAD_BATCH * head_dim],
                            [0, gi * Q_HEAD_PAD * head_dim],
                        )
                        attn_row = pl.assemble(attn_row, ctx_flat_bf16, [0, q_base * head_dim])

                attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            return attn_out

    return Qwen3Scope2


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    use_max_seq: bool = False,
):
    import torch
    from golden import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    max_ctx_blocks = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks_per_seq = max_ctx_blocks
    num_blocks = batch * max_blocks_per_seq
    cache_rows = num_blocks * num_kv_heads * BLOCK_SIZE

    if use_max_seq:
        seq_lens_seed = torch.full((batch,), max_seq, dtype=torch.int32)
    else:
        seq_lens_seed = torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    def init_seq_lens():
        return seq_lens_seed.clone()

    def init_q_proj():
        return torch.rand(batch, hidden) - 0.5

    def init_k_proj():
        return torch.rand(batch, kv_hidden) - 0.5

    def init_v_proj():
        return torch.rand(batch, kv_hidden) - 0.5

    def init_q_norm_weight():
        return torch.ones(1, head_dim)

    def init_k_norm_weight():
        return torch.ones(1, head_dim)

    def init_block_table():
        block_ids = torch.arange(num_blocks, dtype=torch.int32)
        return block_ids.clone()

    def init_slot_mapping():
        slots = torch.empty(batch, dtype=torch.int32)
        for b in range(batch):
            pos = int(seq_lens_seed[b].item()) - 1
            logical_block = pos // BLOCK_SIZE
            page_offset = pos % BLOCK_SIZE
            phys_block = b * max_blocks_per_seq + logical_block
            slots[b] = phys_block * BLOCK_SIZE + page_offset
        return slots

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    return [
        TensorSpec("q_proj", [batch, hidden], torch.float32, init_value=init_q_proj),
        TensorSpec("k_proj", [batch, kv_hidden], torch.float32, init_value=init_k_proj),
        TensorSpec("v_proj", [batch, kv_hidden], torch.float32, init_value=init_v_proj),
        TensorSpec("q_norm_weight", [1, head_dim], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("k_norm_weight", [1, head_dim], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("block_table", [batch * max_blocks_per_seq], torch.int32, init_value=init_block_table),
        TensorSpec("slot_mapping", [batch], torch.int32, init_value=init_slot_mapping),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32, init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_v_cache),
        TensorSpec("attn_out", [batch, hidden], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_scope2(tensors):
    import math
    import torch

    q_proj = tensors["q_proj"]
    k_proj = tensors["k_proj"]
    v_proj = tensors["v_proj"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()

    batch = q_proj.shape[0]
    hidden = q_proj.shape[1]
    kv_hidden = k_proj.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    max_ctx_blocks = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6

    q_heads_all = q_proj.view(batch, num_heads, head_dim)
    k_heads_all = k_proj.view(batch, num_kv_heads, head_dim)
    q_var = q_heads_all.pow(2).mean(dim=-1, keepdim=True)
    k_var = k_heads_all.pow(2).mean(dim=-1, keepdim=True)
    q_heads_all = q_heads_all * torch.rsqrt(q_var + eps) * q_norm_weight.float()
    k_heads_all = k_heads_all * torch.rsqrt(k_var + eps) * k_norm_weight.float()

    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    attn_out = torch.zeros(batch, hidden, dtype=torch.bfloat16)
    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_heads_all[b]
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
            dim=-1,
        )
        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE
        for ki in range(num_kv_heads):
            cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_heads_all[b]
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
            dim=-1,
        )

        attn_row_padded = torch.zeros(1, total_q_groups * Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                gi = kvh * q_groups + qg
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, ctx_len - s0)
                    pbid = int(block_table[b * max_ctx_blocks + sb].item())
                    cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                    k_tile = k_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                    v_tile = v_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < BLOCK_SIZE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale
                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                ctx_flat_padded_bf16 = torch.zeros(1, Q_HEAD_PAD * head_dim, dtype=torch.bfloat16)
                ctx_flat_padded_bf16[:, : Q_HEAD_BATCH * head_dim] = ctx.reshape(1, -1).to(torch.bfloat16)
                attn_row_padded[
                    :,
                    gi * Q_HEAD_PAD * head_dim : (gi + 1) * Q_HEAD_PAD * head_dim,
                ] = ctx_flat_padded_bf16

        attn_row = torch.zeros(1, hidden, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                gi = kvh * q_groups + qg
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                ctx_flat_bf16 = attn_row_padded[
                    :,
                    gi * Q_HEAD_PAD * head_dim : gi * Q_HEAD_PAD * head_dim + Q_HEAD_BATCH * head_dim,
                ]
                attn_row[
                    :,
                    q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim,
                ] = ctx_flat_bf16

        attn_out[b : b + 1, :] = attn_row

    tensors["attn_out"][:] = attn_out


if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_scope2_program(),
        tensor_specs=build_tensor_specs(use_max_seq=args.max_seq),
        golden_fn=golden_qwen3_scope2,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
            compile=dict(dump_passes=True),
            runtime=dict(platform=args.platform, device_id=args.device, runtime_profiling=args.runtime_profiling),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
