# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B single-layer decode forward - tile DSL version.

This file replaces the older monolithic tile rewrite with the current Scope 1,
Scope 2, and Scope 3 tile kernels:

  1. Scope 1: input RMSNorm + Q/K/V projection
  2. Scope 2: RoPE + KV cache update + grouped-query attention
  3. Scope 3: output projection + residual + post RMSNorm + MLP
"""

import pypto.language as pl

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM  # 8192
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 25600

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

PROJ_K_CHUNK = 512
K_CHUNK = 128
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
MLP_OUT_CHUNK = 64
BATCH_TILE = 16

Q_HEAD_BATCH = 8
Q_HEAD_PAD = 16
SEQ_TILE = 64
SB_CHUNK = 64


def build_qwen3_decode_program(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
):
    hidden = hidden_size
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    scope1_hidden_blocks = hidden // PROJ_K_CHUNK
    hidden_blocks = hidden // K_CHUNK
    q_out_blocks = hidden // Q_OUT_CHUNK
    kv_out_blocks = kv_hidden // KV_OUT_CHUNK
    mlp_out_blocks = inter // MLP_OUT_CHUNK
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    total_q_groups = num_kv_heads * q_groups
    cache_rows = batch * num_kv_heads * max_seq
    half_dim = head_dim // 2
    attn_scale = 1.0 / (head_dim**0.5)
    max_ctx_blocks = (max_seq + SEQ_TILE - 1) // SEQ_TILE
    all_q_padded_rows = batch * total_q_groups * Q_HEAD_PAD
    all_cur_rows = max_ctx_blocks * Q_HEAD_BATCH
    all_score_rows = max_ctx_blocks * Q_HEAD_PAD
    ki_outer_blocks = (num_kv_heads + Q_HEAD_BATCH - 1) // Q_HEAD_BATCH

    @pl.program
    class Qwen3DecodeTile:
        # Scope 1: input RMSNorm + Q/K/V projection.
        @pl.function(type=pl.FunctionType.InCore)
        def build_normed_tile(
            self,
            hidden_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            output: pl.Out[pl.Tensor[[BATCH_TILE, hidden], pl.BF16]],
        ) -> pl.Tensor[[BATCH_TILE, hidden], pl.BF16]:
            partial_sq = pl.create_tile(
                [1, BATCH_TILE],
                dtype=pl.FP32,
                target_memory=pl.MemorySpace.Vec,
            )
            partial_sq = pl.mul(partial_sq, 0.0)

            for kb in pl.range(scope1_hidden_blocks):
                k0 = kb * PROJ_K_CHUNK
                tile_x = pl.load(
                    hidden_tile,
                    [0, k0],
                    [BATCH_TILE, PROJ_K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                tile_x_f32 = pl.cast(tile_x, target_type=pl.FP32)
                squared = pl.mul(tile_x_f32, tile_x_f32)
                tmp = pl.create_tile(
                    [BATCH_TILE, PROJ_K_CHUNK],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                row_sum: pl.Tile[[BATCH_TILE, 1], pl.FP32] = pl.row_sum(squared, tmp)
                partial_sq = pl.add(partial_sq, pl.reshape(row_sum, [1, BATCH_TILE]))

            variance_t: pl.Tile[[1, BATCH_TILE], pl.FP32] = pl.add(
                pl.mul(partial_sq, HIDDEN_INV),
                EPS,
            )
            variance: pl.Tile[[BATCH_TILE, 1], pl.FP32] = pl.reshape(
                variance_t,
                [BATCH_TILE, 1],
            )
            rms = pl.sqrt(variance)
            inv_rms = pl.recip(rms)

            for kb, (out_iter,) in pl.range(scope1_hidden_blocks, init_values=(output,)):
                k0 = kb * PROJ_K_CHUNK
                tile_x = pl.load(
                    hidden_tile,
                    [0, k0],
                    [BATCH_TILE, PROJ_K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                tile_x_f32 = pl.cast(tile_x, target_type=pl.FP32)
                tile_gamma = pl.load(
                    input_rms_weight,
                    [0, k0],
                    [1, PROJ_K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                scaled = pl.row_expand_mul(tile_x_f32, inv_rms)
                weighted = pl.col_expand_mul(scaled, tile_gamma)
                weighted_bf16 = pl.cast(weighted, target_type=pl.BF16)
                out_next = pl.store(weighted_bf16, [0, k0], out_iter)
                (out_carry,) = pl.yield_(out_next)

            return out_carry

        @pl.function(type=pl.FunctionType.InCore)
        def q_proj_reduce(
            self,
            normed_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            weight: pl.Tensor[[hidden, hidden], pl.BF16],
            out_row: pl.Scalar[pl.INDEX],
            out_col: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[batch, hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, hidden], pl.FP32]:
            tile_a_l1 = pl.load(
                normed_tile,
                [0, 0],
                [BATCH_TILE, PROJ_K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            tile_b_l1 = pl.load(
                weight,
                [0, out_col],
                [PROJ_K_CHUNK, Q_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(tile_a_l0a, tile_b_l0b)

            for kb in pl.range(1, scope1_hidden_blocks):
                k0 = kb * PROJ_K_CHUNK
                tile_a_i_l1 = pl.load(
                    normed_tile,
                    [0, k0],
                    [BATCH_TILE, PROJ_K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                tile_b_i_l1 = pl.load(
                    weight,
                    [k0, out_col],
                    [PROJ_K_CHUNK, Q_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                tile_a_i_l0a = pl.move(tile_a_i_l1, target_memory=pl.MemorySpace.Left)
                tile_b_i_l0b = pl.move(tile_b_i_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul_acc(acc, tile_a_i_l0a, tile_b_i_l0b)

            out = pl.store(acc, [out_row, out_col], output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def kv_proj_reduce(
            self,
            normed_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            weight: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            out_row: pl.Scalar[pl.INDEX],
            out_col: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[batch, kv_hidden], pl.FP32]],
        ) -> pl.Tensor[[batch, kv_hidden], pl.FP32]:
            tile_a_l1 = pl.load(
                normed_tile,
                [0, 0],
                [BATCH_TILE, PROJ_K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            tile_b_l1 = pl.load(
                weight,
                [0, out_col],
                [PROJ_K_CHUNK, KV_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            tile_a_l0a = pl.move(tile_a_l1, target_memory=pl.MemorySpace.Left)
            tile_b_l0b = pl.move(tile_b_l1, target_memory=pl.MemorySpace.Right)
            acc = pl.matmul(tile_a_l0a, tile_b_l0b)

            for kb in pl.range(1, scope1_hidden_blocks):
                k0 = kb * PROJ_K_CHUNK
                tile_a_i_l1 = pl.load(
                    normed_tile,
                    [0, k0],
                    [BATCH_TILE, PROJ_K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                tile_b_i_l1 = pl.load(
                    weight,
                    [k0, out_col],
                    [PROJ_K_CHUNK, KV_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                tile_a_i_l0a = pl.move(tile_a_i_l1, target_memory=pl.MemorySpace.Left)
                tile_b_i_l0b = pl.move(tile_b_i_l1, target_memory=pl.MemorySpace.Right)
                acc = pl.matmul_acc(acc, tile_a_i_l0a, tile_b_i_l0b)

            out = pl.store(acc, [out_row, out_col], output)
            return out

        # Scope 2: RoPE + KV cache update + grouped-query attention.
        @pl.function(type=pl.FunctionType.InCore)
        def init_q_padded_zero_tail(
            self,
            all_q_padded: pl.Out[pl.Tensor[[all_q_padded_rows, head_dim], pl.BF16]],
        ) -> pl.Tensor[[all_q_padded_rows, head_dim], pl.BF16]:
            for idx, (all_q_padded_iter,) in pl.range(
                batch * total_q_groups,
                init_values=(all_q_padded,),
            ):
                all_q_padded_next = pl.store(
                    pl.cast(
                        pl.tile.full(
                            [Q_HEAD_PAD - Q_HEAD_BATCH, head_dim],
                            dtype=pl.FP32,
                            value=0.0,
                        ),
                        target_type=pl.BF16,
                    ),
                    [idx * Q_HEAD_PAD + Q_HEAD_BATCH, 0],
                    all_q_padded_iter,
                )
                (all_q_padded_carry,) = pl.yield_(all_q_padded_next)

            return all_q_padded_carry

        @pl.function(type=pl.FunctionType.InCore)
        def apply_rope_and_update_kv_cache(
            self,
            k_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            v_proj: pl.Tensor[[batch, kv_hidden], pl.FP32],
            q_proj: pl.Tensor[[batch, hidden], pl.FP32],
            cos_lo: pl.Tensor[[1, half_dim], pl.FP32],
            cos_hi: pl.Tensor[[1, half_dim], pl.FP32],
            sin_lo: pl.Tensor[[1, half_dim], pl.FP32],
            sin_hi: pl.Tensor[[1, half_dim], pl.FP32],
            b: pl.Scalar[pl.INDEX],
            pos: pl.Scalar[pl.INDEX],
            ki_outer: pl.Scalar[pl.INDEX],
            all_q_padded: pl.Out[pl.Tensor[[all_q_padded_rows, head_dim], pl.BF16]],
            k_cache: pl.Out[pl.Tensor[[cache_rows, head_dim], pl.BF16]],
            v_cache: pl.Out[pl.Tensor[[cache_rows, head_dim], pl.BF16]],
        ) -> tuple[
            pl.Tensor[[all_q_padded_rows, head_dim], pl.BF16],
            pl.Tensor[[cache_rows, head_dim], pl.BF16],
            pl.Tensor[[cache_rows, head_dim], pl.BF16],
        ]:
            cos_lo_t = pl.load(cos_lo, [0, 0], [1, half_dim], target_memory=pl.MemorySpace.Vec)
            cos_hi_t = pl.load(cos_hi, [0, 0], [1, half_dim], target_memory=pl.MemorySpace.Vec)
            sin_lo_t = pl.load(sin_lo, [0, 0], [1, half_dim], target_memory=pl.MemorySpace.Vec)
            sin_hi_t = pl.load(sin_hi, [0, 0], [1, half_dim], target_memory=pl.MemorySpace.Vec)
            q_pad_base = b * total_q_groups * Q_HEAD_PAD

            for ki_inner, (
                all_q_padded_iter,
                k_cache_iter,
                v_cache_iter,
            ) in pl.parallel(
                Q_HEAD_BATCH,
                init_values=(all_q_padded, k_cache, v_cache),
                attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
            ):
                ki = ki_outer * Q_HEAD_BATCH + ki_inner
                kv_col = ki * head_dim
                cache_row = b * num_kv_heads * max_seq + ki * max_seq + pos
                q_base = ki * q_per_kv

                k_lo = pl.load(k_proj, [b, kv_col], [1, half_dim], target_memory=pl.MemorySpace.Vec)
                k_hi = pl.load(
                    k_proj,
                    [b, kv_col + half_dim],
                    [1, half_dim],
                    target_memory=pl.MemorySpace.Vec,
                )
                rot_lo = pl.sub(
                    pl.col_expand_mul(k_lo, cos_lo_t),
                    pl.col_expand_mul(k_hi, sin_lo_t),
                )
                rot_hi = pl.add(
                    pl.col_expand_mul(k_hi, cos_hi_t),
                    pl.col_expand_mul(k_lo, sin_hi_t),
                )
                k_cache_next = pl.store(
                    pl.cast(rot_lo, target_type=pl.BF16),
                    [cache_row, 0],
                    k_cache_iter,
                )
                k_cache_next = pl.store(
                    pl.cast(rot_hi, target_type=pl.BF16),
                    [cache_row, half_dim],
                    k_cache_next,
                )

                v_tile = pl.load(v_proj, [b, kv_col], [1, head_dim], target_memory=pl.MemorySpace.Vec)
                v_cache_next = pl.store(
                    pl.cast(v_tile, target_type=pl.BF16),
                    [cache_row, 0],
                    v_cache_iter,
                )

                for qi, (all_q_padded_qi_iter,) in pl.range(
                    Q_HEAD_BATCH,
                    init_values=(all_q_padded_iter,),
                ):
                    q_col = (q_base + qi) * head_dim
                    q_lo = pl.load(q_proj, [b, q_col], [1, half_dim], target_memory=pl.MemorySpace.Vec)
                    q_hi = pl.load(
                        q_proj,
                        [b, q_col + half_dim],
                        [1, half_dim],
                        target_memory=pl.MemorySpace.Vec,
                    )
                    q_rot_lo = pl.sub(
                        pl.col_expand_mul(q_lo, cos_lo_t),
                        pl.col_expand_mul(q_hi, sin_lo_t),
                    )
                    q_rot_hi = pl.add(
                        pl.col_expand_mul(q_hi, cos_hi_t),
                        pl.col_expand_mul(q_lo, sin_hi_t),
                    )
                    all_q_padded_qi_next = pl.store(
                        pl.cast(q_rot_lo, target_type=pl.BF16),
                        [q_pad_base + ki * Q_HEAD_PAD + qi, 0],
                        all_q_padded_qi_iter,
                    )
                    all_q_padded_qi_next = pl.store(
                        pl.cast(q_rot_hi, target_type=pl.BF16),
                        [q_pad_base + ki * Q_HEAD_PAD + qi, half_dim],
                        all_q_padded_qi_next,
                    )
                    (all_q_padded_qi_carry,) = pl.yield_(all_q_padded_qi_next)

                (
                    all_q_padded_inner_carry,
                    k_cache_inner_carry,
                    v_cache_inner_carry,
                ) = pl.yield_(
                    all_q_padded_qi_carry,
                    k_cache_next,
                    v_cache_next,
                )

            return (
                all_q_padded_inner_carry,
                k_cache_inner_carry,
                v_cache_inner_carry,
            )

        @pl.function(type=pl.FunctionType.InCore)
        def compute_qk_scores(
            self,
            q_padded: pl.Tensor[[Q_HEAD_PAD, head_dim], pl.BF16],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            b: pl.Scalar[pl.INDEX],
            kvh: pl.Scalar[pl.INDEX],
            sb_outer: pl.Scalar[pl.INDEX],
            all_raw_scores: pl.Out[pl.Tensor[[all_score_rows, SEQ_TILE], pl.FP32]],
        ) -> pl.Tensor[[all_score_rows, SEQ_TILE], pl.FP32]:
            q_l1 = pl.load(
                q_padded,
                [0, 0],
                [Q_HEAD_PAD, head_dim],
                target_memory=pl.MemorySpace.Mat,
            )
            q_l0a = pl.move(q_l1, target_memory=pl.MemorySpace.Left)

            for sb_inner, (all_raw_scores_iter,) in pl.parallel(
                SB_CHUNK,
                init_values=(all_raw_scores,),
                attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
            ):
                sb = sb_outer * SB_CHUNK + sb_inner
                s0 = sb * SEQ_TILE
                cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0

                k_l1 = pl.load(
                    k_cache,
                    [cache_row0, 0],
                    [SEQ_TILE, head_dim],
                    target_memory=pl.MemorySpace.Mat,
                    transpose=True,
                )
                k_l0b = pl.move(k_l1, target_memory=pl.MemorySpace.Right)
                raw_scores = pl.matmul(q_l0a, k_l0b)
                all_raw_scores_next = pl.store(
                    raw_scores,
                    [sb * Q_HEAD_PAD, 0],
                    all_raw_scores_iter,
                )
                (all_raw_scores_carry,) = pl.yield_(all_raw_scores_next)

            return all_raw_scores_carry

        @pl.function(type=pl.FunctionType.InCore)
        def prepare_softmax_blocks(
            self,
            all_raw_scores: pl.Tensor[[all_score_rows, SEQ_TILE], pl.FP32],
            ctx_len: pl.Scalar[pl.INDEX],
            sb_outer: pl.Scalar[pl.INDEX],
            all_exp_padded: pl.Out[pl.Tensor[[all_score_rows, SEQ_TILE], pl.BF16]],
            all_cur_mi: pl.Out[pl.Tensor[[all_cur_rows, 1], pl.FP32]],
            all_cur_li: pl.Out[pl.Tensor[[all_cur_rows, 1], pl.FP32]],
        ) -> tuple[
            pl.Tensor[[all_score_rows, SEQ_TILE], pl.BF16],
            pl.Tensor[[all_cur_rows, 1], pl.FP32],
            pl.Tensor[[all_cur_rows, 1], pl.FP32],
        ]:
            for sb_inner, (
                all_exp_padded_iter,
                all_cur_mi_iter,
                all_cur_li_iter,
            ) in pl.parallel(
                SB_CHUNK,
                init_values=(all_exp_padded, all_cur_mi, all_cur_li),
                attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
            ):
                sb = sb_outer * SB_CHUNK + sb_inner
                s0 = sb * SEQ_TILE
                valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                scores_valid = pl.load(
                    all_raw_scores,
                    [sb * Q_HEAD_PAD, 0],
                    [Q_HEAD_BATCH, SEQ_TILE],
                    valid_shapes=[Q_HEAD_BATCH, valid_len],
                    target_memory=pl.MemorySpace.Vec,
                )
                scores_padded = pl.fillpad(scores_valid, pad_value=pl.PadValue.min)
                scores = pl.mul(scores_padded, attn_scale)
                tmp = pl.create_tile(
                    [Q_HEAD_BATCH, head_dim],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                cur_mi = pl.row_max(scores, tmp)
                exp_scores = pl.exp(pl.row_expand_sub(scores, cur_mi))
                exp_scores_bf16 = pl.cast(exp_scores, target_type=pl.BF16)
                exp_scores_fp32 = pl.cast(exp_scores_bf16, target_type=pl.FP32)
                cur_li = pl.row_sum(exp_scores_fp32, tmp)

                all_exp_padded_next = pl.store(
                    exp_scores_bf16,
                    [sb * Q_HEAD_PAD, 0],
                    all_exp_padded_iter,
                )
                all_cur_mi_next = pl.store(
                    cur_mi,
                    [sb * Q_HEAD_BATCH, 0],
                    all_cur_mi_iter,
                )
                all_cur_li_next = pl.store(
                    cur_li,
                    [sb * Q_HEAD_BATCH, 0],
                    all_cur_li_iter,
                )
                (
                    all_exp_padded_carry,
                    all_cur_mi_carry,
                    all_cur_li_carry,
                ) = pl.yield_(
                    all_exp_padded_next,
                    all_cur_mi_next,
                    all_cur_li_next,
                )

            return all_exp_padded_carry, all_cur_mi_carry, all_cur_li_carry

        @pl.function(type=pl.FunctionType.InCore)
        def compute_sv_context_blocks(
            self,
            all_exp_padded: pl.Tensor[[all_score_rows, SEQ_TILE], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            b: pl.Scalar[pl.INDEX],
            kvh: pl.Scalar[pl.INDEX],
            sb_outer: pl.Scalar[pl.INDEX],
            all_oi_tmp: pl.Out[pl.Tensor[[all_score_rows, head_dim], pl.FP32]],
        ) -> pl.Tensor[[all_score_rows, head_dim], pl.FP32]:
            for sb_inner, (all_oi_tmp_iter,) in pl.parallel(
                SB_CHUNK,
                init_values=(all_oi_tmp,),
                attrs={"loop_origin": pl.LoopOrigin.ChunkInner},
            ):
                sb = sb_outer * SB_CHUNK + sb_inner
                s0 = sb * SEQ_TILE
                cache_row0 = b * num_kv_heads * max_seq + kvh * max_seq + s0

                exp_l1 = pl.load(
                    all_exp_padded,
                    [sb * Q_HEAD_PAD, 0],
                    [Q_HEAD_PAD, SEQ_TILE],
                    target_memory=pl.MemorySpace.Mat,
                )
                v_l1 = pl.load(
                    v_cache,
                    [cache_row0, 0],
                    [SEQ_TILE, head_dim],
                    target_memory=pl.MemorySpace.Mat,
                )
                exp_l0a = pl.move(exp_l1, target_memory=pl.MemorySpace.Left)
                v_l0b = pl.move(v_l1, target_memory=pl.MemorySpace.Right)
                oi_tmp = pl.matmul(exp_l0a, v_l0b)
                all_oi_tmp_next = pl.store(
                    oi_tmp,
                    [sb * Q_HEAD_PAD, 0],
                    all_oi_tmp_iter,
                )
                (all_oi_tmp_carry,) = pl.yield_(all_oi_tmp_next)

            return all_oi_tmp_carry

        @pl.function(type=pl.FunctionType.InCore)
        def finalize_attention_group(
            self,
            all_oi_tmp: pl.Tensor[[all_score_rows, head_dim], pl.FP32],
            all_cur_mi: pl.Tensor[[all_cur_rows, 1], pl.FP32],
            all_cur_li: pl.Tensor[[all_cur_rows, 1], pl.FP32],
            ctx_blocks: pl.Scalar[pl.INDEX],
            q_base: pl.Scalar[pl.INDEX],
            attn_row: pl.Out[pl.Tensor[[1, hidden], pl.BF16]],
        ) -> pl.Tensor[[1, hidden], pl.BF16]:
            oi = pl.load(all_oi_tmp, [0, 0], [Q_HEAD_BATCH, head_dim], target_memory=pl.MemorySpace.Vec)
            mi = pl.load(all_cur_mi, [0, 0], [Q_HEAD_BATCH, 1], target_memory=pl.MemorySpace.Vec)
            li = pl.load(all_cur_li, [0, 0], [Q_HEAD_BATCH, 1], target_memory=pl.MemorySpace.Vec)

            for sb, (li_iter, mi_iter, oi_iter) in pl.range(
                1,
                ctx_blocks,
                init_values=(li, mi, oi),
            ):
                oi_tmp_valid = pl.load(
                    all_oi_tmp,
                    [sb * Q_HEAD_PAD, 0],
                    [Q_HEAD_BATCH, head_dim],
                    target_memory=pl.MemorySpace.Vec,
                )
                cur_mi = pl.load(
                    all_cur_mi,
                    [sb * Q_HEAD_BATCH, 0],
                    [Q_HEAD_BATCH, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                cur_li = pl.load(
                    all_cur_li,
                    [sb * Q_HEAD_BATCH, 0],
                    [Q_HEAD_BATCH, 1],
                    target_memory=pl.MemorySpace.Vec,
                )
                mi_new = pl.maximum(mi_iter, cur_mi)
                alpha = pl.exp(pl.sub(mi_iter, mi_new))
                beta = pl.exp(pl.sub(cur_mi, mi_new))
                li_next = pl.add(pl.mul(alpha, li_iter), pl.mul(beta, cur_li))
                oi_next = pl.add(
                    pl.row_expand_mul(oi_iter, alpha),
                    pl.row_expand_mul(oi_tmp_valid, beta),
                )
                (li_carry, mi_carry, oi_carry) = pl.yield_(li_next, mi_new, oi_next)

            ctx = pl.row_expand_div(oi_carry, li_carry)
            ctx_flat = pl.reshape(ctx, [1, Q_HEAD_BATCH * head_dim])
            ctx_flat_bf16 = pl.cast(ctx_flat, target_type=pl.BF16)
            attn_row_out = pl.store(ctx_flat_bf16, [0, q_base * head_dim], attn_row)
            return attn_row_out

        # Scope 3: output projection + residual + post RMSNorm + MLP.
        @pl.function(type=pl.FunctionType.InCore)
        def compute_output_projection_chunk(
            self,
            attn_out: pl.Tensor[[batch, hidden], pl.BF16],
            b0: pl.Scalar[pl.INDEX],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            o0: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[BATCH_TILE, Q_OUT_CHUNK], pl.FP32]],
        ) -> pl.Tensor[[BATCH_TILE, Q_OUT_CHUNK], pl.FP32]:
            a_chunk_0 = pl.load(
                attn_out,
                [b0, 0],
                [BATCH_TILE, K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            w_chunk_0 = pl.load(
                wo,
                [0, o0],
                [K_CHUNK, Q_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            o_acc = pl.matmul(a_chunk_0, w_chunk_0)

            for kb, (o_acc_iter,) in pl.range(1, hidden_blocks, init_values=(o_acc,)):
                k0 = kb * K_CHUNK
                a_chunk = pl.load(
                    attn_out,
                    [b0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                w_chunk = pl.load(
                    wo,
                    [k0, o0],
                    [K_CHUNK, Q_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                o_acc_next = pl.matmul_acc(o_acc_iter, a_chunk, w_chunk)
                (o_acc_carry,) = pl.yield_(o_acc_next)

            out = pl.store(o_acc_carry, [0, 0], output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def accumulate_residual_chunk(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            b0: pl.Scalar[pl.INDEX],
            o0: pl.Scalar[pl.INDEX],
            o_acc: pl.Tensor[[BATCH_TILE, Q_OUT_CHUNK], pl.FP32],
            resid1_tile: pl.Out[pl.Tensor[[BATCH_TILE, hidden], pl.FP32]],
        ) -> pl.Tensor[[BATCH_TILE, hidden], pl.FP32]:
            o_acc_tile = pl.load(
                o_acc,
                [0, 0],
                [BATCH_TILE, Q_OUT_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            resid_tile = pl.load(
                hidden_states,
                [b0, o0],
                [BATCH_TILE, Q_OUT_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            resid_f32 = pl.cast(resid_tile, target_type=pl.FP32)
            resid_sum = pl.add(o_acc_tile, resid_f32)
            out = pl.store(resid_sum, [0, o0], resid1_tile)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def apply_post_attention_rmsnorm(
            self,
            resid1_tile: pl.Tensor[[BATCH_TILE, hidden], pl.FP32],
            post_norm_tile: pl.Out[pl.Tensor[[BATCH_TILE, hidden], pl.BF16]],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
        ) -> pl.Tensor[[BATCH_TILE, hidden], pl.BF16]:
            sq_sum = pl.tile.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)

            for kb, (sq_sum_iter,) in pl.range(hidden_blocks, init_values=(sq_sum,)):
                k0 = kb * K_CHUNK
                x_chunk = pl.load(
                    resid1_tile,
                    [0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                squared = pl.mul(x_chunk, x_chunk)
                tmp = pl.create_tile(
                    [BATCH_TILE, K_CHUNK],
                    dtype=pl.FP32,
                    target_memory=pl.MemorySpace.Vec,
                )
                row_sum = pl.row_sum(squared, tmp)
                sq_sum_next = pl.add(sq_sum_iter, pl.reshape(row_sum, [1, BATCH_TILE]))
                (sq_sum_carry,) = pl.yield_(sq_sum_next)

            inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum_carry, HIDDEN_INV), EPS)))

            for kb, (post_norm_iter,) in pl.range(hidden_blocks, init_values=(post_norm_tile,)):
                k0 = kb * K_CHUNK
                x_chunk = pl.load(
                    resid1_tile,
                    [0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                gamma = pl.load(
                    post_rms_weight,
                    [0, k0],
                    [1, K_CHUNK],
                    target_memory=pl.MemorySpace.Vec,
                )
                scaled = pl.row_expand_mul(x_chunk, pl.reshape(inv_rms, [BATCH_TILE, 1]))
                normed = pl.col_expand_mul(scaled, gamma)
                normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                post_norm_next = pl.store(normed_bf16, [0, k0], post_norm_iter)
                (post_norm_carry,) = pl.yield_(post_norm_next)

            return post_norm_carry

        @pl.function(type=pl.FunctionType.InCore)
        def compute_gate_projection_chunk(
            self,
            post_norm_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            w_gate: pl.Tensor[[hidden, inter], pl.BF16],
            o0: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32]],
        ) -> pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32]:
            post_chunk_0 = pl.load(
                post_norm_tile,
                [0, 0],
                [BATCH_TILE, K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            wg_0 = pl.load(
                w_gate,
                [0, o0],
                [K_CHUNK, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            gate_acc = pl.matmul(post_chunk_0, wg_0)

            for kb, (gate_acc_iter,) in pl.range(1, hidden_blocks, init_values=(gate_acc,)):
                k0 = kb * K_CHUNK
                post_chunk = pl.load(
                    post_norm_tile,
                    [0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                wg_chunk = pl.load(
                    w_gate,
                    [k0, o0],
                    [K_CHUNK, MLP_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                gate_acc_next = pl.matmul_acc(gate_acc_iter, post_chunk, wg_chunk)
                (gate_acc_carry,) = pl.yield_(gate_acc_next)

            out = pl.store(gate_acc_carry, [0, 0], output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def compute_up_projection_chunk(
            self,
            post_norm_tile: pl.Tensor[[BATCH_TILE, hidden], pl.BF16],
            w_up: pl.Tensor[[hidden, inter], pl.BF16],
            o0: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32]],
        ) -> pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32]:
            post_chunk_0 = pl.load(
                post_norm_tile,
                [0, 0],
                [BATCH_TILE, K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            wu_0 = pl.load(
                w_up,
                [0, o0],
                [K_CHUNK, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            up_acc = pl.matmul(post_chunk_0, wu_0)

            for kb, (up_acc_iter,) in pl.range(1, hidden_blocks, init_values=(up_acc,)):
                k0 = kb * K_CHUNK
                post_chunk = pl.load(
                    post_norm_tile,
                    [0, k0],
                    [BATCH_TILE, K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                wu_chunk = pl.load(
                    w_up,
                    [k0, o0],
                    [K_CHUNK, MLP_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                up_acc_next = pl.matmul_acc(up_acc_iter, post_chunk, wu_chunk)
                (up_acc_carry,) = pl.yield_(up_acc_next)

            out = pl.store(up_acc_carry, [0, 0], output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def apply_silu_mlp_chunk(
            self,
            gate_acc: pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32],
            up_acc: pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.FP32],
            output: pl.Out[pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.BF16]],
        ) -> pl.Tensor[[BATCH_TILE, MLP_OUT_CHUNK], pl.BF16]:
            gate_acc_tile = pl.load(
                gate_acc,
                [0, 0],
                [BATCH_TILE, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            up_acc_tile = pl.load(
                up_acc,
                [0, 0],
                [BATCH_TILE, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc_tile)), 1.0))
            mlp_chunk = pl.mul(pl.mul(gate_acc_tile, sigmoid), up_acc_tile)
            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)
            out = pl.store(mlp_chunk_bf16, [0, 0], output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def compute_down_projection_chunk(
            self,
            mlp_tile: pl.Tensor[[BATCH_TILE, inter], pl.BF16],
            w_down: pl.Tensor[[inter, hidden], pl.BF16],
            d0: pl.Scalar[pl.INDEX],
            output: pl.Out[pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32]],
        ) -> pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32]:
            mlp_chunk_0 = pl.load(
                mlp_tile,
                [0, 0],
                [BATCH_TILE, MLP_OUT_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            w_down_chunk_0 = pl.load(
                w_down,
                [0, d0],
                [MLP_OUT_CHUNK, K_CHUNK],
                target_memory=pl.MemorySpace.Mat,
            )
            down_acc = pl.matmul(mlp_chunk_0, w_down_chunk_0)

            for ob, (down_acc_iter,) in pl.range(1, mlp_out_blocks, init_values=(down_acc,)):
                o0 = ob * MLP_OUT_CHUNK
                down_mlp_chunk = pl.load(
                    mlp_tile,
                    [0, o0],
                    [BATCH_TILE, MLP_OUT_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                w_down_chunk = pl.load(
                    w_down,
                    [o0, d0],
                    [MLP_OUT_CHUNK, K_CHUNK],
                    target_memory=pl.MemorySpace.Mat,
                )
                down_acc_next = pl.matmul_acc(down_acc_iter, down_mlp_chunk, w_down_chunk)
                (down_acc_carry,) = pl.yield_(down_acc_next)

            out = pl.store(down_acc_carry, [0, 0], output)
            return out

        @pl.function(type=pl.FunctionType.InCore)
        def write_final_output_chunk(
            self,
            resid1_tile: pl.Tensor[[BATCH_TILE, hidden], pl.FP32],
            d0: pl.Scalar[pl.INDEX],
            down_acc: pl.Tensor[[BATCH_TILE, K_CHUNK], pl.FP32],
            out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
            b0: pl.Scalar[pl.INDEX],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            down_acc_tile = pl.load(
                down_acc,
                [0, 0],
                [BATCH_TILE, K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            resid_chunk = pl.load(
                resid1_tile,
                [0, d0],
                [BATCH_TILE, K_CHUNK],
                target_memory=pl.MemorySpace.Vec,
            )
            out_chunk = pl.add(down_acc_tile, resid_chunk)
            out_chunk_bf16 = pl.cast(out_chunk, target_type=pl.BF16)
            result = pl.store(out_chunk_bf16, [b0, d0], out)
            return result

        @pl.function(type=pl.FunctionType.Orchestration)
        def qwen3_decode(
            self,
            hidden_states: pl.Tensor[[batch, hidden], pl.BF16],
            input_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            wq: pl.Tensor[[hidden, hidden], pl.BF16],
            wk: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            wv: pl.Tensor[[hidden, kv_hidden], pl.BF16],
            seq_lens: pl.Tensor[[batch], pl.INT32],
            rope_cos: pl.Tensor[[max_seq, head_dim], pl.FP32],
            rope_sin: pl.Tensor[[max_seq, head_dim], pl.FP32],
            k_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            v_cache: pl.Tensor[[cache_rows, head_dim], pl.BF16],
            wo: pl.Tensor[[hidden, hidden], pl.BF16],
            post_rms_weight: pl.Tensor[[1, hidden], pl.FP32],
            w_gate: pl.Tensor[[hidden, inter], pl.BF16],
            w_up: pl.Tensor[[hidden, inter], pl.BF16],
            w_down: pl.Tensor[[inter, hidden], pl.BF16],
            out: pl.Out[pl.Tensor[[batch, hidden], pl.BF16]],
        ) -> pl.Tensor[[batch, hidden], pl.BF16]:
            q_proj = pl.create_tensor([batch, hidden], dtype=pl.FP32)
            k_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)
            v_proj = pl.create_tensor([batch, kv_hidden], dtype=pl.FP32)

            # Scope 1 range: input RMSNorm + Q/K/V projection.
            for b0 in pl.range(0, batch, BATCH_TILE):
                hidden_tile = pl.slice(hidden_states, [BATCH_TILE, hidden], [b0, 0])
                normed_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)
                normed_tile = self.build_normed_tile(hidden_tile, input_rms_weight, normed_tile)

                for ob in pl.range(q_out_blocks):
                    q0 = ob * Q_OUT_CHUNK
                    q_proj = self.q_proj_reduce(normed_tile, wq, b0, q0, q_proj)

                for ob in pl.range(kv_out_blocks):
                    kv0 = ob * KV_OUT_CHUNK
                    k_proj = self.kv_proj_reduce(normed_tile, wk, b0, kv0, k_proj)
                    v_proj = self.kv_proj_reduce(normed_tile, wv, b0, kv0, v_proj)

            all_q_padded_init = pl.create_tensor([all_q_padded_rows, head_dim], dtype=pl.BF16)
            all_q_padded_init = self.init_q_padded_zero_tail(all_q_padded_init)
            attn_out = pl.create_tensor([batch, hidden], dtype=pl.BF16)

            # Scope 2 range: RoPE + KV cache update + grouped-query attention.
            for b, (
                all_q_padded_iter,
                attn_out_iter,
                k_cache_iter,
                v_cache_iter,
            ) in pl.range(
                batch,
                init_values=(all_q_padded_init, attn_out, k_cache, v_cache),
            ):
                ctx_len = pl.tensor.read(seq_lens, [b])
                pos = ctx_len - 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                cos_row = pl.slice(rope_cos, [1, head_dim], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, head_dim], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, half_dim], [0, 0])
                cos_hi = pl.slice(cos_row, [1, half_dim], [0, half_dim])
                sin_lo = pl.slice(sin_row, [1, half_dim], [0, 0])
                sin_hi = pl.slice(sin_row, [1, half_dim], [0, half_dim])

                for ki_outer, (
                    all_q_padded_rope_iter,
                    k_cache_rope_iter,
                    v_cache_rope_iter,
                ) in pl.parallel(
                    ki_outer_blocks,
                    init_values=(all_q_padded_iter, k_cache_iter, v_cache_iter),
                    attrs={"loop_origin": pl.LoopOrigin.ChunkOuter},
                ):
                    (
                        all_q_padded_rope_next,
                        k_cache_rope_next,
                        v_cache_rope_next,
                    ) = self.apply_rope_and_update_kv_cache(
                        k_proj,
                        v_proj,
                        q_proj,
                        cos_lo,
                        cos_hi,
                        sin_lo,
                        sin_hi,
                        b,
                        pos,
                        ki_outer,
                        all_q_padded_rope_iter,
                        k_cache_rope_iter,
                        v_cache_rope_iter,
                    )
                    (
                        all_q_padded_after_rope,
                        k_cache_after_rope,
                        v_cache_after_rope,
                    ) = pl.yield_(
                        all_q_padded_rope_next,
                        k_cache_rope_next,
                        v_cache_rope_next,
                    )

                attn_row = pl.create_tensor([1, hidden], dtype=pl.BF16)
                for gi, (attn_row_iter,) in pl.range(total_q_groups, init_values=(attn_row,)):
                    kvh = gi // q_groups
                    qg = gi - kvh * q_groups
                    q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                    q_padded = pl.slice(
                        all_q_padded_after_rope,
                        [Q_HEAD_PAD, head_dim],
                        [b * total_q_groups * Q_HEAD_PAD + gi * Q_HEAD_PAD, 0],
                    )

                    all_raw_scores = pl.create_tensor([all_score_rows, SEQ_TILE], dtype=pl.FP32)
                    all_exp_padded = pl.create_tensor([all_score_rows, SEQ_TILE], dtype=pl.BF16)
                    all_oi_tmp = pl.create_tensor([all_score_rows, head_dim], dtype=pl.FP32)
                    all_cur_mi = pl.create_tensor([all_cur_rows, 1], dtype=pl.FP32)
                    all_cur_li = pl.create_tensor([all_cur_rows, 1], dtype=pl.FP32)
                    sb_outer_blocks = (ctx_blocks + SB_CHUNK - 1) // SB_CHUNK

                    for sb_outer, (all_raw_scores_iter,) in pl.parallel(
                        sb_outer_blocks,
                        init_values=(all_raw_scores,),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkOuter},
                    ):
                        all_raw_scores_next = self.compute_qk_scores(
                            q_padded,
                            k_cache_after_rope,
                            b,
                            kvh,
                            sb_outer,
                            all_raw_scores_iter,
                        )
                        (all_raw_scores_after_qk,) = pl.yield_(all_raw_scores_next)

                    for sb_outer, (
                        all_exp_padded_iter,
                        all_cur_mi_iter,
                        all_cur_li_iter,
                    ) in pl.parallel(
                        sb_outer_blocks,
                        init_values=(all_exp_padded, all_cur_mi, all_cur_li),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkOuter},
                    ):
                        (
                            all_exp_padded_next,
                            all_cur_mi_next,
                            all_cur_li_next,
                        ) = self.prepare_softmax_blocks(
                            all_raw_scores_after_qk,
                            ctx_len,
                            sb_outer,
                            all_exp_padded_iter,
                            all_cur_mi_iter,
                            all_cur_li_iter,
                        )
                        (
                            all_exp_padded_after_softmax,
                            all_cur_mi_after_softmax,
                            all_cur_li_after_softmax,
                        ) = pl.yield_(
                            all_exp_padded_next,
                            all_cur_mi_next,
                            all_cur_li_next,
                        )

                    for sb_outer, (all_oi_tmp_iter,) in pl.parallel(
                        sb_outer_blocks,
                        init_values=(all_oi_tmp,),
                        attrs={"loop_origin": pl.LoopOrigin.ChunkOuter},
                    ):
                        all_oi_tmp_next = self.compute_sv_context_blocks(
                            all_exp_padded_after_softmax,
                            v_cache_after_rope,
                            b,
                            kvh,
                            sb_outer,
                            all_oi_tmp_iter,
                        )
                        (all_oi_tmp_after_sv,) = pl.yield_(all_oi_tmp_next)

                    attn_row_next = self.finalize_attention_group(
                        all_oi_tmp_after_sv,
                        all_cur_mi_after_softmax,
                        all_cur_li_after_softmax,
                        ctx_blocks,
                        q_base,
                        attn_row_iter,
                    )
                    (attn_row_after_group,) = pl.yield_(attn_row_next)

                attn_out_next = pl.assemble(attn_out_iter, attn_row_after_group, [b, 0])
                (
                    _,
                    attn_out_batch_carry,
                    _,
                    _,
                ) = pl.yield_(
                    all_q_padded_after_rope,
                    attn_out_next,
                    k_cache_after_rope,
                    v_cache_after_rope,
                )

            # Scope 3 range: output projection + residual + post RMSNorm + MLP.
            for b0, (out_iter,) in pl.range(0, batch, BATCH_TILE, init_values=(out,)):
                resid1_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.FP32)

                for ob, (resid1_tile_iter,) in pl.range(q_out_blocks, init_values=(resid1_tile,)):
                    o0 = ob * Q_OUT_CHUNK
                    o_acc_buf = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                    o_acc = self.compute_output_projection_chunk(
                        attn_out_batch_carry,
                        b0,
                        wo,
                        o0,
                        o_acc_buf,
                    )
                    resid1_tile_next = self.accumulate_residual_chunk(
                        hidden_states,
                        b0,
                        o0,
                        o_acc,
                        resid1_tile_iter,
                    )
                    (resid1_tile_carry,) = pl.yield_(resid1_tile_next)

                post_norm_tile = pl.create_tensor([BATCH_TILE, hidden], dtype=pl.BF16)
                post_norm_tile = self.apply_post_attention_rmsnorm(
                    resid1_tile_carry,
                    post_norm_tile,
                    post_rms_weight,
                )

                mlp_tile = pl.create_tensor([BATCH_TILE, inter], dtype=pl.BF16)
                for ob, (mlp_tile_iter,) in pl.range(mlp_out_blocks, init_values=(mlp_tile,)):
                    o0 = ob * MLP_OUT_CHUNK
                    gate_buf = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    gate_acc = self.compute_gate_projection_chunk(post_norm_tile, w_gate, o0, gate_buf)
                    up_buf = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                    up_acc = self.compute_up_projection_chunk(post_norm_tile, w_up, o0, up_buf)
                    mlp_chunk_buf = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.BF16)
                    mlp_chunk = self.apply_silu_mlp_chunk(gate_acc, up_acc, mlp_chunk_buf)
                    mlp_tile_next = pl.assemble(mlp_tile_iter, mlp_chunk, [0, o0])
                    (mlp_tile_carry,) = pl.yield_(mlp_tile_next)

                for dob, (out_iter_inner,) in pl.range(hidden_blocks, init_values=(out_iter,)):
                    d0 = dob * K_CHUNK
                    down_buf = pl.create_tensor([BATCH_TILE, K_CHUNK], dtype=pl.FP32)
                    down_acc = self.compute_down_projection_chunk(mlp_tile_carry, w_down, d0, down_buf)
                    out_next = self.write_final_output_chunk(
                        resid1_tile_carry,
                        d0,
                        down_acc,
                        out_iter_inner,
                        b0,
                    )
                    (out_carry_inner,) = pl.yield_(out_next)

                (out_carry,) = pl.yield_(out_carry_inner)

            return out_carry

    return Qwen3DecodeTile


def build_tensor_specs(
    batch: int = BATCH,
    max_seq: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    intermediate_size: int = INTERMEDIATE,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    use_max_seq: bool = False,
):
    import torch
    from golden import TensorSpec

    hidden = num_heads * head_dim
    kv_hidden = num_kv_heads * head_dim
    inter = intermediate_size
    cache_rows = batch * num_kv_heads * max_seq

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_wq():
        return torch.rand(hidden_size, hidden_size) / hidden_size**0.5

    def init_wk():
        return torch.rand(hidden_size, kv_hidden) / hidden_size**0.5

    def init_wv():
        return torch.rand(hidden_size, kv_hidden) / hidden_size**0.5

    def init_seq_lens():
        if use_max_seq:
            return torch.full((batch,), max_seq, dtype=torch.int32)
        return torch.randint(1, max_seq + 1, (batch,), dtype=torch.int32)

    def init_rope_cos():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_rope_sin():
        return torch.rand(max_seq, head_dim) - 0.5

    def init_k_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_v_cache():
        return torch.rand(cache_rows, head_dim) - 0.5

    def init_wo():
        return (torch.rand(hidden_size, hidden_size) - 0.5) / hidden_size**0.5

    def init_post_rms_weight():
        return torch.ones(1, hidden_size)

    def init_w_gate():
        return (torch.rand(hidden_size, inter) - 0.5) / hidden_size**0.5

    def init_w_up():
        return (torch.rand(hidden_size, inter) - 0.5) / hidden_size**0.5

    def init_w_down():
        return (torch.rand(inter, hidden_size) - 0.5) / inter**0.5

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, hidden_size], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wq),
        TensorSpec("wk", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wk),
        TensorSpec("wv", [hidden_size, kv_hidden], torch.bfloat16, init_value=init_wv),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=init_seq_lens),
        TensorSpec("rope_cos", [max_seq, head_dim], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [max_seq, head_dim], torch.float32, init_value=init_rope_sin),
        TensorSpec("k_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_k_cache),
        TensorSpec("v_cache", [cache_rows, head_dim], torch.bfloat16, init_value=init_v_cache),
        TensorSpec("wo", [hidden_size, hidden_size], torch.bfloat16, init_value=init_wo),
        TensorSpec("post_rms_weight", [1, hidden_size], torch.float32, init_value=init_post_rms_weight),
        TensorSpec("w_gate", [hidden_size, inter], torch.bfloat16, init_value=init_w_gate),
        TensorSpec("w_up", [hidden_size, inter], torch.bfloat16, init_value=init_w_up),
        TensorSpec("w_down", [inter, hidden_size], torch.bfloat16, init_value=init_w_down),
        TensorSpec("out", [batch, hidden], torch.bfloat16, is_output=True),
    ]


def golden_qwen3_decode(tensors: dict, params: dict | None = None) -> None:
    """PyTorch reference for the combined decode path."""
    import math

    import torch

    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, PROJ_K_CHUNK):
            x_chunk = x_tile[:, k0 : k0 + PROJ_K_CHUNK]
            sq_sum = sq_sum + (x_chunk**2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        rms = torch.sqrt(variance)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

        q_proj[b0:b_end, :] = (normed.float() @ wq.float()).float()
        k_proj[b0:b_end, :] = (normed.float() @ wk.float()).float()
        v_proj[b0:b_end, :] = (normed.float() @ wv.float()).float()

    attn_out = torch.zeros(batch, hidden_size, dtype=torch.float32)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [
                k_lo_h * cos_lo - k_hi_h * sin_lo,
                k_hi_h * cos_hi + k_lo_h * sin_hi,
            ],
            dim=-1,
        )

        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_proj[b].view(num_heads, head_dim)
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [
                q_lo_h * cos_lo - q_hi_h * sin_lo,
                q_hi_h * cos_hi + q_lo_h * sin_hi,
            ],
            dim=-1,
        )

        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * num_kv_heads * max_seq + kvh * max_seq + s0

                    k_tile = k_cache[cb : cb + SEQ_TILE, :]
                    v_tile = v_cache[cb : cb + SEQ_TILE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < SEQ_TILE:
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
                for qi in range(Q_HEAD_BATCH):
                    qh = q_base + qi
                    attn_out[b, qh * head_dim : (qh + 1) * head_dim] = ctx[qi]

    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    tensors["out"][:] = (down + resid1).bfloat16()


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    parser.add_argument("--max-seq", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_qwen3_decode_program(),
        specs=build_tensor_specs(use_max_seq=args.max_seq),
        golden_fn=golden_qwen3_decode,
        config=RunConfig(
            rtol=3e-3,
            atol=3e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
