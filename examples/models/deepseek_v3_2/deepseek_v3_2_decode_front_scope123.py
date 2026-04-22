# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

"""
DeepSeek V3.2-EXP decode front fused kernel (scope1 + scope2 + scope3).

Fused pipeline:
- Scope1: RMSNorm + Q/KV projection + RoPE + KV/PE cache update
- Scope2: indexer projection + indexer RoPE + weight reduction + k_cache_idx update
- Scope3: q_idx x k_cache_idx scoring + topk
"""

import pypto.language as pl


BATCH = 16
MAX_SEQ = 4096
HIDDEN = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 128
KV_A_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM
CACHE_ROWS = BATCH * MAX_SEQ

INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN

# Scope1 tiles
RMSNORM_K = 512
PROJ_K = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
LORA_CHUNK = 64
BATCH_TILE = 16

# Scope2 tiles
K_CHUNK = 128
IDX_OUT_CHUNK = 128
WEIGHTS_OUT_CHUNK = 64
HADAMARD_SCALE = INDEX_HEAD_DIM ** -0.5

# Scope3 tiles
SEQ_TILE = 64
MAX_SEQ_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
Q_VALID = 1
Q_PAD = 16
SORT_LEN = 8192
FP32_NEG_INF = -3.4028234663852886e38


def build_deepseek_v3_2_decode_front_scope123_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
):
    del v_head_dim

    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    NUM_HEADS_CFG = num_heads
    Q_LORA_RANK_CFG = q_lora_rank
    KV_LORA_RANK_CFG = kv_lora_rank
    QK_NOPE_HEAD_DIM_CFG = qk_nope_head_dim
    QK_ROPE_HEAD_DIM_CFG = qk_rope_head_dim
    QK_HEAD_DIM_CFG = qk_nope_head_dim + qk_rope_head_dim
    KV_A_OUT_CFG = kv_lora_rank + qk_rope_head_dim
    CACHE_ROWS_CFG = batch * max_seq_len

    INDEX_HEADS_CFG = index_heads
    INDEX_HEAD_DIM_CFG = index_head_dim
    INDEX_Q_OUT_CFG = index_heads * index_head_dim
    INDEX_HEAD_DIM_INV = 1.0 / index_head_dim
    WEIGHT_SCALE = (index_heads ** -0.5) * (index_head_dim ** -0.5)

    RMSNORM_BLOCKS = (HIDDEN_CFG + RMSNORM_K - 1) // RMSNORM_K
    PROJ_BLOCKS = (HIDDEN_CFG + PROJ_K - 1) // PROJ_K
    QR_BLOCKS = (Q_LORA_RANK_CFG + LORA_CHUNK - 1) // LORA_CHUNK
    KV_A_BLOCKS = (KV_A_OUT_CFG + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    IDX_OUT_BLOCKS = (INDEX_Q_OUT_CFG + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
    WK_OUT_BLOCKS = (INDEX_HEAD_DIM_CFG + IDX_OUT_CHUNK - 1) // IDX_OUT_CHUNK
    WEIGHTS_BLOCKS = (INDEX_HEADS_CFG + WEIGHTS_OUT_CHUNK - 1) // WEIGHTS_OUT_CHUNK

    Q_LORA_INV_CFG = 1.0 / q_lora_rank

    @pl.program
    class DeepSeekV32DecodeFrontScope123:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope123(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            norm_affine_pack: pl.Tensor[[3, HIDDEN_CFG + Q_LORA_RANK_CFG + KV_LORA_RANK_CFG], pl.FP32],
            wq_a: pl.Tensor[[HIDDEN_CFG, Q_LORA_RANK_CFG], pl.BF16],
            wq_b: pl.Tensor[[Q_LORA_RANK_CFG, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], pl.BF16],
            wkv_a: pl.Tensor[[HIDDEN_CFG, KV_A_OUT_CFG], pl.BF16],
            seq_lens: pl.Tensor[[BATCH_CFG], pl.INT32],
            rope_pair: pl.Tensor[[2 * MAX_SEQ_CFG, QK_ROPE_HEAD_DIM_CFG], pl.FP32],
            wq_b_idx: pl.Tensor[[Q_LORA_RANK_CFG, INDEX_Q_OUT_CFG], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
            weights_proj: pl.Tensor[[HIDDEN_CFG, INDEX_HEADS_CFG], pl.FP32],
            idx_init: pl.Tensor[[1, SORT_LEN], pl.UINT32],
            q_proj_out: pl.Tensor[[BATCH_CFG, NUM_HEADS_CFG * QK_HEAD_DIM_CFG], pl.BF16],
            kv_cache: pl.Tensor[[CACHE_ROWS_CFG, KV_LORA_RANK_CFG], pl.BF16],
            pe_cache: pl.Tensor[[CACHE_ROWS_CFG, QK_ROPE_HEAD_DIM_CFG], pl.BF16],
            k_cache_idx: pl.Tensor[[CACHE_ROWS_CFG, INDEX_HEAD_DIM_CFG], pl.BF16],
            topk_idx_out: pl.Tensor[[BATCH_CFG, INDEX_TOPK], pl.INT32],
        ) -> pl.Tensor[[BATCH_CFG, INDEX_TOPK], pl.INT32]:
            # Unpack packed inputs into logical weights to stay within runtime tensor-arg limits.
            input_rms_weight = pl.slice(norm_affine_pack, [1, HIDDEN_CFG], [0, 0])
            q_norm_weight = pl.slice(norm_affine_pack, [1, Q_LORA_RANK_CFG], [0, HIDDEN_CFG])
            kv_norm_weight = pl.slice(norm_affine_pack, [1, KV_LORA_RANK_CFG], [0, HIDDEN_CFG + Q_LORA_RANK_CFG])
            k_norm_weight = pl.slice(norm_affine_pack, [1, INDEX_HEAD_DIM_CFG], [1, 0])
            k_norm_bias = pl.slice(norm_affine_pack, [1, INDEX_HEAD_DIM_CFG], [2, 0])

            # ===== scope1: MLA front path (RMSNorm + Q/KV projection + RoPE + cache writeback) =====
            # Intermediates:
            # - qr_out: q_lora latent
            # - q_nope_out/q_pe_out: split Q head outputs (no-pe / pe parts)
            # - kv_a_out: KV latent output (includes rope part)
            qr_out = pl.create_tensor([BATCH_CFG, Q_LORA_RANK_CFG], dtype=pl.BF16)
            q_nope_out = pl.create_tensor([BATCH_CFG, NUM_HEADS_CFG * QK_NOPE_HEAD_DIM_CFG], dtype=pl.BF16)
            q_pe_out = pl.create_tensor([BATCH_CFG, NUM_HEADS_CFG * QK_ROPE_HEAD_DIM_CFG], dtype=pl.BF16)
            kv_a_out = pl.create_tensor([BATCH_CFG, KV_A_OUT_CFG], dtype=pl.BF16)
            for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                normed_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                qr_fp32_tile = pl.create_tensor([BATCH_TILE, Q_LORA_RANK_CFG], dtype=pl.FP32)
                kv_a_fp32_tile = pl.create_tensor([BATCH_TILE, KV_A_OUT_CFG], dtype=pl.FP32)

                # Stage 1.1: RMSNorm on hidden_states, then assemble blockwise into normed_tile.
                with pl.at(level=pl.Level.CORE_GROUP):
                    partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(RMSNORM_BLOCKS):
                        k0 = kb * RMSNORM_K
                        x_chunk = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, RMSNORM_K], [b0, k0]),
                            target_type=pl.FP32,
                        )
                        partial_sq = pl.add(
                            partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH_TILE]),
                        )

                    variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH_TILE, 1])
                    inv_rms = pl.recip(pl.sqrt(variance))

                    for kb in pl.range(PROJ_BLOCKS):
                        k0 = kb * PROJ_K
                        x_chunk_bf16 = pl.slice(hidden_states, [BATCH_TILE, PROJ_K], [b0, k0])
                        x_tile = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                        gamma = pl.slice(input_rms_weight, [1, PROJ_K], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_tile, inv_rms), gamma)
                        normed_tile = pl.assemble(
                            normed_tile,
                            pl.cast(normed, target_type=pl.BF16),
                            [0, k0],
                        )

                # Stage 1.2: Project q_a into q_lora latent with FP32 accumulation.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(0, QR_BLOCKS, 1, chunk=2):
                        q0 = ob * LORA_CHUNK
                        q_tile_a = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, 0])
                        q_tile_b = pl.slice(wq_a, [PROJ_K, LORA_CHUNK], [0, q0])
                        q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                        for kb in pl.range(1, PROJ_BLOCKS):
                            k0 = kb * PROJ_K
                            q_tile_a_i = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, k0])
                            q_tile_b_i = pl.slice(wq_a, [PROJ_K, LORA_CHUNK], [k0, q0])
                            q_acc = pl.matmul_acc(q_acc, q_tile_a_i, q_tile_b_i)
                        qr_fp32_tile = pl.assemble(qr_fp32_tile, q_acc, [0, q0])

                # Stage 1.3: Apply q_norm on q_lora latent to produce qr_out.
                with pl.at(level=pl.Level.CORE_GROUP):
                    q_partial_sq = pl.full([1, BATCH_TILE], dtype=pl.FP32, value=0.0)
                    for kb in pl.range(QR_BLOCKS):
                        k0 = kb * LORA_CHUNK
                        qr_chunk_fp32 = pl.slice(qr_fp32_tile, [BATCH_TILE, LORA_CHUNK], [0, k0])
                        q_partial_sq = pl.add(
                            q_partial_sq,
                            pl.reshape(pl.row_sum(pl.mul(qr_chunk_fp32, qr_chunk_fp32)), [1, BATCH_TILE]),
                        )

                    q_variance = pl.reshape(
                        pl.add(pl.mul(q_partial_sq, Q_LORA_INV_CFG), EPS),
                        [BATCH_TILE, 1],
                    )
                    q_inv_rms = pl.recip(pl.sqrt(q_variance))

                    for kb in pl.range(QR_BLOCKS):
                        k0 = kb * LORA_CHUNK
                        qr_chunk_bf16 = pl.cast(
                            pl.slice(qr_fp32_tile, [BATCH_TILE, LORA_CHUNK], [0, k0]),
                            target_type=pl.BF16,
                        )
                        qr_chunk_fp32 = pl.cast(qr_chunk_bf16, target_type=pl.FP32)
                        q_gamma = pl.slice(q_norm_weight, [1, LORA_CHUNK], [0, k0])
                        q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk_fp32, q_inv_rms), q_gamma)
                        q_normed_bf16 = pl.cast(q_normed, target_type=pl.BF16)
                        qr_out = pl.assemble(qr_out, q_normed_bf16, [b0, k0])

                # Stage 1.4: Project qr_out -> q_nope (no-pe slice for each head).
                for h in pl.parallel(0, NUM_HEADS_CFG, 1):
                    for q_loop in pl.parallel(0, QK_NOPE_HEAD_DIM_CFG, Q_OUT_CHUNK):
                        with pl.at(level=pl.Level.CORE_GROUP):
                            q_base = h * QK_HEAD_DIM_CFG
                            q0 = q_base + q_loop
                            q_out_acc = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                            for kb in pl.range(QR_BLOCKS):
                                k0 = kb * LORA_CHUNK
                                q_chunk = pl.slice(qr_out, [BATCH_TILE, LORA_CHUNK], [b0, k0])
                                wq_b_chunk = pl.slice(wq_b, [LORA_CHUNK, Q_OUT_CHUNK], [k0, q0])
                                q_out_acc = pl.add(q_out_acc, pl.matmul(q_chunk, wq_b_chunk))
                            q_nope_chunk = pl.cast(q_out_acc, target_type=pl.BF16)
                        q_nope_out = pl.assemble(q_nope_out, q_nope_chunk, [b0, h * QK_NOPE_HEAD_DIM_CFG + q_loop])

                # Stage 1.5: Project qr_out -> q_pe (pe slice for each head, before RoPE).
                for h in pl.parallel(0, NUM_HEADS_CFG, 1):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        q_base = h * QK_HEAD_DIM_CFG
                        q_pe_acc = pl.full([BATCH_TILE, QK_ROPE_HEAD_DIM_CFG], dtype=pl.FP32, value=0.0)
                        q_pe_col = q_base + QK_NOPE_HEAD_DIM_CFG
                        for kb in pl.range(QR_BLOCKS):
                            k0 = kb * LORA_CHUNK
                            q_chunk = pl.slice(qr_out, [BATCH_TILE, LORA_CHUNK], [b0, k0])
                            wq_b_chunk = pl.slice(wq_b, [LORA_CHUNK, QK_ROPE_HEAD_DIM_CFG], [k0, q_pe_col])
                            q_pe_acc = pl.add(q_pe_acc, pl.matmul(q_chunk, wq_b_chunk))
                        q_pe_raw = pl.cast(q_pe_acc, target_type=pl.BF16)
                    q_pe_out = pl.assemble(q_pe_out, q_pe_raw, [b0, h * QK_ROPE_HEAD_DIM_CFG])

                # Stage 1.6: Apply RoPE on q_pe using scope1's per-row update pattern.
                for h in pl.parallel(0, NUM_HEADS_CFG, 1):
                    with pl.at(level=pl.Level.CORE_GROUP):
                        q_pe_col = h * QK_ROPE_HEAD_DIM_CFG
                        for b in pl.range(BATCH_TILE):
                            ctx_len = pl.tensor.read(seq_lens, [b0 + b])
                            pos = ctx_len - 1
                            cos_lo = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [pos, 0])
                            cos_hi = pl.slice(
                                rope_pair,
                                [1, QK_ROPE_HEAD_DIM_CFG // 2],
                                [pos, QK_ROPE_HEAD_DIM_CFG // 2],
                            )
                            sin_lo = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [MAX_SEQ_CFG + pos, 0])
                            sin_hi = pl.slice(
                                rope_pair,
                                [1, QK_ROPE_HEAD_DIM_CFG // 2],
                                [MAX_SEQ_CFG + pos, QK_ROPE_HEAD_DIM_CFG // 2],
                            )
                            q_lo = pl.cast(
                                pl.slice(q_pe_out, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b0 + b, q_pe_col]),
                                target_type=pl.FP32,
                            )
                            q_hi = pl.cast(
                                pl.slice(
                                    q_pe_out,
                                    [1, QK_ROPE_HEAD_DIM_CFG // 2],
                                    [b0 + b, q_pe_col + QK_ROPE_HEAD_DIM_CFG // 2],
                                ),
                                target_type=pl.FP32,
                            )
                            q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                            q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                            q_pe_out = pl.assemble(q_pe_out, pl.cast(q_rot_lo, target_type=pl.BF16), [b0 + b, q_pe_col])
                            q_pe_out = pl.assemble(
                                q_pe_out,
                                pl.cast(q_rot_hi, target_type=pl.BF16),
                                [b0 + b, q_pe_col + QK_ROPE_HEAD_DIM_CFG // 2],
                            )

                # Stage 1.7: Project normed_tile -> kv_a_out (KV latent), FP32 accumulate then cast to BF16.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(0, KV_A_BLOCKS, 1, chunk=4):
                        kv0 = ob * KV_OUT_CHUNK
                        kv_tile_a = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, 0])
                        kv_tile_b = pl.slice(wkv_a, [PROJ_K, KV_OUT_CHUNK], [0, kv0])
                        kv_acc = pl.matmul(kv_tile_a, kv_tile_b, out_dtype=pl.FP32)
                        for kb in pl.range(1, PROJ_BLOCKS):
                            k0 = kb * PROJ_K
                            kv_tile_a_i = pl.slice(normed_tile, [BATCH_TILE, PROJ_K], [0, k0])
                            kv_tile_b_i = pl.slice(wkv_a, [PROJ_K, KV_OUT_CHUNK], [k0, kv0])
                            kv_acc = pl.matmul_acc(kv_acc, kv_tile_a_i, kv_tile_b_i)
                        kv_a_fp32_tile = pl.assemble(kv_a_fp32_tile, kv_acc, [0, kv0])

                # Stage 1.8: Blockwise cast/write for kv_a_out.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for ob in pl.parallel(0, KV_A_BLOCKS, 1, chunk=8):
                        kv0 = ob * KV_OUT_CHUNK
                        kv_chunk = pl.cast(
                            pl.slice(kv_a_fp32_tile, [BATCH_TILE, KV_OUT_CHUNK], [0, kv0]),
                            target_type=pl.BF16,
                        )
                        kv_a_out = pl.assemble(kv_a_out, kv_chunk, [b0, kv0])

            # Compose scope4-facing q projection from split q_nope/q_pe blocks.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    for h in pl.range(NUM_HEADS_CFG):
                        q_nope_row = pl.slice(
                            q_nope_out,
                            [BATCH_TILE, QK_NOPE_HEAD_DIM_CFG],
                            [b0, h * QK_NOPE_HEAD_DIM_CFG],
                        )
                        q_pe_row = pl.slice(
                            q_pe_out,
                            [BATCH_TILE, QK_ROPE_HEAD_DIM_CFG],
                            [b0, h * QK_ROPE_HEAD_DIM_CFG],
                        )
                        q_proj_out = pl.assemble(q_proj_out, q_nope_row, [b0, h * QK_HEAD_DIM_CFG])
                        q_proj_out = pl.assemble(
                            q_proj_out,
                            q_pe_row,
                            [b0, h * QK_HEAD_DIM_CFG + QK_NOPE_HEAD_DIM_CFG],
                        )

            # Stage 1.9: Apply kv_norm on the first KV_LORA_RANK channels of KV latent.
            kv_normed_out = pl.create_tensor([BATCH_CFG, KV_LORA_RANK_CFG], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP):
                kv_rows = pl.cast(pl.slice(kv_a_out, [BATCH_CFG, KV_LORA_RANK_CFG], [0, 0]), target_type=pl.FP32)
                kv_partial_sq = pl.reshape(pl.row_sum(pl.mul(kv_rows, kv_rows)), [1, BATCH_CFG])
                kv_variance = pl.reshape(
                    pl.add(pl.mul(kv_partial_sq, 1.0 / KV_LORA_RANK_CFG), EPS),
                    [BATCH_CFG, 1],
                )
                kv_inv_rms = pl.recip(pl.sqrt(kv_variance))
                kv_gamma = pl.slice(kv_norm_weight, [1, KV_LORA_RANK_CFG], [0, 0])
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rows, kv_inv_rms), kv_gamma)
                kv_normed_out = pl.assemble(kv_normed_out, pl.cast(kv_normed, target_type=pl.BF16), [0, 0])

            # Stage 1.10: Write decode caches.
            # - kv_cache: normalized KV latent
            # - pe_cache: rope component from kv_a_out after RoPE
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    ctx_len = pl.tensor.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cache_row = b * MAX_SEQ_CFG + pos

                    cos_lo = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [pos, 0])
                    cos_hi = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [pos, QK_ROPE_HEAD_DIM_CFG // 2])
                    sin_lo = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [MAX_SEQ_CFG + pos, 0])
                    sin_hi = pl.slice(
                        rope_pair,
                        [1, QK_ROPE_HEAD_DIM_CFG // 2],
                        [MAX_SEQ_CFG + pos, QK_ROPE_HEAD_DIM_CFG // 2],
                    )
                    kv_normed_row = pl.slice(kv_normed_out, [1, KV_LORA_RANK_CFG], [b, 0])

                    pe_lo = pl.cast(
                        pl.slice(kv_a_out, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, KV_LORA_RANK_CFG]),
                        target_type=pl.FP32,
                    )
                    pe_hi = pl.cast(
                        pl.slice(
                            kv_a_out,
                            [1, QK_ROPE_HEAD_DIM_CFG // 2],
                            [b, KV_LORA_RANK_CFG + QK_ROPE_HEAD_DIM_CFG // 2],
                        ),
                        target_type=pl.FP32,
                    )

                    pe_rot_lo = pl.sub(pl.col_expand_mul(pe_lo, cos_lo), pl.col_expand_mul(pe_hi, sin_lo))
                    pe_rot_hi = pl.add(pl.col_expand_mul(pe_hi, cos_hi), pl.col_expand_mul(pe_lo, sin_hi))

                    kv_cache = pl.assemble(kv_cache, kv_normed_row, [cache_row, 0])
                    pe_cache = pl.assemble(pe_cache, pl.cast(pe_rot_lo, target_type=pl.BF16), [cache_row, 0])
                    pe_cache = pl.assemble(
                        pe_cache,
                        pl.cast(pe_rot_hi, target_type=pl.BF16),
                        [cache_row, QK_ROPE_HEAD_DIM_CFG // 2],
                    )

            # ===== scope2: indexer path (prepare q_idx/k_cache_idx) =====
            # Outputs:
            # - q_idx_out: [B, INDEX_HEAD_DIM] aggregated index query vector
            # - k_cache_idx: index key cache row for current decode token
            q_idx_out = pl.create_tensor([BATCH_CFG, INDEX_HEAD_DIM_CFG], dtype=pl.BF16)
            q_idx_full = pl.create_tensor([BATCH_CFG, INDEX_Q_OUT_CFG], dtype=pl.BF16)
            k_idx = pl.create_tensor([BATCH_CFG, INDEX_HEAD_DIM_CFG], dtype=pl.BF16)
            weights = pl.create_tensor([BATCH_CFG, INDEX_HEADS_CFG], dtype=pl.FP32)

            # Stage 2.1: q_idx_full = wq_b_idx(qr_out).
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    for ob in pl.parallel(0, IDX_OUT_BLOCKS, 1, chunk=8):
                        q0 = ob * IDX_OUT_CHUNK
                        s2_q_acc = pl.full([BATCH_TILE, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(QR_BLOCKS):
                            k0 = kb * LORA_CHUNK
                            qr_chunk = pl.slice(qr_out, [BATCH_TILE, LORA_CHUNK], [b0, k0])
                            wq_chunk = pl.slice(wq_b_idx, [LORA_CHUNK, IDX_OUT_CHUNK], [k0, q0])
                            s2_q_acc = pl.add(s2_q_acc, pl.matmul(qr_chunk, wq_chunk, out_dtype=pl.FP32))
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_acc, target_type=pl.BF16), [b0, q0])

            # Stage 2.2: k_idx = wk_idx(hidden_states).
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    for ob in pl.parallel(0, WK_OUT_BLOCKS, 1, chunk=1):
                        k1 = ob * IDX_OUT_CHUNK
                        s2_k_acc = pl.full([BATCH_TILE, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            s2_x_chunk = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
                            wk_chunk = pl.slice(wk_idx, [K_CHUNK, IDX_OUT_CHUNK], [k0, k1])
                            s2_k_acc = pl.add(s2_k_acc, pl.matmul(s2_x_chunk, wk_chunk, out_dtype=pl.FP32))
                        k_idx = pl.assemble(k_idx, pl.cast(s2_k_acc, target_type=pl.BF16), [b0, k1])

            # Stage 2.3: Apply LayerNorm on k_idx (gamma/beta from k_norm_affine).
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    s2_k_tile = pl.cast(
                        pl.slice(k_idx, [BATCH_TILE, INDEX_HEAD_DIM_CFG], [b0, 0]),
                        target_type=pl.FP32,
                    )
                    s2_mean = pl.row_sum(pl.mul(s2_k_tile, INDEX_HEAD_DIM_INV))
                    s2_centered = pl.row_expand_sub(s2_k_tile, s2_mean)
                    s2_var_eps = pl.row_sum(pl.mul(pl.add(pl.mul(s2_centered, s2_centered), EPS), INDEX_HEAD_DIM_INV))
                    s2_std = pl.reshape(pl.sqrt(pl.reshape(s2_var_eps, [1, BATCH_TILE])), [BATCH_TILE, 1])
                    s2_inv_std = pl.recip(s2_std)
                    s2_normed = pl.row_expand_mul(s2_centered, s2_inv_std)
                    s2_gamma = pl.slice(k_norm_weight, [1, INDEX_HEAD_DIM_CFG], [0, 0])
                    s2_beta = pl.slice(k_norm_bias, [1, INDEX_HEAD_DIM_CFG], [0, 0])
                    s2_scaled = pl.col_expand_mul(s2_normed, s2_gamma)
                    s2_ones = pl.add(pl.sub(s2_k_tile, s2_k_tile), 1.0)
                    s2_k_normed = pl.add(s2_scaled, pl.col_expand_mul(s2_ones, s2_beta))
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_normed, target_type=pl.BF16), [b0, 0])

            # Stage 2.4: Apply RoPE on rope dimensions of q_idx_full and k_idx.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cos_lo = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [pos, 0])
                    cos_hi = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [pos, QK_ROPE_HEAD_DIM_CFG // 2])
                    sin_lo = pl.slice(rope_pair, [1, QK_ROPE_HEAD_DIM_CFG // 2], [MAX_SEQ_CFG + pos, 0])
                    sin_hi = pl.slice(
                        rope_pair,
                        [1, QK_ROPE_HEAD_DIM_CFG // 2],
                        [MAX_SEQ_CFG + pos, QK_ROPE_HEAD_DIM_CFG // 2],
                    )

                    for h in pl.range(INDEX_HEADS_CFG):
                        q_col = h * INDEX_HEAD_DIM_CFG
                        s2_q_lo = pl.cast(
                            pl.slice(q_idx_full, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, q_col]),
                            target_type=pl.FP32,
                        )
                        s2_q_hi = pl.cast(
                            pl.slice(q_idx_full, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, q_col + QK_ROPE_HEAD_DIM_CFG // 2]),
                            target_type=pl.FP32,
                        )
                        s2_q_rot_lo = pl.sub(pl.col_expand_mul(s2_q_lo, cos_lo), pl.col_expand_mul(s2_q_hi, sin_lo))
                        s2_q_rot_hi = pl.add(pl.col_expand_mul(s2_q_hi, cos_hi), pl.col_expand_mul(s2_q_lo, sin_hi))
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_rot_lo, target_type=pl.BF16), [b, q_col])
                        q_idx_full = pl.assemble(
                            q_idx_full,
                            pl.cast(s2_q_rot_hi, target_type=pl.BF16),
                            [b, q_col + QK_ROPE_HEAD_DIM_CFG // 2],
                        )

                    s2_k_lo = pl.cast(pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, 0]), target_type=pl.FP32)
                    s2_k_hi = pl.cast(
                        pl.slice(k_idx, [1, QK_ROPE_HEAD_DIM_CFG // 2], [b, QK_ROPE_HEAD_DIM_CFG // 2]),
                        target_type=pl.FP32,
                    )
                    s2_k_rot_lo = pl.sub(pl.col_expand_mul(s2_k_lo, cos_lo), pl.col_expand_mul(s2_k_hi, sin_lo))
                    s2_k_rot_hi = pl.add(pl.col_expand_mul(s2_k_hi, cos_hi), pl.col_expand_mul(s2_k_lo, sin_hi))
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_rot_lo, target_type=pl.BF16), [b, 0])
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_rot_hi, target_type=pl.BF16), [b, QK_ROPE_HEAD_DIM_CFG // 2])

            # Stage 2.5: Apply Hadamard scale (aligned with reference indexer behavior).
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    for ob in pl.parallel(0, IDX_OUT_BLOCKS, 1, chunk=8):
                        q0 = ob * IDX_OUT_CHUNK
                        s2_q_tile = pl.cast(
                            pl.slice(q_idx_full, [BATCH_TILE, IDX_OUT_CHUNK], [b0, q0]),
                            target_type=pl.FP32,
                        )
                        s2_q_scaled = pl.mul(s2_q_tile, HADAMARD_SCALE)
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_scaled, target_type=pl.BF16), [b0, q0])

                    s2_k_tile = pl.cast(
                        pl.slice(k_idx, [BATCH_TILE, INDEX_HEAD_DIM_CFG], [b0, 0]),
                        target_type=pl.FP32,
                    )
                    s2_k_scaled = pl.mul(s2_k_tile, HADAMARD_SCALE)
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_scaled, target_type=pl.BF16), [b0, 0])

            # Stage 2.6: Compute per-index-head weights.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    for ob in pl.parallel(0, WEIGHTS_BLOCKS, 1, chunk=1):
                        w0 = ob * WEIGHTS_OUT_CHUNK
                        s2_w_acc = pl.full([BATCH_TILE, WEIGHTS_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            s2_x_chunk = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
                            wp_chunk = pl.slice(weights_proj, [K_CHUNK, WEIGHTS_OUT_CHUNK], [k0, w0])
                            s2_w_acc = pl.add(
                                s2_w_acc,
                                pl.matmul(s2_x_chunk, pl.cast(wp_chunk, target_type=pl.BF16), out_dtype=pl.FP32),
                            )
                        w_scaled = pl.mul(s2_w_acc, WEIGHT_SCALE)
                        weights = pl.assemble(weights, w_scaled, [b0, w0])

            # Stage 2.7: Write current-token k_idx into k_cache_idx.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ_CFG + pos
                    s2_k_row = pl.slice(k_idx, [1, INDEX_HEAD_DIM_CFG], [b, 0])
                    k_cache_idx = pl.assemble(k_cache_idx, s2_k_row, [cache_row, 0])

            # Stage 2.8: Reduce q_idx_full across heads with weights to get q_idx_out.
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk]):
                s2_weights_t = pl.transpose(weights, axis1=0, axis2=1)
                for b0 in pl.parallel(0, BATCH_CFG, BATCH_TILE, chunk=1):
                    s2_q_idx_acc = pl.full([BATCH_TILE, INDEX_HEAD_DIM_CFG], dtype=pl.FP32, value=0.0)
                    for h in pl.range(INDEX_HEADS_CFG):
                        s2_q_h = pl.cast(
                            pl.slice(q_idx_full, [BATCH_TILE, INDEX_HEAD_DIM_CFG], [b0, h * INDEX_HEAD_DIM_CFG]),
                            target_type=pl.FP32,
                        )
                        s2_w_h_t = pl.slice(s2_weights_t, [1, BATCH_TILE], [h, b0])
                        s2_w_h = pl.reshape(s2_w_h_t, [BATCH_TILE, 1])
                        s2_q_idx_acc = pl.add(s2_q_idx_acc, pl.row_expand_mul(s2_q_h, s2_w_h))
                    q_idx_out = pl.assemble(q_idx_out, pl.cast(s2_q_idx_acc, target_type=pl.BF16), [b0, 0])

            # ===== scope3: index score + topk =====
            # Inputs: q_idx_out, k_cache_idx
            # Output: topk_idx_out (top-k positions per batch)
            topk_vals_out = pl.create_tensor([BATCH_CFG, INDEX_TOPK], dtype=pl.FP32)
            s3_q_padded = pl.create_tensor([BATCH_CFG * Q_PAD, INDEX_HEAD_DIM_CFG], dtype=pl.BF16)
            # Stage 3.1: Pad q_idx_out to Q_PAD rows for backend matmul row-alignment constraints.
            with pl.at(level=pl.Level.CORE_GROUP):
                for b in pl.range(BATCH_CFG):
                    s3_q_row = pl.slice(q_idx_out, [1, INDEX_HEAD_DIM_CFG], [b, 0])
                    s3_q_padded = pl.assemble(s3_q_padded, s3_q_row, [b * Q_PAD, 0])
                    s3_q_padded = pl.assemble(
                        s3_q_padded,
                        pl.cast(pl.full([Q_PAD - Q_VALID, INDEX_HEAD_DIM_CFG], dtype=pl.FP32, value=0.0), target_type=pl.BF16),
                        [b * Q_PAD + Q_VALID, 0],
                    )

            s3_scores = pl.create_tensor([BATCH_CFG, SORT_LEN], dtype=pl.FP32)
            s3_sorted_gm = pl.create_tensor([BATCH_CFG, 2 * SORT_LEN], dtype=pl.FP32)
            s3_raw_idx_gm = pl.create_tensor([BATCH_CFG, INDEX_TOPK], dtype=pl.INT32)

            for b in pl.range(0, BATCH_CFG, 1):
                s3_ctx_len = pl.tensor.read(seq_lens, [b])
                s3_ctx_blocks = (s3_ctx_len + SEQ_TILE - 1) // SEQ_TILE

                # Stage 3.2: Prefill full score row with -inf; only valid context range will be overwritten.
                with pl.at(level=pl.Level.CORE_GROUP):
                    s3_neg_inf_row = pl.full([1, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
                    s3_scores = pl.assemble(s3_scores, s3_neg_inf_row, [b, 0])

                s3_all_scores = pl.create_tensor([MAX_SEQ_BLOCKS * Q_PAD, SEQ_TILE], dtype=pl.FP32)
                # Stage 3.3: Compute tiled dot-product scores between q_idx_out and k_cache_idx.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for sb in pl.parallel(s3_ctx_blocks, chunk=MAX_SEQ_BLOCKS):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * MAX_SEQ_CFG + s0
                        s3_q_b = pl.slice(s3_q_padded, [Q_PAD, INDEX_HEAD_DIM_CFG], [b * Q_PAD, 0])
                        s3_k_tile = pl.slice(k_cache_idx, [SEQ_TILE, INDEX_HEAD_DIM_CFG], [cache_row0, 0])
                        s3_score_tile = pl.matmul(s3_q_b, s3_k_tile, b_trans=True, out_dtype=pl.FP32)
                        s3_all_scores = pl.assemble(s3_all_scores, s3_score_tile, [sb * Q_PAD, 0])

                # Stage 3.4: Fillpad each tile by valid_len, then write back to the global score row.
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for sb in pl.parallel(s3_ctx_blocks, chunk=MAX_SEQ_BLOCKS):
                        s0 = sb * SEQ_TILE
                        s3_valid_len = pl.min(SEQ_TILE, s3_ctx_len - s0)
                        s3_tile_valid = pl.slice(s3_all_scores, [1, SEQ_TILE], [sb * Q_PAD, 0], valid_shape=[1, s3_valid_len])
                        s3_tile_padded = pl.fillpad(s3_tile_valid, pad_value=pl.PadValue.min)
                        s3_scores = pl.assemble(s3_scores, s3_tile_padded, [b, s0])

                # Stage 3.5: Run sort32 + multi-pass merge sort to produce (score, idx) pairs.
                with pl.at(level=pl.Level.CORE_GROUP):
                    s3_score_row = pl.slice(s3_scores, [1, SORT_LEN], [b, 0])
                    s3_sorted_t = pl.tensor.sort32(s3_score_row, idx_init)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=64)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=256)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=1024)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=4096)
                    s3_sorted_gm = pl.assemble(s3_sorted_gm, s3_sorted_t, [b, 0])

                # Stage 3.6: Split top-k values and raw indices from interleaved pairs.
                with pl.at(level=pl.Level.CORE_GROUP):
                    s3_topk_pairs = pl.slice(s3_sorted_gm, [1, 2 * INDEX_TOPK], [b, 0])
                    s3_topk_v = pl.tensor.gather(s3_topk_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                    s3_topk_i_raw = pl.tensor.gather(
                        s3_topk_pairs,
                        mask_pattern=pl.tile.MaskPattern.P1010,
                        output_dtype=pl.INT32,
                    )
                    topk_vals_out = pl.assemble(topk_vals_out, s3_topk_v, [b, 0])
                    s3_raw_idx_gm = pl.assemble(s3_raw_idx_gm, s3_topk_i_raw, [b, 0])

                # Stage 3.7: Mark top-k slots beyond ctx_len with PadValue.min.
                with pl.at(level=pl.Level.CORE_GROUP):
                    s3_valid_topk = pl.min(INDEX_TOPK, s3_ctx_len)
                    s3_idx_valid = pl.slice(s3_raw_idx_gm, [1, INDEX_TOPK], [b, 0], valid_shape=[1, s3_valid_topk])
                    s3_idx_padded = pl.fillpad(s3_idx_valid, pad_value=pl.PadValue.min)
                    topk_idx_out = pl.assemble(topk_idx_out, s3_idx_padded, [b, 0])

            return topk_idx_out

    return DeepSeekV32DecodeFrontScope123


def golden_decode_front_scope123(tensors):
    import torch  # type: ignore[import]

    hidden_states = tensors["hidden_states"].float()
    norm_affine_pack = tensors["norm_affine_pack"].float()
    input_rms_weight = norm_affine_pack[0:1, :HIDDEN]
    q_norm_weight = norm_affine_pack[0:1, HIDDEN : HIDDEN + Q_LORA_RANK]
    kv_norm_weight = norm_affine_pack[0:1, HIDDEN + Q_LORA_RANK : HIDDEN + Q_LORA_RANK + KV_LORA_RANK]
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"].float()
    wkv_a = tensors["wkv_a"].float()
    seq_lens = tensors["seq_lens"]
    rope_pair = tensors["rope_pair"].float()
    rope_cos = rope_pair[:MAX_SEQ]
    rope_sin = rope_pair[MAX_SEQ:]

    wq_b_idx = tensors["wq_b_idx"].float()
    wk_idx = tensors["wk_idx"].float()
    k_norm_weight = norm_affine_pack[1:2, :INDEX_HEAD_DIM]
    k_norm_bias = norm_affine_pack[2:3, :INDEX_HEAD_DIM]
    weights_proj = tensors["weights_proj"].float()

    kv_cache = tensors["kv_cache"]
    pe_cache = tensors["pe_cache"]
    k_cache_idx = tensors["k_cache_idx"]

    sq_sum = torch.sum(hidden_states * hidden_states, dim=1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / HIDDEN) + EPS)
    normed = (hidden_states * inv_rms * input_rms_weight).to(torch.bfloat16).float()

    qr_raw = (normed @ wq_a).to(torch.bfloat16)
    qr_raw_fp32 = qr_raw.float()
    q_var = torch.mean(qr_raw_fp32 * qr_raw_fp32, dim=1, keepdim=True)
    qr = (qr_raw_fp32 * torch.rsqrt(q_var + EPS) * q_norm_weight).to(torch.bfloat16)

    q_proj = (qr.float() @ wq_b).to(torch.bfloat16)
    kv_a = (normed @ wkv_a).to(torch.bfloat16)

    half = QK_ROPE_HEAD_DIM // 2
    q_proj_view = q_proj.float().view(q_proj.shape[0], NUM_HEADS, QK_HEAD_DIM)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cache_row = b * MAX_SEQ + pos

        cos_lo = rope_cos[pos : pos + 1, :half]
        cos_hi = rope_cos[pos : pos + 1, half:]
        sin_lo = rope_sin[pos : pos + 1, :half]
        sin_hi = rope_sin[pos : pos + 1, half:]

        q_pe = q_proj_view[b, :, QK_NOPE_HEAD_DIM:]
        q_lo = q_pe[:, :half].clone()
        q_hi = q_pe[:, half:].clone()
        q_proj_view[b, :, QK_NOPE_HEAD_DIM : QK_NOPE_HEAD_DIM + half] = q_lo * cos_lo - q_hi * sin_lo
        q_proj_view[b, :, QK_NOPE_HEAD_DIM + half :] = q_hi * cos_hi + q_lo * sin_hi

        kv_row = kv_a[b : b + 1, :KV_LORA_RANK].float()
        kv_var = torch.mean(kv_row * kv_row, dim=-1, keepdim=True)
        kv_normed = kv_row * torch.rsqrt(kv_var + EPS) * kv_norm_weight
        kv_cache[cache_row : cache_row + 1].copy_(kv_normed.to(torch.bfloat16))

        pe_lo = kv_a[b : b + 1, KV_LORA_RANK : KV_LORA_RANK + half].float()
        pe_hi = kv_a[b : b + 1, KV_LORA_RANK + half : KV_LORA_RANK + 2 * half].float()
        pe_cache[cache_row : cache_row + 1, :half].copy_((pe_lo * cos_lo - pe_hi * sin_lo).to(torch.bfloat16))
        pe_cache[cache_row : cache_row + 1, half:].copy_((pe_hi * cos_hi + pe_lo * sin_hi).to(torch.bfloat16))

    tensors["q_proj_out"].copy_(q_proj_view.reshape(q_proj.shape[0], -1).to(torch.bfloat16))

    q_idx_full = (qr.float() @ wq_b_idx).to(torch.bfloat16).float()
    k_idx = (hidden_states @ wk_idx).to(torch.bfloat16).float()

    mean = k_idx.mean(dim=-1, keepdim=True)
    centered = k_idx - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    k_idx = (centered * torch.rsqrt(var + EPS) * k_norm_weight + k_norm_bias).to(torch.bfloat16).float()

    q_view = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cos_lo = rope_cos[pos : pos + 1, :half]
        cos_hi = rope_cos[pos : pos + 1, half:QK_ROPE_HEAD_DIM]
        sin_lo = rope_sin[pos : pos + 1, :half]
        sin_hi = rope_sin[pos : pos + 1, half:QK_ROPE_HEAD_DIM]

        q_pe_i = q_view[b, :, :QK_ROPE_HEAD_DIM]
        q_lo = q_pe_i[:, :half].clone()
        q_hi = q_pe_i[:, half:].clone()
        q_view[b, :, :half] = q_lo * cos_lo - q_hi * sin_lo
        q_view[b, :, half:QK_ROPE_HEAD_DIM] = q_hi * cos_hi + q_lo * sin_hi

        k_lo = k_idx[b : b + 1, :half].clone()
        k_hi = k_idx[b : b + 1, half:QK_ROPE_HEAD_DIM].clone()
        k_idx[b : b + 1, :half] = k_lo * cos_lo - k_hi * sin_lo
        k_idx[b : b + 1, half:QK_ROPE_HEAD_DIM] = k_hi * cos_hi + k_lo * sin_hi

    q_idx_full = q_view.reshape(BATCH, INDEX_HEADS * INDEX_HEAD_DIM)
    q_idx_full = (q_idx_full * HADAMARD_SCALE).to(torch.bfloat16).float()
    k_idx = (k_idx * HADAMARD_SCALE).to(torch.bfloat16).float()

    weights = (hidden_states @ weights_proj.to(torch.bfloat16).float()) * (INDEX_HEADS ** -0.5 * INDEX_HEAD_DIM ** -0.5)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        k_cache_idx[b * MAX_SEQ + pos, :].copy_(k_idx[b].to(torch.bfloat16))

    q_heads = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    q_idx = torch.einsum("bhd,bh->bd", q_heads, weights)
    scores = torch.full((BATCH, SORT_LEN), FP32_NEG_INF, dtype=torch.float32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        q_b = q_idx[b : b + 1]
        k_b = k_cache_idx[b * MAX_SEQ : b * MAX_SEQ + ctx_len].float()
        scores[b, :ctx_len] = (q_b @ k_b.T).squeeze(0)

    vals, idx = torch.topk(scores, INDEX_TOPK, dim=1, largest=True, sorted=True)
    del vals

    idx = idx.to(torch.int32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        valid_topk = min(INDEX_TOPK, ctx_len)
        idx[b, valid_topk:] = torch.iinfo(torch.int32).min
    tensors["topk_idx_out"].copy_(idx)


def build_tensor_specs(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    q_lora_rank: int = Q_LORA_RANK,
    kv_lora_rank: int = KV_LORA_RANK,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    qk_rope_head_dim: int = QK_ROPE_HEAD_DIM,
    index_heads: int = INDEX_HEADS,
    index_head_dim: int = INDEX_HEAD_DIM,
):
    import torch  # type: ignore[import]
    from golden import TensorSpec

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_a_out = kv_lora_rank + qk_rope_head_dim
    cache_rows = batch * max_seq_len
    index_q_out = index_heads * index_head_dim
    seq_lens_data = torch.randint(1, max_seq_len + 1, (batch,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(batch, hidden_size) - 0.5

    def init_rms_weight():
        return torch.rand(1, hidden_size) - 0.5

    def init_q_norm_weight():
        return torch.rand(1, q_lora_rank) - 0.5

    def init_wq_a():
        return (torch.rand(hidden_size, q_lora_rank) - 0.5) / hidden_size ** 0.5

    def init_wq_b():
        return (torch.rand(q_lora_rank, num_heads * qk_head_dim) - 0.5) / q_lora_rank ** 0.5

    def init_wkv_a():
        return (torch.rand(hidden_size, kv_a_out) - 0.5) / hidden_size ** 0.5

    def init_kv_norm_weight():
        return torch.rand(1, kv_lora_rank) - 0.5

    def init_norm_affine_pack():
        packed = torch.zeros((3, hidden_size + q_lora_rank + kv_lora_rank), dtype=torch.float32)
        packed[0:1, :] = torch.cat([init_rms_weight(), init_q_norm_weight(), init_kv_norm_weight()], dim=1)
        packed[1:2, :index_head_dim] = init_k_norm_weight()
        packed[2:3, :index_head_dim] = init_k_norm_bias()
        return packed

    def init_wq_b_idx():
        return (torch.rand(q_lora_rank, index_q_out) - 0.5) / q_lora_rank ** 0.5

    def init_wk_idx():
        return (torch.rand(hidden_size, index_head_dim) - 0.5) / hidden_size ** 0.5

    def init_k_norm_weight():
        return torch.rand(1, index_head_dim) - 0.5

    def init_k_norm_bias():
        return torch.rand(1, index_head_dim) - 0.5

    def init_weights_proj():
        return (torch.rand(hidden_size, index_heads) - 0.5) / hidden_size ** 0.5

    def init_rope():
        return torch.rand(max_seq_len, qk_rope_head_dim) - 0.5

    def init_rope_pair():
        return torch.cat([init_rope(), init_rope()], dim=0)

    def init_cache_kv():
        return torch.zeros(cache_rows, kv_lora_rank)

    def init_cache_pe():
        return torch.zeros(cache_rows, qk_rope_head_dim)

    def init_k_cache_idx():
        return torch.rand(cache_rows, index_head_dim) - 0.5

    def init_idx_init():
        return torch.arange(SORT_LEN, dtype=torch.int32).unsqueeze(0)

    def init_topk_idx_out():
        return torch.zeros((batch, INDEX_TOPK), dtype=torch.int32)

    return [
        TensorSpec("hidden_states", [batch, hidden_size], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec(
            "norm_affine_pack",
            [3, hidden_size + q_lora_rank + kv_lora_rank],
            torch.float32,
            init_value=init_norm_affine_pack,
        ),
        TensorSpec("wq_a", [hidden_size, q_lora_rank], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [q_lora_rank, num_heads * qk_head_dim], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv_a", [hidden_size, kv_a_out], torch.bfloat16, init_value=init_wkv_a),
        TensorSpec("seq_lens", [batch], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_pair", [2 * max_seq_len, qk_rope_head_dim], torch.float32, init_value=init_rope_pair),
        TensorSpec("wq_b_idx", [q_lora_rank, index_q_out], torch.bfloat16, init_value=init_wq_b_idx),
        TensorSpec("wk_idx", [hidden_size, index_head_dim], torch.bfloat16, init_value=init_wk_idx),
        TensorSpec("weights_proj", [hidden_size, index_heads], torch.float32, init_value=init_weights_proj),
        TensorSpec("idx_init", [1, SORT_LEN], torch.int32, init_value=init_idx_init),
        TensorSpec("q_proj_out", [batch, num_heads * qk_head_dim], torch.bfloat16, is_output=True),
        TensorSpec("kv_cache", [cache_rows, kv_lora_rank], torch.bfloat16, init_value=init_cache_kv, is_output=True),
        TensorSpec("pe_cache", [cache_rows, qk_rope_head_dim], torch.bfloat16, init_value=init_cache_pe, is_output=True),
        TensorSpec("k_cache_idx", [cache_rows, index_head_dim], torch.bfloat16, init_value=init_k_cache_idx, is_output=True),
        TensorSpec("topk_idx_out", [batch, INDEX_TOPK], torch.int32, init_value=init_topk_idx_out),
    ]


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
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v3_2_decode_front_scope123_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_decode_front_scope123,
        config=RunConfig(
            rtol=4e-3,
            atol=4e-3,
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
