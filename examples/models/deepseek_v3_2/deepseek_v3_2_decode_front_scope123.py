# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP decode front fused kernel (scope1 + scope2 + scope3).

Fused pipeline:
- Scope1: RMSNorm + Q/KV projection + RoPE + KV/PE cache update
- Scope2: indexer projection + indexer RoPE + INT8 quant/dequant + weight
  reduction + k_cache_idx update
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
Q_OUT_CHUNK = 128
KV_OUT_CHUNK = 64
LORA_CHUNK = 64

# Scope2 tiles
K_CHUNK = 128
IDX_OUT_CHUNK = 128
WEIGHTS_OUT_CHUNK = 64
HADAMARD_SCALE = INDEX_HEAD_DIM ** -0.5
INT8_GROUP_SIZE = 128
INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
INT8_SCALE_PACK = 8
INT8_Q_HEAD_TILE = 16  # rows of q_idx_grouped processed per quant step

# Scope3 tiles
SEQ_TILE = 64
MAX_SEQ_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
Q_VALID = 1
Q_PAD = 16
SORT_LEN = 8192
FP32_NEG_INF = -3.4028234663852886e38

# Derived config
INDEX_HEAD_DIM_INV = 1.0 / INDEX_HEAD_DIM
WEIGHT_SCALE = (INDEX_HEADS ** -0.5) * (INDEX_HEAD_DIM ** -0.5)
Q_LORA_INV = 1.0 / Q_LORA_RANK
if INDEX_HEAD_DIM != INT8_GROUP_SIZE:
    raise ValueError(
        f"INT8 quant path expects INDEX_HEAD_DIM == {INT8_GROUP_SIZE}, "
        f"got {INDEX_HEAD_DIM}"
    )


def build_deepseek_v3_2_decode_front_scope123_int8_quant_program():
    @pl.program
    class DeepSeekV32DecodeFrontScope123Int8Quant:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope123(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            q_norm_weight: pl.Tensor[[1, Q_LORA_RANK], pl.FP32],
            kv_norm_weight: pl.Tensor[[1, KV_LORA_RANK], pl.FP32],
            wq_a: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
            wq_b: pl.Tensor[[Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM], pl.BF16],
            wkv_a: pl.Tensor[[HIDDEN, KV_A_OUT], pl.BF16],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            wq_b_idx: pl.Tensor[[Q_LORA_RANK, INDEX_HEADS * INDEX_HEAD_DIM], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN, INDEX_HEAD_DIM], pl.BF16],
            weights_proj: pl.Tensor[[HIDDEN, INDEX_HEADS], pl.FP32],
            k_norm_weight: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            k_norm_bias: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            q_proj_out: pl.Tensor[[BATCH, NUM_HEADS * QK_HEAD_DIM], pl.BF16],
            kv_cache: pl.Tensor[[CACHE_ROWS, KV_LORA_RANK], pl.BF16],
            pe_cache: pl.Tensor[[CACHE_ROWS, QK_ROPE_HEAD_DIM], pl.BF16],
            k_cache_idx_i8: pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.INT8],
            k_cache_idx_scale: pl.Tensor[[BATCH, MAX_SEQ], pl.FP32],
            topk_idx_out: pl.Tensor[[BATCH, INDEX_TOPK], pl.INT32],
            topk_vals_out: pl.Tensor[[BATCH, INDEX_TOPK], pl.FP32],
        ) -> pl.Tensor[[BATCH, INDEX_TOPK], pl.INT32]:
            # ===== scope1: MLA front path (RMSNorm + Q/KV projection + RoPE + cache writeback) =====

            # Stage 1.1: RMSNorm on hidden_states.
            normed_states = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="input_rmsnorm"):
                partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, HIDDEN, RMSNORM_K):
                    x_chunk = pl.cast(hidden_states[:, k0 : k0 + RMSNORM_K], target_type=pl.FP32)
                    partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, BATCH]))
                variance = pl.reshape(pl.add(pl.mul(partial_sq, HIDDEN_INV), EPS), [BATCH, 1])
                inv_rms = pl.recip(pl.sqrt(variance))
                for k0 in pl.range(0, HIDDEN, PROJ_K):
                    x_chunk_bf16 = hidden_states[:, k0 : k0 + PROJ_K]
                    x_tile = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                    gamma = input_rms_weight[:, k0 : k0 + PROJ_K]
                    normed = pl.col_expand_mul(pl.row_expand_mul(x_tile, inv_rms), gamma)
                    normed_states = pl.assemble(normed_states, pl.cast(normed, target_type=pl.BF16), [0, k0])

            # Stage 1.2: Project qr = normed @ wq_a.
            qr_fp32 = pl.create_tensor([BATCH, Q_LORA_RANK], dtype=pl.FP32)
            for q0 in pl.parallel(0, Q_LORA_RANK, LORA_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_lora_proj"):
                    q_tile_a = normed_states[:, 0 : PROJ_K]
                    q_tile_b = wq_a[0 : PROJ_K, q0 : q0 + LORA_CHUNK]
                    q_acc = pl.matmul(q_tile_a, q_tile_b, out_dtype=pl.FP32)
                    for k0 in pl.range(PROJ_K, HIDDEN, PROJ_K):
                        q_tile_a_i = normed_states[:, k0 : k0 + PROJ_K]
                        q_tile_b_i = wq_a[k0 : k0 + PROJ_K, q0 : q0 + LORA_CHUNK]
                        q_acc = pl.matmul_acc(q_acc, q_tile_a_i, q_tile_b_i)
                    qr_fp32 = pl.assemble(qr_fp32, q_acc, [0, q0])

            # Stage 1.3: Apply q_norm on qr.
            qr_out = pl.create_tensor([BATCH, Q_LORA_RANK], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_lora_rmsnorm"):
                q_partial_sq = pl.full([1, BATCH], dtype=pl.FP32, value=0.0)
                for k0 in pl.range(0, Q_LORA_RANK, LORA_CHUNK):
                    qr_chunk_fp32 = qr_fp32[:, k0 : k0 + LORA_CHUNK]
                    q_partial = pl.reshape(pl.row_sum(pl.mul(qr_chunk_fp32, qr_chunk_fp32)), [1, BATCH])
                    q_partial_sq = pl.add(q_partial_sq, q_partial)
                q_variance = pl.reshape(pl.add(pl.mul(q_partial_sq, Q_LORA_INV), EPS), [BATCH, 1])
                q_inv_rms = pl.recip(pl.sqrt(q_variance))
                for k0 in pl.range(0, Q_LORA_RANK, LORA_CHUNK):
                    qr_chunk_bf16 = pl.cast(qr_fp32[:, k0 : k0 + LORA_CHUNK], target_type=pl.BF16)
                    qr_chunk_fp32 = pl.cast(qr_chunk_bf16, target_type=pl.FP32)
                    q_gamma = q_norm_weight[:, k0 : k0 + LORA_CHUNK]
                    q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk_fp32, q_inv_rms), q_gamma)
                    qr_out = pl.assemble(qr_out, pl.cast(q_normed, target_type=pl.BF16), [0, k0])

            # Stage 1.4: Project q_proj = qr @ wq_b.
            for q0 in pl.parallel(0, NUM_HEADS * QK_HEAD_DIM, Q_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_head_proj"):
                    q_chunk_init = qr_out[:, 0 : LORA_CHUNK]
                    wq_b_init = wq_b[0 : LORA_CHUNK, q0 : q0 + Q_OUT_CHUNK]
                    q_out_acc = pl.matmul(q_chunk_init, wq_b_init, out_dtype=pl.FP32)
                    for k0 in pl.range(LORA_CHUNK, Q_LORA_RANK, LORA_CHUNK):
                        q_chunk = qr_out[:, k0 : k0 + LORA_CHUNK]
                        wq_b_chunk = wq_b[k0 : k0 + LORA_CHUNK, q0 : q0 + Q_OUT_CHUNK]
                        q_out_acc = pl.matmul_acc(q_out_acc, q_chunk, wq_b_chunk)
                
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_head_proj_write"):
                    q_proj_out = pl.assemble(q_proj_out, pl.cast(q_out_acc, target_type=pl.BF16), [0, q0])

            # Stage 1.5: Project kv_a = normed @ wkv_a.
            kv_a_out = pl.create_tensor([BATCH, KV_A_OUT], dtype=pl.BF16)
            for kv0 in pl.parallel(0, KV_A_OUT, KV_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_a_proj"):
                    kv_tile_a = normed_states[:, 0 : PROJ_K]
                    kv_tile_b = wkv_a[0 : PROJ_K, kv0 : kv0 + KV_OUT_CHUNK]
                    kv_acc = pl.matmul(kv_tile_a, kv_tile_b, out_dtype=pl.FP32)
                    for k0 in pl.range(PROJ_K, HIDDEN, PROJ_K):
                        kv_tile_a_i = normed_states[:, k0 : k0 + PROJ_K]
                        kv_tile_b_i = wkv_a[k0 : k0 + PROJ_K, kv0 : kv0 + KV_OUT_CHUNK]
                        kv_acc = pl.matmul_acc(kv_acc, kv_tile_a_i, kv_tile_b_i)

                # Stage 1.6: Final KV output cast.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_a_write"):
                    kv_a_out = pl.assemble(kv_a_out, pl.cast(kv_acc, target_type=pl.BF16), [0, kv0])

            # Stage 1.7: Apply RoPE on q_proj_out in-place.
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope"):
                    ctx_len = pl.read(seq_lens, [b])
                    cos_lo = rope_cos[ctx_len - 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    cos_hi = rope_cos[ctx_len - 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    sin_lo = rope_sin[ctx_len - 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    sin_hi = rope_sin[ctx_len - 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    for q_col in pl.range(QK_NOPE_HEAD_DIM, NUM_HEADS * QK_HEAD_DIM, QK_HEAD_DIM):
                        q_lo = pl.cast(q_proj_out[b, q_col : q_col + QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                        q_hi = pl.cast(q_proj_out[b, q_col + QK_ROPE_HEAD_DIM // 2 : q_col + QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        q_proj_out = pl.assemble(q_proj_out, pl.cast(q_rot_lo, target_type=pl.BF16), [b, q_col])
                        q_proj_out = pl.assemble(q_proj_out, pl.cast(q_rot_hi, target_type=pl.BF16), [b, q_col + QK_ROPE_HEAD_DIM // 2])

            # Stage 1.8: Apply kv_norm on KV latent.
            kv_normed_out = pl.create_tensor([BATCH, KV_LORA_RANK], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_rmsnorm"):
                kv_rows = pl.cast(kv_a_out[:, 0 : KV_LORA_RANK], target_type=pl.FP32)
                kv_partial_sq = pl.reshape(pl.row_sum(pl.mul(kv_rows, kv_rows)), [1, BATCH])
                kv_variance = pl.reshape(pl.add(pl.mul(kv_partial_sq, 1.0 / KV_LORA_RANK), EPS), [BATCH, 1])
                kv_inv_rms = pl.recip(pl.sqrt(kv_variance))
                kv_gamma = kv_norm_weight[:, 0 : KV_LORA_RANK]
                kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rows, kv_inv_rms), kv_gamma)
                kv_normed_out = pl.assemble(kv_normed_out, pl.cast(kv_normed, target_type=pl.BF16), [0, 0])

            # Stage 1.9: Write decode caches.
            # - kv_cache: normalized KV latent
            # - pe_cache: rope component from kv_a_out after RoPE
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="decode_cache_write"):
                    ctx_len = pl.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cache_row = b * MAX_SEQ + pos
                    cos_lo = rope_cos[pos, 0 : QK_ROPE_HEAD_DIM // 2]
                    cos_hi = rope_cos[pos, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    sin_lo = rope_sin[pos, 0 : QK_ROPE_HEAD_DIM // 2]
                    sin_hi = rope_sin[pos, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    kv_normed_row = kv_normed_out[b, 0 : KV_LORA_RANK]
                    pe_lo = pl.cast(kv_a_out[b, KV_LORA_RANK : KV_LORA_RANK + QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                    pe_hi = pl.cast(kv_a_out[b, KV_LORA_RANK + QK_ROPE_HEAD_DIM // 2 : KV_LORA_RANK + QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                    pe_rot_lo = pl.sub(pl.col_expand_mul(pe_lo, cos_lo), pl.col_expand_mul(pe_hi, sin_lo))
                    pe_rot_hi = pl.add(pl.col_expand_mul(pe_hi, cos_hi), pl.col_expand_mul(pe_lo, sin_hi))
                    kv_cache = pl.assemble(kv_cache, kv_normed_row, [cache_row, 0])
                    pe_cache = pl.assemble(pe_cache, pl.cast(pe_rot_lo, target_type=pl.BF16), [cache_row, 0])
                    pe_cache = pl.assemble(pe_cache, pl.cast(pe_rot_hi, target_type=pl.BF16), [cache_row, QK_ROPE_HEAD_DIM // 2])

            # ===== scope2: indexer path (prepare q_idx/k_cache_idx) =====
            # Outputs:
            # - k_cache_idx_i8: INT8 index key cache row for current decode token

            # Stage 2.1: q_idx_full = wq_b_idx(qr_out).
            q_idx_full = pl.create_tensor([BATCH, INDEX_HEADS * INDEX_HEAD_DIM], dtype=pl.BF16)
            for q0 in pl.parallel(0, INDEX_HEADS * INDEX_HEAD_DIM, IDX_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_idx_proj"):
                    s2_q_chunk_init = qr_out[:, 0 : LORA_CHUNK]
                    s2_wq_chunk_init = wq_b_idx[0 : LORA_CHUNK, q0 : q0 + IDX_OUT_CHUNK]
                    s2_q_acc = pl.matmul(s2_q_chunk_init, s2_wq_chunk_init, out_dtype=pl.FP32)
                    for k0 in pl.range(LORA_CHUNK, Q_LORA_RANK, LORA_CHUNK):
                        qr_chunk = qr_out[:, k0 : k0 + LORA_CHUNK]
                        wq_chunk = wq_b_idx[k0 : k0 + LORA_CHUNK, q0 : q0 + IDX_OUT_CHUNK]
                        s2_q_acc = pl.matmul_acc(s2_q_acc, qr_chunk, wq_chunk)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_idx_proj_write"):
                    q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_acc, target_type=pl.BF16), [0, q0])

            # Stage 2.2: k_idx = wk_idx(hidden_states).
            # NOTE: pl.parallel currently iterates exactly once because
            # INDEX_HEAD_DIM == IDX_OUT_CHUNK. The loop is kept so scaling
            # INDEX_HEAD_DIM in the future does not require restructuring.
            k_idx = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            for k1 in pl.parallel(0, INDEX_HEAD_DIM, IDX_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_proj"):
                    s2_x_init = hidden_states[:, 0 : K_CHUNK]
                    wk_init = wk_idx[0 : K_CHUNK, k1 : k1 + IDX_OUT_CHUNK]
                    s2_k_acc = pl.matmul(s2_x_init, wk_init, out_dtype=pl.FP32)
                    for k0 in pl.range(K_CHUNK, HIDDEN, K_CHUNK):
                        s2_x_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                        wk_chunk = wk_idx[k0 : k0 + K_CHUNK, k1 : k1 + IDX_OUT_CHUNK]
                        s2_k_acc = pl.matmul_acc(s2_k_acc, s2_x_chunk, wk_chunk)

                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_proj_write"):
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_acc, target_type=pl.BF16), [0, k1])

            # Stage 2.3: Apply LayerNorm on k_idx (gamma/beta from k_norm_affine).
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_layernorm"):
                s2_k_tile = pl.cast(k_idx, target_type=pl.FP32)
                s2_mean = pl.row_sum(pl.mul(s2_k_tile, INDEX_HEAD_DIM_INV))
                s2_centered = pl.row_expand_sub(s2_k_tile, s2_mean)
                s2_var = pl.row_sum(pl.mul(pl.mul(s2_centered, s2_centered), INDEX_HEAD_DIM_INV))
                s2_var_eps = pl.add(s2_var, EPS)
                s2_std = pl.reshape(pl.sqrt(pl.reshape(s2_var_eps, [1, BATCH])), [BATCH, 1])
                s2_inv_std = pl.recip(s2_std)
                s2_normed = pl.row_expand_mul(s2_centered, s2_inv_std)
                s2_scaled = pl.col_expand_mul(s2_normed, k_norm_weight)
                s2_ones = pl.add(pl.sub(s2_k_tile, s2_k_tile), 1.0)
                s2_k_normed = pl.add(s2_scaled, pl.col_expand_mul(s2_ones, k_norm_bias))
                k_idx = pl.cast(s2_k_normed, target_type=pl.BF16)

            # Stage 2.4: Apply RoPE on rope dimensions of q_idx_full and k_idx.
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_idx_rope"):
                    pos = pl.read(seq_lens, [b]) - 1
                    cos_lo = rope_cos[pos, 0 : QK_ROPE_HEAD_DIM // 2]
                    cos_hi = rope_cos[pos, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    sin_lo = rope_sin[pos, 0 : QK_ROPE_HEAD_DIM // 2]
                    sin_hi = rope_sin[pos, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    for q_col in pl.range(0, INDEX_HEADS * INDEX_HEAD_DIM, INDEX_HEAD_DIM):
                        s2_q_lo = pl.cast(q_idx_full[b, q_col : q_col + QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                        s2_q_hi = pl.cast(q_idx_full[b, q_col + QK_ROPE_HEAD_DIM // 2 : q_col + QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                        s2_q_rot_lo = pl.sub(pl.col_expand_mul(s2_q_lo, cos_lo), pl.col_expand_mul(s2_q_hi, sin_lo))
                        s2_q_rot_hi = pl.add(pl.col_expand_mul(s2_q_hi, cos_hi), pl.col_expand_mul(s2_q_lo, sin_hi))
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_rot_lo, target_type=pl.BF16), [b, q_col])
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(s2_q_rot_hi, target_type=pl.BF16), [b, q_col + QK_ROPE_HEAD_DIM // 2])
                    s2_k_lo = pl.cast(k_idx[b, 0 : QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                    s2_k_hi = pl.cast(k_idx[b, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                    s2_k_rot_lo = pl.sub(pl.col_expand_mul(s2_k_lo, cos_lo), pl.col_expand_mul(s2_k_hi, sin_lo))
                    s2_k_rot_hi = pl.add(pl.col_expand_mul(s2_k_hi, cos_hi), pl.col_expand_mul(s2_k_lo, sin_hi))
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_rot_lo, target_type=pl.BF16), [b, 0])
                    k_idx = pl.assemble(k_idx, pl.cast(s2_k_rot_hi, target_type=pl.BF16), [b, QK_ROPE_HEAD_DIM // 2])

            # Stage 2.5: TODO: Apply Hadamard transform.
            # Hadamard is applied to q_idx/k_idx before INT8 quant in the upstream
            # DeepSeek V3.2-EXP reference. Skipping it here is acceptable only when
            # the golden reference also omits it; if the reference is updated, this
            # stage must be implemented to avoid scoring divergence.

            # Stage 2.5b: weights = weights_proj(hidden_states) * n_heads^-0.5 * head_dim^-0.5.
            weights_proj_bf16 = pl.create_tensor([HIDDEN, INDEX_HEADS], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_proj_bf16"):
                for k0 in pl.range(0, HIDDEN, K_CHUNK):
                    wp_tile = pl.cast(weights_proj[k0 : k0 + K_CHUNK, :], target_type=pl.BF16)
                    weights_proj_bf16 = pl.assemble(weights_proj_bf16, wp_tile, [k0, 0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_matmul"):
                s2_x_init = hidden_states[:, 0 : K_CHUNK]
                wp_init = weights_proj_bf16[0 : K_CHUNK, :]
                s2_w_acc = pl.matmul(s2_x_init, wp_init, out_dtype=pl.FP32)
                for k0 in pl.range(K_CHUNK, HIDDEN, K_CHUNK):
                    s2_x_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                    wp_chunk = weights_proj_bf16[k0 : k0 + K_CHUNK, :]
                    s2_w_acc = pl.matmul_acc(s2_w_acc, s2_x_chunk, wp_chunk)
            
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_scale"):
                weights = pl.mul(s2_w_acc, WEIGHT_SCALE)

            # Stage 2.6: Quantize indexer q/k tensors and keep INT8 outputs for later stages.
            q_idx_full_i8 = pl.create_tensor([BATCH * INDEX_HEADS, INDEX_HEAD_DIM], dtype=pl.INT8)
            q_idx_scale_heads = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)
            q_idx_grouped = pl.reshape(q_idx_full, [BATCH * INDEX_HEADS, INDEX_HEAD_DIM])
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_int8_quant_1"):
                    for h in pl.range(0, INDEX_HEADS, INT8_Q_HEAD_TILE):
                        r0 = b * INDEX_HEADS + h
                        s2_q_block = pl.cast(q_idx_grouped[r0 : r0 + INT8_Q_HEAD_TILE, :], target_type=pl.FP32, mode="none")
                        s2_q_abs = pl.maximum(s2_q_block, pl.neg(s2_q_block))
                        s2_q_amax_row = pl.reshape(pl.row_max(s2_q_abs), [1, INT8_Q_HEAD_TILE])
                        s2_q_amax_row = pl.maximum(s2_q_amax_row, pl.full([1, INT8_Q_HEAD_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS))
                        s2_q_scale_quant_row = pl.div(pl.full([1, INT8_Q_HEAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), s2_q_amax_row)
                        s2_q_scale_dequant_row = pl.div(pl.full([1, INT8_Q_HEAD_TILE], dtype=pl.FP32, value=1.0), s2_q_scale_quant_row)
                        s2_q_scale_quant = pl.reshape(s2_q_scale_quant_row, [INT8_Q_HEAD_TILE, 1])
                        s2_q_scaled = pl.row_expand_mul(s2_q_block, s2_q_scale_quant)
                        s2_q_i32 = pl.cast(s2_q_scaled, target_type=pl.INT32, mode="round")
                        s2_q_half = pl.cast(s2_q_i32, target_type=pl.FP16, mode="round")
                        s2_q_i8 = pl.cast(s2_q_half, target_type=pl.INT8, mode="trunc")
                        q_idx_full_i8 = pl.assemble(q_idx_full_i8, s2_q_i8, [r0, 0])
                        q_idx_scale_heads = pl.assemble(q_idx_scale_heads, s2_q_scale_dequant_row, [b, h])
            
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_int8_quant_2"):
                s2_k_block = pl.cast(k_idx, target_type=pl.FP32, mode="none")
                s2_k_abs = pl.maximum(s2_k_block, pl.neg(s2_k_block))
                s2_k_amax_row = pl.reshape(pl.row_max(s2_k_abs), [1, BATCH])
                s2_k_amax_row = pl.maximum(s2_k_amax_row, pl.full([1, BATCH], dtype=pl.FP32, value=INT8_AMAX_EPS))
                s2_k_scale_quant_row = pl.div(pl.full([1, BATCH], dtype=pl.FP32, value=INT8_SCALE_MAX), s2_k_amax_row)
                s2_k_scale_dequant_row = pl.div(pl.full([1, BATCH], dtype=pl.FP32, value=1.0), s2_k_scale_quant_row)
                s2_k_scale_quant = pl.reshape(s2_k_scale_quant_row, [BATCH, 1])
                s2_k_scale_dequant = pl.reshape(s2_k_scale_dequant_row, [BATCH, 1])
                s2_k_scale_pack_target = pl.full([BATCH, INT8_SCALE_PACK], dtype=pl.FP32, value=0.0)
                k_idx_scale = pl.row_expand(s2_k_scale_pack_target, s2_k_scale_dequant)
                s2_k_scaled = pl.row_expand_mul(s2_k_block, s2_k_scale_quant)
                s2_k_i32 = pl.cast(s2_k_scaled, target_type=pl.INT32, mode="round")
                s2_k_half = pl.cast(s2_k_i32, target_type=pl.FP16, mode="round")
                k_idx_i8 = pl.cast(s2_k_half, target_type=pl.INT8, mode="trunc")

            # Stage 2.8: Write current-token k_idx into INT8 cache form.
            k_idx_scale_flat = pl.reshape(k_idx_scale, [BATCH * INT8_SCALE_PACK])
            k_cache_idx_scale_flat = pl.reshape(k_cache_idx_scale, [BATCH * MAX_SEQ])
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_cache_write"):
                    pos = pl.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ + pos
                    s2_k_row_scale = pl.read(k_idx_scale_flat, [b * INT8_SCALE_PACK])
                    k_cache_idx_i8 = pl.assemble(k_cache_idx_i8, k_idx_i8[b, :], [cache_row, 0])
                    pl.write(k_cache_idx_scale_flat, [b * MAX_SEQ + pos], s2_k_row_scale)

            # ===== scope3: index score + topk =====
            # Inputs: q_idx_full_i8, k_cache_idx_i8
            # Output: topk_idx_out (top-k positions per batch)

            for b in pl.parallel(BATCH):
                # Stage 3.0: Pad q_idx_full_i8 and pre-fill scores.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_prefill_scores"):
                    scores = pl.full([1, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
                s3_q_i8_padded = pl.create_tensor([INDEX_HEADS * Q_PAD, INDEX_HEAD_DIM], dtype=pl.INT8)
                s3_q_s_padded = pl.create_tensor([Q_PAD, INDEX_HEADS], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_q_i8_init"):
                    for q_row0 in pl.range(INDEX_HEADS):
                        s3_q_i8_valid = q_idx_full_i8[b * INDEX_HEADS + q_row0, :]
                        s3_q_i8_padded = pl.assemble(s3_q_i8_padded, s3_q_i8_valid, [q_row0 * Q_PAD, 0])
                        s3_q_i8_zero_pad = pl.cast(pl.full([Q_PAD - Q_VALID, INDEX_HEAD_DIM], dtype=pl.INT16, value=0), target_type=pl.INT8)
                        s3_q_i8_padded = pl.assemble(s3_q_i8_padded, s3_q_i8_zero_pad, [q_row0 * Q_PAD + Q_VALID, 0])
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_q_s_init"):
                    s3_weights_row = weights[b, :]
                    s3_q_scales_row = q_idx_scale_heads[b, :]
                    s3_q_s_row = pl.mul(s3_weights_row, s3_q_scales_row)
                    s3_q_s_padded = pl.assemble(s3_q_s_padded, s3_q_s_row, [0, 0])
                    s3_q_s_padded = pl.assemble(s3_q_s_padded, pl.full([Q_PAD - Q_VALID, INDEX_HEADS], dtype=pl.FP32, value=0.0), [Q_VALID, 0])

                s3_ctx_len = pl.read(seq_lens, [b])
                s3_ctx_blocks = (s3_ctx_len + SEQ_TILE - 1) // SEQ_TILE

                # Stage 3.1: Compute tiled INT8 qk logits.
                s3_all_scores_i8 = pl.create_tensor([MAX_SEQ_BLOCKS * INDEX_HEADS * Q_PAD, SEQ_TILE], dtype=pl.INT32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_int8_qk_matmul"):
                    for sb in pl.range(s3_ctx_blocks):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * MAX_SEQ + s0
                        s3_k_tile_i8 = k_cache_idx_i8[cache_row0 : cache_row0 + SEQ_TILE, 0 : INDEX_HEAD_DIM]
                        for h in pl.range(INDEX_HEADS):
                            q_row0 = h * Q_PAD
                            tile_row0 = (sb * INDEX_HEADS + h) * Q_PAD
                            s3_q_tile_i8 = s3_q_i8_padded[q_row0 : q_row0 + Q_PAD, 0 : INDEX_HEAD_DIM]
                            s3_logits_i32 = pl.matmul(s3_q_tile_i8, s3_k_tile_i8, b_trans=True, out_dtype=pl.INT32)
                            s3_all_scores_i8 = pl.assemble(s3_all_scores_i8, s3_logits_i32, [tile_row0, 0])

                # Stage 3.2: Cast to FP32 and apply ReLU.
                s3_relu_rows = pl.create_tensor([MAX_SEQ_BLOCKS * INDEX_HEADS, SEQ_TILE], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_logits_relu"):
                    for sb in pl.range(s3_ctx_blocks):
                        for h in pl.range(INDEX_HEADS):
                            tile_row0 = (sb * INDEX_HEADS + h) * Q_PAD
                            s3_logits_row_i32 = s3_all_scores_i8[tile_row0, :]
                            s3_logits_row_f32 = pl.cast(s3_logits_row_i32, target_type=pl.FP32, mode="none")
                            s3_relu_logits = pl.maximum(s3_logits_row_f32, pl.mul(s3_logits_row_f32, 0.0))
                            relu_row0 = sb * INDEX_HEADS + h
                            s3_relu_rows = pl.assemble(s3_relu_rows, s3_relu_logits, [relu_row0, 0])

                # Stage 3.3: Reduce per-head ReLU logits with q_s weights.
                s3_weighted_scores = pl.create_tensor([MAX_SEQ_BLOCKS * Q_PAD, SEQ_TILE], dtype=pl.FP32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_head_reduce"):
                    for sb in pl.range(s3_ctx_blocks):
                        s3_q_s_tile = s3_q_s_padded[:, :]
                        s3_relu_row0 = sb * INDEX_HEADS
                        s3_relu_tile = s3_relu_rows[s3_relu_row0 : s3_relu_row0 + INDEX_HEADS, :]
                        s3_weighted_tile = pl.matmul(s3_q_s_tile, s3_relu_tile, out_dtype=pl.FP32)
                        weighted_row0 = sb * Q_PAD
                        s3_weighted_scores = pl.assemble(s3_weighted_scores, s3_weighted_tile, [weighted_row0, 0])

                # Stage 3.4: Apply k scale and write valid score tiles.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_score_write"):
                    for sb in pl.range(s3_ctx_blocks):
                        s0 = sb * SEQ_TILE
                        s3_valid_len = pl.min(SEQ_TILE, s3_ctx_len - s0)
                        weighted_row0 = sb * Q_PAD
                        s3_k_scale = k_cache_idx_scale[b, s0 : s0 + SEQ_TILE]
                        s3_score_acc = s3_weighted_scores[weighted_row0, :]
                        s3_score_tile = pl.mul(s3_score_acc, s3_k_scale)
                        s3_score_2d = pl.reshape(s3_score_tile, [1, SEQ_TILE])
                        s3_score_valid = pl.slice(
                            s3_score_2d,
                            [1, SEQ_TILE],
                            [0, 0],
                            valid_shape=[1, s3_valid_len],
                        )
                        scores = pl.assemble(scores, s3_score_valid, [0, s0])

                s3_sorted_gm = pl.create_tensor([1, 2 * SORT_LEN], dtype=pl.FP32)
                # Stage 3.5: Run sort32 + mrgsort.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_sort"):
                    s3_score_row = scores[0, :]
                    idx_init = pl.tensor.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                    s3_sorted_t = pl.tensor.sort32(s3_score_row, idx_init)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=64)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=256)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=1024)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=4096)
                    s3_sorted_gm = pl.assemble(s3_sorted_gm, s3_sorted_t, [0, 0])

                # Stage 3.6: Split top-k values and raw indices.
                s3_raw_idx_local = pl.create_tensor([1, INDEX_TOPK], dtype=pl.INT32)
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_topk_extract"):
                    s3_topk_pairs = s3_sorted_gm[:, 0 : 2 * INDEX_TOPK]
                    s3_topk_v = pl.tensor.gather(s3_topk_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
                    s3_topk_i_raw = pl.tensor.gather(s3_topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    topk_vals_out = pl.assemble(topk_vals_out, s3_topk_v, [b, 0])
                    s3_raw_idx_local = pl.assemble(s3_raw_idx_local, s3_topk_i_raw, [0, 0])
                    s3_valid_topk = pl.min(INDEX_TOPK, s3_ctx_len)
                    s3_idx_valid = pl.slice(s3_raw_idx_local, [1, INDEX_TOPK], [0, 0], valid_shape=[1, s3_valid_topk])
                    s3_idx_padded = pl.fillpad(s3_idx_valid, pad_value=pl.PadValue.min)
                    topk_idx_out = pl.assemble(topk_idx_out, s3_idx_padded, [b, 0])

            return topk_idx_out

    return DeepSeekV32DecodeFrontScope123Int8Quant


def golden_decode_front_scope123_int8_quant(tensors):
    import torch  # type: ignore[import]

    def round_half_away_from_zero(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    def int8_quant_groups(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rows = x.reshape(-1, INDEX_HEAD_DIM).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        scale_dequant = 1.0 / scale_quant
        return out_i8.reshape_as(x), scale_dequant

    hidden_states = tensors["hidden_states"].float()
    input_rms_weight = tensors["input_rms_weight"].float()
    q_norm_weight = tensors["q_norm_weight"].float()
    kv_norm_weight = tensors["kv_norm_weight"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"].float()
    wkv_a = tensors["wkv_a"].float()
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()

    wq_b_idx = tensors["wq_b_idx"].float()
    wk_idx = tensors["wk_idx"].float()
    weights_proj = tensors["weights_proj"].float()
    k_norm_weight = tensors["k_norm_weight"].float()
    k_norm_bias = tensors["k_norm_bias"].float()
    kv_cache = tensors["kv_cache"]
    pe_cache = tensors["pe_cache"]
    k_cache_idx_i8 = tensors["k_cache_idx_i8"]
    k_cache_idx_scale = tensors["k_cache_idx_scale"]

    sq_sum = torch.sum(hidden_states * hidden_states, dim=1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / HIDDEN) + EPS)
    normed = (hidden_states * inv_rms * input_rms_weight).to(torch.bfloat16).float()

    qr_raw_fp32 = (normed @ wq_a)
    q_var = torch.mean(qr_raw_fp32 * qr_raw_fp32, dim=1, keepdim=True)
    qr = (qr_raw_fp32.to(torch.bfloat16).float() * torch.rsqrt(q_var + EPS) * q_norm_weight).to(torch.bfloat16)

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

    q_idx_full = q_view.reshape(BATCH, INDEX_HEADS * INDEX_HEAD_DIM).to(torch.bfloat16).float()
    k_idx = k_idx.to(torch.bfloat16).float()
    weights = (hidden_states @ weights_proj.to(torch.bfloat16).float()) * (
        INDEX_HEADS ** -0.5 * INDEX_HEAD_DIM ** -0.5
    )
    q_idx_full_pre_quant = q_idx_full.clone()
    k_idx_pre_quant = k_idx.clone()
    q_idx_i8_golden, q_idx_scale_golden = int8_quant_groups(q_idx_full_pre_quant)
    k_idx_i8_golden, k_idx_scale_golden = int8_quant_groups(k_idx_pre_quant)

    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        k_cache_idx_i8[b * MAX_SEQ + pos, :].copy_(k_idx_i8_golden[b])
        k_cache_idx_scale[b, pos].copy_(k_idx_scale_golden[b, 0])

    q_idx_i8_view = q_idx_i8_golden.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM).to(torch.int32)
    scores = torch.full((BATCH, SORT_LEN), FP32_NEG_INF, dtype=torch.float32)
    q_scale = q_idx_scale_golden.view(BATCH, INDEX_HEADS)
    q_s = weights * q_scale
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
        for sb in range(ctx_blocks):
            s0 = sb * SEQ_TILE
            valid_len = min(SEQ_TILE, ctx_len - s0)
            cache_row0 = b * MAX_SEQ + s0
            k_tile = k_cache_idx_i8[cache_row0 : cache_row0 + SEQ_TILE].to(torch.int32)
            logits = torch.matmul(q_idx_i8_view[b, :INDEX_HEADS], k_tile.transpose(0, 1)).float()
            score = (torch.relu(logits[:, :valid_len]) * q_s[b, :INDEX_HEADS, None]).sum(dim=0)
            score = score * k_cache_idx_scale[b, s0 : s0 + valid_len].float()
            scores[b, s0 : s0 + valid_len] = score

    sorted_vals, sorted_idx = torch.sort(scores, dim=1, descending=True, stable=True)

    idx = sorted_idx[:, :INDEX_TOPK].to(torch.int32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        valid_topk = min(INDEX_TOPK, ctx_len)
        idx[b, valid_topk:] = torch.iinfo(torch.int32).min
    tensors["topk_idx_out"].copy_(idx)
    tensors["topk_vals_out"].copy_(sorted_vals[:, :INDEX_TOPK])


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    seq_lens_data = torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_rms_weight():
        return torch.rand(1, HIDDEN) - 0.5

    def init_q_norm_weight():
        return torch.rand(1, Q_LORA_RANK) - 0.5

    def init_wq_a():
        return (torch.rand(HIDDEN, Q_LORA_RANK) - 0.5) / HIDDEN ** 0.5

    def init_wq_b():
        return (torch.rand(Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM) - 0.5) / Q_LORA_RANK ** 0.5

    def init_wkv_a():
        return (torch.rand(HIDDEN, KV_A_OUT) - 0.5) / HIDDEN ** 0.5

    def init_kv_norm_weight():
        return torch.rand(1, KV_LORA_RANK) - 0.5

    def init_wq_b_idx():
        return (torch.rand(Q_LORA_RANK, INDEX_HEADS * INDEX_HEAD_DIM) - 0.5) / Q_LORA_RANK ** 0.5

    def init_wk_idx():
        return (torch.rand(HIDDEN, INDEX_HEAD_DIM) - 0.5) / HIDDEN ** 0.5

    def init_weights_proj():
        return (torch.rand(HIDDEN, INDEX_HEADS) - 0.5) / HIDDEN ** 0.5

    def init_k_norm_weight():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_k_norm_bias():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_rope():
        return torch.rand(MAX_SEQ, QK_ROPE_HEAD_DIM) - 0.5

    def init_cache_kv():
        return torch.zeros(BATCH * MAX_SEQ, KV_LORA_RANK)

    def init_cache_pe():
        return torch.zeros(BATCH * MAX_SEQ, QK_ROPE_HEAD_DIM)

    def init_k_cache_idx_i8():
        return torch.randint(-128, 128, (BATCH * MAX_SEQ, INDEX_HEAD_DIM), dtype=torch.int8)

    def init_k_cache_idx_scale():
        return torch.rand((BATCH, MAX_SEQ), dtype=torch.float32) / 127.0

    return [
        TensorSpec("hidden_states", [BATCH, HIDDEN], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, HIDDEN], torch.float32, init_value=init_rms_weight),
        TensorSpec("q_norm_weight", [1, Q_LORA_RANK], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("kv_norm_weight", [1, KV_LORA_RANK], torch.float32, init_value=init_kv_norm_weight),
        TensorSpec("wq_a", [HIDDEN, Q_LORA_RANK], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv_a", [HIDDEN, KV_A_OUT], torch.bfloat16, init_value=init_wkv_a),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_cos", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope),
        TensorSpec("rope_sin", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope),
        TensorSpec("wq_b_idx", [Q_LORA_RANK, INDEX_HEADS * INDEX_HEAD_DIM], torch.bfloat16, init_value=init_wq_b_idx),
        TensorSpec("wk_idx", [HIDDEN, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_wk_idx),
        TensorSpec("weights_proj", [HIDDEN, INDEX_HEADS], torch.float32, init_value=init_weights_proj),
        TensorSpec("k_norm_weight", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("k_norm_bias", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_bias),
        TensorSpec("q_proj_out", [BATCH, NUM_HEADS * QK_HEAD_DIM], torch.bfloat16, is_output=True),
        TensorSpec("kv_cache", [BATCH * MAX_SEQ, KV_LORA_RANK], torch.bfloat16, init_value=init_cache_kv, is_output=True),
        TensorSpec("pe_cache", [BATCH * MAX_SEQ, QK_ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cache_pe, is_output=True),
        TensorSpec("k_cache_idx_i8", [BATCH * MAX_SEQ, INDEX_HEAD_DIM], torch.int8, init_value=init_k_cache_idx_i8),
        TensorSpec("k_cache_idx_scale", [BATCH, MAX_SEQ], torch.float32, init_value=init_k_cache_idx_scale),
        TensorSpec("topk_idx_out", [BATCH, INDEX_TOPK], torch.int32, is_output=True),
        TensorSpec("topk_vals_out", [BATCH, INDEX_TOPK], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_deepseek_v3_2_decode_front_scope123_int8_quant_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_decode_front_scope123_int8_quant,
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
