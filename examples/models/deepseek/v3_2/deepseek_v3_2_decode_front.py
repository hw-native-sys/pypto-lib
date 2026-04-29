# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP decode front fused kernel (scope1 + scope2 + scope3 + scope4).

Fused pipeline:
- Scope1: RMSNorm + Q/KV projection + RoPE + KV/PE cache update
- Scope2: indexer projection + indexer RoPE + weight reduction + k_cache_idx update
- Scope3: q_idx x k_cache_idx scoring + topk
- Scope4: sparse attention with topk indices, online softmax, latent-to-V projection, dispatch buffer write

Decode contract:
- `seq_lens[b]` must be in `[1, MAX_SEQ]` for active rows.
- Zero-length rows are invalid for this fused decode kernel because scope1/scope2
    write the current token at `pos = seq_len - 1` before scope4 runs.
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
INDEX_Q_OUT = INDEX_HEADS * INDEX_HEAD_DIM
INDEX_TOPK = 2048
EP_NODES = 128
ATTN_OUT = NUM_HEADS * V_HEAD_DIM

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
INDEX_HEAD_DIM_INV = 1.0 / INDEX_HEAD_DIM
WEIGHT_SCALE = (INDEX_HEADS ** -0.5) * (INDEX_HEAD_DIM ** -0.5)
Q_LORA_INV = 1.0 / Q_LORA_RANK

# Scope1 tiles
RMSNORM_K = 512
PROJ_K = 512
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 64
LORA_CHUNK = 64

# Scope2 tiles
K_CHUNK = 128
IDX_OUT_CHUNK = 128
KIDX_OUT_CHUNK = 64
QREDUCE_OUT_CHUNK = 64
QREDUCE_BATCH_TILE = 16
WEIGHTS_OUT_CHUNK = 16

# Scope3 tiles
SEQ_TILE = 64
MAX_SEQ_BLOCKS = (MAX_SEQ + SEQ_TILE - 1) // SEQ_TILE
Q_VALID = 1
Q_PAD = 16
SORT_LEN = 8192
FP32_NEG_INF = -3.4028234663852886e38

# Scope4 tiles
ATTN_SCALE = 1.0 / (QK_HEAD_DIM**0.5)
Q_LATENT_CHUNK = 128
V_OUT_CHUNK = 16
MATMUL_ROW_PAD = 16


def build_deepseek_v3_2_decode_front_scope1234_program():
    @pl.program
    class DeepSeekV32DecodeFrontScope1234:
        @pl.function(type=pl.FunctionType.Opaque)
        def deepseek_v3_2_decode_front_scope1234(
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
            wq_b_idx: pl.Tensor[[Q_LORA_RANK, INDEX_Q_OUT], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN, INDEX_HEAD_DIM], pl.BF16],
            weights_proj: pl.Tensor[[HIDDEN, INDEX_HEADS], pl.FP32],
            k_norm_weight: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            k_norm_bias: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            hadamard_q: pl.Tensor[[INDEX_HEAD_DIM, INDEX_HEAD_DIM], pl.BF16],
            hadamard_k: pl.Tensor[[INDEX_HEAD_DIM, INDEX_HEAD_DIM], pl.BF16],
            layer_id_t: pl.Tensor[[1], pl.INT32],
            w_q_nope_to_latent: pl.Tensor[[NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK], pl.BF16],
            w_latent_to_v: pl.Tensor[[NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM], pl.BF16],
            kv_cache: pl.Tensor[[CACHE_ROWS, KV_LORA_RANK], pl.BF16],
            pe_cache: pl.Tensor[[CACHE_ROWS, QK_ROPE_HEAD_DIM], pl.BF16],
            k_cache_idx: pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.BF16],
            dispatch_buf: pl.Tensor[[EP_NODES, BATCH, ATTN_OUT], pl.BF16],
        ) -> pl.Tensor[[EP_NODES, BATCH, ATTN_OUT], pl.BF16]:
            # ===== scope1: MLA front path (RMSNorm + Q/KV projection + RoPE + cache writeback) =====
            # Outputs:
            # - kv_cache: normalized KV latent cache row for the current decode token
            # - pe_cache: rotated rope cache row for the current decode token
            # Internal bridge:
            # - q_proj: rotated decode query heads reused by scope4

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
            q_proj = pl.create_tensor([BATCH, NUM_HEADS * QK_HEAD_DIM], dtype=pl.BF16)
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
                    q_proj = pl.assemble(q_proj, pl.cast(q_out_acc, target_type=pl.BF16), [0, q0])

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

            # Stage 1.7: Apply RoPE on q_proj in-place.
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_rope"):
                    ctx_len = pl.read(seq_lens, [b])
                    cos_lo = rope_cos[ctx_len - 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    cos_hi = rope_cos[ctx_len - 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    sin_lo = rope_sin[ctx_len - 1, 0 : QK_ROPE_HEAD_DIM // 2]
                    sin_hi = rope_sin[ctx_len - 1, QK_ROPE_HEAD_DIM // 2 : QK_ROPE_HEAD_DIM]
                    for q_col in pl.range(QK_NOPE_HEAD_DIM, NUM_HEADS * QK_HEAD_DIM, QK_HEAD_DIM):
                        q_lo = pl.cast(q_proj[b, q_col : q_col + QK_ROPE_HEAD_DIM // 2], target_type=pl.FP32)
                        q_hi = pl.cast(q_proj[b, q_col + QK_ROPE_HEAD_DIM // 2 : q_col + QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                        q_rot_lo = pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo))
                        q_rot_hi = pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi))
                        q_proj = pl.assemble(q_proj, pl.cast(q_rot_lo, target_type=pl.BF16), [b, q_col])
                        q_proj = pl.assemble(q_proj, pl.cast(q_rot_hi, target_type=pl.BF16), [b, q_col + QK_ROPE_HEAD_DIM // 2])

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
            # - q_idx_out: BF16 aggregated index query vector for stage3
            # - k_cache_idx: BF16 index key cache row for the current decode token
            #
            # Scope123 switches to an INT8 index path after the Hadamard stage.
            # Scope1234 keeps the BF16 index contract because this standalone file
            # still validates and exports `k_cache_idx` in BF16, and scope3 keeps the
            # matching exact BF16 score path. The staging and comments below stay as
            # close to scope123 as the scope1234 data contract safely allows.

            # Stage 2.1: q_idx_full = wq_b_idx(qr_out).
            q_idx_full = pl.create_tensor([BATCH, INDEX_Q_OUT], dtype=pl.BF16)
            for q0 in pl.parallel(0, INDEX_Q_OUT, IDX_OUT_CHUNK):
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
            k_idx = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            for k1 in pl.parallel(0, INDEX_HEAD_DIM, KIDX_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_proj"):
                    s2_x_init = hidden_states[:, 0 : K_CHUNK]
                    wk_init = wk_idx[0 : K_CHUNK, k1 : k1 + KIDX_OUT_CHUNK]
                    s2_k_acc = pl.matmul(s2_x_init, wk_init, out_dtype=pl.FP32)
                    for k0 in pl.range(K_CHUNK, HIDDEN, K_CHUNK):
                        s2_x_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                        wk_chunk = wk_idx[k0 : k0 + K_CHUNK, k1 : k1 + KIDX_OUT_CHUNK]
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
                    for q_col in pl.range(0, INDEX_Q_OUT, INDEX_HEAD_DIM):
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

            # Stage 2.5: Apply Hadamard transform (full matrix multiplication).
            # For q_idx_full [B, H*D], reshape conceptually to [B, H, D] and apply Hadamard per head.
            # For k_idx [B, D], apply hadamard_k directly.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_q_hadamard"):
                for h in pl.parallel(INDEX_HEADS):
                    h_offset = h * INDEX_HEAD_DIM
                    for n0 in pl.range(0, INDEX_HEAD_DIM, IDX_OUT_CHUNK):
                        hadamard_q_acc = pl.full([BATCH, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for k0 in pl.range(0, INDEX_HEAD_DIM, K_CHUNK):
                            q_tile = q_idx_full[:, h_offset + k0 : h_offset + k0 + K_CHUNK]
                            hadamard_q_tile = hadamard_q[k0 : k0 + K_CHUNK, n0 : n0 + IDX_OUT_CHUNK]
                            q_h_tile = pl.matmul(q_tile, hadamard_q_tile, out_dtype=pl.FP32)
                            hadamard_q_acc = pl.add(hadamard_q_acc, q_h_tile)
                        q_idx_full = pl.assemble(q_idx_full, pl.cast(hadamard_q_acc, target_type=pl.BF16), [0, h_offset + n0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_hadamard"):
                for n0 in pl.parallel(0, INDEX_HEAD_DIM, IDX_OUT_CHUNK):
                    hadamard_k_acc = pl.full([BATCH, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for k0 in pl.range(0, INDEX_HEAD_DIM, K_CHUNK):
                        k_tile = k_idx[:, k0 : k0 + K_CHUNK]
                        hadamard_k_tile = hadamard_k[k0 : k0 + K_CHUNK, n0 : n0 + IDX_OUT_CHUNK]
                        k_h_tile = pl.matmul(k_tile, hadamard_k_tile, out_dtype=pl.FP32)
                        hadamard_k_acc = pl.add(hadamard_k_acc, k_h_tile)
                    k_idx = pl.assemble(k_idx, pl.cast(hadamard_k_acc, target_type=pl.BF16), [0, n0])

            # Stage 2.6: weights = weights_proj(hidden_states) * n_heads^-0.5 * head_dim^-0.5.
            weights_proj_bf16 = pl.create_tensor([HIDDEN, INDEX_HEADS], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_proj_bf16"):
                for k0 in pl.range(0, HIDDEN, K_CHUNK):
                    wp_tile = pl.cast(weights_proj[k0 : k0 + K_CHUNK, :], target_type=pl.BF16)
                    weights_proj_bf16 = pl.assemble(weights_proj_bf16, wp_tile, [k0, 0])

            weights = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)
            for w0 in pl.parallel(0, INDEX_HEADS, WEIGHTS_OUT_CHUNK):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_matmul"):
                    s2_x_init = hidden_states[:, 0 : K_CHUNK]
                    wp_init = weights_proj_bf16[0 : K_CHUNK, w0 : w0 + WEIGHTS_OUT_CHUNK]
                    s2_w_acc = pl.matmul(s2_x_init, wp_init, out_dtype=pl.FP32)
                    for k0 in pl.range(K_CHUNK, HIDDEN, K_CHUNK):
                        s2_x_chunk = hidden_states[:, k0 : k0 + K_CHUNK]
                        wp_chunk = weights_proj_bf16[k0 : k0 + K_CHUNK, w0 : w0 + WEIGHTS_OUT_CHUNK]
                        s2_w_acc = pl.matmul_acc(s2_w_acc, s2_x_chunk, wp_chunk)
                    weights = pl.assemble(weights, s2_w_acc, [0, w0])

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_weights_scale"):
                weights = pl.mul(weights, WEIGHT_SCALE)

            # Stage 2.7: Write current-token k_idx into BF16 cache form.
            for b in pl.parallel(BATCH):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s2_k_idx_cache_write"):
                    pos = pl.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ + pos
                    k_cache_idx = pl.assemble(k_cache_idx, k_idx[b : b + 1, :], [cache_row, 0])

            # Stage 2.8: Reduce q_idx_full across heads with weights to get q_idx_out.
            q_idx_out = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.auto_chunk], name_hint="s2_q_reduce"):
                s2_weights_t = pl.transpose(weights, axis1=0, axis2=1)
                for d0 in pl.parallel(0, INDEX_HEAD_DIM, QREDUCE_OUT_CHUNK):
                    s2_q_idx_acc = pl.full([BATCH, QREDUCE_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                    for h in pl.range(INDEX_HEADS):
                        s2_q_h = pl.cast(
                            pl.slice(q_idx_full, [BATCH, QREDUCE_OUT_CHUNK], [0, h * INDEX_HEAD_DIM + d0]),
                            target_type=pl.FP32,
                        )
                        s2_w_h_t = pl.slice(s2_weights_t, [1, BATCH], [h, 0])
                        s2_w_h = pl.reshape(s2_w_h_t, [BATCH, 1])
                        s2_q_idx_acc = pl.add(s2_q_idx_acc, pl.row_expand_mul(s2_q_h, s2_w_h))
                    q_idx_out = pl.assemble(q_idx_out, pl.cast(s2_q_idx_acc, target_type=pl.BF16), [0, d0])

            # ===== scope3: index score + topk =====
            # Inputs: q_idx_out, k_cache_idx
            # Internal bridge:
            # - topk_idx: top-k positions per batch for scope4
            #
            # Scope123's INT8 path computes quantized per-head scores and topk values.
            # Scope1234 keeps the exact BF16 `q_idx_out x k_cache_idx` score path so the
            # topk indices stay aligned with the exported BF16 cache contract.

            topk_idx = pl.create_tensor([BATCH, INDEX_TOPK], dtype=pl.INT32)
            scores = pl.create_tensor([BATCH, SORT_LEN], dtype=pl.FP32)
            s3_q_padded = pl.create_tensor([BATCH * Q_PAD, INDEX_HEAD_DIM], dtype=pl.BF16)

            for b in pl.parallel(BATCH):
                s3_score_tiles = pl.create_tensor([MAX_SEQ_BLOCKS * Q_PAD, SEQ_TILE], dtype=pl.FP32)

                # Stage 3.0: Pad q_idx_out and pre-fill scores.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_init"):
                    s3_q_row = q_idx_out[b : b + 1, 0 : INDEX_HEAD_DIM]
                    s3_q_padded = pl.assemble(s3_q_padded, s3_q_row, [b * Q_PAD, 0])
                    s3_q_zero_pad = pl.cast(pl.full([Q_PAD - Q_VALID, INDEX_HEAD_DIM], dtype=pl.FP32, value=0.0), target_type=pl.BF16)
                    s3_q_padded = pl.assemble(s3_q_padded, s3_q_zero_pad, [b * Q_PAD + Q_VALID, 0])
                    s3_neg_inf_row = pl.full([1, SORT_LEN], dtype=pl.FP32, value=FP32_NEG_INF)
                    scores = pl.assemble(scores, s3_neg_inf_row, [b, 0])

                s3_ctx_len = pl.read(seq_lens, [b])
                s3_ctx_blocks = (s3_ctx_len + SEQ_TILE - 1) // SEQ_TILE

                # Stage 3.1: Compute tiled BF16 qk logits.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_score_tile"):
                    for sb in pl.range(s3_ctx_blocks):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * MAX_SEQ + s0
                        s3_q_tile = s3_q_padded[b * Q_PAD : b * Q_PAD + Q_PAD, 0 : INDEX_HEAD_DIM]
                        s3_k_tile = k_cache_idx[cache_row0 : cache_row0 + SEQ_TILE, 0 : INDEX_HEAD_DIM]
                        s3_score_tile = pl.matmul(s3_q_tile, s3_k_tile, b_trans=True, out_dtype=pl.FP32)
                        s3_score_tiles = pl.assemble(s3_score_tiles, s3_score_tile, [sb * Q_PAD, 0])

                # Stage 3.2: Write valid score tiles into the global score row.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_score_write"):
                    for sb in pl.range(s3_ctx_blocks):
                        s0 = sb * SEQ_TILE
                        s3_valid_len = pl.min(SEQ_TILE, s3_ctx_len - s0)
                        score_row0 = sb * Q_PAD
                        s3_score_valid = pl.slice(s3_score_tiles, [1, SEQ_TILE], [score_row0, 0], valid_shape=[1, s3_valid_len])
                        scores = pl.assemble(scores, s3_score_valid, [b, s0])

                s3_sorted_gm = pl.create_tensor([1, 2 * SORT_LEN], dtype=pl.FP32)

                # Stage 3.3: Run sort32 + mrgsort.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_sort"):
                    s3_score_row = scores[b, :]
                    idx_init = pl.tensor.arange(0, [1, SORT_LEN], dtype=pl.UINT32)
                    s3_sorted_t = pl.tensor.sort32(s3_score_row, idx_init)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=64)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=256)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=1024)
                    s3_sorted_t = pl.tensor.mrgsort(s3_sorted_t, block_len=4096)
                    s3_sorted_gm = pl.assemble(s3_sorted_gm, s3_sorted_t, [0, 0])

                s3_raw_idx_local = pl.create_tensor([1, INDEX_TOPK], dtype=pl.INT32)

                # Stage 3.4: Split top-k values and raw indices, then pad invalid tail slots.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s3_topk_extract"):
                    s3_topk_pairs = s3_sorted_gm[:, 0 : 2 * INDEX_TOPK]
                    s3_topk_i_raw = pl.tensor.gather(s3_topk_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    s3_raw_idx_local = pl.assemble(s3_raw_idx_local, s3_topk_i_raw, [0, 0])
                    s3_valid_topk = pl.min(INDEX_TOPK, s3_ctx_len)
                    s3_idx_valid = pl.slice(s3_raw_idx_local, [1, INDEX_TOPK], [0, 0], valid_shape=[1, s3_valid_topk])
                    s3_idx_padded = pl.fillpad(s3_idx_valid, pad_value=pl.PadValue.min)
                    topk_idx = pl.assemble(topk_idx, s3_idx_padded, [b, 0])

            # ===== scope4: sparse attention with online softmax + latent-to-V projection =====
            # Inputs: q_proj, kv_cache, pe_cache, topk_idx
            # Output: dispatch_buf (cross-node dispatch buffer)
            attn_front = pl.create_tensor([BATCH, ATTN_OUT], dtype=pl.BF16)
            topk_idx_flat = pl.reshape(topk_idx, [BATCH * INDEX_TOPK])

            for b in pl.parallel(BATCH):
                attn_row = pl.create_tensor([1, ATTN_OUT], dtype=pl.FP32)
                sparse_k = pl.min(INDEX_TOPK, pl.read(seq_lens, [b]))
                topk_base = b * INDEX_TOPK

                for h in pl.parallel(NUM_HEADS):
                    q_col = h * QK_HEAD_DIM
                    v_col = h * V_HEAD_DIM

                    # Stage 4.1: Load q_pe and project q_nope into latent space.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_q_pe_load"):
                        q_pe = pl.cast(q_proj[b, q_col + QK_NOPE_HEAD_DIM : q_col + QK_HEAD_DIM], target_type=pl.FP32)
                        q_pe_batch = pl.col_expand(
                            pl.full([MATMUL_ROW_PAD, QK_ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
                            q_pe,
                        )
                        q_nope_padded = pl.cast(
                            pl.full([MATMUL_ROW_PAD, QK_NOPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
                            target_type=pl.BF16,
                        )
                        q_nope_padded = pl.col_expand(q_nope_padded, q_proj[b, q_col : q_col + QK_NOPE_HEAD_DIM])
                        q_nope_latent_batch = pl.full([MATMUL_ROW_PAD, KV_LORA_RANK], dtype=pl.FP32, value=0.0)

                    # Stage 4.2: Project q_nope to latent space chunk-by-chunk.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_q_nope_latent_proj"):
                        for q0 in pl.range(0, KV_LORA_RANK, Q_LATENT_CHUNK):
                            w_qn_h = pl.reshape(
                                w_q_nope_to_latent[h : h + 1, 0 : QK_NOPE_HEAD_DIM, q0 : q0 + Q_LATENT_CHUNK],
                                [QK_NOPE_HEAD_DIM, Q_LATENT_CHUNK],
                            )
                            q_nope_latent_part = pl.matmul(q_nope_padded, w_qn_h, out_dtype=pl.FP32)
                            q_nope_latent_batch = pl.assemble(q_nope_latent_batch, q_nope_latent_part, [0, q0])

                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_softmax_init"):
                        # Stage 4.3: Initialize online softmax state with the first topk entry.
                        topk_pos0 = pl.read(topk_idx_flat, [topk_base])
                        cache_s0 = b * MAX_SEQ + topk_pos0
                        kv_s0 = pl.cast(kv_cache[cache_s0 : cache_s0 + 1, 0 : KV_LORA_RANK], target_type=pl.FP32)
                        pe_s0 = pl.cast(pe_cache[cache_s0 : cache_s0 + 1, 0 : QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                        oi = pl.col_expand(pl.full([MATMUL_ROW_PAD, KV_LORA_RANK], dtype=pl.FP32, value=0.0), kv_s0)
                        pe_batch0 = pl.col_expand(pl.full([MATMUL_ROW_PAD, QK_ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0), pe_s0)
                        score_nope0 = pl.row_sum(pl.mul(q_nope_latent_batch, oi))
                        score_pe0 = pl.row_sum(pl.mul(q_pe_batch, pe_batch0))
                        mi = pl.mul(pl.add(score_nope0, score_pe0), ATTN_SCALE)
                        li = pl.exp(pl.sub(mi, mi))

                    # Stage 4.4: Online softmax accumulation over the remaining topk entries.
                    for kk in pl.range(1, sparse_k):
                        with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_softmax_accum"):
                            topk_pos = pl.read(topk_idx_flat, [topk_base + kk])
                            cache_s = b * MAX_SEQ + topk_pos
                            kv_s = pl.cast(kv_cache[cache_s : cache_s + 1, 0 : KV_LORA_RANK], target_type=pl.FP32)
                            pe_s = pl.cast(pe_cache[cache_s : cache_s + 1, 0 : QK_ROPE_HEAD_DIM], target_type=pl.FP32)
                            kv_batch = pl.col_expand(pl.full([MATMUL_ROW_PAD, KV_LORA_RANK], dtype=pl.FP32, value=0.0), kv_s)
                            pe_batch = pl.col_expand(pl.full([MATMUL_ROW_PAD, QK_ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0), pe_s)
                            score_nope = pl.row_sum(pl.mul(q_nope_latent_batch, kv_batch))
                            score_pe = pl.row_sum(pl.mul(q_pe_batch, pe_batch))
                            cur_mi = pl.mul(pl.add(score_nope, score_pe), ATTN_SCALE)
                            mi_new = pl.maximum(mi, cur_mi)
                            alpha = pl.exp(pl.sub(mi, mi_new))
                            beta = pl.exp(pl.sub(cur_mi, mi_new))
                            li = pl.add(pl.mul(alpha, li), beta)
                            oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(kv_batch, beta))
                            mi = mi_new

                    # Stage 4.5: Compute the latent context vector.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_latent_context"):
                        ctx_latent_batch = pl.row_expand_div(oi, li)

                    # Stage 4.6: Project latent context back to V chunks.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_v_proj"):
                        ctx_v_batch = pl.full([MATMUL_ROW_PAD, V_HEAD_DIM], dtype=pl.FP32, value=0.0)
                        for v0 in pl.range(0, V_HEAD_DIM, V_OUT_CHUNK):
                            wv_tile = pl.reshape(
                                w_latent_to_v[h : h + 1, 0 : KV_LORA_RANK, v0 : v0 + V_OUT_CHUNK],
                                [KV_LORA_RANK, V_OUT_CHUNK],
                            )
                            v_part_batch = pl.matmul(pl.cast(ctx_latent_batch, target_type=pl.BF16), wv_tile, out_dtype=pl.FP32)
                            ctx_v_batch = pl.assemble(ctx_v_batch, v_part_batch, [0, v0])

                    # Stage 4.7: Extract the leading V row and assemble it into attn_row.
                    with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_v_assemble"):
                        ctx_v = ctx_v_batch[0 : 1, 0 : V_HEAD_DIM]
                        attn_row = pl.assemble(attn_row, ctx_v, [0, v_col])

                # Stage 4.8: Cast and stash the finished attention row for this batch.
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_attn_row_write"):
                    attn_front = pl.assemble(attn_front, pl.cast(attn_row, target_type=pl.BF16), [b, 0])

            # Stage 4.9: Route finished attention rows into the cross-node dispatch buffer.
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="s4_dispatch"):
                layer_id = pl.read(layer_id_t, [0])
                for b in pl.parallel(BATCH):
                    target_node = (b + layer_id) % EP_NODES
                    token_row = attn_front[b : b + 1, 0 : ATTN_OUT]
                    dispatch_buf = pl.assemble(dispatch_buf, token_row, [target_node, b, 0])

            return dispatch_buf

    return DeepSeekV32DecodeFrontScope1234


def golden_decode_front_scope1234(tensors):
    import torch  # type: ignore[import]

    hidden_states = tensors["hidden_states"].float()
    input_rms_weight = tensors["input_rms_weight"].float()
    q_norm_weight = tensors["q_norm_weight"].float()
    kv_norm_weight = tensors["kv_norm_weight"].float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"].float()
    wkv_a = tensors["wkv_a"].float()
    seq_lens = tensors["seq_lens"]
    if torch.any(seq_lens <= 0):
        raise ValueError("deepseek_v3_2_decode_front_scope1234 expects seq_lens >= 1")
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()

    wq_b_idx = tensors["wq_b_idx"].float()
    wk_idx = tensors["wk_idx"].float()
    k_norm_weight = tensors["k_norm_weight"].float()
    k_norm_bias = tensors["k_norm_bias"].float()
    weights_proj = tensors["weights_proj"].float()

    hadamard_q = tensors["hadamard_q"].float()
    hadamard_k = tensors["hadamard_k"].float()

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

    # Apply Hadamard transform (full matrix multiplication) instead of scalar scale.
    # Query Hadamard: q_idx_full [B, H*D] -> reshape to [B, H, D] -> matmul per head with hadamard_q[D, D]
    q_hadamard = torch.einsum("bhd,dk->bhk", q_view, hadamard_q)
    q_idx_full = q_hadamard.reshape(BATCH, INDEX_HEADS * INDEX_HEAD_DIM).to(torch.bfloat16).float()

    # Key Hadamard: k_idx [B, D] -> matmul with hadamard_k[D, D]
    k_idx = (k_idx @ hadamard_k).to(torch.bfloat16).float()

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

    _, idx = torch.topk(scores, INDEX_TOPK, dim=1, largest=True, sorted=True)

    idx = idx.to(torch.int32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        valid_topk = min(INDEX_TOPK, ctx_len)
        idx[b, valid_topk:] = torch.iinfo(torch.int32).min

    # ===== Scope 4: sparse attention computation =====
    layer_id = int(tensors["layer_id_t"][0].item())
    w_q_nope_to_latent = tensors["w_q_nope_to_latent"].float()
    w_latent_to_v = tensors["w_latent_to_v"].float()
    dispatch_buf = tensors["dispatch_buf"]
    attn_scale = 1.0 / (QK_HEAD_DIM ** 0.5)

    attn_front = torch.zeros(BATCH, NUM_HEADS * V_HEAD_DIM, dtype=torch.float32)
    dispatch_buf.zero_()

    for b in range(BATCH):
        sparse_k = min(INDEX_TOPK, int(seq_lens[b].item()))
        for h in range(NUM_HEADS):
            q_nope = q_proj_view[b : b + 1, h, :QK_NOPE_HEAD_DIM]
            q_pe = q_proj_view[b : b + 1, h, QK_NOPE_HEAD_DIM:]
            q_nope_latent = q_nope @ w_q_nope_to_latent[h]

            oi = torch.zeros(1, KV_LORA_RANK, dtype=torch.float32)
            li = torch.zeros(1, 1, dtype=torch.float32)
            mi = torch.zeros(1, 1, dtype=torch.float32)

            for kk in range(sparse_k):
                topk_pos = int(idx[b, kk].item())
                if topk_pos < 0:
                    continue
                cache_s = b * MAX_SEQ + topk_pos
                kv_s = kv_cache[cache_s : cache_s + 1].float()
                pe_s = pe_cache[cache_s : cache_s + 1].float()
                score_nope = (q_nope_latent * kv_s).sum(dim=-1, keepdim=True)
                score_pe = (q_pe * pe_s).sum(dim=-1, keepdim=True)
                cur_mi = (score_nope + score_pe) * attn_scale
                cur_li = torch.ones(1, 1, dtype=torch.float32)
                oi_tmp = kv_s * cur_li
                if kk == 0:
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

            ctx_latent = oi / li.clamp_min(1e-30)
            ctx_v = ctx_latent @ w_latent_to_v[h]
            v_col = h * V_HEAD_DIM
            attn_front[b, v_col : v_col + V_HEAD_DIM] = ctx_v.squeeze(0)

    for b in range(BATCH):
        target_node = (b + layer_id) % EP_NODES
        dispatch_buf[target_node, b].copy_(attn_front[b].to(torch.bfloat16))


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import TensorSpec

    qk_head_dim = QK_HEAD_DIM
    kv_a_out = KV_A_OUT
    cache_rows = CACHE_ROWS
    index_q_out = INDEX_HEADS * INDEX_HEAD_DIM
    attn_out = ATTN_OUT
    seq_lens_data = torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)
    layer_id_data = torch.tensor([0], dtype=torch.int32)

    def init_hidden_states():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_rms_weight():
        return torch.rand(1, HIDDEN) - 0.5

    def init_q_norm_weight():
        return torch.rand(1, Q_LORA_RANK) - 0.5

    def init_wq_a():
        return (torch.rand(HIDDEN, Q_LORA_RANK) - 0.5) / HIDDEN ** 0.5

    def init_wq_b():
        return (torch.rand(Q_LORA_RANK, NUM_HEADS * qk_head_dim) - 0.5) / Q_LORA_RANK ** 0.5

    def init_wkv_a():
        return (torch.rand(HIDDEN, kv_a_out) - 0.5) / HIDDEN ** 0.5

    def init_kv_norm_weight():
        return torch.rand(1, KV_LORA_RANK) - 0.5

    def init_wq_b_idx():
        return (torch.rand(Q_LORA_RANK, index_q_out) - 0.5) / Q_LORA_RANK ** 0.5

    def init_wk_idx():
        return (torch.rand(HIDDEN, INDEX_HEAD_DIM) - 0.5) / HIDDEN ** 0.5

    def init_k_norm_weight():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_k_norm_bias():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_weights_proj():
        return (torch.rand(HIDDEN, INDEX_HEADS) - 0.5) / HIDDEN ** 0.5

    def init_rope():
        return torch.rand(MAX_SEQ, QK_ROPE_HEAD_DIM) - 0.5

    def init_cache_kv():
        return torch.zeros(cache_rows, KV_LORA_RANK)

    def init_cache_pe():
        return torch.zeros(cache_rows, QK_ROPE_HEAD_DIM)

    def init_k_cache_idx():
        return torch.rand(cache_rows, INDEX_HEAD_DIM) - 0.5

    def init_hadamard_q():
        return (torch.rand(INDEX_HEAD_DIM, INDEX_HEAD_DIM, dtype=torch.float32) - 0.5) / (INDEX_HEAD_DIM ** 0.5)

    def init_hadamard_k():
        return (torch.rand(INDEX_HEAD_DIM, INDEX_HEAD_DIM, dtype=torch.float32) - 0.5) / (INDEX_HEAD_DIM ** 0.5)

    def init_w_q_nope_to_latent():
        return (
            (torch.rand(NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK, dtype=torch.float32) - 0.5)
            / (QK_NOPE_HEAD_DIM ** 0.5)
        )

    def init_w_latent_to_v():
        return (
            (torch.rand(NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM, dtype=torch.float32) - 0.5)
            / (KV_LORA_RANK ** 0.5)
        )

    def init_dispatch_buf():
        return torch.zeros(EP_NODES, BATCH, attn_out, dtype=torch.bfloat16)

    return [
        TensorSpec("hidden_states", [BATCH, HIDDEN], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("input_rms_weight", [1, HIDDEN], torch.float32, init_value=init_rms_weight),
        TensorSpec("q_norm_weight", [1, Q_LORA_RANK], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("kv_norm_weight", [1, KV_LORA_RANK], torch.float32, init_value=init_kv_norm_weight),
        TensorSpec("wq_a", [HIDDEN, Q_LORA_RANK], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA_RANK, NUM_HEADS * qk_head_dim], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv_a", [HIDDEN, kv_a_out], torch.bfloat16, init_value=init_wkv_a),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=seq_lens_data),
        TensorSpec("rope_cos", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope),
        TensorSpec("rope_sin", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope),
        TensorSpec("wq_b_idx", [Q_LORA_RANK, index_q_out], torch.bfloat16, init_value=init_wq_b_idx),
        TensorSpec("wk_idx", [HIDDEN, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_wk_idx),
        TensorSpec("weights_proj", [HIDDEN, INDEX_HEADS], torch.float32, init_value=init_weights_proj),
        TensorSpec("k_norm_weight", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("k_norm_bias", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_bias),
        TensorSpec("hadamard_q", [INDEX_HEAD_DIM, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard_q),
        TensorSpec("hadamard_k", [INDEX_HEAD_DIM, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_hadamard_k),
        TensorSpec("layer_id_t", [1], torch.int32, init_value=layer_id_data),
        TensorSpec(
            "w_q_nope_to_latent",
            [NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK],
            torch.bfloat16,
            init_value=init_w_q_nope_to_latent,
        ),
        TensorSpec("w_latent_to_v", [NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM], torch.bfloat16, init_value=init_w_latent_to_v),
        TensorSpec("kv_cache", [cache_rows, KV_LORA_RANK], torch.bfloat16, init_value=init_cache_kv, is_output=True),
        TensorSpec("pe_cache", [cache_rows, QK_ROPE_HEAD_DIM], torch.bfloat16, init_value=init_cache_pe, is_output=True),
        TensorSpec("k_cache_idx", [cache_rows, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_k_cache_idx, is_output=True),
        TensorSpec("dispatch_buf", [EP_NODES, BATCH, attn_out], torch.bfloat16, init_value=init_dispatch_buf, is_output=True),
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
        program=build_deepseek_v3_2_decode_front_scope1234_program(),
        specs=build_tensor_specs(),
        golden_fn=golden_decode_front_scope1234,
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
