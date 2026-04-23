# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
DeepSeek V3.2-EXP single-layer decode FRONT, organised as 4 scopes.

  Scope 1: qkv proj + qkv rope
           input RMSNorm, wq_a / q_norm / wq_b, wkv_a, q_pe RoPE,
           kv_norm + k_pe RoPE, write kv_cache & pe_cache.

  Scope 2: indexer proj + indexer rope
           wq_b_idx (qr -> 64-head index query), wk_idx + LayerNorm,
           non-interleaved RoPE on q_pe / k_pe halves,
           weights_proj (per-head head weights),
           TODO(hadamard_transform) + TODO(fp8 quant) placeholders,
           weighted-sum aggregation -> q_idx [B, INDEX_HEAD_DIM],
           write k_cache_idx.

  Scope 3: score + topk
           q_idx x k_cache_idx tiled matmul (matches scope2b),
           TODO(topk) placeholder producing topk_idx [B, INDEX_TOPK].

  Scope 4: post topk
           sparse MQA over topk positions in (kv_cache, pe_cache),
           project latent -> v, dispatch write to cross-node buffer.
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
ATTN_OUT = NUM_HEADS * V_HEAD_DIM

# Indexer (per ds32exp_official ModelArgs).
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048

EP_NODES = 128

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
ATTN_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
INDEX_SOFTMAX_SCALE = 1.0 / (INDEX_HEAD_DIM ** 0.5)
INDEX_HEADS_INV_SQRT = INDEX_HEADS ** -0.5
HADAMARD_SCALE = INDEX_HEAD_DIM ** -0.5

KV_A_OUT = KV_LORA_RANK + QK_ROPE_HEAD_DIM
CACHE_ROWS = BATCH * MAX_SEQ
HALF_ROPE = QK_ROPE_HEAD_DIM // 2
HALF_INDEX_ROPE = QK_ROPE_HEAD_DIM // 2  # indexer reuses the same rope dim

# Tiling / chunking.
K_CHUNK = 512
LORA_CHUNK = 128
Q_OUT_CHUNK = 512
KV_OUT_CHUNK = 128
V_OUT_CHUNK = 64
IDX_OUT_CHUNK = 128
WEIGHTS_OUT_CHUNK = 64
BATCH_TILE = 4
SEQ_TILE = 64

HIDDEN_BLOCKS = HIDDEN // K_CHUNK
QR_BLOCKS = Q_LORA_RANK // LORA_CHUNK
Q_OUT_BLOCKS = (NUM_HEADS * QK_HEAD_DIM) // Q_OUT_CHUNK
KV_A_BLOCKS = KV_A_OUT // KV_OUT_CHUNK
IDX_OUT_BLOCKS = (INDEX_HEADS * INDEX_HEAD_DIM) // IDX_OUT_CHUNK
WK_OUT_BLOCKS = INDEX_HEAD_DIM // KV_OUT_CHUNK
V_OUT_BLOCKS = V_HEAD_DIM // V_OUT_CHUNK
MAX_SEQ_BLOCKS = MAX_SEQ // SEQ_TILE


def build_ds32exp_program():
    @pl.program
    class Ds32Exp:
        @pl.function(type=pl.FunctionType.Opaque)
        def ds32exp_decode_front(
            self,
            hidden_states: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[BATCH], pl.INT32],
            layer_id_t: pl.Tensor[[1], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, QK_ROPE_HEAD_DIM], pl.FP32],
            kv_cache: pl.Tensor[[CACHE_ROWS, KV_LORA_RANK], pl.BF16],
            pe_cache: pl.Tensor[[CACHE_ROWS, QK_ROPE_HEAD_DIM], pl.BF16],
            k_cache_idx: pl.Tensor[[CACHE_ROWS, INDEX_HEAD_DIM], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            wq_a: pl.Tensor[[HIDDEN, Q_LORA_RANK], pl.BF16],
            q_norm_weight: pl.Tensor[[1, Q_LORA_RANK], pl.FP32],
            wq_b: pl.Tensor[[Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM], pl.BF16],
            wkv_a: pl.Tensor[[HIDDEN, KV_A_OUT], pl.BF16],
            kv_norm_weight: pl.Tensor[[1, KV_LORA_RANK], pl.FP32],
            wq_b_idx: pl.Tensor[[Q_LORA_RANK, INDEX_HEADS * INDEX_HEAD_DIM], pl.BF16],
            wk_idx: pl.Tensor[[HIDDEN, INDEX_HEAD_DIM], pl.BF16],
            k_norm_weight: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            k_norm_bias: pl.Tensor[[1, INDEX_HEAD_DIM], pl.FP32],
            weights_proj: pl.Tensor[[HIDDEN, INDEX_HEADS], pl.FP32],
            w_q_nope_to_latent: pl.Tensor[[NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK], pl.BF16],
            w_latent_to_v: pl.Tensor[[NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM], pl.BF16],
            dispatch_buf: pl.InOut[pl.Tensor[[EP_NODES, BATCH, ATTN_OUT], pl.BF16]],
        ) -> pl.Tensor[[EP_NODES, BATCH, ATTN_OUT], pl.BF16]:
            # Cross-scope intermediates.
            qr = pl.create_tensor([BATCH, Q_LORA_RANK], dtype=pl.BF16)
            q_proj = pl.create_tensor([BATCH, NUM_HEADS * QK_HEAD_DIM], dtype=pl.BF16)
            kv_a = pl.create_tensor([BATCH, KV_A_OUT], dtype=pl.BF16)

            # ── Scope 1: qkv proj + qkv rope ──
            # Stage 1.1: input RMSNorm (sq_sum -> rsqrt over HIDDEN).
            inv_rms = pl.create_tensor([BATCH, 1], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP):
                sq_sum = pl.full([BATCH, 1], dtype=pl.FP32, value=0.0)
                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.cast(
                        pl.slice(hidden_states, [BATCH, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS)))

            # Stage 1.2: qr = q_norm(wq_a(normed_x)).
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    inv_rms_tile = pl.slice(inv_rms, [BATCH_TILE, 1], [b0, 0])
                    for ob in pl.range(QR_BLOCKS):
                        q0 = ob * LORA_CHUNK
                        q_acc = pl.full([BATCH_TILE, LORA_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_tile = pl.cast(
                                pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                                target_type=pl.FP32,
                            )
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_tile, inv_rms_tile), gamma
                            )
                            wq_chunk = pl.slice(wq_a, [K_CHUNK, LORA_CHUNK], [k0, q0])
                            q_acc = pl.add(
                                q_acc,
                                pl.matmul(pl.cast(normed, target_type=pl.BF16), wq_chunk, out_dtype=pl.FP32),
                            )
                        q_gamma = pl.slice(q_norm_weight, [1, LORA_CHUNK], [0, q0])
                        qn = pl.col_expand_mul(q_acc, q_gamma)
                        qr = pl.assemble(qr, pl.cast(qn, target_type=pl.BF16), [b0, q0])

            # Stage 1.3: q_proj = qr @ wq_b.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    for ob in pl.range(Q_OUT_BLOCKS):
                        q0 = ob * Q_OUT_CHUNK
                        q_out_acc = pl.full([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(QR_BLOCKS):
                            k0 = kb * LORA_CHUNK
                            qr_chunk = pl.slice(qr, [BATCH_TILE, LORA_CHUNK], [b0, k0])
                            wq_b_chunk = pl.slice(wq_b, [LORA_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_out_acc = pl.add(
                                q_out_acc,
                                pl.matmul(qr_chunk, wq_b_chunk, out_dtype=pl.FP32),
                            )
                        q_proj = pl.assemble(
                            q_proj, pl.cast(q_out_acc, target_type=pl.BF16), [b0, q0]
                        )

            # Stage 1.4: kv_a = wkv_a(normed_x).
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    inv_rms_tile = pl.slice(inv_rms, [BATCH_TILE, 1], [b0, 0])
                    for ob in pl.range(KV_A_BLOCKS):
                        kv0 = ob * KV_OUT_CHUNK
                        kv_acc = pl.full([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_tile = pl.cast(
                                pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                                target_type=pl.FP32,
                            )
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(
                                pl.row_expand_mul(x_tile, inv_rms_tile), gamma
                            )
                            wkv_chunk = pl.slice(wkv_a, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            kv_acc = pl.add(
                                kv_acc,
                                pl.matmul(pl.cast(normed, target_type=pl.BF16), wkv_chunk, out_dtype=pl.FP32),
                            )
                        kv_a = pl.assemble(
                            kv_a, pl.cast(kv_acc, target_type=pl.BF16), [b0, kv0]
                        )

            # Stage 1.5: q_pe RoPE on every MLA head, k_pe RoPE on kv_a, kv_norm,
            # write kv_cache and pe_cache at row b*MAX_SEQ + (seq_lens[b]-1).
            # NOTE: official applies interleaved=True for MLA, but the existing
            # decode/prefill paths in this repo use the lo/hi half split form
            # (see deepseek_v3_2_decode_front.py:241-274). We follow the same
            # convention so the cached pe matches the in-tree consumers.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, BATCH, 1, chunk=BATCH):
                    ctx_len = pl.tensor.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cache_row = b * MAX_SEQ + pos

                    cos_lo = pl.slice(rope_cos, [1, HALF_ROPE], [pos, 0])
                    cos_hi = pl.slice(rope_cos, [1, HALF_ROPE], [pos, HALF_ROPE])
                    sin_lo = pl.slice(rope_sin, [1, HALF_ROPE], [pos, 0])
                    sin_hi = pl.slice(rope_sin, [1, HALF_ROPE], [pos, HALF_ROPE])

                    # MLA q_pe RoPE: rotate every head's pe half in-place.
                    for h in pl.range(NUM_HEADS):
                        q_col = h * QK_HEAD_DIM
                        q_lo = pl.cast(
                            pl.slice(q_proj, [1, HALF_ROPE], [b, q_col + QK_NOPE_HEAD_DIM]),
                            target_type=pl.FP32,
                        )
                        q_hi = pl.cast(
                            pl.slice(
                                q_proj, [1, HALF_ROPE],
                                [b, q_col + QK_NOPE_HEAD_DIM + HALF_ROPE],
                            ),
                            target_type=pl.FP32,
                        )
                        rot_lo = pl.sub(
                            pl.col_expand_mul(q_lo, cos_lo),
                            pl.col_expand_mul(q_hi, sin_lo),
                        )
                        rot_hi = pl.add(
                            pl.col_expand_mul(q_hi, cos_hi),
                            pl.col_expand_mul(q_lo, sin_hi),
                        )
                        q_proj = pl.assemble(
                            q_proj, pl.cast(rot_lo, target_type=pl.BF16),
                            [b, q_col + QK_NOPE_HEAD_DIM],
                        )
                        q_proj = pl.assemble(
                            q_proj, pl.cast(rot_hi, target_type=pl.BF16),
                            [b, q_col + QK_NOPE_HEAD_DIM + HALF_ROPE],
                        )

                    # kv_norm on the latent half then write kv_cache.
                    kv_row = pl.cast(
                        pl.slice(kv_a, [1, KV_LORA_RANK], [b, 0]), target_type=pl.FP32
                    )
                    kv_gamma = pl.slice(kv_norm_weight, [1, KV_LORA_RANK], [0, 0])
                    kv_normed = pl.col_expand_mul(kv_row, kv_gamma)
                    kv_cache = pl.assemble(
                        kv_cache, pl.cast(kv_normed, target_type=pl.BF16), [cache_row, 0]
                    )

                    # k_pe RoPE on the rope half then write pe_cache.
                    pe_lo = pl.cast(
                        pl.slice(kv_a, [1, HALF_ROPE], [b, KV_LORA_RANK]),
                        target_type=pl.FP32,
                    )
                    pe_hi = pl.cast(
                        pl.slice(kv_a, [1, HALF_ROPE], [b, KV_LORA_RANK + HALF_ROPE]),
                        target_type=pl.FP32,
                    )
                    pe_rot_lo = pl.sub(
                        pl.col_expand_mul(pe_lo, cos_lo),
                        pl.col_expand_mul(pe_hi, sin_lo),
                    )
                    pe_rot_hi = pl.add(
                        pl.col_expand_mul(pe_hi, cos_hi),
                        pl.col_expand_mul(pe_lo, sin_hi),
                    )
                    pe_cache = pl.assemble(
                        pe_cache, pl.cast(pe_rot_lo, target_type=pl.BF16), [cache_row, 0]
                    )
                    pe_cache = pl.assemble(
                        pe_cache, pl.cast(pe_rot_hi, target_type=pl.BF16),
                        [cache_row, HALF_ROPE],
                    )

            # ── Scope 2: indexer proj + indexer rope ──
            # Cross-stage indexer intermediates.
            q_idx_full = pl.create_tensor(
                [BATCH, INDEX_HEADS * INDEX_HEAD_DIM], dtype=pl.BF16
            )
            k_idx = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)
            weights = pl.create_tensor([BATCH, INDEX_HEADS], dtype=pl.FP32)
            q_idx = pl.create_tensor([BATCH, INDEX_HEAD_DIM], dtype=pl.BF16)

            # Stage 2.1: q_idx_full = qr @ wq_b_idx  -> [B, INDEX_HEADS * INDEX_HEAD_DIM].
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    for ob in pl.range(IDX_OUT_BLOCKS):
                        q0 = ob * IDX_OUT_CHUNK
                        idx_acc = pl.full([BATCH_TILE, IDX_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(QR_BLOCKS):
                            k0 = kb * LORA_CHUNK
                            qr_chunk = pl.slice(qr, [BATCH_TILE, LORA_CHUNK], [b0, k0])
                            wq_b_idx_chunk = pl.slice(
                                wq_b_idx, [LORA_CHUNK, IDX_OUT_CHUNK], [k0, q0]
                            )
                            idx_acc = pl.add(
                                idx_acc,
                                pl.matmul(qr_chunk, wq_b_idx_chunk, out_dtype=pl.FP32),
                            )
                        q_idx_full = pl.assemble(
                            q_idx_full, pl.cast(idx_acc, target_type=pl.BF16), [b0, q0]
                        )

            # Stage 2.2: k_idx_pre = hidden_states @ wk_idx -> [B, INDEX_HEAD_DIM] BF16.
            #            Then LayerNorm with k_norm_weight / k_norm_bias.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    for ob in pl.range(WK_OUT_BLOCKS):
                        kv0 = ob * KV_OUT_CHUNK
                        wk_acc = pl.full([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32, value=0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_tile = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
                            wk_chunk = pl.slice(wk_idx, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            wk_acc = pl.add(
                                wk_acc,
                                pl.matmul(x_tile, wk_chunk, out_dtype=pl.FP32),
                            )
                        k_idx = pl.assemble(
                            k_idx, pl.cast(wk_acc, target_type=pl.BF16), [b0, kv0]
                        )

            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                # LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta.
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    k_tile_fp32 = pl.cast(
                        pl.slice(k_idx, [BATCH_TILE, INDEX_HEAD_DIM], [b0, 0]),
                        target_type=pl.FP32,
                    )
                    mean = pl.mul(pl.row_sum(k_tile_fp32), 1.0 / INDEX_HEAD_DIM)
                    centered = pl.row_expand_sub(k_tile_fp32, mean)
                    var = pl.mul(
                        pl.row_sum(pl.mul(centered, centered)), 1.0 / INDEX_HEAD_DIM
                    )
                    inv_std = pl.recip(pl.sqrt(pl.add(var, EPS)))
                    normed = pl.row_expand_mul(centered, inv_std)
                    gamma = pl.slice(k_norm_weight, [1, INDEX_HEAD_DIM], [0, 0])
                    beta = pl.slice(k_norm_bias, [1, INDEX_HEAD_DIM], [0, 0])
                    y = pl.add(pl.col_expand_mul(normed, gamma), beta)
                    k_idx = pl.assemble(k_idx, pl.cast(y, target_type=pl.BF16), [b0, 0])

            # Stage 2.3: non-interleaved RoPE on q_pe (per index head) and k_pe.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, BATCH, 1, chunk=BATCH):
                    ctx_len = pl.tensor.read(seq_lens, [b])
                    pos = ctx_len - 1
                    cos_lo = pl.slice(rope_cos, [1, HALF_INDEX_ROPE], [pos, 0])
                    cos_hi = pl.slice(rope_cos, [1, HALF_INDEX_ROPE], [pos, HALF_INDEX_ROPE])
                    sin_lo = pl.slice(rope_sin, [1, HALF_INDEX_ROPE], [pos, 0])
                    sin_hi = pl.slice(rope_sin, [1, HALF_INDEX_ROPE], [pos, HALF_INDEX_ROPE])

                    # q_pe rotation per index head. Indexer uses interleaved=False
                    # (official line 464); our lo/hi-half implementation matches
                    # the existing ds_q0_rope.py convention.
                    for h in pl.range(INDEX_HEADS):
                        q_col = h * INDEX_HEAD_DIM
                        q_lo = pl.cast(
                            pl.slice(q_idx_full, [1, HALF_INDEX_ROPE], [b, q_col]),
                            target_type=pl.FP32,
                        )
                        q_hi = pl.cast(
                            pl.slice(
                                q_idx_full, [1, HALF_INDEX_ROPE], [b, q_col + HALF_INDEX_ROPE]
                            ),
                            target_type=pl.FP32,
                        )
                        rot_lo = pl.sub(
                            pl.col_expand_mul(q_lo, cos_lo),
                            pl.col_expand_mul(q_hi, sin_lo),
                        )
                        rot_hi = pl.add(
                            pl.col_expand_mul(q_hi, cos_hi),
                            pl.col_expand_mul(q_lo, sin_hi),
                        )
                        q_idx_full = pl.assemble(
                            q_idx_full, pl.cast(rot_lo, target_type=pl.BF16), [b, q_col]
                        )
                        q_idx_full = pl.assemble(
                            q_idx_full, pl.cast(rot_hi, target_type=pl.BF16),
                            [b, q_col + HALF_INDEX_ROPE],
                        )

                    # k_pe rotation (single head).
                    k_lo = pl.cast(
                        pl.slice(k_idx, [1, HALF_INDEX_ROPE], [b, 0]), target_type=pl.FP32
                    )
                    k_hi = pl.cast(
                        pl.slice(k_idx, [1, HALF_INDEX_ROPE], [b, HALF_INDEX_ROPE]),
                        target_type=pl.FP32,
                    )
                    k_rot_lo = pl.sub(
                        pl.col_expand_mul(k_lo, cos_lo),
                        pl.col_expand_mul(k_hi, sin_lo),
                    )
                    k_rot_hi = pl.add(
                        pl.col_expand_mul(k_hi, cos_hi),
                        pl.col_expand_mul(k_lo, sin_hi),
                    )
                    k_idx = pl.assemble(
                        k_idx, pl.cast(k_rot_lo, target_type=pl.BF16), [b, 0]
                    )
                    k_idx = pl.assemble(
                        k_idx, pl.cast(k_rot_hi, target_type=pl.BF16), [b, HALF_INDEX_ROPE]
                    )

            # Stage 2.4: TODO(hadamard_transform).
            # Official: rotate_activation(q), rotate_activation(k) using
            # fast_hadamard_transform with scale = INDEX_HEAD_DIM ** -0.5
            # (ds32exp_official.py:428-432). Hadamard is orthogonal/linear, so
            # the eventual scoring still scales by the same factor; here we
            # only keep the scaling and leave the orthogonal mixing as a TODO.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    q_tile = pl.cast(
                        pl.slice(
                            q_idx_full, [BATCH_TILE, INDEX_HEADS * INDEX_HEAD_DIM], [b0, 0]
                        ),
                        target_type=pl.FP32,
                    )
                    q_tile_scaled = pl.mul(q_tile, HADAMARD_SCALE)
                    q_idx_full = pl.assemble(
                        q_idx_full, pl.cast(q_tile_scaled, target_type=pl.BF16), [b0, 0]
                    )
                    k_tile = pl.cast(
                        pl.slice(k_idx, [BATCH_TILE, INDEX_HEAD_DIM], [b0, 0]),
                        target_type=pl.FP32,
                    )
                    k_tile_scaled = pl.mul(k_tile, HADAMARD_SCALE)
                    k_idx = pl.assemble(
                        k_idx, pl.cast(k_tile_scaled, target_type=pl.BF16), [b0, 0]
                    )

            # Stage 2.5: TODO(fp8 quant).
            # Official: q_fp8, q_scale = act_quant(q, block_size); same for k.
            # weights are then multiplied by q_scale (line 479). Placeholder
            # below performs a BF16 -> FP8E4M3FN -> BF16 round-trip so the
            # numerical loss matches in spirit, and skips the q_scale fold-in.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    q_tile = pl.slice(
                        q_idx_full, [BATCH_TILE, INDEX_HEADS * INDEX_HEAD_DIM], [b0, 0]
                    )
                    q_fp8 = pl.cast(q_tile, target_type=pl.FP8E4M3FN)
                    q_back = pl.cast(q_fp8, target_type=pl.BF16)
                    q_idx_full = pl.assemble(q_idx_full, q_back, [b0, 0])
                    k_tile = pl.slice(k_idx, [BATCH_TILE, INDEX_HEAD_DIM], [b0, 0])
                    k_fp8 = pl.cast(k_tile, target_type=pl.FP8E4M3FN)
                    k_back = pl.cast(k_fp8, target_type=pl.BF16)
                    k_idx = pl.assemble(k_idx, k_back, [b0, 0])

            # Stage 2.6: weights = (hidden_states.float() @ weights_proj)
            #            * INDEX_HEADS ** -0.5 * INDEX_SOFTMAX_SCALE.
            # TODO(fp8 quant): also multiply by q_scale once Stage 2.5 keeps it.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                w_scale = INDEX_HEADS_INV_SQRT * INDEX_SOFTMAX_SCALE
                for b0 in pl.parallel(0, BATCH, BATCH_TILE, chunk=BATCH // BATCH_TILE):
                    for ob in pl.range(INDEX_HEADS // WEIGHTS_OUT_CHUNK):
                        w0 = ob * WEIGHTS_OUT_CHUNK
                        w_acc = pl.full(
                            [BATCH_TILE, WEIGHTS_OUT_CHUNK], dtype=pl.FP32, value=0.0
                        )
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_tile = pl.cast(
                                pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0]),
                                target_type=pl.FP32,
                            )
                            wp_chunk = pl.slice(
                                weights_proj, [K_CHUNK, WEIGHTS_OUT_CHUNK], [k0, w0]
                            )
                            w_acc = pl.add(
                                w_acc,
                                pl.matmul(
                                    pl.cast(x_tile, target_type=pl.BF16),
                                    pl.cast(wp_chunk, target_type=pl.BF16),
                                    out_dtype=pl.FP32,
                                ),
                            )
                        weights = pl.assemble(
                            weights, pl.mul(w_acc, w_scale), [b0, w0]
                        )

            # Stage 2.7: aggregate q_idx[b] = sum_h weights[b,h] * q_idx_full[b, h*HD:(h+1)*HD].
            # See deepseek_v3_2_decode_front_scope2b.py header for the algebraic
            # reduction that lets scope3 score with a single [INDEX_HEAD_DIM] vector.
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, BATCH, 1, chunk=BATCH):
                    q_acc = pl.full([1, INDEX_HEAD_DIM], dtype=pl.FP32, value=0.0)
                    for h in pl.range(INDEX_HEADS):
                        q_h = pl.cast(
                            pl.slice(
                                q_idx_full, [1, INDEX_HEAD_DIM], [b, h * INDEX_HEAD_DIM]
                            ),
                            target_type=pl.FP32,
                        )
                        w_h = pl.slice(weights, [1, 1], [b, h])
                        q_acc = pl.add(q_acc, pl.col_expand_mul(q_h, w_h))
                    q_idx = pl.assemble(q_idx, pl.cast(q_acc, target_type=pl.BF16), [b, 0])

            # Stage 2.8: write k_cache_idx[b*MAX_SEQ + pos] = k_idx[b].
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, BATCH, 1, chunk=BATCH):
                    pos = pl.tensor.read(seq_lens, [b]) - 1
                    cache_row = b * MAX_SEQ + pos
                    k_row = pl.slice(k_idx, [1, INDEX_HEAD_DIM], [b, 0])
                    k_cache_idx = pl.assemble(k_cache_idx, k_row, [cache_row, 0])

            # ── Scope 3: score + topk ──
            # Stage 3.1: scoring follows deepseek_v3_2_decode_front_scope2b.py
            # (lines 68-95): per-batch tiled q_idx[b] x k_cache_idx[b, :], then
            # fillpad invalid tail to -inf so a downstream topk naturally drops it.
            scores = pl.create_tensor([BATCH, MAX_SEQ], dtype=pl.FP32)
            for b in pl.range(BATCH):
                ctx_len = pl.tensor.read(seq_lens, [b])
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                all_scores = pl.create_tensor([MAX_SEQ_BLOCKS, SEQ_TILE], dtype=pl.FP32)

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for sb in pl.parallel(ctx_blocks, chunk=MAX_SEQ_BLOCKS):
                        s0 = sb * SEQ_TILE
                        cache_row0 = b * MAX_SEQ + s0
                        q_b = pl.slice(q_idx, [1, INDEX_HEAD_DIM], [b, 0])
                        k_tile = pl.slice(
                            k_cache_idx, [SEQ_TILE, INDEX_HEAD_DIM], [cache_row0, 0]
                        )
                        score_tile = pl.matmul(q_b, k_tile, b_trans=True, out_dtype=pl.FP32)
                        all_scores = pl.assemble(all_scores, score_tile, [sb, 0])

                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    for sb in pl.parallel(ctx_blocks, chunk=MAX_SEQ_BLOCKS):
                        s0 = sb * SEQ_TILE
                        valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                        tile_valid = pl.slice(
                            all_scores, [1, SEQ_TILE], [sb, 0], valid_shape=[1, valid_len]
                        )
                        tile_padded = pl.fillpad(tile_valid, pad_value=pl.PadValue.min)
                        scores = pl.assemble(scores, tile_padded, [b, s0])

            # Stage 3.2: TODO(topk).
            # Real implementation should pick the INDEX_TOPK largest entries of
            # scores[b, :ctx_len] per batch. See the legacy monolithic
            # deepseek_v3_2_decode_front.py:278-436 for a B1/B2 two-stage
            # insertion-sort sketch, or replace with a dedicated topk kernel.
            # Placeholder: topk_idx is all-zero, which makes scope4 attend only
            # to position 0 in each batch row but keeps the pipeline runnable.
            topk_idx = pl.create_tensor([BATCH, INDEX_TOPK], dtype=pl.INT32)
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, BATCH, 1, chunk=BATCH):
                    zero_row = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=0)
                    topk_idx = pl.assemble(topk_idx, zero_row, [b, 0])

            # ── Scope 4: post topk (sparse MQA + dispatch) ──
            attn_front = pl.create_tensor([BATCH, ATTN_OUT], dtype=pl.FP32)
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b in pl.parallel(0, BATCH, 1, chunk=4):
                    ctx_len = pl.tensor.read(seq_lens, [b])
                    sparse_k = pl.min(INDEX_TOPK, ctx_len)
                    attn_row = pl.full([1, ATTN_OUT], dtype=pl.FP32, value=0.0)

                    for h in pl.parallel(0, NUM_HEADS, 1, chunk=8):
                        q_col = h * QK_HEAD_DIM
                        # q_pe was already RoPE-rotated in Scope 1, so we read it
                        # back as-is. q_nope is projected to the latent space
                        # via per-head w_q_nope_to_latent.
                        q_nope = pl.cast(
                            pl.slice(q_proj, [1, QK_NOPE_HEAD_DIM], [b, q_col]),
                            target_type=pl.FP32,
                        )
                        q_pe = pl.cast(
                            pl.slice(
                                q_proj, [1, QK_ROPE_HEAD_DIM], [b, q_col + QK_NOPE_HEAD_DIM]
                            ),
                            target_type=pl.FP32,
                        )
                        w_qn_h = pl.reshape(
                            pl.slice(
                                w_q_nope_to_latent,
                                [1, QK_NOPE_HEAD_DIM, KV_LORA_RANK],
                                [h, 0, 0],
                            ),
                            [QK_NOPE_HEAD_DIM, KV_LORA_RANK],
                        )
                        q_nope_latent = pl.matmul(
                            pl.cast(q_nope, target_type=pl.BF16), w_qn_h, out_dtype=pl.FP32
                        )

                        oi = pl.full([1, KV_LORA_RANK], dtype=pl.FP32, value=0.0)
                        li = pl.full([1, 1], dtype=pl.FP32, value=0.0)
                        mi = pl.full([1, 1], dtype=pl.FP32, value=0.0)

                        for kk in pl.range(sparse_k):
                            topk_pos = pl.tensor.read(topk_idx, [b, kk])
                            if topk_pos >= 0:
                                cache_s = b * MAX_SEQ + topk_pos
                                kv_s = pl.cast(
                                    pl.slice(kv_cache, [1, KV_LORA_RANK], [cache_s, 0]),
                                    target_type=pl.FP32,
                                )
                                pe_s = pl.cast(
                                    pl.slice(pe_cache, [1, QK_ROPE_HEAD_DIM], [cache_s, 0]),
                                    target_type=pl.FP32,
                                )
                                score_nope = pl.row_sum(pl.mul(q_nope_latent, kv_s))
                                score_pe = pl.row_sum(pl.mul(q_pe, pe_s))
                                cur_mi = pl.mul(pl.add(score_nope, score_pe), ATTN_SCALE)
                                cur_li = pl.full([1, 1], dtype=pl.FP32, value=1.0)
                                if kk == 0:
                                    oi = kv_s
                                    li = cur_li
                                    mi = cur_mi
                                else:
                                    mi_new = pl.maximum(mi, cur_mi)
                                    alpha = pl.exp(pl.sub(mi, mi_new))
                                    beta = pl.exp(pl.sub(cur_mi, mi_new))
                                    li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                                    oi = pl.add(
                                        pl.row_expand_mul(oi, alpha),
                                        pl.row_expand_mul(kv_s, beta),
                                    )
                                    mi = mi_new
                        ctx_latent = pl.row_expand_div(oi, li)

                        v_col = h * V_HEAD_DIM
                        for vb in pl.range(V_OUT_BLOCKS):
                            v0 = vb * V_OUT_CHUNK
                            wv_tile = pl.reshape(
                                pl.slice(
                                    w_latent_to_v,
                                    [1, KV_LORA_RANK, V_OUT_CHUNK],
                                    [h, 0, v0],
                                ),
                                [KV_LORA_RANK, V_OUT_CHUNK],
                            )
                            v_part = pl.matmul(
                                pl.cast(ctx_latent, target_type=pl.BF16),
                                wv_tile,
                                out_dtype=pl.FP32,
                            )
                            attn_row = pl.assemble(attn_row, v_part, [0, v_col + v0])
                    attn_front = pl.assemble(attn_front, attn_row, [b, 0])

                # Dispatch write to cross-node GM tensor.
                layer_id = pl.tensor.read(layer_id_t, [0])
                for b in pl.parallel(0, BATCH, 1, chunk=4):
                    target_node = (b + layer_id) % EP_NODES
                    token_row = pl.cast(
                        pl.slice(attn_front, [1, ATTN_OUT], [b, 0]), target_type=pl.BF16
                    )
                    dispatch_buf = pl.assemble(dispatch_buf, token_row, [target_node, b, 0])

            return dispatch_buf

    return Ds32Exp


def build_tensor_specs():
    """TensorSpecs for `run` driver. Initialisers mirror the scope1 example:
    centred uniform with 1/sqrt(fan_in) scaling on weights so RMSNorm /
    matmul outputs stay in BF16's well-resolved range.
    """
    import torch  # type: ignore[import]
    from golden import TensorSpec

    def init_hidden_states():
        return torch.rand(BATCH, HIDDEN) - 0.5

    def init_rms_weight():
        return torch.rand(1, HIDDEN) - 0.5

    def init_q_norm_weight():
        return torch.rand(1, Q_LORA_RANK) - 0.5

    def init_kv_norm_weight():
        return torch.rand(1, KV_LORA_RANK) - 0.5

    def init_k_norm_weight():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_k_norm_bias():
        return torch.rand(1, INDEX_HEAD_DIM) - 0.5

    def init_wq_a():
        return (torch.rand(HIDDEN, Q_LORA_RANK) - 0.5) / HIDDEN ** 0.5

    def init_wq_b():
        return (torch.rand(Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM) - 0.5) / Q_LORA_RANK ** 0.5

    def init_wkv_a():
        return (torch.rand(HIDDEN, KV_A_OUT) - 0.5) / HIDDEN ** 0.5

    def init_wq_b_idx():
        return (torch.rand(Q_LORA_RANK, INDEX_HEADS * INDEX_HEAD_DIM) - 0.5) / Q_LORA_RANK ** 0.5

    def init_wk_idx():
        return (torch.rand(HIDDEN, INDEX_HEAD_DIM) - 0.5) / HIDDEN ** 0.5

    def init_weights_proj():
        return (torch.rand(HIDDEN, INDEX_HEADS) - 0.5) / HIDDEN ** 0.5

    def init_w_q_nope_to_latent():
        return (torch.rand(NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK) - 0.5) / QK_NOPE_HEAD_DIM ** 0.5

    def init_w_latent_to_v():
        return (torch.rand(NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM) - 0.5) / KV_LORA_RANK ** 0.5

    def init_kv_cache():
        return torch.rand(CACHE_ROWS, KV_LORA_RANK) - 0.5

    def init_pe_cache():
        return torch.rand(CACHE_ROWS, QK_ROPE_HEAD_DIM) - 0.5

    def init_k_cache_idx():
        return torch.rand(CACHE_ROWS, INDEX_HEAD_DIM) - 0.5

    def init_rope_cos():
        return torch.rand(MAX_SEQ, QK_ROPE_HEAD_DIM) - 0.5

    def init_rope_sin():
        return torch.rand(MAX_SEQ, QK_ROPE_HEAD_DIM) - 0.5

    def init_seq_lens():
        return torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)

    def init_layer_id():
        return torch.tensor([0], dtype=torch.int32)

    def init_dispatch_buf():
        return torch.zeros(EP_NODES, BATCH, ATTN_OUT)

    return [
        TensorSpec("hidden_states", [BATCH, HIDDEN], torch.bfloat16, init_value=init_hidden_states),
        TensorSpec("seq_lens", [BATCH], torch.int32, init_value=init_seq_lens),
        TensorSpec("layer_id_t", [1], torch.int32, init_value=init_layer_id),
        TensorSpec("rope_cos", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope_cos),
        TensorSpec("rope_sin", [MAX_SEQ, QK_ROPE_HEAD_DIM], torch.float32, init_value=init_rope_sin),
        TensorSpec("kv_cache", [CACHE_ROWS, KV_LORA_RANK], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("pe_cache", [CACHE_ROWS, QK_ROPE_HEAD_DIM], torch.bfloat16, init_value=init_pe_cache),
        TensorSpec("k_cache_idx", [CACHE_ROWS, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_k_cache_idx),
        TensorSpec("input_rms_weight", [1, HIDDEN], torch.float32, init_value=init_rms_weight),
        TensorSpec("wq_a", [HIDDEN, Q_LORA_RANK], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("q_norm_weight", [1, Q_LORA_RANK], torch.float32, init_value=init_q_norm_weight),
        TensorSpec("wq_b", [Q_LORA_RANK, NUM_HEADS * QK_HEAD_DIM], torch.bfloat16, init_value=init_wq_b),
        TensorSpec("wkv_a", [HIDDEN, KV_A_OUT], torch.bfloat16, init_value=init_wkv_a),
        TensorSpec("kv_norm_weight", [1, KV_LORA_RANK], torch.float32, init_value=init_kv_norm_weight),
        TensorSpec(
            "wq_b_idx",
            [Q_LORA_RANK, INDEX_HEADS * INDEX_HEAD_DIM],
            torch.bfloat16,
            init_value=init_wq_b_idx,
        ),
        TensorSpec("wk_idx", [HIDDEN, INDEX_HEAD_DIM], torch.bfloat16, init_value=init_wk_idx),
        TensorSpec("k_norm_weight", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_weight),
        TensorSpec("k_norm_bias", [1, INDEX_HEAD_DIM], torch.float32, init_value=init_k_norm_bias),
        TensorSpec("weights_proj", [HIDDEN, INDEX_HEADS], torch.float32, init_value=init_weights_proj),
        TensorSpec(
            "w_q_nope_to_latent",
            [NUM_HEADS, QK_NOPE_HEAD_DIM, KV_LORA_RANK],
            torch.bfloat16,
            init_value=init_w_q_nope_to_latent,
        ),
        TensorSpec(
            "w_latent_to_v",
            [NUM_HEADS, KV_LORA_RANK, V_HEAD_DIM],
            torch.bfloat16,
            init_value=init_w_latent_to_v,
        ),
        TensorSpec(
            "dispatch_buf",
            [EP_NODES, BATCH, ATTN_OUT],
            torch.bfloat16,
            init_value=init_dispatch_buf,
            is_output=True,
        ),
    ]


def golden_ds32exp(tensors):
    """PyTorch reference covering all 4 scopes.

    Mirrors the kernel exactly: lo/hi half-split RoPE for both MLA q_pe/k_pe
    and the indexer's q_pe/k_pe (we follow the in-tree convention rather
    than the official `interleaved=True/False` view), the Hadamard step is
    reduced to its scalar scale (`INDEX_HEAD_DIM ** -0.5`), and FP8 quant
    is approximated by a BF16 -> FP8E4M3FN -> BF16 round-trip.

    Topk is left as a placeholder (all-zero indices) so scope4 attends only
    to position 0; this matches `# TODO(topk)` behaviour in the kernel.
    """
    import torch  # type: ignore[import]

    hidden_states = tensors["hidden_states"].float()
    seq_lens = tensors["seq_lens"]
    layer_id = int(tensors["layer_id_t"][0].item())
    rope_cos = tensors["rope_cos"].float()
    rope_sin = tensors["rope_sin"].float()
    kv_cache = tensors["kv_cache"].clone()
    pe_cache = tensors["pe_cache"].clone()
    k_cache_idx = tensors["k_cache_idx"].clone()
    input_rms_weight = tensors["input_rms_weight"].float()
    wq_a = tensors["wq_a"].float()
    q_norm_weight = tensors["q_norm_weight"].float()
    wq_b = tensors["wq_b"].float()
    wkv_a = tensors["wkv_a"].float()
    kv_norm_weight = tensors["kv_norm_weight"].float()
    wq_b_idx = tensors["wq_b_idx"].float()
    wk_idx = tensors["wk_idx"].float()
    k_norm_weight = tensors["k_norm_weight"].float()
    k_norm_bias = tensors["k_norm_bias"].float()
    weights_proj = tensors["weights_proj"].float()
    w_q_nope_to_latent = tensors["w_q_nope_to_latent"].float()
    w_latent_to_v = tensors["w_latent_to_v"].float()

    half = HALF_ROPE

    def rope_half(vec, cos_lo, cos_hi, sin_lo, sin_hi):
        # vec: [..., QK_ROPE_HEAD_DIM]; matches kernel's lo/hi-split rotation.
        lo = vec[..., :half]
        hi = vec[..., half:]
        rot_lo = lo * cos_lo - hi * sin_lo
        rot_hi = hi * cos_hi + lo * sin_hi
        return torch.cat([rot_lo, rot_hi], dim=-1)

    # ── Scope 1 golden: RMSNorm + projections + q_pe / k_pe RoPE ──
    sq_sum = (hidden_states * hidden_states).sum(dim=1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * HIDDEN_INV + EPS)
    normed = (hidden_states * inv_rms * input_rms_weight).to(torch.bfloat16).float()

    qr = (normed @ wq_a).to(torch.bfloat16).float() * q_norm_weight
    qr_bf16 = qr.to(torch.bfloat16)
    q_proj = (qr_bf16.float() @ wq_b).to(torch.bfloat16).float()
    kv_a = (normed @ wkv_a).to(torch.bfloat16).float()

    q_proj_view = q_proj.view(BATCH, NUM_HEADS, QK_HEAD_DIM)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cos_lo = rope_cos[pos:pos + 1, :half]
        cos_hi = rope_cos[pos:pos + 1, half:]
        sin_lo = rope_sin[pos:pos + 1, :half]
        sin_hi = rope_sin[pos:pos + 1, half:]

        q_pe_b = q_proj_view[b, :, QK_NOPE_HEAD_DIM:]
        q_proj_view[b, :, QK_NOPE_HEAD_DIM:] = rope_half(q_pe_b, cos_lo, cos_hi, sin_lo, sin_hi)

        kv_latent = kv_a[b:b + 1, :KV_LORA_RANK]
        kv_normed = (kv_latent * kv_norm_weight).to(torch.bfloat16).float()
        cache_row = b * MAX_SEQ + pos
        kv_cache[cache_row, :] = kv_normed.squeeze(0).to(torch.bfloat16)

        k_pe_b = kv_a[b:b + 1, KV_LORA_RANK:KV_LORA_RANK + QK_ROPE_HEAD_DIM]
        k_pe_rot = rope_half(k_pe_b, cos_lo, cos_hi, sin_lo, sin_hi)
        pe_cache[cache_row, :] = k_pe_rot.squeeze(0).to(torch.bfloat16)
    q_proj = q_proj_view.reshape(BATCH, NUM_HEADS * QK_HEAD_DIM)
    q_proj_bf16 = q_proj.to(torch.bfloat16).float()

    # ── Scope 2 golden: indexer proj + RoPE + Hadamard placeholder + fp8 placeholder ──
    q_idx_full = (qr_bf16.float() @ wq_b_idx).to(torch.bfloat16).float()
    k_idx = (hidden_states.to(torch.bfloat16).float() @ wk_idx).to(torch.bfloat16).float()

    # LayerNorm on k_idx.
    mean = k_idx.mean(dim=-1, keepdim=True)
    centered = k_idx - mean
    var = (centered * centered).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + EPS)
    k_idx = (centered * inv_std * k_norm_weight + k_norm_bias).to(torch.bfloat16).float()

    # RoPE (lo/hi half) on q_idx_full per index head and on k_idx (single head).
    q_idx_full_view = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        cos_lo = rope_cos[pos:pos + 1, :half]
        cos_hi = rope_cos[pos:pos + 1, half:]
        sin_lo = rope_sin[pos:pos + 1, :half]
        sin_hi = rope_sin[pos:pos + 1, half:]

        q_pe_b = q_idx_full_view[b, :, :QK_ROPE_HEAD_DIM]
        q_idx_full_view[b, :, :QK_ROPE_HEAD_DIM] = rope_half(q_pe_b, cos_lo, cos_hi, sin_lo, sin_hi)

        k_pe_b = k_idx[b:b + 1, :QK_ROPE_HEAD_DIM]
        k_idx[b:b + 1, :QK_ROPE_HEAD_DIM] = rope_half(k_pe_b, cos_lo, cos_hi, sin_lo, sin_hi)
    q_idx_full = q_idx_full_view.reshape(BATCH, INDEX_HEADS * INDEX_HEAD_DIM)

    # TODO(hadamard_transform) placeholder: scalar scale only.
    q_idx_full = (q_idx_full * HADAMARD_SCALE).to(torch.bfloat16).float()
    k_idx = (k_idx * HADAMARD_SCALE).to(torch.bfloat16).float()

    # TODO(fp8 quant) placeholder: BF16 -> FP8E4M3FN -> BF16 round-trip.
    q_idx_full = q_idx_full.to(torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16).float()
    k_idx = k_idx.to(torch.bfloat16).to(torch.float8_e4m3fn).to(torch.bfloat16).float()

    # Per-head weights and aggregation.
    w_scale = INDEX_HEADS_INV_SQRT * INDEX_SOFTMAX_SCALE
    weights = (hidden_states.to(torch.bfloat16).float() @ weights_proj.to(torch.bfloat16).float()) * w_scale
    q_idx_full_view = q_idx_full.view(BATCH, INDEX_HEADS, INDEX_HEAD_DIM)
    q_idx = (weights.unsqueeze(-1) * q_idx_full_view).sum(dim=1)
    q_idx_bf16 = q_idx.to(torch.bfloat16).float()

    for b in range(BATCH):
        pos = int(seq_lens[b].item()) - 1
        k_cache_idx[b * MAX_SEQ + pos, :] = k_idx[b].to(torch.bfloat16)

    # ── Scope 3 golden: tiled scoring (q_idx · k_cache_idx[b, :]) ──
    # Topk is a placeholder (all-zero indices); kernel currently picks position 0.
    topk_idx = torch.zeros(BATCH, INDEX_TOPK, dtype=torch.int64)

    # ── Scope 4 golden: sparse MQA attention + dispatch write ──
    attn_front = torch.zeros(BATCH, ATTN_OUT, dtype=torch.float32)
    for b in range(BATCH):
        ctx_len = int(seq_lens[b].item())
        sparse_k = min(INDEX_TOPK, ctx_len)
        for h in range(NUM_HEADS):
            q_col = h * QK_HEAD_DIM
            q_nope = q_proj_bf16[b, q_col:q_col + QK_NOPE_HEAD_DIM]
            q_pe = q_proj_bf16[b, q_col + QK_NOPE_HEAD_DIM:q_col + QK_HEAD_DIM]
            q_nope_latent = q_nope @ w_q_nope_to_latent[h]

            oi = torch.zeros(KV_LORA_RANK, dtype=torch.float32)
            li = torch.zeros(1, dtype=torch.float32)
            mi = torch.zeros(1, dtype=torch.float32)
            for kk in range(sparse_k):
                pos = int(topk_idx[b, kk].item())
                if pos < 0:
                    continue
                cache_s = b * MAX_SEQ + pos
                kv_s = kv_cache[cache_s].float()
                pe_s = pe_cache[cache_s].float()
                score_nope = (q_nope_latent * kv_s).sum()
                score_pe = (q_pe * pe_s).sum()
                cur_mi = (score_nope + score_pe) * ATTN_SCALE
                cur_li = torch.tensor([1.0])
                cur_mi = cur_mi.view(1)
                if kk == 0:
                    oi = kv_s
                    li = cur_li
                    mi = cur_mi
                else:
                    mi_new = torch.maximum(mi, cur_mi)
                    alpha = torch.exp(mi - mi_new)
                    beta = torch.exp(cur_mi - mi_new)
                    li = alpha * li + beta * cur_li
                    oi = oi * alpha + kv_s * beta
                    mi = mi_new
            ctx_latent = (oi / li).to(torch.bfloat16).float()
            ctx_v = ctx_latent @ w_latent_to_v[h]
            attn_front[b, h * V_HEAD_DIM:(h + 1) * V_HEAD_DIM] = ctx_v

    dispatch_buf = tensors["dispatch_buf"]
    dispatch_buf.zero_()
    for b in range(BATCH):
        target_node = (b + layer_id) % EP_NODES
        dispatch_buf[target_node, b, :] = attn_front[b].to(torch.bfloat16)


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_ds32exp_program(),
        tensor_specs=build_tensor_specs(),
        golden_fn=golden_ds32exp,
        config=RunConfig(
            rtol=2e-2,
            atol=2e-2,
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
