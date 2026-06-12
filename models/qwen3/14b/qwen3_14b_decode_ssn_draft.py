# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B single-layer decode forward, serial tile-DSL style, 4D-blocked.

All tensors -- kernel parameters and internal bridges alike -- use the 4D
pre-blocked layout of qwen3_32b_decode_4d.py:
  - activations / output:  [COL_BLOCKS, 1, BATCH, CHUNK]
  - weights:               [K_BLOCKS, N_BLOCKS, MM_K, MM_N] (one 4 KB tile per block)
  - rms / norm weights:    [K_BLOCKS, 1, 1, CHUNK]
  - rope tables:           [MAX_SEQ, 1, 1, HEAD_DIM]
  - paged KV caches:       [MAX_BLOCKS_PER_SEQ, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM]
  - scalar tables:         [BATCH, ..., 1, 1]
Tensors stay 4D end to end: every access loads one [1, 1, rows, cols] block
(dims 2/3 untouched) and pl.tile.reshape's it to a 2D tile; stores reshape
the 2D tile back to a 4D block. Internal bridges are declared right before
the loop that first writes them.

Style rules:
  1. One @pl.program / @pl.function, body wrapped in a single
     CORE_GROUP pl.at scope (SuperscalarNPU entry form: no cube/vector
     core hierarchy). Inside, control flow is only pl.range and if/else
     (no spmd / parallel / pipeline).
  2. Compute is tile-level (pl.tile.*); the only tensor-level op is
     pl.tensor.read (host scalars for loop bounds / paged addressing).
  3. Every tile's static shape is 4 KB (FP32 1024 / BF16 2048 elems);
     a tile whose live region is smaller declares it via set_validshape.
  4. SSA names: a load tile's type embeds its [idx, ...] offset, so the
     DSL rejects reusing one tile name across two loops -- their loop Vars
     differ even when the source text is identical. Every loop therefore
     prefixes its locals with a unique tag (s1_/q_/k_/v_/qn_/kn_/rp_/fa_/
     op_/s3_/pn_/ml_/dn_). Within one scope set_validshape still rebinds a
     name in place, and loop-carried accumulators (matmul_acc, online-
     softmax oi) rebind.

Implements all three scopes: 1 (RMSNorm -> Q/K/V proj -> per-head q/k
norm), 2 (RoPE + paged KV-cache + flash attention), 3 (out-proj +
residual -> post-RMSNorm -> MLP -> residual).
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

# --- Qwen3-14B model shape (B fixed to its max value 16, no dynamic dim) ----
BATCH = 16
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM        # 5120
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM  # 1024
INTERMEDIATE = 17408                 # MLP hidden size

EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM

VEC_BF16 = 128  # [BATCH, 128] BF16 = 4 KB (full BF16 column tile)
VEC_W = 64      # [BATCH, 64]  FP32 = 4 KB (one half of a BF16 tile)
MM_N = 64       # [BATCH, 64]  FP32 = 4 KB (matmul accumulator)
MM_K = 32       # [32, 64] BF16 = 4 KB weight tile is the binding operand

HALVES_PER_HEAD = 2  # VEC_W blocks per HEAD_DIM head

# --- Scope 2 (RoPE + paged-cache + flash attention) ------------------------
HALF_DIM = HEAD_DIM // 2                  # 64
BLOCK_SIZE = 128                          # paged KV-cache block length
MAX_SEQ = 4096
MAX_BLOCKS_PER_SEQ = (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE  # 32
Q_HEAD_BATCH = 5                          # real Q heads per KV head
Q_HEAD_PAD = 16                           # padded Q rows the cube operates on
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS      # 5
TOTAL_Q_GROUPS = NUM_KV_HEADS             # Q_GROUPS == 1 for Qwen3-14B
NEG_INF = -3.0e38
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)

# Flash-attention sub-tiling, sized so every QK / SV / oi tile is 4 KB.
ATT_SEQ = 64   # online-step width; scores [Q_HEAD_PAD, 64] FP32 = 4 KB
QK_KD = 32     # QK head-dim chunk; k tile [ATT_SEQ, 32] BF16 = 4 KB
QK_KSTEPS = HEAD_DIM // QK_KD   # 4
SV_SEQ = 32    # SV seq chunk; v tile [32, HALF_DIM] BF16 = 4 KB; oi half 4 KB
SV_SSTEPS = ATT_SEQ // SV_SEQ   # 2


def build_qwen3_14b_decode_program():
    """Wrap the decode kernel as a @pl.program for the SuperscalarNPU
    (PR #1680) DDR + TREG IR pipeline. The whole kernel is one flat
    tile-op region inside a single CORE_GROUP scope -- SuperscalarNPU has
    no cube/vector core hierarchy to outline."""

    @pl.program
    class Qwen3Decode14B:
        @pl.function(type=pl.FunctionType.Opaque)
        def qwen3_14b_decode(
            self,
            current_hidden: pl.Tensor[[HIDDEN // VEC_BF16, 1, BATCH, VEC_BF16], pl.BF16],
            input_rms_weight: pl.Tensor[[HIDDEN // VEC_BF16, 1, 1, VEC_BF16], pl.FP32],
            wq: pl.Tensor[[HIDDEN // MM_K, HIDDEN // MM_N, MM_K, MM_N], pl.BF16],
            wk: pl.Tensor[[HIDDEN // MM_K, KV_HIDDEN // MM_N, MM_K, MM_N], pl.BF16],
            wv: pl.Tensor[[HIDDEN // MM_K, KV_HIDDEN // MM_N, MM_K, MM_N], pl.BF16],
            q_norm_weight: pl.Tensor[[1, 1, 1, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[1, 1, 1, HEAD_DIM], pl.FP32],
            seq_lens: pl.Tensor[[BATCH, 1, 1, 1], pl.INT32],
            block_table: pl.Tensor[[BATCH, MAX_BLOCKS_PER_SEQ, 1, 1], pl.INT32],
            slot_mapping: pl.Tensor[[BATCH, 1, 1, 1], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ, 1, 1, HEAD_DIM], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ, 1, 1, HEAD_DIM], pl.FP32],
            k_cache: pl.Tensor[[MAX_BLOCKS_PER_SEQ, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM], pl.BF16],
            v_cache: pl.Tensor[[MAX_BLOCKS_PER_SEQ, NUM_KV_HEADS, BLOCK_SIZE, HEAD_DIM], pl.BF16],
            wo: pl.Tensor[[HIDDEN // MM_K, HIDDEN // MM_N, MM_K, MM_N], pl.BF16],
            post_rms_weight: pl.Tensor[[HIDDEN // VEC_BF16, 1, 1, VEC_BF16], pl.FP32],
            w_gate: pl.Tensor[[HIDDEN // MM_K, INTERMEDIATE // MM_N, MM_K, MM_N], pl.BF16],
            w_up: pl.Tensor[[HIDDEN // MM_K, INTERMEDIATE // MM_N, MM_K, MM_N], pl.BF16],
            w_down: pl.Tensor[[INTERMEDIATE // MM_K, HIDDEN // MM_N, MM_K, MM_N], pl.BF16],
            next_hidden: pl.Out[pl.Tensor[[HIDDEN // MM_N, 1, BATCH, MM_N], pl.BF16]],
        ) -> pl.Tensor[[HIDDEN // MM_N, 1, BATCH, MM_N], pl.BF16]:
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="qwen3_14b_decode"):
                # =====================================================================
                # 1. Input RMSNorm:  normed_all = (x / rms(x)) * gamma
                # =====================================================================
                # sum of squares over HIDDEN (BF16 block load -> two FP32 halves -> square)
                s1_sumsq_pre = pl.tile.full([BATCH, VEC_W], value=0.0, dtype=pl.FP32)
                s1_sumsq = pl.tile.set_validshape(s1_sumsq_pre, BATCH, 1)
                for kb in pl.range(HIDDEN // VEC_BF16):
                    s1_x_blk = pl.tile.load(current_hidden, [1, 1, BATCH, VEC_BF16], [kb, 0, 0, 0])
                    s1_x_bf16 = pl.tile.reshape(s1_x_blk, [BATCH, VEC_BF16])
                    for h in pl.range(2):
                        s1_h0 = h * VEC_W
                        s1_x_half_pre = pl.tile.slice(s1_x_bf16, [BATCH, VEC_W], [0, s1_h0])
                        s1_x_half = pl.tile.set_validshape(s1_x_half_pre, BATCH, VEC_W)
                        s1_x = pl.tile.cast(s1_x_half, dtype=pl.FP32)
                        s1_sq = pl.tile.mul(s1_x, s1_x)
                        s1_part_pre = pl.tile.row_sum(s1_sq)
                        s1_part = pl.tile.set_validshape(s1_part_pre, BATCH, 1)
                        s1_sumsq_acc = pl.tile.add(s1_sumsq, s1_part)
                        s1_sumsq = pl.tile.set_validshape(s1_sumsq_acc, BATCH, 1)

                s1_mean_sq_pre = pl.tile.mul(s1_sumsq, HIDDEN_INV)
                s1_mean_sq = pl.tile.set_validshape(s1_mean_sq_pre, BATCH, 1)
                s1_variance_pre = pl.tile.add(s1_mean_sq, EPS)
                s1_variance = pl.tile.set_validshape(s1_variance_pre, BATCH, 1)
                s1_rms_pre = pl.tile.sqrt(s1_variance)
                s1_rms = pl.tile.set_validshape(s1_rms_pre, BATCH, 1)
                s1_inv_rms_pre = pl.tile.recip(s1_rms)
                s1_inv_rms = pl.tile.set_validshape(s1_inv_rms_pre, BATCH, 1)

                # normalize + scale by gamma -> BF16 bridge buffer (VEC_W column blocks)
                normed_all = pl.create_tensor([HIDDEN // VEC_W, 1, BATCH, VEC_W], dtype=pl.BF16)
                for kb in pl.range(HIDDEN // VEC_BF16):
                    s2_x_blk = pl.tile.load(current_hidden, [1, 1, BATCH, VEC_BF16], [kb, 0, 0, 0])
                    s2_x_bf16 = pl.tile.reshape(s2_x_blk, [BATCH, VEC_BF16])
                    for h in pl.range(2):
                        s2_h0 = h * VEC_W
                        s2_x_half_pre = pl.tile.slice(s2_x_bf16, [BATCH, VEC_W], [0, s2_h0])
                        s2_x_half = pl.tile.set_validshape(s2_x_half_pre, BATCH, VEC_W)
                        s2_x = pl.tile.cast(s2_x_half, dtype=pl.FP32)
                        s2_gamma_blk = pl.tile.load(input_rms_weight, [1, 1, 1, VEC_W], [kb, 0, 0, s2_h0])
                        s2_gamma_pre = pl.tile.reshape(s2_gamma_blk, [1, VEC_W])
                        s2_gamma = pl.tile.set_validshape(s2_gamma_pre, 1, VEC_W)
                        s2_x_scaled = pl.tile.row_expand_mul(s2_x, s1_inv_rms)
                        s2_normed = pl.tile.col_expand_mul(s2_x_scaled, s2_gamma)
                        s2_normed_bf16_pre = pl.tile.cast(s2_normed, dtype=pl.BF16)
                        s2_normed_bf16 = pl.tile.set_validshape(s2_normed_bf16_pre, BATCH, VEC_W)
                        s2_normed_blk = pl.tile.reshape(s2_normed_bf16, [1, 1, BATCH, VEC_W])
                        pl.tile.store(normed_all, s2_normed_blk, [kb * 2 + h, 0, 0, 0])

                # =====================================================================
                # 2. Q / K / V projection:  proj = normed_all @ W
                #    (peeled-first matmul + matmul_acc over the K tiles, per output tile)
                #    An MM_K activation chunk kb sits in VEC_W block kb // 2, half kb % 2.
                # =====================================================================
                # --- Q projection: [BATCH, HIDDEN] @ [HIDDEN, HIDDEN] ---
                q_proj = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
                for nb in pl.range(HIDDEN // MM_N):
                    q_a0_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
                    q_a0_pre = pl.tile.reshape(q_a0_blk, [BATCH, MM_K])
                    q_a0 = pl.tile.set_validshape(q_a0_pre, BATCH, MM_K)
                    q_w0_blk = pl.tile.load(wq, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
                    q_w0 = pl.tile.reshape(q_w0_blk, [MM_K, MM_N])
                    q_acc = pl.tile.matmul(q_a0, q_w0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // MM_K):
                        q_a_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
                        q_a_pre = pl.tile.reshape(q_a_blk, [BATCH, MM_K])
                        q_a = pl.tile.set_validshape(q_a_pre, BATCH, MM_K)
                        q_w_blk = pl.tile.load(wq, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
                        q_w = pl.tile.reshape(q_w_blk, [MM_K, MM_N])
                        q_acc = pl.tile.matmul_acc(q_acc, q_a, q_w)
                    q_acc_blk = pl.tile.reshape(q_acc, [1, 1, BATCH, MM_N])
                    pl.tile.store(q_proj, q_acc_blk, [nb, 0, 0, 0])

                # --- K projection: [BATCH, HIDDEN] @ [HIDDEN, KV_HIDDEN] ---
                k_proj = pl.create_tensor([KV_HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
                for nb in pl.range(KV_HIDDEN // MM_N):
                    k_a0_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
                    k_a0_pre = pl.tile.reshape(k_a0_blk, [BATCH, MM_K])
                    k_a0 = pl.tile.set_validshape(k_a0_pre, BATCH, MM_K)
                    k_w0_blk = pl.tile.load(wk, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
                    k_w0 = pl.tile.reshape(k_w0_blk, [MM_K, MM_N])
                    k_acc = pl.tile.matmul(k_a0, k_w0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // MM_K):
                        k_a_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
                        k_a_pre = pl.tile.reshape(k_a_blk, [BATCH, MM_K])
                        k_a = pl.tile.set_validshape(k_a_pre, BATCH, MM_K)
                        k_w_blk = pl.tile.load(wk, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
                        k_w = pl.tile.reshape(k_w_blk, [MM_K, MM_N])
                        k_acc = pl.tile.matmul_acc(k_acc, k_a, k_w)
                    k_acc_blk = pl.tile.reshape(k_acc, [1, 1, BATCH, MM_N])
                    pl.tile.store(k_proj, k_acc_blk, [nb, 0, 0, 0])

                # --- V projection: [BATCH, HIDDEN] @ [HIDDEN, KV_HIDDEN] ---
                v_proj = pl.create_tensor([KV_HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
                for nb in pl.range(KV_HIDDEN // MM_N):
                    v_a0_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
                    v_a0_pre = pl.tile.reshape(v_a0_blk, [BATCH, MM_K])
                    v_a0 = pl.tile.set_validshape(v_a0_pre, BATCH, MM_K)
                    v_w0_blk = pl.tile.load(wv, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
                    v_w0 = pl.tile.reshape(v_w0_blk, [MM_K, MM_N])
                    v_acc = pl.tile.matmul(v_a0, v_w0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // MM_K):
                        v_a_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
                        v_a_pre = pl.tile.reshape(v_a_blk, [BATCH, MM_K])
                        v_a = pl.tile.set_validshape(v_a_pre, BATCH, MM_K)
                        v_w_blk = pl.tile.load(wv, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
                        v_w = pl.tile.reshape(v_w_blk, [MM_K, MM_N])
                        v_acc = pl.tile.matmul_acc(v_acc, v_a, v_w)
                    v_acc_blk = pl.tile.reshape(v_acc, [1, 1, BATCH, MM_N])
                    pl.tile.store(v_proj, v_acc_blk, [nb, 0, 0, 0])

                # =====================================================================
                # 3. Per-head q_norm / k_norm. A head's HEAD_DIM spans two consecutive
                #    MM_N blocks (lo/hi halves, batch rows): row-wise RMSNorm per head.
                # =====================================================================
                q_proj_norm = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
                for hq in pl.range(NUM_HEADS):
                    qn_b_lo = hq * HALVES_PER_HEAD
                    qn_x_lo_blk = pl.tile.load(q_proj, [1, 1, BATCH, VEC_W], [qn_b_lo, 0, 0, 0])
                    qn_x_lo = pl.tile.reshape(qn_x_lo_blk, [BATCH, VEC_W])
                    qn_x_hi_blk = pl.tile.load(q_proj, [1, 1, BATCH, VEC_W], [qn_b_lo + 1, 0, 0, 0])
                    qn_x_hi = pl.tile.reshape(qn_x_hi_blk, [BATCH, VEC_W])
                    qn_sq_lo = pl.tile.mul(qn_x_lo, qn_x_lo)
                    qn_sumsq_pre = pl.tile.row_sum(qn_sq_lo)
                    qn_sumsq = pl.tile.set_validshape(qn_sumsq_pre, BATCH, 1)
                    qn_sq_hi = pl.tile.mul(qn_x_hi, qn_x_hi)
                    qn_part_pre = pl.tile.row_sum(qn_sq_hi)
                    qn_part = pl.tile.set_validshape(qn_part_pre, BATCH, 1)
                    qn_sumsq_acc = pl.tile.add(qn_sumsq, qn_part)
                    qn_sumsq = pl.tile.set_validshape(qn_sumsq_acc, BATCH, 1)
                    qn_mean_sq_pre = pl.tile.mul(qn_sumsq, HEAD_DIM_INV)
                    qn_mean_sq = pl.tile.set_validshape(qn_mean_sq_pre, BATCH, 1)
                    qn_variance_pre = pl.tile.add(qn_mean_sq, EPS)
                    qn_variance = pl.tile.set_validshape(qn_variance_pre, BATCH, 1)
                    qn_inv_rms_pre = pl.tile.rsqrt(qn_variance)
                    qn_inv_rms = pl.tile.set_validshape(qn_inv_rms_pre, BATCH, 1)
                    qn_g_lo_blk = pl.tile.load(q_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, 0])
                    qn_g_lo_pre = pl.tile.reshape(qn_g_lo_blk, [1, HALF_DIM])
                    qn_g_lo = pl.tile.set_validshape(qn_g_lo_pre, 1, HALF_DIM)
                    qn_g_hi_blk = pl.tile.load(q_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, HALF_DIM])
                    qn_g_hi_pre = pl.tile.reshape(qn_g_hi_blk, [1, HALF_DIM])
                    qn_g_hi = pl.tile.set_validshape(qn_g_hi_pre, 1, HALF_DIM)
                    qn_x_lo_scaled = pl.tile.row_expand_mul(qn_x_lo, qn_inv_rms)
                    qn_n_lo = pl.tile.col_expand_mul(qn_x_lo_scaled, qn_g_lo)
                    qn_x_hi_scaled = pl.tile.row_expand_mul(qn_x_hi, qn_inv_rms)
                    qn_n_hi = pl.tile.col_expand_mul(qn_x_hi_scaled, qn_g_hi)
                    qn_n_lo_blk = pl.tile.reshape(qn_n_lo, [1, 1, BATCH, VEC_W])
                    qn_n_hi_blk = pl.tile.reshape(qn_n_hi, [1, 1, BATCH, VEC_W])
                    pl.tile.store(q_proj_norm, qn_n_lo_blk, [qn_b_lo, 0, 0, 0])
                    pl.tile.store(q_proj_norm, qn_n_hi_blk, [qn_b_lo + 1, 0, 0, 0])

                k_proj_norm = pl.create_tensor([KV_HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
                for hk in pl.range(NUM_KV_HEADS):
                    kn_b_lo = hk * HALVES_PER_HEAD
                    kn_x_lo_blk = pl.tile.load(k_proj, [1, 1, BATCH, VEC_W], [kn_b_lo, 0, 0, 0])
                    kn_x_lo = pl.tile.reshape(kn_x_lo_blk, [BATCH, VEC_W])
                    kn_x_hi_blk = pl.tile.load(k_proj, [1, 1, BATCH, VEC_W], [kn_b_lo + 1, 0, 0, 0])
                    kn_x_hi = pl.tile.reshape(kn_x_hi_blk, [BATCH, VEC_W])
                    kn_sq_lo = pl.tile.mul(kn_x_lo, kn_x_lo)
                    kn_sumsq_pre = pl.tile.row_sum(kn_sq_lo)
                    kn_sumsq = pl.tile.set_validshape(kn_sumsq_pre, BATCH, 1)
                    kn_sq_hi = pl.tile.mul(kn_x_hi, kn_x_hi)
                    kn_part_pre = pl.tile.row_sum(kn_sq_hi)
                    kn_part = pl.tile.set_validshape(kn_part_pre, BATCH, 1)
                    kn_sumsq_acc = pl.tile.add(kn_sumsq, kn_part)
                    kn_sumsq = pl.tile.set_validshape(kn_sumsq_acc, BATCH, 1)
                    kn_mean_sq_pre = pl.tile.mul(kn_sumsq, HEAD_DIM_INV)
                    kn_mean_sq = pl.tile.set_validshape(kn_mean_sq_pre, BATCH, 1)
                    kn_variance_pre = pl.tile.add(kn_mean_sq, EPS)
                    kn_variance = pl.tile.set_validshape(kn_variance_pre, BATCH, 1)
                    kn_inv_rms_pre = pl.tile.rsqrt(kn_variance)
                    kn_inv_rms = pl.tile.set_validshape(kn_inv_rms_pre, BATCH, 1)
                    kn_g_lo_blk = pl.tile.load(k_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, 0])
                    kn_g_lo_pre = pl.tile.reshape(kn_g_lo_blk, [1, HALF_DIM])
                    kn_g_lo = pl.tile.set_validshape(kn_g_lo_pre, 1, HALF_DIM)
                    kn_g_hi_blk = pl.tile.load(k_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, HALF_DIM])
                    kn_g_hi_pre = pl.tile.reshape(kn_g_hi_blk, [1, HALF_DIM])
                    kn_g_hi = pl.tile.set_validshape(kn_g_hi_pre, 1, HALF_DIM)
                    kn_x_lo_scaled = pl.tile.row_expand_mul(kn_x_lo, kn_inv_rms)
                    kn_n_lo = pl.tile.col_expand_mul(kn_x_lo_scaled, kn_g_lo)
                    kn_x_hi_scaled = pl.tile.row_expand_mul(kn_x_hi, kn_inv_rms)
                    kn_n_hi = pl.tile.col_expand_mul(kn_x_hi_scaled, kn_g_hi)
                    kn_n_lo_blk = pl.tile.reshape(kn_n_lo, [1, 1, BATCH, VEC_W])
                    kn_n_hi_blk = pl.tile.reshape(kn_n_hi, [1, 1, BATCH, VEC_W])
                    pl.tile.store(k_proj_norm, kn_n_lo_blk, [kn_b_lo, 0, 0, 0])
                    pl.tile.store(k_proj_norm, kn_n_hi_blk, [kn_b_lo + 1, 0, 0, 0])

                # =====================================================================
                # 2a. RoPE + paged KV-cache write. Per batch row, per KV head: rotate-half
                #     K -> k_cache, copy V -> v_cache, rotate the Q_HEAD_BATCH Q heads and
                #     zero-pad to Q_HEAD_PAD rows -> all_q_padded.
                # =====================================================================
                all_q_padded = pl.create_tensor([BATCH * TOTAL_Q_GROUPS, 1, Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
                for b in pl.range(BATCH):
                    rp_ctx_len = pl.tensor.read(seq_lens, [b, 0, 0, 0])
                    rp_pos = rp_ctx_len - 1
                    rp_slot = pl.tensor.read(slot_mapping, [b, 0, 0, 0])
                    rp_slot_block = rp_slot // BLOCK_SIZE
                    rp_slot_offset = rp_slot - rp_slot_block * BLOCK_SIZE

                    rp_cos_blk = pl.tile.load(rope_cos, [1, 1, 1, HEAD_DIM], [rp_pos, 0, 0, 0])
                    rp_cos_row_pre = pl.tile.reshape(rp_cos_blk, [1, HEAD_DIM])
                    rp_cos_row = pl.tile.set_validshape(rp_cos_row_pre, 1, HEAD_DIM)
                    rp_sin_blk = pl.tile.load(rope_sin, [1, 1, 1, HEAD_DIM], [rp_pos, 0, 0, 0])
                    rp_sin_row_pre = pl.tile.reshape(rp_sin_blk, [1, HEAD_DIM])
                    rp_sin_row = pl.tile.set_validshape(rp_sin_row_pre, 1, HEAD_DIM)
                    rp_cos_lo_pre = pl.tile.slice(rp_cos_row, [1, HALF_DIM], [0, 0])
                    rp_cos_lo = pl.tile.set_validshape(rp_cos_lo_pre, 1, HALF_DIM)
                    rp_cos_hi_pre = pl.tile.slice(rp_cos_row, [1, HALF_DIM], [0, HALF_DIM])
                    rp_cos_hi = pl.tile.set_validshape(rp_cos_hi_pre, 1, HALF_DIM)
                    rp_sin_lo_pre = pl.tile.slice(rp_sin_row, [1, HALF_DIM], [0, 0])
                    rp_sin_lo = pl.tile.set_validshape(rp_sin_lo_pre, 1, HALF_DIM)
                    rp_sin_hi_pre = pl.tile.slice(rp_sin_row, [1, HALF_DIM], [0, HALF_DIM])
                    rp_sin_hi = pl.tile.set_validshape(rp_sin_hi_pre, 1, HALF_DIM)

                    for ki in pl.range(NUM_KV_HEADS):
                        rp_kv_blo = ki * HALVES_PER_HEAD

                        # K head RoPE -> k_cache (rotate-half).
                        rp_k_lo_blk = pl.tile.load(k_proj_norm, [1, 1, 1, HALF_DIM], [rp_kv_blo, 0, b, 0])
                        rp_k_lo_pre = pl.tile.reshape(rp_k_lo_blk, [1, HALF_DIM])
                        rp_k_lo = pl.tile.set_validshape(rp_k_lo_pre, 1, HALF_DIM)
                        rp_k_hi_blk = pl.tile.load(k_proj_norm, [1, 1, 1, HALF_DIM], [rp_kv_blo + 1, 0, b, 0])
                        rp_k_hi_pre = pl.tile.reshape(rp_k_hi_blk, [1, HALF_DIM])
                        rp_k_hi = pl.tile.set_validshape(rp_k_hi_pre, 1, HALF_DIM)
                        rp_klo_cos_pre = pl.tile.col_expand_mul(rp_k_lo, rp_cos_lo)
                        rp_klo_cos = pl.tile.set_validshape(rp_klo_cos_pre, 1, HALF_DIM)
                        rp_khi_sin_pre = pl.tile.col_expand_mul(rp_k_hi, rp_sin_lo)
                        rp_khi_sin = pl.tile.set_validshape(rp_khi_sin_pre, 1, HALF_DIM)
                        rp_k_rot_lo_pre = pl.tile.sub(rp_klo_cos, rp_khi_sin)
                        rp_k_rot_lo = pl.tile.set_validshape(rp_k_rot_lo_pre, 1, HALF_DIM)
                        rp_khi_cos_pre = pl.tile.col_expand_mul(rp_k_hi, rp_cos_hi)
                        rp_khi_cos = pl.tile.set_validshape(rp_khi_cos_pre, 1, HALF_DIM)
                        rp_klo_sin_pre = pl.tile.col_expand_mul(rp_k_lo, rp_sin_hi)
                        rp_klo_sin = pl.tile.set_validshape(rp_klo_sin_pre, 1, HALF_DIM)
                        rp_k_rot_hi_pre = pl.tile.add(rp_khi_cos, rp_klo_sin)
                        rp_k_rot_hi = pl.tile.set_validshape(rp_k_rot_hi_pre, 1, HALF_DIM)
                        rp_k_rot_lo_bf16_pre = pl.tile.cast(rp_k_rot_lo, dtype=pl.BF16)
                        rp_k_rot_lo_bf16 = pl.tile.set_validshape(rp_k_rot_lo_bf16_pre, 1, HALF_DIM)
                        rp_k_rot_hi_bf16_pre = pl.tile.cast(rp_k_rot_hi, dtype=pl.BF16)
                        rp_k_rot_hi_bf16 = pl.tile.set_validshape(rp_k_rot_hi_bf16_pre, 1, HALF_DIM)
                        rp_k_rot_lo_blk = pl.tile.reshape(rp_k_rot_lo_bf16, [1, 1, 1, HALF_DIM])
                        rp_k_rot_hi_blk = pl.tile.reshape(rp_k_rot_hi_bf16, [1, 1, 1, HALF_DIM])
                        pl.tile.store(k_cache, rp_k_rot_lo_blk, [rp_slot_block, ki, rp_slot_offset, 0])
                        pl.tile.store(k_cache, rp_k_rot_hi_blk, [rp_slot_block, ki, rp_slot_offset, HALF_DIM])

                        # V head copy -> v_cache (lo/hi MM_N blocks).
                        rp_v_lo_blk = pl.tile.load(v_proj, [1, 1, 1, HALF_DIM], [rp_kv_blo, 0, b, 0])
                        rp_v_lo_pre = pl.tile.reshape(rp_v_lo_blk, [1, HALF_DIM])
                        rp_v_lo = pl.tile.set_validshape(rp_v_lo_pre, 1, HALF_DIM)
                        rp_v_lo_bf16_pre = pl.tile.cast(rp_v_lo, dtype=pl.BF16)
                        rp_v_lo_bf16 = pl.tile.set_validshape(rp_v_lo_bf16_pre, 1, HALF_DIM)
                        rp_v_lo_out = pl.tile.reshape(rp_v_lo_bf16, [1, 1, 1, HALF_DIM])
                        pl.tile.store(v_cache, rp_v_lo_out, [rp_slot_block, ki, rp_slot_offset, 0])
                        rp_v_hi_blk = pl.tile.load(v_proj, [1, 1, 1, HALF_DIM], [rp_kv_blo + 1, 0, b, 0])
                        rp_v_hi_pre = pl.tile.reshape(rp_v_hi_blk, [1, HALF_DIM])
                        rp_v_hi = pl.tile.set_validshape(rp_v_hi_pre, 1, HALF_DIM)
                        rp_v_hi_bf16_pre = pl.tile.cast(rp_v_hi, dtype=pl.BF16)
                        rp_v_hi_bf16 = pl.tile.set_validshape(rp_v_hi_bf16_pre, 1, HALF_DIM)
                        rp_v_hi_out = pl.tile.reshape(rp_v_hi_bf16, [1, 1, 1, HALF_DIM])
                        pl.tile.store(v_cache, rp_v_hi_out, [rp_slot_block, ki, rp_slot_offset, HALF_DIM])

                        # Q heads RoPE (one row per head) + zero pad -> all_q_padded.
                        rp_q_base = ki * Q_PER_KV
                        rp_pad_idx = b * TOTAL_Q_GROUPS + ki
                        for qi in pl.range(Q_HEAD_BATCH):
                            rp_q_blo = (rp_q_base + qi) * HALVES_PER_HEAD
                            rp_q_lo_blk = pl.tile.load(q_proj_norm, [1, 1, 1, HALF_DIM], [rp_q_blo, 0, b, 0])
                            rp_q_lo_pre = pl.tile.reshape(rp_q_lo_blk, [1, HALF_DIM])
                            rp_q_lo = pl.tile.set_validshape(rp_q_lo_pre, 1, HALF_DIM)
                            rp_q_hi_blk = pl.tile.load(q_proj_norm, [1, 1, 1, HALF_DIM], [rp_q_blo + 1, 0, b, 0])
                            rp_q_hi_pre = pl.tile.reshape(rp_q_hi_blk, [1, HALF_DIM])
                            rp_q_hi = pl.tile.set_validshape(rp_q_hi_pre, 1, HALF_DIM)
                            rp_qlo_cos = pl.tile.col_expand_mul(rp_q_lo, rp_cos_lo)
                            rp_qhi_sin = pl.tile.col_expand_mul(rp_q_hi, rp_sin_lo)
                            rp_q_rot_lo_pre = pl.tile.sub(rp_qlo_cos, rp_qhi_sin)
                            rp_q_rot_lo = pl.tile.set_validshape(rp_q_rot_lo_pre, 1, HALF_DIM)
                            rp_qhi_cos = pl.tile.col_expand_mul(rp_q_hi, rp_cos_hi)
                            rp_qlo_sin = pl.tile.col_expand_mul(rp_q_lo, rp_sin_hi)
                            rp_q_rot_hi_pre = pl.tile.add(rp_qhi_cos, rp_qlo_sin)
                            rp_q_rot_hi = pl.tile.set_validshape(rp_q_rot_hi_pre, 1, HALF_DIM)
                            rp_q_rot_lo_bf16_pre = pl.tile.cast(rp_q_rot_lo, dtype=pl.BF16)
                            rp_q_rot_lo_bf16 = pl.tile.set_validshape(rp_q_rot_lo_bf16_pre, 1, HALF_DIM)
                            rp_q_rot_hi_bf16_pre = pl.tile.cast(rp_q_rot_hi, dtype=pl.BF16)
                            rp_q_rot_hi_bf16 = pl.tile.set_validshape(rp_q_rot_hi_bf16_pre, 1, HALF_DIM)
                            rp_q_rot_lo_blk = pl.tile.reshape(rp_q_rot_lo_bf16, [1, 1, 1, HALF_DIM])
                            rp_q_rot_hi_blk = pl.tile.reshape(rp_q_rot_hi_bf16, [1, 1, 1, HALF_DIM])
                            pl.tile.store(all_q_padded, rp_q_rot_lo_blk, [rp_pad_idx, 0, qi, 0])
                            pl.tile.store(all_q_padded, rp_q_rot_hi_blk, [rp_pad_idx, 0, qi, HALF_DIM])
                        rp_zpad_pre = pl.tile.full([Q_HEAD_PAD, HEAD_DIM], value=0.0, dtype=pl.BF16)
                        rp_zpad = pl.tile.set_validshape(rp_zpad_pre, Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM)
                        rp_zpad_blk = pl.tile.reshape(rp_zpad, [1, 1, Q_HEAD_PAD, HEAD_DIM])
                        pl.tile.store(all_q_padded, rp_zpad_blk, [rp_pad_idx, 0, Q_HEAD_BATCH, 0])

                # =====================================================================
                # 2b. Flash attention, online softmax. Per batch row, per KV head: stream
                #     the KV context in ATT_SEQ-wide steps. QK / SV matmuls are sub-tiled
                #     (QK over head-dim, SV over seq) and the HEAD_DIM output split lo/hi
                #     so every tile stays 4 KB.
                # =====================================================================
                # attn_out blocks mirror q_proj: head h's lo/hi halves at 2h / 2h + 1.
                attn_out = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.BF16)
                for b in pl.range(BATCH):
                    fa_ctx_len = pl.tensor.read(seq_lens, [b, 0, 0, 0])
                    fa_n_steps = (fa_ctx_len + ATT_SEQ - 1) // ATT_SEQ

                    for gi in pl.range(TOTAL_Q_GROUPS):
                        fa_kvh = gi
                        fa_q_base = fa_kvh * Q_HEAD_BATCH
                        fa_pad_idx = b * TOTAL_Q_GROUPS + gi
                        fa_q_pad_blk = pl.tile.load(all_q_padded, [1, 1, Q_HEAD_PAD, HEAD_DIM], [fa_pad_idx, 0, 0, 0])
                        fa_q_padded = pl.tile.reshape(fa_q_pad_blk, [Q_HEAD_PAD, HEAD_DIM])

                        # online accumulators seeded with sentinels mi=-inf, li=0, oi=0
                        fa_mi_pre = pl.tile.full([Q_HEAD_PAD, VEC_W], value=NEG_INF, dtype=pl.FP32)
                        fa_mi = pl.tile.set_validshape(fa_mi_pre, Q_HEAD_PAD, 1)
                        fa_li_pre = pl.tile.full([Q_HEAD_PAD, VEC_W], value=0.0, dtype=pl.FP32)
                        fa_li = pl.tile.set_validshape(fa_li_pre, Q_HEAD_PAD, 1)
                        fa_oi_lo = pl.tile.full([Q_HEAD_PAD, HALF_DIM], value=0.0, dtype=pl.FP32)
                        fa_oi_hi = pl.tile.full([Q_HEAD_PAD, HALF_DIM], value=0.0, dtype=pl.FP32)

                        for st in pl.range(fa_n_steps):
                            fa_g0 = st * ATT_SEQ
                            fa_sb = fa_g0 // BLOCK_SIZE
                            fa_in_block = fa_g0 - fa_sb * BLOCK_SIZE
                            fa_pbid_i32 = pl.tensor.read(block_table, [b, fa_sb, 0, 0])
                            fa_pbid = pl.cast(fa_pbid_i32, pl.INDEX)
                            fa_valid_seq = pl.min(ATT_SEQ, fa_ctx_len - fa_g0)

                            # --- QK matmul: scores[Q_HEAD_PAD, ATT_SEQ] over head-dim chunks ---
                            fa_q_sub0_pre = pl.tile.slice(fa_q_padded, [Q_HEAD_PAD, QK_KD], [0, 0])
                            fa_q_sub0 = pl.tile.set_validshape(fa_q_sub0_pre, Q_HEAD_PAD, QK_KD)
                            fa_k_sub0_blk = pl.tile.load(k_cache, [1, 1, ATT_SEQ, QK_KD], [fa_pbid, fa_kvh, fa_in_block, 0])
                            fa_k_sub0 = pl.tile.reshape(fa_k_sub0_blk, [ATT_SEQ, QK_KD])
                            fa_scores = pl.tile.matmul(fa_q_sub0, fa_k_sub0, b_trans=True, out_dtype=pl.FP32)
                            for kd in pl.range(1, QK_KSTEPS):
                                fa_kd0 = kd * QK_KD
                                fa_q_sub_pre = pl.tile.slice(fa_q_padded, [Q_HEAD_PAD, QK_KD], [0, fa_kd0])
                                fa_q_sub = pl.tile.set_validshape(fa_q_sub_pre, Q_HEAD_PAD, QK_KD)
                                fa_k_sub_blk = pl.tile.load(k_cache, [1, 1, ATT_SEQ, QK_KD], [fa_pbid, fa_kvh, fa_in_block, fa_kd0])
                                fa_k_sub = pl.tile.reshape(fa_k_sub_blk, [ATT_SEQ, QK_KD])
                                fa_scores = pl.tile.matmul_acc(fa_scores, fa_q_sub, fa_k_sub)

                            # --- tail-masked softmax (vec) ---
                            fa_scores_scaled = pl.tile.mul(fa_scores, ATTN_SCALE)
                            fa_scores_valid = pl.tile.set_validshape(fa_scores_scaled, Q_HEAD_PAD, fa_valid_seq)
                            fa_scores_pad = pl.tile.fillpad(fa_scores_valid, pad_value=pl.PadValue.min)
                            fa_cur_mi_pre = pl.tile.row_max(fa_scores_pad)
                            fa_cur_mi = pl.tile.set_validshape(fa_cur_mi_pre, Q_HEAD_PAD, 1)
                            fa_shifted = pl.tile.row_expand_sub(fa_scores_pad, fa_cur_mi)
                            fa_exp_scores = pl.tile.exp(fa_shifted)
                            fa_exp_bf16_pre = pl.tile.cast(fa_exp_scores, dtype=pl.BF16)
                            fa_exp_bf16 = pl.tile.set_validshape(fa_exp_bf16_pre, Q_HEAD_PAD, ATT_SEQ)
                            fa_exp_fp32 = pl.tile.cast(fa_exp_bf16, dtype=pl.FP32)
                            fa_cur_li_pre = pl.tile.row_sum(fa_exp_fp32)
                            fa_cur_li = pl.tile.set_validshape(fa_cur_li_pre, Q_HEAD_PAD, 1)

                            # --- SV matmul: oi halves over the SV_SSTEPS == 2 seq chunks ---
                            fa_exp_sub0_pre = pl.tile.slice(fa_exp_bf16, [Q_HEAD_PAD, SV_SEQ], [0, 0])
                            fa_exp_sub0 = pl.tile.set_validshape(fa_exp_sub0_pre, Q_HEAD_PAD, SV_SEQ)
                            fa_v_lo0_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [fa_pbid, fa_kvh, fa_in_block, 0])
                            fa_v_lo0 = pl.tile.reshape(fa_v_lo0_blk, [SV_SEQ, HALF_DIM])
                            fa_v_hi0_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [fa_pbid, fa_kvh, fa_in_block, HALF_DIM])
                            fa_v_hi0 = pl.tile.reshape(fa_v_hi0_blk, [SV_SEQ, HALF_DIM])
                            fa_oi_lo_tmp = pl.tile.matmul(fa_exp_sub0, fa_v_lo0, out_dtype=pl.FP32)
                            fa_oi_hi_tmp = pl.tile.matmul(fa_exp_sub0, fa_v_hi0, out_dtype=pl.FP32)
                            fa_exp_sub1_pre = pl.tile.slice(fa_exp_bf16, [Q_HEAD_PAD, SV_SEQ], [0, SV_SEQ])
                            fa_exp_sub1 = pl.tile.set_validshape(fa_exp_sub1_pre, Q_HEAD_PAD, SV_SEQ)
                            fa_in_block1 = fa_in_block + SV_SEQ
                            fa_v_lo1_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [fa_pbid, fa_kvh, fa_in_block1, 0])
                            fa_v_lo1 = pl.tile.reshape(fa_v_lo1_blk, [SV_SEQ, HALF_DIM])
                            fa_v_hi1_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [fa_pbid, fa_kvh, fa_in_block1, HALF_DIM])
                            fa_v_hi1 = pl.tile.reshape(fa_v_hi1_blk, [SV_SEQ, HALF_DIM])
                            fa_oi_lo_tmp = pl.tile.matmul_acc(fa_oi_lo_tmp, fa_exp_sub1, fa_v_lo1)
                            fa_oi_hi_tmp = pl.tile.matmul_acc(fa_oi_hi_tmp, fa_exp_sub1, fa_v_hi1)

                            # --- online-softmax recurrence (UB accumulators) ---
                            fa_mi_new_pre = pl.tile.maximum(fa_mi, fa_cur_mi)
                            fa_mi_new = pl.tile.set_validshape(fa_mi_new_pre, Q_HEAD_PAD, 1)
                            fa_mdiff_pre = pl.tile.sub(fa_mi, fa_mi_new)
                            fa_mdiff = pl.tile.set_validshape(fa_mdiff_pre, Q_HEAD_PAD, 1)
                            fa_alpha_pre = pl.tile.exp(fa_mdiff)
                            fa_alpha = pl.tile.set_validshape(fa_alpha_pre, Q_HEAD_PAD, 1)
                            fa_cdiff_pre = pl.tile.sub(fa_cur_mi, fa_mi_new)
                            fa_cdiff = pl.tile.set_validshape(fa_cdiff_pre, Q_HEAD_PAD, 1)
                            fa_beta_pre = pl.tile.exp(fa_cdiff)
                            fa_beta = pl.tile.set_validshape(fa_beta_pre, Q_HEAD_PAD, 1)
                            fa_li_a_pre = pl.tile.mul(fa_alpha, fa_li)
                            fa_li_a = pl.tile.set_validshape(fa_li_a_pre, Q_HEAD_PAD, 1)
                            fa_li_b_pre = pl.tile.mul(fa_beta, fa_cur_li)
                            fa_li_b = pl.tile.set_validshape(fa_li_b_pre, Q_HEAD_PAD, 1)
                            fa_li_acc = pl.tile.add(fa_li_a, fa_li_b)
                            fa_li = pl.tile.set_validshape(fa_li_acc, Q_HEAD_PAD, 1)
                            fa_oi_lo_a = pl.tile.row_expand_mul(fa_oi_lo, fa_alpha)
                            fa_oi_lo_b = pl.tile.row_expand_mul(fa_oi_lo_tmp, fa_beta)
                            fa_oi_lo = pl.tile.add(fa_oi_lo_a, fa_oi_lo_b)
                            fa_oi_hi_a = pl.tile.row_expand_mul(fa_oi_hi, fa_alpha)
                            fa_oi_hi_b = pl.tile.row_expand_mul(fa_oi_hi_tmp, fa_beta)
                            fa_oi_hi = pl.tile.add(fa_oi_hi_a, fa_oi_hi_b)
                            fa_mi = fa_mi_new

                        # ctx = oi / li, trim Q_HEAD_PAD -> Q_HEAD_BATCH rows; per head row
                        # qi, the lo/hi halves land in attn_out blocks 2(q_base+qi) / +1.
                        fa_ctx_lo = pl.tile.row_expand_div(fa_oi_lo, fa_li)
                        fa_ctx_hi = pl.tile.row_expand_div(fa_oi_hi, fa_li)
                        for qi in pl.range(Q_HEAD_BATCH):
                            fa_h_blo = (fa_q_base + qi) * HALVES_PER_HEAD
                            fa_lo1_pre = pl.tile.slice(fa_ctx_lo, [1, HALF_DIM], [qi, 0])
                            fa_lo1 = pl.tile.set_validshape(fa_lo1_pre, 1, HALF_DIM)
                            fa_lo1_bf16_pre = pl.tile.cast(fa_lo1, dtype=pl.BF16)
                            fa_lo1_bf16 = pl.tile.set_validshape(fa_lo1_bf16_pre, 1, HALF_DIM)
                            fa_lo1_blk = pl.tile.reshape(fa_lo1_bf16, [1, 1, 1, HALF_DIM])
                            pl.tile.store(attn_out, fa_lo1_blk, [fa_h_blo, 0, b, 0])
                            fa_hi1_pre = pl.tile.slice(fa_ctx_hi, [1, HALF_DIM], [qi, 0])
                            fa_hi1 = pl.tile.set_validshape(fa_hi1_pre, 1, HALF_DIM)
                            fa_hi1_bf16_pre = pl.tile.cast(fa_hi1, dtype=pl.BF16)
                            fa_hi1_bf16 = pl.tile.set_validshape(fa_hi1_bf16_pre, 1, HALF_DIM)
                            fa_hi1_blk = pl.tile.reshape(fa_hi1_bf16, [1, 1, 1, HALF_DIM])
                            pl.tile.store(attn_out, fa_hi1_blk, [fa_h_blo + 1, 0, b, 0])

                # =====================================================================
                # 3. Output projection + residual -> post-RMSNorm -> MLP -> residual.
                #    An MM_K activation chunk kb sits in MM_N block kb // 2, half kb % 2;
                #    an MM_N residual chunk sits in hidden block nb // 2, half nb % 2.
                # =====================================================================
                # --- out-proj + residual: resid1 = attn_out @ wo + current_hidden ---
                resid1 = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
                for nb in pl.range(HIDDEN // MM_N):
                    op_a0_blk = pl.tile.load(attn_out, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
                    op_a0_pre = pl.tile.reshape(op_a0_blk, [BATCH, MM_K])
                    op_a0 = pl.tile.set_validshape(op_a0_pre, BATCH, MM_K)
                    op_w0_blk = pl.tile.load(wo, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
                    op_w0 = pl.tile.reshape(op_w0_blk, [MM_K, MM_N])
                    op_acc = pl.tile.matmul(op_a0, op_w0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // MM_K):
                        op_a_blk = pl.tile.load(attn_out, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
                        op_a_pre = pl.tile.reshape(op_a_blk, [BATCH, MM_K])
                        op_a = pl.tile.set_validshape(op_a_pre, BATCH, MM_K)
                        op_w_blk = pl.tile.load(wo, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
                        op_w = pl.tile.reshape(op_w_blk, [MM_K, MM_N])
                        op_acc = pl.tile.matmul_acc(op_acc, op_a, op_w)
                    op_resid_blk = pl.tile.load(current_hidden, [1, 1, BATCH, MM_N], [nb // 2, 0, 0, (nb % 2) * MM_N])
                    op_resid_bf16_pre = pl.tile.reshape(op_resid_blk, [BATCH, MM_N])
                    op_resid_bf16 = pl.tile.set_validshape(op_resid_bf16_pre, BATCH, MM_N)
                    op_resid = pl.tile.cast(op_resid_bf16, dtype=pl.FP32)
                    op_out_sum = pl.tile.add(op_acc, op_resid)
                    op_out_blk = pl.tile.reshape(op_out_sum, [1, 1, BATCH, MM_N])
                    pl.tile.store(resid1, op_out_blk, [nb, 0, 0, 0])

                # --- post-attention RMSNorm: post_norm = (resid1 / rms) * post_gamma ---
                s3_sumsq_pre = pl.tile.full([BATCH, VEC_W], value=0.0, dtype=pl.FP32)
                s3_sumsq = pl.tile.set_validshape(s3_sumsq_pre, BATCH, 1)
                for kb in pl.range(HIDDEN // MM_N):
                    s3_x_blk = pl.tile.load(resid1, [1, 1, BATCH, VEC_W], [kb, 0, 0, 0])
                    s3_x = pl.tile.reshape(s3_x_blk, [BATCH, VEC_W])
                    s3_sq = pl.tile.mul(s3_x, s3_x)
                    s3_part_pre = pl.tile.row_sum(s3_sq)
                    s3_part = pl.tile.set_validshape(s3_part_pre, BATCH, 1)
                    s3_sumsq_acc = pl.tile.add(s3_sumsq, s3_part)
                    s3_sumsq = pl.tile.set_validshape(s3_sumsq_acc, BATCH, 1)

                s3_mean_sq_pre = pl.tile.mul(s3_sumsq, HIDDEN_INV)
                s3_mean_sq = pl.tile.set_validshape(s3_mean_sq_pre, BATCH, 1)
                s3_variance_pre = pl.tile.add(s3_mean_sq, EPS)
                s3_variance = pl.tile.set_validshape(s3_variance_pre, BATCH, 1)
                s3_rms_pre = pl.tile.sqrt(s3_variance)
                s3_rms = pl.tile.set_validshape(s3_rms_pre, BATCH, 1)
                s3_inv_rms_pre = pl.tile.recip(s3_rms)
                s3_inv_rms = pl.tile.set_validshape(s3_inv_rms_pre, BATCH, 1)

                # post_gamma block kb // 2 holds the two VEC_W halves of each VEC_BF16 chunk
                post_norm = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, VEC_W], dtype=pl.BF16)
                for kb in pl.range(HIDDEN // MM_N):
                    pn_x_blk = pl.tile.load(resid1, [1, 1, BATCH, VEC_W], [kb, 0, 0, 0])
                    pn_x = pl.tile.reshape(pn_x_blk, [BATCH, VEC_W])
                    pn_gamma_blk = pl.tile.load(post_rms_weight, [1, 1, 1, VEC_W], [kb // 2, 0, 0, (kb % 2) * VEC_W])
                    pn_gamma_pre = pl.tile.reshape(pn_gamma_blk, [1, VEC_W])
                    pn_gamma = pl.tile.set_validshape(pn_gamma_pre, 1, VEC_W)
                    pn_x_scaled = pl.tile.row_expand_mul(pn_x, s3_inv_rms)
                    pn_normed = pl.tile.col_expand_mul(pn_x_scaled, pn_gamma)
                    pn_normed_bf16_pre = pl.tile.cast(pn_normed, dtype=pl.BF16)
                    pn_normed_bf16 = pl.tile.set_validshape(pn_normed_bf16_pre, BATCH, VEC_W)
                    pn_normed_blk = pl.tile.reshape(pn_normed_bf16, [1, 1, BATCH, VEC_W])
                    pl.tile.store(post_norm, pn_normed_blk, [kb, 0, 0, 0])

                # --- MLP gate/up + SiLU: mlp = (silu(post_norm @ w_gate)) * (post_norm @ w_up) ---
                # gate and up share one K-loop over the post_norm activation tiles.
                mlp = pl.create_tensor([INTERMEDIATE // MM_N, 1, BATCH, MM_N], dtype=pl.BF16)
                for nb in pl.range(INTERMEDIATE // MM_N):
                    ml_p0_blk = pl.tile.load(post_norm, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
                    ml_p0_pre = pl.tile.reshape(ml_p0_blk, [BATCH, MM_K])
                    ml_p0 = pl.tile.set_validshape(ml_p0_pre, BATCH, MM_K)
                    ml_wg0_blk = pl.tile.load(w_gate, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
                    ml_wg0 = pl.tile.reshape(ml_wg0_blk, [MM_K, MM_N])
                    ml_wu0_blk = pl.tile.load(w_up, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
                    ml_wu0 = pl.tile.reshape(ml_wu0_blk, [MM_K, MM_N])
                    ml_gate_acc = pl.tile.matmul(ml_p0, ml_wg0, out_dtype=pl.FP32)
                    ml_up_acc = pl.tile.matmul(ml_p0, ml_wu0, out_dtype=pl.FP32)
                    for kb in pl.range(1, HIDDEN // MM_K):
                        ml_p_blk = pl.tile.load(post_norm, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
                        ml_p_pre = pl.tile.reshape(ml_p_blk, [BATCH, MM_K])
                        ml_p = pl.tile.set_validshape(ml_p_pre, BATCH, MM_K)
                        ml_wg_blk = pl.tile.load(w_gate, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
                        ml_wg = pl.tile.reshape(ml_wg_blk, [MM_K, MM_N])
                        ml_wu_blk = pl.tile.load(w_up, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
                        ml_wu = pl.tile.reshape(ml_wu_blk, [MM_K, MM_N])
                        ml_gate_acc = pl.tile.matmul_acc(ml_gate_acc, ml_p, ml_wg)
                        ml_up_acc = pl.tile.matmul_acc(ml_up_acc, ml_p, ml_wu)
                    # SiLU(gate) * up = (gate * sigmoid(gate)) * up
                    ml_neg_gate = pl.tile.neg(ml_gate_acc)
                    ml_exp_gate = pl.tile.exp(ml_neg_gate)
                    ml_denom = pl.tile.add(ml_exp_gate, 1.0)
                    ml_sigmoid = pl.tile.recip(ml_denom)
                    ml_gate_sig = pl.tile.mul(ml_gate_acc, ml_sigmoid)
                    ml_mlp_chunk = pl.tile.mul(ml_gate_sig, ml_up_acc)
                    ml_mlp_bf16_pre = pl.tile.cast(ml_mlp_chunk, dtype=pl.BF16)
                    ml_mlp_bf16 = pl.tile.set_validshape(ml_mlp_bf16_pre, BATCH, MM_N)
                    ml_mlp_blk = pl.tile.reshape(ml_mlp_bf16, [1, 1, BATCH, MM_N])
                    pl.tile.store(mlp, ml_mlp_blk, [nb, 0, 0, 0])

                # --- down-proj + residual: next_hidden = mlp @ w_down + resid1 ---
                for nb in pl.range(HIDDEN // MM_N):
                    dn_m0_blk = pl.tile.load(mlp, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
                    dn_m0_pre = pl.tile.reshape(dn_m0_blk, [BATCH, MM_K])
                    dn_m0 = pl.tile.set_validshape(dn_m0_pre, BATCH, MM_K)
                    dn_wd0_blk = pl.tile.load(w_down, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
                    dn_wd0 = pl.tile.reshape(dn_wd0_blk, [MM_K, MM_N])
                    dn_acc = pl.tile.matmul(dn_m0, dn_wd0, out_dtype=pl.FP32)
                    for kb in pl.range(1, INTERMEDIATE // MM_K):
                        dn_m_blk = pl.tile.load(mlp, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
                        dn_m_pre = pl.tile.reshape(dn_m_blk, [BATCH, MM_K])
                        dn_m = pl.tile.set_validshape(dn_m_pre, BATCH, MM_K)
                        dn_wd_blk = pl.tile.load(w_down, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
                        dn_wd = pl.tile.reshape(dn_wd_blk, [MM_K, MM_N])
                        dn_acc = pl.tile.matmul_acc(dn_acc, dn_m, dn_wd)
                    dn_resid_blk = pl.tile.load(resid1, [1, 1, BATCH, MM_N], [nb, 0, 0, 0])
                    dn_resid = pl.tile.reshape(dn_resid_blk, [BATCH, MM_N])
                    dn_out_sum = pl.tile.add(dn_acc, dn_resid)
                    dn_out_bf16_pre = pl.tile.cast(dn_out_sum, dtype=pl.BF16)
                    dn_out_bf16 = pl.tile.set_validshape(dn_out_bf16_pre, BATCH, MM_N)
                    dn_out_blk = pl.tile.reshape(dn_out_bf16, [1, 1, BATCH, MM_N])
                    pl.tile.store(next_hidden, dn_out_blk, [nb, 0, 0, 0])

            return next_hidden

    return Qwen3Decode14B


def compile_superscalar_npu(dump_dir: str = "build_output/qwen3_14b_decode_ssn"):
    """Lower the decode program through the SuperscalarNPU (PR #1680) pass
    pipeline and dump the TREG-allocated IR. The backend is IR-only (no
    codegen), so this stops after AllocateMemoryAddr -- no golden, no device
    run. Returns the path of the printed IR."""
    import os

    from pypto import backend, passes
    from pypto.ir.pass_manager import OptimizationStrategy, PassManager
    from pypto.ir.printer import python_print

    os.makedirs(dump_dir, exist_ok=True)

    # SuperscalarNPU SoC: DDR + 1 MB TREG register file (256 x 4 KB blocks).
    backend.reset_for_testing()
    backend.set_backend_type(backend.BackendType.SuperscalarNPU)

    program = build_qwen3_14b_decode_program()
    pm = PassManager.get_strategy(OptimizationStrategy.SuperscalarNPU)
    # Codegen is unimplemented and the truncated pipeline does not establish
    # every property a full Ascend run would, so verification is disabled
    # (mirrors tests/ut/ir/transforms/test_superscalar_npu_treg.py).
    with passes.PassContext([], passes.VerificationLevel.NONE):
        lowered = pm.run_passes(program, dump_ir=True, output_dir=dump_dir)

    text = python_print(lowered, prefix="pl")
    ir_path = os.path.join(dump_dir, "qwen3_14b_decode_ssn.lowered.py")
    with open(ir_path, "w") as f:
        f.write(text)
    return ir_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        default="build_output/qwen3_14b_decode_ssn",
        help="Output dir for per-pass IR dumps and the final TREG-allocated IR.",
    )
    args = parser.parse_args()

    ir_path = compile_superscalar_npu(args.dump_dir)
    print(f"[ssn] SuperscalarNPU TREG IR written to {ir_path}")
