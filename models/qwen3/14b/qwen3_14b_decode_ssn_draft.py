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
  1. One plain function; control flow is only pl.range and if/else
     (no jit / program / at / spmd / parallel / pipeline).
  2. Compute is tile-level (pl.tile.*); the only tensor-level op is
     pl.tensor.read (host scalars for loop bounds / paged addressing).
  3. Every tile's static shape is 4 KB (FP32 1024 / BF16 2048 elems);
     a tile whose live region is smaller declares it via set_validshape.
  4. SSA names: only set_validshape may reuse a tile name; full 4 KB
     loop-carried accumulators (matmul_acc, online-softmax oi) rebind.

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


@pl.kernel
def qwen3_14b_decode(
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
):
    # =====================================================================
    # 1. Input RMSNorm:  normed_all = (x / rms(x)) * gamma
    # =====================================================================
    # sum of squares over HIDDEN (BF16 block load -> two FP32 halves -> square)
    sumsq = pl.tile.full([BATCH, VEC_W], value=0.0, dtype=pl.FP32)
    sumsq = pl.tile.set_validshape(sumsq, [BATCH, 1])
    for kb in pl.range(HIDDEN // VEC_BF16):
        x_blk = pl.tile.load(current_hidden, [1, 1, BATCH, VEC_BF16], [kb, 0, 0, 0])
        x_bf16 = pl.tile.reshape(x_blk, [BATCH, VEC_BF16])
        for h in pl.range(2):
            h0 = h * VEC_W
            x_half = pl.tile.slice(x_bf16, [BATCH, VEC_W], [0, h0])
            x_half = pl.tile.set_validshape(x_half, [BATCH, VEC_W])
            x = pl.tile.cast(x_half, dtype=pl.FP32)
            sq = pl.tile.mul(x, x)
            part = pl.tile.row_sum(sq)
            part = pl.tile.set_validshape(part, [BATCH, 1])
            sumsq_acc = pl.tile.add(sumsq, part)
            sumsq = pl.tile.set_validshape(sumsq_acc, [BATCH, 1])

    mean_sq = pl.tile.mul(sumsq, HIDDEN_INV)
    mean_sq = pl.tile.set_validshape(mean_sq, [BATCH, 1])
    variance = pl.tile.add(mean_sq, EPS)
    variance = pl.tile.set_validshape(variance, [BATCH, 1])
    rms = pl.tile.sqrt(variance)
    rms = pl.tile.set_validshape(rms, [BATCH, 1])
    inv_rms = pl.tile.recip(rms)
    inv_rms = pl.tile.set_validshape(inv_rms, [BATCH, 1])

    # normalize + scale by gamma -> BF16 bridge buffer (VEC_W column blocks)
    normed_all = pl.create_tensor([HIDDEN // VEC_W, 1, BATCH, VEC_W], dtype=pl.BF16)
    for kb in pl.range(HIDDEN // VEC_BF16):
        x_blk = pl.tile.load(current_hidden, [1, 1, BATCH, VEC_BF16], [kb, 0, 0, 0])
        x_bf16 = pl.tile.reshape(x_blk, [BATCH, VEC_BF16])
        for h in pl.range(2):
            h0 = h * VEC_W
            x_half = pl.tile.slice(x_bf16, [BATCH, VEC_W], [0, h0])
            x_half = pl.tile.set_validshape(x_half, [BATCH, VEC_W])
            x = pl.tile.cast(x_half, dtype=pl.FP32)
            gamma_blk = pl.tile.load(input_rms_weight, [1, 1, 1, VEC_W], [kb, 0, 0, h0])
            gamma = pl.tile.reshape(gamma_blk, [1, VEC_W])
            gamma = pl.tile.set_validshape(gamma, [1, VEC_W])
            x_scaled = pl.tile.row_expand_mul(x, inv_rms)
            normed = pl.tile.col_expand_mul(x_scaled, gamma)
            normed_bf16 = pl.tile.cast(normed, dtype=pl.BF16)
            normed_bf16 = pl.tile.set_validshape(normed_bf16, [BATCH, VEC_W])
            normed_blk = pl.tile.reshape(normed_bf16, [1, 1, BATCH, VEC_W])
            pl.tile.store(normed_all, normed_blk, [kb * 2 + h, 0, 0, 0])

    # =====================================================================
    # 2. Q / K / V projection:  proj = normed_all @ W
    #    (peeled-first matmul + matmul_acc over the K tiles, per output tile)
    #    An MM_K activation chunk kb sits in VEC_W block kb // 2, half kb % 2.
    # =====================================================================
    # --- Q projection: [BATCH, HIDDEN] @ [HIDDEN, HIDDEN] ---
    q_proj = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
    for nb in pl.range(HIDDEN // MM_N):
        a0_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
        a0 = pl.tile.reshape(a0_blk, [BATCH, MM_K])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0_blk = pl.tile.load(wq, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
        w0 = pl.tile.reshape(w0_blk, [MM_K, MM_N])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            a_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
            a = pl.tile.reshape(a_blk, [BATCH, MM_K])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w_blk = pl.tile.load(wq, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
            w = pl.tile.reshape(w_blk, [MM_K, MM_N])
            acc = pl.tile.matmul_acc(acc, a, w)
        acc_blk = pl.tile.reshape(acc, [1, 1, BATCH, MM_N])
        pl.tile.store(q_proj, acc_blk, [nb, 0, 0, 0])

    # --- K projection: [BATCH, HIDDEN] @ [HIDDEN, KV_HIDDEN] ---
    k_proj = pl.create_tensor([KV_HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
    for nb in pl.range(KV_HIDDEN // MM_N):
        a0_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
        a0 = pl.tile.reshape(a0_blk, [BATCH, MM_K])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0_blk = pl.tile.load(wk, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
        w0 = pl.tile.reshape(w0_blk, [MM_K, MM_N])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            a_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
            a = pl.tile.reshape(a_blk, [BATCH, MM_K])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w_blk = pl.tile.load(wk, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
            w = pl.tile.reshape(w_blk, [MM_K, MM_N])
            acc = pl.tile.matmul_acc(acc, a, w)
        acc_blk = pl.tile.reshape(acc, [1, 1, BATCH, MM_N])
        pl.tile.store(k_proj, acc_blk, [nb, 0, 0, 0])

    # --- V projection: [BATCH, HIDDEN] @ [HIDDEN, KV_HIDDEN] ---
    v_proj = pl.create_tensor([KV_HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
    for nb in pl.range(KV_HIDDEN // MM_N):
        a0_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
        a0 = pl.tile.reshape(a0_blk, [BATCH, MM_K])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0_blk = pl.tile.load(wv, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
        w0 = pl.tile.reshape(w0_blk, [MM_K, MM_N])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            a_blk = pl.tile.load(normed_all, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
            a = pl.tile.reshape(a_blk, [BATCH, MM_K])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w_blk = pl.tile.load(wv, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
            w = pl.tile.reshape(w_blk, [MM_K, MM_N])
            acc = pl.tile.matmul_acc(acc, a, w)
        acc_blk = pl.tile.reshape(acc, [1, 1, BATCH, MM_N])
        pl.tile.store(v_proj, acc_blk, [nb, 0, 0, 0])

    # =====================================================================
    # 3. Per-head q_norm / k_norm. A head's HEAD_DIM spans two consecutive
    #    MM_N blocks (lo/hi halves, batch rows): row-wise RMSNorm per head.
    # =====================================================================
    q_proj_norm = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
    for hq in pl.range(NUM_HEADS):
        b_lo = hq * HALVES_PER_HEAD
        x_lo_blk = pl.tile.load(q_proj, [1, 1, BATCH, VEC_W], [b_lo, 0, 0, 0])
        x_lo = pl.tile.reshape(x_lo_blk, [BATCH, VEC_W])
        x_hi_blk = pl.tile.load(q_proj, [1, 1, BATCH, VEC_W], [b_lo + 1, 0, 0, 0])
        x_hi = pl.tile.reshape(x_hi_blk, [BATCH, VEC_W])
        sq_lo = pl.tile.mul(x_lo, x_lo)
        sumsq = pl.tile.row_sum(sq_lo)
        sumsq = pl.tile.set_validshape(sumsq, [BATCH, 1])
        sq_hi = pl.tile.mul(x_hi, x_hi)
        part = pl.tile.row_sum(sq_hi)
        part = pl.tile.set_validshape(part, [BATCH, 1])
        sumsq_acc = pl.tile.add(sumsq, part)
        sumsq = pl.tile.set_validshape(sumsq_acc, [BATCH, 1])
        mean_sq = pl.tile.mul(sumsq, HEAD_DIM_INV)
        mean_sq = pl.tile.set_validshape(mean_sq, [BATCH, 1])
        variance = pl.tile.add(mean_sq, EPS)
        variance = pl.tile.set_validshape(variance, [BATCH, 1])
        inv_rms = pl.tile.rsqrt(variance)
        inv_rms = pl.tile.set_validshape(inv_rms, [BATCH, 1])
        g_lo_blk = pl.tile.load(q_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, 0])
        g_lo = pl.tile.reshape(g_lo_blk, [1, HALF_DIM])
        g_lo = pl.tile.set_validshape(g_lo, [1, HALF_DIM])
        g_hi_blk = pl.tile.load(q_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, HALF_DIM])
        g_hi = pl.tile.reshape(g_hi_blk, [1, HALF_DIM])
        g_hi = pl.tile.set_validshape(g_hi, [1, HALF_DIM])
        x_lo_scaled = pl.tile.row_expand_mul(x_lo, inv_rms)
        n_lo = pl.tile.col_expand_mul(x_lo_scaled, g_lo)
        x_hi_scaled = pl.tile.row_expand_mul(x_hi, inv_rms)
        n_hi = pl.tile.col_expand_mul(x_hi_scaled, g_hi)
        n_lo_blk = pl.tile.reshape(n_lo, [1, 1, BATCH, VEC_W])
        n_hi_blk = pl.tile.reshape(n_hi, [1, 1, BATCH, VEC_W])
        pl.tile.store(q_proj_norm, n_lo_blk, [b_lo, 0, 0, 0])
        pl.tile.store(q_proj_norm, n_hi_blk, [b_lo + 1, 0, 0, 0])

    k_proj_norm = pl.create_tensor([KV_HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
    for hk in pl.range(NUM_KV_HEADS):
        b_lo = hk * HALVES_PER_HEAD
        x_lo_blk = pl.tile.load(k_proj, [1, 1, BATCH, VEC_W], [b_lo, 0, 0, 0])
        x_lo = pl.tile.reshape(x_lo_blk, [BATCH, VEC_W])
        x_hi_blk = pl.tile.load(k_proj, [1, 1, BATCH, VEC_W], [b_lo + 1, 0, 0, 0])
        x_hi = pl.tile.reshape(x_hi_blk, [BATCH, VEC_W])
        sq_lo = pl.tile.mul(x_lo, x_lo)
        sumsq = pl.tile.row_sum(sq_lo)
        sumsq = pl.tile.set_validshape(sumsq, [BATCH, 1])
        sq_hi = pl.tile.mul(x_hi, x_hi)
        part = pl.tile.row_sum(sq_hi)
        part = pl.tile.set_validshape(part, [BATCH, 1])
        sumsq_acc = pl.tile.add(sumsq, part)
        sumsq = pl.tile.set_validshape(sumsq_acc, [BATCH, 1])
        mean_sq = pl.tile.mul(sumsq, HEAD_DIM_INV)
        mean_sq = pl.tile.set_validshape(mean_sq, [BATCH, 1])
        variance = pl.tile.add(mean_sq, EPS)
        variance = pl.tile.set_validshape(variance, [BATCH, 1])
        inv_rms = pl.tile.rsqrt(variance)
        inv_rms = pl.tile.set_validshape(inv_rms, [BATCH, 1])
        g_lo_blk = pl.tile.load(k_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, 0])
        g_lo = pl.tile.reshape(g_lo_blk, [1, HALF_DIM])
        g_lo = pl.tile.set_validshape(g_lo, [1, HALF_DIM])
        g_hi_blk = pl.tile.load(k_norm_weight, [1, 1, 1, HALF_DIM], [0, 0, 0, HALF_DIM])
        g_hi = pl.tile.reshape(g_hi_blk, [1, HALF_DIM])
        g_hi = pl.tile.set_validshape(g_hi, [1, HALF_DIM])
        x_lo_scaled = pl.tile.row_expand_mul(x_lo, inv_rms)
        n_lo = pl.tile.col_expand_mul(x_lo_scaled, g_lo)
        x_hi_scaled = pl.tile.row_expand_mul(x_hi, inv_rms)
        n_hi = pl.tile.col_expand_mul(x_hi_scaled, g_hi)
        n_lo_blk = pl.tile.reshape(n_lo, [1, 1, BATCH, VEC_W])
        n_hi_blk = pl.tile.reshape(n_hi, [1, 1, BATCH, VEC_W])
        pl.tile.store(k_proj_norm, n_lo_blk, [b_lo, 0, 0, 0])
        pl.tile.store(k_proj_norm, n_hi_blk, [b_lo + 1, 0, 0, 0])

    # =====================================================================
    # 2a. RoPE + paged KV-cache write. Per batch row, per KV head: rotate-half
    #     K -> k_cache, copy V -> v_cache, rotate the Q_HEAD_BATCH Q heads and
    #     zero-pad to Q_HEAD_PAD rows -> all_q_padded.
    # =====================================================================
    all_q_padded = pl.create_tensor([BATCH * TOTAL_Q_GROUPS, 1, Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16)
    for b in pl.range(BATCH):
        ctx_len = pl.tensor.read(seq_lens, [b, 0, 0, 0])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b, 0, 0, 0])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE

        cos_blk = pl.tile.load(rope_cos, [1, 1, 1, HEAD_DIM], [pos, 0, 0, 0])
        cos_row = pl.tile.reshape(cos_blk, [1, HEAD_DIM])
        cos_row = pl.tile.set_validshape(cos_row, [1, HEAD_DIM])
        sin_blk = pl.tile.load(rope_sin, [1, 1, 1, HEAD_DIM], [pos, 0, 0, 0])
        sin_row = pl.tile.reshape(sin_blk, [1, HEAD_DIM])
        sin_row = pl.tile.set_validshape(sin_row, [1, HEAD_DIM])
        cos_lo = pl.tile.slice(cos_row, [1, HALF_DIM], [0, 0])
        cos_lo = pl.tile.set_validshape(cos_lo, [1, HALF_DIM])
        cos_hi = pl.tile.slice(cos_row, [1, HALF_DIM], [0, HALF_DIM])
        cos_hi = pl.tile.set_validshape(cos_hi, [1, HALF_DIM])
        sin_lo = pl.tile.slice(sin_row, [1, HALF_DIM], [0, 0])
        sin_lo = pl.tile.set_validshape(sin_lo, [1, HALF_DIM])
        sin_hi = pl.tile.slice(sin_row, [1, HALF_DIM], [0, HALF_DIM])
        sin_hi = pl.tile.set_validshape(sin_hi, [1, HALF_DIM])

        for ki in pl.range(NUM_KV_HEADS):
            kv_blo = ki * HALVES_PER_HEAD

            # K head RoPE -> k_cache (rotate-half).
            k_lo_blk = pl.tile.load(k_proj_norm, [1, 1, 1, HALF_DIM], [kv_blo, 0, b, 0])
            k_lo = pl.tile.reshape(k_lo_blk, [1, HALF_DIM])
            k_lo = pl.tile.set_validshape(k_lo, [1, HALF_DIM])
            k_hi_blk = pl.tile.load(k_proj_norm, [1, 1, 1, HALF_DIM], [kv_blo + 1, 0, b, 0])
            k_hi = pl.tile.reshape(k_hi_blk, [1, HALF_DIM])
            k_hi = pl.tile.set_validshape(k_hi, [1, HALF_DIM])
            klo_cos = pl.tile.col_expand_mul(k_lo, cos_lo)
            klo_cos = pl.tile.set_validshape(klo_cos, [1, HALF_DIM])
            khi_sin = pl.tile.col_expand_mul(k_hi, sin_lo)
            khi_sin = pl.tile.set_validshape(khi_sin, [1, HALF_DIM])
            k_rot_lo = pl.tile.sub(klo_cos, khi_sin)
            k_rot_lo = pl.tile.set_validshape(k_rot_lo, [1, HALF_DIM])
            khi_cos = pl.tile.col_expand_mul(k_hi, cos_hi)
            khi_cos = pl.tile.set_validshape(khi_cos, [1, HALF_DIM])
            klo_sin = pl.tile.col_expand_mul(k_lo, sin_hi)
            klo_sin = pl.tile.set_validshape(klo_sin, [1, HALF_DIM])
            k_rot_hi = pl.tile.add(khi_cos, klo_sin)
            k_rot_hi = pl.tile.set_validshape(k_rot_hi, [1, HALF_DIM])
            k_rot_lo_bf16 = pl.tile.cast(k_rot_lo, dtype=pl.BF16)
            k_rot_lo_bf16 = pl.tile.set_validshape(k_rot_lo_bf16, [1, HALF_DIM])
            k_rot_hi_bf16 = pl.tile.cast(k_rot_hi, dtype=pl.BF16)
            k_rot_hi_bf16 = pl.tile.set_validshape(k_rot_hi_bf16, [1, HALF_DIM])
            k_rot_lo_blk = pl.tile.reshape(k_rot_lo_bf16, [1, 1, 1, HALF_DIM])
            k_rot_hi_blk = pl.tile.reshape(k_rot_hi_bf16, [1, 1, 1, HALF_DIM])
            pl.tile.store(k_cache, k_rot_lo_blk, [slot_block, ki, slot_offset, 0])
            pl.tile.store(k_cache, k_rot_hi_blk, [slot_block, ki, slot_offset, HALF_DIM])

            # V head copy -> v_cache (lo/hi MM_N blocks).
            v_lo_blk = pl.tile.load(v_proj, [1, 1, 1, HALF_DIM], [kv_blo, 0, b, 0])
            v_lo = pl.tile.reshape(v_lo_blk, [1, HALF_DIM])
            v_lo = pl.tile.set_validshape(v_lo, [1, HALF_DIM])
            v_lo_bf16 = pl.tile.cast(v_lo, dtype=pl.BF16)
            v_lo_bf16 = pl.tile.set_validshape(v_lo_bf16, [1, HALF_DIM])
            v_lo_out = pl.tile.reshape(v_lo_bf16, [1, 1, 1, HALF_DIM])
            pl.tile.store(v_cache, v_lo_out, [slot_block, ki, slot_offset, 0])
            v_hi_blk = pl.tile.load(v_proj, [1, 1, 1, HALF_DIM], [kv_blo + 1, 0, b, 0])
            v_hi = pl.tile.reshape(v_hi_blk, [1, HALF_DIM])
            v_hi = pl.tile.set_validshape(v_hi, [1, HALF_DIM])
            v_hi_bf16 = pl.tile.cast(v_hi, dtype=pl.BF16)
            v_hi_bf16 = pl.tile.set_validshape(v_hi_bf16, [1, HALF_DIM])
            v_hi_out = pl.tile.reshape(v_hi_bf16, [1, 1, 1, HALF_DIM])
            pl.tile.store(v_cache, v_hi_out, [slot_block, ki, slot_offset, HALF_DIM])

            # Q heads RoPE (one row per head) + zero pad -> all_q_padded.
            q_base = ki * Q_PER_KV
            pad_idx = b * TOTAL_Q_GROUPS + ki
            for qi in pl.range(Q_HEAD_BATCH):
                q_blo = (q_base + qi) * HALVES_PER_HEAD
                q_lo_blk = pl.tile.load(q_proj_norm, [1, 1, 1, HALF_DIM], [q_blo, 0, b, 0])
                q_lo = pl.tile.reshape(q_lo_blk, [1, HALF_DIM])
                q_lo = pl.tile.set_validshape(q_lo, [1, HALF_DIM])
                q_hi_blk = pl.tile.load(q_proj_norm, [1, 1, 1, HALF_DIM], [q_blo + 1, 0, b, 0])
                q_hi = pl.tile.reshape(q_hi_blk, [1, HALF_DIM])
                q_hi = pl.tile.set_validshape(q_hi, [1, HALF_DIM])
                qlo_cos = pl.tile.col_expand_mul(q_lo, cos_lo)
                qhi_sin = pl.tile.col_expand_mul(q_hi, sin_lo)
                q_rot_lo = pl.tile.sub(qlo_cos, qhi_sin)
                q_rot_lo = pl.tile.set_validshape(q_rot_lo, [1, HALF_DIM])
                qhi_cos = pl.tile.col_expand_mul(q_hi, cos_hi)
                qlo_sin = pl.tile.col_expand_mul(q_lo, sin_hi)
                q_rot_hi = pl.tile.add(qhi_cos, qlo_sin)
                q_rot_hi = pl.tile.set_validshape(q_rot_hi, [1, HALF_DIM])
                q_rot_lo_bf16 = pl.tile.cast(q_rot_lo, dtype=pl.BF16)
                q_rot_lo_bf16 = pl.tile.set_validshape(q_rot_lo_bf16, [1, HALF_DIM])
                q_rot_hi_bf16 = pl.tile.cast(q_rot_hi, dtype=pl.BF16)
                q_rot_hi_bf16 = pl.tile.set_validshape(q_rot_hi_bf16, [1, HALF_DIM])
                q_rot_lo_blk = pl.tile.reshape(q_rot_lo_bf16, [1, 1, 1, HALF_DIM])
                q_rot_hi_blk = pl.tile.reshape(q_rot_hi_bf16, [1, 1, 1, HALF_DIM])
                pl.tile.store(all_q_padded, q_rot_lo_blk, [pad_idx, 0, qi, 0])
                pl.tile.store(all_q_padded, q_rot_hi_blk, [pad_idx, 0, qi, HALF_DIM])
            zpad = pl.tile.full([Q_HEAD_PAD, HEAD_DIM], value=0.0, dtype=pl.BF16)
            zpad = pl.tile.set_validshape(zpad, [Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM])
            zpad_blk = pl.tile.reshape(zpad, [1, 1, Q_HEAD_PAD, HEAD_DIM])
            pl.tile.store(all_q_padded, zpad_blk, [pad_idx, 0, Q_HEAD_BATCH, 0])

    # =====================================================================
    # 2b. Flash attention, online softmax. Per batch row, per KV head: stream
    #     the KV context in ATT_SEQ-wide steps. QK / SV matmuls are sub-tiled
    #     (QK over head-dim, SV over seq) and the HEAD_DIM output split lo/hi
    #     so every tile stays 4 KB.
    # =====================================================================
    # attn_out blocks mirror q_proj: head h's lo/hi halves at 2h / 2h + 1.
    attn_out = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.BF16)
    for b in pl.range(BATCH):
        ctx_len = pl.tensor.read(seq_lens, [b, 0, 0, 0])
        n_steps = (ctx_len + ATT_SEQ - 1) // ATT_SEQ

        for gi in pl.range(TOTAL_Q_GROUPS):
            kvh = gi
            q_base = kvh * Q_HEAD_BATCH
            pad_idx = b * TOTAL_Q_GROUPS + gi
            q_pad_blk = pl.tile.load(all_q_padded, [1, 1, Q_HEAD_PAD, HEAD_DIM], [pad_idx, 0, 0, 0])
            q_padded = pl.tile.reshape(q_pad_blk, [Q_HEAD_PAD, HEAD_DIM])

            # online accumulators seeded with sentinels mi=-inf, li=0, oi=0
            mi = pl.tile.full([Q_HEAD_PAD, VEC_W], value=NEG_INF, dtype=pl.FP32)
            mi = pl.tile.set_validshape(mi, [Q_HEAD_PAD, 1])
            li = pl.tile.full([Q_HEAD_PAD, VEC_W], value=0.0, dtype=pl.FP32)
            li = pl.tile.set_validshape(li, [Q_HEAD_PAD, 1])
            oi_lo = pl.tile.full([Q_HEAD_PAD, HALF_DIM], value=0.0, dtype=pl.FP32)
            oi_hi = pl.tile.full([Q_HEAD_PAD, HALF_DIM], value=0.0, dtype=pl.FP32)

            for st in pl.range(n_steps):
                g0 = st * ATT_SEQ
                sb = g0 // BLOCK_SIZE
                in_block = g0 - sb * BLOCK_SIZE
                pbid_i32 = pl.tensor.read(block_table, [b, sb, 0, 0])
                pbid = pl.cast(pbid_i32, pl.INDEX)
                valid_seq = pl.min(ATT_SEQ, ctx_len - g0)

                # --- QK matmul: scores[Q_HEAD_PAD, ATT_SEQ] over head-dim chunks ---
                q_sub0 = pl.tile.slice(q_padded, [Q_HEAD_PAD, QK_KD], [0, 0])
                q_sub0 = pl.tile.set_validshape(q_sub0, [Q_HEAD_PAD, QK_KD])
                k_sub0_blk = pl.tile.load(k_cache, [1, 1, ATT_SEQ, QK_KD], [pbid, kvh, in_block, 0])
                k_sub0 = pl.tile.reshape(k_sub0_blk, [ATT_SEQ, QK_KD])
                scores = pl.tile.matmul(q_sub0, k_sub0, b_trans=True, out_dtype=pl.FP32)
                for kd in pl.range(1, QK_KSTEPS):
                    kd0 = kd * QK_KD
                    q_sub = pl.tile.slice(q_padded, [Q_HEAD_PAD, QK_KD], [0, kd0])
                    q_sub = pl.tile.set_validshape(q_sub, [Q_HEAD_PAD, QK_KD])
                    k_sub_blk = pl.tile.load(k_cache, [1, 1, ATT_SEQ, QK_KD], [pbid, kvh, in_block, kd0])
                    k_sub = pl.tile.reshape(k_sub_blk, [ATT_SEQ, QK_KD])
                    scores = pl.tile.matmul_acc(scores, q_sub, k_sub)

                # --- tail-masked softmax (vec) ---
                scores_scaled = pl.tile.mul(scores, ATTN_SCALE)
                scores_valid = pl.tile.set_validshape(scores_scaled, [Q_HEAD_PAD, valid_seq])
                scores_pad = pl.tile.fillpad(scores_valid, pad_value=pl.PadValue.min)
                cur_mi = pl.tile.row_max(scores_pad)
                cur_mi = pl.tile.set_validshape(cur_mi, [Q_HEAD_PAD, 1])
                shifted = pl.tile.row_expand_sub(scores_pad, cur_mi)
                exp_scores = pl.tile.exp(shifted)
                exp_bf16 = pl.tile.cast(exp_scores, dtype=pl.BF16)
                exp_bf16 = pl.tile.set_validshape(exp_bf16, [Q_HEAD_PAD, ATT_SEQ])
                exp_fp32 = pl.tile.cast(exp_bf16, dtype=pl.FP32)
                cur_li = pl.tile.row_sum(exp_fp32)
                cur_li = pl.tile.set_validshape(cur_li, [Q_HEAD_PAD, 1])

                # --- SV matmul: oi halves over the SV_SSTEPS == 2 seq chunks ---
                exp_sub0 = pl.tile.slice(exp_bf16, [Q_HEAD_PAD, SV_SEQ], [0, 0])
                exp_sub0 = pl.tile.set_validshape(exp_sub0, [Q_HEAD_PAD, SV_SEQ])
                v_lo0_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [pbid, kvh, in_block, 0])
                v_lo0 = pl.tile.reshape(v_lo0_blk, [SV_SEQ, HALF_DIM])
                v_hi0_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [pbid, kvh, in_block, HALF_DIM])
                v_hi0 = pl.tile.reshape(v_hi0_blk, [SV_SEQ, HALF_DIM])
                oi_lo_tmp = pl.tile.matmul(exp_sub0, v_lo0, out_dtype=pl.FP32)
                oi_hi_tmp = pl.tile.matmul(exp_sub0, v_hi0, out_dtype=pl.FP32)
                exp_sub1 = pl.tile.slice(exp_bf16, [Q_HEAD_PAD, SV_SEQ], [0, SV_SEQ])
                exp_sub1 = pl.tile.set_validshape(exp_sub1, [Q_HEAD_PAD, SV_SEQ])
                in_block1 = in_block + SV_SEQ
                v_lo1_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [pbid, kvh, in_block1, 0])
                v_lo1 = pl.tile.reshape(v_lo1_blk, [SV_SEQ, HALF_DIM])
                v_hi1_blk = pl.tile.load(v_cache, [1, 1, SV_SEQ, HALF_DIM], [pbid, kvh, in_block1, HALF_DIM])
                v_hi1 = pl.tile.reshape(v_hi1_blk, [SV_SEQ, HALF_DIM])
                oi_lo_tmp = pl.tile.matmul_acc(oi_lo_tmp, exp_sub1, v_lo1)
                oi_hi_tmp = pl.tile.matmul_acc(oi_hi_tmp, exp_sub1, v_hi1)

                # --- online-softmax recurrence (UB accumulators) ---
                mi_new = pl.tile.maximum(mi, cur_mi)
                mi_new = pl.tile.set_validshape(mi_new, [Q_HEAD_PAD, 1])
                mdiff = pl.tile.sub(mi, mi_new)
                mdiff = pl.tile.set_validshape(mdiff, [Q_HEAD_PAD, 1])
                alpha = pl.tile.exp(mdiff)
                alpha = pl.tile.set_validshape(alpha, [Q_HEAD_PAD, 1])
                cdiff = pl.tile.sub(cur_mi, mi_new)
                cdiff = pl.tile.set_validshape(cdiff, [Q_HEAD_PAD, 1])
                beta = pl.tile.exp(cdiff)
                beta = pl.tile.set_validshape(beta, [Q_HEAD_PAD, 1])
                li_a = pl.tile.mul(alpha, li)
                li_a = pl.tile.set_validshape(li_a, [Q_HEAD_PAD, 1])
                li_b = pl.tile.mul(beta, cur_li)
                li_b = pl.tile.set_validshape(li_b, [Q_HEAD_PAD, 1])
                li_acc = pl.tile.add(li_a, li_b)
                li = pl.tile.set_validshape(li_acc, [Q_HEAD_PAD, 1])
                oi_lo_a = pl.tile.row_expand_mul(oi_lo, alpha)
                oi_lo_b = pl.tile.row_expand_mul(oi_lo_tmp, beta)
                oi_lo = pl.tile.add(oi_lo_a, oi_lo_b)
                oi_hi_a = pl.tile.row_expand_mul(oi_hi, alpha)
                oi_hi_b = pl.tile.row_expand_mul(oi_hi_tmp, beta)
                oi_hi = pl.tile.add(oi_hi_a, oi_hi_b)
                mi = mi_new

            # ctx = oi / li, trim Q_HEAD_PAD -> Q_HEAD_BATCH rows; per head row
            # qi, the lo/hi halves land in attn_out blocks 2(q_base+qi) / +1.
            ctx_lo = pl.tile.row_expand_div(oi_lo, li)
            ctx_hi = pl.tile.row_expand_div(oi_hi, li)
            for qi in pl.range(Q_HEAD_BATCH):
                h_blo = (q_base + qi) * HALVES_PER_HEAD
                lo1 = pl.tile.slice(ctx_lo, [1, HALF_DIM], [qi, 0])
                lo1 = pl.tile.set_validshape(lo1, [1, HALF_DIM])
                lo1_bf16 = pl.tile.cast(lo1, dtype=pl.BF16)
                lo1_bf16 = pl.tile.set_validshape(lo1_bf16, [1, HALF_DIM])
                lo1_blk = pl.tile.reshape(lo1_bf16, [1, 1, 1, HALF_DIM])
                pl.tile.store(attn_out, lo1_blk, [h_blo, 0, b, 0])
                hi1 = pl.tile.slice(ctx_hi, [1, HALF_DIM], [qi, 0])
                hi1 = pl.tile.set_validshape(hi1, [1, HALF_DIM])
                hi1_bf16 = pl.tile.cast(hi1, dtype=pl.BF16)
                hi1_bf16 = pl.tile.set_validshape(hi1_bf16, [1, HALF_DIM])
                hi1_blk = pl.tile.reshape(hi1_bf16, [1, 1, 1, HALF_DIM])
                pl.tile.store(attn_out, hi1_blk, [h_blo + 1, 0, b, 0])

    # =====================================================================
    # 3. Output projection + residual -> post-RMSNorm -> MLP -> residual.
    #    An MM_K activation chunk kb sits in MM_N block kb // 2, half kb % 2;
    #    an MM_N residual chunk sits in hidden block nb // 2, half nb % 2.
    # =====================================================================
    # --- out-proj + residual: resid1 = attn_out @ wo + current_hidden ---
    resid1 = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, MM_N], dtype=pl.FP32)
    for nb in pl.range(HIDDEN // MM_N):
        a0_blk = pl.tile.load(attn_out, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
        a0 = pl.tile.reshape(a0_blk, [BATCH, MM_K])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0_blk = pl.tile.load(wo, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
        w0 = pl.tile.reshape(w0_blk, [MM_K, MM_N])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            a_blk = pl.tile.load(attn_out, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
            a = pl.tile.reshape(a_blk, [BATCH, MM_K])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w_blk = pl.tile.load(wo, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
            w = pl.tile.reshape(w_blk, [MM_K, MM_N])
            acc = pl.tile.matmul_acc(acc, a, w)
        resid_blk = pl.tile.load(current_hidden, [1, 1, BATCH, MM_N], [nb // 2, 0, 0, (nb % 2) * MM_N])
        resid_bf16 = pl.tile.reshape(resid_blk, [BATCH, MM_N])
        resid_bf16 = pl.tile.set_validshape(resid_bf16, [BATCH, MM_N])
        resid = pl.tile.cast(resid_bf16, dtype=pl.FP32)
        out_sum = pl.tile.add(acc, resid)
        out_blk = pl.tile.reshape(out_sum, [1, 1, BATCH, MM_N])
        pl.tile.store(resid1, out_blk, [nb, 0, 0, 0])

    # --- post-attention RMSNorm: post_norm = (resid1 / rms) * post_gamma ---
    sumsq = pl.tile.full([BATCH, VEC_W], value=0.0, dtype=pl.FP32)
    sumsq = pl.tile.set_validshape(sumsq, [BATCH, 1])
    for kb in pl.range(HIDDEN // MM_N):
        x_blk = pl.tile.load(resid1, [1, 1, BATCH, VEC_W], [kb, 0, 0, 0])
        x = pl.tile.reshape(x_blk, [BATCH, VEC_W])
        sq = pl.tile.mul(x, x)
        part = pl.tile.row_sum(sq)
        part = pl.tile.set_validshape(part, [BATCH, 1])
        sumsq_acc = pl.tile.add(sumsq, part)
        sumsq = pl.tile.set_validshape(sumsq_acc, [BATCH, 1])

    mean_sq = pl.tile.mul(sumsq, HIDDEN_INV)
    mean_sq = pl.tile.set_validshape(mean_sq, [BATCH, 1])
    variance = pl.tile.add(mean_sq, EPS)
    variance = pl.tile.set_validshape(variance, [BATCH, 1])
    rms = pl.tile.sqrt(variance)
    rms = pl.tile.set_validshape(rms, [BATCH, 1])
    inv_rms = pl.tile.recip(rms)
    inv_rms = pl.tile.set_validshape(inv_rms, [BATCH, 1])

    # post_gamma block kb // 2 holds the two VEC_W halves of each VEC_BF16 chunk
    post_norm = pl.create_tensor([HIDDEN // MM_N, 1, BATCH, VEC_W], dtype=pl.BF16)
    for kb in pl.range(HIDDEN // MM_N):
        x_blk = pl.tile.load(resid1, [1, 1, BATCH, VEC_W], [kb, 0, 0, 0])
        x = pl.tile.reshape(x_blk, [BATCH, VEC_W])
        gamma_blk = pl.tile.load(post_rms_weight, [1, 1, 1, VEC_W], [kb // 2, 0, 0, (kb % 2) * VEC_W])
        gamma = pl.tile.reshape(gamma_blk, [1, VEC_W])
        gamma = pl.tile.set_validshape(gamma, [1, VEC_W])
        x_scaled = pl.tile.row_expand_mul(x, inv_rms)
        normed = pl.tile.col_expand_mul(x_scaled, gamma)
        normed_bf16 = pl.tile.cast(normed, dtype=pl.BF16)
        normed_bf16 = pl.tile.set_validshape(normed_bf16, [BATCH, VEC_W])
        normed_blk = pl.tile.reshape(normed_bf16, [1, 1, BATCH, VEC_W])
        pl.tile.store(post_norm, normed_blk, [kb, 0, 0, 0])

    # --- MLP gate/up + SiLU: mlp = (silu(post_norm @ w_gate)) * (post_norm @ w_up) ---
    # gate and up share one K-loop over the post_norm activation tiles.
    mlp = pl.create_tensor([INTERMEDIATE // MM_N, 1, BATCH, MM_N], dtype=pl.BF16)
    for nb in pl.range(INTERMEDIATE // MM_N):
        p0_blk = pl.tile.load(post_norm, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
        p0 = pl.tile.reshape(p0_blk, [BATCH, MM_K])
        p0 = pl.tile.set_validshape(p0, [BATCH, MM_K])
        wg0_blk = pl.tile.load(w_gate, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
        wg0 = pl.tile.reshape(wg0_blk, [MM_K, MM_N])
        wu0_blk = pl.tile.load(w_up, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
        wu0 = pl.tile.reshape(wu0_blk, [MM_K, MM_N])
        gate_acc = pl.tile.matmul(p0, wg0, out_dtype=pl.FP32)
        up_acc = pl.tile.matmul(p0, wu0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            p_blk = pl.tile.load(post_norm, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
            p = pl.tile.reshape(p_blk, [BATCH, MM_K])
            p = pl.tile.set_validshape(p, [BATCH, MM_K])
            wg_blk = pl.tile.load(w_gate, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
            wg = pl.tile.reshape(wg_blk, [MM_K, MM_N])
            wu_blk = pl.tile.load(w_up, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
            wu = pl.tile.reshape(wu_blk, [MM_K, MM_N])
            gate_acc = pl.tile.matmul_acc(gate_acc, p, wg)
            up_acc = pl.tile.matmul_acc(up_acc, p, wu)
        # SiLU(gate) * up = (gate * sigmoid(gate)) * up
        neg_gate = pl.tile.neg(gate_acc)
        exp_gate = pl.tile.exp(neg_gate)
        denom = pl.tile.add(exp_gate, 1.0)
        sigmoid = pl.tile.recip(denom)
        gate_sig = pl.tile.mul(gate_acc, sigmoid)
        mlp_chunk = pl.tile.mul(gate_sig, up_acc)
        mlp_bf16 = pl.tile.cast(mlp_chunk, dtype=pl.BF16)
        mlp_bf16 = pl.tile.set_validshape(mlp_bf16, [BATCH, MM_N])
        mlp_blk = pl.tile.reshape(mlp_bf16, [1, 1, BATCH, MM_N])
        pl.tile.store(mlp, mlp_blk, [nb, 0, 0, 0])

    # --- down-proj + residual: next_hidden = mlp @ w_down + resid1 ---
    for nb in pl.range(HIDDEN // MM_N):
        m0_blk = pl.tile.load(mlp, [1, 1, BATCH, MM_K], [0, 0, 0, 0])
        m0 = pl.tile.reshape(m0_blk, [BATCH, MM_K])
        m0 = pl.tile.set_validshape(m0, [BATCH, MM_K])
        wd0_blk = pl.tile.load(w_down, [1, 1, MM_K, MM_N], [0, nb, 0, 0])
        wd0 = pl.tile.reshape(wd0_blk, [MM_K, MM_N])
        acc = pl.tile.matmul(m0, wd0, out_dtype=pl.FP32)
        for kb in pl.range(1, INTERMEDIATE // MM_K):
            m_blk = pl.tile.load(mlp, [1, 1, BATCH, MM_K], [kb // 2, 0, 0, (kb % 2) * MM_K])
            m = pl.tile.reshape(m_blk, [BATCH, MM_K])
            m = pl.tile.set_validshape(m, [BATCH, MM_K])
            wd_blk = pl.tile.load(w_down, [1, 1, MM_K, MM_N], [kb, nb, 0, 0])
            wd = pl.tile.reshape(wd_blk, [MM_K, MM_N])
            acc = pl.tile.matmul_acc(acc, m, wd)
        resid_blk = pl.tile.load(resid1, [1, 1, BATCH, MM_N], [nb, 0, 0, 0])
        resid = pl.tile.reshape(resid_blk, [BATCH, MM_N])
        out_sum = pl.tile.add(acc, resid)
        out_bf16 = pl.tile.cast(out_sum, dtype=pl.BF16)
        out_bf16 = pl.tile.set_validshape(out_bf16, [BATCH, MM_N])
        out_blk = pl.tile.reshape(out_bf16, [1, 1, BATCH, MM_N])
        pl.tile.store(next_hidden, out_blk, [nb, 0, 0, 0])
