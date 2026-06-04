# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-14B single-layer decode forward, in a serial tile-DSL style.

Style rules:
  1. One plain function; control flow is only pl.range and if/else
     (no jit / program / at / spmd / parallel / pipeline).
  2. Compute is tile-level (pl.tile.*); the only tensor-level ops are
     pl.tensor.reshape (views) and pl.tensor.read (host scalars for
     loop bounds / paged addressing).
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
HN_ROWS = 8     # [8, 128] FP32 = 4 KB (per-head RMSNorm tile)

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
    current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
    input_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    wq: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    wk: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    wv: pl.Tensor[[HIDDEN, KV_HIDDEN], pl.BF16],
    q_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    k_norm_weight: pl.Tensor[[1, HEAD_DIM], pl.FP32],
    seq_lens: pl.Tensor[[BATCH], pl.INT32],
    block_table: pl.Tensor[[BATCH * MAX_BLOCKS_PER_SEQ], pl.INT32],
    slot_mapping: pl.Tensor[[BATCH], pl.INT32],
    rope_cos: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    rope_sin: pl.Tensor[[MAX_SEQ, HEAD_DIM], pl.FP32],
    k_cache: pl.Tensor[[NUM_KV_HEADS * MAX_BLOCKS_PER_SEQ * BLOCK_SIZE, HEAD_DIM], pl.BF16],
    v_cache: pl.Tensor[[NUM_KV_HEADS * MAX_BLOCKS_PER_SEQ * BLOCK_SIZE, HEAD_DIM], pl.BF16],
    wo: pl.Tensor[[HIDDEN, HIDDEN], pl.BF16],
    post_rms_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
    w_gate: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_up: pl.Tensor[[HIDDEN, INTERMEDIATE], pl.BF16],
    w_down: pl.Tensor[[INTERMEDIATE, HIDDEN], pl.BF16],
    next_hidden: pl.Out[pl.Tensor[[BATCH, HIDDEN], pl.BF16]],
):
    # Bridge buffer carrying the RMSNorm result into the Q/K/V matmuls.
    normed_all = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    # Raw (pre per-head-norm) Q/K projections.
    q_proj = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    # Per-head-normed Q/K and raw V -- internal bridges into scope 2.
    q_proj_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    k_proj_norm = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    v_proj = pl.create_tensor([BATCH, KV_HIDDEN], dtype=pl.FP32)
    # RoPE'd + padded Q, grouped [B, TOTAL_Q_GROUPS, Q_HEAD_PAD] rows of HEAD_DIM.
    all_q_padded = pl.create_tensor(
        [BATCH * TOTAL_Q_GROUPS * Q_HEAD_PAD, HEAD_DIM], dtype=pl.BF16,
    )
    # Attention output -> scope 3 input.
    attn_out = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    # Scope 3 bridges: residual1 (FP32), post-RMSNorm, MLP intermediate.
    resid1 = pl.create_tensor([BATCH, HIDDEN], dtype=pl.FP32)
    post_norm = pl.create_tensor([BATCH, HIDDEN], dtype=pl.BF16)
    mlp = pl.create_tensor([BATCH, INTERMEDIATE], dtype=pl.BF16)

    # =====================================================================
    # 1. Input RMSNorm:  normed_all = (x / rms(x)) * gamma
    # =====================================================================
    # sum of squares over HIDDEN (BF16 load -> two FP32 halves -> square)
    sumsq = pl.tile.full([BATCH, VEC_W], value=0.0, dtype=pl.FP32)
    sumsq = pl.tile.set_validshape(sumsq, [BATCH, 1])
    for kb in pl.range(HIDDEN // VEC_BF16):
        k0 = kb * VEC_BF16
        x_bf16 = pl.tile.load(current_hidden, [BATCH, VEC_BF16], [0, k0])
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

    # normalize + scale by gamma -> BF16 bridge buffer
    for kb in pl.range(HIDDEN // VEC_BF16):
        k0 = kb * VEC_BF16
        x_bf16 = pl.tile.load(current_hidden, [BATCH, VEC_BF16], [0, k0])
        for h in pl.range(2):
            h0 = h * VEC_W
            x_half = pl.tile.slice(x_bf16, [BATCH, VEC_W], [0, h0])
            x_half = pl.tile.set_validshape(x_half, [BATCH, VEC_W])
            x = pl.tile.cast(x_half, dtype=pl.FP32)
            gamma = pl.tile.load(input_rms_weight, [1, VEC_W], [0, k0 + h0])
            gamma = pl.tile.set_validshape(gamma, [1, VEC_W])
            x_scaled = pl.tile.row_expand_mul(x, inv_rms)
            normed = pl.tile.col_expand_mul(x_scaled, gamma)
            normed_bf16 = pl.tile.cast(normed, dtype=pl.BF16)
            normed_bf16 = pl.tile.set_validshape(normed_bf16, [BATCH, VEC_W])
            pl.tile.store(normed_all, normed_bf16, [0, k0 + h0])

    # =====================================================================
    # 2. Q / K / V projection:  proj = normed_all @ W
    #    (peeled-first matmul + matmul_acc over the K tiles, per output tile)
    # =====================================================================
    # --- Q projection: [BATCH, HIDDEN] @ [HIDDEN, HIDDEN] ---
    for nb in pl.range(HIDDEN // MM_N):
        n0 = nb * MM_N
        a0 = pl.tile.load(normed_all, [BATCH, MM_K], [0, 0])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0 = pl.tile.load(wq, [MM_K, MM_N], [0, n0])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            k0 = kb * MM_K
            a = pl.tile.load(normed_all, [BATCH, MM_K], [0, k0])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w = pl.tile.load(wq, [MM_K, MM_N], [k0, n0])
            acc = pl.tile.matmul_acc(acc, a, w)
        pl.tile.store(q_proj, acc, [0, n0])

    # --- K projection: [BATCH, HIDDEN] @ [HIDDEN, KV_HIDDEN] ---
    for nb in pl.range(KV_HIDDEN // MM_N):
        n0 = nb * MM_N
        a0 = pl.tile.load(normed_all, [BATCH, MM_K], [0, 0])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0 = pl.tile.load(wk, [MM_K, MM_N], [0, n0])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            k0 = kb * MM_K
            a = pl.tile.load(normed_all, [BATCH, MM_K], [0, k0])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w = pl.tile.load(wk, [MM_K, MM_N], [k0, n0])
            acc = pl.tile.matmul_acc(acc, a, w)
        pl.tile.store(k_proj, acc, [0, n0])

    # --- V projection: [BATCH, HIDDEN] @ [HIDDEN, KV_HIDDEN] ---
    for nb in pl.range(KV_HIDDEN // MM_N):
        n0 = nb * MM_N
        a0 = pl.tile.load(normed_all, [BATCH, MM_K], [0, 0])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0 = pl.tile.load(wv, [MM_K, MM_N], [0, n0])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            k0 = kb * MM_K
            a = pl.tile.load(normed_all, [BATCH, MM_K], [0, k0])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w = pl.tile.load(wv, [MM_K, MM_N], [k0, n0])
            acc = pl.tile.matmul_acc(acc, a, w)
        pl.tile.store(v_proj, acc, [0, n0])

    # =====================================================================
    # 3. Per-head q_norm / k_norm: reshape (batch, head) -> rows of HEAD_DIM,
    #    then a plain row-wise RMSNorm over each HEAD_DIM row.
    # =====================================================================
    q_rows = BATCH * NUM_HEADS          # 16 * 40  = 640
    k_rows = BATCH * NUM_KV_HEADS       # 16 * 8   = 128
    q_proj_heads = pl.tensor.reshape(q_proj, [q_rows, HEAD_DIM])
    q_norm_heads = pl.tensor.reshape(q_proj_norm, [q_rows, HEAD_DIM])
    k_proj_heads = pl.tensor.reshape(k_proj, [k_rows, HEAD_DIM])
    k_norm_heads = pl.tensor.reshape(k_proj_norm, [k_rows, HEAD_DIM])

    for rb in pl.range(q_rows // HN_ROWS):
        r0 = rb * HN_ROWS
        x = pl.tile.load(q_proj_heads, [HN_ROWS, HEAD_DIM], [r0, 0])
        sq = pl.tile.mul(x, x)
        sumsq = pl.tile.row_sum(sq)
        sumsq = pl.tile.set_validshape(sumsq, [HN_ROWS, 1])
        mean_sq = pl.tile.mul(sumsq, HEAD_DIM_INV)
        mean_sq = pl.tile.set_validshape(mean_sq, [HN_ROWS, 1])
        variance = pl.tile.add(mean_sq, EPS)
        variance = pl.tile.set_validshape(variance, [HN_ROWS, 1])
        inv_rms = pl.tile.rsqrt(variance)
        inv_rms = pl.tile.set_validshape(inv_rms, [HN_ROWS, 1])
        gamma = pl.tile.load(q_norm_weight, [1, HEAD_DIM], [0, 0])
        gamma = pl.tile.set_validshape(gamma, [1, HEAD_DIM])
        x_scaled = pl.tile.row_expand_mul(x, inv_rms)
        normed = pl.tile.col_expand_mul(x_scaled, gamma)
        pl.tile.store(q_norm_heads, normed, [r0, 0])

    for rb in pl.range(k_rows // HN_ROWS):
        r0 = rb * HN_ROWS
        x = pl.tile.load(k_proj_heads, [HN_ROWS, HEAD_DIM], [r0, 0])
        sq = pl.tile.mul(x, x)
        sumsq = pl.tile.row_sum(sq)
        sumsq = pl.tile.set_validshape(sumsq, [HN_ROWS, 1])
        mean_sq = pl.tile.mul(sumsq, HEAD_DIM_INV)
        mean_sq = pl.tile.set_validshape(mean_sq, [HN_ROWS, 1])
        variance = pl.tile.add(mean_sq, EPS)
        variance = pl.tile.set_validshape(variance, [HN_ROWS, 1])
        inv_rms = pl.tile.rsqrt(variance)
        inv_rms = pl.tile.set_validshape(inv_rms, [HN_ROWS, 1])
        gamma = pl.tile.load(k_norm_weight, [1, HEAD_DIM], [0, 0])
        gamma = pl.tile.set_validshape(gamma, [1, HEAD_DIM])
        x_scaled = pl.tile.row_expand_mul(x, inv_rms)
        normed = pl.tile.col_expand_mul(x_scaled, gamma)
        pl.tile.store(k_norm_heads, normed, [r0, 0])

    # =====================================================================
    # 2a. RoPE + paged KV-cache write. Per batch row, per KV head: rotate-half
    #     K -> k_cache, copy V -> v_cache, rotate the Q_HEAD_BATCH Q heads and
    #     zero-pad to Q_HEAD_PAD rows -> all_q_padded.
    # =====================================================================
    q_norm_heads_r = pl.tensor.reshape(q_proj_norm, [BATCH * NUM_HEADS, HEAD_DIM])
    for b in pl.range(BATCH):
        ctx_len = pl.tensor.read(seq_lens, [b])
        pos = ctx_len - 1
        slot = pl.tensor.read(slot_mapping, [b])
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot - slot_block * BLOCK_SIZE

        cos_row = pl.tile.load(rope_cos, [1, HEAD_DIM], [pos, 0])
        cos_row = pl.tile.set_validshape(cos_row, [1, HEAD_DIM])
        sin_row = pl.tile.load(rope_sin, [1, HEAD_DIM], [pos, 0])
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
            kv_col = ki * HEAD_DIM
            cache_row = (slot_block * NUM_KV_HEADS + ki) * BLOCK_SIZE + slot_offset

            # K head RoPE -> k_cache (rotate-half).
            k_lo = pl.tile.load(k_proj_norm, [1, HALF_DIM], [b, kv_col])
            k_lo = pl.tile.set_validshape(k_lo, [1, HALF_DIM])
            k_hi = pl.tile.load(k_proj_norm, [1, HALF_DIM], [b, kv_col + HALF_DIM])
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
            pl.tile.store(k_cache, k_rot_lo_bf16, [cache_row, 0])
            pl.tile.store(k_cache, k_rot_hi_bf16, [cache_row, HALF_DIM])

            # V head copy -> v_cache.
            v_head = pl.tile.load(v_proj, [1, HEAD_DIM], [b, kv_col])
            v_head = pl.tile.set_validshape(v_head, [1, HEAD_DIM])
            v_head_bf16 = pl.tile.cast(v_head, dtype=pl.BF16)
            v_head_bf16 = pl.tile.set_validshape(v_head_bf16, [1, HEAD_DIM])
            pl.tile.store(v_cache, v_head_bf16, [cache_row, 0])

            # Q heads RoPE (Q_HEAD_BATCH rows) + zero pad -> all_q_padded.
            q_base = ki * Q_PER_KV
            qp_row = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + ki * Q_HEAD_PAD
            q_block = pl.tile.load(
                q_norm_heads_r, [Q_HEAD_BATCH, HEAD_DIM], [b * NUM_HEADS + q_base, 0],
            )
            q_block = pl.tile.set_validshape(q_block, [Q_HEAD_BATCH, HEAD_DIM])
            q_lo = pl.tile.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, 0])
            q_lo = pl.tile.set_validshape(q_lo, [Q_HEAD_BATCH, HALF_DIM])
            q_hi = pl.tile.slice(q_block, [Q_HEAD_BATCH, HALF_DIM], [0, HALF_DIM])
            q_hi = pl.tile.set_validshape(q_hi, [Q_HEAD_BATCH, HALF_DIM])
            qlo_cos = pl.tile.col_expand_mul(q_lo, cos_lo)
            qhi_sin = pl.tile.col_expand_mul(q_hi, sin_lo)
            q_rot_lo = pl.tile.sub(qlo_cos, qhi_sin)
            q_rot_lo = pl.tile.set_validshape(q_rot_lo, [Q_HEAD_BATCH, HALF_DIM])
            qhi_cos = pl.tile.col_expand_mul(q_hi, cos_hi)
            qlo_sin = pl.tile.col_expand_mul(q_lo, sin_hi)
            q_rot_hi = pl.tile.add(qhi_cos, qlo_sin)
            q_rot_hi = pl.tile.set_validshape(q_rot_hi, [Q_HEAD_BATCH, HALF_DIM])
            q_rot_lo_bf16 = pl.tile.cast(q_rot_lo, dtype=pl.BF16)
            q_rot_lo_bf16 = pl.tile.set_validshape(q_rot_lo_bf16, [Q_HEAD_BATCH, HALF_DIM])
            q_rot_hi_bf16 = pl.tile.cast(q_rot_hi, dtype=pl.BF16)
            q_rot_hi_bf16 = pl.tile.set_validshape(q_rot_hi_bf16, [Q_HEAD_BATCH, HALF_DIM])
            pl.tile.store(all_q_padded, q_rot_lo_bf16, [qp_row, 0])
            pl.tile.store(all_q_padded, q_rot_hi_bf16, [qp_row, HALF_DIM])
            zpad = pl.tile.full([Q_HEAD_PAD, HEAD_DIM], value=0.0, dtype=pl.BF16)
            zpad = pl.tile.set_validshape(zpad, [Q_HEAD_PAD - Q_HEAD_BATCH, HEAD_DIM])
            pl.tile.store(all_q_padded, zpad, [qp_row + Q_HEAD_BATCH, 0])

    # =====================================================================
    # 2b. Flash attention, online softmax. Per batch row, per KV head: stream
    #     the KV context in ATT_SEQ-wide steps. QK / SV matmuls are sub-tiled
    #     (QK over head-dim, SV over seq) and the HEAD_DIM output split lo/hi
    #     so every tile stays 4 KB.
    # =====================================================================
    # head-as-row view: a KV group's Q_HEAD_BATCH heads are contiguous rows.
    attn_heads = pl.tensor.reshape(attn_out, [BATCH * NUM_HEADS, HEAD_DIM])
    for b in pl.range(BATCH):
        ctx_len = pl.tensor.read(seq_lens, [b])
        bt_base = b * MAX_BLOCKS_PER_SEQ
        n_steps = (ctx_len + ATT_SEQ - 1) // ATT_SEQ

        for gi in pl.range(TOTAL_Q_GROUPS):
            kvh = gi
            q_base = kvh * Q_HEAD_BATCH
            qp_row = b * TOTAL_Q_GROUPS * Q_HEAD_PAD + gi * Q_HEAD_PAD
            q_padded = pl.tile.load(all_q_padded, [Q_HEAD_PAD, HEAD_DIM], [qp_row, 0])

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
                pbid = pl.cast(pl.tensor.read(block_table, [bt_base + sb]), pl.INDEX)
                kv_row0 = (pbid * NUM_KV_HEADS + kvh) * BLOCK_SIZE + in_block
                valid_seq = pl.min(ATT_SEQ, ctx_len - g0)

                # --- QK matmul: scores[Q_HEAD_PAD, ATT_SEQ] over head-dim chunks ---
                q_sub0 = pl.tile.slice(q_padded, [Q_HEAD_PAD, QK_KD], [0, 0])
                q_sub0 = pl.tile.set_validshape(q_sub0, [Q_HEAD_PAD, QK_KD])
                k_sub0 = pl.tile.load(k_cache, [ATT_SEQ, QK_KD], [kv_row0, 0])
                scores = pl.tile.matmul(q_sub0, k_sub0, b_trans=True, out_dtype=pl.FP32)
                for kd in pl.range(1, QK_KSTEPS):
                    kd0 = kd * QK_KD
                    q_sub = pl.tile.slice(q_padded, [Q_HEAD_PAD, QK_KD], [0, kd0])
                    q_sub = pl.tile.set_validshape(q_sub, [Q_HEAD_PAD, QK_KD])
                    k_sub = pl.tile.load(k_cache, [ATT_SEQ, QK_KD], [kv_row0, kd0])
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
                v_lo0 = pl.tile.load(v_cache, [SV_SEQ, HALF_DIM], [kv_row0, 0])
                v_hi0 = pl.tile.load(v_cache, [SV_SEQ, HALF_DIM], [kv_row0, HALF_DIM])
                oi_lo_tmp = pl.tile.matmul(exp_sub0, v_lo0, out_dtype=pl.FP32)
                oi_hi_tmp = pl.tile.matmul(exp_sub0, v_hi0, out_dtype=pl.FP32)
                exp_sub1 = pl.tile.slice(exp_bf16, [Q_HEAD_PAD, SV_SEQ], [0, SV_SEQ])
                exp_sub1 = pl.tile.set_validshape(exp_sub1, [Q_HEAD_PAD, SV_SEQ])
                v_lo1 = pl.tile.load(v_cache, [SV_SEQ, HALF_DIM], [kv_row0 + SV_SEQ, 0])
                v_hi1 = pl.tile.load(v_cache, [SV_SEQ, HALF_DIM], [kv_row0 + SV_SEQ, HALF_DIM])
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

            # ctx = oi / li, trim Q_HEAD_PAD -> Q_HEAD_BATCH, store lo/hi blocks
            ctx_lo = pl.tile.row_expand_div(oi_lo, li)
            ctx_hi = pl.tile.row_expand_div(oi_hi, li)
            head_row0 = b * NUM_HEADS + q_base
            lo5 = pl.tile.slice(ctx_lo, [Q_HEAD_BATCH, HALF_DIM], [0, 0])
            lo5 = pl.tile.set_validshape(lo5, [Q_HEAD_BATCH, HALF_DIM])
            lo5_bf16 = pl.tile.cast(lo5, dtype=pl.BF16)
            lo5_bf16 = pl.tile.set_validshape(lo5_bf16, [Q_HEAD_BATCH, HALF_DIM])
            pl.tile.store(attn_heads, lo5_bf16, [head_row0, 0])
            hi5 = pl.tile.slice(ctx_hi, [Q_HEAD_BATCH, HALF_DIM], [0, 0])
            hi5 = pl.tile.set_validshape(hi5, [Q_HEAD_BATCH, HALF_DIM])
            hi5_bf16 = pl.tile.cast(hi5, dtype=pl.BF16)
            hi5_bf16 = pl.tile.set_validshape(hi5_bf16, [Q_HEAD_BATCH, HALF_DIM])
            pl.tile.store(attn_heads, hi5_bf16, [head_row0, HALF_DIM])

    # =====================================================================
    # 3. Output projection + residual -> post-RMSNorm -> MLP -> residual.
    # =====================================================================
    # --- out-proj + residual: resid1 = attn_out @ wo + current_hidden ---
    for nb in pl.range(HIDDEN // MM_N):
        n0 = nb * MM_N
        a0 = pl.tile.load(attn_out, [BATCH, MM_K], [0, 0])
        a0 = pl.tile.set_validshape(a0, [BATCH, MM_K])
        w0 = pl.tile.load(wo, [MM_K, MM_N], [0, n0])
        acc = pl.tile.matmul(a0, w0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            k0 = kb * MM_K
            a = pl.tile.load(attn_out, [BATCH, MM_K], [0, k0])
            a = pl.tile.set_validshape(a, [BATCH, MM_K])
            w = pl.tile.load(wo, [MM_K, MM_N], [k0, n0])
            acc = pl.tile.matmul_acc(acc, a, w)
        resid_bf16 = pl.tile.load(current_hidden, [BATCH, MM_N], [0, n0])
        resid_bf16 = pl.tile.set_validshape(resid_bf16, [BATCH, MM_N])
        resid = pl.tile.cast(resid_bf16, dtype=pl.FP32)
        out_sum = pl.tile.add(acc, resid)
        pl.tile.store(resid1, out_sum, [0, n0])

    # --- post-attention RMSNorm: post_norm = (resid1 / rms) * post_gamma ---
    sumsq = pl.tile.full([BATCH, VEC_W], value=0.0, dtype=pl.FP32)
    sumsq = pl.tile.set_validshape(sumsq, [BATCH, 1])
    for kb in pl.range(HIDDEN // VEC_W):
        k0 = kb * VEC_W
        x = pl.tile.load(resid1, [BATCH, VEC_W], [0, k0])
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

    for kb in pl.range(HIDDEN // VEC_W):
        k0 = kb * VEC_W
        x = pl.tile.load(resid1, [BATCH, VEC_W], [0, k0])
        gamma = pl.tile.load(post_rms_weight, [1, VEC_W], [0, k0])
        gamma = pl.tile.set_validshape(gamma, [1, VEC_W])
        x_scaled = pl.tile.row_expand_mul(x, inv_rms)
        normed = pl.tile.col_expand_mul(x_scaled, gamma)
        normed_bf16 = pl.tile.cast(normed, dtype=pl.BF16)
        normed_bf16 = pl.tile.set_validshape(normed_bf16, [BATCH, VEC_W])
        pl.tile.store(post_norm, normed_bf16, [0, k0])

    # --- MLP gate/up + SiLU: mlp = (silu(post_norm @ w_gate)) * (post_norm @ w_up) ---
    # gate and up share one K-loop over the post_norm activation tiles.
    for nb in pl.range(INTERMEDIATE // MM_N):
        n0 = nb * MM_N
        p0 = pl.tile.load(post_norm, [BATCH, MM_K], [0, 0])
        p0 = pl.tile.set_validshape(p0, [BATCH, MM_K])
        wg0 = pl.tile.load(w_gate, [MM_K, MM_N], [0, n0])
        wu0 = pl.tile.load(w_up, [MM_K, MM_N], [0, n0])
        gate_acc = pl.tile.matmul(p0, wg0, out_dtype=pl.FP32)
        up_acc = pl.tile.matmul(p0, wu0, out_dtype=pl.FP32)
        for kb in pl.range(1, HIDDEN // MM_K):
            k0 = kb * MM_K
            p = pl.tile.load(post_norm, [BATCH, MM_K], [0, k0])
            p = pl.tile.set_validshape(p, [BATCH, MM_K])
            wg = pl.tile.load(w_gate, [MM_K, MM_N], [k0, n0])
            wu = pl.tile.load(w_up, [MM_K, MM_N], [k0, n0])
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
        pl.tile.store(mlp, mlp_bf16, [0, n0])

    # --- down-proj + residual: next_hidden = mlp @ w_down + resid1 ---
    for nb in pl.range(HIDDEN // MM_N):
        n0 = nb * MM_N
        m0 = pl.tile.load(mlp, [BATCH, MM_K], [0, 0])
        m0 = pl.tile.set_validshape(m0, [BATCH, MM_K])
        wd0 = pl.tile.load(w_down, [MM_K, MM_N], [0, n0])
        acc = pl.tile.matmul(m0, wd0, out_dtype=pl.FP32)
        for kb in pl.range(1, INTERMEDIATE // MM_K):
            k0 = kb * MM_K
            m = pl.tile.load(mlp, [BATCH, MM_K], [0, k0])
            m = pl.tile.set_validshape(m, [BATCH, MM_K])
            wd = pl.tile.load(w_down, [MM_K, MM_N], [k0, n0])
            acc = pl.tile.matmul_acc(acc, m, wd)
        resid = pl.tile.load(resid1, [BATCH, MM_N], [0, n0])
        out_sum = pl.tile.add(acc, resid)
        out_bf16 = pl.tile.cast(out_sum, dtype=pl.BF16)
        out_bf16 = pl.tile.set_validshape(out_bf16, [BATCH, MM_N])
        pl.tile.store(next_hidden, out_bf16, [0, n0])
