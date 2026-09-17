# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Chunked Kimi-delta-attention for prefill.

The per-token recurrence is expanded into chunks of ``CHUNK`` tokens. Within a chunk:
a per-channel cumulative sum of the log decay, a decayed ``K K^T`` Gram matrix, the
inverse of a unit lower-triangular matrix (the UT transform), the ``W`` / ``U``
recomputation, and the intra-chunk output. Across chunks: a scan over the
``[KDA_DIM, KDA_DIM]`` state. q and k are L2-normalised in FP32 with ``eps = 1e-6``
and q carries the ``KDA_DIM ** -0.5`` scale.

Four things decide whether this is correct rather than merely plausible:

* **The decay must be reconstructed around a local pivot, and the pivot spacing is
  not free.** ``exp(gcum_i - gcum_j)`` for ``j`` ahead of ``i`` is ``exp(+large)``;
  taken literally it overflows to ``+inf`` and the mask then computes ``inf * 0 = NaN``
  for the whole chunk, so the column factor is clamped at ``DECAY_CAP``. But the clamp
  truncates the *kept* lower-triangular entries too, not only the masked ones,
  whenever the decay accumulated between the pivot and a column exceeds the cap. The
  gate is bounded by ``|g| <= 5`` per token, so a band of ``DECAY_BAND`` rows bounds
  that accumulation by ``5 * (DECAY_BAND - 1)`` = 75, under the cap, and the clamp
  then provably never touches a kept entry. Measured against a per-token reference in
  FP64: one pivot per 32-row chunk is wrong by 4e-2, one per 16 rows is exact to
  6e-17. A fixture whose gates are near zero never sees any of this. Note that the
  Ascend-tuned reference bands at 64 rows, which this bound does not cover.
* **The tail chunk's padding must be zero in the decay**, so its cumulative sum stays
  flat and the last row remains the true chunk-final decay.
* **The state's stored layout is ``[V, K]``** (``state_v_first`` in the AscendC ABI)
  while the recurrence is natural in ``[K, V]``. Both axes are 128, so a transposed
  state passes every shape check and shows up only as wrong output from the second
  chunk onward. The transposes happen here, at seed and at flush.
* ``decay`` carries the **log** decay. This kernel exponentiates it; the projection
  that produced it does not.

The work is split into two passes over the same chunks. The first computes everything
that does not involve the carried state, which is all of the expensive algebra; the
second is the sequential scan and touches only matmuls against the state. Splitting
them keeps the live set inside UB — a single fused body would need q, k, the
cumulative sum, both decay factors, the Gram matrix and its inverse simultaneously.

``CHUNK`` is 64, the value both the reference and vLLM-Ascend's ``KDA_CHUNK_SIZE``
use. The Ascend-tuned pypto implementation in cann-recipes-infer uses 128, but it
targets a frontend that spills tiles automatically; here every intermediate is
explicit, and 128 does not fit.

The triangular inverse is a plain forward substitution, one row per step, staged in
GM. That is the straightforward correct form, not the fast one — the tuned reference
inverts eight stacked 16x16 leaves and merges them, which needs tile shapes that grow
with the loop index and so cannot be written in this frontend. It is the first thing
to revisit when this kernel is profiled.
"""

import sys
from pathlib import Path

# Run directly and the script's own directory leads sys.path, where this model's
# ``golden.py`` shadows the repo-root ``golden`` harness package. Put the repo root
# first, exactly as models/deepseek_v4_1_flash does.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import B_DYN, KDA_DIM, KDA_STATE_DYN, LOCAL_KDA_H
from models.glm5_3_flash.config import Q_START_DYN, T_DYN
from models.glm5_3_flash.kda_projection import BETA_PAD
from models.glm5_3_flash.golden import l2norm

# Chunk length. The chunked delta rule is exact for any chunk size, so this is a
# tiling parameter, and two constraints pin it down here. At 64 the pass-1 body holds
# eight [64, 128] FP32 intermediates and overruns UB's 184 KiB. And the decay pivot
# has to sit within DECAY_CAP of every column it reconstructs, which bounds a pivot's
# span at 16 tokens; making the chunk longer than that needs several pivots per chunk,
# whose per-band results then have to be stitched together, and doing that through a
# scratch buffer races with the read that follows it. One pivot per chunk keeps both
# constraints satisfied without a stitch. The reference and vLLM-Ascend use 64 and the
# Ascend-tuned pypto implementation uses 128, but both rely on a frontend that spills
# tiles automatically. This is the first thing to revisit when the kernel is profiled.
CHUNK = 16
DECAY_CAP = 80.0  # exp argument ceiling: fp32 exp overflows near 88.7, and the
# masked-out future region would otherwise be +inf, giving inf * 0 = NaN
DECAY_BAND = CHUNK  # one local decay pivot per chunk -- see the module docstring
L2_EPS = 1e-6
Q_SCALE = KDA_DIM ** -0.5

# tiling
V_TILE = 64  # value columns per scan task: the carried state slice is [128, 64] FP32


def golden_prefill_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(output, recurrent_state)``; the state is returned updated.

    ``recurrent_state`` is ``[rows, LOCAL_KDA_H, V, K]``. The reference runs the plain
    per-token recurrence rather than the chunked form: it is the definition the
    chunked kernel has to reproduce, and it cannot share a bug with it.
    """
    tokens = query.shape[0]
    state = recurrent_state.clone().float()
    out = torch.zeros(tokens, LOCAL_KDA_H, KDA_DIM, dtype=torch.float32)
    requests = query_start_loc.numel() - 1
    for b in range(requests):
        s = int(query_start_loc[b])
        e = int(query_start_loc[b + 1])
        row = int(state_rows[b])
        for h in range(LOCAL_KDA_H):
            sh = state[row, h]  # [V, K]
            for t in range(s, e):
                q = l2norm(query[t, h].float(), eps=L2_EPS) * Q_SCALE
                k = l2norm(key[t, h].float(), eps=L2_EPS)
                v = value[t, h].float()
                g = decay[t, h].float()
                bta = float(beta[t, h])
                sh = sh * torch.exp(g).unsqueeze(0)
                kv_mem = (sh * k.unsqueeze(0)).sum(dim=-1)
                delta = (v - kv_mem) * bta
                sh = sh + delta.unsqueeze(-1) * k.unsqueeze(0)
                out[t, h] = (sh * q.unsqueeze(0)).sum(dim=-1)
            state[row, h] = sh
    return out.to(query.dtype), state.to(recurrent_state.dtype)


@pl.jit.inline
def prefill_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK, KDA_DIM], pl.FP32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(query, 0)
    requests = pl.tensor.dim(query_start_loc, 0) - 1
    # Flattened head-major, not token-major: with one row per (token, head) the tokens
    # of one head would sit LOCAL_KDA_H rows apart and a chunk slice would straddle
    # heads. Keeping the row axis as tokens makes a head a contiguous column range.
    head_width = LOCAL_KDA_H * KDA_DIM
    q_rows = pl.reshape(query, [t_dim, head_width])
    k_rows = pl.reshape(key, [t_dim, head_width])
    v_rows = pl.reshape(value, [t_dim, head_width])
    g_rows = pl.reshape(decay, [t_dim, head_width])
    out_rows = pl.reshape(output, [t_dim, head_width])

    # Per-chunk scratch. A chunk of sequence n starting at token ``off`` is parked at
    # row ``off + n * CHUNK``: its last chunk runs past the sequence end, and without
    # the per-sequence stride it would land on the next sequence's rows.
    scratch_rows = t_dim + requests * CHUNK
    qg_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)
    w_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)
    u_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    kdec_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)
    a2_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, CHUNK], dtype=pl.BF16)
    # Flat rather than [head, row, dim]: a store whose source is a tile reshaped to
    # two leading unit axes lowers to a tile with a single-element row, which ptoas
    # rejects on its 32-byte row alignment. One row per chunk, so the index is plain.
    eglast_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, KDA_DIM], dtype=pl.FP32)
    # The triangular inverse is built one row at a time, so it needs a place to stand.
    ainv_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, CHUNK], dtype=pl.FP32)
    # Two alternating slots each, so a squaring never reads the buffer it writes.
    # Kept two-dimensional: storing a matmul result through a reshape into a rank-3
    # view has no producer pipe at this width, while the plain 2D store does.
    # Every slot of these is written whole before it is read, so they need no seeding.
    estep_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, 5 * KDA_DIM], dtype=pl.FP32)
    prod_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, 5 * KDA_DIM], dtype=pl.FP32)
    # a_gram widened to the inverse's working width. Its right-hand columns have to be
    # zero and stay zero, and they are zeroed by their own kernel below: a read that
    # spans two stores from the same task is not ordered against both of them, which is
    # what made a "matrix here, zeros beside it" seed read back as all zeros and
    # collapse the product that followed.
    agw_gm = pl.create_tensor([LOCAL_KDA_H * scratch_rows, KDA_DIM], dtype=pl.FP32)
    agw_rows = LOCAL_KDA_H * scratch_rows
    zero_tiles = (agw_rows + CHUNK - 1) // CHUNK
    with pl.spmd(zero_tiles, name_hint="prefill_kda_zero") as zero_tid:
        z0 = pl.tile.get_block_idx() * CHUNK
        zrows = pl.min(CHUNK, agw_rows - z0)
        agw_gm[z0 : z0 + CHUNK, 0:KDA_DIM] = pl.set_validshape(
            pl.full([CHUNK, KDA_DIM], dtype=pl.FP32, value=0.0), zrows, KDA_DIM)
    agram_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, CHUNK], dtype=pl.FP32)
    # The cumulative sum comes out of the cube in a boxed layout, whose subviews must
    # be whole inner blocks -- a single row of it is not addressable. Parking it in GM
    # and reading the pivot and last rows back is the way to get at them.
    gcum_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    qf_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    kf_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.FP32)
    # Every intermediate gets its own buffer. Reusing one for two values -- parking kg
    # where W will land, or masking a Gram matrix in place -- makes a read and a write
    # of the same rows race inside one task, and a read that wins returns uninitialised
    # memory, which is where the NaNs came from.
    kg_gm = pl.create_tensor([LOCAL_KDA_H, scratch_rows, KDA_DIM], dtype=pl.BF16)

    # --- pass 1: everything that does not touch the carried state ---
    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_chunk",
                 deps=[zero_tid]) as chunk_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos

        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            actual = pl.min(length - s_idx, CHUNK)
            base = off + n * CHUNK
            # Rows past the chunk's real length must read as zero, and valid_shape does
            # not put them there: it marks which rows are real, while the tile still
            # holds whatever the overread picked up -- which for a chunk that runs past
            # the end of the packed stream is whatever follows it in GM. fillpad zeroes
            # them explicitly. For the decay this is not hygiene but load-bearing: a
            # zero pad keeps the cumulative sum flat, so its last row is still the
            # chunk's true final decay, and the padded rows contribute nothing to the
            # state update. Sequences that are a whole number of chunks never expose
            # this; a ragged tail after a full chunk does.
            # Zeroed, then declared valid to the full chunk. Both halves matter. The
            # zeroing is load-bearing for the decay: a zero pad keeps the cumulative sum
            # flat, so its last row is still the chunk's true final decay, and the pad
            # rows contribute nothing to the state. Restoring the valid extent matters
            # because these tiles end up as transposed matmul operands, and the cube
            # lowers a transposed operand to an extract whose row count must be a
            # multiple of 16 -- a ragged valid extent becomes exactly that row count.
            # The simulator enforces neither, so both show up only on device.
            col0 = h * KDA_DIM
            q_raw = pl.cast(pl.set_validshape(pl.fillpad(
                pl.slice(q_rows, [CHUNK, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM]),
                pad_value=pl.PadValue.zero), CHUNK, KDA_DIM), pl.FP32)
            k_raw = pl.cast(pl.set_validshape(pl.fillpad(
                pl.slice(k_rows, [CHUNK, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM]),
                pad_value=pl.PadValue.zero), CHUNK, KDA_DIM), pl.FP32)
            v_c = pl.set_validshape(pl.fillpad(
                pl.slice(v_rows, [CHUNK, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM]),
                pad_value=pl.PadValue.zero), CHUNK, KDA_DIM)
            g_c = pl.set_validshape(pl.fillpad(
                pl.slice(g_rows, [CHUNK, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM]),
                pad_value=pl.PadValue.zero), CHUNK, KDA_DIM)

            qf_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(pl.mul(
                pl.row_expand_mul(q_raw, pl.rsqrt(
                    pl.add(pl.row_sum(pl.mul(q_raw, q_raw)), L2_EPS), high_precision=True)),
                Q_SCALE), [1, CHUNK, KDA_DIM])
            kf_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(
                pl.row_expand_mul(k_raw, pl.rsqrt(
                    pl.add(pl.row_sum(pl.mul(k_raw, k_raw)), L2_EPS), high_precision=True)),
                [1, CHUNK, KDA_DIM])
            q_f = pl.reshape(pl.slice(qf_gm, [1, CHUNK, KDA_DIM], [h, base, 0], drop_dims=[0]),
                             [CHUNK, KDA_DIM])
            k_f = pl.reshape(pl.slice(kf_gm, [1, CHUNK, KDA_DIM], [h, base, 0], drop_dims=[0]),
                             [CHUNK, KDA_DIM])

            # Per-channel cumulative sum along the chunk axis, as one GEMM against the
            # inclusive lower-triangular constant: the DSL has no scan primitive.
            gcum_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(
                pl.matmul(tri_incl[0:CHUNK, 0:CHUNK], g_c, pl.FP32), [1, CHUNK, KDA_DIM])
            gcum = pl.reshape(
                pl.slice(gcum_gm, [1, CHUNK, KDA_DIM], [h, base, 0], drop_dims=[0]),
                [CHUNK, KDA_DIM])
            eg = pl.exp(gcum)

            # Decay reconstructed around the chunk's first row, which is also its only
            # pivot: CHUNK is DECAY_BAND. Rows are at or after the pivot, so row_dec's
            # exponent is <= 0; the column factor's exponent is at most
            # 5 * (CHUNK - 1) = 75 inside the chunk, so the clamp provably touches only
            # the masked-out future. Both Gram matrices stay FP32 -- the decay's
            # dynamic range rounds away in BF16.
            gp = pl.reshape(pl.slice(gcum_gm, [1, 1, KDA_DIM], [h, base, 0], drop_dims=[0]),
                            [1, KDA_DIM])
            gp_full = pl.col_expand_mul(
                pl.full([CHUNK, KDA_DIM], dtype=pl.FP32, value=1.0), gp)
            row_dec = pl.exp(pl.sub(gcum, gp_full))
            ki_a = pl.mul(k_f, pl.exp(pl.minimum(pl.sub(gp_full, gcum), DECAY_CAP)))
            a_full = pl.matmul(pl.mul(k_f, row_dec), ki_a, pl.FP32, b_trans=True)
            a2_raw = pl.matmul(pl.mul(q_f, row_dec), ki_a, pl.FP32, b_trans=True)


            # beta as a column, straight out of a masked reduction. beta is stored one
            # value per head in a 16-wide row, so selecting this head's column with a
            # host-supplied one-hot and summing gives the column directly -- no
            # transpose, and a reduction is the only way a column is a legal tile here.
            beta_block = pl.set_validshape(pl.fillpad(
                pl.slice(beta, [CHUNK, BETA_PAD], [off, 0], valid_shape=[actual, BETA_PAD]),
                pad_value=pl.PadValue.zero), CHUNK, BETA_PAD)
            b_col = pl.row_sum(pl.col_expand_mul(
                beta_block, head_onehot[h : h + 1, 0:BETA_PAD]))
            a_gram = pl.mul(pl.row_expand_mul(a_full, b_col), tri_strict[0:CHUNK, 0:CHUNK])
            agram_gm[h : h + 1, base : base + CHUNK, 0:CHUNK] = pl.reshape(
                a_gram, [1, CHUNK, CHUNK])
            a2_gm[h : h + 1, base : base + CHUNK, 0:CHUNK] = pl.reshape(
                pl.cast(pl.mul(a2_raw, tri_incl[0:CHUNK, 0:CHUNK]), pl.BF16, mode="rint"),
                [1, CHUNK, CHUNK])
            qg_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(
                pl.cast(pl.mul(q_f, eg), pl.BF16, mode="rint"), [1, CHUNK, KDA_DIM])
            # kg = k * exp(gcum) feeds W; kdec carries the chunk-final decay.
            glast = pl.reshape(
                pl.slice(gcum_gm, [1, 1, KDA_DIM], [h, base + CHUNK - 1, 0], drop_dims=[0]),
                [1, KDA_DIM])
            glast_full = pl.col_expand_mul(
                pl.full([CHUNK, KDA_DIM], dtype=pl.FP32, value=1.0), glast)
            kdec_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(
                pl.cast(pl.mul(k_f, pl.exp(pl.sub(glast_full, gcum))), pl.BF16, mode="rint"),
                [1, CHUNK, KDA_DIM])
            eg_row = h * scratch_rows + base
            eglast_gm[eg_row : eg_row + 1, 0:KDA_DIM] = pl.exp(glast)
            kg_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(
                pl.cast(pl.mul(k_f, eg), pl.BF16, mode="rint"), [1, CHUNK, KDA_DIM])

            rb = h * scratch_rows + base
            agw_gm[rb : rb + CHUNK, 0:CHUNK] = a_gram



    # (I + A)^-1 as a finite series. A is strictly lower and therefore nilpotent with
    # A^CHUNK = 0, so (I + A)^-1 = sum_k (-A)^k, and the sum doubles its reach each
    # step. Iterating E_j = I + M_j rather than the power itself keeps both matmul
    # operands in GM:
    #     E_0 = I - A,               P_0 = I
    #     P_{j+1} = E_j P_j,         E_{j+1} = E_j (E_j - 2I) + 2 I
    # (the second line is M_{j+1} = M_j^2 rewritten in terms of E, factored so that it
    # is one matmul and one add). Four steps cover all 16 terms.
    #
    # Each step is its own kernel, chained by an explicit dependency. Within one kernel
    # a store and a later read of the same buffer are not ordered, and the seed read
    # back as zeros, which collapsed the whole product; across kernels the dependency
    # is what orders them. Everything is carried at KDA_DIM width because a 16-wide
    # FP32 tile has no hand-off between the cube and the vector pipe here, and the
    # padding stays zero through both recurrences.
    #
    # The textbook alternative, row-at-a-time forward substitution, has a matmul with
    # M = 1, which the cube lowers to an extract whose row count must be a multiple
    # of 16.

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv_seed", deps=[chunk_tid]) as seed_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            base = off + n * CHUNK
            rb = h * scratch_rows + base
            estep_gm[rb : rb + CHUNK, 0:KDA_DIM] = pl.sub(
                eye_wide[0:CHUNK, 0:KDA_DIM], agw_gm[rb : rb + CHUNK, 0:KDA_DIM])
            prod_gm[rb : rb + CHUNK, 0:KDA_DIM] = eye_wide[0:CHUNK, 0:KDA_DIM]

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv0", deps=[seed_tid]) as inv0_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            base = off + n * CHUNK
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK, 0:0 + CHUNK]
            e_wide = estep_gm[rb : rb + CHUNK, 0:0 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK, 0:KDA_DIM], 2.0)
            prod_gm[rb : rb + CHUNK, 128:128 + KDA_DIM] = pl.matmul(
                e16, prod_gm[rb : rb + CHUNK, 0:0 + KDA_DIM], pl.FP32)
            estep_gm[rb : rb + CHUNK, 128:128 + KDA_DIM] = pl.add(
                pl.matmul(e16, pl.sub(e_wide, two_eye), pl.FP32), two_eye)

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv1", deps=[inv0_tid]) as inv1_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            base = off + n * CHUNK
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK, 128:128 + CHUNK]
            e_wide = estep_gm[rb : rb + CHUNK, 128:128 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK, 0:KDA_DIM], 2.0)
            prod_gm[rb : rb + CHUNK, 256:256 + KDA_DIM] = pl.matmul(
                e16, prod_gm[rb : rb + CHUNK, 128:128 + KDA_DIM], pl.FP32)
            estep_gm[rb : rb + CHUNK, 256:256 + KDA_DIM] = pl.add(
                pl.matmul(e16, pl.sub(e_wide, two_eye), pl.FP32), two_eye)

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv2", deps=[inv1_tid]) as inv2_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            base = off + n * CHUNK
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK, 256:256 + CHUNK]
            e_wide = estep_gm[rb : rb + CHUNK, 256:256 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK, 0:KDA_DIM], 2.0)
            prod_gm[rb : rb + CHUNK, 384:384 + KDA_DIM] = pl.matmul(
                e16, prod_gm[rb : rb + CHUNK, 256:256 + KDA_DIM], pl.FP32)
            estep_gm[rb : rb + CHUNK, 384:384 + KDA_DIM] = pl.add(
                pl.matmul(e16, pl.sub(e_wide, two_eye), pl.FP32), two_eye)

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_inv3", deps=[inv2_tid]) as inv3_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            base = off + n * CHUNK
            rb = h * scratch_rows + base
            e16 = estep_gm[rb : rb + CHUNK, 384:384 + CHUNK]
            e_wide = estep_gm[rb : rb + CHUNK, 384:384 + KDA_DIM]
            two_eye = pl.mul(eye_wide[0:CHUNK, 0:KDA_DIM], 2.0)
            prod_gm[rb : rb + CHUNK, 512:512 + KDA_DIM] = pl.matmul(
                e16, prod_gm[rb : rb + CHUNK, 384:384 + KDA_DIM], pl.FP32)
            estep_gm[rb : rb + CHUNK, 512:512 + KDA_DIM] = pl.add(
                pl.matmul(e16, pl.sub(e_wide, two_eye), pl.FP32), two_eye)

    with pl.spmd(LOCAL_KDA_H * requests, name_hint="prefill_kda_wu", deps=[inv3_tid]) as wu_tid:
        task = pl.tile.get_block_idx()
        h = task // requests
        n = task % requests
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            base = off + n * CHUNK
            rb = h * scratch_rows + base
            actual = pl.min(length - s_idx, CHUNK)
            col0 = h * KDA_DIM
            # W = A^-1 (beta . kg) and U = A^-1 (beta . v). Scaling the inverse's
            # columns by beta is the same as scaling the rows of what it multiplies.
            beta_block = pl.set_validshape(pl.fillpad(
                pl.slice(beta, [CHUNK, BETA_PAD], [off, 0], valid_shape=[actual, BETA_PAD]),
                pad_value=pl.PadValue.zero), CHUNK, BETA_PAD)
            b_col = pl.row_sum(pl.col_expand_mul(
                beta_block, head_onehot[h : h + 1, 0:BETA_PAD]))
            a_inv_bf = pl.cast(
                prod_gm[rb : rb + CHUNK, 4 * KDA_DIM : 4 * KDA_DIM + CHUNK],
                pl.BF16, mode="rint")
            kg_bf = pl.cast(pl.row_expand_mul(pl.cast(pl.reshape(
                pl.slice(kg_gm, [1, CHUNK, KDA_DIM], [h, base, 0], drop_dims=[0]),
                [CHUNK, KDA_DIM]), pl.FP32), b_col), pl.BF16, mode="rint")
            w_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(
                pl.cast(pl.matmul(a_inv_bf, kg_bf, pl.FP32), pl.BF16, mode="rint"),
                [1, CHUNK, KDA_DIM])
            v_c = pl.set_validshape(pl.fillpad(
                pl.slice(v_rows, [CHUNK, KDA_DIM], [off, col0], valid_shape=[actual, KDA_DIM]),
                pad_value=pl.PadValue.zero), CHUNK, KDA_DIM)
            v_beta = pl.cast(pl.row_expand_mul(pl.cast(v_c, pl.FP32), b_col),
                             pl.BF16, mode="rint")
            u_gm[h : h + 1, base : base + CHUNK, 0:KDA_DIM] = pl.reshape(
                pl.matmul(a_inv_bf, v_beta, pl.FP32), [1, CHUNK, KDA_DIM])
    # --- pass 2: the sequential scan over chunks, tiled over the value axis ---
    lanes = KDA_DIM // V_TILE
    state_2d = pl.reshape(
        recurrent_state,
        [pl.tensor.dim(recurrent_state, 0) * LOCAL_KDA_H * KDA_DIM, KDA_DIM])
    # Two slots per task, alternating by chunk. Reading and writing one buffer inside
    # the loop would make each iteration alias its own input, and a read that wins
    # returns uninitialised memory -- which is exactly the NaN a multi-chunk sequence
    # produced while a single-chunk one passed.
    state_gm = pl.create_tensor([LOCAL_KDA_H * requests * lanes * 2 * V_TILE, KDA_DIM],
                                dtype=pl.FP32)
    with pl.spmd(LOCAL_KDA_H * requests * lanes, name_hint="prefill_kda_scan",
                 deps=[wu_tid]):
        task = pl.tile.get_block_idx()
        h = task // (requests * lanes)
        n = (task // lanes) % requests
        v0 = (task % lanes) * V_TILE
        bos = pl.cast(pl.read(query_start_loc, [n]), pl.INDEX)
        eos = pl.cast(pl.read(query_start_loc, [n + 1]), pl.INDEX)
        length = eos - bos
        row = pl.cast(pl.read(state_rows, [n]), pl.INDEX)

        # The scan runs in the pool's own [V, K] orientation. The recurrence is natural
        # in [K, V], but every re-orientation it needs is a matmul operand flag, and
        # keeping [V, K] means the seed and the flush are plain copies. That matters
        # beyond tidiness: the transpose instruction constrains both of its row counts
        # to multiples of 16, which the simulator does not check and the device does.
        pool_row = (row * LOCAL_KDA_H + h) * KDA_DIM + v0
        seed = pl.slice(state_2d, [V_TILE, KDA_DIM], [pool_row, 0])
        slot0 = task * 2 * V_TILE
        state_gm[slot0 : slot0 + V_TILE, 0:KDA_DIM] = seed

        for s_idx in pl.range(0, length, CHUNK):
            off = bos + s_idx
            actual = pl.min(length - s_idx, CHUNK)
            base = off + n * CHUNK
            chunk_idx = s_idx // CHUNK
            src = chunk_idx % 2
            dst = (chunk_idx + 1) % 2
            s_cur = pl.slice(state_gm, [V_TILE, KDA_DIM], [(task * 2 + src) * V_TILE, 0])
            s_bf = pl.cast(s_cur, pl.BF16, mode="rint")
            w_c = pl.reshape(
                pl.slice(w_gm, [1, CHUNK, KDA_DIM], [h, base, 0], drop_dims=[0]),
                [CHUNK, KDA_DIM])
            u_c = pl.reshape(
                pl.slice(u_gm, [1, CHUNK, V_TILE], [h, base, v0], drop_dims=[0]),
                [CHUNK, V_TILE])
            # v' = U - W S: the part of the chunk's values the carried state already
            # explains.
            vi = pl.sub(u_c, pl.matmul(w_c, s_bf, pl.FP32, b_trans=True))
            vi_bf = pl.cast(vi, pl.BF16, mode="rint")
            qg_c = pl.reshape(
                pl.slice(qg_gm, [1, CHUNK, KDA_DIM], [h, base, 0], drop_dims=[0]),
                [CHUNK, KDA_DIM])
            a2_c = pl.reshape(
                pl.slice(a2_gm, [1, CHUNK, CHUNK], [h, base, 0], drop_dims=[0]),
                [CHUNK, CHUNK])
            oc = pl.add(pl.matmul(qg_c, s_bf, pl.FP32, b_trans=True),
                        pl.matmul(a2_c, vi_bf, pl.FP32))
            ocol = h * KDA_DIM + v0
            out_rows[off : off + CHUNK, ocol : ocol + V_TILE] = pl.set_validshape(
                pl.cast(oc, pl.BF16, mode="rint"), actual, V_TILE)

            kdec_c = pl.reshape(
                pl.slice(kdec_gm, [1, CHUNK, KDA_DIM], [h, base, 0], drop_dims=[0]),
                [CHUNK, KDA_DIM])
            # In [V, K] the chunk-final decay is a per-column factor, so it applies as
            # the row it already is.
            eglast_row = pl.slice(eglast_gm, [1, KDA_DIM], [h * scratch_rows + base, 0])
            d_row = (task * 2 + dst) * V_TILE
            state_gm[d_row : d_row + V_TILE, 0:KDA_DIM] = pl.add(
                pl.col_expand_mul(s_cur, eglast_row),
                pl.matmul(vi_bf, kdec_c, pl.FP32, a_trans=True))

        # Flush. The last chunk wrote the slot with the parity of the chunk count.
        chunks = (length + CHUNK - 1) // CHUNK
        last = chunks % 2
        final = pl.slice(state_gm, [V_TILE, KDA_DIM], [(task * 2 + last) * V_TILE, 0])
        state_2d[pool_row : pool_row + V_TILE, 0:KDA_DIM] = final

    return output, recurrent_state


__all__ = [
    "CHUNK",
    "DECAY_CAP",
    "golden_prefill_kda",
    "prefill_kda",
]


@pl.jit
def prefill_kda_test(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    tri_incl: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    tri_strict: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    head_onehot: pl.Tensor[[LOCAL_KDA_H, BETA_PAD], pl.FP32],
    eye_wide: pl.Tensor[[CHUNK, KDA_DIM], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    output, recurrent_state = prefill_kda(
        query, key, value, decay, beta, recurrent_state, query_start_loc, state_rows,
        tri_incl, tri_strict, head_onehot, eye_wide, output)
    return output, recurrent_state


def _golden_tensors(tensors) -> None:
    out, state = golden_prefill_kda(
        tensors["query"], tensors["key"], tensors["value"], tensors["decay"],
        tensors["beta"], tensors["recurrent_state"], tensors["query_start_loc"],
        tensors["state_rows"])
    tensors["output"][:] = out
    tensors["recurrent_state"][:] = state


_POOL_ROWS = 6
# One request spans several chunks with a ragged tail, one is shorter than a chunk,
# one is exactly one chunk. A single-chunk zero-state fixture passes with the state
# transposed and with the carry dropped, so the multi-chunk case is the real test.
# ``ragged`` is the default because it is the case that discriminates: three requests,
# mixed lengths, and a partial chunk after several full ones -- which is the only shape
# that exposes a chunk tail relying on padding it was never given. ``aligned`` is the
# complement, a whole number of chunks. ``zerostate`` and ``nobeta`` each remove one
# term from the recurrence, so a failure in them says which half is wrong.
_LENGTHS = {"ragged": [140, 7, 64], "single": [3], "aligned": [64],
            "zerostate": [3], "nobeta": [70]}


def build_tensor_specs(case: str = "ragged"):
    """``zerostate`` is a diagnostic: it isolates the carried state from everything else."""
    from golden import TensorSpec

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    lengths = _LENGTHS[case]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    rows = torch.tensor([4, 1, 5][: len(lengths)], dtype=torch.int32)

    return [
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        # Keys share a per-head direction. Independent random keys are near-orthogonal
        # at width 128, which makes the Gram matrix ~0.09 and the triangular inverse
        # indistinguishable from the identity -- stubbing the inverse out entirely still
        # passed against the earlier fixture. A shared component puts the off-diagonal
        # terms at O(1), where the inverse is what the result depends on.
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 0.4
                                       + torch.randn(1, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: torch.randn(tokens, LOCAL_KDA_H, KDA_DIM).to(bf)),
        # The real gate spans the whole (-5, 0) range. A near-zero fixture leaves
        # exp(gcum) close to 1, which hides both the decay reconstruction and the
        # overflow clamp that keeps it finite.
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32,
                   init_value=lambda: -5.0 * torch.rand(tokens, LOCAL_KDA_H, KDA_DIM)),
        # beta carries kda_projection's cube-width padding; only the first
        # LOCAL_KDA_H columns are read.
        TensorSpec("beta", [tokens, BETA_PAD], f32,
                   init_value=(lambda: torch.zeros(tokens, BETA_PAD)) if case == "nobeta"
                   else (lambda: torch.rand(tokens, BETA_PAD))),
        # Non-zero, so a dropped carry or a transposed seed is visible.
        TensorSpec("recurrent_state", [_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], f32,
                   init_value=(lambda: torch.zeros(_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM))
                   if case == "zerostate" else
                   (lambda: torch.randn(_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM) * 0.1)),
        TensorSpec("query_start_loc", [len(starts)], i32,
                   init_value=lambda: torch.tensor(starts, dtype=torch.int32)),
        TensorSpec("state_rows", [len(lengths)], i32, init_value=lambda: rows),
        TensorSpec("tri_incl", [CHUNK, CHUNK], f32,
                   init_value=lambda: torch.tril(torch.ones(CHUNK, CHUNK))),
        TensorSpec("tri_strict", [CHUNK, CHUNK], f32,
                   init_value=lambda: torch.tril(torch.ones(CHUNK, CHUNK), diagonal=-1)),
        # Row h selects head h out of beta's padded 16-wide row.
        TensorSpec("head_onehot", [LOCAL_KDA_H, BETA_PAD], f32,
                   init_value=lambda: torch.eye(BETA_PAD)[:LOCAL_KDA_H].contiguous()),
        # 2I in the left CHUNK columns, zero beside it: the padded width the triangular
        # inverse works at.
        # The identity at the padded width the triangular inverse works at.
        TensorSpec("eye_wide", [CHUNK, KDA_DIM], f32,
                   init_value=lambda: torch.nn.functional.pad(
                       torch.eye(CHUNK), (0, KDA_DIM - CHUNK))),
        TensorSpec("output", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged",
                        choices=["ragged", "single", "aligned", "zerostate", "nobeta"])
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run(
        fn=prefill_kda_test,
        specs=build_tensor_specs(args.case),
        golden_fn=_golden_tensors,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
