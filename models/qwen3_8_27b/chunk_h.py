# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_h: the inter-chunk state recurrence. Per chunk, per head,
with S the state entering the chunk,

    snapshot = S
    V_new    = U - W @ S
    S        = exp(g_last) * S + K^T (V_new * exp(g_last - g))

Chunks carry state, so work is parallel over heads and sequential within one.

`state` holds the snapshot ENTERING each chunk, which is what `chunk_o` reads. The state
LEAVING the last chunk is what an inference cache carries forward, and
`_gdn_chunk_h_final` writes that as well -- transposed to `[V, K]` and widened to FP32 --
into a row of the pool `decode_delta_rule` reads. The decode state is FP32 because fp16
does not survive a long rollout, and its value axis leads because that puts the
recurrence's `sum_k` on the contiguous axis.
"""
import pypto.language as pl
import torch

from chunk_cumsum import cumsum_ref
from config import QWEN3_8_27B
from delta_rule_reference import chunk_h_ref
from reference import flat_state, frob_ok, make_inputs, to_hT
from scaled_dot_kkt import kkt_ref
from solve_tril import solve_tril_ref
from wy_fast import CHUNK_TILE, wy_fast_ref

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens
STATE_DYN = pl.dynamic("GDN_STATE_DYN")     # (T // CHUNK_TILE) * H * D, the snapshot rows
FSTATE_DYN = pl.dynamic("GDN_FSTATE_DYN")   # rows of the decode state pool
ROWS_DYN = pl.dynamic("GDN_PACKED_ROWS")    # packed sequences sharing a prefill call

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension
GRP = H // HG                               # value heads sharing one key head
KEY_WIDTH = HG * D
VAL_WIDTH = H * D

# tiling
AIV_LANES = 2
ROW_TILE = CHUNK_TILE // AIV_LANES
STATE_TILE = D // AIV_LANES
HANDOFF_TILE = 64       # value rows per transpose step of the decode hand-off

# validation
# One chunk, the cross-chunk recurrence, then eight chunks. The token count must be a
# multiple of CHUNK_TILE.
DEFAULT_CASES = ((128, (0,)), (256, (0,)), (512, (0, 256)))
FINAL_CASE = 256        # the hand-off body: two chunks, so the state it leaves is not the first
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len


def _gdn_chunk_h(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    w: pl.Tensor[[T_DYN, H, D], pl.FP16],
    u: pl.Tensor[[T_DYN, H, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    row_base: pl.Tensor[[T_DYN], pl.INT32],
    state: pl.Out[pl.Tensor[[STATE_DYN, D], pl.FP16]],
    v_new: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE, and `state` hold (T // CHUNK_TILE) * H * D rows."""
    k.bind_dynamic(0, T_DYN)
    w.bind_dynamic(0, T_DYN)
    u.bind_dynamic(0, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    row_base.bind_dynamic(0, T_DYN)
    state.bind_dynamic(0, STATE_DYN)
    v_new.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(k, 0)
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    w_flat = pl.reshape(w, [t_dim, VAL_WIDTH])
    u_flat = pl.reshape(u, [t_dim, VAL_WIDTH])
    v_flat = pl.reshape(v_new, [t_dim, VAL_WIDTH])
    for hh in pl.spmd(H, name_hint="chunk_h",
                      optimizations=[pl.cross_core_slot(slot_num=1)]):
        for _ in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            s0 = pl.full([STATE_TILE, D], dtype=pl.FP32, value=0.0)
        for c, (s_cur,) in pl.range(t_dim // CHUNK_TILE, init_values=(s0,)):
            t0 = c * CHUNK_TILE
            if c > 0 and pl.read(row_base, [t0]) == t0:
                keep = 0.0
            else:
                keep = 1.0
            row = (c * H + hh) * D
            d0 = hh * D
            # GQA: K comes from key head hh // GRP; W, U and the state are per
            # value head.
            dg0 = (hh // GRP) * D

            g_last = pl.read(g_sum, [hh, t0 + CHUNK_TILE - 1])

            for state_aiv in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                sr0 = state_aiv * STATE_TILE
                s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
                s16 = pl.mul(s16, keep)
                state[row + sr0 : row + sr0 + STATE_TILE, 0:D] = s16   # state ENTERING this chunk
                s16_full = pl.aic_gather(s16)                          # the cube needs both halves
            wc = w_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            ws = pl.matmul(wc, s16_full, out_dtype=pl.FP32)

            # S = exp(g_last) * S + K^T @ (coeff * V_new). The decay goes on V, not
            # on K: K^T (V c) == (K c)^T V. A matmul whose transposed operand came
            # from the vector unit cannot be lowered (pypto#2902).
            for v_aiv in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                r0 = v_aiv * ROW_TILE
                # coeff[i] = exp(g_last - g_i), argument <= 0
                g_lane = g_sum[hh : hh + 1, t0 + r0 : t0 + r0 + ROW_TILE]
                zero_row = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
                neg_g = pl.sub(zero_row, g_lane)
                # exp before the [1,C]->[C,1] reshape, not after (pypto#2947)
                coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [ROW_TILE, 1])
                uc = u_flat[t0 + r0 : t0 + r0 + ROW_TILE, d0 : d0 + D]
                vc = pl.sub(pl.cast(uc, target_type=pl.FP32), pl.aiv_shard(ws))   # V_new = U - W @ S
                vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")
                v_flat[t0 + r0 : t0 + r0 + ROW_TILE, d0 : d0 + D] = vc16
                vs = pl.row_expand_mul(vc, coeff)
                vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
                vs16_full = pl.aic_gather(vs16)
            kc = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            kv = pl.matmul(kc, vs16_full, a_trans=True, out_dtype=pl.FP32)

            for _ in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                # decay = exp(g_last) on every row, so the lane builds its own width
                zero_d = pl.full([1, STATE_TILE], dtype=pl.FP32, value=0.0)
                decay = pl.reshape(pl.exp(pl.add(zero_d, g_last)), [STATE_TILE, 1])
                decay = pl.mul(decay, keep)
                s_next = pl.add(pl.row_expand_mul(s_cur, decay), pl.aiv_shard(kv))
            s_end, = pl.yield_(s_next)
    return state, v_new


gdn_chunk_h = pl.jit.inline(_gdn_chunk_h)
gdn_chunk_h_test = pl.jit(_gdn_chunk_h)


def _gdn_chunk_h_final(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    w: pl.Tensor[[T_DYN, H, D], pl.FP16],
    u: pl.Tensor[[T_DYN, H, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    row_base: pl.Tensor[[T_DYN], pl.INT32],
    state: pl.Out[pl.Tensor[[STATE_DYN, D], pl.FP16]],
    v_new: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
    state_row: pl.Scalar[pl.INDEX],
    final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, D, D], pl.FP32]],
):
    """As `_gdn_chunk_h`, seeded by and updating the decode state pool.

    `state_row` names the pool row this sequence owns, the same indirection
    `decode_delta_rule` uses, so the hand-off is a write into the pool rather than a
    copy of 3 MiB per sequence.
    """
    k.bind_dynamic(0, T_DYN)
    w.bind_dynamic(0, T_DYN)
    u.bind_dynamic(0, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    row_base.bind_dynamic(0, T_DYN)
    state.bind_dynamic(0, STATE_DYN)
    v_new.bind_dynamic(0, T_DYN)
    final_state.bind_dynamic(0, FSTATE_DYN)
    t_dim = pl.tensor.dim(k, 0)
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    w_flat = pl.reshape(w, [t_dim, VAL_WIDTH])
    u_flat = pl.reshape(u, [t_dim, VAL_WIDTH])
    v_flat = pl.reshape(v_new, [t_dim, VAL_WIDTH])
    # The pool is addressed through a flat view of the same buffer, as
    # `decode_delta_rule` addresses it: a `[1, H, D, D]` subview would need a
    # single-element row, which ptoas rejects on its 32-byte row alignment.
    fs_2d = pl.reshape(final_state, [pl.tensor.dim(final_state, 0) * H * D, D])
    fs_row = state_row
    carry = pl.create_tensor([H * D, D], dtype=pl.FP32)
    # The decode pool is [V, K]; the chunk recurrence carries [K, V]. A separate
    # pass keeps the transpose outside the split mixed kernel below.
    for hh0 in pl.spmd(H, name_hint="chunk_h_resume"):
        crow = hh0 * D
        frow = (fs_row * H + hh0) * D
        for v0 in pl.range(0, D, HANDOFF_TILE):
            seed_band = fs_2d[frow + v0 : frow + v0 + HANDOFF_TILE, 0:D]
            carry[crow : crow + D, v0 : v0 + HANDOFF_TILE] = pl.transpose(seed_band, 0, 1)

    for hh in pl.spmd(H, name_hint="chunk_h",
                      optimizations=[pl.cross_core_slot(slot_num=1)]):
        for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            row_start = hh * D + lane * STATE_TILE
            s0 = carry[row_start : row_start + STATE_TILE, 0:D]
        for c, (s_cur,) in pl.range(t_dim // CHUNK_TILE, init_values=(s0,)):
            t0 = c * CHUNK_TILE
            if c > 0 and pl.read(row_base, [t0]) == t0:
                keep = 0.0
            else:
                keep = 1.0
            row = (c * H + hh) * D
            d0 = hh * D
            # GQA: K comes from key head hh // GRP; W, U and the state are per
            # value head.
            dg0 = (hh // GRP) * D

            g_last = pl.read(g_sum, [hh, t0 + CHUNK_TILE - 1])

            for state_aiv in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                sr0 = state_aiv * STATE_TILE
                s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
                s16 = pl.mul(s16, keep)
                state[row + sr0 : row + sr0 + STATE_TILE, 0:D] = s16   # state ENTERING this chunk
                s16_full = pl.aic_gather(s16)                          # the cube needs both halves
            wc = w_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            ws = pl.matmul(wc, s16_full, out_dtype=pl.FP32)

            # S = exp(g_last) * S + K^T @ (coeff * V_new). The decay goes on V, not
            # on K: K^T (V c) == (K c)^T V. A matmul whose transposed operand came
            # from the vector unit cannot be lowered (pypto#2902).
            for v_aiv in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                r0 = v_aiv * ROW_TILE
                # coeff[i] = exp(g_last - g_i), argument <= 0
                g_lane = g_sum[hh : hh + 1, t0 + r0 : t0 + r0 + ROW_TILE]
                zero_row = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
                neg_g = pl.sub(zero_row, g_lane)
                # exp before the [1,C]->[C,1] reshape, not after (pypto#2947)
                coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [ROW_TILE, 1])
                uc = u_flat[t0 + r0 : t0 + r0 + ROW_TILE, d0 : d0 + D]
                vc = pl.sub(pl.cast(uc, target_type=pl.FP32), pl.aiv_shard(ws))   # V_new = U - W @ S
                vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")
                v_flat[t0 + r0 : t0 + r0 + ROW_TILE, d0 : d0 + D] = vc16
                vs = pl.row_expand_mul(vc, coeff)
                vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
                vs16_full = pl.aic_gather(vs16)
            kc = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            kv = pl.matmul(kc, vs16_full, a_trans=True, out_dtype=pl.FP32)

            for _ in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                # decay = exp(g_last) on every row, so the lane builds its own width
                zero_d = pl.full([1, STATE_TILE], dtype=pl.FP32, value=0.0)
                decay = pl.reshape(pl.exp(pl.add(zero_d, g_last)), [STATE_TILE, 1])
                decay = pl.mul(decay, keep)
                s_next = pl.add(pl.row_expand_mul(s_cur, decay), pl.aiv_shard(kv))
            s_end, = pl.yield_(s_next)

        # The state the loop ends with, still [K, V] and still FP32, parked for the
        # transpose pass below.
        for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            row_start = hh * D + lane * STATE_TILE
            carry[row_start : row_start + STATE_TILE, 0:D] = s_end

    # Transpose the FP32 recurrence state into the decode pool layout.
    for hh2 in pl.spmd(H, name_hint="chunk_h_handoff"):
        crow = hh2 * D
        frow = (fs_row * H + hh2) * D
        # In HANDOFF_TILE bands of the value axis: a whole head transposed at once
        # needs three [D, D] FP32 tiles live, 196608 B of a 188416 B buffer, and a
        # band of 64 value rows needs three of half that. `decode_delta_rule` tiles
        # its own state the same way for the same reason.
        for v0 in pl.range(0, D, HANDOFF_TILE):
            band = pl.slice(carry, [D, HANDOFF_TILE], [crow, v0])
            fs_2d[frow + v0 : frow + v0 + HANDOFF_TILE, 0:D] = pl.transpose(band, 0, 1)
    return state, v_new, final_state


gdn_chunk_h_final = pl.jit.inline(_gdn_chunk_h_final)
gdn_chunk_h_final_test = pl.jit(_gdn_chunk_h_final)


def _gdn_chunk_h_final_packed(
    k: pl.Tensor[[T_DYN, HG, D], pl.FP16],
    w: pl.Tensor[[T_DYN, H, D], pl.FP16],
    u: pl.Tensor[[T_DYN, H, D], pl.FP16],
    g_sum: pl.Tensor[[H, T_DYN], pl.FP32],
    state_rows: pl.Tensor[[ROWS_DYN], pl.INT32],
    row_ends: pl.Tensor[[ROWS_DYN], pl.INT32],
    state: pl.Out[pl.Tensor[[STATE_DYN, D], pl.FP16]],
    v_new: pl.Out[pl.Tensor[[T_DYN, H, D], pl.FP16]],
    final_state: pl.InOut[pl.Tensor[[FSTATE_DYN, H, D, D], pl.FP32]],
):
    """Resume and publish the recurrent state for every packed row's cache owner."""
    k.bind_dynamic(0, T_DYN)
    w.bind_dynamic(0, T_DYN)
    u.bind_dynamic(0, T_DYN)
    g_sum.bind_dynamic(1, T_DYN)
    state_rows.bind_dynamic(0, ROWS_DYN)
    row_ends.bind_dynamic(0, ROWS_DYN)
    state.bind_dynamic(0, STATE_DYN)
    v_new.bind_dynamic(0, T_DYN)
    final_state.bind_dynamic(0, FSTATE_DYN)
    t_dim = pl.tensor.dim(k, 0)
    rows = pl.tensor.dim(state_rows, 0)
    k_flat = pl.reshape(k, [t_dim, KEY_WIDTH])
    w_flat = pl.reshape(w, [t_dim, VAL_WIDTH])
    u_flat = pl.reshape(u, [t_dim, VAL_WIDTH])
    v_flat = pl.reshape(v_new, [t_dim, VAL_WIDTH])
    fs_2d = pl.reshape(final_state, [pl.tensor.dim(final_state, 0) * H * D, D])
    seeds = pl.create_tensor([rows * H * D, D], dtype=pl.FP32)
    published = pl.create_tensor([rows * H * D, D], dtype=pl.FP32)
    for seed_idx in pl.spmd(rows * H, name_hint="chunk_h_packed_resume"):
        owner = seed_idx // H
        hh0 = seed_idx % H
        state_row = pl.cast(pl.read(state_rows, [owner]), pl.INDEX)
        crow = seed_idx * D
        frow = (state_row * H + hh0) * D
        for v0 in pl.range(0, D, HANDOFF_TILE):
            seed_band = fs_2d[frow + v0 : frow + v0 + HANDOFF_TILE, 0:D]
            seeds[crow : crow + D, v0 : v0 + HANDOFF_TILE] = pl.transpose(seed_band, 0, 1)

    for work in pl.spmd(rows * H, name_hint="chunk_h_packed",
                      optimizations=[pl.cross_core_slot(slot_num=1)]):
        owner = work // H
        hh = work % H
        end = pl.cast(pl.read(row_ends, [owner]), pl.INDEX)
        if owner == 0:
            start = 0
        else:
            start = pl.cast(pl.read(row_ends, [owner - 1]), pl.INDEX)
        for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            row_start = work * D + lane * STATE_TILE
            s0 = seeds[row_start : row_start + STATE_TILE, 0:D]
        for c, (s_cur,) in pl.range((end - start) // CHUNK_TILE, init_values=(s0,)):
            t0 = start + c * CHUNK_TILE
            row = ((t0 // CHUNK_TILE) * H + hh) * D
            d0 = hh * D
            dg0 = (hh // GRP) * D
            g_last = pl.read(g_sum, [hh, t0 + CHUNK_TILE - 1])

            for state_aiv in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                sr0 = state_aiv * STATE_TILE
                s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
                state[row + sr0 : row + sr0 + STATE_TILE, 0:D] = s16   # state ENTERING this chunk
                s16_full = pl.aic_gather(s16)                          # the cube needs both halves
            wc = w_flat[t0 : t0 + CHUNK_TILE, d0 : d0 + D]
            ws = pl.matmul(wc, s16_full, out_dtype=pl.FP32)

            # S = exp(g_last) * S + K^T @ (coeff * V_new). The decay goes on V, not
            # on K: K^T (V c) == (K c)^T V. A matmul whose transposed operand came
            # from the vector unit cannot be lowered (pypto#2902).
            for v_aiv in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                r0 = v_aiv * ROW_TILE
                # coeff[i] = exp(g_last - g_i), argument <= 0
                g_lane = g_sum[hh : hh + 1, t0 + r0 : t0 + r0 + ROW_TILE]
                zero_row = pl.full([1, ROW_TILE], dtype=pl.FP32, value=0.0)
                neg_g = pl.sub(zero_row, g_lane)
                # exp before the [1,C]->[C,1] reshape, not after (pypto#2947)
                coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [ROW_TILE, 1])
                uc = u_flat[t0 + r0 : t0 + r0 + ROW_TILE, d0 : d0 + D]
                vc = pl.sub(pl.cast(uc, target_type=pl.FP32), pl.aiv_shard(ws))   # V_new = U - W @ S
                vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")
                v_flat[t0 + r0 : t0 + r0 + ROW_TILE, d0 : d0 + D] = vc16
                vs = pl.row_expand_mul(vc, coeff)
                vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
                vs16_full = pl.aic_gather(vs16)
            kc = k_flat[t0 : t0 + CHUNK_TILE, dg0 : dg0 + D]
            kv = pl.matmul(kc, vs16_full, a_trans=True, out_dtype=pl.FP32)

            for _ in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                # decay = exp(g_last) on every row, so the lane builds its own width
                zero_d = pl.full([1, STATE_TILE], dtype=pl.FP32, value=0.0)
                decay = pl.reshape(pl.exp(pl.add(zero_d, g_last)), [STATE_TILE, 1])
                s_next = pl.add(pl.row_expand_mul(s_cur, decay), pl.aiv_shard(kv))
            s_end, = pl.yield_(s_next)
        for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            row_start = work * D + lane * STATE_TILE
            published[row_start : row_start + STATE_TILE, 0:D] = s_end

    for out_idx in pl.spmd(rows * H, name_hint="chunk_h_packed_handoff"):
        owner = out_idx // H
        hh2 = out_idx % H
        state_row = pl.cast(pl.read(state_rows, [owner]), pl.INDEX)
        crow = out_idx * D
        frow = (state_row * H + hh2) * D
        for v0 in pl.range(0, D, HANDOFF_TILE):
            band = pl.slice(published, [D, HANDOFF_TILE], [crow, v0])
            fs_2d[frow + v0 : frow + v0 + HANDOFF_TILE, 0:D] = pl.transpose(band, 0, 1)
    return state, v_new, final_state


gdn_chunk_h_final_packed = pl.jit.inline(_gdn_chunk_h_final_packed)
gdn_chunk_h_final_packed_test = pl.jit(_gdn_chunk_h_final_packed)


_INPUTS: dict = {}


def _inputs(t: int) -> dict:
    """The chain this stage consumes, memoised so the several specs pay for it once."""
    key = t
    if key not in _INPUTS:
        st = make_inputs(t, H, D, HG)
        st["g_sum"] = cumsum_ref(st["g"], CHUNK_TILE)
        a16 = kkt_ref(st["k"], st["beta"], st["g_sum"], CHUNK_TILE).to(torch.float16)
        a_inv16 = solve_tril_ref(a16, CHUNK_TILE).to(torch.float16)
        w, u = wy_fast_ref(st["k"], st["v"], st["beta"], a_inv16, st["g_sum"], CHUNK_TILE)
        st["w16"] = w.to(torch.float16)
        st["u16"] = u.to(torch.float16)
        _INPUTS[key] = st
    return _INPUTS[key]


def row_base_table(t: int, starts: tuple[int, ...] = (0,)) -> torch.Tensor:
    """Give each token the packed start of its 256-aligned sequence row."""
    if not starts or starts[0] != 0 or any(a >= b for a, b in zip(starts, starts[1:])):
        raise ValueError("row starts must be strictly increasing from zero")
    if starts[-1] >= t or any(start % 256 for start in starts):
        raise ValueError("row starts must be 256-aligned and inside the token buffer")
    table = torch.empty(t, dtype=torch.int32)
    for start, end in zip(starts, (*starts[1:], t)):
        table[start:end] = start
    return table


def row_reset_control() -> None:
    """Check packed row isolation both from zero and a resumed first row."""
    gen = torch.Generator().manual_seed(47)
    t, h, hg, d = 512, 2, 1, 4
    k = torch.randn(t, hg, d, generator=gen) * 0.1
    w = torch.randn(t, h, d, generator=gen) * 0.1
    u = torch.randn(t, h, d, generator=gen) * 0.1
    g_sum = torch.zeros(t, h)
    base = row_base_table(t, (0, 256))
    packed = chunk_h_ref(k, w, u, g_sum, CHUNK_TILE, row_base=base)
    first = chunk_h_ref(k[:256], w[:256], u[:256], g_sum[:256], CHUNK_TILE)
    second = chunk_h_ref(k[256:], w[256:], u[256:], g_sum[256:], CHUNK_TILE)
    if not torch.equal(packed[0], torch.cat((first[0], second[0]))):
        raise RuntimeError("packed state snapshots crossed a row boundary")
    if not torch.equal(packed[1], torch.cat((first[1], second[1]))):
        raise RuntimeError("packed corrected values crossed a row boundary")
    if not torch.equal(packed[2], second[2]):
        raise RuntimeError("packed final state differs from the final row")
    unreset = chunk_h_ref(k, w, u, g_sum, CHUNK_TILE)
    if torch.equal(unreset[0][2], packed[0][2]):
        raise RuntimeError("row-reset control cannot detect a carried prior row")
    initial = torch.randn(h, d, d, generator=gen) * 0.03
    resumed = chunk_h_ref(k, w, u, g_sum, CHUNK_TILE, initial, base)
    first_resumed = chunk_h_ref(k[:256], w[:256], u[:256], g_sum[:256],
                                CHUNK_TILE, initial)
    if not torch.equal(resumed[0], torch.cat((first_resumed[0], second[0]))):
        raise RuntimeError("resumed packed state snapshots crossed a row boundary")
    if not torch.equal(resumed[1], torch.cat((first_resumed[1], second[1]))):
        raise RuntimeError("resumed packed corrected values crossed a row boundary")
    if not torch.equal(resumed[2], second[2]):
        raise RuntimeError("resumed packed final state differs from the final row")


def build_tensor_specs(t: int = BENCH_SEQ, starts: tuple[int, ...] = (0,)):
    from golden import TensorSpec

    def draw(key, transform=None):
        def make():
            value = _inputs(t)[key]
            return transform(value) if transform is not None else value

        return make

    nc = t // CHUNK_TILE

    return [
        TensorSpec("k", [t, HG, D], torch.float16,
                   init_value=draw("k")),
        TensorSpec("w", [t, H, D], torch.float16,
                   init_value=draw("w16")),
        TensorSpec("u", [t, H, D], torch.float16,
                   init_value=draw("u16")),
        TensorSpec("g_sum", [H, t], torch.float32,
                   init_value=draw("g_sum", to_hT)),
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: row_base_table(t, starts)),
        TensorSpec("state", [nc * H * D, D], torch.float16),
        TensorSpec("v_new", [t, H, D], torch.float16),
    ]


def golden_gdn_chunk_h(tensors):
    t, _, d = tensors["k"].shape          # k has Hg heads under GQA
    h = tensors["w"].shape[1]             # W, U and the state are per value head
    chunk = t // (tensors["state"].shape[0] // (h * d))
    state, v_new, _ = chunk_h_ref(tensors["k"], tensors["w"], tensors["u"],
                                 tensors["g_sum"].t(), chunk, row_base=tensors["row_base"])
    tensors["state"].copy_(flat_state(state))
    tensors["v_new"].copy_(v_new)


def build_final_tensor_specs(t: int = BENCH_SEQ, pool_rows: int = 4, state_row: int = 2,
                             resume: bool = False, starts: tuple[int, ...] = (0,)):
    """`build_tensor_specs` plus the decode pool and the row this sequence owns.

    `state_row` defaults away from 0 so a kernel that ignored `state_row` and wrote row 0
    would leave the gated row untouched and pass.
    """
    from golden import ScalarSpec, TensorSpec

    def pool():
        # Non-zero, so a row the kernel fails to write shows up as the fill rather than
        # as a plausible zero state.
        rows = torch.full((pool_rows, H, D, D), -1.0)
        if resume:
            gen = torch.Generator().manual_seed(91)
            rows[state_row] = torch.randn(H, D, D, generator=gen) * 0.02
        else:
            rows[state_row] = 0.0
        return rows

    return build_tensor_specs(t=t, starts=starts) + [
        ScalarSpec("state_row", torch.int64, state_row, compile_runtime=True),
        TensorSpec("final_state", [pool_rows, H, D, D], torch.float32, init_value=pool),
    ]


def golden_gdn_chunk_h_final(tensors):
    """The snapshots and V_new as before, plus the hand-off transposed and widened."""
    t, _, d = tensors["k"].shape
    h = tensors["w"].shape[1]
    chunk = t // (tensors["state"].shape[0] // (h * d))
    row = int(tensors["state_row"])
    initial = tensors["final_state"][row].transpose(-2, -1).contiguous()
    state, v_new, final = chunk_h_ref(tensors["k"], tensors["w"], tensors["u"],
                                      tensors["g_sum"].t(), chunk, initial,
                                      row_base=tensors["row_base"])
    tensors["state"].copy_(flat_state(state))
    tensors["v_new"].copy_(v_new)
    # `chunk_h_ref` returns the hand-off as [H, K, V]; the decode pool holds [V, K].
    tensors["final_state"][row] = final.transpose(-2, -1).to(tensors["final_state"].dtype)


def build_packed_final_tensor_specs(t: int = 512, starts: tuple[int, ...] = (0, 256),
                                    state_rows: tuple[int, ...] = (1, 3),
                                    pool_rows: int = 5, resume: bool = False):
    from golden import TensorSpec

    if (len(starts) != len(state_rows) or starts[0] != 0 or t % 256
            or any(start % 256 for start in starts)
            or any(b - a < 256 for a, b in zip(starts, starts[1:] + (t,)))
            or len(set(state_rows)) != len(state_rows)
            or any(row < 0 or row >= pool_rows for row in state_rows)):
        raise ValueError("packed final needs disjoint state rows and complete 256-token spans")
    ends = starts[1:] + (t,)

    def pool():
        value = torch.full((pool_rows, H, D, D), -1.0)
        for owner, row in enumerate(state_rows):
            if resume:
                gen = torch.Generator().manual_seed(91 + owner)
                value[row] = torch.randn(H, D, D, generator=gen) * 0.02
            else:
                value[row] = 0.0
        return value

    base = build_tensor_specs(t=t, starts=starts)
    return base[:4] + [
        TensorSpec("state_rows", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(state_rows, dtype=torch.int32)),
        TensorSpec("row_ends", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(ends, dtype=torch.int32)),
    ] + base[5:] + [
        TensorSpec("final_state", [pool_rows, H, D, D], torch.float32, init_value=pool),
    ]


def golden_gdn_chunk_h_final_packed(tensors):
    ends = tensors["row_ends"].tolist()
    starts = [0] + ends[:-1]
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row = int(tensors["state_rows"][owner])
        initial = tensors["final_state"][row].transpose(-2, -1).contiguous()
        state, v_new, final = chunk_h_ref(
            tensors["k"][start:end], tensors["w"][start:end], tensors["u"][start:end],
            tensors["g_sum"][:, start:end].t(), CHUNK_TILE, initial)
        first = start // CHUNK_TILE * H * D
        last = end // CHUNK_TILE * H * D
        tensors["state"][first:last].copy_(flat_state(state))
        tensors["v_new"][start:end].copy_(v_new)
        tensors["final_state"][row].copy_(final.transpose(-2, -1))


def compare_packed_final_state(state_rows: tuple[int, ...]):
    scored = frob_ok()

    def compare(actual, expected, **kwargs):
        for row in state_rows:
            ok, detail = scored(actual[row], expected[row], **kwargs)
            if not ok:
                return False, f"row {row}: {detail}"
        others = [row for row in range(actual.shape[0]) if row not in state_rows]
        if others and not bool((actual[others] == -1.0).all()):
            return False, f"unowned state rows {others} were modified"
        return True, "packed recurrent states pass; unowned rows untouched"

    return compare


def compare_final_state(state_row: int):
    """Score the row the kernel wrote, and prove it left the others alone.

    A whole-pool norm would be diluted: with four rows and one written, three quarters of
    the tensor is trivially equal. The untouched rows are also the control -- they carry
    -1.0, which no state takes, so a kernel that ignored `state_row` shows up as a row
    that should have been left alone and was not.
    """
    scored = frob_ok()

    def compare(actual, expected, **kwargs):
        ok, detail = scored(actual[state_row], expected[state_row], **kwargs)
        others = [r for r in range(actual.shape[0]) if r != state_row]
        if others and not bool((actual[others] == -1.0).all()):
            return False, f"{detail}\n    rows {others} were written and should not have been"
        return ok, detail

    compare.__name__ = "compare_final_state"
    return compare


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--final", action="store_true", default=False,
                        help="only the body that also writes the hand-off state")
    parser.add_argument("--packed-final", action="store_true", default=False,
                        help="gate independent packed-row recurrent state publication")
    parser.add_argument("--initial-state", choices=["zero", "nonzero"],
                        help="select one initialization for a state-publishing program")
    parser.add_argument("--row-starts", type=str, default="0",
                        help="comma-separated packed row starts, 256-aligned")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    row_reset_control()
    starts = tuple(int(value) for value in args.row_starts.split(","))
    cases = [(args.seq_len, starts)] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")
    if args.final and args.packed_final:
        parser.error("choose either --final or --packed-final")
    if args.initial_state and not (args.final or args.packed_final):
        parser.error("--initial-state requires --final or --packed-final")
    replay = args.golden_data is not None or args.runtime_dir is not None
    if replay and not args.seq_len:
        parser.error("--golden-data and --runtime-dir require --seq-len")
    publish_final = args.final or not (args.packed_final or args.save_data or replay)
    initial_states = (False, True) if args.initial_state is None else (args.initial_state == "nonzero",)

    # Both bodies by default: the hand-off is a second program, not a branch inside the
    # first, so the plain entry cannot cover it.
    plan = [] if args.final or args.packed_final else [
        (f"chunk_h T={t} starts={starts}", t, starts, gdn_chunk_h_test,
         build_tensor_specs, golden_gdn_chunk_h,
         {"state": frob_ok(), "v_new": frob_ok()}) for t, starts in cases]
    plan += [
        (f"chunk_h final T={t} starts={starts} resume={resume}", t, starts, gdn_chunk_h_final_test,
         lambda t, starts, resume=resume: build_final_tensor_specs(t=t, starts=starts, resume=resume),
         golden_gdn_chunk_h_final,
         {"state": frob_ok(), "v_new": frob_ok(), "final_state": compare_final_state(2)})
        for t, starts in (cases if args.seq_len or args.final else [(FINAL_CASE, (0,)), (512, (0, 256))])
        for resume in initial_states] if publish_final else []
    if args.packed_final:
        if not args.seq_len or len(starts) < 2:
            parser.error("--packed-final needs --seq-len and at least two --row-starts")
        state_rows = tuple(2 * owner + 1 for owner in range(len(starts)))
        plan += [
            (f"chunk_h packed final T={args.seq_len} starts={starts} resume={resume}",
             args.seq_len, starts, gdn_chunk_h_final_packed_test,
             lambda t, starts, resume=resume: build_packed_final_tensor_specs(
                 t=t, starts=starts, state_rows=state_rows,
                 pool_rows=2 * len(starts) + 1, resume=resume),
             golden_gdn_chunk_h_final_packed,
             {"state": frob_ok(), "v_new": frob_ok(),
              "final_state": compare_packed_final_state(state_rows)})
            for resume in initial_states]
    if replay and len(plan) != 1:
        parser.error("state-publishing replay requires --initial-state zero or nonzero")

    for label, t, starts, fn, specs_fn, golden_fn, compare_fn in plan:
        print(f"[case] {label}", flush=True)
        result = run(
            fn=fn,
            specs=specs_fn(t=t, starts=starts),
            golden_fn=golden_fn,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            compile_only=args.compile_only,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            compare_fn=compare_fn,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run lane, so
            # every later case in this process reports a failure it did not cause.
            raise SystemExit(f"FAILED at {label}")
