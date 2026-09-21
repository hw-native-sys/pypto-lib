# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The single-token KDA recurrence used by decode.

Per head, with the state ``S`` in FP32 and the value axis leading:

    S       <- S * exp(g)                     # g is per key channel: scales columns
    kv_mem  <- sum_k S[v, k] * key[k]
    delta   <- (value - kv_mem) * beta
    S[v, k] <- S[v, k] + delta[v] * key[k]
    out     <- sum_k S[v, k] * query[k]

q and k are L2-normalised in FP32 with ``eps = 1e-6`` and q carries the
``KDA_DIM ** -0.5`` scale, both inside the kernel, matching
``use_qk_l2norm_in_kernel=True``.

**The state's last two axes are ``[V, K]``, not ``[K, V]``.** That is the layout the
AscendC operator's ABI fixes (``state_v_first=True``) and it also puts the ``sum_k``
reduction along the contiguous axis, where it is a plain row sum on the decode
critical path. ``KDA_DIM`` is 128 on both axes, so **no shape check can ever catch a
transposed state** — it surfaces as wrong output from the first decode step onward.

Every step of the recurrence is independent across ``v``, so the state is tiled over
its value axis: a whole head's ``[128, 128]`` FP32 state is 64 KiB and three live
intermediates of that size would not fit UB, while ``V_TILE``-row slices do.

One token per state row is assumed, which is what a non-speculative decode step
produces. Two tokens sharing a row would be a sequential dependency, and the tasks
here are independent.
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

from models.glm5_3_flash.config import KDA_DIM, KDA_STATE_DYN, LOCAL_KDA_H, T_DYN
from models.glm5_3_flash.kda_projection import BETA_PAD
from models.glm5_3_flash.golden import l2norm

L2_EPS = 1e-6
Q_SCALE = KDA_DIM ** -0.5

# tiling
V_TILE = 64  # value rows per task: a [64, 128] FP32 slice is 32 KiB
NORM_ROWS = 16  # (token, head) rows per normalisation task
COL_REP = 16  # replication width used to turn a row into a legal column


def golden_decode_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    recurrent_state: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(output, recurrent_state)``; the state is returned updated.

    ``recurrent_state`` is ``[rows, LOCAL_KDA_H, V, K]``.
    """
    tokens = query.shape[0]
    state = recurrent_state.clone().float()
    out = torch.zeros(tokens, LOCAL_KDA_H, KDA_DIM, dtype=torch.float32)
    for t in range(tokens):
        row = int(state_rows[t])
        for h in range(LOCAL_KDA_H):
            q = l2norm(query[t, h].float(), eps=L2_EPS) * Q_SCALE  # [K]
            k = l2norm(key[t, h].float(), eps=L2_EPS)  # [K]
            v = value[t, h].float()  # [V]
            g = decay[t, h].float()  # [K]
            b = float(beta[t, h])
            s = state[row, h] * torch.exp(g).unsqueeze(0)  # [V, K]
            kv_mem = (s * k.unsqueeze(0)).sum(dim=-1)  # [V]
            delta = (v - kv_mem) * b  # [V]
            s = s + delta.unsqueeze(-1) * k.unsqueeze(0)
            state[row, h] = s
            out[t, h] = (s * q.unsqueeze(0)).sum(dim=-1)
    return out.to(query.dtype), state.to(recurrent_state.dtype)


@pl.jit.inline
def decode_kda(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(query, 0)
    lanes = KDA_DIM // V_TILE
    # A matmul result is logically one row but physically a whole M fractal, so it
    # cannot be reshaped into the output's rank-3 view. Writing through a 2D view of
    # the output keeps the store a plain 2D assignment, with the valid row declared.
    out_rows = pl.reshape(output, [t_dim, LOCAL_KDA_H * KDA_DIM])
    tasks = t_dim * LOCAL_KDA_H * lanes

    # q and k are L2-normalised first, over a tall view with one row per (token, head).
    # Doing it inside the main task would reduce a single [1, 128] row, and the [1, 1]
    # result is a column of one FP32 -- four bytes, which is not a legal tile column.
    # The taller tile also computes each norm once instead of once per value block.
    rows_total = t_dim * LOCAL_KDA_H
    norm_tiles = (rows_total + NORM_ROWS - 1) // NORM_ROWS
    q_rows = pl.reshape(query, [rows_total, KDA_DIM])
    k_rows = pl.reshape(key, [rows_total, KDA_DIM])
    qn_gm = pl.create_tensor([norm_tiles * NORM_ROWS, KDA_DIM], dtype=pl.FP32)
    kn_gm = pl.create_tensor([norm_tiles * NORM_ROWS, KDA_DIM], dtype=pl.FP32)
    with pl.spmd(norm_tiles, name_hint="decode_kda_l2norm") as norm_tid:
        r0 = pl.tile.get_block_idx() * NORM_ROWS
        rows = pl.min(NORM_ROWS, rows_total - r0)
        qt = pl.cast(pl.slice(q_rows, [NORM_ROWS, KDA_DIM], [r0, 0],
                              valid_shape=[rows, KDA_DIM]), pl.FP32)
        kt = pl.cast(pl.slice(k_rows, [NORM_ROWS, KDA_DIM], [r0, 0],
                              valid_shape=[rows, KDA_DIM]), pl.FP32)
        # FLA adds eps inside the square root rather than clamping, and the a2a3
        # reference uses rsqrt; both are matched here. q takes the 1/sqrt(d) scale.
        qn_gm[r0 : r0 + NORM_ROWS, 0:KDA_DIM] = pl.mul(
            pl.row_expand_mul(qt, pl.rsqrt(pl.add(pl.row_sum(pl.mul(qt, qt)), L2_EPS),
                                           high_precision=True)), Q_SCALE)
        kn_gm[r0 : r0 + NORM_ROWS, 0:KDA_DIM] = pl.row_expand_mul(
            kt, pl.rsqrt(pl.add(pl.row_sum(pl.mul(kt, kt)), L2_EPS), high_precision=True))

    # A store whose source is a tile reshaped to two leading unit axes lowers to a
    # tile with a single-element row, which ptoas rejects on its 32-byte row
    # alignment. Addressing the pool through a flat view of the same buffer keeps
    # the stored tile the shape the arithmetic already produced.
    state_2d = pl.reshape(
        recurrent_state,
        [pl.tensor.dim(recurrent_state, 0) * LOCAL_KDA_H * KDA_DIM, KDA_DIM])

    with pl.spmd(tasks, name_hint="decode_kda", deps=[norm_tid]):
        task = pl.tile.get_block_idx()
        t = task // (LOCAL_KDA_H * lanes)
        h = (task // lanes) % LOCAL_KDA_H
        v0 = (task % lanes) * V_TILE
        row = pl.cast(pl.read(state_rows, [t]), pl.INDEX)

        qk_row = t * LOCAL_KDA_H + h
        q_row = qn_gm[qk_row : qk_row + 1, 0:KDA_DIM]
        k_row = kn_gm[qk_row : qk_row + 1, 0:KDA_DIM]
        # decay holds the log decay; the recurrence exponentiates it.
        g_row = pl.exp(pl.reshape(
            pl.slice(decay, [1, 1, KDA_DIM], [t, h, 0], drop_dims=[0]), [1, KDA_DIM]))
        beta_s = pl.read(beta, [t, h])

        # This task owns value rows [v0, v0 + V_TILE) of one head's state.
        srow = (row * LOCAL_KDA_H + h) * KDA_DIM + v0
        state_tile = pl.slice(state_2d, [V_TILE, KDA_DIM], [srow, 0])
        # kv_mem is built as a column by a reduction, which is the shape the delta and
        # the rank-1 update both want. A matmul would give it as a row, but a matmul
        # result is logically one row and physically a whole M fractal, and that
        # physical shape propagates into every op that touches it -- including through
        # a GM round trip, which the compiler forwards.
        decayed = pl.col_expand_mul(state_tile, g_row)
        kv_col = pl.row_sum(pl.col_expand_mul(decayed, k_row))  # [V_TILE, 1]
        v_row = pl.cast(
            pl.reshape(pl.slice(value, [1, 1, V_TILE], [t, h, v0], drop_dims=[0]),
                       [1, V_TILE]), pl.FP32)
        # value arrives as a row and is needed as a column. Transposing a row into a
        # column is not expressible -- the result is row-major with a four-byte row --
        # but a column is legal when a reduction produces it, so the row is broadcast
        # down COL_REP rows, transposed the legal way round, summed and scaled back.
        v_col = pl.mul(pl.row_sum(pl.transpose(pl.col_expand_mul(
            pl.full([COL_REP, V_TILE], dtype=pl.FP32, value=1.0), v_row), 0, 1)),
            1.0 / COL_REP)
        delta_col = pl.mul(pl.sub(v_col, kv_col), beta_s)  # [V_TILE, 1]
        # The rank-1 update delta (x) k, from the two broadcasts: a width-one cube
        # contraction is not addressable.
        updated = pl.add(decayed, pl.col_expand_mul(
            pl.row_expand_mul(pl.full([V_TILE, KDA_DIM], dtype=pl.FP32, value=1.0), delta_col),
            k_row))
        state_2d[srow : srow + V_TILE, 0:KDA_DIM] = updated

        # The output is the one place a matmul is right: its result feeds a store, so
        # the fractal padding never reaches another operand.
        out_row = pl.matmul(q_row, updated, pl.FP32, b_trans=True)  # [1, V_TILE]
        ocol = h * KDA_DIM + v0
        out_rows[t : t + 1, ocol : ocol + V_TILE] = pl.set_validshape(
            pl.cast(out_row, pl.BF16, mode="rint"), 1, V_TILE)

    return output, recurrent_state


__all__ = [
    "decode_kda",
    "golden_decode_kda",
]


@pl.jit
def decode_kda_test(
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    decay: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.FP32],
    beta: pl.Tensor[[T_DYN, BETA_PAD], pl.FP32],
    recurrent_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    output, recurrent_state = decode_kda(
        query, key, value, decay, beta, recurrent_state, state_rows, output)
    return output, recurrent_state


def _golden_tensors(tensors) -> None:
    out, state = golden_decode_kda(
        tensors["query"], tensors["key"], tensors["value"], tensors["decay"],
        tensors["beta"], tensors["recurrent_state"], tensors["state_rows"])
    tensors["output"][:] = out
    tensors["recurrent_state"][:] = state


_POOL_ROWS = 6


def build_tensor_specs(tokens: int = 5):
    from golden import TensorSpec

    bf, f32, i32 = torch.bfloat16, torch.float32, torch.int32
    # Deliberately not the identity mapping: a kernel that ignored state_rows and used
    # the token index would pass an identity fixture.
    rows = torch.tensor([4, 1, 5, 0, 3][:tokens], dtype=torch.int32)

    return [
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM)).to(bf)),
        # The real gate spans the whole (-5, 0) range. A near-zero fixture would make
        # exp(g) ~ 1 and hide both the decay and its overflow guards.
        TensorSpec("decay", [tokens, LOCAL_KDA_H, KDA_DIM], f32,
                   init_value=lambda: -5.0 * torch.rand(tokens, LOCAL_KDA_H, KDA_DIM)),
        # beta carries kda_projection's cube-width padding; only the first
        # LOCAL_KDA_H columns are read.
        TensorSpec("beta", [tokens, BETA_PAD], f32,
                   init_value=lambda: torch.rand(tokens, BETA_PAD)),
        # A non-zero initial state, so a kernel that silently starts from zero fails.
        TensorSpec("recurrent_state", [_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], f32,
                   init_value=lambda: torch.randn(_POOL_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM) * 0.1),
        TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
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
    parser.add_argument("--case", type=str, default="batch", choices=["batch", "single"])
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run(
        fn=decode_kda_test,
        specs=build_tensor_specs(1 if args.case == "single" else 5),
        golden_fn=_golden_tensors,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
