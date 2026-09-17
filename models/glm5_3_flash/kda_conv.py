# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The KDA short depthwise causal convolution, prefill and decode.

``short_conv_kernel_size = 4``, depthwise over every q/k/v channel, followed by the
SiLU activation. The reference runs one grouped ``Conv1d`` over ``3 * 8192``
channels, but **the checkpoint stores three separate convs** — ``q_conv1d.weight``,
``k_conv1d.weight``, ``v_conv1d.weight``. A depthwise convolution is independent per
channel, so concatenating the three along the channel axis is exact; the loader
stacks this rank's 512 channels from each into one ``[KDA_CONV_K, CONV_DIM]`` weight,
which is what vLLM-Ascend builds on its first forward.

Both entries emit **q, k and v separately**, already viewed as
``[T, LOCAL_KDA_H, KDA_DIM]``, because that is what ``prefill_kda`` and ``decode_kda``
consume.

Two layout choices differ from the landed scaffold, and both follow the AscendC
kernel's own ``[cache, state_len, dim]`` convention:

* ``conv_state`` is time-major, ``[rows, KDA_CONV_K - 1, CONV_DIM]``. A channel-major
  state would need a ``[CONV_DIM, 3] -> [3, CONV_DIM]`` transpose on chip for every
  request, to produce rows the convolution already wants.
* ``weight`` is ``[KDA_CONV_K, CONV_DIM]``, so tap *j* is one contiguous per-channel
  row and applies with a single column-broadcast multiply.

The state holds the last ``KDA_CONV_K - 1`` rows of the **pre-convolution,
pre-activation** ``mixed_qkv`` stream — not the conv output, and not the SiLU output.
For a request contributing fewer than three tokens the new state is a mix of the old
state's tail and the new rows, which is why the update is written as a blend rather
than a plain copy of the last three tokens.

The prefill convolution runs in two passes. The first sweeps the packed token axis
flat, which is correct for every token at least ``KDA_CONV_K - 1`` rows into its own
request; for the first three tokens of a request it reads the previous request's tail
instead. The second pass recomputes exactly those rows from the conv state, and
updates the state. Splitting it this way keeps the bulk pass free of per-request
bounds arithmetic, which is where the token axis is long.
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

from models.glm5_3_flash.config import B_DYN, KDA_CONV_K, KDA_DIM, KDA_STATE_DYN
from models.glm5_3_flash.config import LOCAL_KDA_H, LOCAL_KDA_QKV_DIM, Q_START_DYN, T_DYN

CONV_DIM = 3 * LOCAL_KDA_QKV_DIM
STATE_LEN = KDA_CONV_K - 1

# tiling
T_TILE = 16  # tokens per bulk task
C_TILE = 512  # channels per bulk task: [16, 512] FP32 is 32 KiB, and four taps plus
# the accumulator stay well inside UB


def _silu_torch(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def _conv_reference(
    stream: torch.Tensor, state: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One request: convolve ``stream`` given ``state``, return output and new state.

    ``state`` is ``[STATE_LEN, CONV_DIM]`` of pre-conv rows, ``weight`` is
    ``[KDA_CONV_K, CONV_DIM]``.
    """
    full = torch.cat([state.float(), stream.float()], dim=0)
    length = stream.shape[0]
    out = torch.zeros(length, stream.shape[1], dtype=torch.float32)
    for j in range(KDA_CONV_K):
        out += full[j : j + length] * weight[j].float()
    return _silu_torch(out), full[length : length + STATE_LEN]


def golden_kda_conv_prefill(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(query, key, value, conv_state)``; the state is returned updated.

    ``state_rows`` addresses the pool by request row rather than by batch position,
    so a reordered or paged batch is modelled rather than assumed away.
    """
    tokens, _ = mixed_qkv.shape
    out = torch.zeros(tokens, CONV_DIM, dtype=torch.float32)
    new_state = conv_state.clone()
    requests = query_start_loc.numel() - 1
    for b in range(requests):
        s = int(query_start_loc[b])
        e = int(query_start_loc[b + 1])
        if e <= s:
            continue
        row = int(state_rows[b])
        conv_out, tail = _conv_reference(mixed_qkv[s:e], conv_state[row], weight)
        out[s:e] = conv_out
        new_state[row] = tail.to(conv_state.dtype)
    shaped = out.to(mixed_qkv.dtype).view(tokens, 3, LOCAL_KDA_H, KDA_DIM)
    return shaped[:, 0], shaped[:, 1], shaped[:, 2], new_state


def golden_kda_conv_decode(
    mixed_qkv: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(query, key, value, conv_state)`` for one token per row."""
    tokens, _ = mixed_qkv.shape
    out = torch.zeros(tokens, CONV_DIM, dtype=torch.float32)
    new_state = conv_state.clone()
    for t in range(tokens):
        row = int(state_rows[t])
        conv_out, tail = _conv_reference(
            mixed_qkv[t : t + 1], new_state[row].float(), weight
        )
        out[t] = conv_out[0]
        new_state[row] = tail.to(conv_state.dtype)
    shaped = out.to(mixed_qkv.dtype).view(tokens, 3, LOCAL_KDA_H, KDA_DIM)
    return shaped[:, 0], shaped[:, 1], shaped[:, 2], new_state


@pl.jit.inline
def kda_conv_prefill(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(mixed_qkv, 0)
    requests = pl.tensor.dim(query_start_loc, 0) - 1
    q_flat = pl.reshape(query, [t_dim, LOCAL_KDA_QKV_DIM])
    k_flat = pl.reshape(key, [t_dim, LOCAL_KDA_QKV_DIM])
    v_flat = pl.reshape(value, [t_dim, LOCAL_KDA_QKV_DIM])

    # --- pass 1: flat sweep of every token at least STATE_LEN rows into the stream ---
    # Rows below STATE_LEN, and the first STATE_LEN rows of every later request, are
    # recomputed in pass 2; here they would read the preceding request's tail.
    # A batch shorter than the window has no bulk rows at all, and a zero-task spmd
    # is not dispatchable; pass 2 covers every row in that case.
    # A batch shorter than the window has no bulk rows, and a zero-task spmd is not
    # dispatchable, so the task count floors at one and the body guards itself. Pass 2
    # covers every row in that case.
    bulk_tiles = pl.max((pl.max(t_dim - STATE_LEN, 0) + T_TILE - 1) // T_TILE, 1)
    with pl.spmd(bulk_tiles * (CONV_DIM // C_TILE), name_hint="kda_conv_bulk") as bulk_tid:
        task = pl.tile.get_block_idx()
        lanes = CONV_DIM // C_TILE
        p0 = STATE_LEN + (task // lanes) * T_TILE
        c0 = (task % lanes) * C_TILE
        rows = pl.min(T_TILE, t_dim - p0)
        if rows > 0:
            acc = pl.full([T_TILE, C_TILE], dtype=pl.FP32, value=0.0)
            for j in pl.unroll(KDA_CONV_K):
                tap = pl.slice(mixed_qkv, [T_TILE, C_TILE], [p0 - STATE_LEN + j, c0],
                               valid_shape=[rows, C_TILE])
                acc = pl.add(acc, pl.col_expand_mul(pl.cast(tap, pl.FP32),
                                                    pl.cast(weight[j : j + 1, c0 : c0 + C_TILE],
                                                            pl.FP32)))
            # SiLU, spelled out: the traced body takes only pl.* calls.
            activated = pl.mul(acc, pl.recip(pl.add(pl.exp(pl.neg(acc)), 1.0)))
            out_tile = pl.set_validshape(pl.cast(activated, pl.BF16, mode="rint"), rows, C_TILE)
            if c0 < LOCAL_KDA_QKV_DIM:
                q_flat[p0 : p0 + T_TILE, c0 : c0 + C_TILE] = out_tile
            elif c0 < 2 * LOCAL_KDA_QKV_DIM:
                kc = c0 - LOCAL_KDA_QKV_DIM
                k_flat[p0 : p0 + T_TILE, kc : kc + C_TILE] = out_tile
            else:
                vc = c0 - 2 * LOCAL_KDA_QKV_DIM
                v_flat[p0 : p0 + T_TILE, vc : vc + C_TILE] = out_tile


    # --- pass 2: the first STATE_LEN rows of every request, plus the state update ---
    with pl.spmd(requests * (CONV_DIM // C_TILE), name_hint="kda_conv_head",
                 deps=[bulk_tid]):
        task = pl.tile.get_block_idx()
        lanes = CONV_DIM // C_TILE
        b = task // lanes
        c0 = (task % lanes) * C_TILE
        s = pl.cast(pl.read(query_start_loc, [b]), pl.INDEX)
        e = pl.cast(pl.read(query_start_loc, [b + 1]), pl.INDEX)
        row = pl.cast(pl.read(state_rows, [b]), pl.INDEX)
        length = e - s

        # full[k] is conv_state[row, k] for k < STATE_LEN and mixed_qkv[s + k - STATE_LEN]
        # otherwise. Written straight through rather than looped: a pl iterator's index
        # is a Scalar, so it cannot drive the trace-time choice between the two sources.
        st0 = pl.cast(pl.reshape(pl.slice(conv_state, [1, 1, C_TILE], [row, 0, c0], drop_dims=[0]),
                                 [1, C_TILE]), pl.FP32)
        st1 = pl.cast(pl.reshape(pl.slice(conv_state, [1, 1, C_TILE], [row, 1, c0], drop_dims=[0]),
                                 [1, C_TILE]), pl.FP32)
        st2 = pl.cast(pl.reshape(pl.slice(conv_state, [1, 1, C_TILE], [row, 2, c0], drop_dims=[0]),
                                 [1, C_TILE]), pl.FP32)
        # k - STATE_LEN <= u < length whenever the row below is written, so these reads
        # stay inside the request.
        x0 = pl.cast(pl.slice(mixed_qkv, [1, C_TILE], [s, c0]), pl.FP32)
        x1 = pl.cast(pl.slice(mixed_qkv, [1, C_TILE], [pl.min(s + 1, e - 1), c0]), pl.FP32)
        x2 = pl.cast(pl.slice(mixed_qkv, [1, C_TILE], [pl.min(s + 2, e - 1), c0]), pl.FP32)
        w0 = pl.cast(weight[0:1, c0 : c0 + C_TILE], pl.FP32)
        w1 = pl.cast(weight[1:2, c0 : c0 + C_TILE], pl.FP32)
        w2 = pl.cast(weight[2:3, c0 : c0 + C_TILE], pl.FP32)
        w3 = pl.cast(weight[3:4, c0 : c0 + C_TILE], pl.FP32)

        # out[u] = sum_j w[j] * full[u + j]
        acc0 = pl.add(pl.add(pl.mul(st0, w0), pl.mul(st1, w1)),
                      pl.add(pl.mul(st2, w2), pl.mul(x0, w3)))
        acc1 = pl.add(pl.add(pl.mul(st1, w0), pl.mul(st2, w1)),
                      pl.add(pl.mul(x0, w2), pl.mul(x1, w3)))
        acc2 = pl.add(pl.add(pl.mul(st2, w0), pl.mul(x0, w1)),
                      pl.add(pl.mul(x1, w2), pl.mul(x2, w3)))

        # SiLU and the masked stores, one row at a time. A request shorter than the
        # window leaves the later rows with zero valid rows, so nothing is written.
        act0 = pl.mul(acc0, pl.recip(pl.add(pl.exp(pl.neg(acc0)), 1.0)))
        act1 = pl.mul(acc1, pl.recip(pl.add(pl.exp(pl.neg(acc1)), 1.0)))
        act2 = pl.mul(acc2, pl.recip(pl.add(pl.exp(pl.neg(acc2)), 1.0)))
        v0 = pl.min(pl.max(length, 0), 1)
        v1 = pl.min(pl.max(length - 1, 0), 1)
        v2 = pl.min(pl.max(length - 2, 0), 1)
        o0 = pl.set_validshape(pl.cast(act0, pl.BF16, mode="rint"), v0, C_TILE)
        o1 = pl.set_validshape(pl.cast(act1, pl.BF16, mode="rint"), v1, C_TILE)
        o2 = pl.set_validshape(pl.cast(act2, pl.BF16, mode="rint"), v2, C_TILE)
        # The destination is chosen by writing in each branch, not by rebinding a name
        # to a different tensor: an IfStmt's in-place return must yield one backing
        # value, and three different tensors cannot be unified into it.
        if c0 < LOCAL_KDA_QKV_DIM:
            q_flat[s : s + 1, c0 : c0 + C_TILE] = o0
            q_flat[s + 1 : s + 2, c0 : c0 + C_TILE] = o1
            q_flat[s + 2 : s + 3, c0 : c0 + C_TILE] = o2
        elif c0 < 2 * LOCAL_KDA_QKV_DIM:
            kc0 = c0 - LOCAL_KDA_QKV_DIM
            k_flat[s : s + 1, kc0 : kc0 + C_TILE] = o0
            k_flat[s + 1 : s + 2, kc0 : kc0 + C_TILE] = o1
            k_flat[s + 2 : s + 3, kc0 : kc0 + C_TILE] = o2
        else:
            vc0 = c0 - 2 * LOCAL_KDA_QKV_DIM
            v_flat[s : s + 1, vc0 : vc0 + C_TILE] = o0
            v_flat[s + 1 : s + 2, vc0 : vc0 + C_TILE] = o1
            v_flat[s + 2 : s + 3, vc0 : vc0 + C_TILE] = o2

        # New state row i is full[length + i]: the old state's tail when the request is
        # shorter than the window, the stream's last rows otherwise. Selected by two
        # writes whose row counts are complementary, not by an arithmetic blend: the
        # condition is an index scalar, and index-to-float casts are not supported.
        # A write with zero valid rows stores nothing, so exactly one of each pair lands.
        # Both reads are clamped, so the side that is not stored is still in bounds.
        for i in pl.unroll(STATE_LEN):
            old_idx = pl.min(length + i, STATE_LEN - 1)
            new_idx = pl.max(e - STATE_LEN + i, s)
            keep_old = pl.min(pl.max(STATE_LEN - length - i, 0), 1)
            take_new = pl.min(pl.max(length + i - STATE_LEN + 1, 0), 1)
            # set_validshape takes a 2D tile, so the row count is declared before the
            # reshape to the state's rank-3 view.
            old_row = pl.reshape(pl.slice(conv_state, [1, 1, C_TILE], [row, old_idx, c0],
                                          drop_dims=[0]), [1, C_TILE])
            new_row = pl.slice(mixed_qkv, [1, C_TILE], [new_idx, c0])
            conv_state[row : row + 1, i : i + 1, c0 : c0 + C_TILE] = pl.reshape(
                pl.set_validshape(old_row, keep_old, C_TILE), [1, 1, C_TILE])
            conv_state[row : row + 1, i : i + 1, c0 : c0 + C_TILE] = pl.reshape(
                pl.set_validshape(new_row, take_new, C_TILE), [1, 1, C_TILE])

    return query, key, value, conv_state


@pl.jit.inline
def kda_conv_decode(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    query: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    key: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    value: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(mixed_qkv, 0)
    q_flat = pl.reshape(query, [t_dim, LOCAL_KDA_QKV_DIM])
    k_flat = pl.reshape(key, [t_dim, LOCAL_KDA_QKV_DIM])
    v_flat = pl.reshape(value, [t_dim, LOCAL_KDA_QKV_DIM])

    # One token per row, so the window is the whole state plus this token and every
    # index is a trace-time constant.
    with pl.spmd(t_dim * (CONV_DIM // C_TILE), name_hint="kda_conv_decode"):
        task = pl.tile.get_block_idx()
        lanes = CONV_DIM // C_TILE
        t = task // lanes
        c0 = (task % lanes) * C_TILE
        row = pl.cast(pl.read(state_rows, [t]), pl.INDEX)

        # Straight-line taps: a pl iterator's index is a Scalar, so it cannot drive
        # the trace-time choice between the conv state and the incoming token.
        st0 = pl.cast(pl.reshape(pl.slice(conv_state, [1, 1, C_TILE], [row, 0, c0], drop_dims=[0]),
                                 [1, C_TILE]), pl.FP32)
        st1 = pl.cast(pl.reshape(pl.slice(conv_state, [1, 1, C_TILE], [row, 1, c0], drop_dims=[0]),
                                 [1, C_TILE]), pl.FP32)
        st2 = pl.cast(pl.reshape(pl.slice(conv_state, [1, 1, C_TILE], [row, 2, c0], drop_dims=[0]),
                                 [1, C_TILE]), pl.FP32)
        xt = pl.cast(pl.slice(mixed_qkv, [1, C_TILE], [t, c0]), pl.FP32)
        acc = pl.add(
            pl.add(pl.mul(st0, pl.cast(weight[0:1, c0 : c0 + C_TILE], pl.FP32)),
                   pl.mul(st1, pl.cast(weight[1:2, c0 : c0 + C_TILE], pl.FP32))),
            pl.add(pl.mul(st2, pl.cast(weight[2:3, c0 : c0 + C_TILE], pl.FP32)),
                   pl.mul(xt, pl.cast(weight[3:4, c0 : c0 + C_TILE], pl.FP32))),
        )
        activated = pl.mul(acc, pl.recip(pl.add(pl.exp(pl.neg(acc)), 1.0)))
        out_tile = pl.cast(activated, pl.BF16, mode="rint")
        if c0 < LOCAL_KDA_QKV_DIM:
            q_flat[t : t + 1, c0 : c0 + C_TILE] = out_tile
        elif c0 < 2 * LOCAL_KDA_QKV_DIM:
            k_flat[t : t + 1, c0 - LOCAL_KDA_QKV_DIM : c0 - LOCAL_KDA_QKV_DIM + C_TILE] = out_tile
        else:
            v_flat[t : t + 1, c0 - 2 * LOCAL_KDA_QKV_DIM : c0 - 2 * LOCAL_KDA_QKV_DIM + C_TILE] = out_tile

        # Shift the window: every source row is read above, before any is overwritten.
        conv_state[row : row + 1, 0:1, c0 : c0 + C_TILE] = pl.reshape(
            pl.cast(st1, pl.BF16, mode="rint"), [1, 1, C_TILE])
        conv_state[row : row + 1, 1:2, c0 : c0 + C_TILE] = pl.reshape(
            pl.cast(st2, pl.BF16, mode="rint"), [1, 1, C_TILE])
        conv_state[row : row + 1, 2:3, c0 : c0 + C_TILE] = pl.reshape(
            pl.slice(mixed_qkv, [1, C_TILE], [t, c0]), [1, 1, C_TILE])

    return query, key, value, conv_state


__all__ = [
    "CONV_DIM",
    "STATE_LEN",
    "golden_kda_conv_decode",
    "golden_kda_conv_prefill",
    "kda_conv_decode",
    "kda_conv_prefill",
]


@pl.jit
def kda_conv_prefill_test(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    query_start_loc: pl.Tensor[[Q_START_DYN], pl.INT32],
    state_rows: pl.Tensor[[B_DYN], pl.INT32],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    key: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    value: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    query, key, value, conv_state = kda_conv_prefill(
        mixed_qkv, weight, conv_state, query_start_loc, state_rows, query, key, value)
    return query, key, value, conv_state


@pl.jit
def kda_conv_decode_test(
    mixed_qkv: pl.Tensor[[T_DYN, CONV_DIM], pl.BF16],
    weight: pl.Tensor[[KDA_CONV_K, CONV_DIM], pl.BF16],
    conv_state: pl.InOut[pl.Tensor[[KDA_STATE_DYN, STATE_LEN, CONV_DIM], pl.BF16]],
    state_rows: pl.Tensor[[T_DYN], pl.INT32],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    key: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
    value: pl.Out[pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16]],
):
    query, key, value, conv_state = kda_conv_decode(
        mixed_qkv, weight, conv_state, state_rows, query, key, value)
    return query, key, value, conv_state


def _golden_prefill_tensors(tensors) -> None:
    q, k, v, st = golden_kda_conv_prefill(
        tensors["mixed_qkv"], tensors["weight"], tensors["conv_state"],
        tensors["query_start_loc"], tensors["state_rows"])
    tensors["query"][:] = q
    tensors["key"][:] = k
    tensors["value"][:] = v
    tensors["conv_state"][:] = st


def _golden_decode_tensors(tensors) -> None:
    q, k, v, st = golden_kda_conv_decode(
        tensors["mixed_qkv"], tensors["weight"], tensors["conv_state"], tensors["state_rows"])
    tensors["query"][:] = q
    tensors["key"][:] = k
    tensors["value"][:] = v
    tensors["conv_state"][:] = st


# Lengths chosen so the state blend is exercised: a 1-token and a 2-token request are
# both shorter than the window, so their new state mixes old rows with new ones.
_PREFILL_LENGTHS = {"ragged": [5, 1, 9, 2], "single": [1], "long": [40]}
_POOL_ROWS = 6


def build_tensor_specs(case: str = "ragged", decode: bool = False):
    from golden import TensorSpec

    bf, i32 = torch.bfloat16, torch.int32

    def common(tokens):
        return [
            TensorSpec("mixed_qkv", [tokens, CONV_DIM], bf,
                       init_value=lambda: (torch.randn(tokens, CONV_DIM) * 0.8).to(bf)),
            TensorSpec("weight", [KDA_CONV_K, CONV_DIM], bf,
                       init_value=lambda: (torch.randn(KDA_CONV_K, CONV_DIM) * 0.5).to(bf)),
            # Non-zero, so a dropped state read is visible rather than a no-op.
            TensorSpec("conv_state", [_POOL_ROWS, STATE_LEN, CONV_DIM], bf,
                       init_value=lambda: (torch.randn(_POOL_ROWS, STATE_LEN, CONV_DIM) * 0.6).to(bf)),
        ]

    if decode:
        tokens = 3 if case == "single" else 5
        # Deliberately not the identity mapping: a kernel that ignores state_rows and
        # uses the token index would pass an identity fixture.
        rows = torch.tensor([4, 1, 5, 0, 3][:tokens], dtype=torch.int32)
        return common(tokens) + [
            TensorSpec("state_rows", [tokens], i32, init_value=lambda: rows),
            TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
            TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
            TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        ]

    lengths = _PREFILL_LENGTHS[case]
    starts = [0]
    for length in lengths:
        starts.append(starts[-1] + length)
    tokens = starts[-1]
    rows = torch.tensor([4, 1, 5, 0][: len(lengths)], dtype=torch.int32)
    return common(tokens) + [
        TensorSpec("query_start_loc", [len(starts)], i32,
                   init_value=lambda: torch.tensor(starts, dtype=torch.int32)),
        TensorSpec("state_rows", [len(lengths)], i32, init_value=lambda: rows),
        TensorSpec("query", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        TensorSpec("key", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
        TensorSpec("value", [tokens, LOCAL_KDA_H, KDA_DIM], bf),
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
                        choices=["ragged", "single", "long", "decode", "decode-single"])
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    decode = args.case.startswith("decode")
    case = "single" if args.case == "decode-single" else ("ragged" if decode else args.case)
    result = run(
        fn=kda_conv_decode_test if decode else kda_conv_prefill_test,
        specs=build_tensor_specs(case, decode),
        golden_fn=_golden_decode_tensors if decode else _golden_prefill_tensors,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
