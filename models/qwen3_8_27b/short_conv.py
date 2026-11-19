# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet short convolution: depthwise causal conv over the token axis, then silu.

    y[t, c] = silu(sum_j w[j, c] * x_pad[t + j, c])        j = 0 .. K-1, fp32 accumulate

`x_pad` carries the K-1 tokens preceding the sequence in its first rows, so
`x_pad[t + j]` is `x[t - (K - 1) + j]` and tap K-1 reads the token itself. For
prefill those rows are zero; they are where a decode cache would go. Passing
them in the tensor is what keeps every tap offset non-negative, so no block
needs a head-padding special case. No bias: the model's conv1d has none.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Rows of the conv-state pool decode resumes from. Prefill declares its own dynamic for
# it, as `chunk_h` declares `FSTATE_DYN` against `decode_delta_rule`'s `DSTATE_DYN`: a
# prefill operator may not import a decode one.
CSTATE_DYN = pl.dynamic("GDN_CSTATE_DYN")

# model config
C = QWEN3_8_27B.qkv_width                   # q and k at Hg heads, v at H
K = QWEN3_8_27B.linear_conv_kernel_dim      # taps, newest last
# validation
# one block of TOK_TILE * GROUP rows, then two. The token count must be a multiple of 256.
DEFAULT_CASES = (256, 512)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
TOK_TILE = 16           # output rows per pipelined step
CH_TILE = 256           # channels per block
GROUP = 16              # row-tiles per block, the pipelined inner loop
STATE_TILE = 2048       # channels per task of the hand-off, as decode tiles its own pool


def build_kernel(t: int = BENCH_SEQ, c: int = C, k: int = K, inline: bool = False,
                 tok_tile: int = TOK_TILE, ch_tile: int = CH_TILE, group: int = GROUP,
                 c_off: int = 0, c_out: int | None = None, out_heads: int | None = None,
                 out_dtype=pl.BF16, name: str = "short_conv"):
    """The operator at one shape. `inline` makes it a callee for `gdn_block`.

    A block walks `group` row-tiles of one channel tile, so there is an inner
    loop to pipeline; one tile per block leaves the loads nothing to hide behind.

    `c_off` and `c_out` convolve one contiguous run of channels instead of all
    of them, and `out_heads` declares the output `[t, out_heads, d]` rather than
    `[t, c_out]`. The block builds this three times, once per consumer: q and k
    go to the qk-norm and v goes straight to the delta rule, which wants it in
    fp16, and each consumer reads the rows of one contiguous tensor. Splitting
    at the write is what makes that layout free -- the alternative is one wide
    output that every consumer then has to gather from.
    """
    c_out = c if c_out is None else c_out
    rows = tok_tile * group
    # A tile that does not divide its axis silently leaves the tail unwritten --
    # the kernel returns a correct-looking result missing t % rows tokens.
    if t % rows or c_out % ch_tile:
        raise ValueError(f"t={t} must be a multiple of tok_tile*group={rows} and "
                         f"c_out={c_out} a multiple of ch_tile={ch_tile}")
    if c_off + c_out > c:
        raise ValueError(f"channels {c_off}..{c_off + c_out} run past c={c}")
    if out_heads is not None and c_out % out_heads:
        raise ValueError(f"c_out={c_out} must divide by out_heads={out_heads}")
    out_shape = [t, c_out] if out_heads is None else [t, out_heads, c_out // out_heads]

    @(pl.jit.inline if inline else pl.jit)
    def gdn_short_conv(
        x_pad: pl.Tensor[[t + k - 1, c], pl.BF16],
        w: pl.Tensor[[k, c], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        y: pl.Out[pl.Tensor[out_shape, out_dtype]],
    ):
        y_flat = pl.reshape(y, [t, c_out])
        for blk in pl.spmd((t // rows) * (c_out // ch_tile), name_hint=name):
            r0 = (blk // (c_out // ch_tile)) * rows
            yc = (blk % (c_out // ch_tile)) * ch_tile
            c0 = c_off + yc
            w_taps = w[:, c0 : c0 + ch_tile]
            for i in pl.pipeline(group, stage=2):
                t0 = r0 + i * tok_tile
                acc = pl.full([tok_tile, ch_tile], dtype=pl.FP32, value=0.0)
                base = pl.read(row_base, [t0])
                # one load per tap; the windows overlap by tok_tile - 1 rows
                for j in pl.unroll(k):
                    src = pl.cast(x_pad[t0 + j : t0 + j + tok_tile, c0 : c0 + ch_tile],
                                  target_type=pl.FP32)
                    # A later packed row must not read the preceding row's last
                    # K-1 projected tokens. Row starts are 256-aligned, so only
                    # this row's first token tile can have a crossing window.
                    if base > 0 and t0 == base and j < k - 1:
                        src[0 : k - 1 - j, 0:ch_tile] = pl.full(
                            [k - 1 - j, ch_tile], dtype=pl.FP32, value=0.0)
                    acc = pl.add(acc, pl.col_expand_mul(src, w_taps[j : j + 1, :]))
                # silu as acc / (1 + exp(-acc)): one pass fewer than recip + mul,
                # which is 11% of this kernel. The coding guide prefers recip + mul
                # on hot paths, and that holds when the reciprocal is reused; here
                # it is used once, so the divide simply removes an op.
                den = pl.add(pl.exp(pl.neg(acc)), 1.0)
                y_flat[t0 : t0 + tok_tile, yc : yc + ch_tile] = pl.cast(
                    pl.div(acc, den), target_type=out_dtype, mode="rint")
        return y

    return gdn_short_conv


gdn_short_conv = build_kernel()


def build_packed_kernel(t: int, rows: int, c: int = C, k: int = K,
                        inline: bool = False, tok_tile: int = TOK_TILE,
                        ch_tile: int = CH_TILE, group: int = GROUP,
                        c_off: int = 0, c_out: int | None = None,
                        out_heads: int | None = None, out_dtype=pl.BF16,
                        name: str = "short_conv_packed"):
    """Convolve 256-aligned packed rows using each row's incoming cached halo."""
    c_out = c if c_out is None else c_out
    if tok_tile <= 0 or 256 % tok_tile:
        raise ValueError("packed convolution tok_tile must divide the 256-token request alignment")
    block_rows = tok_tile * group
    if (rows < 1 or t % block_rows or c_out % ch_tile or c_off + c_out > c
            or (out_heads is not None and c_out % out_heads)):
        raise ValueError("packed convolution dimensions are invalid")
    out_shape = [t, c_out] if out_heads is None else [t, out_heads, c_out // out_heads]

    @(pl.jit.inline if inline else pl.jit)
    def gdn_short_conv_packed(
        x_pad: pl.Tensor[[t + k - 1, c], pl.BF16],
        w: pl.Tensor[[k, c], pl.FP32],
        row_base: pl.Tensor[[t], pl.INT32],
        row_owner: pl.Tensor[[t], pl.INT32],
        row_halos: pl.Tensor[[rows * (k - 1), c], pl.BF16],
        y: pl.Out[pl.Tensor[out_shape, out_dtype]],
    ):
        y_flat = pl.reshape(y, [t, c_out])
        for blk in pl.spmd((t // block_rows) * (c_out // ch_tile), name_hint=name):
            r0 = (blk // (c_out // ch_tile)) * block_rows
            yc = (blk % (c_out // ch_tile)) * ch_tile
            c0 = c_off + yc
            w_taps = w[:, c0 : c0 + ch_tile]
            for i in pl.pipeline(group, stage=2):
                t0 = r0 + i * tok_tile
                acc = pl.full([tok_tile, ch_tile], dtype=pl.FP32, value=0.0)
                base = pl.read(row_base, [t0])
                owner = pl.cast(pl.read(row_owner, [t0]), pl.INDEX)
                for j in pl.unroll(k):
                    src = pl.cast(x_pad[t0 + j : t0 + j + tok_tile, c0 : c0 + ch_tile],
                                  target_type=pl.FP32)
                    if t0 == base and j < k - 1:
                        halo = row_halos[owner * (k - 1) + j :
                                         owner * (k - 1) + k - 1, c0 : c0 + ch_tile]
                        src[0 : k - 1 - j, 0:ch_tile] = pl.cast(halo, target_type=pl.FP32)
                    acc = pl.add(acc, pl.col_expand_mul(src, w_taps[j : j + 1, :]))
                den = pl.add(pl.exp(pl.neg(acc)), 1.0)
                y_flat[t0 : t0 + tok_tile, yc : yc + ch_tile] = pl.cast(
                    pl.div(acc, den), target_type=out_dtype, mode="rint")
        return y

    return gdn_short_conv_packed


def build_packed_state_swap_kernel(t: int, rows: int, c: int = C, k: int = K,
                                   inline: bool = False):
    """Snapshot every incoming halo and publish each row's new decode window."""
    if rows < 1 or c % STATE_TILE:
        raise ValueError("packed state swap needs rows and STATE_TILE-aligned channels")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_short_conv_state_swap(
        x_pad: pl.Tensor[[t + k - 1, c], pl.BF16],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[rows], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[CSTATE_DYN, k - 1, c], pl.BF16]],
        row_halos: pl.Out[pl.Tensor[[rows * (k - 1), c], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, CSTATE_DYN)
        cs = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * (k - 1), c])
        bands = c // STATE_TILE
        for block in pl.spmd(rows * bands, name_hint="conv_state_swap"):
            owner = block // bands
            c0 = (block % bands) * STATE_TILE
            end = pl.cast(pl.read(row_ends, [owner]), pl.INDEX)
            state_row = pl.cast(pl.read(state_rows, [owner]), pl.INDEX)
            dst = state_row * (k - 1)
            old = cs[dst : dst + k - 1, c0 : c0 + STATE_TILE]
            row_halos[owner * (k - 1) : owner * (k - 1) + k - 1,
                      c0 : c0 + STATE_TILE] = old
            cs[dst : dst + k - 1, c0 : c0 + STATE_TILE] = x_pad[
                end : end + k - 1, c0 : c0 + STATE_TILE]
        return conv_state, row_halos

    return gdn_short_conv_state_swap


def build_state_kernel(t: int = BENCH_SEQ, c: int = C, k: int = K, inline: bool = False):
    """The window decode resumes from: `x_pad`'s last `k - 1` rows, at `state_row` in the pool.

    Those rows are tokens `t - (k - 1) .. t - 1`, oldest first, which is the order both
    sides already use -- here tap `j` reads `x_pad[i + j]`, and in decode tap `j` reads
    `conv_state[j]` with the incoming token on the last tap -- so nothing is reversed and
    no special case is needed at the boundary. The leading zero rows of a first chunk are
    exactly what this writes for the sequence that follows it.
    """
    if c % STATE_TILE:
        raise ValueError(f"c={c} must be a multiple of STATE_TILE={STATE_TILE}")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_short_conv_state(
        x_pad: pl.Tensor[[t + k - 1, c], pl.BF16],
        state_row: pl.Scalar[pl.INDEX],
        conv_state: pl.InOut[pl.Tensor[[CSTATE_DYN, k - 1, c], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, CSTATE_DYN)
        # Addressed through a flat view of the same buffer, as `decode_short_conv`
        # addresses it, so the stored tile keeps the shape the slice already has.
        cs = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * (k - 1), c])
        row = state_row * (k - 1)
        for band in pl.spmd(c // STATE_TILE, name_hint="conv_state"):
            c0 = band * STATE_TILE
            cs[row : row + k - 1, c0 : c0 + STATE_TILE] = x_pad[t : t + k - 1,
                                                                c0 : c0 + STATE_TILE]
        return conv_state

    return gdn_short_conv_state


gdn_short_conv_state = build_state_kernel()


def build_packed_state_kernel(t: int, rows: int, c: int = C, k: int = K,
                              inline: bool = False):
    """Publish each packed row's final K-1 projected tokens to its own decode state."""
    if rows < 1 or c % STATE_TILE:
        raise ValueError("packed state needs rows >= 1 and STATE_TILE-aligned channels")

    @(pl.jit.inline if inline else pl.jit)
    def gdn_short_conv_state_packed(
        x_pad: pl.Tensor[[t + k - 1, c], pl.BF16],
        row_ends: pl.Tensor[[rows], pl.INT32],
        state_rows: pl.Tensor[[rows], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[CSTATE_DYN, k - 1, c], pl.BF16]],
    ):
        conv_state.bind_dynamic(0, CSTATE_DYN)
        cs = pl.reshape(conv_state, [pl.tensor.dim(conv_state, 0) * (k - 1), c])
        bands = c // STATE_TILE
        for block in pl.spmd(rows * bands, name_hint="conv_state_packed"):
            owner = block // bands
            c0 = (block % bands) * STATE_TILE
            end = pl.cast(pl.read(row_ends, [owner]), pl.INDEX)
            state_row = pl.cast(pl.read(state_rows, [owner]), pl.INDEX)
            dst = state_row * (k - 1)
            cs[dst : dst + k - 1, c0 : c0 + STATE_TILE] = x_pad[
                end : end + k - 1, c0 : c0 + STATE_TILE]
        return conv_state

    return gdn_short_conv_state_packed


def build_packed_state_tensor_specs(t: int = 512, starts: tuple[int, ...] = (0, 256),
                                    state_rows: tuple[int, ...] = (1, 3),
                                    pool_rows: int = 5, c: int = C, k: int = K):
    import torch
    from golden import TensorSpec

    if (not starts or starts[0] != 0 or len(starts) != len(state_rows) or t % 256
            or any(start % 256 for start in starts)
            or any(b - a < 256 for a, b in zip(starts, starts[1:] + (t,)))
            or len(set(state_rows)) != len(state_rows)
            or any(row < 0 or row >= pool_rows for row in state_rows)):
        raise ValueError("packed state needs disjoint state rows and complete 256-token spans")
    ends = starts[1:] + (t,)
    return [
        TensorSpec("x_pad", [t + k - 1, c], torch.bfloat16,
                   init_value=_pad_input(t, c, k, 128, None)),
        TensorSpec("row_ends", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(ends, dtype=torch.int32)),
        TensorSpec("state_rows", [len(starts)], torch.int32,
                   init_value=lambda: torch.tensor(state_rows, dtype=torch.int32)),
        TensorSpec("conv_state", [pool_rows, k - 1, c], torch.bfloat16,
                   init_value=lambda: torch.full((pool_rows, k - 1, c), -99.0,
                                                 dtype=torch.bfloat16)),
    ]


def golden_gdn_short_conv_state_packed(tensors):
    x_pad = tensors["x_pad"]
    k = tensors["conv_state"].shape[1] + 1
    for end, row in zip(tensors["row_ends"].tolist(), tensors["state_rows"].tolist()):
        tensors["conv_state"][row] = x_pad[end : end + k - 1]


def build_packed_halo_tensor_specs(t: int = 512, c: int = 1024,
                                   starts: tuple[int, ...] = (0, 256),
                                   state_rows: tuple[int, ...] = (1, 3),
                                   pool_rows: int = 5, resume: bool = False):
    import torch
    from golden import TensorSpec

    if (len(starts) != len(state_rows) or starts[0] != 0 or t % 256
            or any(start % 256 for start in starts)):
        raise ValueError("packed halo needs 256-aligned rows and state owners")
    ends = starts[1:] + (t,)
    gen = torch.Generator().manual_seed(107)
    pool = torch.full((pool_rows, K - 1, c), -99.0, dtype=torch.bfloat16)
    for owner, row in enumerate(state_rows):
        if resume:
            pool[row] = (torch.randn(K - 1, c, generator=gen) * (owner + 1)).to(torch.bfloat16)
        else:
            pool[row] = 0.0
    x_pad = torch.randn(t + K - 1, c, generator=gen).to(torch.bfloat16)
    x_pad[:K - 1] = pool[state_rows[0]]
    row_base = torch.empty(t, dtype=torch.int32)
    row_owner = torch.empty(t, dtype=torch.int32)
    for owner, (start, end) in enumerate(zip(starts, ends)):
        row_base[start:end] = start
        row_owner[start:end] = owner
    w = torch.randn(K, c, generator=gen).float() * 0.1
    return [
        TensorSpec("x_pad", [t + K - 1, c], torch.bfloat16, init_value=lambda: x_pad),
        TensorSpec("w", [K, c], torch.float32, init_value=lambda: w),
        TensorSpec("row_base", [t], torch.int32, init_value=lambda: row_base),
        TensorSpec("row_owner", [t], torch.int32, init_value=lambda: row_owner),
        TensorSpec("row_halos", [len(starts) * (K - 1), c], torch.bfloat16,
                   init_value=lambda: torch.cat([pool[row] for row in state_rows])),
        TensorSpec("y", [t, c], torch.bfloat16),
    ]


def golden_gdn_short_conv_packed(tensors):
    import torch

    starts = sorted(set(tensors["row_base"].tolist()))
    ends = starts[1:] + [tensors["y"].shape[0]]
    for owner, (start, end) in enumerate(zip(starts, ends)):
        halo = tensors["row_halos"][owner * (K - 1) : (owner + 1) * (K - 1)]
        x = torch.cat((halo, tensors["x_pad"][start + K - 1:end + K - 1]))
        tensors["y"][start:end].copy_(golden_y(x, tensors["w"]).to(torch.bfloat16))


def compare_packed_halo_y(actual, expected, inputs=None, **_kwargs):
    if inputs is None:
        return False, "packed halo needs row boundaries"
    starts = sorted(set(inputs["row_base"].tolist()))
    ends = starts[1:] + [actual.shape[0]]
    for owner, (start, end) in enumerate(zip(starts, ends)):
        dev, ref = actual[start:end].float(), expected[start:end].float()
        rel = float((dev - ref).norm() / ref.norm())
        if not rel <= 1e-3:
            return False, f"packed halo owner {owner}: relative output error {rel:.3e}"
    return True, "both packed halos match; including resumed second row"


def compare_packed_conv_state(actual, expected, **kwargs):
    if not bool((actual == expected).all()):
        return False, "    packed state rows differ or an unowned row was modified"
    return True, "    packed state rows exact; unowned rows untouched"


def golden_gdn_short_conv_state(tensors):
    """The tail of `x_pad`, oldest first, into the row the sequence owns."""
    x_pad, k = tensors["x_pad"], K
    t = x_pad.shape[0] - (k - 1)
    row = int(tensors["state_row"])
    tensors["conv_state"][row] = x_pad[t : t + k - 1]


def build_state_tensor_specs(t: int = BENCH_SEQ, c: int = C, k: int = K, chunk: int = 128,
                             weights: str | None = None, pool_rows: int = 4,
                             state_row: int = 2):
    """`x_pad` as the conv entry draws it, plus the pool and the row this sequence owns.

    `state_row` defaults away from 0 so a kernel that ignored it would leave the gated row
    untouched and pass; the pool fills with a value no activation takes, so a row the
    kernel fails to write shows up as the fill.
    """
    import torch
    from golden import ScalarSpec, TensorSpec

    return [
        TensorSpec("x_pad", [t + k - 1, c], torch.bfloat16,
                   init_value=_pad_input(t, c, k, chunk, weights)),
        ScalarSpec("state_row", torch.int64, state_row, compile_runtime=True),
        TensorSpec("conv_state", [pool_rows, k - 1, c], torch.bfloat16,
                   init_value=lambda: torch.full((pool_rows, k - 1, c), -99.0,
                                                 dtype=torch.bfloat16)),
    ]


def compare_conv_state(state_row: int):
    """Score the row the kernel wrote, and prove it left the others alone.

    The hand-off is a copy, so the row must match exactly -- a tolerance here would hide
    a dtype or an off-by-one in the window. The untouched rows carry the fill, which no
    activation takes, so a kernel that ignored `state_row` shows up as a row that should
    have been left alone and was not.
    """
    def compare(actual, expected, **kwargs):
        if not bool((actual[state_row] == expected[state_row]).all()):
            return False, f"    row {state_row} differs from the expected window"
        others = [r for r in range(actual.shape[0]) if r != state_row]
        if others and not bool((actual[others] == -99.0).all()):
            return False, f"    rows {others} were written and should not have been"
        return True, f"    row {state_row} exact, rows {others} untouched"

    return compare



def golden_y(x_pad, w, k: int = K, row_base=None):
    """The kernel's arithmetic in fp32, in its order. -> fp32 [T, C]."""
    import torch

    t = x_pad.shape[0] - (k - 1)
    xf = x_pad.float()
    wf = w.float()
    acc = sum(xf[j : j + t] * wf[j] for j in range(k))
    if row_base is not None:
        positions = torch.arange(t)
        for j in range(k - 1):
            # masked on the row start, as the kernel is: a row starting at 0 keeps
            # whatever the k-1 rows ahead of it carry, zero pad or resumed conv state
            crossing = (row_base > 0) & (positions + j - (k - 1) < row_base)
            acc[crossing] -= xf[j : j + t][crossing] * wf[j]
    return acc / (torch.exp(-acc) + 1.0)


def golden_gdn_short_conv(tensors):
    import torch

    y = golden_y(tensors["x_pad"], tensors["w"], row_base=tensors["row_base"])
    tensors["y"].copy_(y.to(torch.bfloat16))


def _pad_input(t: int, c: int, k: int, chunk: int, weights: str | None):
    """The projection output with k-1 zero rows in front, bf16. -> [T + k - 1, C]."""
    import torch

    import reference

    src = reference.block_inputs("qkv", t, QWEN3_8_27B, chunk, weights=weights)

    def load():
        x = src().to(torch.bfloat16)
        return torch.cat([torch.zeros(k - 1, c, dtype=torch.bfloat16), x])

    return load


def _taps(c: int, k: int, weights: str | None):
    """`conv1d.weight` as [K, C] fp32: a row per tap, which is what the tile loop reads."""
    import torch

    import reference

    def load():
        if weights is None:
            w = reference.make_block_weights(QWEN3_8_27B)["conv1d.weight"]
        else:
            w = torch.load(weights, weights_only=True)["conv1d.weight"]
        return w.reshape(c, k).t().contiguous().float()

    return load


def row_base_table(t: int, starts: tuple[int, ...] = (0,)):
    """The packed start for every token, with a boundary at a 256-token tile."""
    import torch

    if not starts or starts[0] != 0 or any(a >= b for a, b in zip(starts, starts[1:])):
        raise ValueError("row starts must increase from zero")
    if starts[-1] >= t or any(start % 256 for start in starts):
        raise ValueError("row starts must be 256-aligned and inside the token buffer")
    table = torch.empty(t, dtype=torch.int32)
    for start, end in zip(starts, (*starts[1:], t)):
        table[start:end] = start
    return table


def row_boundary_control() -> None:
    """A nonzero prior row reaches the next packed row only unmasked, and row 0 never.

    The second case is the one a zero pad hides: every spec pads the head of
    ``x_pad`` with zeros, so masking row 0 subtracts zeros there and reads as
    correct. A resumed sequence carries conv state in those rows instead.
    """
    import torch

    x_pad = torch.zeros(512 + K - 1, 1, dtype=torch.bfloat16)
    x_pad[256:259] = 1.0
    w = torch.ones(K, 1)
    base = row_base_table(512, (0, 256))
    packed = golden_y(x_pad, w, row_base=base)
    leaking = golden_y(x_pad, w)
    isolated = golden_y(torch.cat((torch.zeros(K - 1, 1, dtype=torch.bfloat16),
                                   x_pad[256 + K - 1:])), w)
    if not torch.equal(packed[:256], leaking[:256]):
        raise RuntimeError("convolution boundary control changed the preceding row")
    if not torch.equal(packed[256:], isolated):
        raise RuntimeError("packed convolution differs from isolated second row")
    if torch.equal(packed[256:259], leaking[256:259]):
        raise RuntimeError("convolution boundary control cannot detect cross-row taps")

    resumed = torch.zeros(512 + K - 1, 1, dtype=torch.bfloat16)
    resumed[: K - 1] = 1.0
    if not torch.equal(golden_y(resumed, w, row_base=base), golden_y(resumed, w)):
        raise RuntimeError("convolution boundary control masked the first row's carried state")


def build_tensor_specs(t: int = BENCH_SEQ, c: int = C, k: int = K, chunk: int = 128,
                       weights: str | None = None, starts: tuple[int, ...] = (0,), tok_tile: int = TOK_TILE):
    import torch
    from golden import TensorSpec

    if tok_tile <= 0 or any(start % tok_tile for start in starts):
        raise ValueError("packed convolution row starts must align to tok_tile")
    return [
        TensorSpec("x_pad", [t + k - 1, c], torch.bfloat16, init_value=_pad_input(t, c, k, chunk, weights)),
        TensorSpec("w", [k, c], torch.float32, init_value=_taps(c, k, weights)),
        TensorSpec("row_base", [t], torch.int32,
                   init_value=lambda: row_base_table(t, starts)),
        TensorSpec("y", [t, c], torch.bfloat16),
    ]


def compare_y(t: int, chunk: int, weights: str | None):
    """The bf16 gate, plus the distance from the float64 reference the roadmap set."""
    from golden import ratio_allclose

    import reference

    gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)

    def compare(actual, expected, **kw):
        ok, detail = gate(actual, expected, **kw)
        if ok:
            return True, ""
        ref = reference.block_inputs("qkv_conv", t, QWEN3_8_27B, chunk, weights=weights)()
        dev = actual.double()
        max_abs = float((dev - ref).abs().max())
        frob = float((dev - ref).norm() / ref.norm())
        head = (f"    y vs float64 reference: max abs {max_abs:.3e} (bf16 tol 6e-2), "
                  f"rel frob {frob:.3e}; vs golden max abs {float((dev - expected.double()).abs().max()):.3e}")
        return False, f"{head}\n{detail}"

    return compare


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None, help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights")
    parser.add_argument("--tok-tile", type=int, default=TOK_TILE)
    parser.add_argument("--ch-tile", type=int, default=CH_TILE)
    parser.add_argument("--group", type=int, default=GROUP)
    parser.add_argument("--row-starts", type=str, default="0",
                        help="comma-separated packed row starts, 256-aligned")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the kernel over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--state", action="store_true", default=False,
                        help="also gate the conv-state hand-off decode resumes from")
    parser.add_argument("--packed-state-only", action="store_true", default=False,
                        help="gate per-row convolution-state publication")
    parser.add_argument("--packed-halo-only", action="store_true", default=False,
                        help="gate each packed row's incoming convolution halo")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="seed both packed convolution state rows")
    args = parser.parse_args()
    row_boundary_control()
    starts = tuple(int(value) for value in args.row_starts.split(","))
    packed_only = args.packed_halo_only or args.packed_state_only
    cases = [args.seq_len] if args.seq_len is not None else list(DEFAULT_CASES[-1:] if packed_only else DEFAULT_CASES)
    if args.seq_len and args.seq_len % 256:
        parser.error("--seq-len must be a multiple of 256")
    if packed_only and (args.golden_data or args.runtime_dir or args.save_data):
        parser.error("packed convolution validation does not support save or replay options")
    if args.tok_tile <= 0 or any(start % args.tok_tile for start in starts):
        parser.error("--row-starts must align to --tok-tile")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    for t in cases:
        if args.packed_halo_only:
            if len(starts) < 2:
                parser.error("--packed-halo-only needs at least two --row-starts")
            state_rows = tuple(2 * owner + 1 for owner in range(len(starts)))
            print(f"[case] short_conv packed halo T={t} resume={args.resume}", flush=True)
            result = run(
                fn=build_packed_kernel(t=t, rows=len(starts), c=1024,
                                       tok_tile=args.tok_tile, ch_tile=args.ch_tile, group=args.group),
                specs=build_packed_halo_tensor_specs(
                    t=t, starts=starts, state_rows=state_rows,
                    pool_rows=2 * len(starts) + 1, resume=args.resume),
                golden_fn=golden_gdn_short_conv_packed,
                config=dict(backend_type=args.backend_type,
                            execution_mode=args.execution_mode, device_id=args.device),
                compare_fn={"y": compare_packed_halo_y},
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(f"FAILED at packed convolution halo T={t}")
            continue
        if args.packed_state_only:
            if len(starts) < 2:
                parser.error("--packed-state-only needs at least two --row-starts")
            state_rows = tuple(2 * owner + 1 for owner in range(len(starts)))
            print(f"[case] short_conv packed state T={t} rows={len(starts)}", flush=True)
            result = run(
                fn=build_packed_state_kernel(t=t, rows=len(starts)),
                specs=build_packed_state_tensor_specs(
                    t=t, starts=starts, state_rows=state_rows, pool_rows=2 * len(starts) + 1),
                golden_fn=golden_gdn_short_conv_state_packed,
                config=dict(backend_type=args.backend_type,
                            execution_mode=args.execution_mode, device_id=args.device),
                compare_fn={"conv_state": compare_packed_conv_state},
                compile_only=args.compile_only,
            )
            if not result.passed:
                if result.error:
                    print(result.error)
                raise SystemExit(f"FAILED at packed conv-state hand-off, T={t}")
            continue
        print(f"[case] short_conv T={t}", flush=True)
        result = run(
            fn=build_kernel(t=t, tok_tile=args.tok_tile, ch_tile=args.ch_tile, group=args.group),
            specs=build_tensor_specs(t=t, weights=args.weights, starts=starts, tok_tile=args.tok_tile),
            golden_fn=golden_gdn_short_conv,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-5,
            compare_fn={"y": compare_y(t, 128, args.weights)},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at T={t}")

        if not args.state:
            continue
        print(f"[case] short_conv state T={t}", flush=True)
        state_row = 2
        result = run(
            fn=build_state_kernel(t=t),
            specs=build_state_tensor_specs(t=t, weights=args.weights, state_row=state_row),
            golden_fn=golden_gdn_short_conv_state,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            compare_fn={"conv_state": compare_conv_state(state_row)},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at the conv-state hand-off, T={t}")
