# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B plain RMSNorm, `y = bf16(fp32(x) * rsqrt(mean(x^2) + eps) * (1 + fp32(w)))`.

The decoder's `input_layernorm` and `post_attention_layernorm`, one kernel for
both. The weight is zero-centred, so `1 + w` is the formula and not a fold.
Two passes over x: the sum of squares accumulates over K tiles, then the
normalisation re-reads them.

The token tile is a build argument because the task count is `t // tok_tile`, and at
a decode batch that quotient is zero -- which aborts the launch rather than quietly
doing nothing. `decode_layer` passes the decode value.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S

# model config
HIDDEN = QWEN3_8_27B.hidden_size            # the axis the norm reduces over
HIDDEN_INV = 1.0 / HIDDEN
EPS = QWEN3_8_27B.rms_norm_eps
# validation
# one TOK_TILE of tokens, then two, for the two prefill programs; the decode
# fan-out is shaped by the padded batch, so it carries its own pair.
DEFAULT_CASES = (64, 128)
FANNED_CASES = (16, 32)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
TOK_TILE = 64           # tokens per block
K_TILE = 256            # hidden elements per pass step; TOK_TILE * K_TILE fills Vec
DECODE_TOK_TILE = 16    # the padded decode batch, which is one token tile
COL_GROUPS = 5          # column groups the decode form fans the reduction over
VEC_BUFFER_BYTES = 188416       # a2a3's vector buffer, which bounds the column range
STORE_GROUP = 64                # elements per BF16 store group; the guard keeps every store whole

# scheduling
# Speculative early dispatch, producer side: it lets the consumer downstream be staged on a
# core and released by a doorbell when this stage's blocks finish, which removes the
# ready-but-undispatched and pickup parts of the gap in front of it. Pure scheduling, results
# unaffected, and it only does anything when EVERY direct producer of a consumer carries it --
# see docs/debug-and-tune/dependency-and-scheduling.md. A module constant rather than
# a literal so the two schedules can be timed in one process.
#
# With this and `quant_x.EARLY_RESOLVE`, five consumers on a composed layer's critical path
# become eligible, including both of its big matmuls, and a capture proves the window is won:
# their gaps fall from 3.5 to 5.5 us down to 1.7 to 1.9, which is FIN detection alone. The
# layer measures 803.4 -> 791.7 us at batch 16 and 1007.4 -> 1003.0 at 32 (gated attention
# 773.4 -> 765.2 and 1031.0 -> 1019.7), so 43.5 us/token at batch 16 over 64 layers. The
# twelve unflagged producers still on that path are the rest of the item.
EARLY_RESOLVE = True

# gate
MIN_EXACT_VS_GOLDEN = 0.95      # of `y` elements identical to the float64 reference's rounding


def build_kernel(tok_tile: int = TOK_TILE, k_tile: int = K_TILE, inline: bool = False):
    """One norm at a given token tile; `t` must be a multiple of `tok_tile`."""

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_rmsnorm(
        x: pl.Tensor[[T_DYN, HIDDEN], pl.BF16],
        norm_w: pl.Tensor[[HIDDEN], pl.BF16],
        y: pl.Out[pl.Tensor[[T_DYN, HIDDEN], pl.BF16]],
    ):
        """The token count must be a multiple of tok_tile; a block walks one whole tile."""
        x.bind_dynamic(0, T_DYN)
        y.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(x, 0)
        w_row = pl.reshape(norm_w[:], [1, HIDDEN])
        for blk in pl.spmd(t_dim // tok_tile, name_hint="rmsnorm"):
            t0 = blk * tok_tile
            # the sum of squares per token, as a [1, tok_tile] row of partials
            partial_sq = pl.full([1, tok_tile], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, HIDDEN, k_tile):
                sq_x = pl.cast(x[t0 : t0 + tok_tile, k0 : k0 + k_tile], target_type=pl.FP32)
                sq_prod = pl.mul(sq_x, sq_x)
                sq_row = pl.reshape(pl.row_sum(sq_prod), [1, tok_tile])
                partial_sq = pl.add(partial_sq, sq_row)
            mean_sq = pl.mul(partial_sq, HIDDEN_INV)
            variance = pl.reshape(mean_sq, [tok_tile, 1])
            inv_rms = pl.rsqrt(pl.add(variance, EPS), high_precision=True)

            # both multiplies in fp32, narrowed once at the end
            for k0 in pl.range(0, HIDDEN, k_tile):
                norm_x = pl.cast(x[t0 : t0 + tok_tile, k0 : k0 + k_tile], target_type=pl.FP32)
                w_tile = pl.cast(w_row[:, k0 : k0 + k_tile], target_type=pl.FP32)
                gamma = pl.add(w_tile, 1.0)
                scaled = pl.row_expand_mul(norm_x, inv_rms)
                normed = pl.col_expand_mul(scaled, gamma)
                y[t0 : t0 + tok_tile, k0 : k0 + k_tile] = pl.cast(normed, target_type=pl.BF16, mode="rint")
        return y

    return _qwen3_rmsnorm


qwen3_rmsnorm = build_kernel(inline=True)
qwen3_rmsnorm_test = build_kernel()


def build_residual_kernel(tok_tile: int = TOK_TILE, k_tile: int = K_TILE, inline: bool = False):
    """`s = residual + delta` in FP32, then `y = rmsnorm(s)` in BF16, as one stage.

    A decoder layer adds a residual twice and norms twice, and the two always pair: the
    sum one of them produces is the input the next norm reduces, and it is also the
    residual the *following* add needs. Keeping them separate costs two launches of about
    16 us each per pair, 128 of them per decode token.

    **The running residual is FP32 and only the norm's output is BF16.** That is the shape
    `models/qwen3_14b/decode_fwd.py` uses (`resid_fp32`, kept with "no BF16 cast", against
    a BF16 `mlp_norm_in`), and the arithmetic says why: BF16 carries 8 mantissa bits, and
    accumulating 128 terms of comparable size into one costs about half of
    `log2(128)` of them. A BF16 residual stream would reach the last layer with five bits
    of mantissa left. The mixer and MLP deltas stay BF16 -- they are matmul outputs and
    each is added once.
    """

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_residual_rmsnorm(
        residual: pl.Tensor[[T_DYN, HIDDEN], pl.FP32],
        delta: pl.Tensor[[T_DYN, HIDDEN], pl.BF16],
        norm_w: pl.Tensor[[HIDDEN], pl.BF16],
        residual_out: pl.Out[pl.Tensor[[T_DYN, HIDDEN], pl.FP32]],
        y: pl.Out[pl.Tensor[[T_DYN, HIDDEN], pl.BF16]],
    ):
        """The token count must be a multiple of tok_tile; a block walks one whole tile."""
        residual.bind_dynamic(0, T_DYN)
        delta.bind_dynamic(0, T_DYN)
        residual_out.bind_dynamic(0, T_DYN)
        y.bind_dynamic(0, T_DYN)
        t_dim = pl.tensor.dim(residual, 0)
        w_row = pl.reshape(norm_w[:], [1, HIDDEN])
        for blk in pl.spmd(t_dim // tok_tile, name_hint="residual_rmsnorm"):
            t0 = blk * tok_tile
            # The sum is written out on the first pass and read back on the second rather
            # than held: at tok_tile 16 and HIDDEN 5120 the whole row is 320 KiB of FP32,
            # which does not fit a 188416-byte vector buffer.
            partial_sq = pl.full([1, tok_tile], dtype=pl.FP32, value=0.0)
            for k0 in pl.range(0, HIDDEN, k_tile):
                delta_f = pl.cast(delta[t0 : t0 + tok_tile, k0 : k0 + k_tile], target_type=pl.FP32)
                total = pl.add(residual[t0 : t0 + tok_tile, k0 : k0 + k_tile], delta_f)
                residual_out[t0 : t0 + tok_tile, k0 : k0 + k_tile] = total
                sq_row = pl.reshape(pl.row_sum(pl.mul(total, total)), [1, tok_tile])
                partial_sq = pl.add(partial_sq, sq_row)
            mean_sq = pl.mul(partial_sq, HIDDEN_INV)
            variance = pl.reshape(mean_sq, [tok_tile, 1])
            inv_rms = pl.rsqrt(pl.add(variance, EPS), high_precision=True)

            for k0 in pl.range(0, HIDDEN, k_tile):
                total = residual_out[t0 : t0 + tok_tile, k0 : k0 + k_tile]
                w_tile = pl.cast(w_row[:, k0 : k0 + k_tile], target_type=pl.FP32)
                gamma = pl.add(w_tile, 1.0)
                scaled = pl.row_expand_mul(total, inv_rms)
                normed = pl.col_expand_mul(scaled, gamma)
                y[t0 : t0 + tok_tile, k0 : k0 + k_tile] = pl.cast(normed, target_type=pl.BF16, mode="rint")
        return residual_out, y

    return _qwen3_residual_rmsnorm


qwen3_residual_rmsnorm = build_residual_kernel(inline=True)
qwen3_residual_rmsnorm_test = build_residual_kernel()


def build_fanned_residual_kernel(m: int, col_groups: int = COL_GROUPS,
                                 tok_tile: int = DECODE_TOK_TILE, inline: bool = False):
    """The decode form of the stage above: the per-token reduction split across the columns.

    The body above gives one task every token and walks the hidden axis twice, which is right
    at prefill and is **one task** at a decode batch. The token axis is gone, so the fan-out
    moves to the columns, which splits the sum of squares into a partial pass and a fold:

    1. `pl.spmd(col_groups)` -- each task adds its own column range, stores it to
       `residual_out`, and writes that range's partial sum of squares to one row of a
       `[col_groups, m]` FP32 scratch.
    2. `pl.spmd(col_groups)` -- each task folds the scratch, then normalises and stores its
       own column range.

    `act_quant` needs a third group between these two because it has a per-token output to
    write and a 64-byte line to protect; here the folded value is internal, so each task
    re-folds and no `pl.spmd(1)` is needed.

    **This is not bit-identical to the body above**, where `act_quant`'s fan-out is. Maximum
    is associative and exact; a sum is not, so splitting the reduction reorders 5120
    additions. Every term is a square, so there is no cancellation and the difference is at
    the last bits of an FP32 accumulation -- but it is a difference, which is why the gate
    here is a tolerance against the float64 oracle rather than an exactness claim.

    **`col_groups` must divide `HIDDEN // 64 = 80`**, so that every store covers whole
    128-byte rows; `y` is BF16, so 128 bytes is 64 elements. A narrowing store at a row that
    was not a multiple of 128 bytes was observed to write wrong values into the leading half
    of the remainder with no diagnostic, while an FP32 store of `residual_out` at the same
    misalignment was exact -- the cast is what bound. The INT8 form of this was retested on
    the pinned toolchain and no longer reproduces (see `act_quant.build_fanned_kernel`); this
    BF16 path has not been retested, and the guard stands.

    The token axis is not fanned: this form puts every token tile in one task and splits the
    columns instead, so its per-task work grows with the batch.
    """
    if m % tok_tile:
        raise ValueError(f"m={m} must be a multiple of tok_tile={tok_tile}")
    if HIDDEN % col_groups:
        raise ValueError(f"col_groups={col_groups} must divide HIDDEN={HIDDEN}")
    width = HIDDEN // col_groups
    if width % STORE_GROUP:
        raise ValueError(
            f"col_groups={col_groups} leaves a {width}-column range, which is not a multiple "
            f"of {STORE_GROUP}; a partial BF16 store group was observed to write wrong values "
            f"into the leading half of the remainder, and this path has not been retested. "
            f"col_groups must divide {HIDDEN // STORE_GROUP}")
    # 8 bytes per element, calibrated against the compiler's own reported usage rather than
    # counted off the source. Counting the named tiles gives 14 -- an FP32 residual, a BF16
    # delta, their FP32 sum and its square -- and that would refuse two column counts the
    # hardware takes. The compiler holds the real number and says it: at `col_groups=2` it
    # reports *"Vec buffer usage (327744 bytes) exceeds platform limit (188416 bytes)"*,
    # which is `tok_tile * width * 8` plus the scratch row. This check only moves that
    # refusal earlier. Note what does *not* appear: `m`. The tile is `[tok_tile, width]` at
    # every batch, so the bound is on the column range alone.
    live = tok_tile * width * 8
    if live > VEC_BUFFER_BYTES:
        raise ValueError(f"col_groups={col_groups} needs about {live} B of the {VEC_BUFFER_BYTES} B "
                         f"vector buffer; use at least "
                         f"{HIDDEN * tok_tile * 8 // VEC_BUFFER_BYTES + 1} groups")

    @(pl.jit.inline if inline else pl.jit)
    def _qwen3_residual_rmsnorm_fanned(
        residual: pl.Tensor[[m, HIDDEN], pl.FP32],
        delta: pl.Tensor[[m, HIDDEN], pl.BF16],
        norm_w: pl.Tensor[[HIDDEN], pl.BF16],
        residual_out: pl.Out[pl.Tensor[[m, HIDDEN], pl.FP32]],
        y: pl.Out[pl.Tensor[[m, HIDDEN], pl.BF16]],
    ):
        """One task per column group; `m` is the padded decode batch."""
        part = pl.create_tensor([col_groups, m], dtype=pl.FP32)

        for cg in pl.spmd(col_groups, name_hint="residual_rmsnorm_sum",
                          allow_early_resolve=EARLY_RESOLVE):
            c0 = cg * width
            for t0 in pl.range(0, m, tok_tile):
                delta_f = pl.cast(delta[t0 : t0 + tok_tile, c0 : c0 + width], target_type=pl.FP32)
                total = pl.add(residual[t0 : t0 + tok_tile, c0 : c0 + width], delta_f)
                residual_out[t0 : t0 + tok_tile, c0 : c0 + width] = total
                sq_row = pl.row_sum(pl.mul(total, total))
                part[cg : cg + 1, t0 : t0 + tok_tile] = pl.reshape(sq_row, [1, tok_tile])

        w_row = pl.reshape(norm_w[:], [1, HIDDEN])
        for cg in pl.spmd(col_groups, name_hint="residual_rmsnorm_apply",
                          allow_early_resolve=EARLY_RESOLVE):
            c0 = cg * width
            for t0 in pl.range(0, m, tok_tile):
                folded = pl.col_sum(part[0:col_groups, t0 : t0 + tok_tile])
                mean_sq = pl.mul(folded, HIDDEN_INV)
                variance = pl.reshape(mean_sq, [tok_tile, 1])
                inv_rms = pl.rsqrt(pl.add(variance, EPS), high_precision=True)
                total = residual_out[t0 : t0 + tok_tile, c0 : c0 + width]
                w_tile = pl.cast(w_row[:, c0 : c0 + width], target_type=pl.FP32)
                gamma = pl.add(w_tile, 1.0)
                scaled = pl.row_expand_mul(total, inv_rms)
                normed = pl.col_expand_mul(scaled, gamma)
                y[t0 : t0 + tok_tile, c0 : c0 + width] = pl.cast(normed, target_type=pl.BF16,
                                                                 mode="rint")
        return residual_out, y

    return _qwen3_residual_rmsnorm_fanned


def build_tensor_specs(t: int = BENCH_SEQ, hidden: int = HIDDEN):
    import torch
    from golden import TensorSpec

    def init_x():
        gen = torch.Generator().manual_seed(42)
        return torch.randn(t, hidden, generator=gen).to(torch.bfloat16)

    def init_w():
        # a unit-normal weight; the checkpoint's own sit near zero, the module being `1 + w`
        gen = torch.Generator().manual_seed(43)
        return torch.randn(hidden, generator=gen).to(torch.bfloat16)

    return [
        TensorSpec("x", [t, hidden], torch.bfloat16, init_value=init_x),
        TensorSpec("norm_w", [hidden], torch.bfloat16, init_value=init_w),
        TensorSpec("y", [t, hidden], torch.bfloat16),
    ]


def build_residual_tensor_specs(t: int = BENCH_SEQ, hidden: int = HIDDEN):
    """The five tensors `build_residual_kernel` and its fanned form share."""
    import torch
    from golden import TensorSpec

    def draw(seed, dtype, *shape):
        def make():
            gen = torch.Generator().manual_seed(seed)
            return torch.randn(*shape, generator=gen).to(dtype)
        return make

    return [
        TensorSpec("residual", [t, hidden], torch.float32, init_value=draw(42, torch.float32, t, hidden)),
        TensorSpec("delta", [t, hidden], torch.bfloat16, init_value=draw(43, torch.bfloat16, t, hidden)),
        TensorSpec("norm_w", [hidden], torch.bfloat16, init_value=draw(44, torch.bfloat16, hidden)),
        TensorSpec("residual_out", [t, hidden], torch.float32),
        TensorSpec("y", [t, hidden], torch.bfloat16),
    ]


def golden_qwen3_residual_rmsnorm(tensors):
    """The addition in fp32, as the kernel stores it, and the norm of what it stored."""
    import torch

    import reference

    total = (tensors["residual"].double() + tensors["delta"].double()).float()
    tensors["residual_out"].copy_(total)
    normed = reference.rmsnorm(total.double(), tensors["norm_w"], EPS)
    tensors["y"].copy_(normed.to(torch.bfloat16))


def compare_residual_out():
    """Bit for bit: one fp32 addition of two exactly representable inputs has one rounding.

    The device and the oracle round the same sum the same way, so any difference at all is a
    column range read or written wrongly rather than arithmetic. That makes this the gate
    that a fan-out over the columns has to pass, and it has no tolerance to hide in.
    """
    def compare(actual, expected, **_kwargs):
        same = int((actual == expected).sum())
        total = actual.numel()
        line = f"[stats] residual_out: {same}/{total} elements bit-identical"
        if same == total:
            return True, ""
        worst = float((actual.double() - expected.double()).abs().max())
        return False, f"{line}; worst absolute difference {worst:.3e}"

    return compare


def compare_y():
    """`y` within one BF16 step of the float64 norm, with the exact rate as the real gate.

    A step bound on its own accepts a rounding-mode change -- truncation instead of
    round-to-nearest moves a large fraction of an output by exactly one step and passes --
    so the fraction that is bit-identical is what says the arithmetic is the same.
    """
    import torch

    def compare(actual, expected, **_kwargs):
        step = (actual.view(torch.int16).to(torch.int32)
                - expected.view(torch.int16).to(torch.int32)).abs()
        exact = float((step == 0).float().mean())
        worst = int(step.max())
        line = (f"[stats] y: {exact * 100:.4f}% bit-identical (gate "
                f"{MIN_EXACT_VS_GOLDEN * 100:.0f}%), max {worst} BF16 step(s) off")
        ok = exact >= MIN_EXACT_VS_GOLDEN and worst <= 1
        return ok, line if not ok else ""

    return compare


def golden_qwen3_rmsnorm(tensors):
    import torch

    import reference

    ref = reference.rmsnorm(tensors["x"], tensors["norm_w"], EPS)
    tensors["y"].copy_(ref.to(torch.bfloat16))


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", type=str, default="plain", choices=["plain", "residual", "fanned"],
                        help="the plain norm, the fused add-and-norm, or its decode fan-out")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES; the padded batch for --mode fanned")
    parser.add_argument("--col-groups", type=int, default=COL_GROUPS, help="--mode fanned only")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    tile = DECODE_TOK_TILE if args.mode == "fanned" else TOK_TILE
    if args.seq_len and args.seq_len % tile:
        parser.error(f"--seq-len must be a multiple of {tile} in --mode {args.mode}")
    if args.seq_len:
        plan = [(args.mode, args.seq_len)]
    elif args.mode == "fanned":
        plan = [("fanned", m) for m in FANNED_CASES]
    elif args.mode == "residual":
        plan = [("residual", m) for m in DEFAULT_CASES]
    else:
        plan = [("plain", m) for m in DEFAULT_CASES]
        plan += [("residual", DEFAULT_CASES[0]), ("fanned", FANNED_CASES[0])]
    if len(plan) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")


    for mode, m in plan:
        print(f"[case] rmsnorm {mode} T={m}", flush=True)
        if mode == "plain":
            entry = dict(fn=qwen3_rmsnorm_test, specs=build_tensor_specs(t=m),
                         golden_fn=golden_qwen3_rmsnorm, compare_fn=None)
        else:
            fn = (qwen3_residual_rmsnorm_test if mode == "residual"
                  else build_fanned_residual_kernel(m, col_groups=args.col_groups))
            entry = dict(fn=fn, specs=build_residual_tensor_specs(t=m),
                         golden_fn=golden_qwen3_residual_rmsnorm,
                         compare_fn={"residual_out": compare_residual_out(), "y": compare_y()})
        result = run(
            **entry,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=1e-2,
            atol=1e-5,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED in --mode {mode} at T={m}")
