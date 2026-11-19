# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet gated RMSNorm, and the per-token INT8 quantization an A8W8 reader needs:

    n = bf16(o * rsqrt(mean(o^2) + eps))          per (token, head) over the head dim
    y = bf16(n * norm_w) * silu(z)                gate applied after the norm, in fp32
    y_q[t] = rint(y[t] * 127 / amax(|y[t]|))      per token over all H*D values
    y_scale[t] = amax(|y[t]|) / 127

The two bf16 narrowings are the model's own cast sequence (`Qwen3_5RMSNormGated`).
"""
import pypto.language as pl

from chunk_cumsum import CHUNK_TILE
from config import GDN_QUANT, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
D = QWEN3_8_27B.linear_value_head_dim       # head dimension, the norm's axis
EPS = QWEN3_8_27B.rms_norm_eps
VAL_WIDTH = H * D                           # one token's contiguous run

# quantization
SCALE_MAX = GDN_QUANT.scale_max
AMAX_EPS = GDN_QUANT.amax_eps
# validation
# One chunk of the fixture's chain, then eight; the bf16 and fanned bodies are separate
# programs, so the default set runs each of them once rather than leaving two uncovered.
DEFAULT_CASES = (128, 1024)
TAIL_SEQ = 17           # one token past a tile: the clamped tail of both token-tiled bodies
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# gate
MIN_EXACT_VS_REFERENCE = 0.95   # of y_q elements identical to the fake-quant reference

# scheduling: see `rmsnorm.EARLY_RESOLVE`, which carries the note and the measurement
EARLY_RESOLVE = True

# tiling
TOK_TILE = 16           # tokens per block; 16 scalar scale writes fill one 64-byte line
LANES = 16              # amax lanes the decode form publishes; 16 FP32 is one 64-byte line

# `y_scale` is a [1, T] row and not a [T] vector: an A8W8 projection reshapes its
# slice to the [M, 1] column it multiplies by, and a tile store must be 2-D anyway.


def _gdn_gated_rmsnorm(
    o: pl.Tensor[[T_DYN, H, D], pl.FP16],
    z: pl.Tensor[[T_DYN, VAL_WIDTH], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    y_q: pl.Out[pl.Tensor[[T_DYN, VAL_WIDTH], pl.INT8]],
    y_scale: pl.Out[pl.Tensor[[1, T_DYN], pl.FP32]],
):
    """Any token count. A block walks one whole tile; the last one repeats its last token.

    **The block count is a ceiling and the row index is clamped** so a ragged token count is
    correct: a floor division dropped the final partial tile silently, leaving up to
    `TOK_TILE - 1` tokens with their outputs untouched and no error, and below `TOK_TILE` it
    launched no tasks at all. The last tile repeats its last token up to 15 times,
    which is idempotent -- the body is a pure function of its own row -- and a full tile is
    untouched, so `pl.pipeline` keeps its compile-time trip count and its overlap.
    """
    o.bind_dynamic(0, T_DYN)
    z.bind_dynamic(0, T_DYN)
    y_q.bind_dynamic(0, T_DYN)
    y_scale.bind_dynamic(1, T_DYN)
    t_dim = pl.tensor.dim(o, 0)
    o_flat = pl.reshape(o, [t_dim, VAL_WIDTH])
    for blk in pl.spmd((t_dim + TOK_TILE - 1) // TOK_TILE, name_hint="gated_rmsnorm"):
        t0 = blk * TOK_TILE
        w_row = pl.cast(pl.reshape(norm_w[:], [1, D]), target_type=pl.FP32)
        ones_h8 = pl.full([H, 8], dtype=pl.FP32, value=1.0)   # broadcasts the amax partials
        # stage=2: the tokens are independent, so the next one's loads overlap this
        # one's arithmetic. The kernel is latency-bound on the per-token chain.
        for i in pl.pipeline(TOK_TILE, stage=2):
            r0 = pl.min(t0 + i, t_dim - 1)
            # one token is h*d contiguous values: load the whole run, then take the
            # [H, D] view the norm reduces over. Loading [H, D] directly issues h
            # separate row transfers of d elements.
            of_run = pl.cast(o_flat[r0 : r0 + 1, :], target_type=pl.FP32)
            of = pl.reshape(of_run, [H, D])
            inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(of, of)), 1.0 / D), EPS),
                               high_precision=True)
            # the module narrows the normed values, then the weighted ones; both stay
            # narrow until used, the fp32 copy of a bf16-rounded value being only buffer
            normed_bf = pl.cast(pl.row_expand_mul(of, inv_rms), target_type=pl.BF16, mode="rint")
            normed = pl.cast(normed_bf, target_type=pl.FP32)
            weighted_bf = pl.cast(pl.col_expand_mul(normed, w_row), target_type=pl.BF16, mode="rint")

            zf_run = pl.cast(z[r0 : r0 + 1, :], target_type=pl.FP32)
            zf = pl.reshape(zf_run, [H, D])
            gate = pl.div(zf, pl.add(pl.exp(pl.neg(zf)), 1.0))
            y = pl.mul(pl.cast(weighted_bf, target_type=pl.FP32), gate)

            # The token's amax must reach every row of the [H, D] tile, and it
            # cannot be carried through a scalar: a [1, 1] reduction result is a
            # 4-byte tile ptoas refuses to allocate (32-byte row alignment), and
            # folding the last values with scalar `pl.max` builds a kernel the
            # DEVICE compiler then rejects -- ptoas emits its `ptoas_bitcast`
            # helper (the signed-zero path of a scalar max) without `__aicore__`,
            # so both simulators accept the kernel and only a hardware build
            # fails. Reduce on tiles the whole way instead: columns to [1, 8]
            # partials, clamped there while the layout is still row-major, then
            # broadcast down the head axis so `row_max` lands the same maximum in
            # every row of an [h, 1] column. One `pl.read` of row 0 is left, for
            # the scale, and it is a plain load.
            part_8 = pl.maximum(pl.col_max(pl.reshape(pl.col_max(pl.abs(y)), [D // 8, 8])),
                                pl.full([1, 8], dtype=pl.FP32, value=AMAX_EPS))
            amax = pl.row_max(pl.col_expand_mul(ones_h8, part_8))
            pl.write(y_scale, [0, r0], pl.read(amax, [0, 0]) / SCALE_MAX)

            # No clamp: amax is the tile's own maximum, so |y| * SCALE_MAX / amax
            # <= 127 and a +-127 clamp cannot bind. The cast rounds to nearest;
            # an FP16 hop before INT8 would truncate instead, which is a silent
            # one-step shift on 39% of the output.
            # recip, not div, here: one reciprocal feeds a whole tile's multiply
            q = pl.row_expand_mul(y, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
            # INT32 then INT8, not FP32 straight to INT8: the direct cast rounds through
            # FP16, whose spacing at 127 is a sixteenth of an INT8 step, and that moves
            # 0.9% of the output by one step. `quant_x` carries the measurement.
            q32 = pl.cast(q, target_type=pl.INT32, mode="rint")
            q_i8 = pl.cast(q32, target_type=pl.INT8, mode="trunc")
            y_q[r0 : r0 + 1, :] = pl.reshape(q_i8, [1, VAL_WIDTH])
    return y_q


gdn_gated_rmsnorm = pl.jit.inline(_gdn_gated_rmsnorm)
gdn_gated_rmsnorm_test = pl.jit(_gdn_gated_rmsnorm)


def build_fanned_kernel(m: int, inline: bool = False):
    """The decode form: one task per token, and the scale written by a task that owns its line.

    The body above gives one task a whole token tile and walks its 16 rows at `stage=2`. At
    prefill that is right -- T = 8192 is 512 tiles and the cores are full. **At a decode batch
    it is one task**, and the critical path of a composed layer puts it at **68.5 us**, the
    largest vector stage left on that path.

    The token axis is the one to split, and the only thing stopping the obvious `pl.spmd(m)` is
    the scale: 16 FP32 scalars are exactly one 64-byte cache line, and a line written from two
    blocks loses one of them silently. So each task publishes its token's amax as a `[1, LANES]`
    tile into a row of a `[m, LANES]` scratch -- 16 FP32 is the whole line, so no two tasks
    share one -- and a single `pl.spmd(1)` folds the scratch and writes `y_scale`.

    **This is bit-identical to the body above.** Each task computes one token's norm, gate and
    quantisation with the same operations in the same order; only the amax reduction is
    reassociated, from 8 lanes to 16, and a maximum is exact under any association. The scale
    is still `amax / SCALE_MAX` by division rather than by a multiply, which the reciprocal of
    127 would not reproduce. The composed block's output is **exact over all 81920 elements**
    against the shipped build's own device output, on random and on real weights.

    This stage has no decode-shaped specs of its own -- its `o` comes from the prefill chunked
    chain, which needs a chunk multiple -- so it is measured and gated inside
    `decode_gdn_block`. A composed linear-attention layer goes **785.8 -> 727.4 us at batch 16**
    and 1000.0 -> 953.3 at batch 32, which is 175.1 us/token over the 48 layers at batch 16.
    """
    @(pl.jit.inline if inline else pl.jit)
    def _gdn_gated_rmsnorm_fanned(
        o: pl.Tensor[[m, H, D], pl.FP16],
        z: pl.Tensor[[m, VAL_WIDTH], pl.BF16],
        norm_w: pl.Tensor[[D], pl.BF16],
        y_q: pl.Out[pl.Tensor[[m, VAL_WIDTH], pl.INT8]],
        y_scale: pl.Out[pl.Tensor[[1, m], pl.FP32]],
    ):
        """One task per token; `m` is the padded decode batch."""
        o_flat = pl.reshape(o, [m, VAL_WIDTH])
        part = pl.create_tensor([m, LANES], dtype=pl.FP32)

        for r0 in pl.spmd(m, name_hint="gated_rmsnorm_token",
                          allow_early_resolve=EARLY_RESOLVE):
            w_row = pl.cast(pl.reshape(norm_w[:], [1, D]), target_type=pl.FP32)
            ones_h = pl.full([H, LANES], dtype=pl.FP32, value=1.0)
            of_run = pl.cast(o_flat[r0 : r0 + 1, :], target_type=pl.FP32)
            of = pl.reshape(of_run, [H, D])
            inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(of, of)), 1.0 / D), EPS),
                               high_precision=True)
            normed_bf = pl.cast(pl.row_expand_mul(of, inv_rms), target_type=pl.BF16, mode="rint")
            normed = pl.cast(normed_bf, target_type=pl.FP32)
            weighted_bf = pl.cast(pl.col_expand_mul(normed, w_row), target_type=pl.BF16, mode="rint")

            zf_run = pl.cast(z[r0 : r0 + 1, :], target_type=pl.FP32)
            zf = pl.reshape(zf_run, [H, D])
            gate = pl.div(zf, pl.add(pl.exp(pl.neg(zf)), 1.0))
            y = pl.mul(pl.cast(weighted_bf, target_type=pl.FP32), gate)

            # the token's amax as LANES partials, clamped here so the fold inherits it
            lane_max = pl.col_max(pl.reshape(pl.col_max(pl.abs(y)), [D // LANES, LANES]))
            lanes = pl.maximum(lane_max, pl.full([1, LANES], dtype=pl.FP32, value=AMAX_EPS))
            part[r0 : r0 + 1, 0:LANES] = lanes
            amax = pl.row_max(pl.col_expand_mul(ones_h, lanes))

            q = pl.row_expand_mul(y, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
            q32 = pl.cast(q, target_type=pl.INT32, mode="rint")
            q_i8 = pl.cast(q32, target_type=pl.INT8, mode="trunc")
            y_q[r0 : r0 + 1, :] = pl.reshape(q_i8, [1, VAL_WIDTH])

        for _solo in pl.spmd(1, name_hint="gated_rmsnorm_scale",
                             allow_early_resolve=EARLY_RESOLVE):
            folded = pl.row_max(part[0:m, 0:LANES])
            y_scale[0:1, 0:m] = pl.div(pl.reshape(folded, [1, m]), SCALE_MAX)
        return y_q, y_scale

    return _gdn_gated_rmsnorm_fanned


def _gdn_gated_rmsnorm_bf16(
    o: pl.Tensor[[T_DYN, H, D], pl.FP16],
    z: pl.Tensor[[T_DYN, VAL_WIDTH], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    y: pl.Out[pl.Tensor[[T_DYN, VAL_WIDTH], pl.BF16]],
):
    """Any token count. A block walks one whole tile; the last one repeats its last token.

    **The block count is a ceiling and the row index is clamped** so a ragged token count is
    correct: a floor division dropped the final partial tile silently, leaving up to
    `TOK_TILE - 1` tokens with their outputs untouched and no error, and below `TOK_TILE` it
    launched no tasks at all. The last tile repeats its last token up to 15 times,
    which is idempotent -- the body is a pure function of its own row -- and a full tile is
    untouched, so `pl.pipeline` keeps its compile-time trip count and its overlap.
    """
    o.bind_dynamic(0, T_DYN)
    z.bind_dynamic(0, T_DYN)
    y.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(o, 0)
    o_flat = pl.reshape(o, [t_dim, VAL_WIDTH])
    for blk in pl.spmd((t_dim + TOK_TILE - 1) // TOK_TILE, name_hint="gated_rmsnorm_bf16"):
        t0 = blk * TOK_TILE
        w_row = pl.cast(pl.reshape(norm_w[:], [1, D]), target_type=pl.FP32)
        for i in pl.pipeline(TOK_TILE, stage=2):
            r0 = pl.min(t0 + i, t_dim - 1)
            of_run = pl.cast(o_flat[r0 : r0 + 1, :], target_type=pl.FP32)
            of = pl.reshape(of_run, [H, D])
            inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(of, of)), 1.0 / D), EPS),
                               high_precision=True)
            normed_bf = pl.cast(pl.row_expand_mul(of, inv_rms), target_type=pl.BF16, mode="rint")
            normed = pl.cast(normed_bf, target_type=pl.FP32)
            weighted_bf = pl.cast(pl.col_expand_mul(normed, w_row), target_type=pl.BF16, mode="rint")

            zf_run = pl.cast(z[r0 : r0 + 1, :], target_type=pl.FP32)
            zf = pl.reshape(zf_run, [H, D])
            gate = pl.div(zf, pl.add(pl.exp(pl.neg(zf)), 1.0))
            y_bf = pl.cast(pl.mul(pl.cast(weighted_bf, target_type=pl.FP32), gate),
                           target_type=pl.BF16, mode="rint")
            y[r0 : r0 + 1, :] = pl.reshape(y_bf, [1, VAL_WIDTH])
    return y


def build_fanned_bf16_kernel(m: int, inline: bool = False):
    """The decode form of the bf16-output variant: one task per token, and nothing else.

    `build_fanned_kernel` needs a scratch and a second `pl.spmd` group only because of its
    per-token scale: 16 FP32 scalars are one 64-byte line. This variant has no scale, so the
    fan-out is the whole change -- one task per token, no fold, no hand-off.

    Bit-identical to the body above: the same operations on the same token in the same order.
    """
    @(pl.jit.inline if inline else pl.jit)
    def _gdn_gated_rmsnorm_bf16_fanned(
        o: pl.Tensor[[m, H, D], pl.FP16],
        z: pl.Tensor[[m, VAL_WIDTH], pl.BF16],
        norm_w: pl.Tensor[[D], pl.BF16],
        y: pl.Out[pl.Tensor[[m, VAL_WIDTH], pl.BF16]],
    ):
        """One task per token; `m` is the padded decode batch."""
        o_flat = pl.reshape(o, [m, VAL_WIDTH])
        for r0 in pl.spmd(m, name_hint="gated_rmsnorm_bf16_token",
                          allow_early_resolve=EARLY_RESOLVE):
            w_row = pl.cast(pl.reshape(norm_w[:], [1, D]), target_type=pl.FP32)
            of_run = pl.cast(o_flat[r0 : r0 + 1, :], target_type=pl.FP32)
            of = pl.reshape(of_run, [H, D])
            inv_rms = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(of, of)), 1.0 / D), EPS),
                               high_precision=True)
            normed_bf = pl.cast(pl.row_expand_mul(of, inv_rms), target_type=pl.BF16, mode="rint")
            normed = pl.cast(normed_bf, target_type=pl.FP32)
            weighted_bf = pl.cast(pl.col_expand_mul(normed, w_row), target_type=pl.BF16, mode="rint")

            zf_run = pl.cast(z[r0 : r0 + 1, :], target_type=pl.FP32)
            zf = pl.reshape(zf_run, [H, D])
            gate = pl.div(zf, pl.add(pl.exp(pl.neg(zf)), 1.0))
            y_bf = pl.cast(pl.mul(pl.cast(weighted_bf, target_type=pl.FP32), gate),
                           target_type=pl.BF16, mode="rint")
            y[r0 : r0 + 1, :] = pl.reshape(y_bf, [1, VAL_WIDTH])
        return y

    return _gdn_gated_rmsnorm_bf16_fanned


gdn_gated_rmsnorm_bf16 = pl.jit.inline(_gdn_gated_rmsnorm_bf16)
gdn_gated_rmsnorm_bf16_test = pl.jit(_gdn_gated_rmsnorm_bf16)


def golden_y(o, z, norm_w, eps: float = EPS):
    """The kernel's fp32 arithmetic in its order, both bf16 narrowings included. -> fp32 [T, H*D]."""
    import torch

    t, h, d = o.shape
    of = o.float()
    inv_rms = torch.rsqrt((of * of).sum(dim=-1, keepdim=True) * (1.0 / d) + eps)
    normed = (of * inv_rms).to(torch.bfloat16).float()
    weighted = (normed * norm_w.float()).to(torch.bfloat16).float()
    # z arrives as the [T, H*D] row `in_proj_z` writes; the norm is over the head
    zf = z.float().reshape(t, h, d)
    return (weighted * (zf / (torch.exp(-zf) + 1.0))).reshape(t, h * d)


def golden_quant(y):
    """The int8 epilogue on an fp32 [T, C]: -> (int8 [T, C], fp32 scale [T]).

    The kernel reaches the same scale by a high-precision reciprocal and a
    multiply, torch by a divide; the gate's one-step allowance covers the
    difference, and the simulator and hardware runs measure it.
    """
    import torch

    amax = y.abs().amax(dim=-1).clamp_min(AMAX_EPS)
    q = torch.round(y * (SCALE_MAX / amax)[:, None]).clamp(-SCALE_MAX, SCALE_MAX)
    return q.to(torch.int8), (amax / SCALE_MAX).float()


def golden_gdn_gated_rmsnorm(tensors):
    q, scale = golden_quant(golden_y(tensors["o"], tensors["z"], tensors["norm_w"]))
    tensors["y_q"].copy_(q)
    tensors["y_scale"].copy_(scale.reshape(1, -1))


def golden_gdn_gated_rmsnorm_bf16(tensors):
    import torch

    y = golden_y(tensors["o"], tensors["z"], tensors["norm_w"])
    tensors["y"].copy_(y.to(torch.bfloat16))


def _draw(key: str, t: int, chunk: int, weights: str | None):
    """A block-chain tensor as a lazy spec value; one chain serves every spec and the compare.

    **A ragged token count draws the chain at the next chunk multiple and keeps the first `t`
    rows.** The chain cannot produce a ragged count itself -- it slices `k[t0 : t0 + chunk]`
    against a `[chunk, chunk]` mask, so a ragged `t` raises from `reference.kkt` -- but this
    stage is token-local: each row's result is a pure function of that row, and `y`, `y_q`
    and `y_scale` are per-token, so the first `t` rows of the padded chain are exactly the
    reference for a ragged `t`. The kernel's clamped tail is therefore testable here,
    in the stage that owns it, and at the chain's own distribution rather than a drawn one.

    No composed chain reaches it: every stage of the delta rule tiles the token axis by
    `chunk`, so `gdn_block` and `prefill_layer` only ever hand this stage whole tiles. In
    production nothing hands it a ragged count either -- packed prefill aligns each row to
    a multiple of 256.
    """
    import reference

    t_pad = -(-t // chunk) * chunk
    src = reference.block_inputs(key, t_pad, QWEN3_8_27B, chunk, weights=weights)
    if t_pad == t:
        return src
    return lambda: src()[:t]   # every key's dim 0 is the token axis, `y_scale` included


def _norm_weight(weights: str | None):
    """`norm.weight` of the weight set the inputs were drawn from."""
    import torch

    import reference

    if weights is None:
        return reference.make_block_weights(QWEN3_8_27B)["norm.weight"]
    return torch.load(weights, weights_only=True)["norm.weight"]


def build_tensor_specs(t: int = BENCH_SEQ, h: int = H, d: int = D, chunk: int = CHUNK_TILE,
                       weights: str | None = None, quant: bool = True):
    import torch
    from golden import TensorSpec

    def draw(key, dtype, flat: bool = False):
        src = _draw(key, t, chunk, weights)
        if flat:
            return lambda: src().to(dtype).reshape(t, -1)
        return lambda: src().to(dtype)

    specs = [
        TensorSpec("o", [t, h, d], torch.float16, init_value=draw("o", torch.float16)),
        TensorSpec("z", [t, h * d], torch.bfloat16,
                   init_value=draw("z", torch.bfloat16, flat=True)),
        TensorSpec("norm_w", [d], torch.bfloat16, init_value=lambda: _norm_weight(weights)),
    ]
    if quant:
        specs += [TensorSpec("y_q", [t, h * d], torch.int8),
                  TensorSpec("y_scale", [1, t], torch.float32)]
    else:
        specs += [TensorSpec("y", [t, h * d], torch.bfloat16)]
    return specs


def _reference_y(inputs):
    import reference

    o = inputs["o"]
    z = inputs["z"].reshape(o.shape)
    return reference.gated_rmsnorm(o, z, inputs["norm_w"], EPS).flatten(1)


def compare_y_q():
    """Two gates on `y_q`, plus the numbers the gates do not say.

    Against the kernel's own golden: +-1 INT8 step (one step is 0.8% of full
    scale, and an fp32 rsqrt or exp differing in the last bit moves a value
    across a rounding boundary) for all but 0.5% of elements. Against the
    fake-quant reference: the same step bound for all but 1e-4 of elements,
    `MIN_EXACT_VS_REFERENCE` of them identical, and the per-token scale within
    2e-2. The exact-match rate is a gate and not a statistic -- a rounding-mode
    change moves a large fraction of the output by exactly one step, which a
    step bound alone accepts.

    Reported when it fails: the dequantized output's distance from the float64 `y`
    and from the fake-quant reference's dequantized `y`.
    """
    import torch

    from golden import ratio_allclose

    import reference

    golden_gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=0.005)
    ref_gate = ratio_allclose(atol=1, rtol=0, max_error_ratio=1e-4)

    def compare(actual, expected, *, inputs, actual_outputs, **kw):
        ok, detail = golden_gate(actual, expected, inputs=inputs, actual_outputs=actual_outputs, **kw)
        step = (actual.to(torch.int32) - expected.to(torch.int32)).abs()
        lines = [f"y_q vs golden: {float((step == 0).float().mean()) * 100:.4f}% exact, "
                 f"max {int(step.max())} step(s) off"]
        if actual_outputs is not None and "y_scale" in actual_outputs:
            scale_dev = actual_outputs["y_scale"].cpu().float().reshape(-1)
            y_ref = _reference_y(inputs)
            q_ref, scale_ref = reference.quantize_rows(y_ref)
            ref_ok, ref_detail = ref_gate(actual, q_ref, inputs=inputs, actual_outputs=actual_outputs, **kw)
            step = (actual.to(torch.int32) - q_ref.to(torch.int32)).abs()
            exact_ref = float((step == 0).float().mean())
            scale_rel = float(((scale_dev - scale_ref).abs() / scale_ref).max())
            scale_ok = scale_rel <= 2e-2
            exact_ok = exact_ref >= MIN_EXACT_VS_REFERENCE
            ref_ok = ref_ok and exact_ok
            lines.append(f"y_q vs fake-quant reference: "
                         f"{exact_ref * 100:.4f}% exact (gate {MIN_EXACT_VS_REFERENCE * 100:.0f}%), "
                         f"max {int(step.max())} step(s) off, "
                         f"y_scale max rel diff {scale_rel:.2e} "
                         f"-> {'PASS' if ref_ok and scale_ok else 'FAIL'}")
            y_dev = reference.dequantize(actual, scale_dev)
            y_fq = reference.dequantize(q_ref, scale_ref)
            lines.append(f"dequantised y: rel frob {float((y_dev - y_ref).norm() / y_ref.norm()):.3e} "
                         f"vs float64 y, {float((y_dev - y_fq).norm() / y_fq.norm()):.3e} "
                         f"vs fake-quant reference")
            if not (ref_ok and scale_ok):
                why = [] if exact_ok else [f"only {exact_ref * 100:.2f}% exact "
                                           f"(< {MIN_EXACT_VS_REFERENCE * 100:.0f}%): "
                                           f"a systematic rounding change, not boundary flips"]
                if not scale_ok:
                    why.append(f"y_scale off by {scale_rel:.2e}")
                ok, detail = False, "; ".join([detail, f"vs fake-quant reference: {ref_detail}", *why])
        if ok:
            return True, ""
        return False, "\n".join([*(f"    {line}" for line in lines), detail])

    return compare


def compare_y():
    """The bf16 variant: the bf16 gate the roadmap sets for bf16 outputs, plus the truth distance."""
    from golden import ratio_allclose

    gate = ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)

    def compare(actual, expected, *, inputs, **kw):
        ok, detail = gate(actual, expected, inputs=inputs, **kw)
        if ok:
            return True, ""
        y_ref = _reference_y(inputs)
        y_dev = actual.double()
        head = (f"    bf16 y: rel frob {float((y_dev - y_ref).norm() / y_ref.norm()):.3e} vs float64 y, "
                f"{float((y_dev - expected.double()).norm() / y_ref.norm()):.3e} vs golden")
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
    parser.add_argument("--bf16-out", action="store_true", default=False,
                        help="the bf16-output variant, to price the int8 epilogue")
    parser.add_argument("--fanned", action="store_true", default=False,
                        help="the decode form: one task per token (int8 output only)")
    parser.add_argument("--rounds", type=int, default=0,
                        help="also time the kernel over this many rounds (device only)")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    # Four programs live here -- the int8 epilogue and the bf16 one, each in a token-tiled
    # and a per-token form -- so the default set runs every one of them: the int8 epilogue
    # at both shapes and the other three once at the smaller.
    suffix = "_fanned" if args.fanned else ""
    if args.bf16_out or args.fanned:
        plan = [(("bf16" if args.bf16_out else "int8") + suffix, t) for t in cases]
    else:
        plan = [("int8", t) for t in cases]
        if not args.seq_len:
            plan += [("bf16", cases[0]), ("int8_fanned", cases[0]), ("bf16_fanned", cases[0])]
            # and the clamped tail in both token-tiled bodies: the fanned ones take any
            # token count by construction, one task per token and no tile to fall short of.
            plan += [("int8", TAIL_SEQ), ("bf16", TAIL_SEQ)]
    for label, t in plan:
        quant = not label.startswith("bf16")
        if label.endswith("_fanned"):
            entry = build_fanned_kernel(t) if quant else build_fanned_bf16_kernel(t)
        else:
            entry = gdn_gated_rmsnorm_test if quant else gdn_gated_rmsnorm_bf16_test
        print(f"[case] gated_rmsnorm {label} T={t}", flush=True)
        result = run(
            fn=entry,
            specs=build_tensor_specs(t=t, weights=args.weights, quant=quant),
            golden_fn=golden_gdn_gated_rmsnorm if quant else golden_gdn_gated_rmsnorm_bf16,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-3,
            atol=1e-5,
            compare_fn={"y_q": compare_y_q()} if quant else {"y": compare_y()},
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            # Stop at the first failure: one device failure poisons the chip run
            # lane, so every later case reports a failure it did not cause.
            raise SystemExit(f"FAILED at {label} T={t}")
