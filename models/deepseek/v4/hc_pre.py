# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 hc_pre (dynamic shape) fused into a SINGLE pl.spmd scope.

Based on the dynamic-shape hc_pre.py (5 scopes: linear / split_pre_post /
write_post / comb_sinkhorn / mix_x). Here all five are folded into ONE spmd
loop over the dynamic token tiles: per tile we do the RMS+linear matmul into a
GM scratch row, then read it straight back (intra-task) to produce pre/post/
mix_x/comb — no cross-scope GM intermediates (pre_val_store / post_pad_store /
comb_logits are gone).

Requires the SplitVectorKernel cube->vec fix (pypto#1761): fusing the cube
matmul with the vector epilogue in one task previously 507018-deadlocked
because the split=0 cube<->vec pipe ops were replayed into both AIV subblocks.
"""


import pypto.language as pl

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
NORM_EPS = M.rms_norm_eps

# kernel-local
MIX_PAD = 32  # MIX_HC padded for vector ops
HC_PAD = 8  # HC_MULT padded
NEG_INF = -1e20
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)

# tiling
T_TILE = 16  # unified row-tile for the fused spmd
SINK_ST = T_TILE * HC_MULT  # rows of the [T*hc, HC_PAD] reshape (each group -> one row)
SINK_W = HC_MULT * HC_PAD   # padded width of the 4 stacked comb groups (4*8 = 32)
RMS_K_TILE = 128
LINEAR_K_TILE = 128
# 512: the mix_x heads are now accumulated SEQUENTIALLY (load->cast->mac->free
# one head at a time) instead of materializing all 4 x_fp32 + 4 y tiles at once,
# so the Vec live-set drops ~3x and D_TILE=512 (512B = one L2 line) fits.
D_TILE = 512
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0
# The fused pre-mix below is hand-unrolled for HC_MULT == 4: the four pre0..pre3
# residual scales, the four mix_g0..mix_g3 / comb base loads, the comb_off =
# HC_MULT * 2 offset and the in_h * HC_MULT column strides all assume it. Fail
# loudly if the model config changes hc_mult instead of silently mixing the
# wrong comb columns.
assert HC_MULT == 4, (
    f"hc_pre is hand-specialized to HC_MULT == 4, got {HC_MULT}; "
    "regenerate the pre0..pre3 / mix_g0..mix_g3 unrolling for the new hc_mult before using it."
)


@pl.jit.inline
def hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    t_dim = pl.tensor.dim(x, 0)
    x_flat = pl.reshape(x, [t_dim, HC_DIM])
    scale0 = pl.read(hc_scale, [0])
    scale1 = pl.read(hc_scale, [1])
    scale2 = pl.read(hc_scale, [2])
    hc_base_2d = pl.reshape(hc_base, [1, MIX_HC])

    # Raw RMS+linear result, spilled per tile and read straight back within the
    # same task. mixes_gm is sized to the static upper bound T_MAX; the loop
    # below only touches the t_dim real rows. The cube (matmul/AIC) writes it and
    # the vector epilogue (AIV) reads it back in the SAME task — the AIV-side
    # MTE3->MTE2 fence orders the self-RAW correctly; the cube<->vec pipe sync is
    # the part that needs pypto#1761.
    mixes_gm = pl.create_tensor([T_MAX, MIX_PAD], dtype=pl.FP32)

    for ob in pl.spmd(t_dim // T_TILE, name_hint="hc_pre_1spmd"):
        t0 = ob * T_TILE

        # --- linear: RMS norm + hc_fn projection -> mixes_gm[t0] ---
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        mix_acc = pl.create_tensor([T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
            kl0 = kb * LINEAR_K_TILE
            x_lin = pl.cast(x_flat[t0:t0 + T_TILE, kl0:kl0 + LINEAR_K_TILE], target_type=pl.FP32)
            x_sq = pl.mul(x_lin, x_lin)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(x_sq), [1, T_TILE]))
            w_lin = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, kl0], valid_shape=[MIX_HC, LINEAR_K_TILE])
            if kb == 0:
                mix_acc = pl.matmul(x_lin, w_lin, b_trans=True, out_dtype=pl.FP32)
            else:
                mix_acc = pl.matmul_acc(mix_acc, x_lin, w_lin, b_trans=True)
        mean_sq = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms_val = pl.rsqrt(mean_sq, high_precision=True)
        inv_rms_col = pl.reshape(inv_rms_val, [T_TILE, 1])
        mixes_gm[t0:t0 + T_TILE, 0:MIX_PAD] = pl.row_expand_mul(mix_acc, inv_rms_col)

        # Bias bases as tiles (col_expand needs tile-level operands).
        pre_base = pl.load(hc_base_2d, [0, 0], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)
        post_base = pl.load(hc_base_2d, [0, HC_MULT], [1, HC_PAD], target_memory=pl.MemorySpace.Vec)

        # --- pre = sigmoid(mixes[:, :hc]*s0 + base) + eps. Kept in Vec, consumed
        # by mix_x below in the SAME scope (no GM round-trip). ---
        pre_in = pl.load(mixes_gm, [t0, 0], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec)
        pre_scaled = pl.mul(pre_in, scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_eps = pl.add(pre_sig, HC_EPS)

        # --- post = 2*sigmoid(mixes[:, hc:2hc]*s1 + base) -> store ---
        post_in = pl.load(mixes_gm, [t0, HC_MULT], [T_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec)
        post_scaled = pl.mul(post_in, scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_tile = pl.set_validshape(pl.mul(post_sig, 2.0), T_TILE, HC_MULT)
        pl.store(post_tile, [t0, 0], post)

        # --- mix_x = sum_h pre[:, h] * x[:, h, :]. Transpose so each head is a
        # 32B-aligned row, then materialize each [T_TILE,1] scale into its own
        # buffer (tmuls by 1.0). ---
        pre_eps_t = pl.transpose(pre_eps, axis1=0, axis2=1)  # [HC_PAD, T_TILE]
        pre0 = pl.mul(pl.reshape(pre_eps_t[0:1, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre1 = pl.mul(pl.reshape(pre_eps_t[1:2, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre2 = pl.mul(pl.reshape(pre_eps_t[2:3, 0:T_TILE], [T_TILE, 1]), 1.0)
        pre3 = pl.mul(pl.reshape(pre_eps_t[3:4, 0:T_TILE], [T_TILE, 1]), 1.0)
        for db in pl.range(D // D_TILE):
            d0 = db * D_TILE
            # Accumulate the 4 heads SEQUENTIALLY: each head's x is loaded, cast
            # to FP32, weighted by pre_h, added into y, then freed before the next
            # head loads. This keeps only ~1 x_fp32 + y live (vs all 4 x + 4 y),
            # so the Vec live-set is ~3x smaller and D_TILE=512 fits. The
            # left-to-right sum order also matches the torch golden's `y += ...`.
            y = pl.row_expand_mul(pl.cast(pl.load(x_flat, [t0, 0 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32), pre0)
            y = pl.add(y, pl.row_expand_mul(pl.cast(pl.load(x_flat, [t0, 1 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32), pre1))
            y = pl.add(y, pl.row_expand_mul(pl.cast(pl.load(x_flat, [t0, 2 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32), pre2))
            y = pl.add(y, pl.row_expand_mul(pl.cast(pl.load(x_flat, [t0, 3 * D + d0], [T_TILE, D_TILE], target_memory=pl.MemorySpace.Vec), target_type=pl.FP32), pre3))
            pl.store(pl.cast(y, target_type=pl.BF16, mode="rint"), [t0, d0], x_mixed)

        # --- comb = sinkhorn(reshape(mixes[:, 2hc:]*s2 + base, hc, hc)). The 4
        # groups are stacked col-wise (pl.concat) into one padded [T_TILE, hc*8]
        # tile, then a FREE reshape to [T_TILE*hc, 8] makes each group ONE row so
        # the 20-iter loop's per-group row_sum / row_expand_div collapse into a
        # SINGLE op each (the loop is latency-bound on tiny tiles). Groups are
        # padded to 8 cols (not 4) because ptoas requires >=32B row-major tile
        # rows. Col-normalize (over the 4 groups) sums the 4 col-blocks and
        # divides by col_sum replicated back via concat. Pad cols stay 0
        # throughout, so this is bit-identical to the per-group form. ---
        comb_off = HC_MULT * 2
        mix_g0 = pl.load(mixes_gm, [t0, comb_off + 0 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g1 = pl.load(mixes_gm, [t0, comb_off + 1 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g2 = pl.load(mixes_gm, [t0, comb_off + 2 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        mix_g3 = pl.load(mixes_gm, [t0, comb_off + 3 * HC_MULT], [T_TILE, HC_PAD], valid_shapes=[T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb0 = pl.load(hc_base_2d, [0, comb_off + 0 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb1 = pl.load(hc_base_2d, [0, comb_off + 1 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb2 = pl.load(hc_base_2d, [0, comb_off + 2 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        cb3 = pl.load(hc_base_2d, [0, comb_off + 3 * HC_MULT], [1, HC_PAD], valid_shapes=[1, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row0_p = pl.fillpad(pl.add(pl.mul(mix_g0, scale2), pl.col_expand(mix_g0, cb0)), pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(pl.add(pl.mul(mix_g1, scale2), pl.col_expand(mix_g1, cb1)), pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(pl.add(pl.mul(mix_g2, scale2), pl.col_expand(mix_g2, cb2)), pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(pl.add(pl.mul(mix_g3, scale2), pl.col_expand(mix_g3, cb3)), pad_value=pl.PadValue.min)

        # softmax over each group's hc cols (group = one row of the [SINK_ST, HC_PAD] reshape)
        sm = pl.reshape(pl.concat(pl.concat(row0_p, row1_p), pl.concat(row2_p, row3_p)), [SINK_ST, HC_PAD])
        sm_max_tmp = pl.create_tile([SINK_ST, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        sm_sum_tmp = pl.create_tile([SINK_ST, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        sm_max = pl.row_max(sm, sm_max_tmp)
        sm_exp = pl.exp(pl.row_expand_sub(sm, sm_max))
        sm_sum = pl.row_sum(sm_exp, sm_sum_tmp)
        sm_soft = pl.add(pl.row_expand_div(sm_exp, sm_sum), HC_EPS)
        sm_soft = pl.fillpad(pl.set_validshape(sm_soft, SINK_ST, HC_MULT), pad_value=pl.PadValue.zero)
        c32 = pl.reshape(sm_soft, [T_TILE, SINK_W])

        # first col-normalize (over the 4 groups = the 4 padded col-blocks)
        cs = pl.add(pl.add(c32[0:T_TILE, 0 * HC_PAD:1 * HC_PAD], c32[0:T_TILE, 1 * HC_PAD:2 * HC_PAD]),
                    pl.add(c32[0:T_TILE, 2 * HC_PAD:3 * HC_PAD], c32[0:T_TILE, 3 * HC_PAD:4 * HC_PAD]))
        cs = pl.add(cs, HC_EPS)
        c32 = pl.div(c32, pl.concat(pl.concat(cs, cs), pl.concat(cs, cs)))

        sink_sum_tmp = pl.create_tile([SINK_ST, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        for sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            sr = pl.reshape(c32, [SINK_ST, HC_PAD])
            sr_sum = pl.add(pl.row_sum(sr, sink_sum_tmp), HC_EPS)
            c32 = pl.reshape(pl.row_expand_div(sr, sr_sum), [T_TILE, SINK_W])
            cs = pl.add(pl.add(c32[0:T_TILE, 0 * HC_PAD:1 * HC_PAD], c32[0:T_TILE, 1 * HC_PAD:2 * HC_PAD]),
                        pl.add(c32[0:T_TILE, 2 * HC_PAD:3 * HC_PAD], c32[0:T_TILE, 3 * HC_PAD:4 * HC_PAD]))
            cs = pl.add(cs, HC_EPS)
            c32 = pl.div(c32, pl.concat(pl.concat(cs, cs), pl.concat(cs, cs)))

        # Materialize each padded group block (x1.0) into a fresh tile so
        # set_validshape sees a dynamic-validShape source (sub-views don't qualify).
        out0 = pl.mul(c32[0:T_TILE, 0 * HC_PAD:1 * HC_PAD], 1.0)
        out1 = pl.mul(c32[0:T_TILE, 1 * HC_PAD:2 * HC_PAD], 1.0)
        out2 = pl.mul(c32[0:T_TILE, 2 * HC_PAD:3 * HC_PAD], 1.0)
        out3 = pl.mul(c32[0:T_TILE, 3 * HC_PAD:4 * HC_PAD], 1.0)
        pl.store(pl.set_validshape(out0, T_TILE, HC_MULT), [t0, 0 * HC_MULT], comb)
        pl.store(pl.set_validshape(out1, T_TILE, HC_MULT), [t0, 1 * HC_MULT], comb)
        pl.store(pl.set_validshape(out2, T_TILE, HC_MULT), [t0, 2 * HC_MULT], comb)
        pl.store(pl.set_validshape(out3, T_TILE, HC_MULT), [t0, 3 * HC_MULT], comb)
    return x_mixed


@pl.jit
def hc_pre_test(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    post: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    comb: pl.Out[pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    x_mixed.bind_dynamic(0, T_DYN)
    post.bind_dynamic(0, T_DYN)
    comb.bind_dynamic(0, T_DYN)

    x_mixed = hc_pre(x, hc_fn, hc_scale, hc_base, x_mixed, post, comb)
    return x_mixed


def golden_hc_pre(tensors):
    """Torch reference, direct port of model.py Block.hc_pre + hc_split_sinkhorn."""
    import torch

    x = tensors["x"].float()  # [T, hc, D]
    hc_fn = tensors["hc_fn"].float()  # [mix_hc, hc*D]
    hc_scale = tensors["hc_scale"].float()  # [3]
    hc_base = tensors["hc_base"].float()  # [mix_hc]

    t_dim = x.shape[0]
    x_flat_2d = x.reshape(t_dim, HC_DIM)

    sq_sum = torch.zeros(t_dim, 1, dtype=torch.float32)
    for k0 in range(0, HC_DIM, RMS_K_TILE):
        x_chunk = x_flat_2d[:, k0:k0 + RMS_K_TILE]
        sq_sum += (x_chunk * x_chunk).sum(dim=1, keepdim=True)
    rsqrt = torch.rsqrt(sq_sum * HC_DIM_INV + NORM_EPS)

    mix_cols = []
    for m in range(MIX_HC):
        mix_col = torch.zeros(t_dim, 1, dtype=torch.float32)
        for k0 in range(0, HC_DIM, LINEAR_K_TILE):
            x_chunk = x_flat_2d[:, k0:k0 + LINEAR_K_TILE]
            w_chunk = hc_fn[m:m + 1, k0:k0 + LINEAR_K_TILE]
            mix_col += (x_chunk * w_chunk).sum(dim=1, keepdim=True)
        mix_cols.append(mix_col * rsqrt)
    mixes = torch.cat(mix_cols, dim=1)  # [T, mix_hc]

    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(t_dim, HC_MULT, HC_MULT)

    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    for _ in range(HC_SINKHORN_ITER - 1):
        comb_t = comb_t / (comb_t.sum(-1, keepdim=True) + HC_EPS)
        comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)

    y = torch.zeros(t_dim, D, dtype=torch.float32)
    for h in range(HC_MULT):
        y += x[:, h, :] * pre[:, h:h + 1]

    def _to_device_bf16(value):
        rounded = (value.contiguous().view(torch.int32) + 0x8000) & -0x10000
        return rounded.view(torch.float32).to(torch.bfloat16)

    tensors["x_mixed"][:] = _to_device_bf16(y).reshape(t_dim, D)
    tensors["post"][:] = post_t.reshape(t_dim, HC_MULT)
    tensors["comb"][:] = comb_t.reshape(t_dim, HC_MULT * HC_MULT)


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    T = B * S

    def init_x():
        return torch.rand(T, HC_MULT, D) - 0.5
    def init_hc_fn():
        return (torch.randn(MIX_HC, HC_DIM) - 0.5) / (HC_DIM ** 0.5)
    def init_hc_scale():
        return torch.ones(3) * 0.5
    def init_hc_base():
        return torch.zeros(MIX_HC)

    return [
        TensorSpec("x", [T, HC_MULT, D], torch.bfloat16, init_value=init_x),
        TensorSpec("hc_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_fn),
        TensorSpec("hc_scale", [3], torch.float32, init_value=init_hc_scale),
        TensorSpec("hc_base", [MIX_HC], torch.float32, init_value=init_hc_base),
        TensorSpec("x_mixed", [T, D], torch.bfloat16, is_output=True),
        TensorSpec("post", [T, HC_MULT], torch.float32, is_output=True),
        TensorSpec("comb", [T, HC_MULT * HC_MULT], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["decode", "prefill", "all"], default="all",
                        help="Use decode or prefill batch sizes, or 'all' to test both.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- hc_pre 1spmd {mode_name}: B={B}, S={S} ---")
        result = run_jit(
            fn=hc_pre_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_hc_pre,
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_mixed": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "post":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
                "comb":    ratio_allclose(atol=2.5e-5, rtol=5e-3),
            },
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
