# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Hyper-Connections pre-mix (dynamic shape): mixes the hc-stack into
a single sublayer input and produces the post/comb weights used by hc_post."""


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
T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_TILE = 128
LINEAR_K_TILE = 128
D_TILE = 512
assert (DECODE_BATCH * DECODE_SEQ) % T_TILE == 0
assert (PREFILL_BATCH * PREFILL_SEQ) % T_TILE == 0


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
    # mixes GM intermediate is required as the bridge into split_pre_post:
    # fusing split_pre_post here would need sub-MIX_HC-wide vec slicing of the
    # row_expand_mul result, but pto.tpop_from_aic drops valid_shape across
    # the cube->vec bridge (pypto#1507), breaking the downstream subview.
    mixes = pl.create_tensor([T_MAX, MIX_PAD], dtype=pl.FP32)
    for ob in pl.spmd(T_MAX // LINEAR_T_TILE, name_hint="linear"):
        t0 = ob * LINEAR_T_TILE
        sq_sum = pl.full([1, LINEAR_T_TILE], dtype=pl.FP32, value=0.0)

        mix_acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, HC_DIM // LINEAR_K_TILE, stage=2):
            kl0 = kb * LINEAR_K_TILE
            x_lin = pl.cast(x_flat[t0:t0 + LINEAR_T_TILE, kl0:kl0 + LINEAR_K_TILE], target_type=pl.FP32)
            x_sq = pl.mul(x_lin, x_lin)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(x_sq), [1, LINEAR_T_TILE]))
            w_lin = pl.slice(hc_fn, [MIX_PAD, LINEAR_K_TILE], [0, kl0], valid_shape=[MIX_HC, LINEAR_K_TILE])
            if kb == 0:
                mix_acc = pl.matmul(x_lin, w_lin, b_trans=True, out_dtype=pl.FP32)
            else:
                mix_acc = pl.matmul_acc(mix_acc, x_lin, w_lin, b_trans=True)

        mean_sq = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms_val = pl.rsqrt(mean_sq, high_precision=True)
        inv_rms_col = pl.reshape(inv_rms_val, [LINEAR_T_TILE, 1])
        mixes[t0:t0 + LINEAR_T_TILE, 0:MIX_PAD] = pl.row_expand_mul(mix_acc, inv_rms_col)

    pre_val_store = pl.create_tensor([T_MAX, HC_PAD], dtype=pl.FP32)
    post_pad_store = pl.create_tensor([T_MAX, HC_PAD], dtype=pl.FP32)
    comb_logits = pl.create_tensor([T_MAX, HC_MULT * HC_MULT], dtype=pl.FP32)
    for ob in pl.spmd(t_dim // T_TILE, name_hint="split_pre_post"):
        t0 = ob * T_TILE
        pre_base = pl.reshape(hc_base[0:HC_PAD], [1, HC_PAD])
        pre_scaled = pl.mul(mixes[t0:t0 + T_TILE, 0:HC_PAD], scale0)
        pre_logits = pl.add(pre_scaled, pl.col_expand(pre_scaled, pre_base))
        pre_sig = pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0))
        pre_val = pl.add(pre_sig, HC_EPS)
        pre_val_store = pl.assemble(pre_val_store, pre_val, [t0, 0])

        post_base = pl.reshape(hc_base[HC_MULT:HC_MULT + HC_PAD], [1, HC_PAD])
        post_scaled = pl.mul(mixes[t0:t0 + T_TILE, HC_MULT:HC_MULT + HC_PAD], scale1)
        post_logits = pl.add(post_scaled, pl.col_expand(post_scaled, post_base))
        post_sig = pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0))
        post_pad = pl.mul(post_sig, 2.0)
        post_pad_store = pl.assemble(post_pad_store, post_pad, [t0, 0])

        comb_base = pl.reshape(hc_base[HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT], [1, HC_MULT * HC_MULT])
        comb_scaled = pl.mul(mixes[t0:t0 + T_TILE, HC_MULT * 2:HC_MULT * 2 + HC_MULT * HC_MULT], scale2)
        comb_logits_tile = pl.add(comb_scaled, pl.col_expand(comb_scaled, comb_base))
        comb_logits = pl.assemble(comb_logits, comb_logits_tile, [t0, 0])

    for ob in pl.spmd(t_dim // COMB_T_TILE, name_hint="write_post"):
        t0 = ob * COMB_T_TILE
        post_tile = pl.load(post_pad_store, [t0, 0], [COMB_T_TILE, HC_PAD],
                            valid_shapes=[COMB_T_TILE, HC_MULT],
                            target_memory=pl.MemorySpace.Vec)
        pl.store(post_tile, [t0, 0], post)

    for ob in pl.spmd(t_dim // COMB_T_TILE, name_hint="comb_sinkhorn"):
        t0 = ob * COMB_T_TILE
        row0 = pl.load(comb_logits, [t0, 0 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row1 = pl.load(comb_logits, [t0, 1 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row2 = pl.load(comb_logits, [t0, 2 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        row3 = pl.load(comb_logits, [t0, 3 * HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], target_memory=pl.MemorySpace.Vec)
        # Distinct name after fillpad: TileView changes (valid_shape → pad), so
        # the @pl.inline parser rejects a same-name rebind. The JIT specializer
        # alpha-renames automatically — see pypto issue #1603 for the parser
        # discipline difference.
        row0_p = pl.fillpad(row0, pad_value=pl.PadValue.min)
        row1_p = pl.fillpad(row1, pad_value=pl.PadValue.min)
        row2_p = pl.fillpad(row2, pad_value=pl.PadValue.min)
        row3_p = pl.fillpad(row3, pad_value=pl.PadValue.min)

        row_max_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row0_max = pl.row_max(row0_p, row_max_tmp)
        row1_max = pl.row_max(row1_p, row_max_tmp)
        row2_max = pl.row_max(row2_p, row_max_tmp)
        row3_max = pl.row_max(row3_p, row_max_tmp)
        row0_exp = pl.exp(pl.row_expand_sub(row0_p, row0_max))
        row1_exp = pl.exp(pl.row_expand_sub(row1_p, row1_max))
        row2_exp = pl.exp(pl.row_expand_sub(row2_p, row2_max))
        row3_exp = pl.exp(pl.row_expand_sub(row3_p, row3_max))
        row0_sum = pl.row_sum(row0_exp, row_sum_tmp)
        row1_sum = pl.row_sum(row1_exp, row_sum_tmp)
        row2_sum = pl.row_sum(row2_exp, row_sum_tmp)
        row3_sum = pl.row_sum(row3_exp, row_sum_tmp)
        row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
        row1_soft = pl.add(pl.row_expand_div(row1_exp, row1_sum), HC_EPS)
        row2_soft = pl.add(pl.row_expand_div(row2_exp, row2_sum), HC_EPS)
        row3_soft = pl.add(pl.row_expand_div(row3_exp, row3_sum), HC_EPS)

        row0_valid = pl.set_validshape(row0_soft, COMB_T_TILE, HC_MULT)
        row1_valid = pl.set_validshape(row1_soft, COMB_T_TILE, HC_MULT)
        row2_valid = pl.set_validshape(row2_soft, COMB_T_TILE, HC_MULT)
        row3_valid = pl.set_validshape(row3_soft, COMB_T_TILE, HC_MULT)
        row0_eff = pl.fillpad(row0_valid, pad_value=pl.PadValue.zero)
        row1_eff = pl.fillpad(row1_valid, pad_value=pl.PadValue.zero)
        row2_eff = pl.fillpad(row2_valid, pad_value=pl.PadValue.zero)
        row3_eff = pl.fillpad(row3_valid, pad_value=pl.PadValue.zero)

        row_sum_tmp_iter = pl.create_tile([COMB_T_TILE, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        col_sum = pl.add(pl.add(row0_eff, row1_eff), pl.add(row2_eff, row3_eff))
        col_sum = pl.add(col_sum, HC_EPS)
        row0_cur = pl.div(row0_eff, col_sum)
        row1_cur = pl.div(row1_eff, col_sum)
        row2_cur = pl.div(row2_eff, col_sum)
        row3_cur = pl.div(row3_eff, col_sum)

        for sk_it in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0_rowsum = pl.add(pl.row_sum(row0_cur, row_sum_tmp_iter), HC_EPS)
            row1_rowsum = pl.add(pl.row_sum(row1_cur, row_sum_tmp_iter), HC_EPS)
            row2_rowsum = pl.add(pl.row_sum(row2_cur, row_sum_tmp_iter), HC_EPS)
            row3_rowsum = pl.add(pl.row_sum(row3_cur, row_sum_tmp_iter), HC_EPS)
            row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
            row1_norm = pl.row_expand_div(row1_cur, row1_rowsum)
            row2_norm = pl.row_expand_div(row2_cur, row2_rowsum)
            row3_norm = pl.row_expand_div(row3_cur, row3_rowsum)
            col_sum = pl.add(pl.add(row0_norm, row1_norm), pl.add(row2_norm, row3_norm))
            col_sum = pl.add(col_sum, HC_EPS)
            row0_cur = pl.div(row0_norm, col_sum)
            row1_cur = pl.div(row1_norm, col_sum)
            row2_cur = pl.div(row2_norm, col_sum)
            row3_cur = pl.div(row3_norm, col_sum)

        # Narrow tile->GM write via pl.store (respects valid_shape). The
        # equivalent subscript-write `comb_flat[t0:t0+16, k*4:k*4+4] = row_k_cur`
        # is rejected today (static_shape [16,8] vs slot [16,4]) — pypto#1509.
        row0_out = pl.set_validshape(row0_cur, COMB_T_TILE, HC_MULT)
        row1_out = pl.set_validshape(row1_cur, COMB_T_TILE, HC_MULT)
        row2_out = pl.set_validshape(row2_cur, COMB_T_TILE, HC_MULT)
        row3_out = pl.set_validshape(row3_cur, COMB_T_TILE, HC_MULT)
        pl.store(row0_out, [t0, 0 * HC_MULT], comb)
        pl.store(row1_out, [t0, 1 * HC_MULT], comb)
        pl.store(row2_out, [t0, 2 * HC_MULT], comb)
        pl.store(row3_out, [t0, 3 * HC_MULT], comb)

    for ob in pl.spmd(t_dim // T_TILE, name_hint="mix_x"):
        t0 = ob * T_TILE
        pre_tile = pre_val_store[t0:t0 + T_TILE, 0:HC_PAD]
        pre_tile_t = pl.transpose(pre_tile, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        for db in pl.range(D // D_TILE):
            d0 = db * D_TILE
            x0 = pl.cast(x_flat[t0:t0 + T_TILE, 0 * D + d0:0 * D + d0 + D_TILE], target_type=pl.FP32)
            x1 = pl.cast(x_flat[t0:t0 + T_TILE, 1 * D + d0:1 * D + d0 + D_TILE], target_type=pl.FP32)
            x2 = pl.cast(x_flat[t0:t0 + T_TILE, 2 * D + d0:2 * D + d0 + D_TILE], target_type=pl.FP32)
            x3 = pl.cast(x_flat[t0:t0 + T_TILE, 3 * D + d0:3 * D + d0 + D_TILE], target_type=pl.FP32)
            y0 = pl.row_expand_mul(x0, pre0)
            y1 = pl.row_expand_mul(x1, pre1)
            y2 = pl.row_expand_mul(x2, pre2)
            y3 = pl.row_expand_mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            x_mixed[t0:t0 + T_TILE, d0:d0 + D_TILE] = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
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
    """Torch reference, direct port of model.py Block.hc_pre 674-682 + hc_split_sinkhorn."""
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

    # hc_split_sinkhorn (port of kernel.py 372-427)
    pre = torch.sigmoid(mixes[..., :HC_MULT] * hc_scale[0] + hc_base[:HC_MULT]) + HC_EPS
    post_t = 2 * torch.sigmoid(mixes[..., HC_MULT:HC_MULT * 2] * hc_scale[1]
                               + hc_base[HC_MULT:HC_MULT * 2])
    comb_t = (mixes[..., HC_MULT * 2:] * hc_scale[2] + hc_base[HC_MULT * 2:]
              ).view(t_dim, HC_MULT, HC_MULT)

    # First step: row-softmax then col-normalize, with eps after softmax
    comb_t = torch.softmax(comb_t, dim=-1) + HC_EPS
    comb_t = comb_t / (comb_t.sum(-2, keepdim=True) + HC_EPS)
    # Sinkhorn iterations
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
        print(f"--- hc_pre {mode_name}: B={B}, S={S} ---")
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
            # Precision reference: pypto hc_pre —
            # cann-recipes-infer/ops/pypto_python/example/test_hc_pre_pypto.py
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
