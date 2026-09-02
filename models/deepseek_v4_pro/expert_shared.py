# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE shared-expert FFN with MXFP8 W1/W3/W2."""


import pypto.language as pl

from config import ACTIVE as M, MOE_TOKENS


# model config
T = MOE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
SH_M_TILE = 16
SH_ROW_TILE = 2
T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE
SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE
assert T < SH_M_TILE or T % SH_M_TILE == 0
assert SH_VALID_M % SH_ROW_TILE == 0
MX_GROUP = 32
MX_K_TILE = 512
MX_RIGHT_K_TILE = 256
MX_MM_INTER_TILE = 256
MX_K_SCALE_GROUPS = MX_K_TILE // MX_GROUP
MX_RIGHT_K_SCALE_GROUPS = MX_RIGHT_K_TILE // MX_GROUP
MX_W2_RIGHT_K_TILE = 256
MX_W2_RIGHT_K_SCALE_GROUPS = MX_W2_RIGHT_K_TILE // MX_GROUP
assert MOE_INTER % MX_MM_INTER_TILE == 0
assert D % MX_K_TILE == 0
ACT_INTER_TILE = 1024
D_OUT_TILE = 256
QUANT_TILE = 2048 if M.name == "flash" else 1024
assert MX_K_TILE % MX_RIGHT_K_TILE == 0
assert MX_K_TILE % MX_W2_RIGHT_K_TILE == 0


@pl.jit.inline
def expert_shared(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    """Shared expert with device MXFP8 activation quantization before both projections."""
    x_local_scale_mx = pl.tensor.view(x_local_scale, [T_PAD, D // MX_GROUP], layout=pl.MX_A_ZZ)
    for mt in pl.parallel(T_PAD // SH_M_TILE):
        ts0 = mt * SH_M_TILE

        gate_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)

        for nb_idx in pl.spmd(MOE_INTER // MX_MM_INTER_TILE, name_hint="sh_gate_mx_mm"):
            n0 = nb_idx * MX_MM_INTER_TILE
            xs0 = pl.load(x_local, [ts0, 0], [SH_M_TILE, MX_RIGHT_K_TILE])
            xs_scale0 = pl.load(
                x_local_scale_mx,
                [ts0, 0],
                [SH_M_TILE, MX_RIGHT_K_SCALE_GROUPS],
            )
            w1_0 = pl.load(shared_w1, [0, n0], [MX_RIGHT_K_TILE, MX_MM_INTER_TILE])
            w1_scale0 = pl.load(
                shared_w1_scale,
                [0, n0],
                [MX_RIGHT_K_SCALE_GROUPS, MX_MM_INTER_TILE],
            )
            gate_acc = pl.matmul_mx(xs0, xs_scale0, w1_0, w1_scale0)
            for kb in pl.range(1, MX_K_TILE // MX_RIGHT_K_TILE):
                k_local = kb * MX_RIGHT_K_TILE
                xs_part = pl.load(x_local, [ts0, k_local], [SH_M_TILE, MX_RIGHT_K_TILE])
                xs_scale_part = pl.load(
                    x_local_scale_mx,
                    [ts0, k_local // MX_GROUP],
                    [SH_M_TILE, MX_RIGHT_K_SCALE_GROUPS],
                )
                w1_part = pl.load(
                    shared_w1,
                    [k_local, n0],
                    [MX_RIGHT_K_TILE, MX_MM_INTER_TILE],
                )
                w1_scale_part = pl.load(
                    shared_w1_scale,
                    [k_local // MX_GROUP, n0],
                    [MX_RIGHT_K_SCALE_GROUPS, MX_MM_INTER_TILE],
                )
                gate_acc = pl.matmul_mx_acc(
                    gate_acc,
                    xs_part,
                    xs_scale_part,
                    w1_part,
                    w1_scale_part,
                )
            for k0 in pl.pipeline(MX_K_TILE, D, MX_K_TILE, stage=2):
                for kb in pl.range(MX_K_TILE // MX_RIGHT_K_TILE):
                    k_local = kb * MX_RIGHT_K_TILE
                    xs_part = pl.load(
                        x_local,
                        [ts0, k0 + k_local],
                        [SH_M_TILE, MX_RIGHT_K_TILE],
                    )
                    xs_scale_part = pl.load(
                        x_local_scale_mx,
                        [ts0, (k0 + k_local) // MX_GROUP],
                        [SH_M_TILE, MX_RIGHT_K_SCALE_GROUPS],
                    )
                    w1_part = pl.load(
                        shared_w1,
                        [k0 + k_local, n0],
                        [MX_RIGHT_K_TILE, MX_MM_INTER_TILE],
                    )
                    w1_scale_part = pl.load(
                        shared_w1_scale,
                        [(k0 + k_local) // MX_GROUP, n0],
                        [MX_RIGHT_K_SCALE_GROUPS, MX_MM_INTER_TILE],
                    )
                    gate_acc = pl.matmul_mx_acc(
                        gate_acc,
                        xs_part,
                        xs_scale_part,
                        w1_part,
                        w1_scale_part,
                    )
            gate_fp32 = pl.store(gate_acc, [0, n0], gate_fp32)

        up_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)

        for nb_idx in pl.spmd(MOE_INTER // MX_MM_INTER_TILE, name_hint="sh_up_mx_mm"):
            n0 = nb_idx * MX_MM_INTER_TILE
            xs0 = pl.load(x_local, [ts0, 0], [SH_M_TILE, MX_RIGHT_K_TILE])
            xs_scale0 = pl.load(
                x_local_scale_mx,
                [ts0, 0],
                [SH_M_TILE, MX_RIGHT_K_SCALE_GROUPS],
            )
            w3_0 = pl.load(shared_w3, [0, n0], [MX_RIGHT_K_TILE, MX_MM_INTER_TILE])
            w3_scale0 = pl.load(
                shared_w3_scale,
                [0, n0],
                [MX_RIGHT_K_SCALE_GROUPS, MX_MM_INTER_TILE],
            )
            up_acc = pl.matmul_mx(xs0, xs_scale0, w3_0, w3_scale0)
            for kb in pl.range(1, MX_K_TILE // MX_RIGHT_K_TILE):
                k_local = kb * MX_RIGHT_K_TILE
                xs_part = pl.load(x_local, [ts0, k_local], [SH_M_TILE, MX_RIGHT_K_TILE])
                xs_scale_part = pl.load(
                    x_local_scale_mx,
                    [ts0, k_local // MX_GROUP],
                    [SH_M_TILE, MX_RIGHT_K_SCALE_GROUPS],
                )
                w3_part = pl.load(
                    shared_w3,
                    [k_local, n0],
                    [MX_RIGHT_K_TILE, MX_MM_INTER_TILE],
                )
                w3_scale_part = pl.load(
                    shared_w3_scale,
                    [k_local // MX_GROUP, n0],
                    [MX_RIGHT_K_SCALE_GROUPS, MX_MM_INTER_TILE],
                )
                up_acc = pl.matmul_mx_acc(
                    up_acc,
                    xs_part,
                    xs_scale_part,
                    w3_part,
                    w3_scale_part,
                )
            for k0 in pl.pipeline(MX_K_TILE, D, MX_K_TILE, stage=2):
                for kb in pl.range(MX_K_TILE // MX_RIGHT_K_TILE):
                    k_local = kb * MX_RIGHT_K_TILE
                    xs_part = pl.load(
                        x_local,
                        [ts0, k0 + k_local],
                        [SH_M_TILE, MX_RIGHT_K_TILE],
                    )
                    xs_scale_part = pl.load(
                        x_local_scale_mx,
                        [ts0, (k0 + k_local) // MX_GROUP],
                        [SH_M_TILE, MX_RIGHT_K_SCALE_GROUPS],
                    )
                    w3_part = pl.load(
                        shared_w3,
                        [k0 + k_local, n0],
                        [MX_RIGHT_K_TILE, MX_MM_INTER_TILE],
                    )
                    w3_scale_part = pl.load(
                        shared_w3_scale,
                        [(k0 + k_local) // MX_GROUP, n0],
                        [MX_RIGHT_K_SCALE_GROUPS, MX_MM_INTER_TILE],
                    )
                    up_acc = pl.matmul_mx_acc(
                        up_acc,
                        xs_part,
                        xs_scale_part,
                        w3_part,
                        w3_scale_part,
                    )
            up_fp32 = pl.store(up_acc, [0, n0], up_fp32)

        h_tile_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        h_tile_mx = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP8E4M3FN)
        h_scale_backing = pl.create_tensor(
            [1, SH_M_TILE * (MOE_INTER // MX_GROUP)], dtype=pl.FP8E8M0
        )
        for q_idx in pl.spmd(MOE_INTER // QUANT_TILE, name_hint="sh_gate_up_act"):
            k0 = q_idx * QUANT_TILE
            for row_block in pl.range(SH_M_TILE // SH_ROW_TILE):
                row0 = row_block * SH_ROW_TILE
                for n0 in pl.pipeline(k0, k0 + QUANT_TILE, ACT_INTER_TILE, stage=2):
                    gate_rows = gate_fp32[row0 : row0 + SH_ROW_TILE, n0 : n0 + ACT_INTER_TILE]
                    up_rows = up_fp32[row0 : row0 + SH_ROW_TILE, n0 : n0 + ACT_INTER_TILE]
                    if SWIGLU_LIMIT > 0.0:
                        gate_rows = pl.minimum(gate_rows, SWIGLU_LIMIT)
                        up_max = pl.minimum(up_rows, SWIGLU_LIMIT)
                        up_rows = pl.maximum(up_max, -SWIGLU_LIMIT)
                    gate_neg = pl.neg(gate_rows)
                    gate_exp = pl.exp(gate_neg)
                    gate_exp_one = pl.add(gate_exp, 1.0)
                    sigmoid = pl.recip(gate_exp_one)
                    silu = pl.mul(gate_rows, sigmoid)
                    gated = pl.mul(silu, up_rows)
                    h_tile_fp32[row0 : row0 + SH_ROW_TILE, n0 : n0 + ACT_INTER_TILE] = gated
            h_fp32 = pl.load(h_tile_fp32, [0, k0], [SH_M_TILE, QUANT_TILE])
            h_mx, h_scale_mx = pl.quant_mx(h_fp32, group_axis=1)
            h_tile_mx = pl.store(h_mx, [0, k0], h_tile_mx)
            scale_offset = q_idx * SH_M_TILE * (QUANT_TILE // MX_GROUP)
            h_scale_backing = pl.store(
                pl.reshape(h_scale_mx, [1, SH_M_TILE * (QUANT_TILE // MX_GROUP)]),
                [0, scale_offset],
                h_scale_backing,
            )

        h_tile_scale_mx = pl.tensor.view(
            h_scale_backing,
            [SH_M_TILE, MOE_INTER // MX_GROUP],
            layout=pl.MX_A_ZZ,
        )
        for db_idx in pl.spmd(D // D_OUT_TILE, name_hint="sh_w2_mm"):
            d0 = db_idx * D_OUT_TILE
            hs0 = pl.load(h_tile_mx, [0, 0], [SH_M_TILE, MX_W2_RIGHT_K_TILE])
            hs_scale0 = pl.load(
                h_tile_scale_mx,
                [0, 0],
                [SH_M_TILE, MX_W2_RIGHT_K_SCALE_GROUPS],
            )
            sw2_0 = pl.load(shared_w2, [0, d0], [MX_W2_RIGHT_K_TILE, D_OUT_TILE])
            sw2_scale0 = pl.load(
                shared_w2_scale,
                [0, d0],
                [MX_W2_RIGHT_K_SCALE_GROUPS, D_OUT_TILE],
            )
            y_acc = pl.matmul_mx(hs0, hs_scale0, sw2_0, sw2_scale0)
            for kb in pl.range(1, MX_K_TILE // MX_W2_RIGHT_K_TILE):
                k_local = kb * MX_W2_RIGHT_K_TILE
                hs_part = pl.load(h_tile_mx, [0, k_local], [SH_M_TILE, MX_W2_RIGHT_K_TILE])
                hs_scale_part = pl.load(
                    h_tile_scale_mx,
                    [0, k_local // MX_GROUP],
                    [SH_M_TILE, MX_W2_RIGHT_K_SCALE_GROUPS],
                )
                sw2_part = pl.load(
                    shared_w2,
                    [k_local, d0],
                    [MX_W2_RIGHT_K_TILE, D_OUT_TILE],
                )
                sw2_scale_part = pl.load(
                    shared_w2_scale,
                    [k_local // MX_GROUP, d0],
                    [MX_W2_RIGHT_K_SCALE_GROUPS, D_OUT_TILE],
                )
                y_acc = pl.matmul_mx_acc(
                    y_acc,
                    hs_part,
                    hs_scale_part,
                    sw2_part,
                    sw2_scale_part,
                )
            for k0 in pl.pipeline(MX_K_TILE, MOE_INTER, MX_K_TILE, stage=2):
                for kb in pl.range(MX_K_TILE // MX_W2_RIGHT_K_TILE):
                    k_local = kb * MX_W2_RIGHT_K_TILE
                    hs_part = pl.load(
                        h_tile_mx,
                        [0, k0 + k_local],
                        [SH_M_TILE, MX_W2_RIGHT_K_TILE],
                    )
                    hs_scale_part = pl.load(
                        h_tile_scale_mx,
                        [0, (k0 + k_local) // MX_GROUP],
                        [SH_M_TILE, MX_W2_RIGHT_K_SCALE_GROUPS],
                    )
                    sw2_part = pl.load(
                        shared_w2,
                        [k0 + k_local, d0],
                        [MX_W2_RIGHT_K_TILE, D_OUT_TILE],
                    )
                    sw2_scale_part = pl.load(
                        shared_w2_scale,
                        [(k0 + k_local) // MX_GROUP, d0],
                        [MX_W2_RIGHT_K_SCALE_GROUPS, D_OUT_TILE],
                    )
                    y_acc = pl.matmul_mx_acc(
                        y_acc,
                        hs_part,
                        hs_scale_part,
                        sw2_part,
                        sw2_scale_part,
                    )
            y_bf16 = pl.cast(y_acc, target_type=pl.BF16, mode="rint")
            y_valid = pl.set_validshape(y_bf16, SH_VALID_M, D_OUT_TILE)
            sh = pl.store(y_valid, [ts0, d0], sh)

    return sh


@pl.jit
def expert_shared_test(
    x_local: pl.Tensor[[T_PAD, D], pl.FP8E4M3FN],
    x_local_scale: pl.Tensor[[1, T_PAD * (D // MX_GROUP)], pl.FP8E8M0],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local, x_local_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )
    return sh


def golden_expert_shared(tensors):
    """Reference for MXFP8 W1/W3, SwiGLU requantization, and MXFP8 W2."""
    import torch
    import torch.nn.functional as F

    from mx_utils import decode_e8m0_codes, host_quant_mxfp8, matmul_mx_golden

    x_fp8 = tensors["x_local"][:T]
    x_scale = decode_e8m0_codes(
        tensors["x_local_scale"].reshape(T_PAD, D // MX_GROUP), side="a",
    )[:T]
    w1_fp8 = tensors["shared_w1"]
    w1_scale = decode_e8m0_codes(tensors["shared_w1_scale"], side="b")
    w3_fp8 = tensors["shared_w3"]
    w3_scale = decode_e8m0_codes(tensors["shared_w3_scale"], side="b")
    w2_fp8 = tensors["shared_w2"]
    w2_scale = decode_e8m0_codes(tensors["shared_w2_scale"], side="b")

    sh_gate = matmul_mx_golden(x_fp8, x_scale, w1_fp8, w1_scale)
    sh_up = matmul_mx_golden(x_fp8, x_scale, w3_fp8, w3_scale)
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up
    sh_h_fp8, sh_h_scale = host_quant_mxfp8(sh_h, return_e8m0=True)
    sh = matmul_mx_golden(sh_h_fp8, sh_h_scale, w2_fp8, w2_scale)

    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec
    from mx_utils import gen_mxfp8_weight_kn_device, host_mxfp8_activation

    x_local_bf16 = torch.randn(T_PAD, D, dtype=torch.bfloat16)
    if T < T_PAD:
        x_local_bf16[T:, :] = 0
    x_local_fp8, x_local_scale = host_mxfp8_activation(x_local_bf16)

    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_fp8, sw1_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w1"], chan_cv=0.50, seed=1)
    sw3_fp8, sw3_scale = gen_mxfp8_weight_kn_device(MOE_INTER, D, SHARED_DEQUANT_STD["w3"], chan_cv=0.50, seed=2)
    sw2_fp8, sw2_scale = gen_mxfp8_weight_kn_device(
        D, MOE_INTER, SHARED_DEQUANT_STD["w2"], chan_cv=0.33, seed=3
    )

    fp8 = torch.float8_e4m3fn
    fp8_e8m0 = torch.float8_e8m0fnu

    return [
        TensorSpec("x_local", [T_PAD, D], fp8, init_value=lambda: x_local_fp8),
        TensorSpec("x_local_scale", [1, T_PAD * (D // MX_GROUP)], fp8_e8m0, init_value=lambda: x_local_scale.reshape(1, -1)),
        TensorSpec("shared_w1", [D, MOE_INTER], fp8, init_value=lambda: sw1_fp8),
        TensorSpec("shared_w1_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw1_scale),
        TensorSpec("shared_w3", [D, MOE_INTER], fp8, init_value=lambda: sw3_fp8),
        TensorSpec("shared_w3_scale", [D // MX_GROUP, MOE_INTER], fp8_e8m0, init_value=lambda: sw3_scale),
        TensorSpec("shared_w2", [MOE_INTER, D], fp8, init_value=lambda: sw2_fp8),
        TensorSpec("shared_w2_scale", [MOE_INTER // MX_GROUP, D], fp8_e8m0, init_value=lambda: sw2_scale),
        TensorSpec("sh", [T, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    result = run(
        fn=expert_shared_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
        golden_data=args.golden_data,
        save_data=args.save_data,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
