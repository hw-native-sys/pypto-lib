# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE shared expert compute (decode, EP single-card).

Split out of ``expert_routed.py``: only the shared-expert FFN path lives here.
The routed local experts are computed by ``expert_routed.py``; both kernels
are composed inside ``moe.py``.

AscendC's Hybrid MXFP8-MXFP4 path keeps every shared-expert linear in
W8A8 MXFP8: the BF16 input is dynamically quantized for gate/up, and the
SwiGLU output is dynamically re-quantized for down. MXFP4 is used only by
the routed-expert weights, not by this shared-expert path.
"""


import pypto.language as pl

from config import ACTIVE as M, MOE_TOKENS


# model config
T = MOE_TOKENS
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# tiling
SH_M_TILE = 16
SH_ROW_PAD = 8
SH_ROWS_PER_BLOCK = 2
T_PAD = ((T + SH_M_TILE - 1) // SH_M_TILE) * SH_M_TILE
# Decode (T <= SH_M_TILE, single partial block) or prefill (T a multiple of
# SH_M_TILE, fully valid blocks); a T that is neither would need a dynamic
# per-block row count the static valid_shape below can't express.
assert T <= SH_M_TILE or T % SH_M_TILE == 0, \
    "expert_shared needs T <= SH_M_TILE (decode) or T a multiple of SH_M_TILE (prefill)"
SH_VALID_M = T if T < SH_M_TILE else SH_M_TILE
N_MTILES = T_PAD // SH_M_TILE
assert SH_VALID_M % SH_ROWS_PER_BLOCK == 0

ACT_INTER_TILE = 1024
D_OUT_TILE = 256
D_OUT_TILE_ACT = 512

MX_BLOCK_K = 32
MX_K_TILE = 64
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
X_MX_K_TILES = D // MX_K_TILE
H_MX_K_TILES = MOE_INTER // MX_K_TILE
W13_SCALE_ROWS = (
    (MOE_INTER // D_OUT_TILE) * X_MX_K_TILES * MX_K_SCALE_TILE
)
W2_SCALE_ROWS = (D // D_OUT_TILE) * H_MX_K_TILES * MX_K_SCALE_TILE

# Every tile that is used as `<dim> // <tile>` in a loop bound must divide its dim
# exactly, otherwise the loop silently covers only part of the tensor.
assert MOE_INTER % ACT_INTER_TILE == 0
assert D % D_OUT_TILE == 0 and D % D_OUT_TILE_ACT == 0
assert D % MX_K_TILE == 0 and MOE_INTER % MX_K_TILE == 0
assert MX_K_TILE == 2 * MX_BLOCK_K


@pl.jit.inline
def expert_shared(
    x_local: pl.Tensor[[T, D], pl.BF16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[W13_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[W13_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[W2_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    # One M-tile of SH_M_TILE rows per iteration (decode: 1 tile, T<=16 rows valid;
    # prefill: T_PAD/SH_M_TILE fully-valid tiles).
    for mt in pl.parallel(N_MTILES):
        ts0 = mt * SH_M_TILE

        # AscendC's MxFp8LinearMethod dynamically quantizes the BF16 input and
        # uses it for both W8A8 MXFP8 gate/up projections.
        x_tile_mx = pl.create_tensor(
            [SH_M_TILE, D], dtype=pl.FP8E4M3FN
        )
        x_scale_store = pl.create_tensor(
            [X_MX_K_TILES, SH_M_TILE * MX_K_SCALE_TILE],
            dtype=pl.FP8E8M0,
        )
        for quant_kb_idx in pl.spmd(
            X_MX_K_TILES,
            name_hint="sh_x_mx_quant",
            allow_early_resolve=True,
        ):
            quant_k0 = quant_kb_idx * MX_K_TILE
            x_src = pl.load(
                x_local,
                [ts0, quant_k0],
                [SH_M_TILE, MX_K_TILE],
                valid_shape=[SH_VALID_M, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            x_src = pl.fillpad(x_src, pad_value=pl.PadValue.zero)
            x_q, x_scale = pl.quant_mx(x_src, layout=pl.MX_A_ZZ)
            pl.store(x_q, [0, quant_k0], x_tile_mx)
            x_scale_flat = pl.reshape(
                x_scale, [1, SH_M_TILE * MX_K_SCALE_TILE]
            )
            pl.store(x_scale_flat, [quant_kb_idx, 0], x_scale_store)

        gate_partial = pl.create_tensor(
            [X_MX_K_TILES * SH_M_TILE, MOE_INTER], dtype=pl.FP32
        )
        for task_idx in pl.parallel(
            X_MX_K_TILES * (MOE_INTER // D_OUT_TILE)
        ):
            mm_kb_idx = task_idx // (MOE_INTER // D_OUT_TILE)
            nb_idx = task_idx % (MOE_INTER // D_OUT_TILE)
            mm_k0 = mm_kb_idx * MX_K_TILE
            n0 = nb_idx * D_OUT_TILE

            x_scale_store_k = pl.slice(
                x_scale_store,
                [1, SH_M_TILE * MX_K_SCALE_TILE],
                [mm_kb_idx, 0],
            )
            x_scale_mx_k = pl.tensor.view(
                x_scale_store_k,
                [SH_M_TILE, MX_K_SCALE_TILE],
                layout=pl.MX_A_ZZ,
            )
            w1_scale_store_k = pl.slice(
                shared_w1_scale,
                [MX_K_SCALE_TILE, D_OUT_TILE],
                [
                    (nb_idx * X_MX_K_TILES + mm_kb_idx)
                    * MX_K_SCALE_TILE,
                    0,
                ],
            )
            w1_scale_mx_k = pl.tensor.view(
                w1_scale_store_k,
                [MX_K_SCALE_TILE, D_OUT_TILE],
                layout=pl.MX_B_NN,
            )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_gate_mm"):
                x_k = pl.move(
                    pl.load(
                        x_tile_mx,
                        [0, mm_k0],
                        [SH_M_TILE, MX_K_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Left,
                )
                x_scale_k = pl.move(
                    pl.load(
                        x_scale_mx_k,
                        [0, 0],
                        [SH_M_TILE, MX_K_SCALE_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                w1_k = pl.move(
                    pl.load(
                        shared_w1,
                        [mm_k0, n0],
                        [MX_K_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Right,
                )
                w1_scale_k = pl.move(
                    pl.load(
                        w1_scale_mx_k,
                        [0, 0],
                        [MX_K_SCALE_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.RightScale,
                )
                gate_partial_acc = pl.matmul_mx(
                    x_k, x_scale_k, w1_k, w1_scale_k
                )
                pl.store(
                    gate_partial_acc,
                    [mm_kb_idx * SH_M_TILE, n0],
                    gate_partial,
                )

        up_partial = pl.create_tensor(
            [X_MX_K_TILES * SH_M_TILE, MOE_INTER], dtype=pl.FP32
        )
        for task_idx in pl.parallel(
            X_MX_K_TILES * (MOE_INTER // D_OUT_TILE)
        ):
            mm_kb_idx = task_idx // (MOE_INTER // D_OUT_TILE)
            nb_idx = task_idx % (MOE_INTER // D_OUT_TILE)
            mm_k0 = mm_kb_idx * MX_K_TILE
            n0 = nb_idx * D_OUT_TILE

            x_scale_store_k = pl.slice(
                x_scale_store,
                [1, SH_M_TILE * MX_K_SCALE_TILE],
                [mm_kb_idx, 0],
            )
            x_scale_mx_k = pl.tensor.view(
                x_scale_store_k,
                [SH_M_TILE, MX_K_SCALE_TILE],
                layout=pl.MX_A_ZZ,
            )
            w3_scale_store_k = pl.slice(
                shared_w3_scale,
                [MX_K_SCALE_TILE, D_OUT_TILE],
                [
                    (nb_idx * X_MX_K_TILES + mm_kb_idx)
                    * MX_K_SCALE_TILE,
                    0,
                ],
            )
            w3_scale_mx_k = pl.tensor.view(
                w3_scale_store_k,
                [MX_K_SCALE_TILE, D_OUT_TILE],
                layout=pl.MX_B_NN,
            )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="sh_up_mm"):
                x_k = pl.move(
                    pl.load(
                        x_tile_mx,
                        [0, mm_k0],
                        [SH_M_TILE, MX_K_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Left,
                )
                x_scale_k = pl.move(
                    pl.load(
                        x_scale_mx_k,
                        [0, 0],
                        [SH_M_TILE, MX_K_SCALE_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                w3_k = pl.move(
                    pl.load(
                        shared_w3,
                        [mm_k0, n0],
                        [MX_K_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Right,
                )
                w3_scale_k = pl.move(
                    pl.load(
                        w3_scale_mx_k,
                        [0, 0],
                        [MX_K_SCALE_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.RightScale,
                )
                up_partial_acc = pl.matmul_mx(
                    x_k, x_scale_k, w3_k, w3_scale_k
                )
                pl.store(
                    up_partial_acc,
                    [mm_kb_idx * SH_M_TILE, n0],
                    up_partial,
                )

        gate_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        up_fp32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.FP32)
        for nb_idx in pl.spmd(
            MOE_INTER // D_OUT_TILE,
            name_hint="sh_gate_up_reduce",
            allow_early_resolve=True,
        ):
            n0 = nb_idx * D_OUT_TILE
            gate_sum = pl.load(
                gate_partial,
                [0, n0],
                [SH_M_TILE, D_OUT_TILE],
                target_memory=pl.Mem.Vec,
            )
            up_sum = pl.load(
                up_partial,
                [0, n0],
                [SH_M_TILE, D_OUT_TILE],
                target_memory=pl.Mem.Vec,
            )
            for kb_idx, (gate_sum_iter, up_sum_iter) in pl.pipeline(
                1,
                X_MX_K_TILES,
                init_values=(gate_sum, up_sum),
                stage=2,
            ):
                gate_partial_k = pl.load(
                    gate_partial,
                    [kb_idx * SH_M_TILE, n0],
                    [SH_M_TILE, D_OUT_TILE],
                    target_memory=pl.Mem.Vec,
                )
                up_partial_k = pl.load(
                    up_partial,
                    [kb_idx * SH_M_TILE, n0],
                    [SH_M_TILE, D_OUT_TILE],
                    target_memory=pl.Mem.Vec,
                )
                gate_sum_next = pl.add(gate_sum_iter, gate_partial_k)
                up_sum_next = pl.add(up_sum_iter, up_partial_k)
                gate_sum, up_sum = pl.yield_(gate_sum_next, up_sum_next)
            pl.store(gate_sum, [0, n0], gate_fp32)
            pl.store(up_sum, [0, n0], up_fp32)

        # Each AIV block owns two rows across the full intermediate axis.
        h_tile_fp32 = pl.create_tensor(
            [SH_M_TILE, MOE_INTER], dtype=pl.FP32, init_value=0.0
        )
        for row_block in pl.spmd(
            SH_VALID_M // SH_ROWS_PER_BLOCK,
            name_hint="sh_gate_up_act_q",
            allow_early_resolve=True,
        ):
            row0 = row_block * SH_ROWS_PER_BLOCK
            for part in pl.pipeline(0, MOE_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_rows_fp32 = pl.slice(
                    gate_fp32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                up_rows_fp32 = pl.slice(
                    up_fp32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                gate_value = pl.add(gate_rows_fp32, 0.0)
                up_value = pl.add(up_rows_fp32, 0.0)
                if SWIGLU_LIMIT > 0.0:
                    gate_value = pl.minimum(gate_value, SWIGLU_LIMIT)
                    up_value = pl.maximum(
                        pl.minimum(up_value, SWIGLU_LIMIT),
                        -SWIGLU_LIMIT,
                    )
                sigmoid = pl.recip(
                    pl.add(pl.exp(pl.neg(gate_value)), 1.0)
                )
                gated = pl.mul(pl.mul(gate_value, sigmoid), up_value)
                h_tile_fp32[
                    row0 : row0 + SH_ROWS_PER_BLOCK,
                    n0 : n0 + ACT_INTER_TILE,
                ] = gated[0:SH_ROWS_PER_BLOCK, :]

        h_tile_mx = pl.create_tensor(
            [SH_M_TILE, MOE_INTER], dtype=pl.FP8E4M3FN
        )
        h_scale_store = pl.create_tensor(
            [H_MX_K_TILES, SH_M_TILE * MX_K_SCALE_TILE], dtype=pl.FP8E8M0
        )
        for quant_kb_idx in pl.spmd(
            H_MX_K_TILES,
            name_hint="sh_h_mx_quant",
            allow_early_resolve=True,
        ):
            quant_k0 = quant_kb_idx * MX_K_TILE
            h_src = pl.load(
                h_tile_fp32,
                [0, quant_k0],
                [SH_M_TILE, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            h_q, h_scale = pl.quant_mx(h_src, layout=pl.MX_A_ZZ)
            pl.store(h_q, [0, quant_k0], h_tile_mx)
            h_scale_flat = pl.reshape(
                h_scale, [1, SH_M_TILE * MX_K_SCALE_TILE]
            )
            pl.store(h_scale_flat, [quant_kb_idx, 0], h_scale_store)

        y_partial = pl.create_tensor(
            [H_MX_K_TILES * SH_M_TILE, D], dtype=pl.FP32
        )
        for task_idx in pl.parallel(H_MX_K_TILES * (D // D_OUT_TILE)):
            mm_kb_idx = task_idx // (D // D_OUT_TILE)
            db_idx = task_idx % (D // D_OUT_TILE)
            mm_k0 = mm_kb_idx * MX_K_TILE
            d0 = db_idx * D_OUT_TILE

            hs_scale_store_k = pl.slice(
                h_scale_store,
                [1, SH_M_TILE * MX_K_SCALE_TILE],
                [mm_kb_idx, 0],
            )
            hs_scale_mx_k = pl.tensor.view(
                hs_scale_store_k,
                [SH_M_TILE, MX_K_SCALE_TILE],
                layout=pl.MX_A_ZZ,
            )
            sw2_scale_store_k = pl.slice(
                shared_w2_scale,
                [MX_K_SCALE_TILE, D_OUT_TILE],
                [
                    (db_idx * H_MX_K_TILES + mm_kb_idx) * MX_K_SCALE_TILE,
                    0,
                ],
            )
            sw2_scale_mx_k = pl.tensor.view(
                sw2_scale_store_k,
                [MX_K_SCALE_TILE, D_OUT_TILE],
                layout=pl.MX_B_NN,
            )

            with pl.at(
                level=pl.Level.CORE_GROUP,
                name_hint="sh_w2_mm",
            ):
                hs_k = pl.move(
                    pl.load(
                        h_tile_mx,
                        [0, mm_k0],
                        [SH_M_TILE, MX_K_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Left,
                )
                hs_scale_k = pl.move(
                    pl.load(
                        hs_scale_mx_k,
                        [0, 0],
                        [SH_M_TILE, MX_K_SCALE_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                sw2_k = pl.move(
                    pl.load(
                        shared_w2,
                        [mm_k0, d0],
                        [MX_K_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Right,
                )
                sw2_scale_k = pl.move(
                    pl.load(
                        sw2_scale_mx_k,
                        [0, 0],
                        [MX_K_SCALE_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.RightScale,
                )
                y_partial_acc = pl.matmul_mx(
                    hs_k, hs_scale_k, sw2_k, sw2_scale_k
                )
                pl.store(
                    y_partial_acc,
                    [mm_kb_idx * SH_M_TILE, d0],
                    y_partial,
                )

        y_fp32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.FP32)
        for db_idx in pl.spmd(
            D // D_OUT_TILE,
            name_hint="sh_w2_reduce",
            allow_early_resolve=True,
        ):
            d0 = db_idx * D_OUT_TILE
            y_sum = pl.load(
                y_partial,
                [0, d0],
                [SH_M_TILE, D_OUT_TILE],
                target_memory=pl.Mem.Vec,
            )
            for kb_idx, (y_sum_iter,) in pl.pipeline(
                1,
                H_MX_K_TILES,
                init_values=(y_sum,),
                stage=2,
            ):
                y_partial_k = pl.load(
                    y_partial,
                    [kb_idx * SH_M_TILE, d0],
                    [SH_M_TILE, D_OUT_TILE],
                    target_memory=pl.Mem.Vec,
                )
                y_sum_next = pl.add(y_sum_iter, y_partial_k)
                y_sum = pl.yield_(y_sum_next)
            pl.store(y_sum, [0, d0], y_fp32)

        for db_idx in pl.spmd(
            D // D_OUT_TILE_ACT,
            name_hint="sh_w2_act",
        ):
            d0 = db_idx * D_OUT_TILE_ACT
            y_2d = y_fp32[:, d0 : d0 + D_OUT_TILE_ACT]
            y_bf16 = pl.cast(y_2d, target_type=pl.BF16, mode="rint")
            sh[
                ts0 : ts0 + SH_VALID_M,
                d0 : d0 + D_OUT_TILE_ACT,
            ] = y_bf16[0:SH_VALID_M, :]

    # The @pl.inline parser requires inline call expressions to have a return
    # value; sh is convenient because it's already pl.Out.
    return sh


@pl.jit
def expert_shared_test(
    x_local: pl.Tensor[[T, D], pl.BF16],
    shared_w1: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w1_scale: pl.Tensor[[W13_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    shared_w3: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    shared_w3_scale: pl.Tensor[[W13_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[W2_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )
    return sh


def _ocp_e8m0_and_inv_scale(amax):
    """Create the E8M0 bytes and reciprocal scales used by hardware TQuant."""
    import torch

    amax = amax.to(torch.float32).contiguous()
    bits = amax.view(torch.int32)
    biased_exp = ((bits >> 23) & 0xFF).to(torch.int32)
    mantissa = bits & 0x007FFFFF
    is_nan = (biased_exp == 0xFF) & (mantissa != 0)
    e8m0 = torch.where(
        biased_exp <= 8,
        torch.zeros_like(biased_exp),
        biased_exp - 8,
    )
    e8m0 = torch.where(is_nan, torch.full_like(e8m0, 0xFF), e8m0)
    scale_exp = torch.where(
        biased_exp <= 8,
        torch.full_like(biased_exp, 254),
        254 - (biased_exp - 8),
    )
    scale_exp = torch.where(
        is_nan,
        torch.full_like(scale_exp, 0xFF),
        scale_exp,
    ).clamp(0, 255)
    inv_scale = (scale_exp.to(torch.int32) << 23).view(torch.float32)
    inv_scale = torch.where(
        inv_scale == 0.0,
        torch.full_like(inv_scale, 2.0**-127),
        inv_scale,
    )
    inv_scale = torch.where(
        is_nan,
        torch.full_like(inv_scale, float("nan")),
        inv_scale,
    )
    return e8m0.to(torch.uint8), inv_scale


def _dynamic_mx_quant_e4m3(x):
    """Quantize the last dimension in block-32 groups like ``pl.quant_mx``."""
    import torch

    x_fp32 = x.to(torch.float32)
    shape = x_fp32.shape
    blocks = x_fp32.reshape(*shape[:-1], shape[-1] // MX_BLOCK_K, MX_BLOCK_K)
    scale_u8, inv_scale = _ocp_e8m0_and_inv_scale(blocks.abs().amax(dim=-1))
    quant = (blocks * inv_scale.unsqueeze(-1)).clamp(-448.0, 448.0)
    quant = quant.to(torch.float8_e4m3fn).reshape(shape)
    return quant, scale_u8.contiguous().view(torch.float8_e8m0fnu)


def _e8m0_to_float(scale):
    """Decode E8M0 payload bytes to positive FP32 powers of two."""
    import torch

    scale_u8 = scale.contiguous().view(torch.uint8)
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def _dequant_mxfp8_a(value, scale):
    import torch

    value_fp32 = value.to(torch.float32)
    blocks = value_fp32.reshape(
        *value_fp32.shape[:-1],
        value_fp32.shape[-1] // MX_BLOCK_K,
        MX_BLOCK_K,
    )
    return (blocks * _e8m0_to_float(scale).unsqueeze(-1)).reshape_as(value_fp32)


def _dequant_mxfp8_b(value, scale):
    import torch

    value_fp32 = value.to(torch.float32)
    blocks = value_fp32.reshape(
        value_fp32.shape[0] // MX_BLOCK_K,
        MX_BLOCK_K,
        value_fp32.shape[1],
    )
    return (blocks * _e8m0_to_float(scale).unsqueeze(1)).reshape_as(value_fp32)


def _mx_matmul_fp8(a_value, a_scale, b_value, b_scale):
    return _dequant_mxfp8_a(a_value, a_scale) @ _dequant_mxfp8_b(
        b_value, b_scale
    )


def _pack_b_scale_block(scale):
    """Pack one unpadded logical scale tile to the MX_B_NN byte order."""
    import torch

    k_scale, n = scale.shape
    scale_u8 = scale.contiguous().view(torch.uint8)
    packed = scale_u8.reshape(k_scale // 2, 2, n // 16, 16)
    packed = packed.permute(2, 0, 3, 1).contiguous().reshape(k_scale, n)
    return packed.view(torch.float8_e8m0fnu)


def _unpack_b_scale_block(scale):
    """Restore one unpadded MX_B_NN scale tile to logical byte order."""
    import torch

    k_scale, n = scale.shape
    scale_u8 = scale.contiguous().view(torch.uint8)
    logical = scale_u8.reshape(n // 16, k_scale // 2, 16, 2)
    logical = logical.permute(1, 3, 0, 2).contiguous().reshape(k_scale, n)
    return logical.view(torch.float8_e8m0fnu)


def _pack_b_scale_tiled(scale, k_tile=MX_K_TILE, n_tile=D_OUT_TILE):
    """Pack logical B scales independently in the tiles consumed by the kernel."""
    k_scale, n = scale.shape
    k_scale_tile = k_tile // MX_BLOCK_K
    assert k_scale % k_scale_tile == 0
    assert n % n_tile == 0
    k_tiles = k_scale // k_scale_tile
    parts = []
    for nb in range(n // n_tile):
        for kb in range(k_tiles):
            block = scale[
                kb * k_scale_tile : (kb + 1) * k_scale_tile,
                nb * n_tile : (nb + 1) * n_tile,
            ]
            parts.append(_pack_b_scale_block(block))
    import torch

    return torch.cat(parts, dim=0)


def _unpack_b_scale_tiled(scale, k, n, k_tile=MX_K_TILE, n_tile=D_OUT_TILE):
    """Restore the kernel's tile-major MX_B_NN scale tensor to logical order."""
    import torch

    k_scale_tile = k_tile // MX_BLOCK_K
    assert k % k_tile == 0
    assert n % n_tile == 0
    logical = torch.empty(
        (k // MX_BLOCK_K, n),
        dtype=torch.uint8,
        device=scale.device,
    )
    scale_u8 = scale.contiguous().view(torch.uint8)
    index = 0
    for nb in range(n // n_tile):
        for kb in range(k // k_tile):
            block = scale_u8[
                index * k_scale_tile : (index + 1) * k_scale_tile
            ].view(torch.float8_e8m0fnu)
            block = _unpack_b_scale_block(block).view(torch.uint8)
            logical[
                kb * k_scale_tile : (kb + 1) * k_scale_tile,
                nb * n_tile : (nb + 1) * n_tile,
            ] = block
            index += 1
    return logical.contiguous().view(torch.float8_e8m0fnu)


def _quantize_mxfp8_weight_kn(weight):
    """Quantize a logical right matrix along its reduction dimension."""
    import torch

    weight_fp32 = weight.to(torch.float32)
    k, n = weight_fp32.shape
    blocks = weight_fp32.reshape(k // MX_BLOCK_K, MX_BLOCK_K, n)
    scale_u8, inv_scale = _ocp_e8m0_and_inv_scale(blocks.abs().amax(dim=1))
    quant = (blocks * inv_scale.unsqueeze(1)).clamp(-448.0, 448.0)
    quant = quant.to(torch.float8_e4m3fn).reshape(k, n)
    return quant, scale_u8.contiguous().view(torch.float8_e8m0fnu)


def _gen_mxfp8_weight_kn(
    shape, dequant_std, chan_cv, k_tile=MX_K_TILE, n_tile=D_OUT_TILE
):
    """Generate a tiled MXFP8 right weight and MX_B_NN scale fixture."""
    import torch

    k, n = shape
    weight = torch.randn(k, n) * torch.exp(chan_cv * torch.randn(1, n))
    weight_q, scale = _quantize_mxfp8_weight_kn(weight)
    weight_dq = _dequant_mxfp8_b(weight_q, scale)
    weight = weight_dq * (dequant_std / weight_dq.std().clamp_min(1e-12))
    weight_q, scale = _quantize_mxfp8_weight_kn(weight)
    return weight_q, _pack_b_scale_tiled(scale, k_tile=k_tile, n_tile=n_tile)


def _dynamic_mxfp8_matmul(a, b_value, b_scale):
    """Match the kernel's K=64 dynamic-MX quant and FP32 partial reduction."""
    result = None
    for k0 in range(0, a.shape[-1], MX_K_TILE):
        a_q, a_scale = _dynamic_mx_quant_e4m3(
            a[:, k0 : k0 + MX_K_TILE]
        )
        partial = _mx_matmul_fp8(
            a_q,
            a_scale,
            b_value[k0 : k0 + MX_K_TILE, :],
            b_scale[
                k0 // MX_BLOCK_K : (k0 + MX_K_TILE) // MX_BLOCK_K,
                :,
            ],
        )
        result = partial if result is None else result + partial
    return result


def golden_expert_shared(tensors):
    """Torch reference for AscendC's W8A8 MXFP8 shared expert."""
    import torch
    import torch.nn.functional as F

    x_local = tensors["x_local"]
    w1_fp8 = tensors["shared_w1"]
    w1_scale = _unpack_b_scale_tiled(
        tensors["shared_w1_scale"], D, MOE_INTER
    )
    w3_fp8 = tensors["shared_w3"]
    w3_scale = _unpack_b_scale_tiled(
        tensors["shared_w3_scale"], D, MOE_INTER
    )
    w2_fp8 = tensors["shared_w2"]
    w2_scale = _unpack_b_scale_tiled(
        tensors["shared_w2_scale"], MOE_INTER, D
    )

    # AscendC dynamically quantizes the same BF16 input independently for the
    # two linears. MX groups are deterministic, so one logical reference is
    # sufficient for both projections.
    sh_gate = _dynamic_mxfp8_matmul(x_local, w1_fp8, w1_scale)
    sh_up = _dynamic_mxfp8_matmul(x_local, w3_fp8, w3_scale)
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up

    sh = _dynamic_mxfp8_matmul(sh_h, w2_fp8, w2_scale)

    tensors["sh"][:] = sh.to(torch.bfloat16)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    x_local_bf16 = torch.randn(T, D, dtype=torch.bfloat16)

    # AscendC's Hybrid MXFP8-MXFP4 conversion keeps all shared weights in
    # MXFP8; MXFP4 is reserved for the routed experts.
    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_fp8, sw1_s = _gen_mxfp8_weight_kn(
        (D, MOE_INTER), SHARED_DEQUANT_STD["w1"], chan_cv=0.50
    )
    sw3_fp8, sw3_s = _gen_mxfp8_weight_kn(
        (D, MOE_INTER), SHARED_DEQUANT_STD["w3"], chan_cv=0.50
    )
    sw2_fp8, sw2_s = _gen_mxfp8_weight_kn(
        (MOE_INTER, D),
        SHARED_DEQUANT_STD["w2"],
        chan_cv=0.33,
    )

    return [
        TensorSpec(
            "x_local", [T, D], torch.bfloat16,
            init_value=lambda: x_local_bf16,
        ),
        TensorSpec(
            "shared_w1", [D, MOE_INTER], torch.float8_e4m3fn,
            init_value=lambda: sw1_fp8,
        ),
        TensorSpec(
            "shared_w1_scale",
            [W13_SCALE_ROWS, D_OUT_TILE],
            torch.float8_e8m0fnu,
            init_value=lambda: sw1_s,
        ),
        TensorSpec(
            "shared_w3", [D, MOE_INTER], torch.float8_e4m3fn,
            init_value=lambda: sw3_fp8,
        ),
        TensorSpec(
            "shared_w3_scale",
            [W13_SCALE_ROWS, D_OUT_TILE],
            torch.float8_e8m0fnu,
            init_value=lambda: sw3_s,
        ),
        TensorSpec(
            "shared_w2", [MOE_INTER, D], torch.float8_e4m3fn,
            init_value=lambda: sw2_fp8,
        ),
        TensorSpec(
            "shared_w2_scale",
            [W2_SCALE_ROWS, D_OUT_TILE],
            torch.float8_e8m0fnu,
            init_value=lambda: sw2_s,
        ),
        TensorSpec("sh", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=expert_shared_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_shared,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 output; allow a small tail from the three MXFP8 matmuls.
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
