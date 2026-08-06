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

The shared expert reuses the per-token INT8 quant already produced by
``gate`` (``x_norm_i8`` + ``x_norm_scale``) — the same INT8 view
that ``dispatch`` packs for the routed path. Gate/up remain INT8 while the
SwiGLU output and down projection use dynamic MXFP8.
"""


import pypto.language as pl

from config import (PRO_KERNEL as M, MOE_TOKENS, INT8_SCALE_MAX, INT8_AMAX_EPS)


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

K_TILE = 512
MM_INTER_TILE = 256
ACT_INTER_TILE = 1024
D_OUT_TILE = 256
D_OUT_TILE_ACT = 512
# 14 (not 8): the w2-dequant task count is `D // (W2_ACT_INNER * D_OUT_TILE_ACT)`.
# For FLASH that was 4096 // 4096 == 1 task covering all of D. PRO's D = 7168 is
# NOT a multiple of 8 * 512 = 4096, so the same expression truncates to 1 task
# covering only 4096 columns and silently leaves 3072 columns of D un-dequantized.
# W2_ACT_INNER must divide D // D_OUT_TILE_ACT (= 14 for PRO); 14 keeps the single
# outer task FLASH had and moves the whole range into the inner pipeline.
W2_ACT_INNER = 14

# TQuant emits one flat [1, M*K/32] scale tile. K=64 makes that [1, 32],
# whose byte order is exactly one [16, 2] MX_A_ZZ block. Four consecutive
# blocks occupy one row of the packed ND backing tensor.
MX_BLOCK_K = 32
MX_K_TILE = 64
MX_K_SCALE_TILE = MX_K_TILE // MX_BLOCK_K
MX_K_TILES = MOE_INTER // MX_K_TILE
MX_SCALE_STORE_COLS = 128
MX_SCALE_TILES_PER_ROW = MX_SCALE_STORE_COLS // (SH_M_TILE * MX_K_SCALE_TILE)
MX_SCALE_STORE_ROWS = MX_K_TILES // MX_SCALE_TILES_PER_ROW
W2_SCALE_ROWS = (D // D_OUT_TILE) * MX_K_TILES * MX_K_SCALE_TILE

# Every tile that is used as `<dim> // <tile>` in a loop bound must divide its dim
# exactly, otherwise the loop silently covers only part of the tensor.
assert D % K_TILE == 0
assert MOE_INTER % MM_INTER_TILE == 0 and MOE_INTER % ACT_INTER_TILE == 0
assert D % D_OUT_TILE == 0 and D % D_OUT_TILE_ACT == 0
assert D % (W2_ACT_INNER * D_OUT_TILE_ACT) == 0, \
    "W2_ACT_INNER * D_OUT_TILE_ACT must divide D (otherwise the w2-dequant task count truncates)"
assert MOE_INTER % MX_K_TILE == 0
assert MX_K_TILE == 2 * MX_BLOCK_K
assert MX_SCALE_STORE_COLS % (SH_M_TILE * MX_K_SCALE_TILE) == 0
assert MX_K_TILES % MX_SCALE_TILES_PER_ROW == 0


@pl.jit.inline
def expert_shared(
    x_local_i8: pl.Tensor[[T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[W2_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Tensor[[T, D], pl.BF16],
):
    # One M-tile of SH_M_TILE rows per iteration (decode: 1 tile, T<=16 rows valid;
    # prefill: T_PAD/SH_M_TILE fully-valid tiles).
    for mt in pl.parallel(N_MTILES):
        ts0 = mt * SH_M_TILE

        gate_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)
        up_i32 = pl.create_tensor([SH_M_TILE, MOE_INTER], dtype=pl.INT32)

        # gate (w1) cube matmul -> INT32 GM accumulator.
        for nb_idx in pl.spmd(
            MOE_INTER // MM_INTER_TILE,
            name_hint="sh_gate_mm",
            allow_early_resolve=True,
        ):
            n0 = nb_idx * MM_INTER_TILE
            gate_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw1_k = shared_w1[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    gate_acc = pl.matmul(xs_k, sw1_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    gate_acc = pl.matmul_acc(gate_acc, xs_k, sw1_k, b_trans=True)
            gate_i32[:, n0 : n0 + MM_INTER_TILE] = gate_acc

        # up (w3) cube matmul -> INT32 GM accumulator.
        for nb_idx in pl.spmd(
            MOE_INTER // MM_INTER_TILE,
            name_hint="sh_up_mm",
            allow_early_resolve=True,
        ):
            n0 = nb_idx * MM_INTER_TILE
            up_acc = pl.create_tensor([SH_M_TILE, MM_INTER_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                xs_k = pl.slice(x_local_i8, [SH_M_TILE, K_TILE], [ts0, k0], valid_shape=[SH_VALID_M, K_TILE])
                sw3_k = shared_w3[n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                if k0 == 0:
                    up_acc = pl.matmul(xs_k, sw3_k, b_trans=True, out_dtype=pl.INT32)
                else:
                    up_acc = pl.matmul_acc(up_acc, xs_k, sw3_k, b_trans=True)
            up_i32[:, n0 : n0 + MM_INTER_TILE] = up_acc

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
            x_scale = pl.slice(
                x_local_scale_dq,
                [SH_ROW_PAD, 1],
                [ts0 + row0, 0],
                valid_shape=[SH_ROWS_PER_BLOCK, 1],
            )
            for part in pl.pipeline(0, MOE_INTER // ACT_INTER_TILE, stage=1):
                n0 = part * ACT_INTER_TILE
                gate_rows_i32 = pl.slice(
                    gate_i32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                up_rows_i32 = pl.slice(
                    up_i32,
                    [SH_ROW_PAD, ACT_INTER_TILE],
                    [row0, n0],
                    valid_shape=[SH_ROWS_PER_BLOCK, ACT_INTER_TILE],
                )
                w1_scale = pl.reshape(
                    shared_w1_scale[n0 : n0 + ACT_INTER_TILE],
                    [1, ACT_INTER_TILE],
                )
                w3_scale = pl.reshape(
                    shared_w3_scale[n0 : n0 + ACT_INTER_TILE],
                    [1, ACT_INTER_TILE],
                )
                gate_fp32 = pl.cast(
                    gate_rows_i32, target_type=pl.FP32, mode="none"
                )
                up_fp32 = pl.cast(
                    up_rows_i32, target_type=pl.FP32, mode="none"
                )
                gate_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(gate_fp32, x_scale), w1_scale
                )
                up_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(up_fp32, x_scale), w3_scale
                )
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(
                        pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT
                    )
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                gated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                h_tile_fp32[
                    row0 : row0 + SH_ROWS_PER_BLOCK,
                    n0 : n0 + ACT_INTER_TILE,
                ] = gated[0:SH_ROWS_PER_BLOCK, :]

        # Store quantized h tile-major so every down-projection K tile is a
        # standalone [16, 64] tensor window.
        h_tile_mx = pl.create_tensor(
            [MX_K_TILES * SH_M_TILE, MX_K_TILE], dtype=pl.FP8E4M3FN
        )
        h_scale_store = pl.create_tensor(
            [MX_SCALE_STORE_ROWS, MX_SCALE_STORE_COLS], dtype=pl.FP8E8M0
        )
        for kb_idx in pl.spmd(
            MX_K_TILES,
            name_hint="sh_h_mx_quant",
            allow_early_resolve=True,
        ):
            k0 = kb_idx * MX_K_TILE
            h_src = pl.load(
                h_tile_fp32,
                [0, k0],
                [SH_M_TILE, MX_K_TILE],
                target_memory=pl.Mem.Vec,
            )
            h_q, h_scale = pl.quant_mx(h_src)
            pl.store(h_q, [kb_idx * SH_M_TILE, 0], h_tile_mx)
            scale_row = kb_idx // MX_SCALE_TILES_PER_ROW
            scale_col = (
                kb_idx % MX_SCALE_TILES_PER_ROW
            ) * SH_M_TILE * MX_K_SCALE_TILE
            pl.store(h_scale, [scale_row, scale_col], h_scale_store)

        h_scale_mx = pl.tensor.view(
            h_scale_store,
            [MX_K_TILES * SH_M_TILE, MX_K_SCALE_TILE],
            layout=pl.MX_A_ZZ,
        )

        # Accumulate all 48 tile-major K windows in the down projection.
        y_fp32 = pl.create_tensor([SH_M_TILE, D], dtype=pl.FP32)
        for db_idx in pl.spmd(
            D // D_OUT_TILE,
            name_hint="sh_w2_mm",
            allow_early_resolve=True,
        ):
            d0 = db_idx * D_OUT_TILE
            k0 = 0
            hs_k = pl.move(
                pl.load(
                    h_tile_mx,
                    [0, k0],
                    [SH_M_TILE, MX_K_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Left,
            )
            hs_scale_k = pl.move(
                pl.load(
                    h_scale_mx,
                    [0, 0],
                    [SH_M_TILE, MX_K_SCALE_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.LeftScale,
            )
            sw2_k = pl.move(
                pl.load(
                    shared_w2,
                    [k0, d0],
                    [MX_K_TILE, D_OUT_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.Right,
            )
            sw2_scale_k = pl.move(
                pl.load(
                    shared_w2_scale,
                    [db_idx * MX_K_TILES * MX_K_SCALE_TILE, 0],
                    [MX_K_SCALE_TILE, D_OUT_TILE],
                    target_memory=pl.Mem.Mat,
                ),
                target_memory=pl.Mem.RightScale,
            )
            y_acc = pl.matmul_mx(
                hs_k, hs_scale_k, sw2_k, sw2_scale_k
            )
            for kb in pl.unroll(MX_K_TILES - 1):
                kb_idx = kb + 1
                k0 = kb_idx * MX_K_TILE
                h_row = kb_idx * SH_M_TILE
                hs_k = pl.move(
                    pl.load(
                        h_tile_mx,
                        [h_row, 0],
                        [SH_M_TILE, MX_K_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Left,
                )
                hs_scale_k = pl.move(
                    pl.load(
                        h_scale_mx,
                        [h_row, 0],
                        [SH_M_TILE, MX_K_SCALE_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.LeftScale,
                )
                sw2_k = pl.move(
                    pl.load(
                        shared_w2,
                        [k0, d0],
                        [MX_K_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.Right,
                )
                sw2_scale_k = pl.move(
                    pl.load(
                        shared_w2_scale,
                        [
                            (db_idx * MX_K_TILES + kb_idx)
                            * MX_K_SCALE_TILE,
                            0,
                        ],
                        [MX_K_SCALE_TILE, D_OUT_TILE],
                        target_memory=pl.Mem.Mat,
                    ),
                    target_memory=pl.Mem.RightScale,
                )
                y_acc = pl.matmul_mx_acc(
                    y_acc, hs_k, hs_scale_k, sw2_k, sw2_scale_k
                )
            pl.store(y_acc, [0, d0], y_fp32)

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
    x_local_i8: pl.Tensor[[T, D], pl.INT8],
    x_local_scale_dq: pl.Tensor[[T, 1], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    shared_w2_scale: pl.Tensor[[W2_SCALE_ROWS, D_OUT_TILE], pl.FP8E8M0, pl.MX_B_NN],
    sh: pl.Out[pl.Tensor[[T, D], pl.BF16]],
):
    expert_shared(
        x_local_i8, x_local_scale_dq,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        sh,
    )
    return sh


def _int8_quant_per_row(x):
    """Per-row (per-token) INT8 symmetric quant matching v3.2 scope2 Stage 2.6."""
    import torch
    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


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


def _pack_b_scale_tiled(scale):
    """Pack logical B scales independently in the tiles consumed by the kernel."""
    parts = []
    for nb in range(D // D_OUT_TILE):
        for kb in range(MX_K_TILES):
            block = scale[
                kb * MX_K_SCALE_TILE : (kb + 1) * MX_K_SCALE_TILE,
                nb * D_OUT_TILE : (nb + 1) * D_OUT_TILE,
            ]
            parts.append(_pack_b_scale_block(block))
    import torch

    return torch.cat(parts, dim=0)


def _unpack_b_scale_tiled(scale):
    """Restore the kernel's tile-major MX_B_NN scale tensor to logical order."""
    import torch

    logical = torch.empty(
        (MOE_INTER // MX_BLOCK_K, D),
        dtype=torch.uint8,
        device=scale.device,
    )
    scale_u8 = scale.contiguous().view(torch.uint8)
    index = 0
    for nb in range(D // D_OUT_TILE):
        for kb in range(MX_K_TILES):
            block = scale_u8[
                index * MX_K_SCALE_TILE : (index + 1) * MX_K_SCALE_TILE
            ].view(torch.float8_e8m0fnu)
            block = _unpack_b_scale_block(block).view(torch.uint8)
            logical[
                kb * MX_K_SCALE_TILE : (kb + 1) * MX_K_SCALE_TILE,
                nb * D_OUT_TILE : (nb + 1) * D_OUT_TILE,
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


def _gen_mxfp8_weight_kn(shape, dequant_std, chan_cv):
    """Generate a tiled MXFP8 right weight and MX_B_NN scale fixture."""
    import torch

    k, n = shape
    weight = torch.randn(k, n) * torch.exp(chan_cv * torch.randn(1, n))
    weight_q, scale = _quantize_mxfp8_weight_kn(weight)
    weight_dq = _dequant_mxfp8_b(weight_q, scale)
    weight = weight_dq * (dequant_std / weight_dq.std().clamp_min(1e-12))
    weight_q, scale = _quantize_mxfp8_weight_kn(weight)
    return weight_q, _pack_b_scale_tiled(scale)


def golden_expert_shared(tensors):
    """Torch reference for the shared expert.

    Input is the per-token INT8 quant produced by gate (shared with
    dispatch / routed expert); we dequant inside to match the kernel's
    dequant-then-matmul pattern."""
    import torch
    import torch.nn.functional as F

    # Gate/up keep the mainline exact INT8 x INT8 -> INT32 path.
    x_local_i8 = tensors["x_local_i8"]                       # [T, D] int8
    x_local_scale_dq = tensors["x_local_scale_dq"].float()   # [T, 1]
    w1_i8 = tensors["shared_w1"]                        # [MOE_INTER, D] int8
    w1_scale = tensors["shared_w1_scale"].float()       # [MOE_INTER]
    w3_i8 = tensors["shared_w3"]                        # [MOE_INTER, D] int8
    w3_scale = tensors["shared_w3_scale"].float()       # [MOE_INTER]
    w2_fp8 = tensors["shared_w2"]                       # [MOE_INTER, D] fp8
    w2_scale = _unpack_b_scale_tiled(tensors["shared_w2_scale"])

    gate_int = x_local_i8.to(torch.int32) @ w1_i8.to(torch.int32).T
    sh_gate = gate_int.to(torch.float32) * x_local_scale_dq * w1_scale.unsqueeze(0)
    up_int = x_local_i8.to(torch.int32) @ w3_i8.to(torch.int32).T
    sh_up = up_int.to(torch.float32) * x_local_scale_dq * w3_scale.unsqueeze(0)
    if SWIGLU_LIMIT > 0:
        sh_gate = sh_gate.clamp(max=SWIGLU_LIMIT)
        sh_up = sh_up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    sh_h = F.silu(sh_gate) * sh_up

    # Match the complete tile-major matmul_mx/matmul_mx_acc device path.
    sh = None
    for k0 in range(0, MOE_INTER, MX_K_TILE):
        h_q, h_scale = _dynamic_mx_quant_e4m3(
            sh_h[:, k0 : k0 + MX_K_TILE]
        )
        partial = _mx_matmul_fp8(
            h_q,
            h_scale,
            w2_fp8[k0 : k0 + MX_K_TILE, :],
            w2_scale[
                k0 // MX_BLOCK_K : (k0 + MX_K_TILE) // MX_BLOCK_K,
                :,
            ],
        )
        sh = partial if sh is None else sh + partial

    tensors["sh"][:] = sh.to(torch.bfloat16)


def gen_shared_weight(shape, dequant_std, chan_cv):
    """Synthesize a shared-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP8 shared-expert quant grid (e4m3, 128x128-block
    E8M0 scale), then re-quantizing per-output-channel. Unlike routed (MXFP4 -> ~37 discrete
    levels), shared stays near-Gaussian (~200 levels). The coarse 128-block scale does NOT
    flatten the real per-output-channel magnitude spread, so ``chan_cv`` (log-space source-gain
    std) injects it to reproduce the real INT8 scale CV (~0.5 gate/up, ~0.35 down). Per-output-
    channel INT8 is scale-invariant, so the grid sets the level shape and ``dequant_std`` only
    sets the absolute scale magnitude. (routed experts use a different grid -- see
    expert_routed.gen_routed_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([out, in] -> scale [out]).
    """
    import torch

    FP8_MAX, TINY = 448.0, 1e-20

    def sim_fp8(W, block=128):   # e4m3 + 128x128-block E8M0 (round-up) scale on (out, in)
        out, inn = W.shape
        Wb = W.reshape(out // block, block, inn // block, block)
        scale = torch.exp2(torch.ceil(torch.log2((Wb.abs().amax(dim=(1, 3), keepdim=True) / FP8_MAX).clamp_min(TINY))))
        q = (Wb / scale).to(torch.float8_e4m3fn).float() * scale
        return q.reshape(out, inn)

    W = torch.randn(*shape) * torch.exp(chan_cv * torch.randn(*shape[:-1], 1))  # per-channel gain
    Wq = sim_fp8(W)
    amax = Wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale = amax / INT8_SCALE_MAX
    w_i8 = torch.round(Wq / scale).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8, scale


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    # Pre-quantize x_local once so the i8 / scale specs see consistent values
    # (mirrors what gate produces in the full pipeline).
    x_local_bf16 = torch.randn(T, D, dtype=torch.bfloat16)
    x_local_i8_pre, x_local_sd_pre = _int8_quant_per_row(x_local_bf16)

    # Synthesize (int8, per-channel scale) by simulating the real MXFP8 shared-expert
    # quant grid (gen_shared_weight). chan_cv reproduces the real per-output-channel scale
    # CV (~0.5 gate/up, ~0.35 down) the coarse FP8 block scale leaves behind.
    SHARED_DEQUANT_STD = {"w1": 1.71e-2, "w2": 1.68e-2, "w3": 1.70e-2}
    sw1_i8, sw1_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w1"], chan_cv=0.50)
    sw3_i8, sw3_s = gen_shared_weight((MOE_INTER, D), SHARED_DEQUANT_STD["w3"], chan_cv=0.50)
    sw2_fp8, sw2_s = _gen_mxfp8_weight_kn(
        (MOE_INTER, D),
        SHARED_DEQUANT_STD["w2"],
        chan_cv=0.33,
    )

    return [
        TensorSpec("x_local_i8", [T, D], torch.int8, init_value=lambda: x_local_i8_pre),
        TensorSpec("x_local_scale_dq", [T, 1], torch.float32, init_value=lambda: x_local_sd_pre.float()),
        TensorSpec("shared_w1", [MOE_INTER, D], torch.int8, init_value=lambda: sw1_i8),
        TensorSpec("shared_w1_scale", [MOE_INTER], torch.float32, init_value=lambda: sw1_s),
        TensorSpec("shared_w3", [MOE_INTER, D], torch.int8, init_value=lambda: sw3_i8),
        TensorSpec("shared_w3_scale", [MOE_INTER], torch.float32, init_value=lambda: sw3_s),
        TensorSpec("shared_w2", [MOE_INTER, D], torch.float8_e4m3fn, init_value=lambda: sw2_fp8),
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
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
            # BF16 sh, ~1 ULP. Gen weights reproduce real(L21): 0.01% vs 0.004% of points > 1e-3.
            "sh": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
