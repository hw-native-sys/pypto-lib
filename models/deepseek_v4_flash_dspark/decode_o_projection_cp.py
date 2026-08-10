# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 decode DSA-CP receive-side grouped output projection."""

import pypto.language as pl

from config import FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX
from decode_attention_cp import GROUP_T_PAD, LOCAL_O_GROUPS, LOCAL_T, O_GROUP_IN, SP_SIZE


# model config
D = M.hidden_size
O_LORA = M.o_lora_rank
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA

# tiling
O_A_T_TILE = 16
O_A_K_TILE = 256
O_A_N_TILE = 128
QUANT_T_TILE = 8
O_B_T_TILE = 128
O_B_K_TILE = 256
O_B_N_TILE = 256
O_B_D_TILE = 512
ACT_T_TILE = 8
ACT_N_TILE = 512

if O_GROUP_IN % O_A_K_TILE != 0:
    raise ValueError(f"O-A input {O_GROUP_IN} must be divisible by K tile {O_A_K_TILE}")
if O_LORA % O_A_N_TILE != 0:
    raise ValueError(f"O-A output {O_LORA} must be divisible by N tile {O_A_N_TILE}")
if O_LORA % O_B_K_TILE != 0:
    raise ValueError(f"O-B group width {O_LORA} must be divisible by K tile {O_B_K_TILE}")
if D % O_B_D_TILE != 0 or O_B_D_TILE % O_B_N_TILE != 0:
    raise ValueError("O-B output tiles must divide the hidden dimension")
if D % ACT_N_TILE != 0:
    raise ValueError(f"O-B activation tile {ACT_N_TILE} must divide hidden size {D}")


@pl.jit.inline
def decode_o_projection_cp(
    attention_local_groups: pl.Tensor[[LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    local_t: pl.Scalar[pl.INT32],
    o_partial: pl.Tensor[[GROUP_T_PAD, D], pl.FP32],
) -> tuple[pl.Tensor, pl.Scalar[pl.TASK_ID]]:
    """Project the compact A2A receive prefix into one rank's FP32 O-B partial."""
    # A2A receive rows use [local_group, source_rank * local_t] offsets.
    group_t = SP_SIZE * local_t
    o_a_rows = (group_t + O_A_T_TILE - 1) // O_A_T_TILE
    o_b_rows = (group_t + O_B_T_TILE - 1) // O_B_T_TILE
    act_rows = (group_t + ACT_T_TILE - 1) // ACT_T_TILE

    attn_2d = pl.reshape(attention_local_groups, [LOCAL_O_GROUPS * GROUP_T_PAD, O_GROUP_IN])
    wo_a_flat = pl.reshape(wo_a, [LOCAL_O_WIDTH, O_GROUP_IN])
    o_a_fp32 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_WIDTH], dtype=pl.FP32)
    o_a_i8 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_WIDTH], dtype=pl.INT8)
    act_scale_dq = pl.create_tensor([LOCAL_O_GROUPS, GROUP_T_PAD], dtype=pl.FP32)
    o_b_i32 = pl.create_tensor([GROUP_T_PAD, LOCAL_O_GROUPS * D], dtype=pl.INT32)
    proj_b_tids = pl.array.create(LOCAL_O_GROUPS, pl.TASK_ID)

    for local_group in pl.parallel(LOCAL_O_GROUPS):
        attention_row = local_group * GROUP_T_PAD
        o_a_col = local_group * O_LORA
        with pl.spmd(o_a_rows * (O_LORA // O_A_N_TILE), name_hint="cp_o_a") as proj_a_tid:
            proj_a_unit = pl.tile.get_block_idx()
            row_block = proj_a_unit // (O_LORA // O_A_N_TILE)
            n_block = proj_a_unit - row_block * (O_LORA // O_A_N_TILE)
            t0 = row_block * O_A_T_TILE
            n0 = n_block * O_A_N_TILE
            a_rows = pl.min(O_A_T_TILE, group_t - t0)
            src_row = attention_row + t0
            weight_row = o_a_col + n0
            o_a_x0 = pl.slice(attn_2d, [O_A_T_TILE, O_A_K_TILE], [src_row, 0], valid_shape=[a_rows, O_A_K_TILE])
            o_a_w0 = wo_a_flat[weight_row : weight_row + O_A_N_TILE, 0:O_A_K_TILE]
            o_a_acc = pl.matmul(o_a_x0, o_a_w0, b_trans=True, out_dtype=pl.FP32)
            for k0 in pl.pipeline(O_A_K_TILE, O_GROUP_IN, O_A_K_TILE, stage=2):
                o_a_xk = pl.slice(attn_2d, [O_A_T_TILE, O_A_K_TILE], [src_row, k0], valid_shape=[a_rows, O_A_K_TILE])
                o_a_wk = wo_a_flat[weight_row : weight_row + O_A_N_TILE, k0 : k0 + O_A_K_TILE]
                o_a_acc = pl.matmul_acc(o_a_acc, o_a_xk, o_a_wk, b_trans=True)
            o_a_valid = pl.set_validshape(o_a_acc, a_rows, O_A_N_TILE)
            o_a_fp32[t0 : t0 + O_A_T_TILE, weight_row : weight_row + O_A_N_TILE] = o_a_valid

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_o_a_quant", deps=[proj_a_tid]) as quant_tid:
            for qt in pl.pipeline(0, group_t, QUANT_T_TILE, stage=2):
                quant_rows = pl.min(QUANT_T_TILE, group_t - qt)
                o_a_tile = pl.slice(o_a_fp32, [QUANT_T_TILE, O_LORA], [qt, o_a_col], valid_shape=[quant_rows, O_LORA])
                o_a_abs = pl.abs(o_a_tile)
                row_amax = pl.reshape(pl.row_max(o_a_abs), [1, QUANT_T_TILE])
                amax_floor = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
                row_amax = pl.maximum(amax_floor, row_amax)
                scale_max = pl.full([1, QUANT_T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
                scale_q_row = pl.div(scale_max, row_amax)
                scale_dq_row = pl.recip(scale_q_row)
                scale_dq_valid = pl.set_validshape(scale_dq_row, 1, quant_rows)
                act_scale_dq[local_group : local_group + 1, qt : qt + QUANT_T_TILE] = scale_dq_valid
                scale_q_col = pl.reshape(scale_q_row, [QUANT_T_TILE, 1])
                o_a_scaled = pl.row_expand_mul(o_a_tile, scale_q_col)
                o_a_i32 = pl.cast(o_a_scaled, target_type=pl.INT32, mode="rint")
                o_a_fp16 = pl.cast(o_a_i32, target_type=pl.FP16, mode="round")
                o_a_quant = pl.cast(o_a_fp16, target_type=pl.INT8, mode="trunc")
                o_a_quant_valid = pl.set_validshape(o_a_quant, quant_rows, O_LORA)
                o_a_i8[qt : qt + QUANT_T_TILE, o_a_col : o_a_col + O_LORA] = o_a_quant_valid

        with pl.spmd(o_b_rows * (D // O_B_D_TILE), name_hint="cp_o_b", deps=[quant_tid]) as proj_b_tid:
            proj_b_unit = pl.tile.get_block_idx()
            row_block = proj_b_unit // (D // O_B_D_TILE)
            d_block = proj_b_unit - row_block * (D // O_B_D_TILE)
            t0 = row_block * O_B_T_TILE
            d0 = d_block * O_B_D_TILE
            b_rows = pl.min(O_B_T_TILE, group_t - t0)
            for n0 in pl.range(d0, d0 + O_B_D_TILE, O_B_N_TILE):
                o_b_x0 = pl.slice(o_a_i8, [O_B_T_TILE, O_B_K_TILE], [t0, o_a_col], valid_shape=[b_rows, O_B_K_TILE])
                o_b_w0 = wo_b[n0 : n0 + O_B_N_TILE, o_a_col : o_a_col + O_B_K_TILE]
                o_b_acc = pl.matmul(o_b_x0, o_b_w0, b_trans=True, out_dtype=pl.INT32)
                for k0 in pl.pipeline(O_B_K_TILE, O_LORA, O_B_K_TILE, stage=2):
                    b_k0 = o_a_col + k0
                    o_b_xk = pl.slice(o_a_i8, [O_B_T_TILE, O_B_K_TILE], [t0, b_k0], valid_shape=[b_rows, O_B_K_TILE])
                    o_b_wk = wo_b[n0 : n0 + O_B_N_TILE, b_k0 : b_k0 + O_B_K_TILE]
                    o_b_acc = pl.matmul_acc(o_b_acc, o_b_xk, o_b_wk, b_trans=True)
                o_b_i32_valid = pl.set_validshape(o_b_acc, b_rows, O_B_N_TILE)
                partial_col = local_group * D + n0
                o_b_i32[t0 : t0 + O_B_T_TILE, partial_col : partial_col + O_B_N_TILE] = o_b_i32_valid
        proj_b_tids[local_group] = proj_b_tid

    with pl.spmd(
        act_rows * (D // ACT_N_TILE), name_hint="cp_o_b_dequant",
        deps=[proj_b_tids[group] for group in range(LOCAL_O_GROUPS)],
    ) as completion_tid:
        act_unit = pl.tile.get_block_idx()
        row_block = act_unit // (D // ACT_N_TILE)
        n_block = act_unit - row_block * (D // ACT_N_TILE)
        t0 = row_block * ACT_T_TILE
        n0 = n_block * ACT_N_TILE
        out_rows = pl.min(ACT_T_TILE, group_t - t0)
        acc = pl.full([ACT_T_TILE, ACT_N_TILE], dtype=pl.FP32, value=0.0)
        for local_group in pl.pipeline(LOCAL_O_GROUPS, stage=2):
            part_col = local_group * D + n0
            group_i32 = pl.slice(o_b_i32, [ACT_T_TILE, ACT_N_TILE], [t0, part_col], valid_shape=[out_rows, ACT_N_TILE])
            group_partial_fp32 = pl.cast(group_i32, target_type=pl.FP32, mode="none")
            scale_row = pl.slice(act_scale_dq, [1, ACT_T_TILE], [local_group, t0], valid_shape=[1, out_rows])
            scale_col = pl.reshape(scale_row, [ACT_T_TILE, 1])
            group_dequant = pl.row_expand_mul(group_partial_fp32, scale_col)
            acc = pl.add(acc, group_dequant)
        weight_scale = pl.reshape(wo_b_scale[n0 : n0 + ACT_N_TILE], [1, ACT_N_TILE])
        o_b_fp32 = pl.col_expand_mul(acc, weight_scale)
        o_partial_valid = pl.set_validshape(o_b_fp32, out_rows, ACT_N_TILE)
        o_partial[t0 : t0 + ACT_T_TILE, n0 : n0 + ACT_N_TILE] = o_partial_valid
    return o_partial, completion_tid


@pl.jit
def decode_o_projection_cp_test(
    attention_local_groups: pl.Tensor[[LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    local_t: pl.Scalar[pl.INT32],
    o_partial: pl.InOut[pl.Tensor[[GROUP_T_PAD, D], pl.FP32]],
):
    """Run one receive-side output projection at a runtime token count."""
    o_partial, completion_tid = decode_o_projection_cp(
        attention_local_groups, wo_a, wo_b, wo_b_scale,
        local_t, o_partial,
    )
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_o_projection_complete", deps=[completion_tid]):
        completion_anchor = pl.read(o_partial, [0, 0])
        pl.write(o_partial, [0, 0], completion_anchor)
    return o_partial


def build_tensor_specs(local_t):
    """Build deterministic capacity-static inputs with a poisoned receive tail."""
    import torch

    from golden import ScalarSpec, TensorSpec

    group_t = SP_SIZE * local_t

    def init_attention_local_groups():
        shape = (LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN)
        values = torch.arange(LOCAL_O_GROUPS * GROUP_T_PAD * O_GROUP_IN, dtype=torch.int32)
        attention = values.remainder(31).sub(15).reshape(shape).to(torch.bfloat16)
        attention[:, :group_t, 0] = 127.0
        attention[:, group_t:, :] = float("nan")
        return attention

    def init_wo_a():
        weight = torch.zeros(LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16)
        diagonal = torch.arange(O_LORA)
        for local_group in range(LOCAL_O_GROUPS):
            group_scale = local_group + 1
            for k_block, coefficient in enumerate((0.5, -0.25, 0.125, -0.0625)):
                k_diagonal = diagonal + k_block * O_LORA
                weight[local_group, diagonal, k_diagonal] = group_scale * coefficient
        return weight

    def init_wo_b():
        values = torch.arange(D * LOCAL_O_WIDTH, dtype=torch.int32)
        return values.remainder(7).sub(3).reshape(D, LOCAL_O_WIDTH).to(torch.int8)

    def init_o_partial():
        return torch.zeros(GROUP_T_PAD, D)

    def init_wo_b_scale():
        values = torch.arange(D, dtype=torch.int32).remainder(4).to(torch.float32)
        return values * 0.25 + 0.5

    attention_shape = [LOCAL_O_GROUPS, GROUP_T_PAD, O_GROUP_IN]
    return [
        TensorSpec("attention_local_groups", attention_shape, torch.bfloat16, init_value=init_attention_local_groups),
        TensorSpec("wo_a", [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, LOCAL_O_WIDTH], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        ScalarSpec("local_t", torch.int32, local_t),
        TensorSpec("o_partial", [GROUP_T_PAD, D], torch.float32, init_value=init_o_partial, is_output=True),
    ]


def golden_decode_o_projection_cp(tensors):
    """Compute the per-group A8 projection over the compact receive prefix."""
    import torch

    group_t = SP_SIZE * int(tensors["local_t"])
    attention = tensors["attention_local_groups"][:, :group_t].float()
    wo_a = tensors["wo_a"].float()
    o_a = torch.einsum("gti,gri->gtr", attention, wo_a)
    row_amax = o_a.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / row_amax
    o_a_i8 = torch.round(o_a * scale_q).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = 1.0 / scale_q
    wo_b = tensors["wo_b"].reshape(D, LOCAL_O_GROUPS, O_LORA)
    o_partial = torch.zeros(group_t, D, dtype=torch.float32)
    for local_group in range(LOCAL_O_GROUPS):
        group_i32 = o_a_i8[local_group].to(torch.int32)
        weight_i32 = wo_b[:, local_group].to(torch.int32)
        group_partial = group_i32 @ weight_i32.T
        o_partial = o_partial + group_partial.float() * scale_dq[local_group]
    o_partial = o_partial * tensors["wo_b_scale"].float().unsqueeze(0)
    tensors["o_partial"][:group_t] = o_partial


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=("all", "max", "subcapacity"), default="all")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    case_local_t = {"max": LOCAL_T, "subcapacity": LOCAL_T - 1}
    selected_cases = tuple(case_local_t) if args.case == "all" else (args.case,)
    for case in selected_cases:
        local_t = case_local_t[case]
        group_t = SP_SIZE * local_t
        partial_compare = ratio_allclose(
            atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0,
            valid_rows=group_t, zero_tail=True,
        )
        result = run_jit(
            fn=decode_o_projection_cp_test,
            specs=build_tensor_specs(local_t),
            golden_fn=golden_decode_o_projection_cp,
            compile_only=args.compile_only,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            compare_fn={"o_partial": partial_compare},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
