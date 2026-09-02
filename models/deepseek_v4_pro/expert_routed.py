# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed local expert compute (decode, EP single-card).

Only the routed-expert path lives here. The shared expert was split out
into ``expert_shared.py``; both kernels are composed in ``moe.py``.
"""


import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, EP_WORLD_SIZE, RECV_MAX


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE

# tiling
RECV_TILE = 16
MX_GROUP = 32
K_SCALE = D // MX_GROUP
H_SCALE = MOE_INTER // MX_GROUP
MX_K_TILE = 256
MX_MM_INTER_TILE = 256
MX_K_SCALE_GROUPS = MX_K_TILE // MX_GROUP
MX_MM_TASK_TILE = 1024
assert MOE_INTER % MX_MM_INTER_TILE == 0
assert D % MX_K_TILE == 0
ACT_INTER_TILE = 128
ACT_GATE_INNER = 4
MX_W2_D_OUT_TILE = 256
MX_W2_TASK_TILE = 1024
ROUTE_D_OUT_TILE = 512
ROUTE_TASK_TILE = D

assert RECV_MAX % RECV_TILE == 0, "RECV_MAX must be a whole number of RECV_TILE row-tiles"
# Every `<dim> // <tile>` used as a loop/task bound must divide exactly, or the
# bound silently truncates and part of the tensor is never written.
assert MOE_INTER % MX_MM_TASK_TILE == 0 and D % MX_W2_TASK_TILE == 0
assert MX_MM_TASK_TILE % (ACT_GATE_INNER * ACT_INTER_TILE) == 0
assert MX_W2_TASK_TILE % MX_W2_D_OUT_TILE == 0
assert ROUTE_TASK_TILE % ROUTE_D_OUT_TILE == 0


@pl.jit.inline(auto_scope=False)
def expert_routed(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    recv_y: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    recv_mx_scale_view = pl.tensor.view(
        recv_mx_scale,
        [N_LOCAL_EXPERTS * RECV_MAX, K_SCALE],
        layout=pl.MX_A_ZZ,
    )
    routed_w1_flat = pl.reshape(routed_w1, [N_LOCAL_EXPERTS * D, MOE_INTER])
    routed_w3_flat = pl.reshape(routed_w3, [N_LOCAL_EXPERTS * D, MOE_INTER])
    routed_w2_flat = pl.reshape(routed_w2, [N_LOCAL_EXPERTS * MOE_INTER, D])
    with pl.scope():
        h_mx = pl.create_tensor(
            [N_LOCAL_EXPERTS * RECV_MAX, MOE_INTER], dtype=pl.FP8E4M3FN
        )
        h_scale_backing = pl.create_tensor(
            [1, N_LOCAL_EXPERTS * RECV_MAX * H_SCALE], dtype=pl.FP8E8M0
        )
        for local_i in pl.parallel(N_LOCAL_EXPERTS):
            flat_base = local_i * RECV_MAX

            n_rows = pl.read(recv_expert_count, [local_i, 0])
            n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE

            for t in pl.parallel(n_tiles):
                t0 = t * RECV_TILE
                flat_t0 = flat_base + t0
                valid_rows = pl.min(RECV_TILE, n_rows - t0)

                with pl.scope():
                    gate_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
                    up_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
                    h_tile_mx = h_mx[flat_t0 : flat_t0 + RECV_TILE]
                    h_tile_scale_backing = h_scale_backing[
                        :, flat_t0 * H_SCALE : (flat_t0 + RECV_TILE) * H_SCALE
                    ]

                    with pl.spmd(MOE_INTER // MX_MM_TASK_TILE, name_hint="exp_gate_mx_mm"):
                        nb_idx = pl.tile.get_block_idx()
                        n_base = nb_idx * MX_MM_TASK_TILE
                        for ng in pl.range(MX_MM_TASK_TILE // MX_MM_INTER_TILE):
                            n0 = n_base + ng * MX_MM_INTER_TILE
                            w1_row_base = local_i * D
                            w1_scale_row_base = local_i * K_SCALE
                            xs0 = pl.load(recv_x_flat, [flat_t0, 0], [RECV_TILE, MX_K_TILE])
                            xs_scale0 = pl.load(recv_mx_scale_view, [flat_t0, 0], [RECV_TILE, MX_K_SCALE_GROUPS])
                            w1_k0 = pl.load(routed_w1_flat, [w1_row_base, n0], [MX_K_TILE, MX_MM_INTER_TILE])
                            w1_scale0 = pl.load(routed_w1_scale, [w1_scale_row_base, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE])
                            gate_acc = pl.matmul_mx(xs0, xs_scale0, w1_k0, w1_scale0)
                            for k0 in pl.pipeline(MX_K_TILE, D, MX_K_TILE, stage=2):
                                ks = k0 // MX_GROUP
                                xs_k = pl.load(recv_x_flat, [flat_t0, k0], [RECV_TILE, MX_K_TILE])
                                xs_scale_k = pl.load(recv_mx_scale_view, [flat_t0, ks], [RECV_TILE, MX_K_SCALE_GROUPS])
                                w1_k = pl.load(routed_w1_flat, [w1_row_base + k0, n0], [MX_K_TILE, MX_MM_INTER_TILE])
                                w1_scale_k = pl.load(routed_w1_scale, [w1_scale_row_base + ks, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE])
                                gate_acc = pl.matmul_mx_acc(gate_acc, xs_k, xs_scale_k, w1_k, w1_scale_k)
                            gate_tile_fp32 = pl.store(gate_acc, [0, n0], gate_tile_fp32)

                    with pl.spmd(MOE_INTER // MX_MM_TASK_TILE, name_hint="exp_up_mx_mm"):
                        ub_idx = pl.tile.get_block_idx()
                        n_base = ub_idx * MX_MM_TASK_TILE
                        for ug in pl.range(MX_MM_TASK_TILE // MX_MM_INTER_TILE):
                            n0 = n_base + ug * MX_MM_INTER_TILE
                            w3_row_base = local_i * D
                            w3_scale_row_base = local_i * K_SCALE
                            xs0 = pl.load(recv_x_flat, [flat_t0, 0], [RECV_TILE, MX_K_TILE])
                            xs_scale0 = pl.load(recv_mx_scale_view, [flat_t0, 0], [RECV_TILE, MX_K_SCALE_GROUPS])
                            w3_k0 = pl.load(routed_w3_flat, [w3_row_base, n0], [MX_K_TILE, MX_MM_INTER_TILE])
                            w3_scale0 = pl.load(routed_w3_scale, [w3_scale_row_base, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE])
                            up_acc = pl.matmul_mx(xs0, xs_scale0, w3_k0, w3_scale0)
                            for k0 in pl.pipeline(MX_K_TILE, D, MX_K_TILE, stage=2):
                                ks = k0 // MX_GROUP
                                xs_k = pl.load(recv_x_flat, [flat_t0, k0], [RECV_TILE, MX_K_TILE])
                                xs_scale_k = pl.load(recv_mx_scale_view, [flat_t0, ks], [RECV_TILE, MX_K_SCALE_GROUPS])
                                w3_k = pl.load(routed_w3_flat, [w3_row_base + k0, n0], [MX_K_TILE, MX_MM_INTER_TILE])
                                w3_scale_k = pl.load(routed_w3_scale, [w3_scale_row_base + ks, n0], [MX_K_SCALE_GROUPS, MX_MM_INTER_TILE])
                                up_acc = pl.matmul_mx_acc(up_acc, xs_k, xs_scale_k, w3_k, w3_scale_k)
                            up_tile_fp32 = pl.store(up_acc, [0, n0], up_tile_fp32)

                    h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
                    with pl.spmd(
                        MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE),
                        name_hint="exp_gate_up_act",
                    ):
                        ab_idx = pl.tile.get_block_idx()
                        a_base = ab_idx * (ACT_GATE_INNER * ACT_INTER_TILE)
                        for ag in pl.pipeline(ACT_GATE_INNER, stage=2):
                            a0 = a_base + ag * ACT_INTER_TILE
                            gate_2d = gate_tile_fp32[:, a0 : a0 + ACT_INTER_TILE]
                            up_2d = up_tile_fp32[:, a0 : a0 + ACT_INTER_TILE]
                            if SWIGLU_LIMIT > 0.0:
                                gate_2d = pl.minimum(gate_2d, SWIGLU_LIMIT)
                                up_2d = pl.maximum(pl.minimum(up_2d, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_2d)), 1.0))
                            silu = pl.mul(gate_2d, sigmoid)
                            gated = pl.mul(silu, up_2d)
                            gated_valid = pl.set_validshape(gated, valid_rows, ACT_INTER_TILE)
                            h_tile_fp32[:, a0 : a0 + ACT_INTER_TILE] = pl.fillpad(
                                gated_valid, pad_value=pl.PadValue.zero
                            )
                    with pl.spmd(
                        MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE),
                        name_hint="exp_h_mx_q",
                    ):
                        q_idx = pl.tile.get_block_idx()
                        q0 = q_idx * (ACT_GATE_INNER * ACT_INTER_TILE)
                        h_fp32 = pl.load(
                            h_tile_fp32,
                            [0, q0],
                            [RECV_TILE, ACT_GATE_INNER * ACT_INTER_TILE],
                        )
                        h_quant, h_scale = pl.quant_mx(h_fp32, group_axis=1)
                        h_tile_mx = pl.store(h_quant, [0, q0], h_tile_mx)
                        scale_offset = q_idx * RECV_TILE * (ACT_GATE_INNER * ACT_INTER_TILE // MX_GROUP)
                        h_tile_scale_backing = pl.store(
                            pl.reshape(h_scale, [1, RECV_TILE * (ACT_GATE_INNER * ACT_INTER_TILE // MX_GROUP)]),
                            [0, scale_offset],
                            h_tile_scale_backing,
                        )

        with pl.scope():
            for local_e in pl.parallel(N_LOCAL_EXPERTS):
                e_flat_base = local_e * RECV_MAX

                e_rows = pl.read(recv_expert_count, [local_e, 0])
                e_tiles = (e_rows + RECV_TILE - 1) // RECV_TILE

                for tt in pl.parallel(e_tiles):
                    tt0 = tt * RECV_TILE
                    flat_tt0 = e_flat_base + tt0
                    valid_rows = pl.min(RECV_TILE, e_rows - tt0)
                    h_tile_mx = h_mx[flat_tt0 : flat_tt0 + RECV_TILE]
                    h_tile_scale_backing = h_scale_backing[
                        :, flat_tt0 * H_SCALE : (flat_tt0 + RECV_TILE) * H_SCALE
                    ]
                    h_tile_scale_mx = pl.tensor.view(
                        h_tile_scale_backing,
                        [RECV_TILE, H_SCALE],
                        layout=pl.MX_A_ZZ,
                    )
                    recv_y_tile_fp32 = pl.create_tensor([RECV_TILE, D], dtype=pl.FP32)
                    recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.BF16)
                    with pl.spmd(D // MX_W2_TASK_TILE, name_hint="exp_w2_mx_mm"):
                        wb_idx = pl.tile.get_block_idx()
                        d_base = wb_idx * MX_W2_TASK_TILE
                        for dg in pl.range(MX_W2_TASK_TILE // MX_W2_D_OUT_TILE):
                            d0 = d_base + dg * MX_W2_D_OUT_TILE
                            w2_row_base = local_e * MOE_INTER
                            w2_scale_row_base = local_e * H_SCALE
                            h0 = pl.load(h_tile_mx, [0, 0], [RECV_TILE, MX_K_TILE])
                            h_scale0 = pl.load(h_tile_scale_mx, [0, 0], [RECV_TILE, MX_K_SCALE_GROUPS])
                            w2_0 = pl.load(routed_w2_flat, [w2_row_base, d0], [MX_K_TILE, MX_W2_D_OUT_TILE])
                            w2_scale0 = pl.load(routed_w2_scale, [w2_scale_row_base, d0], [MX_K_SCALE_GROUPS, MX_W2_D_OUT_TILE])
                            y_acc = pl.matmul_mx(h0, h_scale0, w2_0, w2_scale0)
                            for k0 in pl.pipeline(MX_K_TILE, MOE_INTER, MX_K_TILE, stage=2):
                                ks = k0 // MX_GROUP
                                h_k = pl.load(h_tile_mx, [0, k0], [RECV_TILE, MX_K_TILE])
                                h_scale_k = pl.load(h_tile_scale_mx, [0, ks], [RECV_TILE, MX_K_SCALE_GROUPS])
                                w2_k = pl.load(routed_w2_flat, [w2_row_base + k0, d0], [MX_K_TILE, MX_W2_D_OUT_TILE])
                                w2_scale_k = pl.load(routed_w2_scale, [w2_scale_row_base + ks, d0], [MX_K_SCALE_GROUPS, MX_W2_D_OUT_TILE])
                                y_acc = pl.matmul_mx_acc(y_acc, h_k, h_scale_k, w2_k, w2_scale_k)
                            recv_y_tile_fp32 = pl.store(y_acc, [0, d0], recv_y_tile_fp32)

                    with pl.spmd(D // ROUTE_TASK_TILE, name_hint="exp_route_weight"):
                        wb_idx = pl.tile.get_block_idx()
                        d_base = wb_idx * ROUTE_TASK_TILE
                        w_row_blk = pl.load(recv_weights, [local_e, tt0], [1, RECV_TILE])
                        w_col_blk = pl.reshape(w_row_blk, [RECV_TILE, 1])
                        for dg in pl.range(ROUTE_TASK_TILE // ROUTE_D_OUT_TILE):
                            d0 = d_base + dg * ROUTE_D_OUT_TILE
                            y_fp32 = pl.load(recv_y_tile_fp32, [0, d0], [RECV_TILE, ROUTE_D_OUT_TILE])
                            y_weighted = pl.row_expand_mul(y_fp32, w_col_blk)
                            y_valid = pl.set_validshape(y_weighted, valid_rows, ROUTE_D_OUT_TILE)
                            y_padded = pl.fillpad(y_valid, pad_value=pl.PadValue.zero)
                            recv_y_tile = pl.store(
                                pl.cast(y_padded, target_type=pl.BF16, mode="rint"),
                                [0, d0],
                                recv_y_tile,
                            )
                    recv_y_flat = pl.assemble(recv_y_flat, recv_y_tile, [flat_tt0, 0])

    return recv_y


@pl.jit
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.FP8E4M3FN],
    recv_mx_scale: pl.Tensor[[1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], pl.FP8E8M0],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.FP8E4M3FN],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.FP8E4M3FN],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS * H_SCALE, D], pl.FP8E8M0, pl.MX_B_NN],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    expert_routed(
        recv_x, recv_mx_scale, recv_weights, recv_expert_count,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        recv_y,
    )
    return recv_y


def golden_expert_routed(tensors):
    """Torch reference for MXFP4-grid W1/W3/W2 with MXFP8 activations."""
    import torch
    import torch.nn.functional as F

    from mx_utils import decode_e8m0_codes, host_quant_mxfp8, matmul_mx_golden

    recv_x = tensors["recv_x"]
    recv_mx_scale = decode_e8m0_codes(
        tensors["recv_mx_scale"].reshape(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE),
        side="a",
    ).reshape(
        N_LOCAL_EXPERTS, RECV_MAX, K_SCALE,
    )
    recv_weights = tensors["recv_weights"].float()
    recv_expert_count = tensors["recv_expert_count"]
    w1_fp8 = tensors["routed_w1"]
    w1_scale = decode_e8m0_codes(tensors["routed_w1_scale"], side="b").reshape(
        N_LOCAL_EXPERTS, K_SCALE, MOE_INTER,
    )
    w3_fp8 = tensors["routed_w3"]
    w3_scale = decode_e8m0_codes(tensors["routed_w3_scale"], side="b").reshape(
        N_LOCAL_EXPERTS, K_SCALE, MOE_INTER,
    )
    w2_fp8 = tensors["routed_w2"]
    w2_scale = decode_e8m0_codes(tensors["routed_w2_scale"], side="b").reshape(
        N_LOCAL_EXPERTS, H_SCALE, D,
    )

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub = recv_x[e, :n_rows, :]
        x_scale = recv_mx_scale[e, :n_rows, :]
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)

        gate = matmul_mx_golden(x_sub, x_scale, w1_fp8[e], w1_scale[e])
        up = matmul_mx_golden(x_sub, x_scale, w3_fp8[e], w3_scale[e])
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        h_fp8, h_scale = host_quant_mxfp8(h, return_e8m0=True)
        y = matmul_mx_golden(h_fp8, h_scale, w2_fp8[e], w2_scale[e])
        recv_y[e, :n_rows, :] = y * w_per_row

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_mx_weights(n_experts, dequant_std, seed_base=0):
    """Expand synthetic MXFP4 W1/W3/W2 through the checkpoint conversion path."""
    import torch
    from mx_utils import gen_mxfp4_weight_kn_device, pack_b_scale, unpack_b_scale

    w1_list, w1s_list, w3_list, w3s_list, w2_list, w2s_list = [], [], [], [], [], []
    for e in range(n_experts):
        seed = seed_base + e * 3
        w1, w1s = gen_mxfp4_weight_kn_device(MOE_INTER, D, dequant_std["w1"], seed=seed)
        w3, w3s = gen_mxfp4_weight_kn_device(MOE_INTER, D, dequant_std["w3"], seed=seed + 1)
        w2, w2s = gen_mxfp4_weight_kn_device(D, MOE_INTER, dequant_std["w2"], seed=seed + 2)
        w1_list.append(w1)
        w1s_list.append(w1s)
        w3_list.append(w3)
        w3s_list.append(w3s)
        w2_list.append(w2)
        w2s_list.append(w2s)
    def pack_expert_scales(scales):
        logical = torch.stack(
            [unpack_b_scale(scale.contiguous().view(torch.uint8)) for scale in scales]
        ).flatten(0, 1)
        return pack_b_scale(logical).view(torch.float8_e8m0fnu)

    return (
        torch.stack(w1_list),
        pack_expert_scales(w1s_list),
        torch.stack(w3_list),
        pack_expert_scales(w3s_list),
        torch.stack(w2_list),
        pack_expert_scales(w2s_list),
    )


def build_tensor_specs():
    import torch
    from golden import TensorSpec
    from mx_utils import host_quant_mxfp8, pack_a_scale

    # Across-layer-mean dequant std (typical layer) of the real routed experts.
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(
        torch.randint(0, N_LOCAL_EXPERTS, (total,)),
        minlength=N_LOCAL_EXPERTS,
    ).to(torch.int32)
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (
        torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1)
    )
    recv_x_pre = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.float8_e4m3fn)
    recv_scale_codes = torch.zeros(N_LOCAL_EXPERTS * RECV_MAX, K_SCALE, dtype=torch.uint8)
    for e in range(N_LOCAL_EXPERTS):
        n = int(counts_2d[e, 0].item())
        if n > 0:
            xf, xs = host_quant_mxfp8(x_bf16[e, :n, :], return_e8m0=True)
            recv_x_pre[e, :n, :] = xf
            recv_scale_codes[e * RECV_MAX : e * RECV_MAX + n, :] = xs.view(torch.uint8)
    recv_mx_scale_pre = pack_a_scale(recv_scale_codes).view(torch.float8_e8m0fnu)
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_x_pre = torch.where(
        valid_mask_3d,
        recv_x_pre,
        torch.zeros_like(recv_x_pre),
    )

    def init_recv_x():
        return recv_x_pre

    def init_recv_mx_scale():
        return recv_mx_scale_pre.reshape(1, -1)

    def init_recv_expert_count():
        return counts_2d

    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(
        valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre)
    )

    def init_recv_weights():
        return recv_weights_pre

    rw1_fp8, rw1_scale, rw3_fp8, rw3_scale, rw2_fp8, rw2_scale = gen_routed_mx_weights(
        N_LOCAL_EXPERTS, ROUTED_DEQUANT_STD, seed_base=10
    )

    fp8 = torch.float8_e4m3fn
    fp8_e8m0 = torch.float8_e8m0fnu

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], fp8, init_value=init_recv_x),
        TensorSpec("recv_mx_scale", [1, N_LOCAL_EXPERTS * RECV_MAX * K_SCALE], fp8_e8m0, init_value=init_recv_mx_scale),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("routed_w1", [N_LOCAL_EXPERTS, D, MOE_INTER], fp8, init_value=lambda: rw1_fp8),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], fp8_e8m0, init_value=lambda: rw1_scale),
        TensorSpec("routed_w3", [N_LOCAL_EXPERTS, D, MOE_INTER], fp8, init_value=lambda: rw3_fp8),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS * K_SCALE, MOE_INTER], fp8_e8m0, init_value=lambda: rw3_scale),
        TensorSpec("routed_w2", [N_LOCAL_EXPERTS, MOE_INTER, D], fp8, init_value=lambda: rw2_fp8),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS * H_SCALE, D], fp8_e8m0, init_value=lambda: rw2_scale),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16),
    ]


def active_recv_ratio_reldiff(*, diff_thd, pct_thd):
    """Validate compact expert rows without borrowing padding capacity.

    ``recv_y`` has a static ``RECV_MAX`` extent for every local expert, but
    only the prefix described by ``recv_expert_count`` is computed. Numerical
    tolerance applies to those active rows only; every inactive row must stay
    bitwise equal to golden so stale or out-of-range writes cannot be hidden by
    the much larger padded buffer.
    """
    import torch

    from golden import ratio_reldiff

    active_compare = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd)

    def compare(actual, expected, **kwargs):
        if actual.shape != expected.shape:
            return False, (
                f"    output shape mismatch: actual={tuple(actual.shape)} "
                f"expected={tuple(expected.shape)}"
            )
        if actual.ndim != 3:
            return False, f"    recv_y must have rank 3, got {tuple(actual.shape)}"

        counts = kwargs.get("inputs", {}).get("recv_expert_count")
        if counts is None:
            return False, "    compare_fn misconfigured: missing input 'recv_expert_count'"
        counts = counts.cpu().to(torch.int64).reshape(-1)
        expert_count, recv_max, _ = actual.shape
        if counts.numel() != expert_count:
            return False, (
                f"    recv_expert_count has {counts.numel()} values, "
                f"expected {expert_count}"
            )
        invalid = (counts < 0) | (counts > recv_max)
        if invalid.any().item():
            expert = int(invalid.nonzero(as_tuple=False)[0].item())
            return False, (
                f"    recv_expert_count[{expert}]={int(counts[expert].item())} "
                f"is outside [0, {recv_max}]"
            )

        rows = torch.arange(recv_max, dtype=torch.int64).reshape(1, -1)
        active = rows < counts.reshape(-1, 1)
        inactive = ~active

        actual_f = actual.float()
        expected_f = expected.float()
        for label, values in (("actual", actual_f), ("expected", expected_f)):
            invalid_values = ~torch.isfinite(values)
            if invalid_values.any().item():
                return False, (
                    f"    illegal values in {label}: "
                    f"count={int(invalid_values.sum().item())}"
                )

        if inactive.any().item() and not torch.equal(actual[inactive], expected[inactive]):
            changed = int((actual[inactive] != expected[inactive]).sum().item())
            return False, f"    inactive recv_y rows changed: changed_values={changed}"

        if not active.any().item():
            return True, ""
        ok, detail = active_compare(actual[active], expected[active], **kwargs)
        if ok:
            return True, ""
        return False, f"    active recv_y rows:\n{detail}"

    compare.__name__ = (
        f"active_recv_ratio_reldiff(diff_thd={diff_thd}, pct_thd={pct_thd})"
    )
    return compare


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    result = run(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
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
            # BF16 recv_y, ~1 ULP. Gen weights reproduce real(L21): 0.016% vs 0.015% of points > 1e-3.
            "recv_y": active_recv_ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
