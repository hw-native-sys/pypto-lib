# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE routed expert: gate/up matmul, SwiGLU, requant, W2, output scaling."""


import pypto.language as pl

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS,
                    EP, RECV_MAX)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# EP layout / recv buffers (single-card view: kernel only sees the local shard)
N_LOCAL_EXPERTS = M.n_routed_experts // EP

# tiling
RECV_TILE = 64
K_TILE = 512
INTER_K = 512
MM_INTER_TILE = 256
MM_GATE_INNER = 4
ACT_INTER_TILE = 64
ACT_GATE_INNER = 4
D_OUT_TILE = 256
QUANT_TILE = 512
QUANT_ROW_TILE = 16
QUANT_SCALE_PAD = 8
D_OUT_TILE_ACT = 256
W2_INNER = 1
W2_ACT_INNER = 8
TILES_PER_EXPERT = RECV_MAX // RECV_TILE

assert RECV_MAX % RECV_TILE == 0, "RECV_MAX must be a whole number of RECV_TILE row-tiles"


@pl.jit.inline(auto_scope=False)
def expert_routed_tile(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y_tile: pl.Tensor[[RECV_TILE, D], pl.BF16],
    local_e: pl.Scalar[pl.INDEX],
    tile_row: pl.Scalar[pl.INDEX],
    valid_rows: pl.Scalar[pl.INDEX],
    inputs_ready: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    recv_x_flat = pl.reshape(recv_x, [N_LOCAL_EXPERTS * RECV_MAX, D])
    flat_tile_row = local_e * RECV_MAX + tile_row
    h_tile_i8 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT8)
    h_tile_scale_dq = pl.create_tensor([RECV_TILE, QUANT_SCALE_PAD], dtype=pl.FP32, manual_dep=True)
    quant_tids = pl.array.create(1, pl.TASK_ID)

    with pl.scope():
        gate_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)
        up_tile_i32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.INT32)

        with pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_gate_mm", deps=[inputs_ready]):
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * MM_INTER_TILE
                gate_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + K_TILE]
                    w1_chunk = routed_w1[local_e : local_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                    gate_acc = pl.matmul_acc(gate_acc, x_chunk, w1_chunk, b_trans=True, init_cond=(k0 == 0))
                gate_tile_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(gate_acc, [RECV_TILE, MM_INTER_TILE])

        with pl.spmd(MOE_INTER // (MM_GATE_INNER * MM_INTER_TILE), name_hint="exp_up_mm", deps=[inputs_ready]):
            block = pl.tile.get_block_idx()
            n_base = block * (MM_GATE_INNER * MM_INTER_TILE)
            for inner in pl.range(MM_GATE_INNER):
                n0 = n_base + inner * MM_INTER_TILE
                up_acc = pl.create_tensor([1, RECV_TILE, MM_INTER_TILE], dtype=pl.INT32)
                for k0 in pl.pipeline(0, D, K_TILE, stage=2):
                    x_chunk = recv_x_flat[flat_tile_row : flat_tile_row + RECV_TILE, k0 : k0 + K_TILE]
                    w3_chunk = routed_w3[local_e : local_e + 1, n0 : n0 + MM_INTER_TILE, k0 : k0 + K_TILE]
                    up_acc = pl.matmul_acc(up_acc, x_chunk, w3_chunk, b_trans=True, init_cond=(k0 == 0))
                up_tile_i32[:, n0 : n0 + MM_INTER_TILE] = pl.reshape(up_acc, [RECV_TILE, MM_INTER_TILE])

        h_tile_fp32 = pl.create_tensor([RECV_TILE, MOE_INTER], dtype=pl.FP32)
        with pl.spmd(MOE_INTER // (ACT_GATE_INNER * ACT_INTER_TILE), name_hint="exp_gate_up_act"):
            block = pl.tile.get_block_idx()
            inter_base = block * (ACT_GATE_INNER * ACT_INTER_TILE)
            for inner in pl.pipeline(ACT_GATE_INNER, stage=2):
                inter0 = inter_base + inner * ACT_INTER_TILE
                gate_i32 = gate_tile_i32[:, inter0 : inter0 + ACT_INTER_TILE]
                up_i32 = up_tile_i32[:, inter0 : inter0 + ACT_INTER_TILE]
                x_scale = pl.reshape(
                    recv_scale_dq[
                        local_e : local_e + 1,
                        tile_row : tile_row + RECV_TILE,
                    ],
                    [RECV_TILE, 1],
                )
                gate_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(
                        pl.cast(gate_i32, target_type=pl.FP32, mode="none"),
                        x_scale,
                    ),
                    routed_w1_scale[
                        local_e : local_e + 1,
                        inter0 : inter0 + ACT_INTER_TILE,
                    ],
                )
                up_fp32 = pl.col_expand_mul(
                    pl.row_expand_mul(
                        pl.cast(up_i32, target_type=pl.FP32, mode="none"),
                        x_scale,
                    ),
                    routed_w3_scale[
                        local_e : local_e + 1,
                        inter0 : inter0 + ACT_INTER_TILE,
                    ],
                )
                if SWIGLU_LIMIT > 0.0:
                    gate_fp32 = pl.minimum(gate_fp32, SWIGLU_LIMIT)
                    up_fp32 = pl.maximum(pl.minimum(up_fp32, SWIGLU_LIMIT), -SWIGLU_LIMIT)
                sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_fp32)), 1.0))
                activated = pl.mul(pl.mul(gate_fp32, sigmoid), up_fp32)
                activated = pl.set_validshape(activated, valid_rows, ACT_INTER_TILE)
                h_tile_fp32[:, inter0 : inter0 + ACT_INTER_TILE] = pl.fillpad(activated, pad_value=pl.PadValue.zero)

        with pl.spmd(RECV_TILE // QUANT_ROW_TILE, name_hint="exp_h_q") as quant_tid:
            quant_block = pl.tile.get_block_idx()
            quant_row = quant_block * QUANT_ROW_TILE
            row_amax = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                h_amax_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + QUANT_TILE]
                h_abs = pl.maximum(h_amax_chunk, pl.neg(h_amax_chunk))
                row_amax = pl.maximum(row_amax, pl.reshape(pl.row_max(h_abs), [1, QUANT_ROW_TILE]))
            quant_scale = pl.div(pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), row_amax)
            dequant_scale_col = pl.reshape(pl.recip(quant_scale), [QUANT_ROW_TILE, 1])
            dequant_scale_zeros = pl.full([QUANT_ROW_TILE, QUANT_SCALE_PAD], dtype=pl.FP32, value=0.0)
            h_tile_scale_dq[
                quant_row : quant_row + QUANT_ROW_TILE,
                :,
            ] = pl.row_expand(dequant_scale_zeros, dequant_scale_col)
            quant_scale_col = pl.reshape(quant_scale, [QUANT_ROW_TILE, 1])
            for k0 in pl.pipeline(0, MOE_INTER, QUANT_TILE, stage=2):
                h_quant_chunk = h_tile_fp32[quant_row : quant_row + QUANT_ROW_TILE, k0 : k0 + QUANT_TILE]
                h_scaled = pl.row_expand_mul(h_quant_chunk, quant_scale_col)
                h_i32 = pl.cast(h_scaled, target_type=pl.INT32, mode="rint")
                h_fp16 = pl.cast(h_i32, target_type=pl.FP16, mode="round")
                h_tile_i8[
                    quant_row : quant_row + QUANT_ROW_TILE,
                    k0 : k0 + QUANT_TILE,
                ] = pl.cast(
                    h_fp16,
                    target_type=pl.INT8,
                    mode="trunc",
                )
        quant_tids[0] = quant_tid

    y_i32 = pl.create_tensor([RECV_TILE, D], dtype=pl.INT32)
    with pl.spmd(
        D // (W2_INNER * D_OUT_TILE),
        name_hint="exp_w2_mm",
        deps=[quant_tids[0]],
        allow_early_resolve=True,
    ) as w2_tid:
        block = pl.tile.get_block_idx()
        d_base = block * (W2_INNER * D_OUT_TILE)
        for inner in pl.range(W2_INNER):
            d0 = d_base + inner * D_OUT_TILE
            y_acc = pl.create_tensor([1, RECV_TILE, D_OUT_TILE], dtype=pl.INT32)
            for k0 in pl.pipeline(0, MOE_INTER, INTER_K, stage=2):
                h_w2_chunk = h_tile_i8[:, k0 : k0 + INTER_K]
                w2_chunk = routed_w2[local_e : local_e + 1, d0 : d0 + D_OUT_TILE, k0 : k0 + INTER_K]
                y_acc = pl.matmul_acc(y_acc, h_w2_chunk, w2_chunk, b_trans=True, init_cond=(k0 == 0))
            y_i32[:, d0 : d0 + D_OUT_TILE] = pl.reshape(y_acc, [RECV_TILE, D_OUT_TILE])

    with pl.spmd(
        D // (W2_ACT_INNER * D_OUT_TILE_ACT),
        name_hint="exp_w2_act",
        deps=[w2_tid],
        allow_early_resolve=True,
    ) as w2_act_tid:
        block = pl.tile.get_block_idx()
        d_base = block * (W2_ACT_INNER * D_OUT_TILE_ACT)
        route_weight = pl.reshape(recv_weights[local_e : local_e + 1, tile_row : tile_row + RECV_TILE], [RECV_TILE, 1])
        row_scale = pl.mul(pl.row_max(h_tile_scale_dq), route_weight)
        for inner in pl.pipeline(W2_ACT_INNER, stage=2):
            d0 = d_base + inner * D_OUT_TILE_ACT
            y_fp32 = pl.cast(y_i32[:, d0 : d0 + D_OUT_TILE_ACT], target_type=pl.FP32, mode="none")
            y_fp32 = pl.col_expand_mul(
                pl.row_expand_mul(y_fp32, row_scale),
                routed_w2_scale[
                    local_e : local_e + 1,
                    d0 : d0 + D_OUT_TILE_ACT,
                ],
            )
            recv_y_tile[:, d0 : d0 + D_OUT_TILE_ACT] = pl.cast(y_fp32, target_type=pl.BF16, mode="rint")
    return w2_act_tid


@pl.jit(auto_scope=False)
def expert_routed_test(
    recv_x: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.INT8],
    recv_scale_dq: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_weights: pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX], pl.FP32],
    recv_expert_count: pl.Tensor[[N_LOCAL_EXPERTS, 1], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL_EXPERTS, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL_EXPERTS, D], pl.FP32],
    recv_y: pl.Out[pl.Tensor[[N_LOCAL_EXPERTS, RECV_MAX, D], pl.BF16]],
):
    recv_y_flat = pl.reshape(recv_y, [N_LOCAL_EXPERTS * RECV_MAX, D])
    d_tiles = D // D_OUT_TILE_ACT
    # Rows past an expert's receive count are never written by a tile; zero them
    # so the whole recv_y slab matches the golden.
    with pl.spmd(N_LOCAL_EXPERTS * TILES_PER_EXPERT * d_tiles, name_hint="expert_recv_y_zero") as inputs_ready:
        block = pl.tile.get_block_idx()
        zero_row = (block // d_tiles) * RECV_TILE
        zero_col = (block % d_tiles) * D_OUT_TILE_ACT
        recv_y_flat[
            zero_row : zero_row + RECV_TILE,
            zero_col : zero_col + D_OUT_TILE_ACT,
        ] = pl.full([RECV_TILE, D_OUT_TILE_ACT], dtype=pl.BF16, value=0.0)
    for local_e in pl.parallel(N_LOCAL_EXPERTS):
        n_rows = pl.cast(pl.read(recv_expert_count, [local_e, 0]), pl.INDEX)
        n_tiles = (n_rows + RECV_TILE - 1) // RECV_TILE
        for tile in pl.parallel(n_tiles):
            tile_row = tile * RECV_TILE
            valid_rows = pl.min(RECV_TILE, n_rows - tile_row)
            flat_tile_row = local_e * RECV_MAX + tile_row
            with pl.scope():
                recv_y_tile = pl.create_tensor([RECV_TILE, D], dtype=pl.BF16)
                expert_routed_tile(
                    recv_x, recv_scale_dq, recv_weights,
                    routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
                    recv_y_tile,
                    local_e, tile_row, valid_rows, inputs_ready,
                )
                recv_y_flat[flat_tile_row : flat_tile_row + RECV_TILE, :] = recv_y_tile
    return recv_y


def golden_expert_routed(tensors):
    """Torch reference for the routed expert. recv_y is the per-row routing-
    weight-scaled SwiGLU output, ready for combine reduce to simply sum.

    Per-expert layout: recv_x[e, 0:cnt[e], :] is the valid INT8 receive
    payload; recv_y[e, cnt[e]:, :] stays at zero."""
    from utils import int8_quant_per_row
    import torch
    import torch.nn.functional as F

    def dequant_w(w_i8, w_scale):
        return w_i8.to(torch.float32) * w_scale.unsqueeze(-1)

    recv_x_i8 = tensors["recv_x"]  # INT8, pre-quantized in dispatch
    recv_scale_dq = tensors["recv_scale_dq"].float()  # [E, RECV_MAX]
    recv_weights = tensors["recv_weights"].float()  # [E, RECV_MAX]
    recv_expert_count = tensors["recv_expert_count"]  # [E, 1] int32
    w1 = dequant_w(tensors["routed_w1"], tensors["routed_w1_scale"].float())
    w3 = dequant_w(tensors["routed_w3"], tensors["routed_w3_scale"].float())
    w2 = dequant_w(tensors["routed_w2"], tensors["routed_w2_scale"].float())

    recv_y = torch.zeros(N_LOCAL_EXPERTS, RECV_MAX, D)
    for e in range(N_LOCAL_EXPERTS):
        n_rows = int(recv_expert_count[e, 0].item())
        if n_rows == 0:
            continue
        x_sub_i8 = recv_x_i8[e, :n_rows, :]
        x_sub_sd = recv_scale_dq[e, :n_rows].reshape(-1, 1)
        x_sub_q = x_sub_i8.float() * x_sub_sd
        w_per_row = recv_weights[e, :n_rows].reshape(-1, 1)

        gate = x_sub_q @ w1[e].T
        up = x_sub_q @ w3[e].T
        if SWIGLU_LIMIT > 0:
            gate = gate.clamp(max=SWIGLU_LIMIT)
            up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
        h = F.silu(gate) * up
        # A8 requant before w2 matmul.
        h_i8, h_sd = int8_quant_per_row(h)
        h = h_i8.float() * (h_sd * w_per_row)
        recv_y[e, :n_rows, :] = h @ w2[e].T

    tensors["recv_y"][:] = recv_y.to(torch.bfloat16)


def gen_routed_weight(shape, dequant_std):
    """Synthesize a routed-expert per-channel-symmetric INT8 weight + FP32 scale by
    simulating the real DeepSeek-V4-Flash MXFP4 routed-expert quant grid (e2m1, per-32-group
    E8M0 scale), then re-quantizing per-output-channel. A plain ``randn`` INT8 is wrong:
    routed collapses onto ~37 discrete levels with an ~11.6% zero spike (the FP4 grid) and a
    per-channel scale CV ~0.09 (the fine group scale flattens it). Per-output-channel INT8 is
    scale-invariant, so the level structure / zero spike emerge from the grid alone and
    ``dequant_std`` only sets the absolute scale magnitude. (shared experts use a different
    grid -- see expert_shared.gen_shared_weight.)

    ``shape`` last dim = reduction (in) dim; leading dims map to the per-output-channel
    scale shape ([E, out, in] -> scale [E, out]).
    """
    import torch

    FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    FP4_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])  # nearest-grid bounds
    FP4_MAX, TINY = 6.0, 1e-20
    GROUP = 32
    CHUNK_ELEMS = 1 << 25    # elements per pass; unchunked this walks GiB-sized temporaries

    *lead, out, inn = shape
    n_lead = 1
    for dim in lead:
        n_lead *= dim

    W = torch.randn(*shape).reshape(n_lead, out, inn)
    w_i8 = torch.empty(n_lead, out, inn, dtype=torch.int8)
    scale = torch.empty(n_lead, out, 1, dtype=torch.float32)

    # e2m1 + per-32-group E8M0 (round-up) scale on the in dim, then per-output-channel
    # INT8. Chunked over the leading dim: every reduction here is confined to one
    # (out, in) row, so the chunk boundary cannot change a result.
    step = max(1, CHUNK_ELEMS // (out * inn))
    for i0 in range(0, n_lead, step):
        w = W[i0:i0 + step]
        wg = w.reshape(-1, out, inn // GROUP, GROUP)
        absw = wg.abs()
        grp_scale = torch.exp2(torch.ceil(torch.log2((absw.amax(-1, keepdim=True) / FP4_MAX).clamp_min(TINY))))
        idx = torch.bucketize(absw.div_(grp_scale), FP4_MID).clamp_max_(7)
        wq = (torch.sign(wg) * FP4_MAG[idx]).mul_(grp_scale).reshape(w.shape)
        amax = wq.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        chan_scale = amax / INT8_SCALE_MAX
        w_i8[i0:i0 + step] = torch.round(wq.div_(chan_scale)).clamp_(-INT8_SCALE_MAX, INT8_SCALE_MAX).to(torch.int8)
        scale[i0:i0 + step] = chan_scale
    del W

    scale = (scale * (dequant_std / (w_i8.float() * scale).std())).squeeze(-1).float()
    return w_i8.reshape(*shape), scale.reshape(*lead, out)


def build_tensor_specs():
    from utils import int8_quant_per_row
    import torch
    from golden import TensorSpec

    # Across-layer-mean dequant std of the real DeepSeek-V4-Flash MXFP4 routed experts.
    ROUTED_DEQUANT_STD = {"w1": 2.47e-2, "w2": 2.44e-2, "w3": 2.46e-2}

    # Distribute B*S*TOPK token-expert pairs uniformly across local experts.
    total = B * S * M.num_experts_per_tok
    counts = torch.bincount(torch.randint(0, N_LOCAL_EXPERTS, (total,)), minlength=N_LOCAL_EXPERTS).to(torch.int32)
    counts_2d = counts.reshape(N_LOCAL_EXPERTS, 1)

    # INT8 recv_x + per-row dequant scale. Tail rows are INT8 0 with scale 0,
    # so dequant produces 0.
    x_bf16 = torch.randn(N_LOCAL_EXPERTS, RECV_MAX, D, dtype=torch.bfloat16)
    valid_mask_3d = (torch.arange(RECV_MAX).reshape(1, RECV_MAX, 1) < counts.reshape(N_LOCAL_EXPERTS, 1, 1))
    recv_x_i8_pre, recv_scale_dq_pre = int8_quant_per_row(x_bf16)
    recv_x_i8_pre = torch.where(valid_mask_3d, recv_x_i8_pre, torch.zeros_like(recv_x_i8_pre))
    valid_mask_2d = valid_mask_3d.squeeze(-1)
    recv_scale_dq_pre = torch.where(
        valid_mask_2d,
        recv_scale_dq_pre.squeeze(-1),
        torch.zeros_like(recv_scale_dq_pre.squeeze(-1)),
    )

    def init_recv_x():
        return recv_x_i8_pre

    def init_recv_scale_dq():
        return recv_scale_dq_pre.float()

    def init_recv_expert_count():
        return counts_2d

    # Per-row routing weight in [0, 1); tail rows (slot >= count) stay 0.
    recv_weights_pre = torch.rand(N_LOCAL_EXPERTS, RECV_MAX, dtype=torch.float32)
    recv_weights_pre = torch.where(valid_mask_2d, recv_weights_pre, torch.zeros_like(recv_weights_pre))

    def init_recv_weights():
        return recv_weights_pre

    # (int8, per-channel scale) on the real MXFP4 routed-expert quant grid.
    w1_i8, w1_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w1"])
    w3_i8, w3_s = gen_routed_weight((N_LOCAL_EXPERTS, MOE_INTER, D), ROUTED_DEQUANT_STD["w3"])
    w2_i8, w2_s = gen_routed_weight((N_LOCAL_EXPERTS, D, MOE_INTER), ROUTED_DEQUANT_STD["w2"])

    return [
        TensorSpec("recv_x", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.int8, init_value=init_recv_x),
        TensorSpec("recv_scale_dq", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_scale_dq),
        TensorSpec("recv_weights", [N_LOCAL_EXPERTS, RECV_MAX], torch.float32, init_value=init_recv_weights),
        TensorSpec("recv_expert_count", [N_LOCAL_EXPERTS, 1], torch.int32, init_value=init_recv_expert_count),
        TensorSpec("routed_w1", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w1_i8),
        TensorSpec("routed_w1_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w1_s),
        TensorSpec("routed_w3", [N_LOCAL_EXPERTS, MOE_INTER, D], torch.int8, init_value=lambda: w3_i8),
        TensorSpec("routed_w3_scale", [N_LOCAL_EXPERTS, MOE_INTER], torch.float32, init_value=lambda: w3_s),
        TensorSpec("routed_w2", [N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8, init_value=lambda: w2_i8),
        TensorSpec("routed_w2_scale", [N_LOCAL_EXPERTS, D], torch.float32, init_value=lambda: w2_s),
        TensorSpec("recv_y", [N_LOCAL_EXPERTS, RECV_MAX, D], torch.bfloat16),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=expert_routed_test,
        specs=build_tensor_specs(),
        golden_fn=golden_expert_routed,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            # BF16 recv_y, ~1 ULP.
            "recv_y": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
