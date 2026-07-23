# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 dynamic-token MoE router: RMSNorm + gate + topk + normalize."""


import pypto.language as pl

from config import (FLASH as M, MOE_TOKENS, FP32_NEG_INF,
                    INT8_SCALE_MAX, INT8_AMAX_EPS)


T_DYN = pl.dynamic("GATE_TOKENS_DYN")


# model config
T = MOE_TOKENS
D = M.hidden_size
NORM_EPS = M.rms_norm_eps
# Routing space: every rank routes over the full global expert set so dispatch
# can fan tokens across ranks. moe.py shrinks config.FLASH.n_routed_experts to
# 32*EP before importing this module, so N_EXPERTS follows the active EP world.
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
VOCAB = M.vocab_size
N_HASH_LAYERS = M.num_hash_layers

# tiling
T_TILE = 8
GATE_T_TILE = 8
GATE_M_TILE = 16        # cube M-tile: matmul rows must be a multiple of 16 (fractal)
GATE_N_TILE = 16        # expert columns per gate spmd block
assert N_EXPERTS % GATE_N_TILE == 0
D_TILE = 256
GATE_D_TILE = 256
QUANT_TILE = 256
SCORE_PAD = 256         # padded expert row for sort32 + mrgsort
TOPK_PAD = 8            # TOPK padded to 32B-aligned width
SORT_PAD = TOPK_PAD * 2 # (val, idx) interleaved slice width
assert TOPK <= TOPK_PAD

@pl.jit.inline
def _gate_core(
    x_mixed: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    active_tokens_arg: pl.Scalar[pl.INDEX],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor,
    x_norm: pl.Tensor,
    x_norm_i8: pl.Tensor,
    x_norm_scale: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
):
    token_dim = pl.tensor.dim(x_mixed, 0)
    token_pad = ((token_dim + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
    x_norm_gate_buf = pl.create_tensor([token_pad, D], dtype=pl.FP32)
    route_scores_buf = pl.create_tensor([token_pad, SCORE_PAD], dtype=pl.FP32)
    biased_scores_buf = pl.create_tensor([token_pad, SCORE_PAD], dtype=pl.FP32)
    active_tokens = active_tokens_arg
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > token_dim:
        active_tokens = token_dim
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    active_compute_tokens = active_gate_tiles * GATE_M_TILE
    if active_compute_tokens > token_dim:
        active_compute_tokens = token_dim
    active_norm_tokens = ((active_compute_tokens + T_TILE - 1) // T_TILE) * T_TILE
    norm_tiles = active_norm_tokens // T_TILE

    with pl.spmd(norm_tiles, name_hint="ffn_norm", allow_early_resolve=True) as norm_tid:
        t0 = pl.tile.get_block_idx() * T_TILE
        norm_rows = pl.min(T_TILE, token_dim - t0)
        norm_reduce_tmp = pl.create_tile([T_TILE, D_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        rms_x_bf16 = pl.load(
            x_mixed,
            [t0, 0],
            [T_TILE, D_TILE],
            valid_shapes=[norm_rows, D_TILE],
            target_memory=pl.MemorySpace.Vec,
        )
        rms_x = pl.cast(rms_x_bf16, pl.FP32)
        sq_sum = pl.row_sum(pl.mul(rms_x, rms_x), norm_reduce_tmp)
        for rms_db in pl.range(1, D // D_TILE):
            rms_d0 = rms_db * D_TILE
            rms_x_bf16 = pl.load(
                x_mixed,
                [t0, rms_d0],
                [T_TILE, D_TILE],
                valid_shapes=[norm_rows, D_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            rms_x = pl.cast(rms_x_bf16, pl.FP32)
            sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(rms_x, rms_x), norm_reduce_tmp))
        inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
        for an_d0 in pl.pipeline(0, D, D_TILE, stage=2):
            an_x_bf16 = pl.load(
                x_mixed,
                [t0, an_d0],
                [T_TILE, D_TILE],
                valid_shapes=[norm_rows, D_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            an_x = pl.cast(an_x_bf16, pl.FP32)
            an_w_input = pl.load(
                norm_w,
                [an_d0],
                [D_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            an_w = pl.cast(pl.reshape(an_w_input, [1, D_TILE]), pl.FP32)
            an_normed = pl.col_expand_mul(pl.row_expand_mul(an_x, inv_rms), an_w)
            an_bf16 = pl.cast(an_normed, pl.BF16, mode="rint")
            an_gate_f32 = pl.cast(an_bf16, pl.FP32)
            an_gate_out = pl.set_validshape(an_gate_f32, norm_rows, D_TILE)
            pl.store(an_gate_out, [t0, an_d0], x_norm_gate_buf)
            an_out = pl.set_validshape(an_bf16, norm_rows, D_TILE)
            pl.store(an_out, [t0, an_d0], x_norm)

    # Per-token symmetric INT8 quant of x_norm (read the bf16 output directly;
    # x_norm_gate_buf holds the same bf16 values widened to fp32).
    with pl.spmd(
        norm_tiles,
        name_hint="x_norm_quant",
        deps=[norm_tid],
        allow_early_resolve=True,
    ) as _quant_tid:
        t0 = pl.tile.get_block_idx() * T_TILE
        quant_rows = pl.min(T_TILE, token_dim - t0)
        quant_reduce_tmp = pl.create_tile(
            [T_TILE, QUANT_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
        )
        xn_a_f32 = pl.load(
            x_norm_gate_buf,
            [t0, 0],
            [T_TILE, QUANT_TILE],
            target_memory=pl.MemorySpace.Vec,
        )
        xn_amax = pl.row_max(pl.abs(xn_a_f32), quant_reduce_tmp)
        for xq_a_b in pl.range(1, D // QUANT_TILE):
            xq_a_k = xq_a_b * QUANT_TILE
            xn_a_f32 = pl.load(
                x_norm_gate_buf,
                [t0, xq_a_k],
                [T_TILE, QUANT_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            xn_a_max = pl.row_max(pl.abs(xn_a_f32), quant_reduce_tmp)
            xn_amax = pl.maximum(xn_amax, xn_a_max)
        xn_amax = pl.maximum(xn_amax, INT8_AMAX_EPS)
        xn_scale_quant = pl.mul(pl.recip(xn_amax), INT8_SCALE_MAX)
        xn_scale_dq = pl.mul(xn_amax, 1.0 / INT8_SCALE_MAX)
        xn_scale_out = pl.set_validshape(xn_scale_dq, quant_rows, 1)
        pl.store(xn_scale_out, [t0, 0], x_norm_scale)
        for xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
            xn_q_f32 = pl.load(
                x_norm_gate_buf,
                [t0, xq_b_k],
                [T_TILE, QUANT_TILE],
                target_memory=pl.MemorySpace.Vec,
            )
            xn_q_scaled = pl.row_expand_mul(
                xn_q_f32,
                xn_scale_quant,
            )
            xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
            xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
            xn_q_i8 = pl.cast(xn_q_half, pl.INT8, mode="trunc")
            xn_q_out = pl.set_validshape(xn_q_i8, quant_rows, QUANT_TILE)
            pl.store(xn_q_out, [t0, xq_b_k], x_norm_i8)

    # Pre-route setup: zero the inactive-token outputs.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
        for zt in pl.range(token_dim):
            if zt >= active_tokens:
                pl.write(x_norm_scale, [zt, 0], pl.cast(0.0, pl.FP32))
                for zk in pl.range(TOPK):
                    pl.write(indices, [zt, zk], pl.cast(0, pl.INT32))
                    pl.write(weights, [zt, zk], pl.cast(0.0, pl.FP32))

    if N_EXPERTS < SCORE_PAD and layer_id >= N_HASH_LAYERS:
        for pad_tg in pl.spmd(active_gate_tiles, name_hint="gate_score_pad"):
            pad_t0 = pad_tg * GATE_M_TILE
            biased_scores_buf[pad_t0 : pad_t0 + GATE_M_TILE, N_EXPERTS:SCORE_PAD] = pl.full(
                [GATE_M_TILE, SCORE_PAD - N_EXPERTS],
                dtype=pl.FP32,
                value=FP32_NEG_INF,
            )

    # Gate matmul + post: x_norm @ gate_w.T → sqrt(softplus(logits)) (+bias).
    # Fan the matmul over expert columns so each block computes a [GATE_M_TILE,
    # GATE_N_TILE] slice on its own core; token-tile is the dynamic dim, so it
    # stays outermost and // % divide by the compile-time GATE_N_BLOCKS.
    GATE_N_BLOCKS = N_EXPERTS // GATE_N_TILE
    with pl.spmd(
        active_gate_tiles * GATE_N_BLOCKS,
        name_hint="gate",
        deps=[norm_tid],
    ) as gate_tid:
        gb_idx = pl.tile.get_block_idx()
        tg = gb_idx // GATE_N_BLOCKS
        nb = gb_idx % GATE_N_BLOCKS
        t1 = tg * GATE_M_TILE
        n0 = nb * GATE_N_TILE
        gp_bias_row = pl.reshape(gate_bias[n0 : n0 + GATE_N_TILE], [1, GATE_N_TILE])
        gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
        for kb in pl.range(0, D // GATE_D_TILE):
            gd_kd = kb * GATE_D_TILE
            gd_x = x_norm_gate_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE]
            gd_w = gate_w[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
            if gd_kd == 0:
                gate_logits_tile = pl.matmul(gd_x, gd_w, out_dtype=pl.FP32, b_trans=True)
            else:
                gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True)
        gp_relu = pl.maximum(gate_logits_tile, 0.0)
        gp_abs = pl.maximum(gate_logits_tile, pl.neg(gate_logits_tile))
        gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
        gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(gate_logits_tile), 10.0), 0.0), 1.0)
        gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(gate_logits_tile, 0.0)))
        gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
        gp_score = pl.sqrt(gp_softplus)
        route_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_score
        if layer_id >= N_HASH_LAYERS:
            gp_bias = pl.col_expand_mul(pl.full([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32, value=1.0), gp_bias_row)
            gp_biased = pl.add(gp_score, gp_bias)
            biased_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_biased

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    # Hash layers index via tid2eid[input_ids]; score layers sort+gather.
    if layer_id < N_HASH_LAYERS:
        with pl.spmd(active_route_tiles, name_hint="route_hash", deps=[gate_tid]) as _route_tid:
            th_idx = pl.tile.get_block_idx()
            t1 = th_idx * GATE_T_TILE
            # Scalar gather TOPK (eid, unbiased score) per row; tail
            # [TOPK, TOPK_PAD) stays zero so row_sum below sums only TOPK.
            hs_vals_buf = pl.tile.full([GATE_T_TILE, TOPK_PAD], dtype=pl.FP32, value=0.0)
            hs_idx_buf = pl.tile.full([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32, value=0)
            hs_reduce_tmp = pl.create_tile(
                [GATE_T_TILE, TOPK_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            for hs_tt in pl.range(GATE_T_TILE):
                if t1 + hs_tt < active_tokens:
                    hs_token = pl.cast(pl.read(input_ids, [t1 + hs_tt]), pl.INDEX)
                    for hs_k in pl.range(TOPK):
                        hs_eid = pl.read(tid2eid, [hs_token, hs_k])
                        hs_epos = pl.cast(hs_eid, pl.INDEX)
                        hs_unbiased = pl.read(route_scores_buf, [t1 + hs_tt, hs_epos])
                        pl.write(hs_idx_buf, [hs_tt, hs_k], hs_eid)
                        pl.write(hs_vals_buf, [hs_tt, hs_k], hs_unbiased)
            # Normalize+scale, then scalar-scatter to GM. Slice-assign would
            # alloc a [GATE_T_TILE, TOPK=6] temp (24B row, under alloc_tile's
            # 32B alignment), so write element-by-element.
            hs_denom = pl.reshape(pl.row_sum(hs_vals_buf, hs_reduce_tmp), [GATE_T_TILE, 1])
            hs_weights_buf = pl.mul(pl.row_expand_div(hs_vals_buf, hs_denom), ROUTE_SCALE)
            for hs_wt_tt in pl.range(GATE_T_TILE):
                if t1 + hs_wt_tt < active_tokens:
                    for hs_wt_k in pl.range(TOPK):
                        pl.write(indices, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_idx_buf, [hs_wt_tt, hs_wt_k]))
                        pl.write(weights, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_weights_buf, [hs_wt_tt, hs_wt_k]))
    else:
        with pl.spmd(active_route_tiles, name_hint="route_sort", deps=[gate_tid]) as _route_tid:
            ts_idx = pl.tile.get_block_idx()
            t1 = ts_idx * GATE_T_TILE
            # topk_idx_tile stays Tensor (created here, not a pl.full Tile) so
            # the batched pl.gather below accepts it — Tile-against-Tensor src
            # is rejected.
            topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            # ptoas pto.tmrgsort requires src rows == 1; sort path iterates
            # row-by-row. sort32: [1,256]→[1,512] (8 runs of 64). mrgsort
            # format1 4-way: 8→2 runs of 256. format2 2-way: 2→1 run of 512.
            for sr_tt in pl.range(GATE_T_TILE):
                sr_row = biased_scores_buf[t1 + sr_tt : t1 + sr_tt + 1, :]
                sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                sr_sorted = pl.sort32(sr_row, sr_idx_init)
                sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
                sr_sorted = pl.mrgsort(sr_sorted[:, 0:256], sr_sorted[:, 256:512])
                sr_pairs = sr_sorted[:, 0:SORT_PAD]
                sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i
            # Batched gather; set_validshape+fillpad zeros the [TOPK, TOPK_PAD)
            # tail so the normalize sum below sees only real TOPK entries.
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            nm_reduce_tmp = pl.create_tile(
                [GATE_T_TILE, TOPK_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec
            )
            # Copy topk_idx_tile to dodge the tensor_view-vs-ptr SSA conflict
            # between sort's slice-assign and scalar pl.read (pypto #1493).
            topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            nm_denom = pl.reshape(pl.row_sum(topk_vals_pad, nm_reduce_tmp), [GATE_T_TILE, 1])
            nm_weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, nm_denom), ROUTE_SCALE)
            for nm_tt in pl.range(GATE_T_TILE):
                if t1 + nm_tt < active_tokens:
                    for nm_k in pl.range(TOPK):
                        pl.write(indices, [t1 + nm_tt, nm_k], pl.read(topk_idx_read, [nm_tt, nm_k]))
                        pl.write(weights, [t1 + nm_tt, nm_k], pl.read(nm_weights_pad, [nm_tt, nm_k]))

    return _quant_tid


@pl.jit.inline
def gate(
    x_mixed: pl.Tensor,
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor,
    x_norm: pl.Tensor,
    x_norm_i8: pl.Tensor,
    x_norm_scale: pl.Tensor,
    indices: pl.Tensor,
    weights: pl.Tensor,
):
    token_dim = pl.tensor.dim(x_mixed, 0)
    gate_done = _gate_core(
        x_mixed,
        norm_w,
        gate_w,
        gate_bias,
        layer_id,
        token_dim,
        tid2eid,
        input_ids,
        x_norm,
        x_norm_i8,
        x_norm_scale,
        indices,
        weights,
    )
    return gate_done


@pl.jit
def gate_test(
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    x_norm: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_norm_i8: pl.Out[pl.Tensor[[T_DYN, D], pl.INT8]],
    x_norm_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T_DYN, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T_DYN, TOPK], pl.FP32]],
):
    x_mixed.bind_dynamic(0, T_DYN)
    input_ids.bind_dynamic(0, T_DYN)
    x_norm.bind_dynamic(0, T_DYN)
    x_norm_i8.bind_dynamic(0, T_DYN)
    x_norm_scale.bind_dynamic(0, T_DYN)
    indices.bind_dynamic(0, T_DYN)
    weights.bind_dynamic(0, T_DYN)

    _gate_done = gate(
        x_mixed,
        norm_w, gate_w, gate_bias,
        layer_id,
        tid2eid, input_ids,
        x_norm, x_norm_i8, x_norm_scale, indices, weights,
    )
    return x_norm, x_norm_i8, x_norm_scale, indices, weights


def _per_token_int8_quant(x_bf16):
    import torch
    x_f32 = x_bf16.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = INT8_SCALE_MAX / amax
    scaled = x_f32 * scale_q
    x_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = (1.0 / scale_q).reshape(-1)  # [T]
    return x_i8, scale_dq


def golden_gate_core(tensors):
    import torch

    token_count = tensors["x_mixed"].shape[0]

    # FFN RMSNorm.
    x_f = tensors["x_mixed"].float().view(token_count, D)
    norm_w = tensors["norm_w"].float()
    sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)
    x_normalized = (x_f * inv_rms) * norm_w.view(1, D)
    x_flat = x_normalized.to(torch.bfloat16)

    # Per-token symmetric INT8 quant of bf16(x_norm).
    x_norm_i8, x_norm_scale = _per_token_int8_quant(x_flat)

    # Gate matmul + sqrtsoftplus router score. Use the log1p stable form, which
    # equals F.softplus while keeping the negative-logit tail alive: the naive
    # log(exp(-|x|)+1) rounds 1+tiny back to 1.0 in fp32 and zeros the tail for
    # logits below ~-16.
    gate_w = tensors["gate_w"].float()
    gate_bias = tensors["gate_bias"].float()
    logits = x_flat.float() @ gate_w.T
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.view(1, -1)

    # Choose TOPK ids: hash layers take ids from tid2eid[input_ids]; score
    # layers argsort biased (stable to match NPU sort32 deterministic order).
    layer_id = int(tensors["layer_id"])
    if layer_id < N_HASH_LAYERS:
        tid2eid = tensors["tid2eid"]
        input_ids = tensors["input_ids"]
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]

    # Gather unbiased scores, normalize over TOPK, scale by ROUTE_SCALE.
    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = (topk_vals / denom) * ROUTE_SCALE
    tensors["x_norm"][:] = x_flat
    tensors["x_norm_i8"][:] = x_norm_i8
    tensors["x_norm_scale"][:] = x_norm_scale.reshape(token_count, 1)
    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def build_tensor_specs(token_count=T, layer_id=0):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_mixed():
        # Mirror post-RMSNorm activation magnitude (~ N(0, 1)).
        return torch.randn(token_count, D)
    def init_norm_w():
        return torch.ones(D)
    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():
        return torch.randn(N_EXPERTS) * 0.1
    def init_tid2eid():
        return torch.randint(0, N_EXPERTS, (VOCAB, TOPK), dtype=torch.int32)
    def init_input_ids():
        return torch.randint(0, VOCAB, (token_count,), dtype=torch.int64)
    return [
        TensorSpec("x_mixed", [token_count, D], torch.bfloat16, init_value=init_x_mixed),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("gate_w", [N_EXPERTS, D], torch.float32, init_value=init_gate_w),
        TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=init_gate_bias),
        ScalarSpec("layer_id", torch.int32, layer_id),
        TensorSpec("tid2eid", [VOCAB, TOPK], torch.int32, init_value=init_tid2eid),
        TensorSpec("input_ids", [token_count], torch.int64, init_value=init_input_ids),
        TensorSpec("x_norm", [token_count, D], torch.bfloat16, is_output=True),
        TensorSpec("x_norm_i8", [token_count, D], torch.int8, is_output=True),
        TensorSpec("x_norm_scale", [token_count, 1], torch.float32, is_output=True),
        TensorSpec("indices", [token_count, TOPK], torch.int32, is_output=True),
        TensorSpec("weights", [token_count, TOPK], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, T, 17])
    parser.add_argument("--enable-l2-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    runtime_dir = None
    for token_count in args.tokens:
        if token_count <= 0:
            parser.error("--tokens values must be positive")
        print(f"--- gate_test tokens={token_count} ---")
        result = run_jit(
            fn=gate_test,
            specs=build_tensor_specs(token_count=token_count, layer_id=args.layer_id),
            golden_fn=golden_gate_core,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_norm": ratio_allclose(atol=1e-3, rtol=1.0 / 128),
                "x_norm_i8": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.001),
                "indices": topk_pair_compare("weights"),
            },
            runtime_dir=runtime_dir,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
        runtime_dir = str(result.work_dir)
