# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 MoE FFN router (decode): RMSNorm + gate + topk + normalize."""


import pypto.language as pl

from config import (FLASH as M, MOE_TOKENS, FP32_NEG_INF,
                    INT8_SCALE_MAX, INT8_AMAX_EPS)


# model config
T = MOE_TOKENS
D = M.hidden_size
NORM_EPS = M.rms_norm_eps
# Routing space: every rank routes over the full global expert set so dispatch
# can fan tokens across ranks. moe.py sets config.FLASH.n_routed_experts to
# a fixed per-rank count * EP before importing this module (256 at EP8).
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
VOCAB = M.vocab_size
N_HASH_LAYERS = M.num_hash_layers

# tiling
NORM_TOKEN_TILE = 4
GATE_T_TILE = 8
GATE_M_TILE = min(64, T)  # Fully initialized token tiles at every supported TP.
GATE_N_TILE = 32        # expert columns per gate spmd block
assert N_EXPERTS % GATE_N_TILE == 0
T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
D_TILE = 256
ROW_PAD = 8
FFN_REDUCE_TILE = D // ROW_PAD
assert D % ROW_PAD == 0
GATE_D_TILE = 512
assert D % GATE_D_TILE == 0, "gate K-loop must cover D"
SCORE_PAD = 256         # padded expert row for sort32 + mrgsort
TOPK_PAD = 8            # TOPK padded to 32B-aligned width
SORT_PAD = TOPK_PAD * 2 # (val, idx) interleaved slice width
assert TOPK <= TOPK_PAD

def _gate(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    x_norm_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
    x_norm_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    # Deferred RMSNorm (qwen3-style): store xg = x*gamma (NOT *inv_rms), because
    # the per-token positive scalar inv_rms factors out of everything downstream:
    #   - gate logits: inv_rms * (xg @ gate_w.T)  -> applied as a [16,1] row-scale
    #   - int8 quant : symmetric per-token quant of xg CANCELS inv_rms exactly;
    #                  inv_rms rides only x_norm_scale (= inv_rms * amax(xg)/127).
    # The early xg producer overlaps projection with RMS statistics and INT8
    # quantization, which read the stored FP32 product independently.
    xg_buf = pl.create_tensor([T_PAD, D], dtype=pl.FP32)
    inv_rms_buf = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    gate_logits_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    # Keep one producer tile alive for an owner with zero local tokens.  The
    # following pre-route task overwrites the whole fixed-capacity output with
    # zeroes, while the rank still publishes zero route counts and participates
    # in the EP barriers.  A zero-sized producer loop leaves x_norm_i8 without a
    # producer and can stall its fixed-shape shared-expert consumer.
    if active_gate_tiles < 1:
        active_gate_tiles = pl.cast(1, pl.INDEX)
    active_gate_tokens = active_gate_tiles * GATE_M_TILE
    if active_gate_tokens > T:
        active_gate_tokens = pl.cast(T, pl.INDEX)

    # Each block normalizes a group of tokens with full-row reductions.
    norm_w_2d = pl.reshape(norm_w, [1, D])
    norm_blocks = (active_gate_tokens + NORM_TOKEN_TILE - 1) // NORM_TOKEN_TILE
    for xg_block in pl.spmd(norm_blocks, name_hint="ffn_xg", allow_early_resolve=True):
        xg_tok0 = xg_block * NORM_TOKEN_TILE
        xg_prod_w = pl.cast(pl.tile.load(norm_w_2d, [0, 0], [1, D]), pl.FP32)
        xg_prod_input = pl.tile.load(x_mixed, [xg_tok0, 0], [NORM_TOKEN_TILE, D])
        xg_prod_x = pl.cast(xg_prod_input, pl.FP32)
        xg_prod = pl.col_expand_mul(xg_prod_x, xg_prod_w)
        pl.tile.store(xg_prod, [xg_tok0, 0], xg_buf, shapes=[NORM_TOKEN_TILE, D])

    for tok_block in pl.spmd(norm_blocks, name_hint="ffn_norm", allow_early_resolve=True):
        tok0 = tok_block * NORM_TOKEN_TILE
        for tok in pl.range(tok0, pl.min(tok0 + NORM_TOKEN_TILE, active_gate_tokens)):
            rms_x = pl.cast(pl.tile.load(x_mixed, [tok, 0], [1, D]), pl.FP32)
            xg = pl.tile.load(xg_buf, [tok, 0], [1, D])

            sq_rows = pl.reshape(pl.mul(rms_x, rms_x), [ROW_PAD, FFN_REDUCE_TILE])
            sq_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            sq_partial = pl.row_sum(sq_rows, sq_partial_tmp)
            sq_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_reduce[0:1, :] = pl.reshape(sq_partial, [1, ROW_PAD])
            sq_reduce = pl.set_validshape(sq_reduce, 1, ROW_PAD)
            sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_sum = pl.row_sum(sq_reduce, sq_sum_tmp)
            sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
            inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
            pl.tile.store(inv_rms, [tok, 0], inv_rms_buf, shapes=[1, 1])

            xg_abs_rows = pl.reshape(pl.abs(xg), [ROW_PAD, FFN_REDUCE_TILE])
            amax_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            amax_partial = pl.row_max(xg_abs_rows, amax_partial_tmp)
            amax_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            amax_reduce[0:1, :] = pl.reshape(amax_partial, [1, ROW_PAD])
            amax_reduce = pl.set_validshape(amax_reduce, 1, ROW_PAD)
            amax_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            xg_amax = pl.row_max(amax_reduce, amax_tmp)
            xg_amax = pl.set_validshape(pl.reshape(xg_amax, [1, ROW_PAD]), 1, 1)
            amax_eps = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_AMAX_EPS)
            amax_eps = pl.set_validshape(amax_eps, 1, 1)
            xg_amax = pl.maximum(xg_amax, amax_eps)
            # quant scale = INT8_SCALE_MAX / amax(xg); dequant scale rides inv_rms.
            scale_max = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
            scale_max = pl.set_validshape(scale_max, 1, 1)
            xg_sq = pl.div(scale_max, xg_amax)
            xg_dequant_scale = pl.mul(xg_amax, 1.0 / INT8_SCALE_MAX)
            x_norm_dequant_scale = pl.mul(xg_dequant_scale, inv_rms)
            pl.tile.store(x_norm_dequant_scale, [tok, 0], x_norm_scale, shapes=[1, 1])
            # Per-token symmetric INT8 quantization of xg.
            xg_sq_scalar = pl.read(xg_sq, [0, 0])
            xn_q_scaled = pl.mul(xg, xg_sq_scalar)
            xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
            xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
            xn_q_i8 = pl.cast(xn_q_half, pl.INT8, mode="trunc")
            pl.tile.store(xn_q_i8, [tok, 0], x_norm_i8, shapes=[1, D])

    # Zero inactive fixed-tile inputs/outputs and mask padded expert scores.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
        for zt in pl.range(T):
            if zt >= active_tokens:
                # Positive FP16 zero has zero bytes; no numeric conversion is needed.
                inactive_x_norm_f16 = pl.full([1, D // 2], dtype=pl.FP16, value=0.0)
                inactive_x_norm_i8 = pl.reinterpret_view(inactive_x_norm_f16, pl.INT8)
                x_norm_i8[zt : zt + 1, :] = inactive_x_norm_i8
                pl.write(x_norm_scale, [zt, 0], pl.cast(0.0, pl.FP32))
                for zk in pl.range(TOPK):
                    pl.write(indices, [zt, zk], pl.cast(0, pl.INT32))
                    pl.write(weights, [zt, zk], pl.cast(0.0, pl.FP32))
    # Gate projection over expert columns.
    # Fan the matmul over expert columns so each block computes a [GATE_M_TILE,
    # GATE_N_TILE] slice on its own core; token-tile is the dynamic dim, so it
    # stays outermost and // % divide by the compile-time GATE_N_BLOCKS.
    GATE_N_BLOCKS = N_EXPERTS // GATE_N_TILE
    for gb_idx in pl.spmd(active_gate_tiles * GATE_N_BLOCKS, name_hint="gate", allow_early_resolve=True):
        tg = gb_idx // GATE_N_BLOCKS
        nb = gb_idx % GATE_N_BLOCKS
        t1 = tg * GATE_M_TILE
        n0 = nb * GATE_N_TILE
        gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // GATE_D_TILE, stage=2):
            gd_kd = kb * GATE_D_TILE
            gd_x = xg_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE]
            gd_w = gate_w[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
            gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True, init_cond=(gd_kd == 0))
        gate_logits_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gate_logits_tile

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    # PyPTO cannot launch an SPMD task with zero blocks.  The guarded store in
    # either routing path keeps this dummy tile from publishing any route.
    if active_route_tiles < 1:
        active_route_tiles = pl.cast(1, pl.INDEX)
    for route_idx in pl.spmd(active_route_tiles, name_hint="route", allow_early_resolve=True):
        gp_t0 = route_idx * GATE_T_TILE
        t1 = gp_t0
        gp_logits = gate_logits_buf[gp_t0 : gp_t0 + GATE_T_TILE, 0:N_EXPERTS]
        gp_logits = pl.row_expand_mul(gp_logits, inv_rms_buf[gp_t0 : gp_t0 + GATE_T_TILE, 0:1])
        gp_relu = pl.maximum(gp_logits, 0.0)
        gp_abs = pl.maximum(gp_logits, pl.neg(gp_logits))
        gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
        gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(gp_logits), 10.0), 0.0), 1.0)
        gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(gp_logits, 0.0)))
        gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
        gp_score = pl.sqrt(gp_softplus)
        local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
        local_scores[:, :] = pl.full([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32, value=0.0)
        local_scores[:, 0:N_EXPERTS] = gp_score
        # Hash layers index via tid2eid[input_ids]; score layers sort+gather.
        if layer_id < N_HASH_LAYERS:
            # eids from tid2eid[input_ids]: scalar lookup (dynamic row index) into a
            # Tensor. tid2eid row load hits the 32B tile floor (TOPK*4=24B), so the
            # index build stays scalar; the score fetch is a batched gather below.
            # Tail [TOPK, TOPK_PAD) zeroed so fillpad drops it from the sum.
            hs_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            for hs_tt in pl.range(GATE_T_TILE):
                hs_token = pl.cast(pl.read(input_ids, [t1 + hs_tt]), pl.INDEX)
                for hs_k in pl.range(TOPK):
                    pl.write(hs_idx_tile, [hs_tt, hs_k], pl.read(tid2eid, [hs_token, hs_k]))
                for hs_pad_k in pl.range(TOPK, TOPK_PAD):
                    pl.write(hs_idx_tile, [hs_tt, hs_pad_k], pl.cast(0, pl.INT32))
            # Batched score gather (replaces per-eid scalar reads); set_validshape +
            # fillpad zero the [TOPK, TOPK_PAD) tail so row_sum sees only TOPK.
            gather_all = pl.gather(local_scores, dim=-1, index=hs_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            hs_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            # Copy to dodge the tensor_view-vs-ptr SSA conflict between the gather
            # and the scalar pl.read below (pypto #1493).
            hs_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            hs_idx_read[:, :] = hs_idx_tile[:, :]
            hs_denom = pl.reshape(pl.row_sum(hs_vals_pad), [GATE_T_TILE, 1])
            hs_weights_pad = pl.mul(pl.row_expand_div(hs_vals_pad, hs_denom), ROUTE_SCALE)
            for hs_wt_tt in pl.range(GATE_T_TILE):
                if t1 + hs_wt_tt < active_tokens:
                    for hs_wt_k in pl.range(TOPK):
                        pl.write(indices, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_idx_read, [hs_wt_tt, hs_wt_k]))
                        pl.write(weights, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_weights_pad, [hs_wt_tt, hs_wt_k]))
        else:
            gp_bias_row = pl.reshape(gate_bias, [1, N_EXPERTS])
            gp_ones = pl.full([GATE_T_TILE, N_EXPERTS], dtype=pl.FP32, value=1.0)
            gp_bias = pl.col_expand_mul(gp_ones, gp_bias_row)
            gp_biased = pl.add(gp_score, gp_bias)
            sort_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            sort_scores[:, :] = pl.full([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
            sort_scores[:, 0:N_EXPERTS] = gp_biased
            # topk_idx_tile stays Tensor (created here, not a pl.full Tile) so
            # the batched pl.gather below accepts it — Tile-against-Tensor src
            # is rejected.
            topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            # ptoas pto.tmrgsort requires src rows == 1; sort path iterates
            # row-by-row. sort32: [1,256]→[1,512] (8 runs of 64). mrgsort
            # format1 4-way: 8→2 runs of 256. format2 2-way: 2→1 run of 512.
            for sr_tt in pl.range(GATE_T_TILE):
                sr_row = sort_scores[sr_tt : sr_tt + 1, :]
                sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                sr_sorted = pl.sort32(sr_row, sr_idx_init)
                sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
                sr_sorted = pl.mrgsort(sr_sorted[:, 0:SORT_PAD], sr_sorted[:, 256:256 + SORT_PAD])
                sr_pairs = sr_sorted[:, 0:SORT_PAD]
                sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i
            # Batched gather; set_validshape+fillpad zeros the [TOPK, TOPK_PAD)
            # tail so the normalize sum below sees only real TOPK entries.
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            # Copy topk_idx_tile to dodge the tensor_view-vs-ptr SSA conflict
            # between sort's slice-assign and scalar pl.read (pypto #1493).
            topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            nm_denom = pl.reshape(pl.row_sum(topk_vals_pad), [GATE_T_TILE, 1])
            nm_weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, nm_denom), ROUTE_SCALE)
            for nm_tt in pl.range(GATE_T_TILE):
                if t1 + nm_tt < active_tokens:
                    for nm_k in pl.range(TOPK):
                        pl.write(indices, [t1 + nm_tt, nm_k], pl.read(topk_idx_read, [nm_tt, nm_k]))
                        pl.write(weights, [t1 + nm_tt, nm_k], pl.read(nm_weights_pad, [nm_tt, nm_k]))

    # The @pl.inline parser requires inline call expressions to have a return
    # value. weights is convenient because it's already pl.Out and reads as
    # the same SSA name on the caller side.
    return weights


gate = pl.jit.inline(_gate)
gate_test = pl.jit(_gate)


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

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    # FFN RMSNorm, deferred (qwen3-style): xg = bf16(x * gamma), with the
    # per-token inv_rms scalar folded downstream instead of applied per-element.
    x_f = tensors["x_mixed"].float().view(T, D)
    norm_w = tensors["norm_w"].float()
    sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)   # [T,1]
    xg = x_f * norm_w.view(1, D)

    # Symmetric INT8 quant of xg: inv_rms cancels in the int8 values (a positive
    # per-token scalar), so it rides only the dequant scale.
    x_norm_i8, scale_dq_g = _per_token_int8_quant(xg)
    x_norm_scale = scale_dq_g.reshape(T, 1) * inv_rms   # inv_rms * amax(xg)/127

    # Gate matmul + sqrtsoftplus router score. logits = inv_rms * (xg @ gate_w.T).
    # Use the log1p stable form, which equals F.softplus while keeping the
    # negative-logit tail alive: the naive log(exp(-|x|)+1) rounds 1+tiny back to
    # 1.0 in fp32 and zeros the tail for logits below ~-16.
    gate_w = tensors["gate_w"].float()
    gate_bias = tensors["gate_bias"].float()
    logits = inv_rms * (xg.float() @ gate_w.T)
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
    if num_tokens < T:
        x_norm_i8[num_tokens:] = 0
        x_norm_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0

    tensors["x_norm_i8"][:] = x_norm_i8
    tensors["x_norm_scale"][:] = x_norm_scale.reshape(T, 1)
    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def build_tensor_specs(layer_id=0, num_tokens=T):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_mixed():
        # Mirror post-RMSNorm activation magnitude (~ N(0, 1)).
        return torch.randn(T, D)
    def init_norm_w():
        return torch.ones(D)
    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():
        return torch.randn(N_EXPERTS) * 0.1
    def init_tid2eid():
        return torch.randint(0, N_EXPERTS, (VOCAB, TOPK), dtype=torch.int32)
    def init_input_ids():
        return torch.randint(0, VOCAB, (T,), dtype=torch.int64)
    return [
        TensorSpec("x_mixed", [T, D], torch.bfloat16, init_value=init_x_mixed),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=init_norm_w),
        TensorSpec("gate_w", [N_EXPERTS, D], torch.float32, init_value=init_gate_w),
        TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=init_gate_bias),
        ScalarSpec("layer_id", torch.int32, layer_id),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
        TensorSpec("tid2eid", [VOCAB, TOPK], torch.int32, init_value=init_tid2eid),
        TensorSpec("input_ids", [T], torch.int64, init_value=init_input_ids),
        TensorSpec("x_norm_i8", [T, D], torch.int8),
        TensorSpec("x_norm_scale", [T, 1], torch.float32),
        TensorSpec("indices", [T, TOPK], torch.int32),
        TensorSpec("weights", [T, TOPK], torch.float32),
    ]


def gate_active_rows(num_tokens):
    """Active token count rounded up to the gate M-tile, capped at T."""
    active_count = max(0, min(T, int(num_tokens)))
    return min(T, ((active_count + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE)


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=gate_test,
        specs=build_tensor_specs(layer_id=args.layer_id, num_tokens=args.num_tokens),
        golden_fn=golden_gate_core,
        config=dict(
            dump_passes=args.dump_passes,
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_norm_i8": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.001,
                                        valid_rows=gate_active_rows(args.num_tokens)),
            "indices": topk_pair_compare("weights"),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
