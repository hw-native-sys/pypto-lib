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

from config import ACTIVE as M, FP32_NEG_INF, INT8_AMAX_EPS, INT8_SCALE_MAX, MOE_TOKENS


# model config
T = MOE_TOKENS
D = M.hidden_size
NORM_EPS = M.rms_norm_eps
N_EXPERTS = M.n_routed_experts
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
VOCAB = M.vocab_size
N_HASH_LAYERS = M.num_hash_layers

# tiling
T_TILE = 8
GATE_T_TILE = 8
assert T % GATE_T_TILE == 0
GATE_M_TILE = 16
GATE_N_TILE = 16
T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE
ROW_TILE = 8
FFN_REDUCE_TILE = D // ROW_TILE
GATE_D_TILE = 2048 if M.name == "flash" else 512
assert (D // GATE_D_TILE) % 2 == 0, "gate K-loop trip count must be even (A5 accumulator-buffer constraint)"
QUANT_TILE = 256
SCORE_PAD = 256 if M.name == "flash" else 384
TOPK_PAD = 8
SORT_PAD = TOPK_PAD * 2


if M.name == "flash":
    @pl.jit.inline
    def route_topk_row(
        score_row: pl.Tensor[[1, 256], pl.FP32],
        idx_init: pl.Tensor[[1, 256], pl.UINT32],
    ) -> pl.Tensor[[1, TOPK_PAD], pl.INT32]:
        """Sort one Flash router row and return its leading expert ids."""

        sorted_32 = pl.sort32(score_row, idx_init)
        sorted_64 = pl.mrgsort(sorted_32, block_len=64)
        sorted_pairs = pl.mrgsort(sorted_64[:, 0:256], sorted_64[:, 256:512])
        return pl.gather(
            sorted_pairs[:, 0:SORT_PAD],
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
else:
    @pl.jit.inline
    def route_topk_row(
        score_row: pl.Tensor[[1, 384], pl.FP32],
        idx_init: pl.Tensor[[1, 384], pl.UINT32],
    ) -> pl.Tensor[[1, TOPK_PAD], pl.INT32]:
        """Sort one Pro router row and return its leading expert ids."""

        sorted_32 = pl.sort32(score_row, idx_init)
        sorted_64 = pl.mrgsort(sorted_32, block_len=64)
        sorted_pairs = pl.mrgsort(
            sorted_64[:, 0:256],
            sorted_64[:, 256:512],
            sorted_64[:, 512:768],
        )
        return pl.gather(
            sorted_pairs[:, 0:SORT_PAD],
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )


@pl.jit.inline
def gate(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
):
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    active_gate_tokens = active_gate_tiles * GATE_M_TILE
    if active_gate_tokens > T:
        active_gate_tokens = pl.cast(T, pl.INDEX)

    norm_w_2d = pl.reshape(norm_w, [1, D])
    xg_buf = pl.create_tensor([T_PAD, D], dtype=pl.FP32)
    inv_rms_buf = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    xn_scale_buf = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    for tok in pl.spmd(active_gate_tokens, name_hint="ffn_norm"):
        rms_x_bf16 = pl.tile.load(x_mixed, [tok, 0], [1, D])
        rms_x = pl.cast(rms_x_bf16, pl.FP32)
        rms_w_bf16 = pl.tile.load(norm_w_2d, [0, 0], [1, D])
        rms_w = pl.cast(rms_w_bf16, pl.FP32)
        xg = pl.mul(rms_x, rms_w)
        pl.tile.store(xg, [tok, 0], xg_buf, shapes=[1, D])

        rms_sq = pl.mul(rms_x, rms_x)
        sq_rows = pl.reshape(rms_sq, [ROW_TILE, FFN_REDUCE_TILE])
        sq_partial_tmp = pl.create_tile([ROW_TILE, FFN_REDUCE_TILE], dtype=pl.FP32)
        sq_partial = pl.row_sum(sq_rows, sq_partial_tmp)
        sq_reduce_tile = pl.create_tile([ROW_TILE, ROW_TILE], dtype=pl.FP32)
        sq_partial_row = pl.reshape(sq_partial, [1, ROW_TILE])
        sq_reduce_tile[0:1, :] = sq_partial_row
        sq_reduce_valid = pl.set_validshape(sq_reduce_tile, 1, ROW_TILE)
        sq_sum_tmp = pl.create_tile([ROW_TILE, ROW_TILE], dtype=pl.FP32)
        sq_sum_raw = pl.row_sum(sq_reduce_valid, sq_sum_tmp)
        sq_sum_row = pl.reshape(sq_sum_raw, [1, ROW_TILE])
        sq_sum_valid = pl.set_validshape(sq_sum_row, 1, 1)
        sq_mean = pl.mul(sq_sum_valid, 1.0 / D)
        rms_arg = pl.add(sq_mean, NORM_EPS)
        rms_root = pl.sqrt(rms_arg)
        inv_rms = pl.recip(rms_root)
        pl.tile.store(inv_rms, [tok, 0], inv_rms_buf, shapes=[1, 1])

        xg_abs = pl.abs(xg)
        xg_abs_rows = pl.reshape(xg_abs, [ROW_TILE, FFN_REDUCE_TILE])
        amax_partial_tmp = pl.create_tile([ROW_TILE, FFN_REDUCE_TILE], dtype=pl.FP32)
        amax_partial = pl.row_max(xg_abs_rows, amax_partial_tmp)
        amax_reduce_tile = pl.create_tile([ROW_TILE, ROW_TILE], dtype=pl.FP32)
        amax_partial_row = pl.reshape(amax_partial, [1, ROW_TILE])
        amax_reduce_tile[0:1, :] = amax_partial_row
        amax_reduce_valid = pl.set_validshape(amax_reduce_tile, 1, ROW_TILE)
        amax_tmp = pl.create_tile([ROW_TILE, ROW_TILE], dtype=pl.FP32)
        xg_amax_raw = pl.row_max(amax_reduce_valid, amax_tmp)
        xg_amax_row = pl.reshape(xg_amax_raw, [1, ROW_TILE])
        xg_amax_valid = pl.set_validshape(xg_amax_row, 1, 1)
        amax_eps_tile = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        amax_eps_valid = pl.set_validshape(amax_eps_tile, 1, 1)
        xg_amax = pl.maximum(xg_amax_valid, amax_eps_valid)
        scale_max_tile = pl.tile.full([1, ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        scale_max = pl.set_validshape(scale_max_tile, 1, 1)
        xg_sq = pl.div(scale_max, xg_amax)
        xg_dequant_scale = pl.mul(xg_amax, 1.0 / INT8_SCALE_MAX)
        x_norm_dequant_scale = pl.mul(xg_dequant_scale, inv_rms)
        pl.tile.store(x_norm_dequant_scale, [tok, 0], x_norm_scale, shapes=[1, 1])
        pl.tile.store(xg_sq, [tok, 0], xn_scale_buf, shapes=[1, 1])

    for t0 in pl.parallel(0, active_gate_tokens, T_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="x_norm_quant"):
            xn_sq_col = xn_scale_buf[t0 : t0 + T_TILE, 0:1]
            for xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
                xn_q_chunk = xg_buf[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_TILE]
                xn_q_scaled = pl.row_expand_mul(xn_q_chunk, xn_sq_col)
                xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
                xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
                xn_q_i8 = pl.cast(xn_q_half, pl.INT8, mode="trunc")
                x_norm_i8[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_TILE] = xn_q_i8

    biased_scores_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="gate_pre_route"):
        for zt in pl.range(T):
            if zt >= active_tokens:
                zero_scale = pl.cast(0.0, pl.FP32)
                pl.write(x_norm_scale, [zt, 0], zero_scale)
                for zk in pl.range(TOPK):
                    zero_index = pl.cast(0, pl.INT32)
                    pl.write(indices, [zt, zk], zero_index)
                    zero_weight = pl.cast(0.0, pl.FP32)
                    pl.write(weights, [zt, zk], zero_weight)
        if N_EXPERTS < SCORE_PAD:
            biased_pad = pl.full([T_PAD, SCORE_PAD - N_EXPERTS], dtype=pl.FP32, value=FP32_NEG_INF)
            biased_scores_buf[:, N_EXPERTS:SCORE_PAD] = biased_pad

    route_scores_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    for gb_idx in pl.spmd(active_gate_tiles * (N_EXPERTS // GATE_N_TILE), name_hint="gate"):
        tg = gb_idx // (N_EXPERTS // GATE_N_TILE)
        nb = gb_idx % (N_EXPERTS // GATE_N_TILE)
        t1 = tg * GATE_M_TILE
        n0 = nb * GATE_N_TILE
        gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // GATE_D_TILE, stage=2):
            gd_kd = kb * GATE_D_TILE
            gd_x = xg_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE]
            gd_w = gate_w[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
            if gd_kd == 0:
                gate_logits_tile = pl.matmul(gd_x, gd_w, out_dtype=pl.FP32, b_trans=True)
            else:
                gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True)
        inv_rms_tile = inv_rms_buf[t1 : t1 + GATE_M_TILE, 0:1]
        gate_logits_tile = pl.row_expand_mul(gate_logits_tile, inv_rms_tile)
        gp_relu = pl.maximum(gate_logits_tile, 0.0)
        gp_abs = pl.abs(gate_logits_tile)
        gp_neg_abs = pl.neg(gp_abs)
        gp_exp_abs = pl.exp(gp_neg_abs)
        gp_exp_plus = pl.add(gp_exp_abs, 1.0)
        gp_softplus_tail = pl.log(gp_exp_plus)
        gp_softplus_log = pl.add(gp_relu, gp_softplus_tail)
        gp_neg_logits = pl.neg(gate_logits_tile)
        gp_neg_shift = pl.sub(gp_neg_logits, 10.0)
        gp_neg_mask_floor = pl.maximum(gp_neg_shift, 0.0)
        gp_neg_floor_mask = pl.minimum(gp_neg_mask_floor, 1.0)
        gp_logits_floor = pl.minimum(gate_logits_tile, 0.0)
        gp_neg_exp = pl.exp(gp_logits_floor)
        gp_neg_floor = pl.mul(gp_neg_floor_mask, gp_neg_exp)
        gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
        gp_score = pl.sqrt(gp_softplus)
        route_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_score
        gp_bias_row = pl.reshape(gate_bias[n0 : n0 + GATE_N_TILE], [1, GATE_N_TILE])
        if layer_id >= N_HASH_LAYERS:
            gp_biased = pl.col_expand_add(gp_score, gp_bias_row)
            biased_scores_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gp_biased

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    if layer_id < N_HASH_LAYERS:
        for th_idx in pl.spmd(active_route_tiles, name_hint="route_hash"):
            t1 = th_idx * GATE_T_TILE
            hs_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            for hs_tt in pl.range(GATE_T_TILE):
                hs_input_id = pl.read(input_ids, [t1 + hs_tt])
                hs_token = pl.cast(hs_input_id, pl.INDEX)
                for hs_k in pl.range(TOPK):
                    hs_eid = pl.read(tid2eid, [hs_token, hs_k])
                    pl.write(hs_idx_tile, [hs_tt, hs_k], hs_eid)
                for hs_pad_k in pl.range(TOPK, TOPK_PAD):
                    hs_pad = pl.cast(0, pl.INT32)
                    pl.write(hs_idx_tile, [hs_tt, hs_pad_k], hs_pad)
            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=hs_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            hs_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            hs_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            hs_idx_read[:, :] = hs_idx_tile[:, :]
            hs_sum = pl.row_sum(hs_vals_pad)
            hs_denom = pl.reshape(hs_sum, [GATE_T_TILE, 1])
            hs_normalized = pl.row_expand_div(hs_vals_pad, hs_denom)
            hs_weights_pad = pl.mul(hs_normalized, ROUTE_SCALE)
            for hs_wt_tt in pl.range(GATE_T_TILE):
                hs_out_t = t1 + hs_wt_tt
                if hs_out_t < active_tokens:
                    for hs_wt_k in pl.range(TOPK):
                        hs_out_idx = pl.read(hs_idx_read, [hs_wt_tt, hs_wt_k])
                        pl.write(indices, [hs_out_t, hs_wt_k], hs_out_idx)
                        hs_out_weight = pl.read(hs_weights_pad, [hs_wt_tt, hs_wt_k])
                        pl.write(weights, [hs_out_t, hs_wt_k], hs_out_weight)
    else:
        for ts_idx in pl.spmd(active_route_tiles, name_hint="route_sort"):
            t1 = ts_idx * GATE_T_TILE
            # ptoas pto.tmrgsort requires a single source row.
            topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            sr_idx_init = pl.create_tensor([1, SCORE_PAD], dtype=pl.UINT32)
            sr_index_range = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
            sr_idx_init[:, :] = sr_index_range
            for sr_tt in pl.range(GATE_T_TILE):
                sr_t = t1 + sr_tt
                sr_row = pl.slice(biased_scores_buf, [1, SCORE_PAD], [sr_t, 0])
                sr_i = route_topk_row(sr_row, sr_idx_init)
                topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i

            local_scores = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
            local_scores[:, :] = route_scores_buf[t1 : t1 + GATE_T_TILE, :]
            gather_all = pl.gather(local_scores, dim=-1, index=topk_idx_tile)
            gather_valid = pl.set_validshape(gather_all, GATE_T_TILE, TOPK)
            topk_vals_pad = pl.fillpad(gather_valid, pad_value=pl.PadValue.zero)
            topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            topk_sum = pl.row_sum(topk_vals_pad)
            denom = pl.reshape(topk_sum, [GATE_T_TILE, 1])
            topk_normalized = pl.row_expand_div(topk_vals_pad, denom)
            normalized_weights = pl.mul(topk_normalized, ROUTE_SCALE)
            for wt_tt in pl.range(GATE_T_TILE):
                wt_out_t = t1 + wt_tt
                if wt_out_t < active_tokens:
                    for wt_k in pl.range(TOPK):
                        wt_out_idx = pl.read(topk_idx_read, [wt_tt, wt_k])
                        pl.write(indices, [wt_out_t, wt_k], wt_out_idx)
                        wt_out_weight = pl.read(normalized_weights, [wt_tt, wt_k])
                        pl.write(weights, [wt_out_t, wt_k], wt_out_weight)

    return weights


@pl.jit
def gate_test(
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
    gate(
        x_mixed,
        norm_w, gate_w, gate_bias,
        layer_id, num_tokens,
        tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )
    return x_norm_i8, x_norm_scale, indices, weights


def _per_token_int8_quant(x_bf16):
    import torch
    x_f32 = x_bf16.float()
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_q = torch.full_like(amax, INT8_SCALE_MAX) / amax
    scaled = x_f32 * scale_q
    x_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dq = (1.0 / scale_q).reshape(-1)  # [T]
    return x_i8, scale_dq


def _golden_gate_scores(tensors):
    """Recompute the host router scores from the immutable gate inputs."""
    import torch

    x_f = tensors["x_mixed"].cpu().float().view(T, D)
    norm_w = tensors["norm_w"].cpu().float()
    sq_rows = (x_f * x_f).reshape(T, ROW_TILE, FFN_REDUCE_TILE)
    sq_partial = sq_rows.sum(dim=-1)
    sq_sum = sq_partial.sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)
    xg = x_f * norm_w.view(1, D)

    gate_w = tensors["gate_w"].cpu().float()
    gate_bias = tensors["gate_bias"].cpu().float()
    logits_acc = xg @ gate_w.T
    logits = inv_rms * logits_acc
    softplus = logits.clamp(min=0) + torch.log1p(torch.exp(-logits.abs()))
    scores = softplus.sqrt()
    biased = scores + gate_bias.view(1, -1)
    return xg, inv_rms, scores, biased


def golden_gate_core(tensors):
    import torch

    num_tokens = max(0, min(T, int(tensors.get("num_tokens", T))))

    xg, inv_rms, scores, biased = _golden_gate_scores(tensors)

    x_norm_i8, scale_dq_g = _per_token_int8_quant(xg)
    x_norm_scale = scale_dq_g.reshape(T, 1) * inv_rms

    layer_id = int(tensors["layer_id"])
    if layer_id < N_HASH_LAYERS:
        tid2eid = tensors["tid2eid"]
        input_ids = tensors["input_ids"]
        indices = tid2eid[input_ids.flatten().long()]
    else:
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]

    topk_vals = torch.gather(scores, dim=-1, index=indices.long())
    denom = topk_vals.sum(dim=-1, keepdim=True)
    weights = (topk_vals / denom) * ROUTE_SCALE
    if num_tokens < T:
        x_norm_scale[num_tokens:] = 0
        indices[num_tokens:] = 0
        weights[num_tokens:] = 0

    tensors["x_norm_i8"][:] = x_norm_i8
    tensors["x_norm_scale"][:] = x_norm_scale.reshape(T, 1)
    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def gate_indices_compare(
    layer_id,
    num_tokens,
    *,
    score_atol=1e-4,
    score_rtol=2e-5,
    max_show=10,
):
    """Validate router ids against exact hash routes or bounded score routes."""
    import torch

    active_tokens = max(0, min(T, int(num_tokens)))

    def cmp(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, rtol, atol
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape:
            return False, (
                f"    index shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )

        inactive = actual[active_tokens:]
        inactive_nonzero = int(inactive.count_nonzero().item())
        if inactive_nonzero:
            return False, (
                f"    inactive index tail contains {inactive_nonzero} nonzero values"
            )

        actual = actual[:active_tokens].to(torch.int64)
        expected = expected[:active_tokens].to(torch.int64)
        if actual.numel() == 0:
            return True, ""

        mismatch = actual != expected
        if int(layer_id) < N_HASH_LAYERS:
            if not mismatch.any().item():
                return True, ""
            bad = mismatch.nonzero(as_tuple=False)
            lines = [
                f"    hash route ids must match exactly: {bad.shape[0]} mismatch(es)"
            ]
            for row, pos in bad[:max_show].tolist():
                lines.append(
                    f"      [{row},{pos}] actual={int(actual[row, pos])} "
                    f"expected={int(expected[row, pos])}"
                )
            return False, "\n".join(lines)

        invalid = (actual < 0) | (actual >= N_EXPERTS)
        if invalid.any().item():
            bad = invalid.nonzero(as_tuple=False)
            lines = [
                f"    score route contains {bad.shape[0]} out-of-range id(s); "
                f"valid range is [0, {N_EXPERTS})"
            ]
            for row, pos in bad[:max_show].tolist():
                lines.append(f"      [{row},{pos}] actual={int(actual[row, pos])}")
            return False, "\n".join(lines)

        sorted_ids = torch.sort(actual, dim=-1).values
        duplicate = sorted_ids[:, 1:] == sorted_ids[:, :-1]
        if duplicate.any().item():
            rows = duplicate.any(dim=-1).nonzero(as_tuple=False).flatten()
            lines = [f"    score route contains duplicate ids in {rows.numel()} row(s)"]
            for row in rows[:max_show].tolist():
                lines.append(f"      row {row}: ids={actual[row].tolist()}")
            return False, "\n".join(lines)

        _, _, scores, biased = _golden_gate_scores(inputs)
        scores = scores[:active_tokens]
        biased = biased[:active_tokens]
        if not torch.isfinite(biased).all().item():
            return False, "    CPU router reference contains NaN or Inf"

        score_error = score_atol + score_rtol * scores.abs()
        actual_biased = torch.gather(biased, dim=-1, index=actual)
        actual_error = torch.gather(score_error, dim=-1, index=actual)
        selected_mask = torch.zeros_like(biased, dtype=torch.bool)
        selected_mask.scatter_(dim=-1, index=actual, value=True)
        omitted_lower = (biased - score_error).masked_fill(
            selected_mask,
            float("-inf"),
        )
        best_omitted_lower, best_omitted_id = omitted_lower.max(dim=-1, keepdim=True)
        selected_upper = actual_biased + actual_error
        selected_floor_upper, selected_floor_pos = selected_upper.min(
            dim=-1,
            keepdim=True,
        )
        omitted_better = best_omitted_lower > selected_floor_upper

        earlier_upper = selected_upper.unsqueeze(-1)
        later_lower = (actual_biased - actual_error).unsqueeze(-2)
        order_matrix = later_lower > earlier_upper
        order_bad = torch.triu(order_matrix, diagonal=1)
        if not omitted_better.any().item() and not order_bad.any().item():
            return True, ""

        lines = [
            "    score route exceeds the calibrated FP32 top-k ambiguity band "
            f"(score_atol={score_atol:g}, score_rtol={score_rtol:g})"
        ]
        bad_rows = omitted_better.nonzero(as_tuple=False)[:max_show, 0].tolist()
        for row in bad_rows:
            omitted_id = int(best_omitted_id[row, 0])
            selected_pos = int(selected_floor_pos[row, 0])
            selected_id = int(actual[row, selected_pos])
            regret = float(biased[row, omitted_id] - biased[row, selected_id])
            budget = float(score_error[row, omitted_id] + actual_error[row, selected_pos])
            lines.append(
                f"      row {row}: omitted id={omitted_id} beats selected "
                f"id={selected_id}; raw_regret={regret:.8g} budget={budget:.8g}"
            )
        shown = len(bad_rows)
        remaining = max_show - shown
        if remaining:
            for row, earlier, later in order_bad.nonzero(as_tuple=False)[:remaining].tolist():
                lines.append(
                    f"      row {row} order [{earlier},{later}]: "
                    f"id={int(actual[row, later])} is unambiguously above "
                    f"id={int(actual[row, earlier])}"
                )
        return False, "\n".join(lines)

    cmp.__name__ = (
        f"gate_indices_compare(score_atol={score_atol},score_rtol={score_rtol})"
    )
    return cmp


def gate_weights_compare(
    num_tokens,
    *,
    score_atol=1e-4,
    score_rtol=2e-5,
    weight_math_atol=2e-5,
    weight_sum_atol=2e-5,
    max_show=10,
):
    """Validate weights against unbiased CPU scores at the device-selected ids."""
    import torch

    active_tokens = max(0, min(T, int(num_tokens)))

    def cmp(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del expected_outputs, rtol, atol
        actual = actual.cpu().to(torch.float32)
        if actual.shape != expected.shape:
            return False, (
                f"    weight shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if "indices" not in actual_outputs:
            return False, "    compare_fn misconfigured: actual indices output is missing"

        inactive = actual[active_tokens:]
        inactive_nonzero = int(inactive.count_nonzero().item())
        if inactive_nonzero:
            return False, (
                f"    inactive weight tail contains {inactive_nonzero} nonzero values"
            )
        actual = actual[:active_tokens]
        if actual.numel() == 0:
            return True, ""
        if not torch.isfinite(actual).all().item():
            return False, "    device router weights contain NaN or Inf"
        if (actual <= 0).any().item():
            return False, "    active device router weights must be positive"
        weight_sum = actual.sum(dim=-1)
        sum_bad = (weight_sum - ROUTE_SCALE).abs() > weight_sum_atol
        if sum_bad.any().item():
            rows = sum_bad.nonzero(as_tuple=False).flatten()
            lines = [
                f"    router weight sum differs from ROUTE_SCALE={ROUTE_SCALE} "
                f"in {rows.numel()} row(s), atol={weight_sum_atol}"
            ]
            for row in rows[:max_show].tolist():
                lines.append(f"      row {row}: sum={float(weight_sum[row]):.8g}")
            return False, "\n".join(lines)

        indices = actual_outputs["indices"].cpu()[:active_tokens].to(torch.int64)
        if indices.shape != actual.shape:
            return False, (
                f"    index/weight shape mismatch: indices={tuple(indices.shape)} "
                f"weights={tuple(actual.shape)}"
            )
        invalid = (indices < 0) | (indices >= N_EXPERTS)
        if invalid.any().item():
            return False, "    cannot validate weights: device route contains invalid ids"
        sorted_ids = torch.sort(indices, dim=-1).values
        if (sorted_ids[:, 1:] == sorted_ids[:, :-1]).any().item():
            return False, "    cannot validate weights: device route contains duplicate ids"

        _, _, scores, _ = _golden_gate_scores(inputs)
        selected_scores = torch.gather(scores[:active_tokens], dim=-1, index=indices)
        selected_error = score_atol + score_rtol * selected_scores.abs()
        score_sum = selected_scores.sum(dim=-1, keepdim=True)
        error_sum = selected_error.sum(dim=-1, keepdim=True)
        if (score_sum <= error_sum).any().item():
            return False, "    router score uncertainty is larger than selected score sum"
        reference = selected_scores / score_sum
        reference = reference * ROUTE_SCALE
        tolerance = ROUTE_SCALE * (
            score_sum * selected_error + selected_scores * error_sum
        ) / (score_sum * (score_sum - error_sum))
        tolerance = tolerance + weight_math_atol
        close = (actual - reference).abs() <= tolerance
        if close.all().item():
            return True, ""

        bad = (~close).nonzero(as_tuple=False)
        lines = [
            f"    weights do not match unbiased CPU scores at device-selected ids: "
            f"{bad.shape[0]}/{actual.numel()} mismatch(es)"
        ]
        for row, pos in bad[:max_show].tolist():
            lines.append(
                f"      [{row},{pos}] id={int(indices[row, pos])} "
                f"actual={float(actual[row, pos]):.8g} "
                f"expected_for_id={float(reference[row, pos]):.8g} "
                f"tol={float(tolerance[row, pos]):.8g}"
            )
        return False, "\n".join(lines)

    cmp.__name__ = "gate_weights_compare"
    return cmp


def build_tensor_specs(layer_id=0, num_tokens=T):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_mixed():
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
        TensorSpec("x_norm_i8", [T, D], torch.int8, is_output=True),
        TensorSpec("x_norm_scale", [T, 1], torch.float32, is_output=True),
        TensorSpec("indices", [T, TOPK], torch.int32, is_output=True),
        TensorSpec("weights", [T, TOPK], torch.float32, is_output=True),
    ]


def gate_active_rows(num_tokens):
    """Active token count rounded up to the gate M-tile, capped at T."""
    active_count = max(0, min(T, int(num_tokens)))
    return min(T, ((active_count + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE)


def gate_x_norm_scale_compare(num_tokens):
    """Validate active scales numerically and require an exact-zero inactive tail."""
    from golden import ratio_allclose

    active_tokens = max(0, min(T, int(num_tokens)))
    return ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.0, valid_rows=active_tokens, zero_tail=True)


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=10)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    result = run_jit(
        fn=gate_test,
        specs=build_tensor_specs(layer_id=args.layer_id, num_tokens=args.num_tokens),
        golden_fn=golden_gate_core,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "x_norm_i8": ratio_allclose(
                atol=1, rtol=0, max_error_ratio=0.001,
                valid_rows=gate_active_rows(args.num_tokens),
            ),
            "x_norm_scale": gate_x_norm_scale_compare(args.num_tokens),
            "indices": gate_indices_compare(args.layer_id, args.num_tokens),
            "weights": gate_weights_compare(args.num_tokens),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
