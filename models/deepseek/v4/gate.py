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

from config import (FLASH as M, DECODE_BATCH, DECODE_SEQ, FP32_NEG_INF, EP_WORLD_SIZE,
                    INT8_SCALE_MAX, INT8_AMAX_EPS)


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
D = M.hidden_size
NORM_EPS = M.rms_norm_eps
N_EXPERTS = M.n_routed_experts // EP_WORLD_SIZE   # single-card view: route only over local shard
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor
VOCAB = M.vocab_size
N_HASH_LAYERS = M.num_hash_layers

# tiling
T_TILE = 8
GATE_T_TILE = 16
D_CHUNK = 128
GATE_D_CHUNK = 512
QUANT_CHUNK = 32
SCORE_PAD = 256         # padded expert row for sort32 + mrgsort
TOPK_PAD = 8            # TOPK padded to 32B-aligned width
SORT_PAD = TOPK_PAD * 2 # (val, idx) interleaved slice width
assert TOPK <= TOPK_PAD

@pl.jit.inline
def gate(
    x_mixed: pl.Tensor[[B, S, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S], pl.INT64],
    x_norm: pl.Tensor[[T, D], pl.BF16],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    indices: pl.Tensor[[T, TOPK], pl.INT32],
    weights: pl.Tensor[[T, TOPK], pl.FP32],
):
    x_mixed_flat = pl.reshape(x_mixed, [T, D])
    input_ids_flat = pl.reshape(input_ids, [T])

    x_norm_gate_buf = pl.create_tensor([T, D], dtype=pl.FP32)
    route_scores_buf = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    biased_scores_buf = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)

    for ob_n in pl.spmd(T // T_TILE, name_hint="ffn_norm"):
        t0 = ob_n * T_TILE
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for rms_d0 in pl.pipeline(0, D, D_CHUNK, stage=2):
            rms_x = pl.cast(x_mixed_flat[t0 : t0 + T_TILE, rms_d0 : rms_d0 + D_CHUNK], pl.FP32)
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x, rms_x)), [1, T_TILE]))
        inv_rms = pl.reshape(pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS))), [T_TILE, 1])
        for an_d0 in pl.pipeline(0, D, D_CHUNK, stage=2):
            an_x = pl.cast(x_mixed_flat[t0 : t0 + T_TILE, an_d0 : an_d0 + D_CHUNK], pl.FP32)
            an_w = pl.reshape(norm_w[an_d0 : an_d0 + D_CHUNK], [1, D_CHUNK])
            an_normed = pl.col_expand_mul(pl.row_expand_mul(an_x, inv_rms), an_w)
            an_bf16 = pl.cast(an_normed, pl.BF16, mode="rint")
            x_norm_gate_buf[t0 : t0 + T_TILE, an_d0 : an_d0 + D_CHUNK] = pl.cast(an_bf16, pl.FP32)
            x_norm[t0 : t0 + T_TILE, an_d0 : an_d0 + D_CHUNK] = an_bf16

    # Per-token symmetric INT8 quant of x_norm.
    for ob_q in pl.spmd(T // T_TILE, name_hint="x_norm_quant"):
        t0 = ob_q * T_TILE
        xn_amax = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for xq_a_k in pl.pipeline(0, D, QUANT_CHUNK, stage=2):
            xn_a_f32 = x_norm_gate_buf[t0 : t0 + T_TILE, xq_a_k : xq_a_k + QUANT_CHUNK]
            xn_a_abs = pl.maximum(xn_a_f32, pl.neg(xn_a_f32))
            xn_a_max = pl.reshape(pl.row_max(xn_a_abs), [1, T_TILE])
            xn_amax = pl.maximum(xn_amax, xn_a_max)
        xn_sq_row = pl.div(pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), xn_amax)
        x_norm_scale[t0 : t0 + T_TILE, 0:1] = pl.reshape(pl.recip(xn_sq_row), [T_TILE, 1])
        xn_sq_col = pl.reshape(xn_sq_row, [T_TILE, 1])
        for xq_b_k in pl.pipeline(0, D, QUANT_CHUNK, stage=2):
            xn_q_scaled = pl.row_expand_mul(x_norm_gate_buf[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_CHUNK], xn_sq_col)
            xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
            xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
            x_norm_i8[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_CHUNK] = pl.cast(xn_q_half, pl.INT8, mode="trunc")

    # Gate matmul + post: x_norm @ gate_w.T → sqrt(softplus(logits)) (+bias).
    for ob_g in pl.spmd(T // GATE_T_TILE, name_hint="gate"):
        t1 = ob_g * GATE_T_TILE
        gp_bias_row = pl.reshape(gate_bias, [1, N_EXPERTS])
        gate_logits_tile = pl.create_tensor([GATE_T_TILE, N_EXPERTS], dtype=pl.FP32)
        for kb in pl.range(0, D // GATE_D_CHUNK):
            gd_kd = kb * GATE_D_CHUNK
            gd_x = x_norm_gate_buf[t1 : t1 + GATE_T_TILE, gd_kd : gd_kd + GATE_D_CHUNK]
            gd_w = gate_w[:, gd_kd : gd_kd + GATE_D_CHUNK]
            if gd_kd == 0:
                gate_logits_tile = pl.matmul(gd_x, gd_w, out_dtype=pl.FP32, b_trans=True)
            else:
                gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True)
        route_scores_buf[t1 : t1 + GATE_T_TILE, :] = pl.full([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32, value=0.0)
        biased_scores_buf[t1 : t1 + GATE_T_TILE, :] = pl.full([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
        gp_relu = pl.maximum(gate_logits_tile, 0.0)
        gp_abs = pl.maximum(gate_logits_tile, pl.neg(gate_logits_tile))
        gp_softplus = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
        gp_score = pl.sqrt(gp_softplus)
        gp_bias = pl.col_expand_mul(pl.full([GATE_T_TILE, N_EXPERTS], dtype=pl.FP32, value=1.0), gp_bias_row)
        gp_biased = pl.add(gp_score, gp_bias)
        route_scores_buf[t1 : t1 + GATE_T_TILE, 0:N_EXPERTS] = gp_score
        biased_scores_buf[t1 : t1 + GATE_T_TILE, 0:N_EXPERTS] = gp_biased

    # Hash layers index via tid2eid[input_ids]; score layers sort+gather.
    if layer_id < N_HASH_LAYERS:
        for ob_h in pl.spmd(T // GATE_T_TILE, name_hint="route_hash"):
            t1 = ob_h * GATE_T_TILE
            # Scalar gather TOPK (eid, unbiased score) per row; tail
            # [TOPK, TOPK_PAD) stays zero so row_sum below sums only TOPK.
            hs_vals_buf = pl.full([GATE_T_TILE, TOPK_PAD], dtype=pl.FP32, value=0.0)
            hs_idx_buf = pl.full([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32, value=0)
            for hs_tt in pl.range(GATE_T_TILE):
                hs_token = pl.cast(pl.read(input_ids_flat, [t1 + hs_tt]), pl.INDEX)
                for hs_k in pl.range(TOPK):
                    hs_eid = pl.read(tid2eid, [hs_token, hs_k])
                    hs_epos = pl.cast(hs_eid, pl.INDEX)
                    hs_unbiased = pl.read(route_scores_buf, [t1 + hs_tt, hs_epos])
                    pl.write(hs_idx_buf, [hs_tt, hs_k], hs_eid)
                    pl.write(hs_vals_buf, [hs_tt, hs_k], hs_unbiased)
            # Normalize+scale, then scalar-scatter to GM. Slice-assign would
            # alloc a [GATE_T_TILE, TOPK=6] temp (24B row, under alloc_tile's
            # 32B alignment), so write element-by-element.
            hs_denom = pl.reshape(pl.row_sum(hs_vals_buf), [GATE_T_TILE, 1])
            hs_weights_buf = pl.mul(pl.row_expand_div(hs_vals_buf, hs_denom), ROUTE_SCALE)
            for hs_wt_tt in pl.range(GATE_T_TILE):
                for hs_wt_k in pl.range(TOPK):
                    pl.write(indices, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_idx_buf, [hs_wt_tt, hs_wt_k]))
                    pl.write(weights, [t1 + hs_wt_tt, hs_wt_k], pl.read(hs_weights_buf, [hs_wt_tt, hs_wt_k]))
    else:
        for ob_s in pl.spmd(T // GATE_T_TILE, name_hint="route_sort"):
            t1 = ob_s * GATE_T_TILE
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
            # Copy topk_idx_tile to dodge the tensor_view-vs-ptr SSA conflict
            # between sort's slice-assign and scalar pl.read (pypto #1493).
            topk_idx_read = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
            topk_idx_read[:, :] = topk_idx_tile[:, :]
            nm_denom = pl.reshape(pl.row_sum(topk_vals_pad), [GATE_T_TILE, 1])
            nm_weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, nm_denom), ROUTE_SCALE)
            for nm_tt in pl.range(GATE_T_TILE):
                for nm_k in pl.range(TOPK):
                    pl.write(indices, [t1 + nm_tt, nm_k], pl.read(topk_idx_read, [nm_tt, nm_k]))
                    pl.write(weights, [t1 + nm_tt, nm_k], pl.read(nm_weights_pad, [nm_tt, nm_k]))


@pl.jit
def gate_test(
    x_mixed: pl.Tensor[[B, S, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[B, S], pl.INT64],
    x_norm: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    x_norm_i8: pl.Out[pl.Tensor[[T, D], pl.INT8]],
    x_norm_scale: pl.Out[pl.Tensor[[T, 1], pl.FP32]],
    indices: pl.Out[pl.Tensor[[T, TOPK], pl.INT32]],
    weights: pl.Out[pl.Tensor[[T, TOPK], pl.FP32]],
):
    gate(
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

    # FFN RMSNorm.
    x_f = tensors["x_mixed"].float().view(T, D)
    norm_w = tensors["norm_w"].float()
    sq_sum = (x_f * x_f).sum(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sq_sum * (1.0 / D) + NORM_EPS)
    x_normalized = (x_f * inv_rms) * norm_w.view(1, D)
    x_flat = x_normalized.to(torch.bfloat16)

    # Per-token symmetric INT8 quant of bf16(x_norm).
    x_norm_i8, x_norm_scale = _per_token_int8_quant(x_flat)

    # Gate matmul + numerically stable softplus(x) = relu(x) + log(exp(-|x|)+1).
    gate_w = tensors["gate_w"].float()
    gate_bias = tensors["gate_bias"].float()
    logits = x_flat.float() @ gate_w.T
    relu_logits = torch.maximum(logits, torch.zeros_like(logits))
    abs_logits = torch.maximum(logits, -logits)
    softplus = relu_logits + torch.log(torch.exp(-abs_logits) + 1.0)
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
    tensors["x_norm_scale"][:] = x_norm_scale.reshape(T, 1)
    tensors["indices"][:] = indices.to(torch.int32)
    tensors["weights"][:] = weights.to(torch.float32)


def build_tensor_specs(layer_id=0):
    import torch
    from golden import ScalarSpec, TensorSpec

    def init_x_mixed():
        # Mirror post-RMSNorm activation magnitude (~ N(0, 1)).
        return torch.randn(B, S, D)
    def init_norm_w():
        return torch.ones(D)
    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():
        return torch.randn(N_EXPERTS) * 0.1
    def init_tid2eid():
        return torch.randint(0, N_EXPERTS, (VOCAB, TOPK), dtype=torch.int32)
    def init_input_ids():
        return torch.randint(0, VOCAB, (B, S), dtype=torch.int64)
    return [
        TensorSpec("x_mixed", [B, S, D], torch.bfloat16, init_value=init_x_mixed),
        TensorSpec("norm_w", [D], torch.float32, init_value=init_norm_w),
        TensorSpec("gate_w", [N_EXPERTS, D], torch.float32, init_value=init_gate_w),
        TensorSpec("gate_bias", [N_EXPERTS], torch.float32, init_value=init_gate_bias),
        ScalarSpec("layer_id", torch.int32, layer_id),
        TensorSpec("tid2eid", [VOCAB, TOPK], torch.int32, init_value=init_tid2eid),
        TensorSpec("input_ids", [B, S], torch.int64, init_value=init_input_ids),
        TensorSpec("x_norm", [T, D], torch.bfloat16, is_output=True),
        TensorSpec("x_norm_i8", [T, D], torch.int8, is_output=True),
        TensorSpec("x_norm_scale", [T, 1], torch.float32, is_output=True),
        TensorSpec("indices", [T, TOPK], torch.int32, is_output=True),
        TensorSpec("weights", [T, TOPK], torch.float32, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit, topk_pair_compare

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=gate_test,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_gate_core,
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
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
