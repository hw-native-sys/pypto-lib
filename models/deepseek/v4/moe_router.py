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

from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, FP32_NEG_INF


# model config
B             = DECODE_BATCH
S             = DECODE_SEQ
T             = B * S
D             = M.hidden_size
NORM_EPS      = M.rms_norm_eps
N_EXPERTS     = M.n_routed_experts
TOPK          = M.num_experts_per_tok
ROUTE_SCALE   = M.routed_scaling_factor
VOCAB         = M.vocab_size
N_HASH_LAYERS = M.num_hash_layers

# tiling
D_CHUNK          = 128 if T >= 128 else (256 if T >= 64 else 512)
D_BLOCKS         = D // D_CHUNK
# SCORE_PAD = padded expert row width. sort32 handles 32-value runs; the
# 512-wide path uses two mrgsort passes to cover FLASH/PRO expert counts.
if N_EXPERTS <= 32:
    SCORE_PAD    = 32
elif N_EXPERTS <= 512:
    SCORE_PAD    = 512
else:
    raise ValueError(f"moe_router supports at most 512 routed experts, got {N_EXPERTS}")
# PAIR_PAD covers the (val, idx) interleaved topk slice. FP32_NEG_INF seeds
# the unused tail so padding ranks last.
PAIR_PAD         = 32
TOPK_GATHER_PAD  = PAIR_PAD // 2
assert TOPK <= TOPK_GATHER_PAD
RMS_PIPE_STAGE   = 1 if T >= 64 else 4

@pl.jit.inline
def moe_router(
    x_mixed:      pl.Tensor[[B, S, D],                    pl.BF16],
    norm_w:       pl.Tensor[[D],                           pl.FP32],
    gate_w:       pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:    pl.Tensor[[N_EXPERTS],                   pl.FP32],
    layer_id:     pl.Scalar[pl.INT32],
    tid2eid:      pl.Tensor[[VOCAB, TOPK],                 pl.INT32],
    input_ids:    pl.Tensor[[B, S],                        pl.INT64],
    x_norm:       pl.Tensor[[T, D],                        pl.BF16],
    indices:      pl.Tensor[[T, TOPK],                     pl.INT32],
    weights:      pl.Tensor[[T, TOPK],                     pl.FP32],
):
    # Stage 0: FFN RMSNorm over x_mixed.
    x_mixed_flat = pl.reshape(x_mixed, [T, D])
    inv_rms = pl.create_tensor([1, T], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="ffn_norm_rms"):
        sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for db in pl.pipeline(D_BLOCKS, stage=RMS_PIPE_STAGE):
            d0 = db * D_CHUNK
            x_chunk = pl.cast(x_mixed_flat[:, d0 : d0 + D_CHUNK], target_type=pl.FP32)
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]),
            )
        inv_rms_val = pl.rsqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS), high_precision=True)
        inv_rms[:, :] = inv_rms_val

    # Stage 1: ffn_norm apply. Doubles as the gate-dot input and as the
    # entry's `x_norm` output.
    x_norm_bf16 = pl.create_tensor([T, D], dtype=pl.BF16)
    for db in pl.parallel(0, D_BLOCKS, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_norm_apply"):
            d0 = db * D_CHUNK
            inv_rms_col = pl.reshape(inv_rms[:, :], [T, 1])
            x_chunk = pl.cast(x_mixed_flat[:, d0 : d0 + D_CHUNK], target_type=pl.FP32)
            norm_w_chunk = pl.reshape(norm_w[d0 : d0 + D_CHUNK], [1, D_CHUNK])
            x_normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_col), norm_w_chunk)
            # mode="rint" = round half to even, matches torch's `.to(bfloat16)`.
            x_normed_bf16 = pl.cast(x_normed, target_type=pl.BF16, mode="rint")
            x_norm_bf16[:, d0 : d0 + D_CHUNK] = x_normed_bf16
            x_norm[:, d0 : d0 + D_CHUNK] = x_normed_bf16

    # Stage 2: gate.forward — dot(x_norm, gate_w) → sqrt(softplus(.)) + bias.
    # Single fused pl.at across all N_EXPERTS: per-expert kernels would
    # collapse to identical outputs for tokens 1..15 on hardware.
    route_scores = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    biased_scores = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    route_flat = pl.reshape(route_scores, [T * SCORE_PAD])
    biased_flat = pl.reshape(biased_scores, [T * SCORE_PAD])
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="gate_dot"):
        # Pad-tail seeded to -inf so sort32 ranks padded slots after real
        # scores (otherwise pad-zero beats negative reals and topk picks pads).
        route_scores[:, :] = pl.full([T, SCORE_PAD], dtype=pl.FP32, value=0.0)
        biased_scores[:, :] = pl.full([T, SCORE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
        score_acc_buf = pl.create_tensor([1, T], dtype=pl.FP32)
        for expert_i in pl.range(N_EXPERTS):
            score_acc = pl.full([1, T], dtype=pl.FP32, value=0.0)
            for db in pl.range(D_BLOCKS):
                d0 = db * D_CHUNK
                x_chunk = pl.cast(x_norm_bf16[:, d0 : d0 + D_CHUNK], target_type=pl.FP32)
                w_row = gate_w[expert_i : expert_i + 1, d0 : d0 + D_CHUNK]
                prod = pl.col_expand_mul(x_chunk, w_row)
                score_acc = pl.add(
                    score_acc,
                    pl.reshape(pl.row_sum(prod), [1, T]),
                )
            score_acc_buf[:, :] = score_acc
            bias = pl.read(gate_bias, [expert_i])
            logits = pl.load(score_acc_buf, [0, 0], [1, T])
            zero = pl.mul(logits, 0.0)
            relu_logits = pl.maximum(logits, zero)
            abs_logits = pl.maximum(logits, pl.neg(logits))
            softplus = pl.add(relu_logits, pl.log(pl.add(pl.exp(pl.neg(abs_logits)), 1.0)))
            score_tile = pl.sqrt(softplus)
            biased_tile = pl.add(score_tile, bias)
            score_row_flat = pl.reshape(score_tile, [T])
            biased_row_flat = pl.reshape(biased_tile, [T])
            for t in pl.unroll(T):
                pl.write(route_flat, [t * SCORE_PAD + expert_i], pl.read(score_row_flat, [t]))
                pl.write(biased_flat, [t * SCORE_PAD + expert_i], pl.read(biased_row_flat, [t]))
    route_scores = pl.reshape(route_flat, [T, SCORE_PAD])
    biased_scores = pl.reshape(biased_flat, [T, SCORE_PAD])

    # Stage 3: choose routed expert ids. Hash layers take ids directly from
    # tid2eid[input_ids]; score-routed layers use biased top-k.
    topk_vals_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    topk_idx_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.INT32)
    weight_out_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    if layer_id < N_HASH_LAYERS:
        tid2eid_flat = pl.reshape(tid2eid, [VOCAB * TOPK])
        input_ids_flat = pl.reshape(input_ids, [T])
        route_scores_flat = pl.reshape(route_scores, [T * SCORE_PAD])
        topk_vals_pad_flat = pl.reshape(topk_vals_pad, [T * SCORE_PAD])
        topk_idx_pad_flat = pl.reshape(topk_idx_pad, [T * SCORE_PAD])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_hash_indices_and_scores"):
            for p in pl.range(T * SCORE_PAD):
                pl.write(topk_vals_pad_flat, [p], 0.0)
            for t in pl.unroll(T):
                token_id = pl.cast(pl.read(input_ids_flat, [t]), pl.INDEX)
                src_base = token_id * TOPK
                dst_score_base = t * SCORE_PAD
                for k in pl.unroll(TOPK):
                    hash_expert_i = pl.read(tid2eid_flat, [src_base + k])
                    hash_expert_pos = pl.cast(hash_expert_i, pl.INDEX)
                    hash_score = pl.read(route_scores_flat, [dst_score_base + hash_expert_pos])
                    pl.write(topk_idx_pad_flat, [dst_score_base + k], hash_expert_i)
                    pl.write(topk_vals_pad_flat, [dst_score_base + k], hash_score)
        topk_idx_pad = pl.reshape(topk_idx_pad_flat, [T, SCORE_PAD])
        topk_vals_pad = pl.reshape(topk_vals_pad_flat, [T, SCORE_PAD])
    else:
        topk_idx_work = pl.create_tensor([T, TOPK_GATHER_PAD], dtype=pl.INT32)
        # Per-token sort with paired UINT32 indices.
        sorted_rows = pl.create_tensor([T, 2 * SCORE_PAD], dtype=pl.FP32)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_sort_topk"):
            for t in pl.unroll(T):
                score_row = biased_scores[t : t + 1, :]
                idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                sorted_t = pl.tensor.sort32(score_row, idx_init)
                if SCORE_PAD == 512:
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=64)
                    sorted_t = pl.tensor.mrgsort(sorted_t, block_len=256)
                sorted_rows[t : t + 1, :] = sorted_t

        # Extract topk indices from the biased sort result.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_extract_topk"):
            for t in pl.unroll(T):
                topk_pairs = sorted_rows[t : t + 1, 0 : PAIR_PAD]
                topk_i_raw = pl.tensor.gather(
                    topk_pairs,
                    mask_pattern=pl.tile.MaskPattern.P1010,
                    output_dtype=pl.INT32,
                )
                topk_idx_work[t : t + 1, :] = topk_i_raw
                topk_idx_pad[t : t + 1, 0 : TOPK_GATHER_PAD] = topk_i_raw

        # Stage 4: gather unbiased score values for the selected expert ids.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_gather_weights"):
            topk_scores = pl.tensor.gather(route_scores, dim=-1, index=topk_idx_work)
            topk_vals_pad[:, :] = pl.full([T, SCORE_PAD], dtype=pl.FP32, value=0.0)
            for t in pl.unroll(T):
                # `valid_shape` carries the meaningful [1, TOPK] subspan so
                # fillpad zeros the padding columns; no [:] equivalent.
                topk_vals_valid = pl.slice(
                    topk_scores,
                    [1, TOPK_GATHER_PAD],
                    [t, 0],
                    valid_shape=[1, TOPK],
                )
                topk_vals_padded = pl.fillpad(topk_vals_valid, pad_value=pl.PadValue.zero)
                topk_vals_pad[t : t + 1, 0 : TOPK_GATHER_PAD] = topk_vals_padded

    # Stage 5: normalize weights, then scatter first TOPK cols to GM outputs.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_normalize_weights"):
        denom = pl.reshape(pl.row_sum(topk_vals_pad), [T, 1])
        weights_pad = pl.mul(pl.row_expand_div(topk_vals_pad, denom), ROUTE_SCALE)
        weight_out_pad[:, :] = weights_pad

    indices_flat = pl.reshape(indices, [T * TOPK])
    weights_flat = pl.reshape(weights, [T * TOPK])
    topk_idx_flat = pl.reshape(topk_idx_pad, [T * SCORE_PAD])
    weight_out_flat = pl.reshape(weight_out_pad, [T * SCORE_PAD])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="write_route_outputs"):
        for t in pl.unroll(T):
            dst_base = t * TOPK
            src_base = t * SCORE_PAD
            for k in pl.unroll(TOPK):
                indices_flat = pl.write(indices_flat, [dst_base + k], pl.read(topk_idx_flat, [src_base + k]))
                weights_flat = pl.write(weights_flat, [dst_base + k], pl.read(weight_out_flat, [src_base + k]))


@pl.jit
def moe_router_test(
    x_mixed:      pl.Tensor[[B, S, D],                     pl.BF16],
    norm_w:       pl.Tensor[[D],                           pl.FP32],
    gate_w:       pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:    pl.Tensor[[N_EXPERTS],                   pl.FP32],
    layer_id:     pl.Scalar[pl.INT32],
    tid2eid:      pl.Tensor[[VOCAB, TOPK],                 pl.INT32],
    input_ids:    pl.Tensor[[B, S],                        pl.INT64],
    x_norm:       pl.Out[pl.Tensor[[T, D],                 pl.BF16]],
    indices:      pl.Out[pl.Tensor[[T, TOPK],              pl.INT32]],
    weights:      pl.Out[pl.Tensor[[T, TOPK],              pl.FP32]],
):
    moe_router(
        x_mixed,
        norm_w, gate_w, gate_bias,
        layer_id,
        tid2eid, input_ids,
        x_norm, indices, weights,
    )
    return x_norm, indices, weights


def golden_moe_router_core(tensors):
    import torch
    import torch.nn.functional as F

    # RMSNorm; cast back to bf16 to match what downstream gate/expert see.
    norm_w = tensors["norm_w"].float()
    x_f = tensors["x_mixed"].float()
    var = x_f.square().mean(-1, keepdim=True)
    x_n = x_f * torch.rsqrt(var + NORM_EPS)
    x_normalized = (norm_w * x_n).to(torch.bfloat16)
    x_flat = x_normalized.view(T, D)

    gate_w = tensors["gate_w"].float()
    gate_bias = tensors["gate_bias"].float()
    scores = F.softplus(x_flat.float() @ gate_w.T).sqrt()
    original_scores = scores

    layer_id = int(tensors["layer_id"])
    if layer_id < N_HASH_LAYERS:
        tid2eid = tensors["tid2eid"]
        input_ids = tensors["input_ids"]
        indices = tid2eid[input_ids.flatten().long()]
    else:
        biased = scores + gate_bias
        # Stable sort: deterministic tie-break to match the NPU sort32 order.
        indices = torch.argsort(-biased, dim=-1, stable=True)[..., :TOPK]

    weights = original_scores.gather(1, indices.long())
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    tensors["x_norm"][:]   = x_flat
    tensors["indices"][:]  = indices.to(torch.int32)
    tensors["weights"][:]  = weights.to(torch.float32)


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
        TensorSpec("x_mixed",      [B, S, D],                  torch.bfloat16, init_value=init_x_mixed),
        TensorSpec("norm_w",       [D],                        torch.float32,  init_value=init_norm_w),
        TensorSpec("gate_w",       [N_EXPERTS, D],             torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",    [N_EXPERTS],                torch.float32,  init_value=init_gate_bias),
        ScalarSpec("layer_id",     torch.int32,                layer_id),
        TensorSpec("tid2eid",      [VOCAB, TOPK],              torch.int32,    init_value=init_tid2eid),
        TensorSpec("input_ids",    [B, S],                     torch.int64,    init_value=init_input_ids),
        TensorSpec("x_norm",       [T, D],                     torch.bfloat16, is_output=True),
        TensorSpec("indices",      [T, TOPK],                  torch.int32,    is_output=True),
        TensorSpec("weights",      [T, TOPK],                  torch.float32,  is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import run_jit

    def bf16_allclose(rtol, atol):
        """Loosened comparator for BF16 outputs whose kernel reduction order
        differs from torch's, occasionally crossing a BF16 rounding boundary."""
        def cmp(actual, expected, **_):
            return torch.allclose(actual, expected, rtol=rtol, atol=atol), ""
        return cmp

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--layer-id", type=int, default=0)
    args = parser.parse_args()

    result = run_jit(
        fn=moe_router_test,
        specs=build_tensor_specs(layer_id=args.layer_id),
        golden_fn=golden_moe_router_core,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
        ),
        rtol=1e-5,
        atol=1e-5,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
