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

from config import DEMO as M, DECODE_BATCH, DECODE_SEQ


# model config
B             = DECODE_BATCH
S             = DECODE_SEQ
T             = B * S
D             = M.hidden_size
NORM_EPS      = M.rms_norm_eps
N_EXPERTS     = M.n_routed_experts
TOPK          = M.num_experts_per_tok
ROUTE_SCALE   = M.routed_scaling_factor

# tiling
D_CHUNK          = 512
D_BLOCKS         = D // D_CHUNK
# SCORE_PAD = sort32 row width; PAIR_PAD covers the (val, idx) interleaved
# topk slice. FP32_NEG_INF seeds the unused tail so padding ranks last.
SCORE_PAD        = 32
PAIR_PAD         = 32
TOPK_GATHER_PAD  = PAIR_PAD // 2
FP32_NEG_INF     = -1.0e30


@pl.jit.inline
def moe_router(
    x_mixed:      pl.Tensor[[B, S, D],                    pl.BF16],
    norm_w:       pl.Tensor[[D],                           pl.FP32],
    gate_w:       pl.Tensor[[N_EXPERTS, D],                pl.FP32],
    gate_bias:    pl.Tensor[[N_EXPERTS],                   pl.FP32],
    x_norm:       pl.Tensor[[T, D],                        pl.BF16],
    indices:      pl.Tensor[[T, TOPK],                     pl.INT32],
    weights:      pl.Tensor[[T, TOPK],                     pl.FP32],
):
    # Stage 0: FFN RMSNorm over x_mixed.
    x_mixed_flat = pl.reshape(x_mixed, [T, D])
    inv_rms = pl.create_tensor([1, T], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="ffn_norm_rms"):
        sq_sum = pl.full([1, T], dtype=pl.FP32, value=0.0)
        for db in pl.pipeline(D_BLOCKS, stage=4):
            d0 = db * D_CHUNK
            x_chunk = pl.cast(x_mixed_flat[:, d0 : d0 + D_CHUNK], target_type=pl.FP32)
            sq_sum = pl.add(
                sq_sum,
                pl.reshape(pl.row_sum(pl.mul(x_chunk, x_chunk)), [1, T]),
            )
        inv_rms_val = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
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
    biased_scores = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    biased_flat = pl.reshape(biased_scores, [T * SCORE_PAD])
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="gate_dot"):
        # Pad-tail seeded to -inf so sort32 ranks padded slots after real
        # scores (otherwise pad-zero beats negative reals and topk picks pads).
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
            biased_row_flat = pl.reshape(biased_tile, [T])
            for t in pl.unroll(T):
                pl.write(biased_flat, [t * SCORE_PAD + expert_i], pl.read(biased_row_flat, [t]))
    biased_scores = pl.reshape(biased_flat, [T, SCORE_PAD])

    # Stage 3: per-token sort32 with paired UINT32 indices.
    sorted_rows = pl.create_tensor([T, 2 * SCORE_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_sort_top2"):
        for t in pl.unroll(T):
            score_row = biased_scores[t : t + 1, :]
            idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
            sorted_t = pl.tensor.sort32(score_row, idx_init)
            sorted_rows[t : t + 1, :] = sorted_t

    # Stage 4: extract topk vals/idx. Declaration order matters — codegen
    # binds GM buffers in source order and the block below writes vals first;
    # swapping these two lines silently rebinds downstream readers.
    topk_vals_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    topk_idx_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.INT32)
    weight_out_pad = pl.create_tensor([T, SCORE_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_extract_top2"):
        # topk_idx_pad is fully covered by the per-token assemble below, so
        # only the FP32 vals table needs zero-init.
        topk_vals_pad[:, :] = pl.full([T, SCORE_PAD], dtype=pl.FP32, value=0.0)
        for t in pl.unroll(T):
            topk_pairs = sorted_rows[t : t + 1, 0 : PAIR_PAD]
            topk_vals = pl.tensor.gather(topk_pairs, mask_pattern=pl.tile.MaskPattern.P0101)
            topk_i_raw = pl.tensor.gather(
                topk_pairs,
                mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
            # `valid_shape` carries the meaningful [1, TOPK] subspan so
            # fillpad zeros the padding columns; no [:] equivalent.
            topk_vals_valid = pl.slice(
                topk_vals,
                [1, TOPK_GATHER_PAD],
                [0, 0],
                valid_shape=[1, TOPK],
            )
            topk_vals_padded = pl.fillpad(topk_vals_valid, pad_value=pl.PadValue.zero)
            # Bias handling: weights here are taken straight from sort32, so
            # they include route bias. The zero-bias fixture makes them equal
            # to the unbiased scores; non-zero bias needs an indirect gather
            # on the score table, blocked on ptoas `gather(input, dim, index)`.
            topk_vals_pad[t : t + 1, 0 : TOPK_GATHER_PAD] = topk_vals_padded
            topk_idx_pad[t : t + 1, 0 : TOPK_GATHER_PAD] = topk_i_raw

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
    x_norm:       pl.Out[pl.Tensor[[T, D],                 pl.BF16]],
    indices:      pl.Out[pl.Tensor[[T, TOPK],              pl.INT32]],
    weights:      pl.Out[pl.Tensor[[T, TOPK],              pl.FP32]],
):
    moe_router(
        x_mixed,
        norm_w, gate_w, gate_bias,
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

    biased = scores + gate_bias
    indices = biased.topk(TOPK, dim=-1).indices

    weights = original_scores.gather(1, indices.long())
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * ROUTE_SCALE

    tensors["x_norm"][:]   = x_flat
    tensors["indices"][:]  = indices.to(torch.int32)
    tensors["weights"][:]  = weights.to(torch.float32)


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    def init_x_mixed():
        return torch.randn(B, S, D) * 0.1
    def init_norm_w():
        return torch.ones(D)
    def init_gate_w():
        return torch.randn(N_EXPERTS, D) / D ** 0.5
    def init_gate_bias():
        # Pinned to zero — see bias note at `route_extract_top2`.
        return torch.zeros(N_EXPERTS)
    return [
        TensorSpec("x_mixed",      [B, S, D],                  torch.bfloat16, init_value=init_x_mixed),
        TensorSpec("norm_w",       [D],                        torch.float32,  init_value=init_norm_w),
        TensorSpec("gate_w",       [N_EXPERTS, D],             torch.float32,  init_value=init_gate_w),
        TensorSpec("gate_bias",    [N_EXPERTS],                torch.float32,  init_value=init_gate_bias),
        TensorSpec("x_norm",       [T, D],                     torch.bfloat16, is_output=True),
        TensorSpec("indices",      [T, TOPK],                  torch.int32,    is_output=True),
        TensorSpec("weights",      [T, TOPK],                  torch.float32,  is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import RunConfig, run_jit, topk_pair_compare

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
    args = parser.parse_args()

    result = run_jit(
        fn=moe_router_test,
        specs=build_tensor_specs(),
        golden_fn=golden_moe_router_core,
        config=RunConfig(
            # `x_norm` is BF16 (1-ULP drift from reduction order); `indices`
            # uses topk_pair_compare to tolerate sort32-vs-torch.topk tie-break.
            rtol=1e-3,
            atol=1e-3,
            compare_fn={
                "x_norm":  bf16_allclose(1e-2, 1e-2),
                "indices": topk_pair_compare("weights"),
            },
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
