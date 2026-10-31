# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FP32 MoE gate, text/vision expert selection, and normalized routing weights."""

import pypto.language as pl


# tiling
D_TILE = 256
EXPERT_TILE = 8
TOPK_TILE = 8


def make_moe_router_inline(capacity, config):
    """Build full-precision gate projection and 384-expert top-six routing."""
    hidden = config["hidden_size"]
    experts = config["n_routed_experts"]
    topk = config["num_experts_per_tok"]
    route_scale = config["routed_scaling_factor"]
    sentinel_base = config["image_sentinel_base_id"]
    sentinel_end = sentinel_base + 5
    if experts != 384 or topk != 6:
        raise ValueError("The A2 router requires 384 experts and six selected experts")
    if hidden % D_TILE:
        raise ValueError("The gate hidden size must be divisible by D_TILE")

    @pl.jit.inline
    def moe_router_inline(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        text_bias: pl.Tensor[[experts], pl.FP32],
        vision_bias: pl.Tensor[[experts], pl.FP32],
        input_ids: pl.Tensor[[capacity], pl.INT64],
        logits: pl.Tensor[[capacity, experts], pl.FP32],
        route_ids: pl.Tensor[[capacity, topk], pl.INT32],
        route_weights: pl.Tensor[[capacity, topk], pl.FP32],
        active: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(active * (experts // EXPERT_TILE), name_hint="a2_moe_gate_fp32"):
            token = block // (experts // EXPERT_TILE)
            expert = block % (experts // EXPERT_TILE) * EXPERT_TILE
            dot_sum = pl.full([1, EXPERT_TILE], dtype=pl.FP32, value=0.0)
            for inner in pl.range(0, hidden, D_TILE):
                source = pl.cast(x[token:token + 1, inner:inner + D_TILE], pl.FP32)
                weight = gate_weight[expert:expert + EXPERT_TILE, inner:inner + D_TILE]
                product = pl.col_expand_mul(weight, source)
                partial = pl.reshape(pl.row_sum(product), [1, EXPERT_TILE])
                dot_sum = pl.add(dot_sum, partial)
            logits[token:token + 1, expert:expert + EXPERT_TILE] = dot_sum

        raw_scores = pl.create_tensor([capacity, experts], dtype=pl.FP32)
        selected_ids = pl.create_tensor([capacity, TOPK_TILE], dtype=pl.INT32)
        for token in pl.spmd(active, name_hint="a2_moe_route_sort"):
            gate_logits = logits[token:token + 1, :]
            positive = pl.maximum(gate_logits, 0.0)
            negative_abs = pl.neg(pl.abs(gate_logits))
            exponential = pl.exp(negative_abs)
            log_tail = pl.log(pl.add(exponential, 1.0))
            softplus_log = pl.add(positive, log_tail)
            negative_shift = pl.sub(pl.neg(gate_logits), 10.0)
            negative_mask = pl.minimum(pl.maximum(negative_shift, 0.0), 1.0)
            negative_exp = pl.exp(pl.minimum(gate_logits, 0.0))
            negative_floor = pl.mul(negative_mask, negative_exp)
            softplus = pl.maximum(softplus_log, negative_floor)
            scores = pl.sqrt(softplus)
            raw_scores[token:token + 1, :] = scores

            bias = pl.reshape(text_bias[:], [1, experts])
            token_id = pl.read(input_ids, [token])
            if token_id >= sentinel_base:
                if token_id < sentinel_end:
                    bias = pl.reshape(vision_bias[:], [1, experts])
            biased_scores = pl.add(scores, bias)
            index_range = pl.arange(0, [1, experts], dtype=pl.UINT32)
            sorted_32 = pl.sort32(biased_scores, index_range)
            sorted_64 = pl.mrgsort(sorted_32, block_len=64)
            first = sorted_64[:, 0:256]
            second = sorted_64[:, 256:512]
            third = sorted_64[:, 512:768]
            sorted_pairs = pl.mrgsort(first, second, third)
            top_pairs = sorted_pairs[:, 0:16]
            top_ids = pl.gather(top_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected_ids[token:token + 1, :] = top_ids

        # Each row is only 6 INT32/FP32 values (24 bytes), so parallel scalar
        # stores would share a 64-byte cache line and lose neighboring rows.
        # Serialize this small write-back until the output is padded or tile
        # stores can provide an atomic full-line update.
        for _block in pl.spmd(1, name_hint="a2_moe_route_normalize"):
            for token in pl.range(active):
                score_sum = pl.cast(0.0, pl.FP32)
                for rank in pl.range(topk):
                    selected_expert_idx = pl.cast(pl.read(selected_ids, [token, rank]), pl.INDEX)
                    score = pl.read(raw_scores, [token, selected_expert_idx])
                    score_sum = score_sum + score
                for rank in pl.range(topk):
                    selected_expert_idx = pl.cast(pl.read(selected_ids, [token, rank]), pl.INDEX)
                    score = pl.read(raw_scores, [token, selected_expert_idx])
                    normalized = (score / score_sum) * route_scale
                    pl.write(route_ids, [token, rank], pl.cast(selected_expert_idx, pl.INT32))
                    pl.write(route_weights, [token, rank], normalized)
        return logits, route_ids, route_weights

    return moe_router_inline


def make_moe_router(capacity, config):
    """Build the standalone MoE router entry with untouched inactive rows."""
    hidden = config["hidden_size"]
    experts = config["n_routed_experts"]
    topk = config["num_experts_per_tok"]
    compute = make_moe_router_inline(capacity, config)

    @pl.jit
    def moe_router(
        x: pl.Tensor[[capacity, hidden], pl.BF16],
        gate_weight: pl.Tensor[[experts, hidden], pl.FP32],
        text_bias: pl.Tensor[[experts], pl.FP32],
        vision_bias: pl.Tensor[[experts], pl.FP32],
        input_ids: pl.Tensor[[capacity], pl.INT64],
        logits: pl.InOut[pl.Tensor[[capacity, experts], pl.FP32]],
        route_ids: pl.InOut[pl.Tensor[[capacity, topk], pl.INT32]],
        route_weights: pl.InOut[pl.Tensor[[capacity, topk], pl.FP32]],
        active: pl.Scalar[pl.INT32],
    ):
        compute(x, gate_weight, text_bias, vision_bias, input_ids, logits, route_ids, route_weights, active)
        return logits, route_ids, route_weights

    return moe_router
