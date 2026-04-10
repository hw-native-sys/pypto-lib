# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Kimi K2 single-layer decode forward implementation in PyPTO.

Kimi K2 Architecture Highlights:
- MoE (Mixture of Experts) with shared expert
- Hybrid attention: sliding window + global attention
- Multi-token prediction support
- Optimized for ultra-long context (128K+)

This implementation focuses on:
- decode only (single token generation)
- single Transformer layer with MoE
- batch = 16 by default
- KV cache with sliding window (4096)
- 8 experts + 1 shared expert

Reference: https://github.com/MoonshotAI/Kimi-K2
"""

import pypto.language as pl


# =============================================================================
# Kimi K2 Base Configuration (Kimi-K2-Base like)
# =============================================================================
BATCH = 16
MAX_SEQ = 4096  # Sliding window size
HIDDEN = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 12288  # SwiGLU intermediate
NUM_EXPERTS = 8
NUM_ACTIVE_EXPERTS = 4
EPS = 1e-6
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
HIDDEN_INV = 1.0 / HIDDEN

# Tile configuration for InCore memory optimization
K_CHUNK = 256
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 32
SEQ_TILE = 128
MLP_OUT_CHUNK = 32
BATCH_TILE = 4
EXPERT_CHUNK = 64


def build_kimi_k2_decode_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
    num_experts: int = NUM_EXPERTS,
    num_active_experts: int = NUM_ACTIVE_EXPERTS,
):
    """Build Kimi K2 decode program with MoE support."""
    
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    NUM_HEADS_CFG = num_heads
    NUM_KV_HEADS_CFG = num_kv_heads
    HEAD_DIM_CFG = head_dim
    KV_HIDDEN_CFG = num_kv_heads * head_dim
    INTER_CFG = intermediate_size
    NUM_EXPERTS_CFG = num_experts
    NUM_ACTIVE_CFG = num_active_experts
    Q_PER_KV_CFG = num_heads // num_kv_heads

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    KV_OUT_BLOCKS = (KV_HIDDEN_CFG + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_CFG + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK
    CACHE_ROWS = BATCH_CFG * NUM_KV_HEADS_CFG * MAX_SEQ_CFG

    @pl.program
    class KimiK2Decode:
        @pl.function(type=pl.FunctionType.Opaque)
        def kimi_k2_decode_layer(
            self,
            hidden_states: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
            cache_pos: pl.Tensor[[BATCH_CFG], pl.INT32],
            rope_cos: pl.Tensor[[MAX_SEQ_CFG, HEAD_DIM_CFG], pl.FP32],
            rope_sin: pl.Tensor[[MAX_SEQ_CFG, HEAD_DIM_CFG], pl.FP32],
            k_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM_CFG], pl.BF16],
            v_cache: pl.Tensor[[CACHE_ROWS, HEAD_DIM_CFG], pl.BF16],
            input_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            # Attention weights
            wq: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            wk: pl.Tensor[[HIDDEN_CFG, KV_HIDDEN_CFG], pl.BF16],
            wv: pl.Tensor[[HIDDEN_CFG, KV_HIDDEN_CFG], pl.BF16],
            wo: pl.Tensor[[HIDDEN_CFG, HIDDEN_CFG], pl.BF16],
            # Post-attention RMSNorm weight
            post_rms_weight: pl.Tensor[[1, HIDDEN_CFG], pl.FP32],
            # MoE gate & experts
            moe_gate: pl.Tensor[[HIDDEN_CFG, NUM_EXPERTS_CFG], pl.FP32],
            w_gate_shared: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],  # Shared expert gate
            w_up_shared: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],    # Shared expert up
            w_down_shared: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],  # Shared expert down
            # Expert weights (flattened: [num_experts, hidden, inter])
            w_gate_experts: pl.Tensor[[NUM_EXPERTS_CFG, HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up_experts: pl.Tensor[[NUM_EXPERTS_CFG, HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down_experts: pl.Tensor[[NUM_EXPERTS_CFG, INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            """
            Kimi K2 decode layer with MoE.
            
            Flow:
            1. Input RMSNorm
            2. QKV Projection
            3. RoPE + KV Cache Update
            4. Flash Decoding Attention (sliding window)
            5. Output Projection + Residual
            6. Post RMSNorm
            7. MoE (Shared Expert + Routed Experts)
            8. Final Residual
            """
            q_proj = pl.create_tensor([BATCH_CFG, HIDDEN_CFG], dtype=pl.BF16)
            k_proj = pl.create_tensor([BATCH_CFG, KV_HIDDEN_CFG], dtype=pl.BF16)
            v_proj = pl.create_tensor([BATCH_CFG, KV_HIDDEN_CFG], dtype=pl.BF16)
            attn_out = pl.create_tensor([BATCH_CFG, HIDDEN_CFG], dtype=pl.FP32)

            # =========================================================================
            # Scope 1: Input RMSNorm + QKV Projection
            # =========================================================================
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                sq_sum = pl.create_tensor([BATCH_CFG, 1], dtype=pl.FP32)
                sq_sum = pl.mul(sq_sum, 0.0)

                # Compute RMSNorm: sum of squares
                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.cast(
                        pl.slice(hidden_states, [BATCH_CFG, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))

                inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                # QKV projection with chunked computation
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    inv_rms_tile = pl.slice(inv_rms, [BATCH_TILE, 1], [b0, 0])

                    # Q projection
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=4):
                        q0 = ob * Q_OUT_CHUNK
                        q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        q_acc = pl.mul(q_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk_bf16 = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
                            x_chunk = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                            wq_chunk = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_acc = pl.add(q_acc, pl.matmul(pl.cast(normed, target_type=pl.BF16), wq_chunk))
                        q_proj = pl.assemble(q_proj, pl.cast(q_acc, target_type=pl.BF16), [b0, q0])

                    # K/V projection
                    for ob in pl.parallel(0, KV_OUT_BLOCKS, 1, chunk=8):
                        kv0 = ob * KV_OUT_CHUNK
                        k_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        v_acc = pl.create_tensor([BATCH_TILE, KV_OUT_CHUNK], dtype=pl.FP32)
                        k_acc = pl.mul(k_acc, 0.0)
                        v_acc = pl.mul(v_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk_bf16 = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
                            x_chunk = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                            normed_bf16 = pl.cast(normed, target_type=pl.BF16)
                            wk_chunk = pl.slice(wk, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            wv_chunk = pl.slice(wv, [K_CHUNK, KV_OUT_CHUNK], [k0, kv0])
                            k_acc = pl.add(k_acc, pl.matmul(normed_bf16, wk_chunk))
                            v_acc = pl.add(v_acc, pl.matmul(normed_bf16, wv_chunk))
                        k_proj = pl.assemble(k_proj, pl.cast(k_acc, target_type=pl.BF16), [b0, kv0])
                        v_proj = pl.assemble(v_proj, pl.cast(v_acc, target_type=pl.BF16), [b0, kv0])

            # =========================================================================
            # Scope 2: RoPE + KV Cache Update + Flash Decoding Attention
            # =========================================================================
            for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                pos = pl.tensor.read(cache_pos, [b])
                ctx_len = pos + 1
                ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE
                
                # Load RoPE coefficients
                cos_row = pl.slice(rope_cos, [1, HEAD_DIM_CFG], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, HEAD_DIM_CFG], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                cos_hi = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                sin_lo = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                sin_hi = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])

                # RoPE for K/V and update cache
                for kvh in pl.parallel(0, NUM_KV_HEADS_CFG, 1, chunk=4):
                    kv_col = kvh * HEAD_DIM_CFG
                    k_row = pl.cast(
                        pl.slice(k_proj, [1, HEAD_DIM_CFG], [b, kv_col]),
                        target_type=pl.FP32,
                    )
                    k_lo = pl.slice(k_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                    k_hi = pl.slice(k_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                    
                    # Apply RoPE
                    k_rot = pl.create_tensor([1, HEAD_DIM_CFG], dtype=pl.FP32)
                    k_rot = pl.assemble(
                        k_rot,
                        pl.sub(pl.col_expand_mul(k_lo, cos_lo), pl.col_expand_mul(k_hi, sin_lo)),
                        [0, 0],
                    )
                    k_rot = pl.assemble(
                        k_rot,
                        pl.add(pl.col_expand_mul(k_hi, cos_hi), pl.col_expand_mul(k_lo, sin_hi)),
                        [0, HEAD_DIM_CFG // 2],
                    )
                    
                    # Update KV cache
                    cache_row = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + kvh * MAX_SEQ_CFG + pos
                    k_cache = pl.assemble(k_cache, pl.cast(k_rot, target_type=pl.BF16), [cache_row, 0])
                    v_cache = pl.assemble(
                        v_cache,
                        pl.slice(v_proj, [1, HEAD_DIM_CFG], [b, kv_col]),
                        [cache_row, 0],
                    )

                # Flash Decoding Attention per head
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    attn_row = pl.create_tensor([1, HIDDEN_CFG], dtype=pl.FP32)
                    attn_row = pl.mul(attn_row, 0.0)

                    for h in pl.parallel(0, NUM_HEADS_CFG, 1, chunk=8):
                        kvh = h // Q_PER_KV_CFG
                        q_col = h * HEAD_DIM_CFG

                        # RoPE for Q
                        q_row = pl.cast(
                            pl.slice(q_proj, [1, HEAD_DIM_CFG], [b, q_col]),
                            target_type=pl.FP32,
                        )
                        q_lo = pl.slice(q_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                        q_hi = pl.slice(q_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                        q_rot = pl.create_tensor([1, HEAD_DIM_CFG], dtype=pl.FP32)
                        q_rot = pl.assemble(
                            q_rot,
                            pl.sub(pl.col_expand_mul(q_lo, cos_lo), pl.col_expand_mul(q_hi, sin_lo)),
                            [0, 0],
                        )
                        q_rot = pl.assemble(
                            q_rot,
                            pl.add(pl.col_expand_mul(q_hi, cos_hi), pl.col_expand_mul(q_lo, sin_hi)),
                            [0, HEAD_DIM_CFG // 2],
                        )
                        q_rot_bf16 = pl.cast(q_rot, target_type=pl.BF16)

                        # Online softmax state
                        oi = pl.create_tensor([1, HEAD_DIM_CFG], dtype=pl.FP32)
                        li = pl.create_tensor([1, 1], dtype=pl.FP32)
                        mi = pl.create_tensor([1, 1], dtype=pl.FP32)
                        oi = pl.mul(oi, 0.0)
                        li = pl.mul(li, 0.0)
                        mi = pl.mul(mi, 0.0)

                        # Process KV cache in chunks (sliding window)
                        for sb in pl.range(ctx_blocks):
                            s0 = sb * SEQ_TILE
                            valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                            cache_row0 = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + kvh * MAX_SEQ_CFG + s0
                            
                            k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM_CFG], [cache_row0, 0])
                            v_tile = pl.slice(v_cache, [SEQ_TILE, HEAD_DIM_CFG], [cache_row0, 0])
                            
                            # Q @ K^T * scale
                            scores = pl.mul(pl.matmul(q_rot_bf16, k_tile, b_trans=True), ATTN_SCALE)
                            scores_valid = pl.slice(scores, [1, valid_len], [0, 0])
                            
                            # Online softmax
                            cur_mi = pl.cast(pl.row_max(scores_valid), target_type=pl.FP32)
                            exp_scores = pl.exp(pl.row_expand_sub(scores_valid, cur_mi))
                            cur_li = pl.cast(pl.row_sum(exp_scores), target_type=pl.FP32)
                            
                            exp_pad = pl.create_tensor([1, SEQ_TILE], dtype=pl.FP32)
                            exp_pad = pl.mul(exp_pad, 0.0)
                            exp_pad = pl.assemble(exp_pad, exp_scores, [0, 0])
                            
                            oi_tmp = pl.matmul(
                                pl.cast(exp_pad, target_type=pl.BF16),
                                v_tile,
                                out_dtype=pl.FP32,
                            )

                            if sb == 0:
                                oi = oi_tmp
                                li = cur_li
                                mi = cur_mi
                            else:
                                mi_new = pl.maximum(mi, cur_mi)
                                alpha = pl.exp(pl.sub(mi, mi_new))
                                beta = pl.exp(pl.sub(cur_mi, mi_new))
                                li = pl.add(pl.mul(alpha, li), pl.mul(beta, cur_li))
                                oi = pl.add(pl.row_expand_mul(oi, alpha), pl.row_expand_mul(oi_tmp, beta))
                                mi = mi_new

                        ctx = pl.row_expand_div(oi, li)
                        attn_row = pl.assemble(attn_row, ctx, [0, q_col])

                    attn_out = pl.assemble(attn_out, attn_row, [b, 0])

            # =========================================================================
            # Scope 3: Output Projection + Residual + Post RMSNorm + MoE
            # =========================================================================
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    # Output projection + residual
                    resid1_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)

                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=8):
                        o0 = ob * Q_OUT_CHUNK
                        o_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        o_acc = pl.mul(o_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            a_chunk = pl.cast(
                                pl.slice(attn_out, [BATCH_TILE, K_CHUNK], [b0, k0]),
                                target_type=pl.BF16,
                            )
                            w_chunk = pl.slice(wo, [K_CHUNK, Q_OUT_CHUNK], [k0, o0])
                            o_acc = pl.add(o_acc, pl.matmul(a_chunk, w_chunk))
                        resid = pl.cast(
                            pl.slice(hidden_states, [BATCH_TILE, Q_OUT_CHUNK], [b0, o0]),
                            target_type=pl.FP32,
                        )
                        resid1_tile = pl.assemble(resid1_tile, pl.add(o_acc, resid), [0, o0])

                    # Post RMSNorm
                    sq_sum = pl.create_tensor([BATCH_TILE, 1], dtype=pl.FP32)
                    sq_sum = pl.mul(sq_sum, 0.0)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))
                    inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                    post_norm_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.BF16)
                    for kb in pl.range(HIDDEN_BLOCKS):
                        k0 = kb * K_CHUNK
                        x_chunk = pl.slice(resid1_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                        gamma = pl.slice(post_rms_weight, [1, K_CHUNK], [0, k0])
                        normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms), gamma)
                        post_norm_tile = pl.assemble(post_norm_tile, pl.cast(normed, target_type=pl.BF16), [0, k0])

                    # =================================================================
                    # MoE Layer: Shared Expert + Routed Experts
                    # =================================================================
                    moe_out = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    moe_out = pl.mul(moe_out, 0.0)

                    # Compute gating scores (accumulate over HIDDEN_BLOCKS)
                    gate_scores = pl.create_tensor([BATCH_TILE, NUM_EXPERTS_CFG], dtype=pl.FP32)
                    gate_scores = pl.mul(gate_scores, 0.0)
                    for eb in pl.range(NUM_EXPERTS_CFG):
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            gate_chunk = pl.slice(moe_gate, [K_CHUNK, 1], [k0, eb])
                            score = pl.row_sum(pl.mul(pl.cast(post_chunk, target_type=pl.FP32), pl.cast(gate_chunk, target_type=pl.FP32)))
                            # Accumulate score: gate_scores[0, eb] += score
                            prev_score = pl.slice(gate_scores, [BATCH_TILE, 1], [0, eb])
                            gate_scores = pl.assemble(gate_scores, pl.add(prev_score, score), [0, eb])

                    # Softmax over experts and select top-K
                    gate_max = pl.row_max(gate_scores)
                    gate_exp = pl.exp(pl.row_expand_sub(gate_scores, gate_max))
                    gate_sum = pl.row_sum(gate_exp)
                    gate_prob = pl.row_expand_div(gate_exp, gate_sum)

                    # Shared Expert (always active)
                    shared_out = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    shared_out = pl.mul(shared_out, 0.0)
                    
                    for ob in pl.range(MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        gate_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        up_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        gate_acc = pl.mul(gate_acc, 0.0)
                        up_acc = pl.mul(up_acc, 0.0)

                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate_shared, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            wu = pl.slice(w_up_shared, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                            up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                        # SwiGLU activation
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                        for dob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=4):
                            d0 = dob * Q_OUT_CHUNK
                            down_prev = pl.slice(shared_out, [BATCH_TILE, Q_OUT_CHUNK], [0, d0])
                            w_down_chunk = pl.slice(w_down_shared, [MLP_OUT_CHUNK, Q_OUT_CHUNK], [o0, d0])
                            down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                            shared_out = pl.assemble(shared_out, down_next, [0, d0])

                    # Routed Experts (top-K selection)
                    # Simplified: process all experts with gating weight
                    for exp_idx in pl.range(NUM_EXPERTS_CFG):
                        expert_out = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                        expert_out = pl.mul(expert_out, 0.0)
                        
                        for ob in pl.range(MLP_OUT_BLOCKS):
                            o0 = ob * MLP_OUT_CHUNK
                            gate_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                            up_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                            gate_acc = pl.mul(gate_acc, 0.0)
                            up_acc = pl.mul(up_acc, 0.0)

                            for kb in pl.range(HIDDEN_BLOCKS):
                                k0 = kb * K_CHUNK
                                post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                                # Slice 3D expert weights and reshape to 2D
                                wg = pl.slice(w_gate_experts, [NUM_EXPERTS_CFG, K_CHUNK, MLP_OUT_CHUNK], [exp_idx, k0, o0])
                                wu = pl.slice(w_up_experts, [NUM_EXPERTS_CFG, K_CHUNK, MLP_OUT_CHUNK], [exp_idx, k0, o0])
                                # Reshape from [1, K_CHUNK, MLP_OUT_CHUNK] to [K_CHUNK, MLP_OUT_CHUNK]
                                wg = pl.reshape(wg, [K_CHUNK, MLP_OUT_CHUNK])
                                wu = pl.reshape(wu, [K_CHUNK, MLP_OUT_CHUNK])
                                gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                                up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                            sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                            mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                            mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                            for dob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=4):
                                d0 = dob * Q_OUT_CHUNK
                                down_prev = pl.slice(expert_out, [BATCH_TILE, Q_OUT_CHUNK], [0, d0])
                                # Slice and reshape 3D down weights to 2D
                                w_down_chunk = pl.slice(w_down_experts, [NUM_EXPERTS_CFG, MLP_OUT_CHUNK, Q_OUT_CHUNK], [exp_idx, o0, d0])
                                w_down_chunk = pl.reshape(w_down_chunk, [MLP_OUT_CHUNK, Q_OUT_CHUNK])
                                down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                                expert_out = pl.assemble(expert_out, down_next, [0, d0])

                        # Weight by gating probability
                        gate_weight = pl.slice(gate_prob, [BATCH_TILE, 1], [0, exp_idx])
                        for ob in pl.range(Q_OUT_BLOCKS):
                            o0 = ob * Q_OUT_CHUNK
                            prev = pl.slice(moe_out, [BATCH_TILE, Q_OUT_CHUNK], [0, o0])
                            weighted = pl.row_expand_mul(pl.slice(expert_out, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]), gate_weight)
                            moe_out = pl.assemble(moe_out, pl.add(prev, weighted), [0, o0])

                    # Combine shared + routed experts
                    for ob in pl.range(Q_OUT_BLOCKS):
                        o0 = ob * Q_OUT_CHUNK
                        combined = pl.add(pl.slice(shared_out, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]), 
                                         pl.slice(moe_out, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]))
                        final = pl.add(combined, pl.slice(resid1_tile, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]))
                        out = pl.assemble(out, pl.cast(final, target_type=pl.BF16), [b0, o0])

            return out

    return KimiK2Decode


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("Building Kimi K2 Decode Program...")
    program = build_kimi_k2_decode_program()
    print(f"Program: {program}")
    print("Kimi K2 PyPTO implementation ready!")
