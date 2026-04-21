# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Xiaomi MiLM single-layer decode forward implementation in PyPTO.

MiLM Architecture Highlights:
- Llama-style Transformer with SwiGLU
- RoPE (Rotary Position Embedding)
- RMSNorm (pre-normalization)
- GQA (Grouped Query Attention)
- Optimized for mobile/edge deployment

This implementation focuses on:
- decode only (single token generation)
- single Transformer layer
- batch = 16 by default
- KV cache <= 4096
- Fused kernel operations for efficiency

Reference: Xiaomi AI Lab publications
"""

import pypto.language as pl


# =============================================================================
# MiLM Base Configuration (MiLM-7B like)
# =============================================================================
BATCH = 16
MAX_SEQ = 4096
HIDDEN = 4096
NUM_HEADS = 32
NUM_KV_HEADS = 8  # GQA: 4 query heads per KV head
HEAD_DIM = 128
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
INTERMEDIATE = 11008  # SwiGLU: 2/3 * 4 * hidden
EPS = 1e-6
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
HIDDEN_INV = 1.0 / HIDDEN

# Tile configuration for InCore memory optimization
# Tuned for Ascend NPU A3/A5 architecture
K_CHUNK = 256
Q_OUT_CHUNK = 64
KV_OUT_CHUNK = 32
SEQ_TILE = 128
MLP_OUT_CHUNK = 32
BATCH_TILE = 4


def build_milm_decode_program(
    batch: int = BATCH,
    max_seq_len: int = MAX_SEQ,
    hidden_size: int = HIDDEN,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
    intermediate_size: int = INTERMEDIATE,
):
    """Build Xiaomi MiLM decode program."""
    
    BATCH_CFG = batch
    MAX_SEQ_CFG = max_seq_len
    HIDDEN_CFG = hidden_size
    NUM_HEADS_CFG = num_heads
    NUM_KV_HEADS_CFG = num_kv_heads
    HEAD_DIM_CFG = head_dim
    KV_HIDDEN_CFG = num_kv_heads * head_dim
    INTER_CFG = intermediate_size
    Q_PER_KV_CFG = num_heads // num_kv_heads

    HIDDEN_BLOCKS = (HIDDEN_CFG + K_CHUNK - 1) // K_CHUNK
    Q_OUT_BLOCKS = (HIDDEN_CFG + Q_OUT_CHUNK - 1) // Q_OUT_CHUNK
    KV_OUT_BLOCKS = (KV_HIDDEN_CFG + KV_OUT_CHUNK - 1) // KV_OUT_CHUNK
    MLP_OUT_BLOCKS = (INTER_CFG + MLP_OUT_CHUNK - 1) // MLP_OUT_CHUNK
    CACHE_ROWS = BATCH_CFG * NUM_KV_HEADS_CFG * MAX_SEQ_CFG

    @pl.program
    class MiLMDecode:
        @pl.function(type=pl.FunctionType.Opaque)
        def milm_decode_layer(
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
            # MLP weights (SwiGLU)
            w_gate: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_up: pl.Tensor[[HIDDEN_CFG, INTER_CFG], pl.BF16],
            w_down: pl.Tensor[[INTER_CFG, HIDDEN_CFG], pl.BF16],
            out: pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16],
        ) -> pl.Tensor[[BATCH_CFG, HIDDEN_CFG], pl.BF16]:
            """
            Xiaomi MiLM decode layer.
            
            Flow:
            1. Input RMSNorm
            2. QKV Projection
            3. RoPE + KV Cache Update
            4. Flash Decoding Attention (GQA)
            5. Output Projection + Residual
            6. Post RMSNorm
            7. SwiGLU MLP
            8. Final Residual
            """
            q_proj = pl.create_tensor([BATCH_CFG, HIDDEN_CFG], dtype=pl.BF16)
            k_proj = pl.create_tensor([BATCH_CFG, KV_HIDDEN_CFG], dtype=pl.BF16)
            v_proj = pl.create_tensor([BATCH_CFG, KV_HIDDEN_CFG], dtype=pl.BF16)
            attn_out = pl.create_tensor([BATCH_CFG, HIDDEN_CFG], dtype=pl.FP32)

            # =========================================================================
            # Scope 1: Input RMSNorm + QKV Projection
            # Optimized with chunked computation to reduce InCore pressure
            # =========================================================================
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                # Compute sum of squares for RMSNorm
                sq_sum = pl.create_tensor([BATCH_CFG, 1], dtype=pl.FP32)
                sq_sum = pl.mul(sq_sum, 0.0)

                for kb in pl.range(HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    x_chunk = pl.cast(
                        pl.slice(hidden_states, [BATCH_CFG, K_CHUNK], [0, k0]),
                        target_type=pl.FP32,
                    )
                    sq_sum = pl.add(sq_sum, pl.row_sum(pl.mul(x_chunk, x_chunk)))

                # RMSNorm: 1/sqrt(sum(x^2)/n + eps)
                inv_rms = pl.rsqrt(pl.add(pl.mul(sq_sum, HIDDEN_INV), EPS))

                # QKV projection with batch tiling
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    inv_rms_tile = pl.slice(inv_rms, [BATCH_TILE, 1], [b0, 0])

                    # Q projection (parallel over output chunks)
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=4):
                        q0 = ob * Q_OUT_CHUNK
                        q_acc = pl.create_tensor([BATCH_TILE, Q_OUT_CHUNK], dtype=pl.FP32)
                        q_acc = pl.mul(q_acc, 0.0)
                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            x_chunk_bf16 = pl.slice(hidden_states, [BATCH_TILE, K_CHUNK], [b0, k0])
                            x_chunk = pl.cast(x_chunk_bf16, target_type=pl.FP32)
                            gamma = pl.slice(input_rms_weight, [1, K_CHUNK], [0, k0])
                            # RMSNorm: (x / rms) * gamma
                            normed = pl.col_expand_mul(pl.row_expand_mul(x_chunk, inv_rms_tile), gamma)
                            wq_chunk = pl.slice(wq, [K_CHUNK, Q_OUT_CHUNK], [k0, q0])
                            q_acc = pl.add(q_acc, pl.matmul(pl.cast(normed, target_type=pl.BF16), wq_chunk))
                        q_proj = pl.assemble(q_proj, pl.cast(q_acc, target_type=pl.BF16), [b0, q0])

                    # K/V projection (parallel over output chunks)
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
                
                # Load RoPE coefficients for current position
                cos_row = pl.slice(rope_cos, [1, HEAD_DIM_CFG], [pos, 0])
                sin_row = pl.slice(rope_sin, [1, HEAD_DIM_CFG], [pos, 0])
                cos_lo = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                cos_hi = pl.slice(cos_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                sin_lo = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                sin_hi = pl.slice(sin_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])

                # Apply RoPE to K/V and update cache
                for kvh in pl.parallel(0, NUM_KV_HEADS_CFG, 1, chunk=4):
                    kv_col = kvh * HEAD_DIM_CFG
                    k_row = pl.cast(
                        pl.slice(k_proj, [1, HEAD_DIM_CFG], [b, kv_col]),
                        target_type=pl.FP32,
                    )
                    k_lo = pl.slice(k_row, [1, HEAD_DIM_CFG // 2], [0, 0])
                    k_hi = pl.slice(k_row, [1, HEAD_DIM_CFG // 2], [0, HEAD_DIM_CFG // 2])
                    
                    # RoPE: [k_lo, k_hi] -> [k_lo*cos - k_hi*sin, k_hi*cos + k_lo*sin]
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

                # Flash Decoding Attention (per head with GQA)
                with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                    attn_row = pl.create_tensor([1, HIDDEN_CFG], dtype=pl.FP32)
                    attn_row = pl.mul(attn_row, 0.0)

                    for h in pl.parallel(0, NUM_HEADS_CFG, 1, chunk=8):
                        kvh = h // Q_PER_KV_CFG  # GQA: multiple Q heads share one KV head
                        q_col = h * HEAD_DIM_CFG

                        # Apply RoPE to Q
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

                        # Online softmax state for Flash Decoding
                        oi = pl.create_tensor([1, HEAD_DIM_CFG], dtype=pl.FP32)
                        li = pl.create_tensor([1, 1], dtype=pl.FP32)
                        mi = pl.create_tensor([1, 1], dtype=pl.FP32)
                        oi = pl.mul(oi, 0.0)
                        li = pl.mul(li, 0.0)
                        mi = pl.mul(mi, 0.0)

                        # Process KV cache in chunks
                        for sb in pl.range(ctx_blocks):
                            s0 = sb * SEQ_TILE
                            valid_len = pl.min(SEQ_TILE, ctx_len - s0)
                            cache_row0 = b * NUM_KV_HEADS_CFG * MAX_SEQ_CFG + kvh * MAX_SEQ_CFG + s0
                            
                            k_tile = pl.slice(k_cache, [SEQ_TILE, HEAD_DIM_CFG], [cache_row0, 0])
                            v_tile = pl.slice(v_cache, [SEQ_TILE, HEAD_DIM_CFG], [cache_row0, 0])
                            
                            # Q @ K^T * scale
                            scores = pl.mul(pl.matmul(q_rot_bf16, k_tile, b_trans=True), ATTN_SCALE)
                            scores_valid = pl.slice(scores, [1, valid_len], [0, 0])
                            
                            # Online softmax (numerically stable)
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
            # Scope 3: Output Projection + Residual + Post RMSNorm + SwiGLU MLP
            # =========================================================================
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for b0 in pl.range(0, BATCH_CFG, BATCH_TILE):
                    # Output projection + residual (first residual connection)
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

                    # Post RMSNorm (before MLP)
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

                    # SwiGLU MLP: down(silu(gate(x)) * up(x))
                    down_proj_tile = pl.create_tensor([BATCH_TILE, HIDDEN_CFG], dtype=pl.FP32)
                    down_proj_tile = pl.mul(down_proj_tile, 0.0)

                    for ob in pl.range(MLP_OUT_BLOCKS):
                        o0 = ob * MLP_OUT_CHUNK
                        gate_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        up_acc = pl.create_tensor([BATCH_TILE, MLP_OUT_CHUNK], dtype=pl.FP32)
                        gate_acc = pl.mul(gate_acc, 0.0)
                        up_acc = pl.mul(up_acc, 0.0)

                        for kb in pl.range(HIDDEN_BLOCKS):
                            k0 = kb * K_CHUNK
                            post_chunk = pl.slice(post_norm_tile, [BATCH_TILE, K_CHUNK], [0, k0])
                            wg = pl.slice(w_gate, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            wu = pl.slice(w_up, [K_CHUNK, MLP_OUT_CHUNK], [k0, o0])
                            gate_acc = pl.add(gate_acc, pl.matmul(post_chunk, wg))
                            up_acc = pl.add(up_acc, pl.matmul(post_chunk, wu))

                        # SiLU activation: sigmoid(x) * x
                        sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_acc)), 1.0))
                        mlp_chunk = pl.mul(pl.mul(gate_acc, sigmoid), up_acc)
                        mlp_chunk_bf16 = pl.cast(mlp_chunk, target_type=pl.BF16)

                        # Down projection
                        for dob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=4):
                            d0 = dob * Q_OUT_CHUNK
                            down_prev = pl.slice(down_proj_tile, [BATCH_TILE, Q_OUT_CHUNK], [0, d0])
                            w_down_chunk = pl.slice(w_down, [MLP_OUT_CHUNK, Q_OUT_CHUNK], [o0, d0])
                            down_next = pl.add(down_prev, pl.matmul(mlp_chunk_bf16, w_down_chunk))
                            down_proj_tile = pl.assemble(down_proj_tile, down_next, [0, d0])

                    # Final residual connection
                    for ob in pl.parallel(0, Q_OUT_BLOCKS, 1, chunk=4):
                        o0 = ob * Q_OUT_CHUNK
                        down_acc = pl.add(
                            pl.slice(down_proj_tile, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]),
                            pl.slice(resid1_tile, [BATCH_TILE, Q_OUT_CHUNK], [0, o0]),
                        )
                        out = pl.assemble(out, pl.cast(down_acc, target_type=pl.BF16), [b0, o0])

            return out

    return MiLMDecode


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    print("Building Xiaomi MiLM Decode Program...")
    program = build_milm_decode_program()
    print(f"Program: {program}")
    print("Xiaomi MiLM PyPTO implementation ready!")
