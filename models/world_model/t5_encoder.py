# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""T5 Encoder — pypto3.0 kernel for the T5 text encoder sub-network.

Implements the full T5 encoder stack: RMSNorm → Self-Attention (with relative
position bias) → Residual → RMSNorm → Gated GELU-tanh FFN → Residual, repeated
for T5_LAYERS layers, followed by a final RMSNorm.

The golden reference uses ``T5LayerNorm``, ``T5RelativeEmbedding``, and
``T5SelfAttention`` from ``test_golden_fun_control_full.py``, which mirrors
the T5 encoder in ``infer_fun_control_1_3b_text.py`` at reduced dimensions.

Usage::

    python t5_encoder.py    # golden-case precision test (T5_DIM=128)
"""

import argparse
import sys

import pypto.language as pl
import torch

from config import (
    T5_DIM, T5_FFN, T5_HEADS, T5_HEAD_DIM, T5_LAYERS, T5_NUM_BUCKETS, T5_SEQ,
)

# ── Golden reference (shared classes with infer_fun_control_1_3b_text.py) ──
sys.path.insert(0, '/data/x00952168/pypto3.0/cann-recipes-embodied-ai/world_model/agibot-arm-world-model/infer_with_torch')
from test_golden_fun_control_full import (  # noqa: E402
    T5LayerNorm, T5RelativeEmbedding, T5SelfAttention,
)


# ══════════════════════════════════════════════════════════════════════════════
# Optimization config — tiling parameters per functional stage.
# ══════════════════════════════════════════════════════════════════════════════

EPS = 1e-6  # RMSNorm epsilon

# ── Tiling geometry ──
# SEQ_TILE = row tile for the seq-length dimension (pl.parallel step).
# K_CHUNK = column tile for the hidden-dimension loops (K dimension).
# N_CHUNK = column tile for the output-dimension matmul loops (N dimension).
SEQ_TILE = 16
K_CHUNK = 512
N_CHUNK = 64

# Dynamic chunking — scale down for the golden case (T5_DIM=128) so loops
# don't exceed the dimension bounds.
if T5_DIM >= 4096:  # Real case (t5_umt5-xxl)
    pass  # defaults above
else:  # Golden case (T5_DIM=128)
    K_CHUNK = 128

# ── Derived block counts ──
HIDDEN_BLOCKS = T5_DIM // K_CHUNK       # dim splits for RMSNorm / QKV / out_proj
FFN_BLOCKS = T5_FFN // K_CHUNK          # dim splits for the fc2 (down) matmul

# Geometry assertions — keep at the bottom of the config block.
assert T5_DIM % T5_HEADS == 0, "T5_HEADS must divide T5_DIM"
assert T5_HEAD_DIM * T5_HEADS == T5_DIM
assert T5_DIM % K_CHUNK == 0, "K_CHUNK must divide T5_DIM"
assert T5_DIM % N_CHUNK == 0, "N_CHUNK must divide T5_DIM"
assert T5_FFN % K_CHUNK == 0, "K_CHUNK must divide T5_FFN"
assert T5_FFN % N_CHUNK == 0, "N_CHUNK must divide T5_FFN"
assert T5_SEQ % SEQ_TILE == 0, "SEQ_TILE must divide T5_SEQ"


@pl.jit.inline
def _rmsnorm(
    x_in: pl.Tensor[[T5_SEQ, T5_DIM], pl.FP32],
    weight: pl.Tensor[[1, T5_DIM], pl.FP32],
    out: pl.Tensor[[T5_SEQ, T5_DIM], pl.BF16],
) -> pl.Tensor[[T5_SEQ, T5_DIM], pl.BF16]:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.  FP32 in → BF16 out."""
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm"):
            partial_sq = pl.full([1, SEQ_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc = pl.slice(x_in, [SEQ_TILE, K_CHUNK], [st, k0])
                partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(xc, xc)), [1, SEQ_TILE]))
            var = pl.reshape(pl.add(pl.mul(partial_sq, 1.0 / T5_DIM), EPS), [SEQ_TILE, 1])
            inv = pl.recip(pl.sqrt(var))
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc = pl.slice(x_in, [SEQ_TILE, K_CHUNK], [st, k0])
                g = pl.slice(weight, [1, K_CHUNK], [0, k0])
                n = pl.col_expand_mul(pl.row_expand_mul(xc, inv), g)
                out = pl.assemble(out, pl.cast(n, target_type=pl.BF16), [st, k0])
    return out


@pl.jit.inline
def t5_encoder_layer(
    x_in: pl.Tensor[[T5_SEQ, T5_DIM], pl.FP32],
    norm1_w: pl.Tensor[[T5_LAYERS, 1, T5_DIM], pl.FP32],
    q_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    k_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    v_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    o_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    pos_bias: pl.Tensor[[T5_LAYERS, T5_HEADS, T5_SEQ, T5_SEQ], pl.FP32],
    norm2_w: pl.Tensor[[T5_LAYERS, 1, T5_DIM], pl.FP32],
    wi_0: pl.Tensor[[T5_LAYERS, T5_FFN, T5_DIM], pl.BF16],
    wi_1: pl.Tensor[[T5_LAYERS, T5_FFN, T5_DIM], pl.BF16],
    wo: pl.Tensor[[T5_LAYERS, T5_DIM, T5_FFN], pl.BF16],
    out: pl.Tensor[[T5_SEQ, T5_DIM], pl.FP32],
    layer_idx: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T5_SEQ, T5_DIM], pl.FP32]:
    # ── Per-layer weight slicing ──
    norm1_w_layer = pl.slice(norm1_w, [1, 1, T5_DIM], [layer_idx, 0, 0])
    norm1_w_layer = pl.reshape(norm1_w_layer, [1, T5_DIM])
    q_w_layer = pl.slice(q_w, [1, T5_DIM, T5_DIM], [layer_idx, 0, 0])
    q_w_layer = pl.reshape(q_w_layer, [T5_DIM, T5_DIM])
    k_w_layer = pl.slice(k_w, [1, T5_DIM, T5_DIM], [layer_idx, 0, 0])
    k_w_layer = pl.reshape(k_w_layer, [T5_DIM, T5_DIM])
    v_w_layer = pl.slice(v_w, [1, T5_DIM, T5_DIM], [layer_idx, 0, 0])
    v_w_layer = pl.reshape(v_w_layer, [T5_DIM, T5_DIM])
    o_w_layer = pl.slice(o_w, [1, T5_DIM, T5_DIM], [layer_idx, 0, 0])
    o_w_layer = pl.reshape(o_w_layer, [T5_DIM, T5_DIM])
    pos_bias_layer = pl.slice(pos_bias, [1, T5_HEADS, T5_SEQ, T5_SEQ], [layer_idx, 0, 0, 0])
    pos_bias_layer = pl.reshape(pos_bias_layer, [T5_HEADS, T5_SEQ, T5_SEQ])
    norm2_w_layer = pl.slice(norm2_w, [1, 1, T5_DIM], [layer_idx, 0, 0])
    norm2_w_layer = pl.reshape(norm2_w_layer, [1, T5_DIM])
    wi_0_layer = pl.slice(wi_0, [1, T5_FFN, T5_DIM], [layer_idx, 0, 0])
    wi_0_layer = pl.reshape(wi_0_layer, [T5_FFN, T5_DIM])
    wi_1_layer = pl.slice(wi_1, [1, T5_FFN, T5_DIM], [layer_idx, 0, 0])
    wi_1_layer = pl.reshape(wi_1_layer, [T5_FFN, T5_DIM])
    wo_layer = pl.slice(wo, [1, T5_DIM, T5_FFN], [layer_idx, 0, 0])
    wo_layer = pl.reshape(wo_layer, [T5_DIM, T5_FFN])

    # ── Scope 1 · RMSNorm (pre-attention) ──
    normed_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    normed_gm = _rmsnorm(x_in, norm1_w_layer, normed_gm)

    # ── Scope 2a · Q projection ──
    q_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                act_tile = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                w_chunk = pl.slice(q_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                q_acc = pl.matmul(act_tile, w_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    act_chunk = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    w_chunk_i = pl.slice(q_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    q_acc = pl.matmul_acc(q_acc, act_chunk, w_chunk_i, b_trans=True)
                q_gm = pl.assemble(q_gm, pl.cast(q_acc, target_type=pl.BF16), [st, n0])

    # ── Scope 2b · K projection ──
    k_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                act_tile = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                w_chunk = pl.slice(k_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                k_acc = pl.matmul(act_tile, w_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    act_chunk = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    w_chunk_i = pl.slice(k_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    k_acc = pl.matmul_acc(k_acc, act_chunk, w_chunk_i, b_trans=True)
                k_gm = pl.assemble(k_gm, pl.cast(k_acc, target_type=pl.BF16), [st, n0])

    # ── Scope 2c · V projection ──
    v_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                act_tile = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                w_chunk = pl.slice(v_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                v_acc = pl.matmul(act_tile, w_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    act_chunk = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    w_chunk_i = pl.slice(v_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    v_acc = pl.matmul_acc(v_acc, act_chunk, w_chunk_i, b_trans=True)
                v_gm = pl.assemble(v_gm, pl.cast(v_acc, target_type=pl.BF16), [st, n0])

    # ── Scope 3 · Multi-head self-attention (with relative position bias) ──
    ctx_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mha"):
        for h in pl.range(T5_HEADS):
            head_col = h * T5_HEAD_DIM
            q_head = pl.slice(q_gm, [T5_SEQ, T5_HEAD_DIM], [0, head_col])
            k_head = pl.slice(k_gm, [T5_SEQ, T5_HEAD_DIM], [0, head_col])
            v_head = pl.slice(v_gm, [T5_SEQ, T5_HEAD_DIM], [0, head_col])
            raw_scores = pl.matmul(q_head, k_head, b_trans=True, out_dtype=pl.FP32)
            
            # Add relative position bias for this head
            # pos_bias_h: [T5_SEQ, T5_SEQ] - precomputed on host
            pos_bias_h = pl.slice(pos_bias_layer, [1, T5_SEQ, T5_SEQ], [h, 0, 0])
            pos_bias_h = pl.reshape(pos_bias_h, [T5_SEQ, T5_SEQ])
            # Add to scores (both are FP32)
            raw_scores = pl.add(raw_scores, pos_bias_h)
            
            row_max_val = pl.row_max(raw_scores)
            shifted  = pl.row_expand_sub(raw_scores, row_max_val)
            exp_scores = pl.exp(shifted)
            exp_sum    = pl.row_sum(exp_scores)
            attn_weights = pl.row_expand_div(exp_scores, exp_sum)
            attn_weights_bf = pl.cast(attn_weights, target_type=pl.BF16)
            head_out = pl.matmul(attn_weights_bf, v_head, out_dtype=pl.FP32)
            ctx_gm = pl.assemble(ctx_gm, pl.cast(head_out, target_type=pl.BF16), [0, head_col])

    # ── Scope 4 · Output projection ──
    attn_out_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                ctx_tile = pl.slice(ctx_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                o_w_chunk = pl.slice(o_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                oproj_acc = pl.matmul(ctx_tile, o_w_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    ctx_chunk = pl.slice(ctx_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    o_w_chunk_i = pl.slice(o_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    oproj_acc = pl.matmul_acc(oproj_acc, ctx_chunk, o_w_chunk_i, b_trans=True)
                attn_out_gm = pl.assemble(attn_out_gm, pl.cast(oproj_acc, target_type=pl.BF16), [st, n0])

    # ── Scope 5 · Residual add (attention) ──
    x_after_attn_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.FP32)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                orig_tile = pl.slice(x_in, [SEQ_TILE, N_CHUNK], [st, n0])
                attn_out_fp32 = pl.cast(pl.slice(attn_out_gm, [SEQ_TILE, N_CHUNK], [st, n0]), target_type=pl.FP32)
                attention_residual = pl.add(orig_tile, attn_out_fp32)
                x_after_attn_gm = pl.assemble(x_after_attn_gm, attention_residual, [st, n0])

    # ── Scope 6 · RMSNorm (pre-FFN) ──
    normed2_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    normed2_gm = _rmsnorm(x_after_attn_gm, norm2_w_layer, normed2_gm)

    # ── Scope 7 · FFN gate + fc1 (GELU-tanh activation) ──
    fc1_gm = pl.create_tensor([T5_SEQ, T5_FFN], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_gate_fc1"):
            for ob in pl.range(T5_FFN // N_CHUNK):
                n0 = ob * N_CHUNK
                ffn_act_tile = pl.slice(normed2_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                gate_w_chunk = pl.slice(wi_0_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                gate_acc = pl.matmul(ffn_act_tile, gate_w_chunk, b_trans=True, out_dtype=pl.FP32)
                fc1_w_chunk = pl.slice(wi_1_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                fc1_acc = pl.matmul(ffn_act_tile, fc1_w_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    act_chunk = pl.slice(normed2_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    gate_w_chunk_i = pl.slice(wi_0_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    gate_acc = pl.matmul_acc(gate_acc, act_chunk, gate_w_chunk_i, b_trans=True)
                    fc1_w_chunk_i = pl.slice(wi_1_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    fc1_acc = pl.matmul_acc(fc1_acc, act_chunk, fc1_w_chunk_i, b_trans=True)
                # GELU-tanh: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                x_cubed     = pl.mul(pl.mul(gate_acc, gate_acc), gate_acc)
                gelu_inner  = pl.add(gate_acc, pl.mul(x_cubed, 0.044715))
                gelu_scaled = pl.mul(gelu_inner, 0.7978845608)
                gelu_2z     = pl.mul(gelu_scaled, 2.0)
                gelu_exp    = pl.exp(gelu_2z)
                tanh_val    = pl.div(pl.sub(gelu_exp, 1.0), pl.add(gelu_exp, 1.0))
                gelu_out    = pl.mul(pl.mul(gate_acc, 0.5), pl.add(tanh_val, 1.0))
                gated_fc1   = pl.mul(fc1_acc, gelu_out)
                fc1_gm = pl.assemble(fc1_gm, pl.cast(gated_fc1, target_type=pl.BF16), [st, n0])

    # ── Scope 8 · FFN fc2 + residual ──
    x_final_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.FP32)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_fc2"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                fc1_out_tile = pl.slice(fc1_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                fc2_w_chunk = pl.slice(wo_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                fc2_acc = pl.matmul(fc1_out_tile, fc2_w_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, FFN_BLOCKS):
                    k0 = kb * K_CHUNK
                    fc1_out_chunk = pl.slice(fc1_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    fc2_w_chunk_i = pl.slice(wo_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    fc2_acc = pl.matmul_acc(fc2_acc, fc1_out_chunk, fc2_w_chunk_i, b_trans=True)
                attn_residual = pl.slice(x_after_attn_gm, [SEQ_TILE, N_CHUNK], [st, n0])
                ffn_residual  = pl.add(attn_residual, fc2_acc)
                x_final_gm = pl.assemble(x_final_gm, ffn_residual, [st, n0])

    out = pl.assemble(out, x_final_gm, [0, 0])
    return out


@pl.jit
def t5_encoder(
    x_in: pl.Tensor[[T5_SEQ, T5_DIM], pl.FP32],
    norm1_w: pl.Tensor[[T5_LAYERS, 1, T5_DIM], pl.FP32],
    q_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    k_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    v_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    o_w: pl.Tensor[[T5_LAYERS, T5_DIM, T5_DIM], pl.BF16],
    pos_bias: pl.Tensor[[T5_LAYERS, T5_HEADS, T5_SEQ, T5_SEQ], pl.FP32],
    norm2_w: pl.Tensor[[T5_LAYERS, 1, T5_DIM], pl.FP32],
    wi_0: pl.Tensor[[T5_LAYERS, T5_FFN, T5_DIM], pl.BF16],
    wi_1: pl.Tensor[[T5_LAYERS, T5_FFN, T5_DIM], pl.BF16],
    wo: pl.Tensor[[T5_LAYERS, T5_DIM, T5_FFN], pl.BF16],
    final_norm_w: pl.Tensor[[1, T5_DIM], pl.FP32],
    out: pl.Out[pl.Tensor[[T5_SEQ, T5_DIM], pl.BF16]],
):
    # ── Copy input → internal tensor ──
    cur = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.FP32)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_input"):
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc = pl.slice(x_in, [SEQ_TILE, K_CHUNK], [st, k0])
                cur = pl.assemble(cur, xc, [st, k0])
    
    # ── Layer loop ──
    for layer_idx in pl.range(T5_LAYERS):
        next_hidden = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.FP32)
        cur = t5_encoder_layer(
            cur, norm1_w, q_w, k_w, v_w, o_w, pos_bias, norm2_w,
            wi_0, wi_1, wo, next_hidden, layer_idx
        )
    
    # ── Final RMSNorm ──
    x_final_normed_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    x_final_normed_gm = _rmsnorm(cur, final_norm_w, x_final_normed_gm)

    # ── Output copy (BF16) ──
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_output"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                xf_chunk = pl.slice(x_final_normed_gm, [SEQ_TILE, N_CHUNK], [st, n0])
                out = pl.assemble(out, xf_chunk, [st, n0])

    return out


def _golden_t5_encoder(x, norm1_w, q_w, k_w, v_w, o_w, rel_pos_bias, norm2_w, wi_0, wi_1, wo, final_norm_w):
    """Golden reference using the EXACT same classes as test_golden_fun_control_full.py."""
    layer_weights = {
        'norm1_w': norm1_w,
        'norm2_w': norm2_w,
        'attn': {'q': q_w, 'k': k_w, 'v': v_w, 'o': o_w},
        'ffn': {'wi_0': wi_0, 'wi_1': wi_1, 'wo': wo},
        'pos_embedding': rel_pos_bias,
    }
    block = T5SelfAttention(
        layer_weights, T5_DIM, T5_DIM, T5_FFN, T5_HEADS, T5_NUM_BUCKETS,
        shared_pos=False, dropout=0.1,
    )
    x = block(x, mask=None, pos_bias=None)
    x = T5LayerNorm(final_norm_w, T5_DIM)(x)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Test harness — build_tensor_specs / golden_fn / __main__.
# ══════════════════════════════════════════════════════════════════════════════

_GOLDEN_DATA = None  # populated by build_tensor_specs, consumed by golden_t5_encoder_fn


def build_tensor_specs():
    global _GOLDEN_DATA
    from golden import TensorSpec

    g = torch.Generator().manual_seed(42)

    norm1_w_1d = torch.ones(T5_DIM, dtype=torch.float32)
    norm2_w_1d = torch.ones(T5_DIM, dtype=torch.float32)
    final_norm_w_1d = torch.ones(T5_DIM, dtype=torch.float32)

    def rn(shape):
        return torch.empty(shape).normal_(generator=g) * 0.02

    q_w = rn([T5_DIM, T5_DIM])
    k_w = rn([T5_DIM, T5_DIM])
    v_w = rn([T5_DIM, T5_DIM])
    o_w = rn([T5_DIM, T5_DIM])
    rel_pos_bias = rn([T5_NUM_BUCKETS, T5_HEADS])
    wi_0 = rn([T5_FFN, T5_DIM])
    wi_1 = rn([T5_FFN, T5_DIM])
    wo = rn([T5_DIM, T5_FFN])

    x_batch = torch.empty(1, T5_SEQ, T5_DIM).normal_(generator=g)
    x_flat = x_batch.squeeze(0).contiguous()

    norm1_w_stacked = norm1_w_1d.unsqueeze(0).unsqueeze(0).expand(T5_LAYERS, 1, T5_DIM).contiguous()
    norm2_w_stacked = norm2_w_1d.unsqueeze(0).unsqueeze(0).expand(T5_LAYERS, 1, T5_DIM).contiguous()
    final_norm_w_2d = final_norm_w_1d.unsqueeze(0)

    q_w_stacked = q_w.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    k_w_stacked = k_w.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    v_w_stacked = v_w.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    o_w_stacked = o_w.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    wi_0_stacked = wi_0.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_FFN, T5_DIM).contiguous()
    wi_1_stacked = wi_1.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_FFN, T5_DIM).contiguous()
    wo_stacked = wo.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_FFN).contiguous()

    pos_emb = T5RelativeEmbedding(rel_pos_bias, T5_NUM_BUCKETS, T5_HEADS, bidirectional=True)
    pos_bias = pos_emb(T5_SEQ, T5_SEQ).squeeze(0)
    pos_bias_stacked = pos_bias.unsqueeze(0).expand(T5_LAYERS, T5_HEADS, T5_SEQ, T5_SEQ).contiguous()

    def _identity(t):
        return lambda: t

    specs = [
        TensorSpec("x_in",       [T5_SEQ, T5_DIM],                        torch.float32,  init_value=_identity(x_flat)),
        TensorSpec("norm1_w",    [T5_LAYERS, 1, T5_DIM],                  torch.float32,  init_value=_identity(norm1_w_stacked)),
        TensorSpec("q_w",        [T5_LAYERS, T5_DIM, T5_DIM],             torch.bfloat16, init_value=_identity(q_w_stacked)),
        TensorSpec("k_w",        [T5_LAYERS, T5_DIM, T5_DIM],             torch.bfloat16, init_value=_identity(k_w_stacked)),
        TensorSpec("v_w",        [T5_LAYERS, T5_DIM, T5_DIM],             torch.bfloat16, init_value=_identity(v_w_stacked)),
        TensorSpec("o_w",        [T5_LAYERS, T5_DIM, T5_DIM],             torch.bfloat16, init_value=_identity(o_w_stacked)),
        TensorSpec("pos_bias",   [T5_LAYERS, T5_HEADS, T5_SEQ, T5_SEQ],   torch.float32,  init_value=_identity(pos_bias_stacked)),
        TensorSpec("norm2_w",    [T5_LAYERS, 1, T5_DIM],                  torch.float32,  init_value=_identity(norm2_w_stacked)),
        TensorSpec("wi_0",       [T5_LAYERS, T5_FFN, T5_DIM],             torch.bfloat16, init_value=_identity(wi_0_stacked)),
        TensorSpec("wi_1",       [T5_LAYERS, T5_FFN, T5_DIM],             torch.bfloat16, init_value=_identity(wi_1_stacked)),
        TensorSpec("wo",         [T5_LAYERS, T5_DIM, T5_FFN],             torch.bfloat16, init_value=_identity(wo_stacked)),
        TensorSpec("final_norm_w", [1, T5_DIM],                           torch.float32,  init_value=_identity(final_norm_w_2d)),
    ]
    specs.append(TensorSpec("out", [T5_SEQ, T5_DIM], torch.bfloat16, is_output=True))

    _GOLDEN_DATA = {
        "x_batch": x_batch, "norm1_w": norm1_w_1d, "q_w": q_w, "k_w": k_w,
        "v_w": v_w, "o_w": o_w, "rel_pos_bias": rel_pos_bias,
        "norm2_w": norm2_w_1d, "wi_0": wi_0, "wi_1": wi_1, "wo": wo,
        "final_norm_w": final_norm_w_1d,
    }
    return specs


def golden_t5_encoder_fn(tensors):
    """Run golden T5 encoder and fill tensors['out']."""
    x_expected = _golden_t5_encoder(
        _GOLDEN_DATA["x_batch"], _GOLDEN_DATA["norm1_w"],
        _GOLDEN_DATA["q_w"], _GOLDEN_DATA["k_w"], _GOLDEN_DATA["v_w"], _GOLDEN_DATA["o_w"],
        _GOLDEN_DATA["rel_pos_bias"], _GOLDEN_DATA["norm2_w"],
        _GOLDEN_DATA["wi_0"], _GOLDEN_DATA["wi_1"], _GOLDEN_DATA["wo"],
        _GOLDEN_DATA["final_norm_w"],
    )
    tensors["out"][:] = x_expected.squeeze(0).bfloat16()


if __name__ == "__main__":
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="T5 Encoder pypto3.0 kernel test")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    specs = build_tensor_specs()

    result = run_jit(
        fn=t5_encoder,
        specs=specs,
        golden_fn=golden_t5_encoder_fn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=3e-3,
        atol=3e-3,
        compare_fn={"out": ratio_allclose(atol=3e-3, rtol=3e-3, max_error_ratio=0.02)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
