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

    python t5_encoder.py -p a2a3 -d 0
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
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm1"):
            partial_sq = pl.full([1, SEQ_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc = pl.slice(x_in, [SEQ_TILE, K_CHUNK], [st, k0])
                partial_sq = pl.add(partial_sq, pl.reshape(pl.row_sum(pl.mul(xc, xc)), [1, SEQ_TILE]))
            var = pl.reshape(pl.add(pl.mul(partial_sq, 1.0 / T5_DIM), EPS), [SEQ_TILE, 1])
            inv = pl.rsqrt(var)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc = pl.slice(x_in, [SEQ_TILE, K_CHUNK], [st, k0])
                g = pl.slice(norm1_w_layer, [1, K_CHUNK], [0, k0])
                n = pl.col_expand_mul(pl.row_expand_mul(xc, inv), g)
                normed_gm = pl.assemble(normed_gm, pl.cast(n, target_type=pl.BF16), [st, k0])

    # ── Scope 2 · Q projection ──
    q_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="q_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                ta = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                tw = pl.slice(q_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                qa = pl.matmul(ta, tw, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    ta_i = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    tw_i = pl.slice(q_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    qa = pl.matmul_acc(qa, ta_i, tw_i, b_trans=True)
                q_gm = pl.assemble(q_gm, pl.cast(qa, target_type=pl.BF16), [st, n0])

    # ── Scope 2 · K projection ──
    k_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="k_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                ta = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                tw = pl.slice(k_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                ka = pl.matmul(ta, tw, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    ta_i = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    tw_i = pl.slice(k_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    ka = pl.matmul_acc(ka, ta_i, tw_i, b_trans=True)
                k_gm = pl.assemble(k_gm, pl.cast(ka, target_type=pl.BF16), [st, n0])

    # ── Scope 2 · V projection ──
    v_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="v_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                ta = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                tw = pl.slice(v_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                va = pl.matmul(ta, tw, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    ta_i = pl.slice(normed_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    tw_i = pl.slice(v_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    va = pl.matmul_acc(va, ta_i, tw_i, b_trans=True)
                v_gm = pl.assemble(v_gm, pl.cast(va, target_type=pl.BF16), [st, n0])

    # ── Scope 3 · Multi-head self-attention (with relative position bias) ──
    ctx_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="mha"):
        for h in pl.range(T5_HEADS):
            hc = h * T5_HEAD_DIM
            qh = pl.slice(q_gm, [T5_SEQ, T5_HEAD_DIM], [0, hc])
            kh = pl.slice(k_gm, [T5_SEQ, T5_HEAD_DIM], [0, hc])
            vh = pl.slice(v_gm, [T5_SEQ, T5_HEAD_DIM], [0, hc])
            sc = pl.matmul(qh, kh, b_trans=True, out_dtype=pl.FP32)
            
            # Add relative position bias for this head
            # pos_bias_h: [T5_SEQ, T5_SEQ] - precomputed on host
            pos_bias_h = pl.slice(pos_bias_layer, [1, T5_SEQ, T5_SEQ], [h, 0, 0])
            pos_bias_h = pl.reshape(pos_bias_h, [T5_SEQ, T5_SEQ])
            # Add to scores (both are FP32)
            sc = pl.add(sc, pos_bias_h)
            
            rm = pl.row_max(sc)
            sh = pl.row_expand_sub(sc, rm)
            es = pl.exp(sh)
            dn = pl.row_sum(es)
            sm = pl.row_expand_div(es, dn)
            sb = pl.cast(sm, target_type=pl.BF16)
            cx = pl.matmul(sb, vh, out_dtype=pl.FP32)
            ctx_gm = pl.assemble(ctx_gm, pl.cast(cx, target_type=pl.BF16), [0, hc])

    # ── Scope 4 · Output projection ──
    attn_out_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="out_proj"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                ta = pl.slice(ctx_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                tw = pl.slice(o_w_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                oa = pl.matmul(ta, tw, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    ta_i = pl.slice(ctx_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    tw_i = pl.slice(o_w_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    oa = pl.matmul_acc(oa, ta_i, tw_i, b_trans=True)
                attn_out_gm = pl.assemble(attn_out_gm, pl.cast(oa, target_type=pl.BF16), [st, n0])

    # ── Scope 5 · Residual add (attention) ──
    x_after_attn_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.FP32)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="residual"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                xa = pl.slice(x_in, [SEQ_TILE, N_CHUNK], [st, n0])
                ao = pl.cast(pl.slice(attn_out_gm, [SEQ_TILE, N_CHUNK], [st, n0]), target_type=pl.FP32)
                xr = pl.add(xa, ao)
                x_after_attn_gm = pl.assemble(x_after_attn_gm, xr, [st, n0])

    # ── Scope 6 · RMSNorm (pre-FFN) ──
    normed2_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm2"):
            partial_sq2 = pl.full([1, SEQ_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc2 = pl.slice(x_after_attn_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                partial_sq2 = pl.add(partial_sq2, pl.reshape(pl.row_sum(pl.mul(xc2, xc2)), [1, SEQ_TILE]))
            var2 = pl.reshape(pl.add(pl.mul(partial_sq2, 1.0 / T5_DIM), EPS), [SEQ_TILE, 1])
            inv2 = pl.rsqrt(var2)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc2 = pl.slice(x_after_attn_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                g2 = pl.slice(norm2_w_layer, [1, K_CHUNK], [0, k0])
                n2 = pl.col_expand_mul(pl.row_expand_mul(xc2, inv2), g2)
                normed2_gm = pl.assemble(normed2_gm, pl.cast(n2, target_type=pl.BF16), [st, k0])

    # ── Scope 7 · FFN gate + fc1 (GELU-tanh activation) ──
    fc1_gm = pl.create_tensor([T5_SEQ, T5_FFN], dtype=pl.BF16)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_gate_fc1"):
            for ob in pl.range(T5_FFN // N_CHUNK):
                n0 = ob * N_CHUNK
                ta_f = pl.slice(normed2_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                tw0 = pl.slice(wi_0_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                ga = pl.matmul(ta_f, tw0, b_trans=True, out_dtype=pl.FP32)
                tw1 = pl.slice(wi_1_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                fa = pl.matmul(ta_f, tw1, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, HIDDEN_BLOCKS):
                    k0 = kb * K_CHUNK
                    ta_fi = pl.slice(normed2_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    tw0i = pl.slice(wi_0_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    ga = pl.matmul_acc(ga, ta_fi, tw0i, b_trans=True)
                    tw1i = pl.slice(wi_1_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    fa = pl.matmul_acc(fa, ta_fi, tw1i, b_trans=True)
                g3 = pl.mul(pl.mul(ga, ga), ga)
                gi = pl.add(ga, pl.mul(g3, 0.044715))
                gs = pl.mul(gi, 0.7978845608)
                g2x = pl.mul(gs, 2.0)
                ge = pl.exp(g2x)
                gt = pl.div(pl.sub(ge, 1.0), pl.add(ge, 1.0))
                go = pl.mul(pl.mul(ga, 0.5), pl.add(gt, 1.0))
                mo = pl.mul(fa, go)
                fc1_gm = pl.assemble(fc1_gm, pl.cast(mo, target_type=pl.BF16), [st, n0])

    # ── Scope 8 · FFN fc2 + residual ──
    x_final_gm = pl.create_tensor([T5_SEQ, T5_DIM], dtype=pl.FP32)
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn_fc2"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                ta_fc2 = pl.slice(fc1_gm, [SEQ_TILE, K_CHUNK], [st, 0])
                tw_fc2 = pl.slice(wo_layer, [N_CHUNK, K_CHUNK], [n0, 0])
                f2 = pl.matmul(ta_fc2, tw_fc2, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, FFN_BLOCKS):
                    k0 = kb * K_CHUNK
                    ta_fci = pl.slice(fc1_gm, [SEQ_TILE, K_CHUNK], [st, k0])
                    tw_fci = pl.slice(wo_layer, [N_CHUNK, K_CHUNK], [n0, k0])
                    f2 = pl.matmul_acc(f2, ta_fci, tw_fci, b_trans=True)
                xa_fc2 = pl.slice(x_after_attn_gm, [SEQ_TILE, N_CHUNK], [st, n0])
                x_final = pl.add(xa_fc2, f2)
                x_final_gm = pl.assemble(x_final_gm, x_final, [st, n0])

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
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_rmsnorm"):
            partial_sq_final = pl.full([1, SEQ_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc_final = pl.slice(cur, [SEQ_TILE, K_CHUNK], [st, k0])
                partial_sq_final = pl.add(partial_sq_final, pl.reshape(pl.row_sum(pl.mul(xc_final, xc_final)), [1, SEQ_TILE]))
            var_final = pl.reshape(pl.add(pl.mul(partial_sq_final, 1.0 / T5_DIM), EPS), [SEQ_TILE, 1])
            inv_final = pl.rsqrt(var_final)
            for kb in pl.range(HIDDEN_BLOCKS):
                k0 = kb * K_CHUNK
                xc_final = pl.slice(cur, [SEQ_TILE, K_CHUNK], [st, k0])
                g_final = pl.slice(final_norm_w, [1, K_CHUNK], [0, k0])
                n_final = pl.col_expand_mul(pl.row_expand_mul(xc_final, inv_final), g_final)
                x_final_normed_gm = pl.assemble(x_final_normed_gm, pl.cast(n_final, target_type=pl.BF16), [st, k0])

    # ── Output copy (BF16) ──
    for st in pl.parallel(0, T5_SEQ, SEQ_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="final_output"):
            for ob in pl.range(T5_DIM // N_CHUNK):
                n0 = ob * N_CHUNK
                xf_chunk = pl.slice(x_final_normed_gm, [SEQ_TILE, N_CHUNK], [st, n0])
                out = pl.assemble(out, xf_chunk, [st, n0])

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Test harness — build_tensor_specs / golden_fn / __main__.
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor_specs():
    from golden import TensorSpec

    torch.manual_seed(42)

    norm1_w_1d = torch.ones(T5_DIM, dtype=torch.float32)
    norm2_w_1d = torch.ones(T5_DIM, dtype=torch.float32)
    final_norm_w_1d = torch.ones(T5_DIM, dtype=torch.float32)

    q_w_fp32 = torch.randn(T5_DIM, T5_DIM, dtype=torch.float32) * 0.02
    k_w_fp32 = torch.randn(T5_DIM, T5_DIM, dtype=torch.float32) * 0.02
    v_w_fp32 = torch.randn(T5_DIM, T5_DIM, dtype=torch.float32) * 0.02
    o_w_fp32 = torch.randn(T5_DIM, T5_DIM, dtype=torch.float32) * 0.02
    rel_pos_bias_fp32 = torch.randn(T5_NUM_BUCKETS, T5_HEADS, dtype=torch.float32) * 0.02
    wi_0_fp32 = torch.randn(T5_FFN, T5_DIM, dtype=torch.float32) * 0.02
    wi_1_fp32 = torch.randn(T5_FFN, T5_DIM, dtype=torch.float32) * 0.02
    wo_fp32 = torch.randn(T5_DIM, T5_FFN, dtype=torch.float32) * 0.02

    norm1_w_stacked = norm1_w_1d.unsqueeze(0).unsqueeze(0).expand(T5_LAYERS, 1, T5_DIM).contiguous()
    norm2_w_stacked = norm2_w_1d.unsqueeze(0).unsqueeze(0).expand(T5_LAYERS, 1, T5_DIM).contiguous()
    final_norm_w_2d = final_norm_w_1d.unsqueeze(0)

    q_w_stacked = q_w_fp32.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    k_w_stacked = k_w_fp32.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    v_w_stacked = v_w_fp32.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    o_w_stacked = o_w_fp32.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_DIM).contiguous()
    wi_0_stacked = wi_0_fp32.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_FFN, T5_DIM).contiguous()
    wi_1_stacked = wi_1_fp32.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_FFN, T5_DIM).contiguous()
    wo_stacked = wo_fp32.bfloat16().unsqueeze(0).expand(T5_LAYERS, T5_DIM, T5_FFN).contiguous()

    pos_emb = T5RelativeEmbedding(rel_pos_bias_fp32, T5_NUM_BUCKETS, T5_HEADS, bidirectional=True)
    pos_bias = pos_emb(T5_SEQ, T5_SEQ).squeeze(0)  # [H, S, S]
    pos_bias_stacked = pos_bias.unsqueeze(0).expand(T5_LAYERS, T5_HEADS, T5_SEQ, T5_SEQ).contiguous()

    return [
        TensorSpec("x_in",         [T5_SEQ, T5_DIM],                       torch.float32,  init_value=torch.randn),
        TensorSpec("norm1_w",      [T5_LAYERS, 1, T5_DIM],                torch.float32,  init_value=norm1_w_stacked),
        TensorSpec("q_w",          [T5_LAYERS, T5_DIM, T5_DIM],           torch.bfloat16, init_value=q_w_stacked),
        TensorSpec("k_w",          [T5_LAYERS, T5_DIM, T5_DIM],           torch.bfloat16, init_value=k_w_stacked),
        TensorSpec("v_w",          [T5_LAYERS, T5_DIM, T5_DIM],           torch.bfloat16, init_value=v_w_stacked),
        TensorSpec("o_w",          [T5_LAYERS, T5_DIM, T5_DIM],           torch.bfloat16, init_value=o_w_stacked),
        TensorSpec("pos_bias",     [T5_LAYERS, T5_HEADS, T5_SEQ, T5_SEQ], torch.float32,  init_value=pos_bias_stacked),
        TensorSpec("norm2_w",      [T5_LAYERS, 1, T5_DIM],                torch.float32,  init_value=norm2_w_stacked),
        TensorSpec("wi_0",         [T5_LAYERS, T5_FFN, T5_DIM],           torch.bfloat16, init_value=wi_0_stacked),
        TensorSpec("wi_1",         [T5_LAYERS, T5_FFN, T5_DIM],           torch.bfloat16, init_value=wi_1_stacked),
        TensorSpec("wo",           [T5_LAYERS, T5_DIM, T5_FFN],           torch.bfloat16, init_value=wo_stacked),
        TensorSpec("final_norm_w", [1, T5_DIM],                           torch.float32,  init_value=final_norm_w_2d),
        TensorSpec("out",          [T5_SEQ, T5_DIM],                      torch.bfloat16, is_output=True),
    ]


def golden_t5_encoder(tensors):
    """Golden reference using precomputed position bias (shared_pos=True).

    Extracts layer-0 weights from the stacked kernel tensors and passes the
    precomputed ``pos_bias`` through ``T5SelfAttention`` with ``shared_pos=True``,
    producing identical results to the per-layer ``shared_pos=False`` path used by
    ``t5_encode`` in ``test_golden_fun_control_full.py``.
    """
    x = tensors["x_in"].unsqueeze(0)  # [1, S, D]

    pos_bias_4d = tensors["pos_bias"][0].unsqueeze(0)  # [1, H, S, S]

    layer_weights = {
        'norm1_w': tensors["norm1_w"][0, 0, :],
        'norm2_w': tensors["norm2_w"][0, 0, :],
        'attn': {
            'q': tensors["q_w"][0].float(),
            'k': tensors["k_w"][0].float(),
            'v': tensors["v_w"][0].float(),
            'o': tensors["o_w"][0].float(),
        },
        'ffn': {
            'wi_0': tensors["wi_0"][0].float(),
            'wi_1': tensors["wi_1"][0].float(),
            'wo': tensors["wo"][0].float(),
        },
    }

    block = T5SelfAttention(
        layer_weights, T5_DIM, T5_DIM, T5_FFN, T5_HEADS, T5_NUM_BUCKETS,
        shared_pos=True, dropout=0.1,
    )
    x = block(x, mask=None, pos_bias=pos_bias_4d)
    x = T5LayerNorm(tensors["final_norm_w"].squeeze(0), T5_DIM)(x)

    tensors["out"][:] = x.squeeze(0).bfloat16()


if __name__ == "__main__":
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=t5_encoder,
        specs=build_tensor_specs(),
        golden_fn=golden_t5_encoder,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1.2e-1,
        atol=1.2e-1,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
