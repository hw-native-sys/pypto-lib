# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CLIP image encoder for Fun-Control 1.3B — pypto3.0 implementation.

Computation logic follows ``test_golden_fun_control_full.py::clip_encode``
exactly, using **QuickGELU** activation: ``x * sigmoid(1.702 * x)``.

Architecture::

    PatchConv2d → CLS+PosEmbed → PreLN →
    [LN → MHA → Proj → Res → LN → FFN(QuickGELU) → Res] × L

Layer count is parameterised via ``CLIP_LAYERS``; weights are concatenated into
2-D tensors and sliced per layer inside a ``pl.unroll`` loop (compile-time
expansion avoids GM buffer dependency issues with ``pl.range``).

Hardware constraints (Ascend 910B / A2A3):
  • Tile rows must be a multiple of ``innerRows = 16`` → ``T = 16`` (pad from 5).
  • Row byte-size must be 32-byte aligned → ``COL_PAD = 608`` (from 588).
  • Right buffer ≤ 65 536 B → ``N_CHUNK = 32`` for output-N chunking.
  • Large-K matmul (K = 608) causes ``tmov acc→acc`` codegen failure →
    each N-chunk of the patch matmul gets its own ``pl.at`` block.
  • Padding rows [CLIP_TOKENS..T) must be zeroed after LayerNorm (bias would
    activate them and corrupt attention scores).

Usage::

    python clip_encoder.py -p a2a3 -d 0
"""

import argparse

import pypto.language as pl
import torch
import torch.nn.functional as F

from config import (
    CLIP_DIM, CLIP_HEADS, CLIP_LAYERS, CLIP_PATCH, CLIP_IMG, CLIP_TOKENS,
)


# ══════════════════════════════════════════════════════════════════════════════
# Derived constants — hardware-aligned padding / tiling.
# ══════════════════════════════════════════════════════════════════════════════

CLIP_HEAD_DIM = CLIP_DIM // CLIP_HEADS          # 32
FFN_DIM = CLIP_DIM * 4                          # 512
EPS = 1e-5
ATTN_SCALE = CLIP_HEAD_DIM ** -0.5              # 1/√32

T = ((CLIP_TOKENS + 15) // 16) * 16             # CLIP_TOKENS padded to innerRows=16
NUM_PATCHES = (CLIP_IMG // CLIP_PATCH) ** 2
PATCH_H = CLIP_IMG // CLIP_PATCH
PATCH_W = CLIP_IMG // CLIP_PATCH
COL_SIZE = 3 * CLIP_PATCH * CLIP_PATCH          # 588  (im2col column width)
COL_PAD = 608                                   # 32-byte aligned, avoids tmov bug
IMG_FLAT_SIZE = 3 * CLIP_IMG * CLIP_IMG         # 2352
N_CHUNK = 32                                    # output-N tile (Right buffer safe)

# ── QuickGELU constant (matches golden reference: x * sigmoid(1.702 * x)) ──
QUICK_GELU_SCALE = 1.702

# Geometry assertions — keep at the bottom of the derived-constants block.
assert CLIP_DIM % CLIP_HEADS == 0, "CLIP_HEADS must divide CLIP_DIM"
assert CLIP_HEAD_DIM * CLIP_HEADS == CLIP_DIM
assert T % 16 == 0, "T must be a multiple of innerRows=16"
assert T >= CLIP_TOKENS, "T must accommodate all real tokens"
assert CLIP_TOKENS == NUM_PATCHES + 1, "expected CLS token + NUM_PATCHES patch tokens"
assert COL_PAD >= COL_SIZE and COL_PAD % 4 == 0, "COL_PAD must be ≥ COL_SIZE and 32-byte aligned"
assert FFN_DIM == CLIP_DIM * 4
assert IMG_FLAT_SIZE == 3 * CLIP_IMG * CLIP_IMG
assert CLIP_DIM % N_CHUNK == 0, "N_CHUNK must divide CLIP_DIM"
assert FFN_DIM % N_CHUNK == 0, "N_CHUNK must divide FFN_DIM"
assert (3 * CLIP_DIM) % N_CHUNK == 0, "N_CHUNK must divide 3*CLIP_DIM (QKV width)"


# ══════════════════════════════════════════════════════════════════════════════
# Reusable inline kernels.
# ══════════════════════════════════════════════════════════════════════════════

@pl.jit.inline
def _layernorm(
    x: pl.Tensor[[T, CLIP_DIM], pl.FP32],
    w: pl.Tensor[[1, CLIP_DIM], pl.FP32],
    b: pl.Tensor[[1, CLIP_DIM], pl.FP32],
    y: pl.Tensor[[T, CLIP_DIM], pl.FP32],
):
    """LayerNorm with padding-row masking.

    Computes ``(x - mean) / std * w + b`` over all T rows, then zeroes rows
    ``[CLIP_TOKENS, T)`` so that padding rows stay exactly 0 (the bias would
    otherwise activate them and corrupt downstream attention).
    """
    scaled   = pl.mul(x, 1.0 / CLIP_DIM)
    mean     = pl.row_sum(scaled)
    centred  = pl.row_expand_sub(x, mean)
    sq       = pl.mul(centred, centred)
    var      = pl.row_sum(sq)
    var      = pl.mul(var, 1.0 / CLIP_DIM)
    var_eps  = pl.add(var, EPS)
    var_eps  = pl.reshape(var_eps, [1, T])
    std      = pl.sqrt(var_eps)
    std      = pl.reshape(std, [T, 1])
    normed   = pl.row_expand_div(centred, std)

    scaled_n = pl.col_expand_mul(normed, w)
    # A2/A3 does not support pl.full(dtype=pl.FP32).  Work around by zeroing
    # a same-shape tile then adding 1.0 — produces an all-ones FP32 tile.
    ones     = pl.sub(x, x)
    ones     = pl.add(ones, 1.0)
    biased   = pl.add(scaled_n, pl.col_expand_mul(ones, b))

    for r in pl.range(CLIP_TOKENS, T):
        zero_row = pl.mul(pl.slice(biased, [1, CLIP_DIM], [r, 0]), 0.0)
        biased   = pl.assemble(biased, zero_row, [r, 0])

    y = pl.assemble(y, biased, [0, 0])
    return y


# ══════════════════════════════════════════════════════════════════════════════
# Main kernel — full CLIP encoder forward pass.
# ══════════════════════════════════════════════════════════════════════════════

@pl.jit
def clip_encoder(
    # ── Inputs ──
    img_flat:     pl.Tensor[[1, IMG_FLAT_SIZE], pl.FP32],
    # ── Patch embedding ──
    patch_conv_w: pl.Tensor[[CLIP_DIM, COL_PAD], pl.BF16],
    cls_token:    pl.Tensor[[1, CLIP_DIM], pl.BF16],
    pos_embed:    pl.Tensor[[T, CLIP_DIM], pl.BF16],
    # ── Pre-LayerNorm ──
    pre_ln_w:     pl.Tensor[[1, CLIP_DIM], pl.FP32],
    pre_ln_b:     pl.Tensor[[1, CLIP_DIM], pl.FP32],
    # ── Stacked transformer weights ──
    norm1_w:      pl.Tensor[[CLIP_LAYERS, CLIP_DIM], pl.FP32],
    norm1_b:      pl.Tensor[[CLIP_LAYERS, CLIP_DIM], pl.FP32],
    qkv_w:        pl.Tensor[[CLIP_LAYERS * 3 * CLIP_DIM, CLIP_DIM], pl.BF16],
    proj_w:       pl.Tensor[[CLIP_DIM, CLIP_LAYERS * CLIP_DIM], pl.BF16],
    norm2_w:      pl.Tensor[[CLIP_LAYERS, CLIP_DIM], pl.FP32],
    norm2_b:      pl.Tensor[[CLIP_LAYERS, CLIP_DIM], pl.FP32],
    fc1_w:        pl.Tensor[[CLIP_DIM, CLIP_LAYERS * FFN_DIM], pl.BF16],
    fc2_w:        pl.Tensor[[CLIP_LAYERS * FFN_DIM, CLIP_DIM], pl.BF16],
    # ── Output ──
    out:          pl.Out[pl.Tensor[[CLIP_TOKENS, CLIP_DIM], pl.BF16]],
):

    # ── Scope 1: Patch embedding — im2col (kernel-side, scalar access). ──
    # Follows the same pl.read + pl.tensor.write pattern as common/conv2d.py.
    col_buf = pl.create_tensor([T, COL_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="im2col"):
        for ph in pl.range(PATCH_H):
            for pw in pl.range(PATCH_W):
                patch_idx = ph * PATCH_W + pw
                col_offset = 0
                for c in pl.range(3):
                    for kh in pl.range(CLIP_PATCH):
                        for kw in pl.range(CLIP_PATCH):
                            src_h = ph * CLIP_PATCH + kh
                            src_w = pw * CLIP_PATCH + kw
                            src_col = c * CLIP_IMG * CLIP_IMG + src_h * CLIP_IMG + src_w
                            val = pl.read(img_flat, [0, src_col])
                            pl.tensor.write(col_buf, [patch_idx, col_offset], val)
                            col_offset = col_offset + 1

    # ── Scope 2: Patch matmul — N-chunked (each chunk in its own pl.at). ──
    # Separate pl.at blocks avoid tmov acc→acc codegen failure on K=608.
    patch_emb_gm = pl.create_tensor([T, CLIP_DIM], dtype=pl.BF16)
    for nc in pl.unroll(CLIP_DIM // N_CHUNK):
        n0 = nc * N_CHUNK
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="patch_mm"):
            col_tile = pl.slice(col_buf, [T, COL_PAD], [0, 0])
            col_bf   = pl.cast(col_tile, target_type=pl.BF16)
            pw_chunk = pl.slice(patch_conv_w, [N_CHUNK, COL_PAD], [n0, 0])
            patch_out = pl.matmul(col_bf, pw_chunk, b_trans=True, out_dtype=pl.FP32)
            patch_emb_gm = pl.assemble(
                patch_emb_gm, pl.cast(patch_out, target_type=pl.BF16), [0, n0],
            )

    # ── Scope 3: Prepend CLS token + add positional embedding. ──
    x_pos_gm = pl.create_tensor([T, CLIP_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cls_pos"):
        x_pos_gm = pl.assemble(x_pos_gm, pl.slice(cls_token, [1, CLIP_DIM], [0, 0]), [0, 0])
        for i in pl.range(NUM_PATCHES):
            x_pos_gm = pl.assemble(
                x_pos_gm, pl.slice(patch_emb_gm, [1, CLIP_DIM], [i, 0]), [i + 1, 0],
            )

    x_ln_gm = pl.create_tensor([T, CLIP_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="add_pos"):
        x_tile = pl.slice(x_pos_gm, [T, CLIP_DIM], [0, 0])
        p_tile = pl.slice(pos_embed,  [T, CLIP_DIM], [0, 0])
        x_ln_gm = pl.assemble(
            x_ln_gm,
            pl.add(pl.cast(x_tile, target_type=pl.FP32),
                   pl.cast(p_tile, target_type=pl.FP32)),
            [0, 0],
        )

    # ── Scope 4: Pre-LayerNorm. ──
    layer_x_gm = pl.create_tensor([T, CLIP_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="pre_ln"):
        x_tile = pl.slice(x_ln_gm, [T, CLIP_DIM], [0, 0])
        layer_x_gm = _layernorm(x_tile, pre_ln_w, pre_ln_b, layer_x_gm)

    # ══════════════════════════════════════════════════════════════════════
    # Transformer layers — parameterised loop over CLIP_LAYERS.
    #
    # GM buffers are pre-allocated once and reused each iteration
    # (each scope overwrites before the next scope reads).
    # ══════════════════════════════════════════════════════════════════════
    qkv_gm     = pl.create_tensor([T, 3 * CLIP_DIM], dtype=pl.BF16)
    attn_gm    = pl.create_tensor([T, CLIP_DIM],     dtype=pl.FP32)
    attn_res_gm = pl.create_tensor([T, CLIP_DIM],    dtype=pl.FP32)
    ln1_gm     = pl.create_tensor([T, CLIP_DIM],     dtype=pl.FP32)
    ln2_gm     = pl.create_tensor([T, CLIP_DIM],     dtype=pl.FP32)
    gelu_gm    = pl.create_tensor([T, FFN_DIM],      dtype=pl.FP32)

    for layer_idx in pl.unroll(CLIP_LAYERS):

        # ── Scope 5: LN1 + QKV projection. ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ln_qkv"):
            ln1_w = pl.slice(norm1_w, [1, CLIP_DIM], [layer_idx, 0])
            ln1_b = pl.slice(norm1_b, [1, CLIP_DIM], [layer_idx, 0])
            x_tile = pl.slice(layer_x_gm, [T, CLIP_DIM], [0, 0])
            ln1_gm = _layernorm(x_tile, ln1_w, ln1_b, ln1_gm)
            normed_tile = pl.slice(ln1_gm, [T, CLIP_DIM], [0, 0])
            normed_bf = pl.cast(normed_tile, target_type=pl.BF16)
            qkv_offset = layer_idx * (3 * CLIP_DIM)
            for n0 in pl.range(0, 3 * CLIP_DIM, N_CHUNK):
                qkv_chunk = pl.matmul(
                    normed_bf,
                    pl.slice(qkv_w, [N_CHUNK, CLIP_DIM], [qkv_offset + n0, 0]),
                    b_trans=True, out_dtype=pl.FP32,
                )
                qkv_gm = pl.assemble(
                    qkv_gm, pl.cast(qkv_chunk, target_type=pl.BF16), [0, n0],
                )
            for r in pl.range(CLIP_TOKENS, T):
                zero_row = pl.full([1, 3 * CLIP_DIM], dtype=pl.BF16, value=0.0)
                qkv_gm = pl.assemble(qkv_gm, zero_row, [r, 0])

        # ── Scope 6: Multi-head self-attention (per-head, with scale). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mha"):
            for h in pl.range(CLIP_HEADS):
                h_col  = h * CLIP_HEAD_DIM
                q_h    = pl.slice(qkv_gm, [T, CLIP_HEAD_DIM], [0, h_col])
                k_h    = pl.slice(qkv_gm, [T, CLIP_HEAD_DIM], [0, CLIP_DIM + h_col])
                v_h    = pl.slice(qkv_gm, [T, CLIP_HEAD_DIM], [0, 2 * CLIP_DIM + h_col])

                scores = pl.mul(
                    pl.matmul(q_h, k_h, b_trans=True, out_dtype=pl.FP32), ATTN_SCALE,
                )
                shifted = pl.row_expand_sub(scores, pl.row_max(scores))
                exp_s   = pl.exp(shifted)
                sm      = pl.row_expand_div(exp_s, pl.row_sum(exp_s))
                ctx     = pl.matmul(pl.cast(sm, target_type=pl.BF16), v_h, out_dtype=pl.FP32)
                attn_gm = pl.assemble(attn_gm, ctx, [0, h_col])
            for r in pl.range(CLIP_TOKENS, T):
                zero_attn = pl.full([1, CLIP_DIM], dtype=pl.FP32, value=0.0)
                attn_gm = pl.assemble(attn_gm, zero_attn, [r, 0])

        # ── Scope 7: Output projection + residual. ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="proj_res"):
            attn_tile = pl.cast(pl.slice(attn_gm, [T, CLIP_DIM], [0, 0]), target_type=pl.BF16)
            proj_offset = layer_idx * CLIP_DIM
            proj_w_l  = pl.slice(proj_w, [CLIP_DIM, CLIP_DIM], [0, proj_offset])
            proj      = pl.matmul(attn_tile, proj_w_l, out_dtype=pl.FP32)
            residual  = pl.slice(layer_x_gm, [T, CLIP_DIM], [0, 0])
            attn_res_gm = pl.assemble(attn_res_gm, pl.add(residual, proj), [0, 0])
            for r in pl.range(CLIP_TOKENS, T):
                zero_res = pl.full([1, CLIP_DIM], dtype=pl.FP32, value=0.0)
                attn_res_gm = pl.assemble(attn_res_gm, zero_res, [r, 0])

        # ── Scope 8: LN2 + FC1 + QuickGELU (x * sigmoid(1.702 * x)). ──
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="ln_fc1"):
            ln2_w_l = pl.slice(norm2_w, [1, CLIP_DIM], [layer_idx, 0])
            ln2_b_l = pl.slice(norm2_b, [1, CLIP_DIM], [layer_idx, 0])
            ar_tile = pl.slice(attn_res_gm, [T, CLIP_DIM], [0, 0])
            ln2_gm  = _layernorm(ar_tile, ln2_w_l, ln2_b_l, ln2_gm)
            ln2_tile = pl.slice(ln2_gm, [T, CLIP_DIM], [0, 0])
            normed_bf = pl.cast(ln2_tile, target_type=pl.BF16)
            fc1_offset = layer_idx * FFN_DIM
            for n0 in pl.range(0, FFN_DIM, N_CHUNK):
                fc1 = pl.matmul(
                    normed_bf,
                    pl.slice(fc1_w, [CLIP_DIM, N_CHUNK], [0, fc1_offset + n0]),
                    out_dtype=pl.FP32,
                )
                # QuickGELU: x * sigmoid(1.702 * x)
                scaled   = pl.mul(fc1, QUICK_GELU_SCALE)
                neg_sc   = pl.neg(scaled)
                exp_neg  = pl.exp(neg_sc)
                denom    = pl.add(exp_neg, 1.0)
                sigmoid  = pl.recip(denom)
                qgelu    = pl.mul(fc1, sigmoid)
                gelu_gm = pl.assemble(gelu_gm, qgelu, [0, n0])

        # ── Scope 9: FC2 (K-chunked accumulation) + residual → layer_x_gm. ──
        # qgelu [T, FFN_DIM] @ fc2_w [FFN_DIM, CLIP_DIM] → [T, CLIP_DIM]
        # K-chunk: [T, N_CHUNK] @ [N_CHUNK, CLIP_DIM] (no b_trans needed)
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="fc2_res"):
            fc2_offset = layer_idx * FFN_DIM
            g_chunk0 = pl.cast(pl.slice(gelu_gm, [T, N_CHUNK], [0, 0]), target_type=pl.BF16)
            w_chunk0 = pl.slice(fc2_w, [N_CHUNK, CLIP_DIM], [fc2_offset, 0])
            acc = pl.matmul(g_chunk0, w_chunk0, out_dtype=pl.FP32)
            for k0 in pl.range(N_CHUNK, FFN_DIM, N_CHUNK):
                g_chunk = pl.cast(pl.slice(gelu_gm, [T, N_CHUNK], [0, k0]), target_type=pl.BF16)
                w_chunk = pl.slice(fc2_w, [N_CHUNK, CLIP_DIM], [fc2_offset + k0, 0])
                acc = pl.matmul_acc(acc, g_chunk, w_chunk)
            residual = pl.slice(attn_res_gm, [T, CLIP_DIM], [0, 0])
            layer_x_gm = pl.assemble(layer_x_gm, pl.add(residual, acc), [0, 0])

    # ── Scope 10: Final output — only valid rows (no padding). ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="final"):
        final_tile = pl.slice(layer_x_gm, [CLIP_TOKENS, CLIP_DIM], [0, 0])
        out = pl.assemble(out, pl.cast(final_tile, target_type=pl.BF16), [0, 0])

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Golden reference — mirrors test_golden_fun_control_full.py::clip_encode
# with QuickGELU: x * sigmoid(1.702 * x) (matching golden reference line 742).
# ══════════════════════════════════════════════════════════════════════════════

def _bf16(t):
    """Round through BF16 then back to FP32 — emulates kernel pl.cast(BF16)."""
    if isinstance(t, torch.Tensor):
        return t.to(torch.bfloat16).to(torch.float32)
    return t


def _im2col_clip(img_4d, patch_h, patch_w, clip_patch):
    """im2col for CLIP patch conv: matches kernel's scalar read/write order.
    img_4d: [B, 3, H, W].  Returns [B, num_patches, col_size].
    """
    B, C, H, W = img_4d.shape
    num_patches = patch_h * patch_w
    col_size = C * clip_patch * clip_patch
    cols = torch.zeros(B, num_patches, col_size)
    for b in range(B):
        for ph in range(patch_h):
            for pw in range(patch_w):
                patch_idx = ph * patch_w + pw
                col_offset = 0
                for c in range(C):
                    for kh in range(clip_patch):
                        for kw in range(clip_patch):
                            src_h = ph * clip_patch + kh
                            src_w = pw * clip_patch + kw
                            cols[b, patch_idx, col_offset] = img_4d[b, c, src_h, src_w]
                            col_offset += 1
    return cols


def golden_clip_encode(img_tensor, w):
    """BF16-aware golden matching kernel's exact computation path.

    Implements custom LayerNorm and N-chunked matmul to match kernel exactly.
    Output: [CLIP_TOKENS, CLIP_DIM] — only valid rows, no padding.
    """
    B = img_tensor.shape[0]

    def custom_layernorm(x, w, b):
        """Match kernel's _layernorm exactly."""
        scaled = x * (1.0 / CLIP_DIM)
        mean = scaled.sum(dim=-1, keepdim=True)
        centred = x - mean
        sq = centred * centred
        var = sq.sum(dim=-1, keepdim=True) * (1.0 / CLIP_DIM)
        var_eps = var + EPS
        std = var_eps.sqrt()
        normed = centred / std
        scaled_n = normed * w
        biased = scaled_n + b
        # Zero padding rows
        if biased.shape[1] > CLIP_TOKENS:
            biased[:, CLIP_TOKENS:, :] = 0.0
        return biased

    def n_chunked_matmul(x, w, n_chunk=N_CHUNK):
        """Match kernel's N-chunked matmul accumulation order."""
        result = torch.zeros(x.shape[0], w.shape[0], dtype=x.dtype)
        for n0 in range(0, w.shape[0], n_chunk):
            w_chunk = w[n0:n0+n_chunk]
            result[:, n0:n0+n_chunk] = _bf16(x) @ _bf16(w_chunk).T
        return result

    col = _im2col_clip(img_tensor, PATCH_H, PATCH_W, CLIP_PATCH)
    col_padded = torch.zeros(T, COL_PAD)
    col_padded[:NUM_PATCHES, :COL_SIZE] = col.squeeze(0)
    w_raw = w["clip_patch_conv"].reshape(CLIP_DIM, -1)
    w_padded = torch.zeros(CLIP_DIM, COL_PAD, dtype=w_raw.dtype)
    w_padded[:, :w_raw.shape[1]] = w_raw
    patch_emb = _bf16(n_chunked_matmul(_bf16(col_padded), _bf16(w_padded)))
    x = patch_emb[:NUM_PATCHES, :].unsqueeze(0)

    cls = w["clip_cls"].expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)
    x = _bf16(x + w["clip_pos"])
    x = custom_layernorm(x, w["clip_pre_norm_w"], w["clip_pre_norm_b"])
    for i in range(CLIP_LAYERS):
        p = f"clip_l{i}"
        h = custom_layernorm(x, w[f"{p}_norm1_w"], w[f"{p}_norm1_b"])
        qkv_w = w[f"{p}_qkv"]
        qkv = _bf16(n_chunked_matmul(_bf16(h.squeeze(0)), _bf16(qkv_w), n_chunk=3*CLIP_DIM)).unsqueeze(0)
        qkv = qkv.view(B, -1, 3, CLIP_HEADS, CLIP_HEAD_DIM)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = _bf16(q) @ _bf16(k).transpose(-2, -1) * ATTN_SCALE
        attn = F.softmax(scores, dim=-1)
        attn_out = _bf16(_bf16(attn) @ _bf16(v)).transpose(1, 2).contiguous().view(B, -1, CLIP_DIM)
        proj_w = w[f"{p}_proj"]
        proj = _bf16(n_chunked_matmul(_bf16(attn_out.squeeze(0)), _bf16(proj_w))).unsqueeze(0)
        x = _bf16(x + proj)
        # Zero padding after residual
        if x.shape[1] > CLIP_TOKENS:
            x[:, CLIP_TOKENS:, :] = 0.0
        h = custom_layernorm(x, w[f"{p}_norm2_w"], w[f"{p}_norm2_b"])
        fc1_w = w[f"{p}_fc1"]
        ff = _bf16(n_chunked_matmul(_bf16(h.squeeze(0)), _bf16(fc1_w), n_chunk=FFN_DIM)).unsqueeze(0)
        ff = ff * torch.sigmoid(1.702 * ff)
        fc2_w = w[f"{p}_fc2"]
        fc2 = _bf16(n_chunked_matmul(_bf16(ff.squeeze(0)), _bf16(fc2_w))).unsqueeze(0)
        x = _bf16(x + fc2)
        # Zero padding after residual
        if x.shape[1] > CLIP_TOKENS:
            x[:, CLIP_TOKENS:, :] = 0.0
    return x.squeeze(0)[:CLIP_TOKENS, :]


# ══════════════════════════════════════════════════════════════════════════════
# Test harness — build_tensor_specs / golden_fn / __main__.
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor_specs():
    from golden import TensorSpec

    g = torch.Generator().manual_seed(42)

    patch_conv_w_4d = torch.randn(CLIP_DIM, 3, CLIP_PATCH, CLIP_PATCH, generator=g) * 0.02
    patch_conv_w_padded = torch.zeros(CLIP_DIM, COL_PAD, dtype=torch.bfloat16)
    patch_conv_w_padded[:, :COL_SIZE] = patch_conv_w_4d.reshape(CLIP_DIM, COL_SIZE).bfloat16()

    cls_3d = torch.randn(1, 1, CLIP_DIM, generator=g) * 0.02
    pos_3d = torch.randn(1, CLIP_TOKENS, CLIP_DIM, generator=g) * 0.02
    pos_padded = torch.zeros(T, CLIP_DIM, dtype=torch.bfloat16)
    pos_padded[:CLIP_TOKENS, :] = pos_3d.squeeze(0).bfloat16()

    specs = [
        TensorSpec("img_flat",     [1, IMG_FLAT_SIZE],     torch.float32, init_value=torch.randn),
        TensorSpec("patch_conv_w", [CLIP_DIM, COL_PAD],    torch.bfloat16, init_value=patch_conv_w_padded),
        TensorSpec("cls_token",    [1, CLIP_DIM],          torch.bfloat16, init_value=cls_3d.squeeze(0).bfloat16()),
        TensorSpec("pos_embed",    [T, CLIP_DIM],          torch.bfloat16, init_value=pos_padded),
        TensorSpec("pre_ln_w",     [1, CLIP_DIM],          torch.float32, init_value=torch.ones),
        TensorSpec("pre_ln_b",     [1, CLIP_DIM],          torch.float32, init_value=torch.zeros),
    ]

    stacked_norm1_w = torch.ones(CLIP_LAYERS, CLIP_DIM, dtype=torch.float32)
    stacked_norm1_b = torch.zeros(CLIP_LAYERS, CLIP_DIM, dtype=torch.float32)
    stacked_qkv_w = torch.cat([
        (torch.randn(3 * CLIP_DIM, CLIP_DIM, generator=g) * 0.02).bfloat16()
        for _ in range(CLIP_LAYERS)
    ], dim=0)
    stacked_proj_w = torch.cat([
        (torch.randn(CLIP_DIM, CLIP_DIM, generator=g) * 0.02).bfloat16().T
        for _ in range(CLIP_LAYERS)
    ], dim=1)
    stacked_norm2_w = torch.ones(CLIP_LAYERS, CLIP_DIM, dtype=torch.float32)
    stacked_norm2_b = torch.zeros(CLIP_LAYERS, CLIP_DIM, dtype=torch.float32)
    stacked_fc1_w = torch.cat([
        (torch.randn(FFN_DIM, CLIP_DIM, generator=g) * 0.02).bfloat16().T
        for _ in range(CLIP_LAYERS)
    ], dim=1)
    stacked_fc2_w = torch.cat([
        (torch.randn(CLIP_DIM, FFN_DIM, generator=g) * 0.02).bfloat16().T
        for _ in range(CLIP_LAYERS)
    ], dim=0)

    specs.extend([
        TensorSpec("norm1_w", [CLIP_LAYERS, CLIP_DIM],              torch.float32,  init_value=stacked_norm1_w),
        TensorSpec("norm1_b", [CLIP_LAYERS, CLIP_DIM],              torch.float32,  init_value=stacked_norm1_b),
        TensorSpec("qkv_w",   [CLIP_LAYERS * 3 * CLIP_DIM, CLIP_DIM], torch.bfloat16, init_value=stacked_qkv_w),
        TensorSpec("proj_w",  [CLIP_DIM, CLIP_LAYERS * CLIP_DIM],   torch.bfloat16, init_value=stacked_proj_w),
        TensorSpec("norm2_w", [CLIP_LAYERS, CLIP_DIM],              torch.float32,  init_value=stacked_norm2_w),
        TensorSpec("norm2_b", [CLIP_LAYERS, CLIP_DIM],              torch.float32,  init_value=stacked_norm2_b),
        TensorSpec("fc1_w",   [CLIP_DIM, CLIP_LAYERS * FFN_DIM],    torch.bfloat16, init_value=stacked_fc1_w),
        TensorSpec("fc2_w",   [CLIP_LAYERS * FFN_DIM, CLIP_DIM],    torch.bfloat16, init_value=stacked_fc2_w),
    ])

    specs.append(TensorSpec("out", [CLIP_TOKENS, CLIP_DIM], torch.bfloat16, is_output=True))
    return specs


def golden_clip_encoder(tensors):
    """Run golden_clip_encode with QuickGELU, output only valid rows."""
    img_flat = tensors["img_flat"]
    img_4d = img_flat.reshape(1, 3, CLIP_IMG, CLIP_IMG).float()

    w = {
        "clip_patch_conv": tensors["patch_conv_w"][:, :COL_SIZE].float()
                           .reshape(CLIP_DIM, 3, CLIP_PATCH, CLIP_PATCH),
        "clip_cls":        tensors["cls_token"].float().unsqueeze(0),
        "clip_pos":        tensors["pos_embed"][:CLIP_TOKENS, :].float().unsqueeze(0),
        "clip_pre_norm_w": tensors["pre_ln_w"].squeeze(0).float(),
        "clip_pre_norm_b": tensors["pre_ln_b"].squeeze(0).float(),
    }
    for i in range(CLIP_LAYERS):
        p = f"clip_l{i}"
        w[f"{p}_norm1_w"] = tensors["norm1_w"][i].float()
        w[f"{p}_norm1_b"] = tensors["norm1_b"][i].float()
        qkv_start = i * 3 * CLIP_DIM
        w[f"{p}_qkv"]     = tensors["qkv_w"][qkv_start:qkv_start + 3 * CLIP_DIM].float()
        proj_start = i * CLIP_DIM
        w[f"{p}_proj"]    = tensors["proj_w"][:, proj_start:proj_start + CLIP_DIM].float().T
        w[f"{p}_norm2_w"] = tensors["norm2_w"][i].float()
        w[f"{p}_norm2_b"] = tensors["norm2_b"][i].float()
        fc1_start = i * FFN_DIM
        w[f"{p}_fc1"]     = tensors["fc1_w"][:, fc1_start:fc1_start + FFN_DIM].float().T
        fc2_start = i * FFN_DIM
        w[f"{p}_fc2"]     = tensors["fc2_w"][fc2_start:fc2_start + FFN_DIM].float().T

    result = golden_clip_encode(img_4d, w)
    tensors["out"][:] = result.bfloat16()


if __name__ == "__main__":
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=clip_encoder,
        specs=build_tensor_specs(),
        golden_fn=golden_clip_encoder,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=3e-3,
        atol=3e-3,
        compare_fn={"out": ratio_allclose(atol=0.1, rtol=0.1, max_error_ratio=0.02)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
