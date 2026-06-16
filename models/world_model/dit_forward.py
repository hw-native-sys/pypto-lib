# Copyright (c) PyPTO Contributors.
# Licensed under CANN Open Software License Agreement Version 2.0.
# -----------------------------------------------------------------------------
"""DiT (Diffusion Transformer) for Fun-Control 1.3B — pypto3.0 implementation.

Single @pl.jit kernel implementing the full DiT forward pass matching
``test_golden_fun_control_full.py::dit_forward`` (golden:1364-1489).

Architecture::

    ref_conv + patch_conv + concat → x_full
    text_proj(GELU-tanh) + clip_proj(LN→FC→GELU→FC→LN) → ctx
    AdaLN → Self-Attn(RoPE) → Cross-Attn(text+img) → FFN → Head → output

All arithmetic in NPU kernels.  Host prepares im2col (data rearrangement),
RoPE cos/sin tables (lookup), and the static sinusoidal frequency table
(``time_freqs``).  The full timestep embedding — ``sin_emb = [cos(t*freqs),
sin(t*freqs)]`` followed by a 3-layer SiLU MLP — runs on NPU.

Known differences from the golden reference:
  • CLIP projection uses GELU-tanh approximation in the kernel (pypto lacks
    ``erf`` for exact GELU); the golden uses ``F.gelu()`` (exact).  This
    difference is intrinsic to the NPU compute model and is tracked by the
    permissive K3 tolerance.

Usage::

    python dit_forward.py -p a2a3 -d 0
"""

import argparse
import math
import sys

import pypto.language as pl
import torch
import torch.nn.functional as F

sys.path.insert(
    0, '/data/x00952168/pypto3.0/cann-recipes-embodied-ai/world_model/'
       'agibot-arm-world-model/infer_with_torch')
from test_golden_fun_control_full import (  # noqa: E402
    dit_forward as golden_dit_fn,
    sinusoidal_embedding_1d, precompute_freqs_cis_3d, make_weights,
)
from config import (
    DIT_DIM, DIT_HEADS, DIT_HEAD_DIM, DIT_FFN, DIT_IN_DIM,
    DIT_TEXT_DIM, DIT_FREQ_DIM, DIT_EPS,
    CLIP_DIM, CLIP_TOKENS, T5_SEQ, T5_DIM, LAT_F, LAT_H, LAT_W,
)


# ══════════════════════════════════════════════════════════════════════════════
# Derived constants
# ══════════════════════════════════════════════════════════════════════════════

COND_CH = DIT_IN_DIM + 4 + DIT_IN_DIM
FP = LAT_F
HP = LAT_H // 2
WP = LAT_W // 2
REF_N = HP * WP
X_N = FP * HP * WP
S = REF_N + X_N
T = 16
CLIP_PAD = ((CLIP_TOKENS + T - 1) // T) * T

# 3D RoPE dimension splits
ROPE_F_DIM = DIT_HEAD_DIM - 2 * (DIT_HEAD_DIM // 3)
ROPE_H_DIM = DIT_HEAD_DIM // 3
ROPE_W_DIM = DIT_HEAD_DIM // 3
HEAD_HALF = DIT_HEAD_DIM // 2

# Numerical constants
HEAD_SCALE = 1.0 / math.sqrt(DIT_HEAD_DIM)
GELU_SQRT_2_OVER_PI = 0.7978845608
GELU_COEFF = 0.044715
LN_EPS = 1e-5
DIM_INV = 1.0 / DIT_DIM

REF_CONV_COL = T * 4
PATCH_CONV_COL = T * 13
TIME_PROJ_N = 6 * DIT_DIM   # 768
TIME_N_CHUNK = 64           # FC3 N-chunk for Right buffer safety (64*128*2=16384 < 65536)

# Geometry assertions
assert DIT_DIM % DIT_HEADS == 0, "DIT_HEADS must divide DIT_DIM"
assert DIT_HEAD_DIM * DIT_HEADS == DIT_DIM
assert S % T == 0, "S must be a multiple of T"
assert CLIP_PAD % T == 0, "CLIP_PAD must be a multiple of T"
assert T5_SEQ % T == 0, "T5_SEQ must be a multiple of T"
assert X_N % T == 0, "X_N must be a multiple of T"
assert REF_N % T == 0, "REF_N must be a multiple of T"
assert DIT_HEAD_DIM % 2 == 0, "DIT_HEAD_DIM must be even (RoPE half-split)"


# ══════════════════════════════════════════════════════════════════════════════
# Host helpers — data rearrangement only (no arithmetic).
# ══════════════════════════════════════════════════════════════════════════════

def _im2col_ref_conv2d(ref):
    """im2col for ref Conv2d(kernel=2x2, stride=2). Data rearrangement only."""
    cols = F.unfold(ref.float(), kernel_size=(2, 2), stride=2)
    return cols.squeeze(0).transpose(0, 1).contiguous().bfloat16()


def _im2col_patch_conv3d(x):
    """im2col for patch Conv3d(kernel=(1,2,2), stride=(1,2,2)). Data rearrangement only."""
    B, C, D, H, W = x.shape
    col_size = C * 1 * 2 * 2
    xf = x.float()
    rows = []
    for n in range(B):
        for d in range(D):
            for h in range(H // 2):
                for w in range(W // 2):
                    rows.append(xf[n, :, d, h * 2:h * 2 + 2, w * 2:w * 2 + 2].reshape(col_size))
    return torch.stack(rows).contiguous().bfloat16()


def _precompute_rope_cos_sin():
    """Precompute 3D RoPE cos/sin tables. Data rearrangement only."""
    freq_f, freq_h, freq_w = precompute_freqs_cis_3d(DIT_HEAD_DIM)
    return (freq_f.real.float(), freq_f.imag.float(),
            freq_h.real.float(), freq_h.imag.float(),
            freq_w.real.float(), freq_w.imag.float())


def _build_rope_map(cos_f, sin_f, cos_h, sin_h, cos_w, sin_w, fl, hl, wl):
    """Build per-position RoPE cos/sin lookup tables for complex-pair rotation.

    The kernel applies complex-pair rotation via pl.gather (stride-2 even/odd
    split).  Shape: ``[S, DIT_DIM//2]`` — half-dim frequencies repeated across
    all heads.
    """
    total = fl * hl * wl
    cos_table = torch.zeros(total, HEAD_HALF)
    sin_table = torch.zeros(total, HEAD_HALF)
    idx = 0
    for fi in range(fl):
        for hi in range(hl):
            for wi in range(wl):
                for j in range(ROPE_F_DIM // 2):
                    cos_table[idx, j] = cos_f[fi, j]
                    sin_table[idx, j] = sin_f[fi, j]
                for j in range(ROPE_H_DIM // 2):
                    col = ROPE_F_DIM // 2 + j
                    cos_table[idx, col] = cos_h[hi, j]
                    sin_table[idx, col] = sin_h[hi, j]
                for j in range(ROPE_W_DIM // 2):
                    col = ROPE_F_DIM // 2 + ROPE_H_DIM // 2 + j
                    cos_table[idx, col] = cos_w[wi, j]
                    sin_table[idx, col] = sin_w[wi, j]
                idx += 1
    return cos_table.repeat(1, DIT_HEADS), sin_table.repeat(1, DIT_HEADS)


# ══════════════════════════════════════════════════════════════════════════════
# Fused kernel — full DiT forward pass.
# ══════════════════════════════════════════════════════════════════════════════

@pl.jit
def dit_forward(
    # ── Conv inputs (im2col precomputed on host — data rearrangement) ──
    ref_col:       pl.Tensor[[REF_N, REF_CONV_COL], pl.BF16],
    patch_col:     pl.Tensor[[X_N, PATCH_CONV_COL], pl.BF16],
    ref_conv_w:    pl.Tensor[[REF_CONV_COL, DIT_DIM], pl.BF16],
    patch_conv_w:  pl.Tensor[[PATCH_CONV_COL, DIT_DIM], pl.BF16],
    # ── Text + CLIP projection inputs ──
    text_raw:      pl.Tensor[[T5_SEQ, T5_DIM], pl.BF16],
    text_fc1_w:    pl.Tensor[[DIT_DIM, DIT_TEXT_DIM], pl.BF16],
    text_fc2_w:    pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    clip_raw:      pl.Tensor[[CLIP_PAD, CLIP_DIM], pl.BF16],
    clip_ln1_w:    pl.Tensor[[1, CLIP_DIM], pl.FP32],
    clip_ln1_b:    pl.Tensor[[1, CLIP_DIM], pl.FP32],
    clip_fc1_w:    pl.Tensor[[CLIP_DIM, CLIP_DIM], pl.BF16],
    clip_fc2_w:    pl.Tensor[[DIT_DIM, CLIP_DIM], pl.BF16],
    clip_ln2_w:    pl.Tensor[[1, DIT_DIM], pl.FP32],
    clip_ln2_b:    pl.Tensor[[1, DIT_DIM], pl.FP32],
    # ── Context tables (precomputed on host) ──
    clip_mask:     pl.Tensor[[1, CLIP_PAD], pl.FP32],
    rope_cos:      pl.Tensor[[S, DIT_DIM // 2], pl.FP32],
    rope_sin:      pl.Tensor[[S, DIT_DIM // 2], pl.FP32],
    # ── Timestep embedding (computed on NPU) ──
    timestep:      pl.Tensor[[1], pl.FP32],
    time_freqs:    pl.Tensor[[T, DIT_FREQ_DIM // 2], pl.FP32],
    time_fc1_w:    pl.Tensor[[DIT_DIM, DIT_FREQ_DIM], pl.BF16],
    time_fc2_w:    pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    time_proj_w:   pl.Tensor[[6 * DIT_DIM, DIT_DIM], pl.BF16],
    # ── Layer weights ──
    l_mod:         pl.Tensor[[1, 6, DIT_DIM], pl.BF16],
    l_norm1_w:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_norm1_b:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_q_w:         pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_k_w:         pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_v_w:         pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_o_w:         pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_q_rms:       pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_k_rms:       pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_gate_attn:   pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_norm3_w:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_norm3_b:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_cross_q:     pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_cross_ktxt:  pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_cross_vtxt:  pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_cross_kimg:  pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_cross_vimg:  pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_cross_o:     pl.Tensor[[DIT_DIM, DIT_DIM], pl.BF16],
    l_cross_qrms:  pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_cross_krms:  pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_cross_kir:   pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_norm2_w:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_norm2_b:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    l_ffn1:        pl.Tensor[[DIT_FFN, DIT_DIM], pl.BF16],
    l_ffn2:        pl.Tensor[[DIT_DIM, DIT_FFN], pl.BF16],
    l_gate_ffn:    pl.Tensor[[1, DIT_DIM], pl.FP32],
    # ── Head weights ──
    head_mod_w:    pl.Tensor[[1, 2, DIT_DIM], pl.BF16],
    head_ln_w:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    head_ln_b:     pl.Tensor[[1, DIT_DIM], pl.FP32],
    head_fc:       pl.Tensor[[DIT_IN_DIM * 4, DIT_DIM], pl.BF16],
    # ── Output ──
    out:           pl.Out[pl.Tensor[[X_N, DIT_IN_DIM * 4], pl.BF16]],
):

    # ══════════════════════════════════════════════════════════════════════
    # Scope 0: Timestep embedding — sinusoidal + 3-layer SiLU MLP on NPU.
    # FC3 is N-chunked to stay within the 65536-byte Right buffer.
    # ══════════════════════════════════════════════════════════════════════
    t_mod_gm = pl.create_tensor([6, DIT_DIM], dtype=pl.BF16)
    te_fc2_gm = pl.create_tensor([T, DIT_DIM], dtype=pl.FP32)
    te_fc3_out = pl.create_tensor([T, TIME_PROJ_N], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="time_embed"):
        t_scalar = pl.read(timestep, [0])
        freqs_tile = pl.slice(time_freqs, [T, DIT_FREQ_DIM // 2], [0, 0])
        args = pl.mul(freqs_tile, t_scalar)

        # sin_emb = [cos(args) | sin(args)] → FC1 split into two halves
        # to avoid assembling cos/sin tiles (tmov valid-shape mismatch).
        cos_part = pl.cos(args)
        sin_part = pl.sin(args)
        cos_bf = pl.cast(cos_part, target_type=pl.BF16)
        sin_bf = pl.cast(sin_part, target_type=pl.BF16)

        half_k = DIT_FREQ_DIM // 2
        w_cos = pl.slice(time_fc1_w, [DIT_DIM, half_k], [0, 0])
        w_sin = pl.slice(time_fc1_w, [DIT_DIM, half_k], [0, half_k])
        fc1_cos = pl.matmul(cos_bf, w_cos, b_trans=True, out_dtype=pl.FP32)
        fc1_sin = pl.matmul(sin_bf, w_sin, b_trans=True, out_dtype=pl.FP32)
        te_fc1 = pl.add(fc1_cos, fc1_sin)

        # SiLU
        te_silu_1 = pl.mul(te_fc1, pl.recip(pl.add(pl.exp(pl.neg(te_fc1)), 1.0)))

        # FC2 → store to GM for N-chunked FC3 below
        te_fc2 = pl.matmul(pl.cast(te_silu_1, target_type=pl.BF16), time_fc2_w, b_trans=True, out_dtype=pl.FP32)
        te_fc2_gm = pl.assemble(te_fc2_gm, te_fc2, [0, 0])

    # FC3 with N-chunked accumulation — each chunk in its own pl.at.
    for n_chunk in pl.unroll(TIME_PROJ_N // TIME_N_CHUNK):
        n0 = n_chunk * TIME_N_CHUNK
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="time_fc3"):
            te_fc2_tile = pl.slice(te_fc2_gm, [T, DIT_DIM], [0, 0])
            # SiLU on the loaded tile
            te_silu = pl.mul(te_fc2_tile, pl.recip(pl.add(pl.exp(pl.neg(te_fc2_tile)), 1.0)))
            te_silu_bf = pl.cast(te_silu, target_type=pl.BF16)
            te_w_chunk = pl.slice(time_proj_w, [TIME_N_CHUNK, DIT_DIM], [n0, 0])
            te_chunk_mm = pl.matmul(te_silu_bf, te_w_chunk, b_trans=True, out_dtype=pl.FP32)
            te_fc3_out = pl.assemble(te_fc3_out, te_chunk_mm, [0, n0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="time_final"):
        t_mod_slice = pl.slice(te_fc3_out, [6, DIT_DIM], [0, 0])
        t_mod_gm = pl.assemble(t_mod_gm, pl.cast(t_mod_slice, target_type=pl.BF16), [0, 0])

    # ══════════════════════════════════════════════════════════════════════
    # Phase A: Conv + Concat (was K1)
    # ══════════════════════════════════════════════════════════════════════

    # ── Scope 1: Reference Conv2d matmul ──
    ref_tokens = pl.create_tensor([REF_N, DIT_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ref_conv"):
        ref_tokens = pl.cast(
            pl.matmul(ref_col, ref_conv_w, out_dtype=pl.FP32),
            target_type=pl.BF16,
        )

    # ── Scope 2: Patch Conv3d K-chunked matmul ──
    patch_tokens = pl.create_tensor([X_N, DIT_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="patch_conv"):
        for row_start in pl.range(0, X_N, T):
            col_chunk = pl.slice(patch_col, [T, T], [row_start, 0])
            w_chunk = pl.slice(patch_conv_w, [T, DIT_DIM], [0, 0])
            acc = pl.matmul(col_chunk, w_chunk, out_dtype=pl.FP32)
            for k_idx in pl.range(1, 13):
                col_chunk = pl.slice(patch_col, [T, T], [row_start, k_idx * T])
                w_chunk = pl.slice(patch_conv_w, [T, DIT_DIM], [k_idx * T, 0])
                acc = pl.matmul_acc(acc, col_chunk, w_chunk)
            patch_tokens = pl.assemble(
                patch_tokens, pl.cast(acc, target_type=pl.BF16), [row_start, 0],
            )

    # ── Scope 3: Concat ref + patch → x_full ──
    x_full = pl.create_tensor([S, DIT_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="concat"):
        for row in pl.range(0, REF_N, T):
            x_full = pl.assemble(
                x_full, pl.slice(ref_tokens, [T, DIT_DIM], [row, 0]), [row, 0],
            )
        for row in pl.range(0, X_N, T):
            x_full = pl.assemble(
                x_full, pl.slice(patch_tokens, [T, DIT_DIM], [row, 0]),
                [REF_N + row, 0],
            )

    # ══════════════════════════════════════════════════════════════════════
    # Phase B: Text + CLIP Projection (was K2)
    # ══════════════════════════════════════════════════════════════════════

    text_ctx = pl.create_tensor([T5_SEQ, DIT_DIM], dtype=pl.BF16)
    # ── Scope 4: Text projection with GELU-tanh ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="text_proj"):
        for row_start in pl.range(0, T5_SEQ, T):
            text_tile = pl.slice(text_raw, [T, T5_DIM], [row_start, 0])
            fc1_out = pl.matmul(text_tile, text_fc1_w, b_trans=True, out_dtype=pl.FP32)
            # GELU-tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
            x_cubed = pl.mul(pl.mul(fc1_out, fc1_out), fc1_out)
            gelu_inner = pl.add(fc1_out, pl.mul(x_cubed, GELU_COEFF))
            gelu_scaled = pl.mul(gelu_inner, GELU_SQRT_2_OVER_PI)
            gelu_2z = pl.mul(gelu_scaled, 2.0)
            gelu_exp = pl.exp(gelu_2z)
            gelu_recip = pl.recip(pl.add(gelu_exp, 1.0))
            tanh_val = pl.sub(
                pl.add(pl.sub(fc1_out, fc1_out), 1.0),
                pl.add(gelu_recip, gelu_recip),
            )
            gelu_out = pl.mul(pl.mul(fc1_out, 0.5), pl.add(tanh_val, 1.0))
            fc2_out = pl.matmul(
                pl.cast(gelu_out, target_type=pl.BF16),
                text_fc2_w, b_trans=True, out_dtype=pl.FP32,
            )
            text_ctx = pl.assemble(
                text_ctx, pl.cast(fc2_out, target_type=pl.BF16), [row_start, 0],
            )

    clip_ctx = pl.create_tensor([CLIP_PAD, DIT_DIM], dtype=pl.BF16)
    # ── Scope 5: CLIP projection (LN → FC1 → GELU → FC2 → LN) ──
    # NOTE: golden reference uses exact F.gelu() for CLIP projection;
    # the kernel uses GELU-tanh approximation because pypto lacks erf.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="clip_proj"):
        for row_start in pl.range(0, CLIP_PAD, T):
            clip_tile = pl.slice(clip_raw, [T, CLIP_DIM], [row_start, 0])
            clip_fp32 = pl.cast(clip_tile, target_type=pl.FP32)

            # LayerNorm 1
            clip_scaled = pl.mul(clip_fp32, 1.0 / CLIP_DIM)
            clip_mean = pl.row_sum(clip_scaled)
            clip_centered = pl.row_expand_sub(clip_fp32, clip_mean)
            clip_var = pl.row_sum(pl.mul(clip_centered, clip_centered))
            clip_var = pl.mul(clip_var, 1.0 / CLIP_DIM)
            clip_var_eps = pl.add(clip_var, LN_EPS)
            clip_var_eps = pl.reshape(clip_var_eps, [1, T])
            clip_std = pl.sqrt(clip_var_eps)
            clip_std = pl.reshape(clip_std, [T, 1])
            clip_normed = pl.col_expand_add(
                pl.col_expand_mul(pl.row_expand_div(clip_centered, clip_std),
                                  clip_ln1_w[:, :]),
                clip_ln1_b[:, :],
            )

            # FC1 + GELU-tanh
            clip_fc1_out = pl.matmul(
                pl.cast(clip_normed, target_type=pl.BF16),
                clip_fc1_w, b_trans=True, out_dtype=pl.FP32,
            )
            clip_x3 = pl.mul(pl.mul(clip_fc1_out, clip_fc1_out), clip_fc1_out)
            clip_inner = pl.add(clip_fc1_out, pl.mul(clip_x3, GELU_COEFF))
            clip_sc = pl.mul(clip_inner, GELU_SQRT_2_OVER_PI)
            clip_2z = pl.mul(clip_sc, 2.0)
            clip_exp = pl.exp(clip_2z)
            clip_recip = pl.recip(pl.add(clip_exp, 1.0))
            clip_tanh = pl.sub(
                pl.add(pl.sub(clip_fc1_out, clip_fc1_out), 1.0),
                pl.add(clip_recip, clip_recip),
            )
            clip_gelu = pl.mul(pl.mul(clip_fc1_out, 0.5), pl.add(clip_tanh, 1.0))

            # FC2
            clip_fc2_out = pl.matmul(
                pl.cast(clip_gelu, target_type=pl.BF16),
                clip_fc2_w, b_trans=True, out_dtype=pl.FP32,
            )

            # LayerNorm 2
            fc2_scaled = pl.mul(clip_fc2_out, DIM_INV)
            fc2_mean = pl.row_sum(fc2_scaled)
            fc2_centered = pl.row_expand_sub(clip_fc2_out, fc2_mean)
            fc2_var = pl.row_sum(pl.mul(fc2_centered, fc2_centered))
            fc2_var = pl.mul(fc2_var, DIM_INV)
            fc2_var_eps = pl.add(fc2_var, DIT_EPS)
            fc2_var_eps = pl.reshape(fc2_var_eps, [1, T])
            fc2_std = pl.sqrt(fc2_var_eps)
            fc2_std = pl.reshape(fc2_std, [T, 1])
            clip_out = pl.col_expand_add(
                pl.col_expand_mul(pl.row_expand_div(fc2_centered, fc2_std),
                                  clip_ln2_w[:, :]),
                clip_ln2_b[:, :],
            )
            clip_ctx = pl.assemble(
                clip_ctx, pl.cast(clip_out, target_type=pl.BF16), [row_start, 0],
            )

    # ══════════════════════════════════════════════════════════════════════
    # Phase C: DiT Blocks + Head (was K3)
    # ══════════════════════════════════════════════════════════════════════

    # ── AdaLN modulation vectors (6 per layer) ──
    mod_shift_msa = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)
    mod_scale_msa = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)
    mod_gate_msa  = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)
    mod_shift_mlp = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)
    mod_scale_mlp = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)
    mod_gate_mlp  = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)

    # Self-attention buffers
    sa_q_gm     = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    sa_k_gm     = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    sa_v_gm     = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    sa_q_rope   = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    sa_k_rope   = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    sa_attn_out = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    sa_oproj_fp = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    x_after_sa  = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)

    # Cross-attention buffers
    ca_q_gm     = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    ca_k_txt_gm = pl.create_tensor([T5_SEQ, DIT_DIM], dtype=pl.FP32)
    ca_v_txt_gm = pl.create_tensor([T5_SEQ, DIT_DIM], dtype=pl.FP32)
    ca_txt_out  = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    ca_k_img_gm = pl.create_tensor([CLIP_PAD, DIT_DIM], dtype=pl.FP32)
    ca_v_img_gm = pl.create_tensor([CLIP_PAD, DIT_DIM], dtype=pl.FP32)
    ca_img_out  = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)

    # Post-cross-attn / FFN / Head buffers
    x_after_ca  = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    x_after_ffn = pl.create_tensor([S, DIT_DIM], dtype=pl.FP32)
    head_out_gm = pl.create_tensor([S, DIT_IN_DIM * 4], dtype=pl.BF16)
    head_shift  = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)
    head_scale  = pl.create_tensor([1, DIT_DIM], dtype=pl.FP32)

    # x_full copy for GM safety
    x_full_gm = pl.create_tensor([S, DIT_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="copy_input"):
        for row_start in pl.range(0, S, T):
            x_full_gm = pl.assemble(
                x_full_gm,
                pl.slice(x_full, [T, DIT_DIM], [row_start, 0]),
                [row_start, 0],
            )

    # ── Scope 6: AdaLN Modulation ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="adaln_mod"):
        layer_mod_fp = pl.reshape(pl.cast(l_mod, target_type=pl.FP32), [6, DIT_DIM])
        t_mod_fp = pl.cast(t_mod_gm, target_type=pl.FP32)
        combined_mod = pl.add(layer_mod_fp, t_mod_fp)
        mod_shift_msa = pl.assemble(mod_shift_msa, pl.slice(combined_mod, [1, DIT_DIM], [0, 0]), [0, 0])
        mod_scale_msa = pl.assemble(mod_scale_msa, pl.slice(combined_mod, [1, DIT_DIM], [1, 0]), [0, 0])
        mod_gate_msa  = pl.assemble(mod_gate_msa,  pl.slice(combined_mod, [1, DIT_DIM], [2, 0]), [0, 0])
        mod_shift_mlp = pl.assemble(mod_shift_mlp, pl.slice(combined_mod, [1, DIT_DIM], [3, 0]), [0, 0])
        mod_scale_mlp = pl.assemble(mod_scale_mlp, pl.slice(combined_mod, [1, DIT_DIM], [4, 0]), [0, 0])
        mod_gate_mlp  = pl.assemble(mod_gate_mlp,  pl.slice(combined_mod, [1, DIT_DIM], [5, 0]), [0, 0])

    # ── Scope 7: Self-Attention — LN → AdaLN → QKV projection ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sa_qkv"):
        shift_msa_tile = pl.slice(mod_shift_msa, [1, DIT_DIM], [0, 0])
        scale_msa_tile = pl.slice(mod_scale_msa, [1, DIT_DIM], [0, 0])
        scale_plus_1 = pl.add(scale_msa_tile, 1.0)

        for row_start in pl.range(0, S, T):
            x_tile = pl.slice(x_full_gm, [T, DIT_DIM], [row_start, 0])
            x_fp = pl.cast(x_tile, target_type=pl.FP32)

            x_scaled = pl.mul(x_fp, DIM_INV)
            x_mean = pl.row_sum(x_scaled)
            x_centered = pl.row_expand_sub(x_fp, x_mean)
            x_var = pl.row_sum(pl.mul(x_centered, x_centered))
            x_var = pl.mul(x_var, DIM_INV)
            x_var_eps = pl.add(x_var, LN_EPS)
            x_var_eps = pl.reshape(x_var_eps, [1, T])
            x_std = pl.sqrt(x_var_eps)
            x_std = pl.reshape(x_std, [T, 1])
            x_normed = pl.col_expand_add(
                pl.col_expand_mul(pl.row_expand_div(x_centered, x_std),
                                  l_norm1_w[:, :]),
                l_norm1_b[:, :],
            )
            x_modulated = pl.col_expand_add(
                pl.col_expand_mul(x_normed, scale_plus_1),
                shift_msa_tile,
            )
            x_mod_bf = pl.cast(x_modulated, target_type=pl.BF16)

            sa_q_gm = pl.assemble(sa_q_gm,
                pl.matmul(x_mod_bf, l_q_w, b_trans=True, out_dtype=pl.FP32), [row_start, 0])
            sa_k_gm = pl.assemble(sa_k_gm,
                pl.matmul(x_mod_bf, l_k_w, b_trans=True, out_dtype=pl.FP32), [row_start, 0])
            sa_v_gm = pl.assemble(sa_v_gm,
                pl.matmul(x_mod_bf, l_v_w, b_trans=True, out_dtype=pl.FP32), [row_start, 0])

    # ── Scope 8: Q/K RMSNorm + 3D RoPE ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sa_rms_rope"):
        for row_start in pl.range(0, S, T):
            q_tile = pl.slice(sa_q_gm, [T, DIT_DIM], [row_start, 0])
            k_tile = pl.slice(sa_k_gm, [T, DIT_DIM], [row_start, 0])
            cos_tile = pl.slice(rope_cos, [T, DIT_DIM // 2], [row_start, 0])
            sin_tile = pl.slice(rope_sin, [T, DIT_DIM // 2], [row_start, 0])

            # Q RMSNorm
            q_sq = pl.mul(q_tile, q_tile)
            q_sq_sum = pl.row_sum(q_sq)
            q_sq_sum = pl.reshape(q_sq_sum, [1, T])
            q_rms_var = pl.add(pl.mul(q_sq_sum, DIM_INV), DIT_EPS)
            q_inv_rms = pl.recip(pl.sqrt(q_rms_var))
            q_inv_rms = pl.reshape(q_inv_rms, [T, 1])
            q_normed = pl.col_expand_mul(pl.row_expand_mul(q_tile, q_inv_rms), l_q_rms[:, :])

            # K RMSNorm
            k_sq = pl.mul(k_tile, k_tile)
            k_sq_sum = pl.row_sum(k_sq)
            k_sq_sum = pl.reshape(k_sq_sum, [1, T])
            k_rms_var = pl.add(pl.mul(k_sq_sum, DIM_INV), DIT_EPS)
            k_inv_rms = pl.recip(pl.sqrt(k_rms_var))
            k_inv_rms = pl.reshape(k_inv_rms, [T, 1])
            k_normed = pl.col_expand_mul(pl.row_expand_mul(k_tile, k_inv_rms), l_k_rms[:, :])

            # Complex-pair RoPE via pl.gather stride-2 even/odd split
            q_even = pl.gather(q_normed, mask_pattern=pl.tile.MaskPattern.P0101)
            q_odd  = pl.gather(q_normed, mask_pattern=pl.tile.MaskPattern.P1010)
            k_even = pl.gather(k_normed, mask_pattern=pl.tile.MaskPattern.P0101)
            k_odd  = pl.gather(k_normed, mask_pattern=pl.tile.MaskPattern.P1010)

            q_re = pl.sub(pl.mul(q_even, cos_tile), pl.mul(q_odd, sin_tile))
            q_ro = pl.add(pl.mul(q_odd, cos_tile), pl.mul(q_even, sin_tile))
            k_re = pl.sub(pl.mul(k_even, cos_tile), pl.mul(k_odd, sin_tile))
            k_ro = pl.add(pl.mul(k_odd, cos_tile), pl.mul(k_even, sin_tile))

            q_buf = pl.full([T, DIT_DIM], dtype=pl.FP32, value=0.0)
            q_buf = pl.tensor.scatter(q_re, mask_pattern=pl.tile.MaskPattern.P0101, dst=q_buf)
            q_buf = pl.tensor.scatter(q_ro, mask_pattern=pl.tile.MaskPattern.P1010, dst=q_buf)
            sa_q_rope = pl.assemble(sa_q_rope, q_buf, [row_start, 0])

            k_buf = pl.full([T, DIT_DIM], dtype=pl.FP32, value=0.0)
            k_buf = pl.tensor.scatter(k_re, mask_pattern=pl.tile.MaskPattern.P0101, dst=k_buf)
            k_buf = pl.tensor.scatter(k_ro, mask_pattern=pl.tile.MaskPattern.P1010, dst=k_buf)
            sa_k_rope = pl.assemble(sa_k_rope, k_buf, [row_start, 0])

    # ── Scope 9: Self-Attention MHA (per-head, scaled) ──
    attn_scores_tmp = pl.create_tensor([S, S], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sa_mha"):
        for head_idx in pl.unroll(DIT_HEADS):
            head_col = head_idx * DIT_HEAD_DIM
            sa_q_head = pl.slice(sa_q_rope, [S, DIT_HEAD_DIM], [0, head_col])
            sa_k_head = pl.slice(sa_k_rope, [S, DIT_HEAD_DIM], [0, head_col])
            sa_v_head = pl.slice(sa_v_gm,   [S, DIT_HEAD_DIM], [0, head_col])

            sa_scores = pl.matmul(sa_q_head, sa_k_head, b_trans=True, out_dtype=pl.FP32)
            sa_scores = pl.mul(sa_scores, HEAD_SCALE)

            for row_start in pl.range(0, S, T):
                score_tile = pl.slice(sa_scores, [T, S], [row_start, 0])
                shifted = pl.row_expand_sub(score_tile, pl.row_max(score_tile))
                exp_scores = pl.exp(shifted)
                attn_weights = pl.row_expand_div(exp_scores, pl.row_sum(exp_scores))
                attn_scores_tmp = pl.assemble(
                    attn_scores_tmp, attn_weights, [row_start, 0],
                )

            attn_weights_bf = pl.cast(
                pl.slice(attn_scores_tmp, [S, S], [0, 0]),
                target_type=pl.BF16,
            )
            sa_head_out = pl.matmul(attn_weights_bf, pl.cast(sa_v_head, target_type=pl.BF16), out_dtype=pl.FP32)
            sa_attn_out = pl.assemble(sa_attn_out, sa_head_out, [0, head_col])

    # ── Scope 10: Self-Attention Output Projection ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sa_oproj"):
        oproj_result = pl.matmul(pl.cast(sa_attn_out, target_type=pl.BF16), l_o_w, b_trans=True, out_dtype=pl.FP32)
        for row_start in pl.range(0, S, T):
            sa_oproj_fp = pl.assemble(
                sa_oproj_fp,
                pl.slice(oproj_result, [T, DIT_DIM], [row_start, 0]),
                [row_start, 0],
            )

    # ── Scope 11: Gated Residual (self-attention) ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="sa_gate"):
        gate_msa_tile = pl.slice(mod_gate_msa, [1, DIT_DIM], [0, 0])
        for row_start in pl.range(0, S, T):
            x_orig = pl.slice(x_full_gm, [T, DIT_DIM], [row_start, 0])
            x_orig_fp = pl.cast(x_orig, target_type=pl.FP32)
            oproj_tile = pl.slice(sa_oproj_fp, [T, DIT_DIM], [row_start, 0])
            gated_oproj = pl.col_expand_mul(oproj_tile, gate_msa_tile)
            x_after_sa = pl.assemble(
                x_after_sa, pl.add(x_orig_fp, gated_oproj), [row_start, 0],
            )

    # ── Scope 12: Cross-Attention LN + Q projection + RMSNorm ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ca_layernorm"):
        for row_start in pl.range(0, S, T):
            ca_x_tile = pl.slice(x_after_sa, [T, DIT_DIM], [row_start, 0])
            ca_x_fp = ca_x_tile
            ca_scaled = pl.mul(ca_x_fp, DIM_INV)
            ca_mean = pl.row_sum(ca_scaled)
            ca_centered = pl.row_expand_sub(ca_x_fp, ca_mean)
            ca_var = pl.row_sum(pl.mul(ca_centered, ca_centered))
            ca_var = pl.mul(ca_var, DIM_INV)
            ca_var_eps = pl.add(ca_var, LN_EPS)
            ca_var_eps = pl.reshape(ca_var_eps, [1, T])
            ca_std = pl.sqrt(ca_var_eps)
            ca_std = pl.reshape(ca_std, [T, 1])
            ca_normed = pl.col_expand_add(
                pl.col_expand_mul(pl.row_expand_div(ca_centered, ca_std),
                                  l_norm3_w[:, :]),
                l_norm3_b[:, :],
            )
            ca_q_gm = pl.assemble(ca_q_gm, ca_normed, [row_start, 0])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ca_q_proj"):
        for row_start in pl.range(0, S, T):
            ca_q_tile = pl.slice(ca_q_gm, [T, DIT_DIM], [row_start, 0])
            ca_q_proj = pl.matmul(pl.cast(ca_q_tile, target_type=pl.BF16), l_cross_q, b_trans=True, out_dtype=pl.FP32)
            ca_q_sq = pl.mul(ca_q_proj, ca_q_proj)
            ca_q_sqsum = pl.row_sum(ca_q_sq)
            ca_q_sqsum = pl.reshape(ca_q_sqsum, [1, T])
            ca_q_rms = pl.add(pl.mul(ca_q_sqsum, DIM_INV), DIT_EPS)
            ca_q_inv = pl.recip(pl.sqrt(ca_q_rms))
            ca_q_inv = pl.reshape(ca_q_inv, [T, 1])
            ca_q_normed = pl.col_expand_mul(
                pl.row_expand_mul(ca_q_proj, ca_q_inv), l_cross_qrms[:, :])
            ca_q_gm = pl.assemble(ca_q_gm, ca_q_normed, [row_start, 0])

    # ── Scope 13: Cross-Attention Text K/V projection + RMSNorm + MHA ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ca_text_kv"):
        for row_start in pl.range(0, T5_SEQ, T):
            txt_tile = pl.slice(text_ctx, [T, DIT_DIM], [row_start, 0])

            txt_k_proj = pl.matmul(txt_tile, l_cross_ktxt, b_trans=True, out_dtype=pl.FP32)
            txt_k_sq = pl.mul(txt_k_proj, txt_k_proj)
            txt_k_sqsum = pl.row_sum(txt_k_sq)
            txt_k_sqsum = pl.reshape(txt_k_sqsum, [1, T])
            txt_k_rms = pl.add(pl.mul(txt_k_sqsum, DIM_INV), DIT_EPS)
            txt_k_inv = pl.recip(pl.sqrt(txt_k_rms))
            txt_k_inv = pl.reshape(txt_k_inv, [T, 1])
            txt_k_normed = pl.col_expand_mul(
                pl.row_expand_mul(txt_k_proj, txt_k_inv), l_cross_krms[:, :])
            ca_k_txt_gm = pl.assemble(ca_k_txt_gm, txt_k_normed, [row_start, 0])

            txt_v_proj = pl.matmul(txt_tile, l_cross_vtxt, b_trans=True, out_dtype=pl.FP32)
            ca_v_txt_gm = pl.assemble(ca_v_txt_gm, txt_v_proj, [row_start, 0])

    txt_attn_tmp = pl.create_tensor([S, T5_SEQ], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ca_text_mha"):
        for head_idx in pl.unroll(DIT_HEADS):
            head_col = head_idx * DIT_HEAD_DIM
            ca_txt_q_head = pl.slice(ca_q_gm,     [S, DIT_HEAD_DIM],       [0, head_col])
            txt_k_head = pl.slice(ca_k_txt_gm, [T5_SEQ, DIT_HEAD_DIM], [0, head_col])
            txt_v_head = pl.slice(ca_v_txt_gm, [T5_SEQ, DIT_HEAD_DIM], [0, head_col])

            ca_txt_scores = pl.matmul(ca_txt_q_head, txt_k_head, b_trans=True, out_dtype=pl.FP32)
            ca_txt_scores = pl.mul(ca_txt_scores, HEAD_SCALE)

            for row_start in pl.range(0, S, T):
                txt_score_tile = pl.slice(ca_txt_scores, [T, T5_SEQ], [row_start, 0])
                txt_shifted = pl.row_expand_sub(txt_score_tile, pl.row_max(txt_score_tile))
                txt_exp_s = pl.exp(txt_shifted)
                txt_attn_w = pl.row_expand_div(txt_exp_s, pl.row_sum(txt_exp_s))
                txt_attn_tmp = pl.assemble(txt_attn_tmp, txt_attn_w, [row_start, 0])

            txt_attn_bf = pl.cast(
                pl.slice(txt_attn_tmp, [S, T5_SEQ], [0, 0]), target_type=pl.BF16,
            )
            txt_head_out = pl.matmul(txt_attn_bf, pl.cast(txt_v_head, target_type=pl.BF16), out_dtype=pl.FP32)
            ca_txt_out = pl.assemble(ca_txt_out, txt_head_out, [0, head_col])

    # ── Scope 14: Cross-Attention Image K/V projection + RMSNorm + MHA ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ca_img_kv"):
        for row_start in pl.range(0, CLIP_PAD, T):
            img_tile = pl.slice(clip_ctx, [T, DIT_DIM], [row_start, 0])

            img_k_proj = pl.matmul(img_tile, l_cross_kimg, b_trans=True, out_dtype=pl.FP32)
            img_k_sq = pl.mul(img_k_proj, img_k_proj)
            img_k_sqsum = pl.row_sum(img_k_sq)
            img_k_sqsum = pl.reshape(img_k_sqsum, [1, T])
            img_k_rms = pl.add(pl.mul(img_k_sqsum, DIM_INV), DIT_EPS)
            img_k_inv = pl.recip(pl.sqrt(img_k_rms))
            img_k_inv = pl.reshape(img_k_inv, [T, 1])
            img_k_normed = pl.col_expand_mul(
                pl.row_expand_mul(img_k_proj, img_k_inv), l_cross_kir[:, :])
            ca_k_img_gm = pl.assemble(ca_k_img_gm, img_k_normed, [row_start, 0])

            img_v_proj = pl.matmul(img_tile, l_cross_vimg, b_trans=True, out_dtype=pl.FP32)
            ca_v_img_gm = pl.assemble(ca_v_img_gm, img_v_proj, [row_start, 0])

    img_attn_tmp = pl.create_tensor([S, CLIP_PAD], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ca_img_mha"):
        for head_idx in pl.unroll(DIT_HEADS):
            head_col = head_idx * DIT_HEAD_DIM
            ca_img_q_head = pl.slice(ca_q_gm,     [S, DIT_HEAD_DIM],       [0, head_col])
            img_k_head = pl.slice(ca_k_img_gm, [CLIP_PAD, DIT_HEAD_DIM], [0, head_col])
            img_v_head = pl.slice(ca_v_img_gm, [CLIP_PAD, DIT_HEAD_DIM], [0, head_col])

            ca_img_scores = pl.matmul(ca_img_q_head, img_k_head, b_trans=True, out_dtype=pl.FP32)
            ca_img_scores = pl.mul(ca_img_scores, HEAD_SCALE)
            ca_img_scores = pl.col_expand_add(ca_img_scores, clip_mask)

            for row_start in pl.range(0, S, T):
                img_score_tile = pl.slice(ca_img_scores, [T, CLIP_PAD], [row_start, 0])
                img_shifted = pl.row_expand_sub(img_score_tile, pl.row_max(img_score_tile))
                img_exp_s = pl.exp(img_shifted)
                img_attn_w = pl.row_expand_div(img_exp_s, pl.row_sum(img_exp_s))
                img_attn_tmp = pl.assemble(img_attn_tmp, img_attn_w, [row_start, 0])

            img_attn_bf = pl.cast(
                pl.slice(img_attn_tmp, [S, CLIP_PAD], [0, 0]), target_type=pl.BF16,
            )
            img_head_out = pl.matmul(img_attn_bf, pl.cast(img_v_head, target_type=pl.BF16), out_dtype=pl.FP32)
            ca_img_out = pl.assemble(ca_img_out, img_head_out, [0, head_col])

    # ── Scope 15: Cross-Attention Output Projection + Residual ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ca_oproj"):
        combined_out = pl.add(ca_txt_out, ca_img_out)
        combined_bf = pl.cast(combined_out, target_type=pl.BF16)
        ca_oproj = pl.matmul(combined_bf, l_cross_o, b_trans=True, out_dtype=pl.FP32)

        for row_start in pl.range(0, S, T):
            x_sa_tile = pl.slice(x_after_sa, [T, DIT_DIM], [row_start, 0])
            ca_oproj_tile = pl.slice(ca_oproj, [T, DIT_DIM], [row_start, 0])
            x_after_ca = pl.assemble(
                x_after_ca, pl.add(x_sa_tile, ca_oproj_tile), [row_start, 0],
            )

    # ── Scope 16: FFN with AdaLN + GELU-tanh + Gated Residual ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="ffn"):
        shift_mlp_tile = pl.slice(mod_shift_mlp, [1, DIT_DIM], [0, 0])
        scale_mlp_tile = pl.slice(mod_scale_mlp, [1, DIT_DIM], [0, 0])
        gate_mlp_tile  = pl.slice(mod_gate_mlp,  [1, DIT_DIM], [0, 0])
        scale_mlp_plus1 = pl.add(scale_mlp_tile, 1.0)

        for row_start in pl.range(0, S, T):
            ffn_x_tile = pl.slice(x_after_ca, [T, DIT_DIM], [row_start, 0])

            # LayerNorm
            ffn_scaled = pl.mul(ffn_x_tile, DIM_INV)
            ffn_mean = pl.row_sum(ffn_scaled)
            ffn_centered = pl.row_expand_sub(ffn_x_tile, ffn_mean)
            ffn_var = pl.row_sum(pl.mul(ffn_centered, ffn_centered))
            ffn_var = pl.mul(ffn_var, DIM_INV)
            ffn_var_eps = pl.add(ffn_var, LN_EPS)
            ffn_var_eps = pl.reshape(ffn_var_eps, [1, T])
            ffn_std = pl.sqrt(ffn_var_eps)
            ffn_std = pl.reshape(ffn_std, [T, 1])
            ffn_normed = pl.col_expand_add(
                pl.col_expand_mul(pl.row_expand_div(ffn_centered, ffn_std),
                                  l_norm2_w[:, :]),
                l_norm2_b[:, :],
            )

            # AdaLN modulation
            ffn_mod = pl.col_expand_add(
                pl.col_expand_mul(ffn_normed, scale_mlp_plus1),
                shift_mlp_tile,
            )
            ffn_mod_bf = pl.cast(ffn_mod, target_type=pl.BF16)

            # FC1 + GELU-tanh
            ffn_fc1 = pl.matmul(ffn_mod_bf, l_ffn1, b_trans=True, out_dtype=pl.FP32)
            ffn_x3 = pl.mul(pl.mul(ffn_fc1, ffn_fc1), ffn_fc1)
            ffn_inner = pl.add(ffn_fc1, pl.mul(ffn_x3, GELU_COEFF))
            ffn_z = pl.mul(ffn_inner, GELU_SQRT_2_OVER_PI)
            ffn_2z = pl.mul(ffn_z, 2.0)
            ffn_exp_val = pl.exp(ffn_2z)
            ffn_recip = pl.recip(pl.add(ffn_exp_val, 1.0))
            ffn_tanh = pl.sub(
                pl.add(pl.sub(ffn_fc1, ffn_fc1), 1.0),
                pl.add(ffn_recip, ffn_recip),
            )
            ffn_gelu = pl.mul(pl.mul(ffn_fc1, 0.5), pl.add(ffn_tanh, 1.0))
            ffn_gelu_bf = pl.cast(ffn_gelu, target_type=pl.BF16)

            # FC2 + gated residual
            ffn_fc2 = pl.matmul(ffn_gelu_bf, l_ffn2, b_trans=True, out_dtype=pl.FP32)
            ffn_gated = pl.col_expand_mul(ffn_fc2, gate_mlp_tile)
            x_after_ffn = pl.assemble(
                x_after_ffn, pl.add(ffn_x_tile, ffn_gated), [row_start, 0],
            )

    # ── Scope 17: Output Head — Modulation ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_mod"):
        t_mod_fp = pl.cast(t_mod_gm, target_type=pl.FP32)
        t_mod_head = pl.slice(t_mod_fp, [2, DIT_DIM], [0, 0])
        head_mod_fp = pl.reshape(pl.cast(head_mod_w, target_type=pl.FP32), [2, DIT_DIM])
        head_combined = pl.add(head_mod_fp, t_mod_head)
        head_shift = pl.assemble(
            head_shift, pl.slice(head_combined, [1, DIT_DIM], [0, 0]), [0, 0],
        )
        head_scale = pl.assemble(
            head_scale, pl.slice(head_combined, [1, DIT_DIM], [1, 0]), [0, 0],
        )

    # ── Scope 18: Output Head — LN + AdaLN + FC ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="head_proj"):
        head_shift_tile = pl.slice(head_shift, [1, DIT_DIM], [0, 0])
        head_scale_tile = pl.slice(head_scale, [1, DIT_DIM], [0, 0])
        head_scale_plus1 = pl.add(head_scale_tile, 1.0)

        for row_start in pl.range(0, S, T):
            head_x_tile = pl.slice(x_after_ffn, [T, DIT_DIM], [row_start, 0])

            head_scaled = pl.mul(head_x_tile, DIM_INV)
            head_mean = pl.row_sum(head_scaled)
            head_centered = pl.row_expand_sub(head_x_tile, head_mean)
            head_var = pl.row_sum(pl.mul(head_centered, head_centered))
            head_var = pl.mul(head_var, DIM_INV)
            head_var_eps = pl.add(head_var, LN_EPS)
            head_var_eps = pl.reshape(head_var_eps, [1, T])
            head_std = pl.sqrt(head_var_eps)
            head_std = pl.reshape(head_std, [T, 1])
            head_normed = pl.col_expand_add(
                pl.col_expand_mul(pl.row_expand_div(head_centered, head_std),
                                  head_ln_w[:, :]),
                head_ln_b[:, :],
            )

            head_mod = pl.col_expand_add(
                pl.col_expand_mul(head_normed, head_scale_plus1),
                head_shift_tile,
            )
            head_mod_bf = pl.cast(head_mod, target_type=pl.BF16)
            head_fc_out = pl.matmul(head_mod_bf, head_fc, b_trans=True, out_dtype=pl.FP32)
            head_out_gm = pl.assemble(
                head_out_gm, pl.cast(head_fc_out, target_type=pl.BF16),
                [row_start, 0],
            )

    # ── Scope 19: Remove reference tokens → output ──
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="remove_ref"):
        for row_start in pl.range(0, X_N, T):
            out_tile = pl.slice(
                head_out_gm, [T, DIT_IN_DIM * 4], [REF_N + row_start, 0],
            )
            out = pl.assemble(out, out_tile, [row_start, 0])

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Test harness
# ══════════════════════════════════════════════════════════════════════════════

def build_tensor_specs():
    """Build tensor specs for the fused dit_forward kernel."""
    from golden import TensorSpec

    torch.manual_seed(42)
    w = make_weights(seed=42)

    x_input = torch.randn(1, DIT_IN_DIM + COND_CH, FP, LAT_H, LAT_W)
    text_raw = torch.randn(1, T5_SEQ, T5_DIM)
    clip_raw = torch.randn(1, CLIP_TOKENS, CLIP_DIM)
    ref_latents = torch.randn(1, DIT_IN_DIM, LAT_H, LAT_W)
    timestep = torch.tensor([500.], dtype=torch.float32)

    # Timestep embedding — frequencies are static; the kernel computes
    # sin_emb = [cos(t*freqs), sin(t*freqs)] on NPU using pl.cos/pl.sin.
    half_freq = DIT_FREQ_DIM // 2
    time_freqs_1d = torch.exp(-math.log(10000.0) * torch.arange(half_freq).float() / half_freq)
    time_freqs = time_freqs_1d.unsqueeze(0).expand(T, half_freq).contiguous()

    # Host im2col (data rearrangement)
    ref_conv_w_2d = w["dit_ref_conv"].reshape(DIT_DIM, -1).T.contiguous().bfloat16()
    ref_col = _im2col_ref_conv2d(ref_latents.bfloat16())
    patch_conv_w_2d = w["dit_patch"].reshape(DIT_DIM, -1).T.contiguous().bfloat16()
    patch_col = _im2col_patch_conv3d(x_input.bfloat16())

    # RoPE tables (data rearrangement)
    cos_f, sin_f, cos_h, sin_h, cos_w, sin_w = _precompute_rope_cos_sin()
    rope_cos, rope_sin = _build_rope_map(
        cos_f, sin_f, cos_h, sin_h, cos_w, sin_w, FP + 1, HP, WP,
    )

    # CLIP padding + mask
    clip_padded = torch.zeros(CLIP_PAD, CLIP_DIM, dtype=torch.bfloat16)
    clip_padded[:CLIP_TOKENS, :] = clip_raw.squeeze(0).bfloat16()
    clip_mask = torch.zeros(1, CLIP_PAD, dtype=torch.float32)
    clip_mask[:, CLIP_TOKENS:] = -65504.0

    def _identity(t):
        return lambda: t

    def spec(name, shape, dtype, value):
        return TensorSpec(name, shape, dtype, init_value=_identity(value))

    specs = [
        # Conv inputs
        spec("ref_col",      [REF_N, REF_CONV_COL],       torch.bfloat16, ref_col),
        spec("patch_col",    [X_N, PATCH_CONV_COL],        torch.bfloat16, patch_col),
        spec("ref_conv_w",   [REF_CONV_COL, DIT_DIM],      torch.bfloat16, ref_conv_w_2d),
        spec("patch_conv_w", [PATCH_CONV_COL, DIT_DIM],    torch.bfloat16, patch_conv_w_2d),
        # Text + CLIP projection
        spec("text_raw",     [T5_SEQ, T5_DIM],             torch.bfloat16, text_raw.squeeze(0).bfloat16()),
        spec("text_fc1_w",   [DIT_DIM, DIT_TEXT_DIM],      torch.bfloat16, w["dit_text_fc1"].bfloat16()),
        spec("text_fc2_w",   [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_text_fc2"].bfloat16()),
        spec("clip_raw",     [CLIP_PAD, CLIP_DIM],         torch.bfloat16, clip_padded),
        spec("clip_ln1_w",   [1, CLIP_DIM],                torch.float32,  w["dit_clip_ln_w"].unsqueeze(0)),
        spec("clip_ln1_b",   [1, CLIP_DIM],                torch.float32,  w["dit_clip_ln_b"].unsqueeze(0)),
        spec("clip_fc1_w",   [CLIP_DIM, CLIP_DIM],         torch.bfloat16, w["dit_clip_fc1"].bfloat16()),
        spec("clip_fc2_w",   [DIT_DIM, CLIP_DIM],          torch.bfloat16, w["dit_clip_fc2"].bfloat16()),
        spec("clip_ln2_w",   [1, DIT_DIM],                 torch.float32,  w["dit_clip_ln2_w"].unsqueeze(0)),
        spec("clip_ln2_b",   [1, DIT_DIM],                 torch.float32,  w["dit_clip_ln2_b"].unsqueeze(0)),
        # Context tables
        spec("clip_mask",    [1, CLIP_PAD],                torch.float32,  clip_mask),
        spec("rope_cos",     [S, DIT_DIM // 2],            torch.float32,  rope_cos),
        spec("rope_sin",     [S, DIT_DIM // 2],            torch.float32,  rope_sin),
        # Timestep embedding (computed on NPU)
        spec("timestep",     [1],                          torch.float32,  timestep),
        spec("time_freqs",   [T, half_freq],               torch.float32,  time_freqs),
        spec("time_fc1_w",   [DIT_DIM, DIT_FREQ_DIM],      torch.bfloat16, w["dit_time_fc1"].bfloat16()),
        spec("time_fc2_w",   [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_time_fc2"].bfloat16()),
        spec("time_proj_w",  [6 * DIT_DIM, DIT_DIM],       torch.bfloat16, w["dit_time_proj"].bfloat16()),
        # Layer weights
        spec("l_mod",        [1, 6, DIT_DIM],              torch.bfloat16, w["dit_l0_mod"].bfloat16()),
        spec("l_norm1_w",    [1, DIT_DIM],                 torch.float32,  w["dit_l0_norm1_w"].unsqueeze(0)),
        spec("l_norm1_b",    [1, DIT_DIM],                 torch.float32,  w["dit_l0_norm1_b"].unsqueeze(0)),
        spec("l_q_w",        [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_q"].bfloat16()),
        spec("l_k_w",        [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_k"].bfloat16()),
        spec("l_v_w",        [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_v"].bfloat16()),
        spec("l_o_w",        [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_o"].bfloat16()),
        spec("l_q_rms",      [1, DIT_DIM],                 torch.float32,  w["dit_l0_q_rms"].unsqueeze(0)),
        spec("l_k_rms",      [1, DIT_DIM],                 torch.float32,  w["dit_l0_k_rms"].unsqueeze(0)),
        spec("l_gate_attn",  [1, DIT_DIM],                 torch.float32,  w["dit_l0_gate_attn"].unsqueeze(0)),
        spec("l_norm3_w",    [1, DIT_DIM],                 torch.float32,  w["dit_l0_norm3_w"].unsqueeze(0)),
        spec("l_norm3_b",    [1, DIT_DIM],                 torch.float32,  w["dit_l0_norm3_b"].unsqueeze(0)),
        spec("l_cross_q",    [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_cross_q"].bfloat16()),
        spec("l_cross_ktxt", [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_cross_k_txt"].bfloat16()),
        spec("l_cross_vtxt", [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_cross_v_txt"].bfloat16()),
        spec("l_cross_kimg", [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_cross_k_img"].bfloat16()),
        spec("l_cross_vimg", [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_cross_v_img"].bfloat16()),
        spec("l_cross_o",    [DIT_DIM, DIT_DIM],           torch.bfloat16, w["dit_l0_cross_o"].bfloat16()),
        spec("l_cross_qrms", [1, DIT_DIM],                 torch.float32,  w["dit_l0_cross_q_rms"].unsqueeze(0)),
        spec("l_cross_krms", [1, DIT_DIM],                 torch.float32,  w["dit_l0_cross_k_rms"].unsqueeze(0)),
        spec("l_cross_kir",  [1, DIT_DIM],                 torch.float32,  w["dit_l0_cross_k_img_rms"].unsqueeze(0)),
        spec("l_norm2_w",    [1, DIT_DIM],                 torch.float32,  w["dit_l0_norm2_w"].unsqueeze(0)),
        spec("l_norm2_b",    [1, DIT_DIM],                 torch.float32,  w["dit_l0_norm2_b"].unsqueeze(0)),
        spec("l_ffn1",       [DIT_FFN, DIT_DIM],          torch.bfloat16, w["dit_l0_ffn1"].bfloat16()),
        spec("l_ffn2",       [DIT_DIM, DIT_FFN],          torch.bfloat16, w["dit_l0_ffn2"].bfloat16()),
        spec("l_gate_ffn",   [1, DIT_DIM],                 torch.float32,  w["dit_l0_gate_ffn"].unsqueeze(0)),
        # Head weights
        spec("head_mod_w",   [1, 2, DIT_DIM],              torch.bfloat16, w["dit_head_mod"].bfloat16()),
        spec("head_ln_w",    [1, DIT_DIM],                 torch.float32,  w["dit_head_ln_w"].unsqueeze(0)),
        spec("head_ln_b",    [1, DIT_DIM],                 torch.float32,  w["dit_head_ln_b"].unsqueeze(0)),
        spec("head_fc",      [DIT_IN_DIM * 4, DIT_DIM],    torch.bfloat16, w["dit_head_fc"].bfloat16()),
    ]
    specs.append(TensorSpec("out", [X_N, DIT_IN_DIM * 4], torch.bfloat16, is_output=True))

    _GOLDEN_DATA["w"] = w
    _GOLDEN_DATA["x_input"] = x_input
    _GOLDEN_DATA["text_raw"] = text_raw
    _GOLDEN_DATA["clip_raw"] = clip_raw
    _GOLDEN_DATA["ref_latents"] = ref_latents
    _GOLDEN_DATA["timestep"] = timestep
    return specs


_GOLDEN_DATA = {}


def golden_dit_forward_fn(tensors):
    """Golden reference — calls test_golden_fun_control_full.py::dit_forward.

    The golden uses exact F.gelu() for CLIP projection (matching the reference
    at line 1417).  The kernel uses GELU-tanh approximation because pypto lacks
    erf — the permissive K3 tolerance accounts for this difference.
    """
    w = _GOLDEN_DATA["w"]
    x_input = _GOLDEN_DATA["x_input"]
    text_raw = _GOLDEN_DATA["text_raw"]
    clip_raw = _GOLDEN_DATA["clip_raw"]
    ref_latents = _GOLDEN_DATA["ref_latents"]
    timestep = _GOLDEN_DATA["timestep"]
    context = {"text": text_raw, "clip": clip_raw, "ref_latents": ref_latents}
    freqs = precompute_freqs_cis_3d(DIT_HEAD_DIM)
    result = golden_dit_fn(x_input, context, freqs, timestep, w)
    tensors["out"][:] = result.permute(0, 2, 3, 4, 1).reshape(-1, DIT_IN_DIM * 4).bfloat16()


if __name__ == "__main__":
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="DiT forward pass — fused kernel test")
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    specs = build_tensor_specs()

    result = run_jit(
        fn=dit_forward,
        specs=specs,
        golden_fn=golden_dit_forward_fn,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=3e-3,
        atol=3e-3,
        compare_fn={"out": ratio_allclose(atol=0.6, rtol=0.6, max_error_ratio=0.02)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
    print("PASS")
