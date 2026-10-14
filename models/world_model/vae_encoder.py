# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""VAE encoder for Fun-Control 1.3B — pypto3.0 implementation.

Implements the VAE encoder sub-network from the pipeline in
``infer_fun_control_1_3b_text.py`` (DiffSynth WanVideoPipeline VAE).
All arithmetic (matmul, SiLU, RMS norm, softmax, attention) runs inside
pypto kernels.  im2col / reshape / permute are data-rearrangement only.

The golden reference for precision comparison is
``test_golden_fun_control_full.py::vae_encode``, a pure-PyTorch
reimplementation of the SAME VAE logic used by the DiffSynth pipeline.
The two are logically identical because ``test_golden_fun_control_full.py``
replicates every VAE computation from the DiffSynth source that
``infer_fun_control_1_3b_text.py`` invokes via ``WanVideoPipeline``.

Legitimate differences from the golden reference:

  1. **Feature caching not implemented.**  The golden reference maintains
     ``feat_cache`` / ``feat_idx`` lists for chunked autoregressive
     inference where successive chunks share temporal context.  The pypto
     implementation targets single-chunk inference (CHUNK_SIZE=5 golden
     case); multi-chunk caching is a host-side orchestration concern
     handled at the pipeline integration level, not inside the NPU kernel.
  2. **Output temporal dimension = 1** (not 2).  For CHUNK_SIZE=5 the
     encoder's natural temporal output is 1 (3× CausalConv3d stride=(2,1,1)
     downsample: 5→3→2→1).  The golden test uses LAT_F=2 as a
     DiT-compatible target and interpolates.  The pypto encoder/decoder
     work at the natural resolution; DiT interpolation is a
     pipeline-level remap.
  3. **Attention processes per-temporal-slice** (loop over ``batch_temporal``).
     The golden uses ``F.scaled_dot_product_attention`` which handles all
     heads/slices at once.  The pypto version processes each temporal slice
     independently because spatial dimensions are small (8×8 →
     batch_temporal ≤ 2).

Architecture::

    Input [1,3,5,64,64]
    → CausalConv3d(3→96)
    → Stage 0: ResBlock(96→192)+ResBlock(192)+Downsample  → [1,192,3,32,32]
    → Stage 1: ResBlock(192→384)+ResBlock(384)+Downsample  → [1,384,2,16,16]
    → Stage 2: ResBlock(384→768)+ResBlock(768)+Downsample  → [1,768,1,8,8]
    → Stage 3: ResBlock(768→1536)+ResBlock(1536)           → [1,1536,1,8,8]
    → Middle: ResBlock+Attention+ResBlock                   → [1,1536,1,8,8]
    → CausalConv3d(1536→32) → chunk → mu [1,16,1,8,8]
    → scale_norm → output [1,16,1,8,8]

Multi-kernel design rationale:
    Convolution im2col (F.pad + .unfold) is host-side data rearrangement.
    Each convolution's matmul + each elementwise op (RMS_norm, SiLU, etc.)
    runs in its own @pl.jit kernel, dispatched from host.  This is the
    standard pattern for conv-heavy networks in pypto.

Precision comparison uses ``ratio_allclose`` with tolerance atol=0.1 /
rtol=0.1 / max_error_ratio=0.02 — same pattern as decode_layer.py but with
looser tolerances justified by the deeper VAE network (25+ RMSNorm ops,
each contributing ~0.012 hardware accuracy loss on Ascend vector unit;
see memory.md Phase 3 for per-op precision analysis).

Usage::

    python vae_encoder.py -p a2a3 -d 0
"""

import argparse
import sys
import time

import pypto.language as pl
import torch
import torch.nn.functional as F

from config import VAE_Z_DIM, H, W, CHUNK_SIZE, VAE_ENC_CH

# ══════════════════════════════════════════════════════════════════════════════
# Functional config — model architecture + workload.
# ══════════════════════════════════════════════════════════════════════════════

VAE_CH = VAE_ENC_CH             # [96, 192, 384, 768, 1536]
VAE_OUT_CH = VAE_Z_DIM * 2     # 32
T_IN = CHUNK_SIZE            # 5

# ── Tiling parameters (optimization config) ──
# See decode_layer.py pattern: each kernel type gets its own tile size tuned
# to the NPU's M/N/K alignment requirements.  Values verified on Ascend910B.

R_TILE = 16           # row-tile for pl.parallel dispatch (mm_col_w / softmax / bias / scale / resadd)
K_CHUNK = 1024        # inner K-dim chunk for matmul accumulation loop
N_CHUNK = 64          # outer N-dim chunk for matmul output tiling
SILU_TILE = 64        # vector-row tile for SiLU elementwise
SILU_COL = 192        # vector-col tile for SiLU elementwise
RMS_TILE = 16         # vector-row tile for RMS norm
COL_CHUNK = 192       # vector-col tile for wide-column elementwise ops (add / bias / scale / scale_norm)
RMS_COL = 96          # vector-col chunk for RMS norm inner reduction loop

RMS_EPS = 1e-12       # matching F.normalize default eps

# ── Geometry / platform assertions (same pattern as decode_layer.py) ──
assert len(VAE_CH) == 5, f"VAE_CH must have 5 entries (4 stages + middle), got {len(VAE_CH)}"
assert VAE_CH[0] == 96, f"VAE_CH[0] must be 96 (first stage in channels), got {VAE_CH[0]}"
assert VAE_CH[4] == 1536, f"VAE_CH[4] must be 1536 (middle bottleneck), got {VAE_CH[4]}"
assert VAE_OUT_CH == VAE_Z_DIM * 2, f"VAE_OUT_CH must be VAE_Z_DIM*2={VAE_Z_DIM*2}, got {VAE_OUT_CH}"
assert T_IN == CHUNK_SIZE, f"T_IN must equal CHUNK_SIZE={CHUNK_SIZE}, got {T_IN}"


# ── Pad helpers (host-side, data rearrangement only) ──

def _pad_cols(t, align):
    c = t.shape[1]
    p = ((c + align - 1) // align) * align
    if p == c:
        return t
    o = torch.zeros(t.shape[0], p, dtype=t.dtype)
    o[:, :c] = t
    return o


def _pad_rows(t, align):
    r = t.shape[0]
    p = ((r + align - 1) // align) * align
    if p == r:
        return t
    o = torch.zeros(p, t.shape[1], dtype=t.dtype)
    o[:r] = t
    return o


# ══════════════════════════════════════════════════════════════════════════════
# NPU Kernels — each @pl.jit kernel runs one operation on NPU.
# ══════════════════════════════════════════════════════════════════════════════

@pl.jit
def mm_col_w(
    col: pl.Tensor,
    w: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """matmul(col, w^T) with K-chunked accumulation → FP32 output."""
    R = col.shape[0]
    K = col.shape[1]
    N = w.shape[0]
    for r0 in pl.parallel(0, R, R_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="mm"):
            for n0 in pl.range(0, N, N_CHUNK):
                ct0 = pl.slice(col, [R_TILE, K_CHUNK], [r0, 0])
                wt0 = pl.slice(w, [N_CHUNK, K_CHUNK], [n0, 0])
                acc = pl.matmul(ct0, wt0, b_trans=True, out_dtype=pl.FP32)
                for k0 in pl.range(K_CHUNK, K, K_CHUNK):
                    ct1 = pl.slice(col, [R_TILE, K_CHUNK], [r0, k0])
                    wt1 = pl.slice(w, [N_CHUNK, K_CHUNK], [n0, k0])
                    acc = pl.matmul_acc(acc, ct1, wt1, b_trans=True)
                out = pl.assemble(out, acc, [r0, n0])
    return out


@pl.jit
def softmax_k(
    x: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Row-wise softmax."""
    N = x.shape[1]
    for r0 in pl.parallel(0, x.shape[0], R_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax"):
            row = pl.cast(pl.slice(x, [R_TILE, N], [r0, 0]), target_type=pl.FP32)
            shifted = pl.row_expand_sub(row, pl.row_max(row))
            e = pl.exp(shifted)
            s = pl.row_expand_div(e, pl.row_sum(e))
            out = pl.assemble(out, pl.cast(s, target_type=pl.BF16), [r0, 0])
    return out


@pl.jit
def scale_k(
    x: pl.Tensor,
    scale: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """Element-wise multiply by scalar.  FP32 in → FP32 out."""
    COLS = x.shape[1]
    s = pl.read(scale, [0, 0])
    for r0 in pl.parallel(0, x.shape[0], R_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="scale"):
            for c0 in pl.range(0, COLS, COL_CHUNK):
                row = pl.slice(x, [R_TILE, COL_CHUNK], [r0, c0])
                r = pl.mul(row, s)
                out = pl.assemble(out, r, [r0, c0])
    return out


@pl.jit
def rms_norm_k(
    x: pl.Tensor,
    weight: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """RMS_norm(x, weight) FP32 in → FP32 out.  Matches F.normalize*√C*w."""
    R = x.shape[0]
    COLS = x.shape[1]
    cols_fp = pl.cast(COLS, target_type=pl.FP32)
    for r0 in pl.parallel(0, R, RMS_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="rms"):
            row0 = pl.slice(x, [RMS_TILE, RMS_COL], [r0, 0])
            sq_sum = pl.reshape(pl.row_sum(pl.mul(row0, row0)), [RMS_TILE, 1])
            for c0 in pl.range(RMS_COL, COLS, RMS_COL):
                rc = pl.slice(x, [RMS_TILE, RMS_COL], [r0, c0])
                sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rc, rc)), [RMS_TILE, 1]))
            variance = pl.add(pl.mul(sq_sum, 1.0 / cols_fp), RMS_EPS)
            inv_rms = pl.recip(pl.sqrt(variance))
            for c0 in pl.range(0, COLS, RMS_COL):
                rc = pl.slice(x, [RMS_TILE, RMS_COL], [r0, c0])
                g = pl.slice(weight, [1, RMS_COL], [0, c0])
                n = pl.col_expand_mul(pl.row_expand_mul(rc, inv_rms), g)
                out = pl.assemble(out, n, [r0, c0])
    return out


@pl.jit
def silu_k(
    x: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """SiLU(x) FP32 in → FP32 out."""
    R = x.shape[0]
    COLS = x.shape[1]
    for r0 in pl.parallel(0, R, SILU_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="silu"):
            for c0 in pl.range(0, COLS, SILU_COL):
                row = pl.slice(x, [SILU_TILE, SILU_COL], [r0, c0])
                sig = pl.recip(pl.add(pl.exp(pl.neg(row)), 1.0))
                y = pl.mul(row, sig)
                out = pl.assemble(out, y, [r0, c0])
    return out


@pl.jit
def residual_add_k(
    x: pl.Tensor,
    y: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """x + y, FP32 in → FP32 out.  Same shape."""
    COLS = x.shape[1]
    for r0 in pl.parallel(0, x.shape[0], R_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="resadd"):
            for c0 in pl.range(0, COLS, COL_CHUNK):
                xr = pl.slice(x, [R_TILE, COL_CHUNK], [r0, c0])
                yr = pl.slice(y, [R_TILE, COL_CHUNK], [r0, c0])
                z = pl.add(xr, yr)
                out = pl.assemble(out, z, [r0, c0])
    return out


@pl.jit
def bias_add_k(
    x: pl.Tensor,
    bias: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """x + bias, FP32 in → FP32 out."""
    COLS = x.shape[1]
    for r0 in pl.parallel(0, x.shape[0], R_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="bias"):
            for c0 in pl.range(0, COLS, COL_CHUNK):
                row = pl.slice(x, [R_TILE, COL_CHUNK], [r0, c0])
                bc = pl.slice(bias, [1, COL_CHUNK], [0, c0])
                z = pl.col_expand_add(row, bc)
                out = pl.assemble(out, z, [r0, c0])
    return out


@pl.jit
def scale_norm_k(
    x: pl.Tensor,
    mean: pl.Tensor,
    inv_std: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """(x - mean) * inv_std, FP32 in → FP32 out."""
    COLS = x.shape[1]
    for r0 in pl.parallel(0, x.shape[0], R_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="snorm"):
            for c0 in pl.range(0, COLS, COL_CHUNK):
                row = pl.slice(x, [R_TILE, COL_CHUNK], [r0, c0])
                mc = pl.slice(mean, [1, COL_CHUNK], [0, c0])
                sc = pl.slice(inv_std, [1, COL_CHUNK], [0, c0])
                r = pl.col_expand_add(row, pl.neg(mc))
                r = pl.col_expand_mul(r, sc)
                out = pl.assemble(out, r, [r0, c0])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Host helpers — im2col, reshape, pad (data rearrangement only, no arithmetic).
# ══════════════════════════════════════════════════════════════════════════════

def _to_2d(x):
    """5D [B,C,T,H,W] or 4D [B,C,H,W] → 2D [B*T*H*W, C]."""
    if x.dim() == 5:
        B, C, T, Hh, Ww = x.shape
        return x.permute(0, 2, 3, 4, 1).reshape(B * T * Hh * Ww, C).contiguous(), (B, T, Hh, Ww, C)
    B, C, Hh, Ww = x.shape
    return x.permute(0, 2, 3, 1).reshape(B * Hh * Ww, C).contiguous(), (B, Hh, Ww, C)


def _from_2d(x2, info):
    """Reverse of _to_2d."""
    if len(info) == 5:
        B, T, Hh, Ww, C = info
        return x2.reshape(B, T, Hh, Ww, C).permute(0, 4, 1, 2, 3)
    B, Hh, Ww, C = info
    return x2.reshape(B, Hh, Ww, C).permute(0, 3, 1, 2)


def _real_rows(info):
    if len(info) == 5:
        return info[0] * info[1] * info[2] * info[3]
    return info[0] * info[1] * info[2]


# ══════════════════════════════════════════════════════════════════════════════
# Kernel runners — compile cache + device dispatch.
# ══════════════════════════════════════════════════════════════════════════════

_cache = {}
_DEVICE_ID = 0


def set_device(device_id):
    global _DEVICE_ID
    _DEVICE_ID = device_id


def _compile(key, kernel, *dummies):
    from pypto.runtime import RunConfig
    if key not in _cache:
        _cache[key] = kernel.compile(*dummies, config=RunConfig(platform="a2a3", device_id=_DEVICE_ID))
    return _cache[key]


def _run(compiled_info, args):
    from pypto.runtime import execute_compiled
    execute_compiled(compiled_info.output_dir, args, platform="a2a3", device_id=_DEVICE_ID)


# ── Host wrappers (pad inputs → compile/run kernel → slice result) ──

def _npu_matmul(col, w):
    """matmul(col, w^T) → FP32."""
    M_real, K_real, N_real = col.shape[0], col.shape[1], w.shape[0]
    col_p = _pad_cols(col, K_CHUNK)
    col_p = _pad_rows(col_p, R_TILE)
    w_p = _pad_cols(w, K_CHUNK)
    w_N_pad = ((N_real + N_CHUNK - 1) // N_CHUNK) * N_CHUNK
    if w_N_pad > N_real:
        wp2 = torch.zeros(w_N_pad, w_p.shape[1], dtype=w_p.dtype)
        wp2[:N_real] = w_p
        w_p = wp2
    out = torch.zeros(col_p.shape[0], w_N_pad, dtype=torch.float32)
    key = ('mm_col_w', col_p.shape[0], w_p.shape[1], w_N_pad)
    compiled = _compile(key, mm_col_w, col_p, w_p, out)
    _run(compiled, [col_p, w_p, out])
    return out[:M_real, :N_real]


def _npu_softmax(x):
    """Row-wise softmax."""
    N = x.shape[1]
    xp = _pad_rows(x, R_TILE)
    out = torch.zeros(xp.shape[0], N, dtype=torch.bfloat16)
    key = ('softmax', xp.shape[0], N)
    compiled = _compile(key, softmax_k, xp, out)
    _run(compiled, [xp, out])
    return out[:x.shape[0], :]


def _npu_scale_fp32(x, scale_val):
    """Multiply FP32 tensor by scalar on NPU."""
    R, C = x.shape[0], x.shape[1]
    xp = _pad_cols(x, COL_CHUNK)
    xp = _pad_rows(xp, R_TILE)
    out = torch.zeros(xp.shape[0], xp.shape[1], dtype=torch.float32)
    scale_t = torch.tensor([[scale_val]], dtype=torch.float32)
    key = ('scale', xp.shape[0], xp.shape[1])
    compiled = _compile(key, scale_k, xp, scale_t, out)
    _run(compiled, [xp, scale_t, out])
    return out[:R, :C]


def _npu_scale_norm_fp32(x, mean, inv_std):
    """(x - mean) * inv_std on NPU."""
    R, C = x.shape[0], x.shape[1]
    xp = _pad_cols(x, COL_CHUNK)
    xp = _pad_rows(xp, R_TILE)
    mp = _pad_cols(mean.reshape(1, -1), COL_CHUNK)
    sp = _pad_cols(inv_std.reshape(1, -1), COL_CHUNK)
    out = torch.zeros(xp.shape[0], xp.shape[1], dtype=torch.float32)
    key = ('snorm', xp.shape[0], xp.shape[1])
    compiled = _compile(key, scale_norm_k, xp, mp, sp, out)
    _run(compiled, [xp, mp, sp, out])
    return out[:R, :C]


def _npu_rms_norm_fp32(x, weight_1d):
    """RMS_norm FP32 in → FP32 out."""
    R, C = x.shape[0], x.shape[1]
    xp = _pad_cols(x, RMS_COL)
    xp = _pad_rows(xp, RMS_TILE)
    wp = _pad_cols(weight_1d.reshape(1, -1), RMS_COL)
    out = torch.zeros(xp.shape[0], xp.shape[1], dtype=torch.float32)
    key = ('rms', xp.shape[0], xp.shape[1])
    compiled = _compile(key, rms_norm_k, xp, wp, out)
    _run(compiled, [xp, wp, out])
    return out[:R, :C]


def _npu_silu_fp32(x):
    """SiLU FP32 in → FP32 out."""
    R, C = x.shape[0], x.shape[1]
    xp = _pad_cols(x, SILU_COL)
    xp = _pad_rows(xp, SILU_TILE)
    out = torch.zeros(xp.shape[0], xp.shape[1], dtype=torch.float32)
    key = ('silu', xp.shape[0], xp.shape[1])
    compiled = _compile(key, silu_k, xp, out)
    _run(compiled, [xp, out])
    return out[:R, :C]


def _npu_residual_add_fp32(a, b):
    """a + b on NPU."""
    R, C = a.shape[0], a.shape[1]
    ap = _pad_cols(a, COL_CHUNK)
    ap = _pad_rows(ap, R_TILE)
    bp = _pad_cols(b, COL_CHUNK)
    bp = _pad_rows(bp, R_TILE)
    out = torch.zeros(ap.shape[0], ap.shape[1], dtype=torch.float32)
    key = ('resadd', ap.shape[0], ap.shape[1])
    compiled = _compile(key, residual_add_k, ap, bp, out)
    _run(compiled, [ap, bp, out])
    return out[:R, :C]


def _npu_bias_add_fp32(x, bias_1d):
    """x + bias on NPU."""
    R, C = x.shape[0], x.shape[1]
    xp = _pad_cols(x, COL_CHUNK)
    xp = _pad_rows(xp, R_TILE)
    bp = _pad_cols(bias_1d.reshape(1, -1), COL_CHUNK)
    out = torch.zeros(xp.shape[0], xp.shape[1], dtype=torch.float32)
    key = ('bias', xp.shape[0], xp.shape[1])
    compiled = _compile(key, bias_add_k, xp, bp, out)
    _run(compiled, [xp, bp, out])
    return out[:R, :C]


def _npu_matmul_with_bias(col, w, bias):
    """matmul(col, w^T) + bias on NPU."""
    r = _npu_matmul(col, w)
    if bias is not None:
        r = _npu_bias_add_fp32(r, bias.float())
    return r


# ══════════════════════════════════════════════════════════════════════════════
# Building blocks — FP32 internal path, BF16 at matmul inputs only.
# ALL arithmetic runs on NPU kernels.
# ══════════════════════════════════════════════════════════════════════════════

def causal_conv3d_npu_fp32(x, weight, bias=None, stride=1, padding=1):
    """Causal 3D conv.  im2col on host (data rearrangement), matmul on NPU."""
    out_channels = weight.shape[0]
    kd, kh, kw = weight.shape[2], weight.shape[3], weight.shape[4]
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    sd, sh, sw = stride
    pd, ph, pw = padding

    if sd == 1 and pd == 1:
        tpf, tpb = pd, pd
    else:
        tpf, tpb = pd * 2, 0

    xp = F.pad(x, [pw, pw, ph, ph, tpf, tpb])
    b, ic, depth, hp, wp = xp.shape
    od = (depth - kd) // sd + 1
    oh = (hp - kh) // sh + 1
    ow = (wp - kw) // sw + 1
    xu = xp.unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw)
    xu = xu.contiguous().view(b, ic, od, oh, ow, kd * kh * kw)
    xu = xu.permute(0, 2, 3, 4, 1, 5).contiguous()
    col = xu.reshape(b * od * oh * ow, ic * kd * kh * kw).bfloat16()

    w2d = weight.reshape(out_channels, -1).contiguous().bfloat16()
    bias_2d = bias.float().reshape(1, -1) if bias is not None else None
    result = _npu_matmul_with_bias(col, w2d, bias_2d)
    real_rows = od * oh * ow
    result_trimmed = result[:real_rows, :]
    return result_trimmed.reshape(1, od, oh, ow, out_channels).permute(0, 4, 1, 2, 3).contiguous()


def conv2d_npu_fp32(x, weight, bias=None, stride=1, padding=1):
    """2D conv.  im2col on host, matmul on NPU."""
    out_channels = weight.shape[0]
    kh, kw = weight.shape[2], weight.shape[3]
    sh = stride if not isinstance(stride, tuple) else stride[0]
    sw = stride if not isinstance(stride, tuple) else stride[1]
    if isinstance(padding, int):
        ph, pw = padding, padding
    else:
        ph, pw = padding[0], padding[1]

    xp = F.pad(x, [pw, pw, ph, ph])
    b, ic, hp, wp = xp.shape
    oh = (hp - kh) // sh + 1
    ow = (wp - kw) // sw + 1
    xu = xp.unfold(2, kh, sh).unfold(3, kw, sw)
    xu = xu.contiguous().view(b, ic, oh, ow, kh * kw)
    xu = xu.permute(0, 2, 3, 1, 4).contiguous()
    col = xu.reshape(b * oh * ow, ic * kh * kw).bfloat16()

    w2d = weight.reshape(out_channels, -1).contiguous().bfloat16()
    bias_2d = bias.float().reshape(1, -1) if bias is not None else None
    result = _npu_matmul_with_bias(col, w2d, bias_2d)
    real_rows = b * oh * ow
    return result[:real_rows, :].reshape(b, oh, ow, out_channels).permute(0, 3, 1, 2).contiguous()


def rms_norm_npu_fp32(x, weight_1d):
    """RMS_norm FP32 in → FP32 out on NPU."""
    x_2d, info = _to_2d(x.float())
    result_2d = _npu_rms_norm_fp32(x_2d, weight_1d.float())
    return _from_2d(result_2d[:_real_rows(info)], info)


def silu_npu_fp32(x):
    """SiLU FP32 in → FP32 out on NPU."""
    x_2d, info = _to_2d(x.float())
    result_2d = _npu_silu_fp32(x_2d)
    return _from_2d(result_2d[:_real_rows(info)], info)


def resblock_npu(x, weight_dict, in_dim, out_dim):
    """ResBlock: norm→SiLU→conv→norm→SiLU→conv + shortcut.  All arithmetic on NPU."""
    x = x.float()
    hidden = rms_norm_npu_fp32(x, weight_dict['norm1_w'])
    hidden = silu_npu_fp32(hidden)
    hidden = causal_conv3d_npu_fp32(hidden, weight_dict['conv1_w'], weight_dict.get('conv1_b'),
                                    stride=1, padding=1)
    hidden = rms_norm_npu_fp32(hidden, weight_dict['norm2_w'])
    hidden = silu_npu_fp32(hidden)
    hidden = causal_conv3d_npu_fp32(hidden, weight_dict['conv2_w'], weight_dict.get('conv2_b'),
                                    stride=1, padding=1)

    if in_dim != out_dim:
        shortcut = causal_conv3d_npu_fp32(x, weight_dict['shortcut_w'], weight_dict.get('shortcut_b'),
                                          stride=1, padding=0)
    else:
        shortcut = x

    hidden_2d, info = _to_2d(hidden)
    shortcut_2d, _ = _to_2d(shortcut)
    result_2d = _npu_residual_add_fp32(hidden_2d[:_real_rows(info)], shortcut_2d[:_real_rows(info)])
    return _from_2d(result_2d, info)


def attention_block_npu(x, weight_dict, dim):
    """AttentionBlock: norm→QKV(1x1conv)→SDPA→proj + residual.  All arithmetic on NPU."""
    x = x.float()
    identity = x
    batch, channels, temporal, height, width = x.size()
    batch_temporal = batch * temporal
    spatial_size = height * width

    x_2d = x.permute(0, 2, 1, 3, 4).reshape(batch_temporal, channels, height, width).contiguous()
    x_2d_norm = rms_norm_npu_fp32(x_2d, weight_dict['norm_w'])

    qkv = conv2d_npu_fp32(x_2d_norm, weight_dict['to_qkv_w'], weight_dict.get('to_qkv_b'),
                          stride=1, padding=0)
    qkv_r = qkv.reshape(batch_temporal, channels * 3, spatial_size).permute(0, 2, 1).contiguous()
    q_pf = qkv_r[:, :, :channels].contiguous()
    k_pf = qkv_r[:, :, channels:2 * channels].contiguous()
    v_pf = qkv_r[:, :, 2 * channels:].contiguous()

    scale_factor = 1.0 / (channels ** 0.5)
    attn_out_flat = torch.zeros(batch_temporal * spatial_size, channels, dtype=torch.float32)

    for i in range(batch_temporal):
        q_i = q_pf[i]
        k_i = k_pf[i]
        v_i = v_pf[i]
        scores = _npu_matmul(q_i.bfloat16(), k_i.bfloat16())
        scores = _npu_scale_fp32(scores, scale_factor)
        softmax_out = _npu_softmax(scores.bfloat16())
        head_out = _npu_matmul(softmax_out, v_i.bfloat16().transpose(0, 1))
        attn_out_flat[i * spatial_size:(i + 1) * spatial_size, :] = head_out[:spatial_size, :channels]

    proj_w2d = weight_dict['proj_w'].reshape(channels, channels).bfloat16()
    proj_result = _npu_matmul(attn_out_flat.bfloat16(), proj_w2d)
    if weight_dict.get('proj_b') is not None:
        proj_result = _npu_bias_add_fp32(proj_result, weight_dict['proj_b'].float())

    x_2d_out = proj_result[:batch_temporal * spatial_size, :channels].reshape(
        batch_temporal, spatial_size, channels).permute(0, 2, 1)
    x_2d_out = x_2d_out.reshape(batch_temporal, channels, height, width).contiguous()
    x_5d_out = x_2d_out.reshape(batch, temporal, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()

    x_5d_2d, info = _to_2d(x_5d_out)
    identity_2d, _ = _to_2d(identity)
    result_2d = _npu_residual_add_fp32(x_5d_2d[:_real_rows(info)], identity_2d[:_real_rows(info)])
    return _from_2d(result_2d, info)


def downsample3d_npu(x, weight_dict, dim):
    """Downsample3d: spatial Conv2d(s=2) + temporal CausalConv3d(s=(2,1,1))."""
    x = x.float()
    batch, channels, temporal, height, width = x.size()

    x_2d = x.permute(0, 2, 1, 3, 4).reshape(batch * temporal, channels, height, width).contiguous()
    x_2d = F.pad(x_2d, (0, 1, 0, 1))
    x_2d_out = conv2d_npu_fp32(x_2d, weight_dict['conv_w'], weight_dict.get('conv_b'),
                               stride=2, padding=0)
    _, _, hd, wd = x_2d_out.shape
    x_5d = x_2d_out.reshape(batch, temporal, channels, hd, wd).permute(0, 2, 1, 3, 4).contiguous()

    x_5d = causal_conv3d_npu_fp32(x_5d, weight_dict['time_conv_w'], weight_dict.get('time_conv_b'),
                                  stride=(2, 1, 1), padding=(1, 0, 0))
    return x_5d


# ══════════════════════════════════════════════════════════════════════════════
# Full VAE encoder — FP32 internal path, all arithmetic on NPU.
# ══════════════════════════════════════════════════════════════════════════════

def vae_encode_npu(video, w):
    """Full VAE encoder: video [1,3,T,64,64] → mu [1,16,1,8,8].

    All arithmetic runs on NPU kernels.  im2col (F.pad + .unfold), reshape,
    permute, and dtype cast are host-side data rearrangement only.
    """
    hidden = video.float()
    hidden = causal_conv3d_npu_fp32(hidden, w["vae_enc_in_w"], w["vae_enc_in_b"],
                                    stride=1, padding=1)

    channels = VAE_CH
    for stage_idx in range(4):
        r1w = {k: w[f"vae_enc_s{stage_idx}_r1_{k}"]
               for k in ['norm1_w', 'conv1_w', 'conv1_b',
                         'norm2_w', 'conv2_w', 'conv2_b']}
        if channels[stage_idx] != channels[stage_idx + 1]:
            r1w['shortcut_w'] = w[f"vae_enc_s{stage_idx}_r1_shortcut_w"]
            r1w['shortcut_b'] = w.get(f"vae_enc_s{stage_idx}_r1_shortcut_b")
        hidden = resblock_npu(hidden, r1w, channels[stage_idx], channels[stage_idx + 1])

        r2w = {k: w[f"vae_enc_s{stage_idx}_r2_{k}"]
               for k in ['norm1_w', 'conv1_w', 'conv1_b',
                         'norm2_w', 'conv2_w', 'conv2_b']}
        hidden = resblock_npu(hidden, r2w, channels[stage_idx + 1], channels[stage_idx + 1])

        if stage_idx < 3:
            rsw = {k: w[f"vae_enc_s{stage_idx}_resample_{k}"]
                   for k in ['conv_w', 'conv_b', 'time_conv_w', 'time_conv_b']}
            hidden = downsample3d_npu(hidden, rsw, channels[stage_idx + 1])

    mr1w = {k: w[f"vae_enc_mid_r1_{k}"]
            for k in ['norm1_w', 'conv1_w', 'conv1_b',
                      'norm2_w', 'conv2_w', 'conv2_b']}
    hidden = resblock_npu(hidden, mr1w, 1536, 1536)

    maw = {k: w[f"vae_enc_mid_attn_{k}"]
           for k in ['norm_w', 'to_qkv_w', 'to_qkv_b', 'proj_w', 'proj_b']}
    hidden = attention_block_npu(hidden, maw, 1536)

    mr2w = {k: w[f"vae_enc_mid_r2_{k}"]
            for k in ['norm1_w', 'conv1_w', 'conv1_b',
                      'norm2_w', 'conv2_w', 'conv2_b']}
    hidden = resblock_npu(hidden, mr2w, 1536, 1536)

    hidden = causal_conv3d_npu_fp32(hidden, w["vae_enc_out_w"], w["vae_enc_out_b"],
                                    stride=1, padding=1)
    mu, _ = hidden.chunk(2, dim=1)
    mu_2d, info = _to_2d(mu)
    mu_2d = mu_2d[:_real_rows(info)]
    result_2d = _npu_scale_norm_fp32(mu_2d, w["vae_scale_mean"].float().flatten(),
                                     w["vae_scale_inv_std"].float().flatten())
    return _from_2d(result_2d, info)


# ══════════════════════════════════════════════════════════════════════════════
# Weight generation — matches golden reference make_weights (encoder part).
# ══════════════════════════════════════════════════════════════════════════════

def make_vae_enc_weights(seed=42):
    gen = torch.Generator().manual_seed(seed)
    weights = {}
    weights["vae_enc_in_w"] = torch.randn(96, 3, 3, 3, 3, generator=gen) * 0.02
    weights["vae_enc_in_b"] = torch.zeros(96)
    channels = VAE_CH
    for stage in range(4):
        prefix = f"vae_enc_s{stage}_r1"
        weights[f"{prefix}_norm1_w"] = torch.ones(channels[stage])
        weights[f"{prefix}_conv1_w"] = torch.randn(channels[stage+1], channels[stage], 3, 3, 3, generator=gen) * 0.02
        weights[f"{prefix}_conv1_b"] = torch.zeros(channels[stage+1])
        weights[f"{prefix}_norm2_w"] = torch.ones(channels[stage+1])
        weights[f"{prefix}_conv2_w"] = torch.randn(channels[stage+1], channels[stage+1], 3, 3, 3, generator=gen) * 0.02
        weights[f"{prefix}_conv2_b"] = torch.zeros(channels[stage+1])
        if channels[stage] != channels[stage+1]:
            weights[f"{prefix}_shortcut_w"] = torch.randn(channels[stage+1], channels[stage], 1, 1, 1, generator=gen) * 0.02
            weights[f"{prefix}_shortcut_b"] = torch.zeros(channels[stage+1])
        prefix = f"vae_enc_s{stage}_r2"
        weights[f"{prefix}_norm1_w"] = torch.ones(channels[stage+1])
        weights[f"{prefix}_conv1_w"] = torch.randn(channels[stage+1], channels[stage+1], 3, 3, 3, generator=gen) * 0.02
        weights[f"{prefix}_conv1_b"] = torch.zeros(channels[stage+1])
        weights[f"{prefix}_norm2_w"] = torch.ones(channels[stage+1])
        weights[f"{prefix}_conv2_w"] = torch.randn(channels[stage+1], channels[stage+1], 3, 3, 3, generator=gen) * 0.02
        weights[f"{prefix}_conv2_b"] = torch.zeros(channels[stage+1])
        if stage < 3:
            prefix = f"vae_enc_s{stage}_resample"
            weights[f"{prefix}_conv_w"] = torch.randn(channels[stage+1], channels[stage+1], 3, 3, generator=gen) * 0.02
            weights[f"{prefix}_conv_b"] = torch.zeros(channels[stage+1])
            weights[f"{prefix}_time_conv_w"] = torch.randn(channels[stage+1], channels[stage+1], 3, 1, 1, generator=gen) * 0.02
            weights[f"{prefix}_time_conv_b"] = torch.zeros(channels[stage+1])

    prefix = "vae_enc_mid_r1"
    for key in ['norm1_w', 'norm2_w']:
        weights[f"{prefix}_{key}"] = torch.ones(1536)
    for key in ['conv1_w', 'conv2_w']:
        weights[f"{prefix}_{key}"] = torch.randn(1536, 1536, 3, 3, 3, generator=gen) * 0.02
    for key in ['conv1_b', 'conv2_b']:
        weights[f"{prefix}_{key}"] = torch.zeros(1536)
    prefix = "vae_enc_mid_attn"
    weights[f"{prefix}_norm_w"] = torch.ones(1536)
    weights[f"{prefix}_to_qkv_w"] = torch.randn(1536*3, 1536, 1, 1, generator=gen) * 0.02
    weights[f"{prefix}_to_qkv_b"] = torch.zeros(1536*3)
    weights[f"{prefix}_proj_w"] = torch.randn(1536, 1536, 1, 1, generator=gen) * 0.02
    weights[f"{prefix}_proj_b"] = torch.zeros(1536)
    prefix = "vae_enc_mid_r2"
    for key in ['norm1_w', 'norm2_w']:
        weights[f"{prefix}_{key}"] = torch.ones(1536)
    for key in ['conv1_w', 'conv2_w']:
        weights[f"{prefix}_{key}"] = torch.randn(1536, 1536, 3, 3, 3, generator=gen) * 0.02
    for key in ['conv1_b', 'conv2_b']:
        weights[f"{prefix}_{key}"] = torch.zeros(1536)

    weights["vae_enc_out_w"] = torch.randn(VAE_OUT_CH, 1536, 3, 3, 3, generator=gen) * 0.02
    weights["vae_enc_out_b"] = torch.zeros(VAE_OUT_CH)

    vae_mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921])
    vae_std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160])
    weights["vae_scale_mean"] = vae_mean.view(VAE_Z_DIM, 1, 1, 1)
    weights["vae_scale_inv_std"] = (1.0 / vae_std).view(VAE_Z_DIM, 1, 1, 1)
    return weights


# ══════════════════════════════════════════════════════════════════════════════
# Golden reference — uses the EXACT vae_encode from test_golden_fun_control_full.py
# ══════════════════════════════════════════════════════════════════════════════

def _golden_vae_encode(video, w):
    """Golden reference: imports and runs the SAME code as the test file."""
    _GOLDEN_DIR = '/data/x00952168/pypto3.0/cann-recipes-embodied-ai/world_model/agibot-arm-world-model/infer_with_torch'
    if _GOLDEN_DIR not in sys.path:
        sys.path.insert(0, _GOLDEN_DIR)
    from test_golden_fun_control_full import vae_encode as ref_vae_encode
    return ref_vae_encode(video, w)


# ══════════════════════════════════════════════════════════════════════════════
# Test — uses golden.runner._Stage + golden.validation.validate_golden,
# same harness primitives as run_jit in t5_encoder.py.
# (Multi-kernel architecture: individual @pl.jit kernels are JIT-compiled
# on first use during the runtime stage; no standalone compile pass.)
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    from golden.runner import _Stage
    from golden.validation import validate_golden, ratio_allclose

    parser = argparse.ArgumentParser(description="VAE Encoder — pypto3.0")
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="NPU device id (soldered chip index)")
    args = parser.parse_args()

    set_device(args.device)
    t_total = time.time()

    with _Stage("generate inputs"):
        torch.manual_seed(42)
        w = make_vae_enc_weights(seed=42)
        video = (torch.randn(1, 3, T_IN, H, W) * 0.1).bfloat16()

    with _Stage("compute golden"):
        ref = _golden_vae_encode(video.float(), w)

    with _Stage("runtime"):
        npu = vae_encode_npu(video, w)

    with _Stage("validate"):
        validate_golden(
            outputs={"out": npu},
            golden={"out": ref},
            rtol=0.1,
            atol=0.1,
            compare_fn={"out": ratio_allclose(atol=0.1, rtol=0.1, max_error_ratio=0.02)},
        )

    total = time.time() - t_total
    print(f"[RUN] PASS ({total:.2f}s)", flush=True)
