# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""VAE decoder for Fun-Control 1.3B — pypto3.0 implementation.

Implements the VAE decoder sub-network from the pipeline in
``infer_fun_control_1_3b_text.py`` (DiffSynth WanVideoPipeline VAE).
All arithmetic runs in pypto kernels.  im2col / reshape / permute /
nearest-neighbour upsample are host-side data rearrangement only.

The golden reference for precision comparison is
``test_golden_fun_control_full.py::vae_decode``, a pure-PyTorch
reimplementation of the SAME VAE logic used by the DiffSynth pipeline.
The two are logically identical because ``test_golden_fun_control_full.py``
replicates every VAE computation from the DiffSynth source that
``infer_fun_control_1_3b_text.py`` invokes via ``WanVideoPipeline``.

Reuses all building blocks from ``vae_encoder`` (ResBlock, AttentionBlock,
CausalConv3d, RMS norm, SiLU, softmax, matmul, etc.).  Two new components:
  - ``Upsample3d``: temporal CausalConv3d + nearest x2 + spatial Conv2d
  - ``clamp_k``:    NPU kernel for torch.clamp(-1, 1)

Legitimate differences from the golden reference:

  1. **Feature caching not implemented.**  The golden reference maintains
     ``feat_cache`` / ``feat_idx`` lists for chunked autoregressive
     inference where successive chunks share temporal context.  The pypto
     implementation targets single-chunk inference (CHUNK_SIZE=5 golden
     case); multi-chunk caching is a host-side orchestration concern
     handled at the pipeline integration level, not inside the NPU kernel.
  2. **LAT_F = 1** (not config.LAT_F = 2).  For CHUNK_SIZE=5 the encoder's
     natural temporal output is 1 frame (3× downsample: 5→3→2→1).  The
     golden test uses LAT_F=2 as a DiT-compatible target and interpolates.
     The pypto encoder/decoder work at the encoder's natural resolution
     (T=1); any DiT-required interpolation is pipeline-level remapping.
  3. **Attention processes per-temporal-slice** (loop over ``batch_temporal``).
     The golden uses ``F.scaled_dot_product_attention`` which handles all
     heads/slices at once.  The pypto version processes each temporal slice
     independently because spatial dimensions for the golden case are small
     (8×8 → batch_temporal ≤ 2).

Architecture (cache=None, CHUNK_SIZE=5)::

    Input [1, 16, 1, 8, 8]
    → CausalConv3d(16→1536, k=3, s=1)
    → Middle: ResBlock(1536)+Attention(1536)+ResBlock(1536)
    → Stage 0: ResBlock(1536→768)+ResBlock(768)+Upsample3d(t_up=T)  → [1,768,2,16,16]
    → Stage 1: ResBlock(768→384)+ResBlock(384)+Upsample3d(t_up=T)   → [1,384,4,32,32]
    → Stage 2: ResBlock(384→192)+ResBlock(192)+Upsample3d(t_up=F)   → [1,192,4,64,64]
    → Stage 3: ResBlock(192→96)+ResBlock(96)
    → CausalConv3d(96→3, k=3, s=1)                                   → [1,3,4,64,64]
    → clamp(-1, 1)

Precision comparison uses ``ratio_allclose`` with tolerance atol=0.1 /
rtol=0.1 / max_error_ratio=0.02 — same pattern as decode_layer.py but with
looser tolerances justified by the deeper VAE network (25+ RMSNorm ops,
each contributing ~0.012 hardware accuracy loss on Ascend vector unit;
see memory.md Phase 3b for per-op precision analysis).

Usage::

    python vae_decoder.py -p a2a3 -d 0
"""

import argparse
import sys
import time

import pypto.language as pl
import torch
import torch.nn.functional as F

import vae_encoder as ve
from config import VAE_Z_DIM, H, W, VAE_DEC_CH

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

# LAT_F = 1 is the encoder's natural temporal output for CHUNK_SIZE=5
# (3× CausalConv3d stride=(2,1,1) downsample: 5→3→2→1).
# config.LAT_F = 2 is the DiT-compatible target; the golden pipeline
# interpolates to it.  The pypto encoder/decoder pair uses the natural
# resolution; DiT interpolation is a pipeline-level remap.
LAT_F = 1

# ── Tiling constants for the clamp kernel (reuse encoder's proven tile sizes) ──

_PAD_COL = ve.COL_CHUNK   # 192 — column tile for elementwise clamp (same as encoder)
_R_TILE = ve.R_TILE        # 16 — row tile for pl.parallel dispatch (same as encoder)

# ── Geometry assertions (same pattern as decode_layer.py) ──
assert len(VAE_DEC_CH) == 5, f"VAE_DEC_CH must have 5 entries (middle + 4 stages), got {len(VAE_DEC_CH)}"
assert VAE_DEC_CH[0] == 1536, f"VAE_DEC_CH[0] must be 1536 (middle bottleneck), got {VAE_DEC_CH[0]}"
assert VAE_DEC_CH[4] == 96, f"VAE_DEC_CH[4] must be 96 (last stage out channels), got {VAE_DEC_CH[4]}"
assert LAT_F == 1, f"LAT_F must be 1 (encoder's natural T output for CHUNK_SIZE=5), got {LAT_F}"


# ══════════════════════════════════════════════════════════════════════════════
# NPU Kernel — clamp(-1, 1)
# ══════════════════════════════════════════════════════════════════════════════

@pl.jit
def clamp_k(
    x: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    """clamp(x, -1, 1) on 2D tensor.  Pre-padded to R_TILE rows and COL_CHUNK cols."""
    R = x.shape[0]
    COLS = x.shape[1]
    for r0 in pl.parallel(0, R, _R_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="clamp"):
            for c0 in pl.range(0, COLS, _PAD_COL):
                row = pl.slice(x, [_R_TILE, _PAD_COL], [r0, c0])
                clipped = pl.minimum(pl.maximum(row, -1.0), 1.0)
                out = pl.assemble(out, pl.cast(clipped, target_type=pl.BF16), [r0, c0])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Host runner — clamp on NPU
# ══════════════════════════════════════════════════════════════════════════════

def _npu_clamp(x):
    """clamp(x, -1, 1) → BF16.  x: 2D FP32 tensor."""
    R, C = x.shape[0], x.shape[1]
    xp = ve._pad_cols(x, _PAD_COL)
    xp = ve._pad_rows(xp, _R_TILE)
    out = torch.zeros(xp.shape[0], xp.shape[1], dtype=torch.bfloat16)

    key = ('clamp', xp.shape[0], xp.shape[1])
    compiled = ve._compile(key, clamp_k, xp, out)
    ve._run(compiled, [xp, out])
    return out[:R, :C]


# ══════════════════════════════════════════════════════════════════════════════
# Building block — Upsample3d (host orchestration, NPU arithmetic)
# ══════════════════════════════════════════════════════════════════════════════

def upsample3d_npu(x, wd, temporal_up):
    """Upsample3d: temporal CausalConv3d + nearest x2 + spatial Conv2d.

    All arithmetic (conv matmul) on NPU.  Nearest-neighbour interpolation
    and reshape/permute on host (data rearrangement, no arithmetic).
    """
    b, c, t, h, w = x.size()

    # ── Temporal upsample ──
    if temporal_up:
        x = ve.causal_conv3d_npu_fp32(x, wd['time_conv_w'], wd.get('time_conv_b'),
                                       stride=1, padding=(1, 0, 0))
        x = x.reshape(b, 2, c, t, h, w)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]

    # ── Spatial upsample ──
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w).contiguous()
    # Nearest-neighbour x2 — data rearrangement (sampling, no arithmetic)
    x = F.interpolate(x, scale_factor=2.0, mode='nearest')
    x = ve.conv2d_npu_fp32(x, wd['conv_w'], wd.get('conv_b'),
                           stride=1, padding=1)
    x = x.reshape(b, t, -1, h * 2, w * 2).permute(0, 2, 1, 3, 4).contiguous()
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Full VAE decoder — FP32 internal path, all arithmetic on NPU.
# ══════════════════════════════════════════════════════════════════════════════

def vae_decode_npu(latents, w):
    """Full VAE decoder: latents [1,16,1,8,8] → video [1,3,4,64,64].

    All arithmetic runs on NPU kernels.  im2col, nearest-neighbour upsample,
    reshape, and permute are host-side data rearrangement only.
    """
    h = latents.float()

    # Input conv: 16 → 1536
    h = ve.causal_conv3d_npu_fp32(h, w["vae_dec_in_w"], w["vae_dec_in_b"],
                                   stride=1, padding=1)

    # Middle: ResBlock(1536) + AttentionBlock(1536) + ResBlock(1536)
    mr1w = {k: w[f"vae_dec_mid_r1_{k}"]
            for k in ['norm1_w', 'conv1_w', 'conv1_b',
                      'norm2_w', 'conv2_w', 'conv2_b']}
    h = ve.resblock_npu(h, mr1w, 1536, 1536)

    maw = {k: w[f"vae_dec_mid_attn_{k}"]
           for k in ['norm_w', 'to_qkv_w', 'to_qkv_b', 'proj_w', 'proj_b']}
    h = ve.attention_block_npu(h, maw, 1536)

    mr2w = {k: w[f"vae_dec_mid_r2_{k}"]
            for k in ['norm1_w', 'conv1_w', 'conv1_b',
                      'norm2_w', 'conv2_w', 'conv2_b']}
    h = ve.resblock_npu(h, mr2w, 1536, 1536)

    # 4 stages: ResBlocks + Upsample
    vae_dec_ch = VAE_DEC_CH
    for s in range(4):
        # ResBlock 1 (may change channels)
        r1w = {k: w[f"vae_dec_s{s}_r1_{k}"]
               for k in ['norm1_w', 'conv1_w', 'conv1_b',
                         'norm2_w', 'conv2_w', 'conv2_b']}
        if vae_dec_ch[s] != vae_dec_ch[s + 1]:
            r1w['shortcut_w'] = w[f"vae_dec_s{s}_r1_shortcut_w"]
            r1w['shortcut_b'] = w.get(f"vae_dec_s{s}_r1_shortcut_b")
        h = ve.resblock_npu(h, r1w, vae_dec_ch[s], vae_dec_ch[s + 1])

        # ResBlock 2 (same channels)
        r2w = {k: w[f"vae_dec_s{s}_r2_{k}"]
               for k in ['norm1_w', 'conv1_w', 'conv1_b',
                         'norm2_w', 'conv2_w', 'conv2_b']}
        h = ve.resblock_npu(h, r2w, vae_dec_ch[s + 1], vae_dec_ch[s + 1])

        # Upsample (stages 0,1,2): temporal_up for s < 2
        if s < 3:
            rsw = {k: w[f"vae_dec_s{s}_resample_{k}"]
                   for k in ['conv_w', 'conv_b',
                             'time_conv_w', 'time_conv_b']}
            h = upsample3d_npu(h, rsw, temporal_up=(s < 2))

    # Output conv: 96 → 3
    h = ve.causal_conv3d_npu_fp32(h, w["vae_dec_out_w"], w["vae_dec_out_b"],
                                   stride=1, padding=1)

    # clamp(-1, 1)
    h2, info = ve._to_2d(h)
    h2 = h2[:ve._real_rows(info)]
    r2 = _npu_clamp(h2)
    return ve._from_2d(r2, info)


# ══════════════════════════════════════════════════════════════════════════════
# Weight generation — matches golden reference make_weights (decoder part).
# ══════════════════════════════════════════════════════════════════════════════

def make_vae_dec_weights(seed=42):
    g = torch.Generator().manual_seed(seed)
    w = {}
    w["vae_dec_in_w"] = torch.randn(1536, VAE_Z_DIM, 3, 3, 3, generator=g) * 0.02
    w["vae_dec_in_b"] = torch.zeros(1536)

    p = "vae_dec_mid_r1"
    w[f"{p}_norm1_w"] = torch.ones(1536)
    w[f"{p}_conv1_w"] = torch.randn(1536, 1536, 3, 3, 3, generator=g) * 0.02
    w[f"{p}_conv1_b"] = torch.zeros(1536)
    w[f"{p}_norm2_w"] = torch.ones(1536)
    w[f"{p}_conv2_w"] = torch.randn(1536, 1536, 3, 3, 3, generator=g) * 0.02
    w[f"{p}_conv2_b"] = torch.zeros(1536)

    p = "vae_dec_mid_attn"
    w[f"{p}_norm_w"] = torch.ones(1536)
    w[f"{p}_to_qkv_w"] = torch.randn(1536 * 3, 1536, 1, 1, generator=g) * 0.02
    w[f"{p}_to_qkv_b"] = torch.zeros(1536 * 3)
    w[f"{p}_proj_w"] = torch.randn(1536, 1536, 1, 1, generator=g) * 0.02
    w[f"{p}_proj_b"] = torch.zeros(1536)

    p = "vae_dec_mid_r2"
    w[f"{p}_norm1_w"] = torch.ones(1536)
    w[f"{p}_conv1_w"] = torch.randn(1536, 1536, 3, 3, 3, generator=g) * 0.02
    w[f"{p}_conv1_b"] = torch.zeros(1536)
    w[f"{p}_norm2_w"] = torch.ones(1536)
    w[f"{p}_conv2_w"] = torch.randn(1536, 1536, 3, 3, 3, generator=g) * 0.02
    w[f"{p}_conv2_b"] = torch.zeros(1536)

    vae_dec_ch = VAE_DEC_CH
    for s in range(4):
        p = f"vae_dec_s{s}_r1"
        w[f"{p}_norm1_w"] = torch.ones(vae_dec_ch[s])
        w[f"{p}_conv1_w"] = torch.randn(vae_dec_ch[s + 1], vae_dec_ch[s], 3, 3, 3, generator=g) * 0.02
        w[f"{p}_conv1_b"] = torch.zeros(vae_dec_ch[s + 1])
        w[f"{p}_norm2_w"] = torch.ones(vae_dec_ch[s + 1])
        w[f"{p}_conv2_w"] = torch.randn(vae_dec_ch[s + 1], vae_dec_ch[s + 1], 3, 3, 3, generator=g) * 0.02
        w[f"{p}_conv2_b"] = torch.zeros(vae_dec_ch[s + 1])
        if vae_dec_ch[s] != vae_dec_ch[s + 1]:
            w[f"{p}_shortcut_w"] = torch.randn(vae_dec_ch[s + 1], vae_dec_ch[s], 1, 1, 1, generator=g) * 0.02
            w[f"{p}_shortcut_b"] = torch.zeros(vae_dec_ch[s + 1])

        p = f"vae_dec_s{s}_r2"
        w[f"{p}_norm1_w"] = torch.ones(vae_dec_ch[s + 1])
        w[f"{p}_conv1_w"] = torch.randn(vae_dec_ch[s + 1], vae_dec_ch[s + 1], 3, 3, 3, generator=g) * 0.02
        w[f"{p}_conv1_b"] = torch.zeros(vae_dec_ch[s + 1])
        w[f"{p}_norm2_w"] = torch.ones(vae_dec_ch[s + 1])
        w[f"{p}_conv2_w"] = torch.randn(vae_dec_ch[s + 1], vae_dec_ch[s + 1], 3, 3, 3, generator=g) * 0.02
        w[f"{p}_conv2_b"] = torch.zeros(vae_dec_ch[s + 1])

        if s < 3:
            p = f"vae_dec_s{s}_resample"
            w[f"{p}_conv_w"] = torch.randn(vae_dec_ch[s + 1], vae_dec_ch[s + 1], 3, 3, generator=g) * 0.02
            w[f"{p}_conv_b"] = torch.zeros(vae_dec_ch[s + 1])
            w[f"{p}_time_conv_w"] = torch.randn(vae_dec_ch[s + 1] * 2, vae_dec_ch[s + 1], 3, 1, 1, generator=g) * 0.02
            w[f"{p}_time_conv_b"] = torch.zeros(vae_dec_ch[s + 1] * 2)

    w["vae_dec_out_w"] = torch.randn(3, 96, 3, 3, 3, generator=g) * 0.02
    w["vae_dec_out_b"] = torch.zeros(3)
    return w


# ══════════════════════════════════════════════════════════════════════════════
# Golden reference — uses the EXACT vae_decode from test_golden_fun_control_full.py
# ══════════════════════════════════════════════════════════════════════════════

def _golden_vae_decode(latents, w):
    """Golden reference: imports and runs the SAME code as the test file."""
    _GOLDEN_DIR = '/data/x00952168/pypto3.0/cann-recipes-embodied-ai/world_model/agibot-arm-world-model/infer_with_torch'
    if _GOLDEN_DIR not in sys.path:
        sys.path.insert(0, _GOLDEN_DIR)
    from test_golden_fun_control_full import vae_decode as ref_vae_decode
    return ref_vae_decode(latents, w)


# ══════════════════════════════════════════════════════════════════════════════
# Test — uses golden.runner._Stage + golden.validation.validate_golden,
# same harness primitives as run_jit in t5_encoder.py.
# (Multi-kernel architecture: individual @pl.jit kernels are JIT-compiled
# on first use during the runtime stage; no standalone compile pass.)
# ══════════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    from golden.runner import _Stage
    from golden.validation import validate_golden, ratio_allclose

    parser = argparse.ArgumentParser(description="VAE Decoder — pypto3.0")
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0,
                        help="NPU device id (soldered chip index)")
    args = parser.parse_args()

    ve.set_device(args.device)
    t_total = time.time()

    with _Stage("generate inputs"):
        torch.manual_seed(42)
        w = make_vae_dec_weights(seed=42)
        latents = (torch.randn(1, VAE_Z_DIM, LAT_F, H // 8, W // 8) * 0.1).bfloat16()

    with _Stage("compute golden"):
        ref = _golden_vae_decode(latents.float(), w)

    with _Stage("runtime"):
        npu = vae_decode_npu(latents, w)

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
