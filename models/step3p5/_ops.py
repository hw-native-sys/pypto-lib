# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 shared kernel helpers (Phase 3 dedup target).

Hoists the four inline helpers that were duplicated across the Phase 2
drafts ``single_layer_decode_full_draft.py`` and
``single_layer_decode_swa_draft.py`` into a single import surface so
``attention_full.py`` / ``attention_swa.py`` consume identical building
blocks.

Function names match the brief in MIGRATION_PLAN.md Phase 3 verbatim:

  - ``zero_centered_rmsnorm_apply(normed_fp32, gamma_slice)``
       Fold the +1.0 zero-centring shift into the per-channel gamma
       broadcast multiply: ``normed_fp32 * (gamma_slice + 1.0)``.

  - ``per_head_qk_norm(qk_per_head, gamma_128)``
       Per-head RMSNorm with a single [HEAD_DIM=128] zero-centred gamma
       broadcast across all heads of the bundle. Used for both q_norm
       (Q_PER_KV heads per KV head bundle) and k_norm (1 head per KV
       head bundle).

  - ``partial_rope_rotate(lo, hi, cos_lo, cos_hi, sin_lo, sin_hi)``
       One (lo, hi) llama-style RoPE rotation step. The caller picks the
       half-dim by slicing the cos/sin tables appropriately, so the same
       helper covers both the partial-0.5 (rotary_half=32 for full
       attention) and partial-1.0 (rotary_half=64 for SWA) layer flavours.

  - ``head_wise_gate_apply(attn_head_slice_bf16, gate_logit_col_fp32)``
       Step3p5 head-wise attention gate: sigmoid(gate logits) broadcast
       across HEAD_DIM lanes and multiplied into a single head slice of
       the fa_fused output.

Host-side torch helpers (used by both the kernel input-gen path and the
torch goldens):

  - ``build_llama3_yarn_rope_tables(seq_len, head_rotary_dim,
        rope_theta, factor=2.0, low=1.0, high=32.0,
        orig_max=131072)``
       FP32 cos/sin tables sized ``[seq_len, head_rotary_dim]`` with the
       llama3 yarn freq-spectrum re-scaling applied.

  - ``build_plain_rope_tables(seq_len, head_rotary_dim, rope_theta)``
       FP32 cos/sin tables sized ``[seq_len, head_rotary_dim]`` with no
       scaling.

Cos/sin sizing convention: the tables are ``[max_seq, rotary_dim]`` --
the rotary slice width, NOT the full HEAD_DIM. This matches the layer-0
draft's bandwidth-efficient layout and lets the SWA path naturally
consume ``rotary_dim == HEAD_DIM`` while the full path consumes the
narrower ``rotary_dim == HEAD_DIM * 0.5``.

g_proj weight layout convention: the kernel reads g_proj as
``[HIDDEN, NUM_HEADS]`` after a host-side transpose from the checkpoint's
canonical ``[NUM_HEADS, HIDDEN]`` orientation. Do not change orientation
in the kernel.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl

from .config import EPS, HEAD_DIM_INV


# =============================================================================
# Inline kernel helpers (run inside an InCore region).
# =============================================================================


@pl.jit.inline
def zero_centered_rmsnorm_apply(
    normed_fp32,    # pl.Tensor[[rows, k], pl.FP32]
    gamma_slice,    # pl.Tensor[[1, k], pl.FP32]  (stored, zero-centred)
):
    """Fold +1.0 into the stored zero-centred gamma and broadcast-multiply.

    Step3p5 stores RMSNorm gammas centred at 0; the effective per-channel
    scale is ``stored_gamma + 1.0``. We add the +1 at runtime so the host
    loader can pass the raw checkpoint weights unchanged.

    Args:
        normed_fp32: FP32 tile ``[rows, k]`` -- typically
            ``row_expand_mul(x_fp32, inv_rms)``, i.e. the unscaled normed
            activations before the per-channel gamma broadcast.
        gamma_slice: FP32 ``[1, k]`` slice of the stored gamma.

    Returns:
        FP32 tile ``[rows, k]`` equal to ``normed_fp32 * (gamma + 1.0)``.
    """
    gamma_eff = pl.adds(gamma_slice, 1.0)
    return pl.col_expand_mul(normed_fp32, gamma_eff)


@pl.jit.inline
def per_head_qk_norm(
    qk_per_head,    # pl.Tensor[[rows_per_head, HEAD_DIM], pl.FP32]
    gamma_128,      # pl.Tensor[[1, HEAD_DIM], pl.FP32]  (stored, zero-centred)
):
    """Per-head zero-centred RMSNorm with a single shared [HEAD_DIM] gamma.

    Step3p5 applies RMSNorm per head along the HEAD_DIM=128 axis BEFORE
    RoPE. The same [HEAD_DIM] gamma vector is broadcast across all heads
    in the bundle. Callers flatten their (batch, heads, HEAD_DIM) tile to
    ``[rows_per_head, HEAD_DIM]`` then reshape back after this helper.

    Returns:
        FP32 tile ``[rows_per_head, HEAD_DIM]`` of normed activations.
    """
    sq = pl.row_sum(pl.mul(qk_per_head, qk_per_head))
    inv = pl.rsqrt(pl.add(pl.mul(sq, HEAD_DIM_INV), EPS))
    scaled = pl.row_expand_mul(qk_per_head, inv)
    return zero_centered_rmsnorm_apply(scaled, gamma_128)


@pl.jit.inline
def partial_rope_rotate(
    lo,        # pl.Tensor[[rows, rotary_half], pl.FP32]
    hi,        # pl.Tensor[[rows, rotary_half], pl.FP32]
    cos_lo,    # pl.Tensor[[1, rotary_half], pl.FP32]
    cos_hi,    # pl.Tensor[[1, rotary_half], pl.FP32]
    sin_lo,    # pl.Tensor[[1, rotary_half], pl.FP32]
    sin_hi,    # pl.Tensor[[1, rotary_half], pl.FP32]
):
    """One llama-style RoPE rotation step on an (lo, hi) half-pair.

    The rotated pair is::

        rot_lo = lo * cos_lo - hi * sin_lo
        rot_hi = hi * cos_hi + lo * sin_hi

    Step3p5 layer flavours differ only in the half-dim the caller slices:

      - Full attention (partial=0.5): rotary_dim=64, rotary_half=32. The
        caller leaves the trailing ``HEAD_DIM - rotary_dim = 64`` lanes
        un-rotated (pass-through).
      - SWA (partial=1.0): rotary_dim=128, rotary_half=64. No pass-through
        tail (rotary slice covers the full HEAD_DIM).
    """
    rot_lo = pl.sub(pl.col_expand_mul(lo, cos_lo), pl.col_expand_mul(hi, sin_lo))
    rot_hi = pl.add(pl.col_expand_mul(hi, cos_hi), pl.col_expand_mul(lo, sin_hi))
    return rot_lo, rot_hi


@pl.jit.inline
def head_wise_gate_apply(
    attn_head_slice_bf16,   # pl.Tensor[[rows, HEAD_DIM], pl.BF16]
    gate_logit_col_fp32,    # pl.Tensor[[rows, 1], pl.FP32]
):
    """Multiply one attn-out head slab by its per-head sigmoid gate.

    Step3p5's head-wise attention gate uses
    ``gate = sigmoid(current_hidden @ w_g)`` precomputed during scope-1
    (w_g is laid out as ``[HIDDEN, NUM_HEADS]`` -- the host transposes
    the checkpoint's ``[NUM_HEADS, HIDDEN]`` orientation at load time).

    For each Q head ``h``, the column ``gate_logits[:, h:h+1]`` is
    broadcast across the HEAD_DIM lanes of that head's attn_out slab and
    multiplied in. This helper handles one such head's broadcast.
    """
    gate = pl.recip(pl.add(pl.exp(pl.neg(gate_logit_col_fp32)), 1.0))
    gated_fp32 = pl.col_expand_mul(
        pl.cast(attn_head_slice_bf16, target_type=pl.FP32), gate,
    )
    return pl.cast(gated_fp32, target_type=pl.BF16)


# =============================================================================
# Host-side torch RoPE table builders (used by input-gen and goldens).
# =============================================================================


def build_llama3_yarn_rope_tables(
    seq_len: int,
    head_rotary_dim: int,
    rope_theta: float,
    factor: float = 2.0,
    low: float = 1.0,
    high: float = 32.0,
    orig_max: int = 131072,
):
    """Build llama3-yarn-scaled cos/sin tables for full-attention layers.

    Mirrors vllm's ``_compute_llama3_parameters``. The freq spectrum is
    split into three regions by ``low_freq_wavelen`` and
    ``high_freq_wavelen``:

      - wavelen < high_freq_wavelen: no scaling.
      - wavelen > low_freq_wavelen:  scale down by ``factor``.
      - in-between:                  smooth interpolation.

    Args:
        seq_len: number of positions to tabulate (``max_seq``).
        head_rotary_dim: width of the rope rotation slice
            (``= HEAD_DIM * partial_rotary_factor`` -- 64 for full).
        rope_theta: layer rope_theta (5_000_000.0 for step3p5 full).
        factor / low / high / orig_max: yarn parameters from
            ``config.ROPE_SCALING``.

    Returns:
        ``(cos, sin)`` each shape ``[seq_len, head_rotary_dim]`` FP32.
    """
    import math

    import torch

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_rotary_dim, 2, dtype=torch.float32) / head_rotary_dim)
    )

    low_freq_wavelen = orig_max / low
    high_freq_wavelen = orig_max / high
    wavelen = 2.0 * math.pi / inv_freq

    smooth_factor = (orig_max / wavelen - low) / (high - low)
    smoothed_inv_freq = (
        (1.0 - smooth_factor) * inv_freq / factor + smooth_factor * inv_freq
    )

    inv_freq_scaled = torch.where(
        wavelen < high_freq_wavelen,
        inv_freq,
        torch.where(
            wavelen > low_freq_wavelen,
            inv_freq / factor,
            smoothed_inv_freq,
        ),
    )

    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
    angles = positions * inv_freq_scaled.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return cos_full, sin_full


def build_plain_rope_tables(
    seq_len: int,
    head_rotary_dim: int,
    rope_theta: float,
):
    """Build unscaled cos/sin tables for SWA layers (no yarn)."""
    import torch

    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_rotary_dim, 2, dtype=torch.float32) / head_rotary_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1)
    angles = positions * inv_freq.unsqueeze(0)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    return cos_full, sin_full


__all__ = [
    "zero_centered_rmsnorm_apply",
    "per_head_qk_norm",
    "partial_rope_rotate",
    "head_wise_gate_apply",
    "build_llama3_yarn_rope_tables",
    "build_plain_rope_tables",
    # Distributed collectives — re-exported from .collectives (Phase 9 Wave 1).
    # See ``collectives.py`` module docstring for the assumed topology
    # (TP/EP both = 8, co-located on a single 8-card node) and the
    # caller-managed buffer-window contract.
    "tp_all_reduce",
    "tp_all_gather",
    "tp_reduce_scatter",
    "ep_all_to_all",
]


# =============================================================================
# Distributed collectives — thin re-export (Phase 9 Wave 1).
#
# The actual implementations live in ``collectives.py`` to keep this file
# focused on per-card vector / scalar helpers. The wrappers below are pure
# pypto-IR builders (no decorator) — call them inside an
# ``@pl.function(type=pl.FunctionType.InCore)`` body of the consumer
# kernel; ``host_orch`` allocates the windowed scratch via
# ``pld.alloc_window_buffer`` (see ``collectives.py`` for the
# host-orchestrator pattern this mirrors).
#
# Assumed topology (see ``config.py``): single node, 8 cards, one process
# per card, TP_WORLD_SIZE = EP_WORLD_SIZE = 8 (co-located groups).
# =============================================================================
from .collectives import (  # noqa: E402,F401  (deliberate trailing re-export)
    ep_all_to_all,
    tp_all_gather,
    tp_all_reduce,
    tp_reduce_scatter,
)
