# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Paged-attention KV-cache helpers shared by hf_compare cases.

The kernels write KV into a flat row-major paged layout::

    cache_row(pbid, kvh, offset) = (pbid * num_kv_heads + kvh) * block_size + offset

For fused multi-layer kernels, layers are concatenated along axis 0; pass
``layer_offset_rows = layer_idx * layer_cache_rows`` to address a specific layer.

This module provides:

* ``init_block_table_identity`` / ``compute_*_slot_mapping`` -- build paged
  metadata that case fixtures hand to kernels.
* ``select_kv_at_decode_slot`` -- gather the [batch, num_kv_heads, head_dim]
  row written by a decode step.
* ``gather_prefill_kv`` -- assemble the [batch, num_kv_heads, seq_len, head_dim]
  block written by prefill across all positions.
* ``paged_to_dense_history`` -- stitch paged cache rows into the dense
  [batch, num_kv_heads, ctx_len, head_dim] tensors HF DynamicCache expects.
"""
from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Paged metadata initialisers (block_table / slot_mapping)
# ---------------------------------------------------------------------------
def init_block_table_identity(
    block_table: torch.Tensor, *, batch: int, max_blocks_per_seq: int
) -> None:
    """Fill block_table with logical->physical identity mapping (block i -> i)."""
    block_table[:] = torch.arange(batch * max_blocks_per_seq, dtype=torch.int32)


def compute_decode_slot_mapping(
    slot_mapping: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    batch: int,
    block_size: int,
    max_blocks_per_seq: int,
) -> None:
    """Set slot_mapping[b] for the current decode position (pos = seq_lens[b] - 1)."""
    for b in range(batch):
        pos = int(seq_lens[b].item()) - 1
        logical_block = pos // block_size
        page_off = pos - logical_block * block_size
        phys_block = b * max_blocks_per_seq + logical_block
        slot_mapping[b] = phys_block * block_size + page_off


def compute_prefill_slot_mapping(
    slot_mapping: torch.Tensor,
    *,
    batch: int,
    max_seq: int,
    block_size: int,
    max_blocks_per_seq: int,
) -> None:
    """Set slot_mapping[b*max_seq + pos] for every (b, pos) in [0, max_seq)."""
    for b in range(batch):
        base_block = b * max_blocks_per_seq
        for pos in range(max_seq):
            logical_block = pos // block_size
            page_off = pos - logical_block * block_size
            phys_block = base_block + logical_block
            slot_mapping[b * max_seq + pos] = phys_block * block_size + page_off


# ---------------------------------------------------------------------------
# Cache-readback helpers (gather kernel-written KV for comparison)
# ---------------------------------------------------------------------------
def select_kv_at_decode_slot(
    cache_flat: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    layer_offset_rows: int = 0,
) -> torch.Tensor:
    """Read the row a decode step wrote, returning [batch, num_kv_heads, head_dim].

    For a fused multi-layer cache pass ``layer_offset_rows = layer_idx * layer_cache_rows``.
    """
    batch = int(slot_mapping.shape[0])
    out = torch.empty(
        (batch, num_kv_heads, head_dim),
        dtype=cache_flat.dtype, device=cache_flat.device,
    )
    for b in range(batch):
        slot = int(slot_mapping[b].item())
        slot_block = slot // block_size
        slot_off = slot - slot_block * block_size
        ofs = layer_offset_rows + slot_block * num_kv_heads * block_size + slot_off
        for kvh in range(num_kv_heads):
            out[b, kvh, :] = cache_flat[ofs + kvh * block_size, :]
    return out.clone()


def gather_prefill_kv(
    cache_flat: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    batch: int,
    max_seq: int,
    seq_len: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """Gather all prefill-written rows into a [B, num_kv_heads, seq_len, head_dim] tensor."""
    out = torch.empty(
        (batch, num_kv_heads, seq_len, head_dim),
        dtype=cache_flat.dtype, device=cache_flat.device,
    )
    for b in range(batch):
        for pos in range(seq_len):
            slot = int(slot_mapping[b * max_seq + pos].item())
            slot_block = slot // block_size
            slot_off = slot - slot_block * block_size
            base = slot_block * num_kv_heads * block_size + slot_off
            for kvh in range(num_kv_heads):
                out[b, kvh, pos, :] = cache_flat[base + kvh * block_size, :]
    return out.clone()


def paged_to_dense_history(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    batch: int,
    ctx_len: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    max_blocks_per_seq: int,
    hf_dtype: torch.dtype,
    layer_offset_rows: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stitch the paged KV cache into dense ``[B, num_kv_heads, ctx_len, head_dim]``.

    Walks ``block_table`` to assemble per-sequence histories, then truncates
    each to ``ctx_len``. Used by HF references that seed ``DynamicCache``.
    For a fused multi-layer cache pass ``layer_offset_rows`` to point at the
    layer of interest.
    """
    ctx_blocks = (ctx_len + block_size - 1) // block_size
    k_hist = torch.empty((batch, num_kv_heads, ctx_len, head_dim), dtype=hf_dtype)
    v_hist = torch.empty((batch, num_kv_heads, ctx_len, head_dim), dtype=hf_dtype)
    for b in range(batch):
        for kvh in range(num_kv_heads):
            k_blocks: list[torch.Tensor] = []
            v_blocks: list[torch.Tensor] = []
            for sb in range(ctx_blocks):
                pbid = int(block_table[b * max_blocks_per_seq + sb].item())
                row0 = layer_offset_rows + (pbid * num_kv_heads + kvh) * block_size
                k_blocks.append(k_cache[row0:row0 + block_size, :])
                v_blocks.append(v_cache[row0:row0 + block_size, :])
            k_hist[b, kvh, :, :] = torch.cat(k_blocks, dim=0)[:ctx_len, :].to(hf_dtype)
            v_hist[b, kvh, :, :] = torch.cat(v_blocks, dim=0)[:ctx_len, :].to(hf_dtype)
    return k_hist, v_hist
