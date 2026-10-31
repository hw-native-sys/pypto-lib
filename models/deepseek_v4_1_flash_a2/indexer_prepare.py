"""Prepare QLI output for the A2/A3 sparse-attention consumer.

The native QLI output is score ordered.  vLLM-Ascend's
``prepare_indexer_indices`` filters entries beyond the visible compressed
prefix, sorts the surviving logical positions chronologically, and emits a
``-1`` tail.  The CPU reference also returns the valid count used by sparse
metadata.  Keeping this as a separate PyPTO boundary matches the vLLM graph
and gives cache gather a stable INT32 ABI.
"""

from __future__ import annotations

import torch
import pypto.language as pl


INDEXER_TOPK = 512
SORT_LEAF = 512
NEGATIVE_INF = -3.0e38


def prepare_indexer_indices_reference(
    selected: torch.Tensor,
    positions: torch.Tensor,
    *,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """CPU reference for vLLM's visible-prefix filter and chronological sort."""

    if selected.ndim != 2 or selected.dtype != torch.int32:
        raise ValueError("selected must be a 2-D INT32 tensor")
    if positions.shape != (selected.shape[0],) or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be a 1-D INT32/INT64 tensor")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")
    tokens, topk = selected.shape
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=topk)
    valid = selected.ge(0) & selected.lt(visible[:, None])
    sentinel = torch.iinfo(torch.int32).max
    sortable = torch.where(valid, selected, torch.full_like(selected, sentinel))
    ordered = sortable.sort(dim=-1, stable=True).values
    output = torch.where(ordered.eq(sentinel), torch.full_like(ordered, -1), ordered)
    lengths = valid.sum(dim=-1, keepdim=True, dtype=torch.int32)
    return {"indices": output, "lengths": lengths}


def make_indexer_prepare(
    capacity: int,
    topk: int = INDEXER_TOPK,
    *,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket A2/A3 index preparation kernel."""

    if capacity <= 0 or capacity % 16:
        raise ValueError("capacity must be a positive multiple of 16")
    if topk <= 0 or topk % 32 or topk > SORT_LEAF:
        raise ValueError("topk must be a positive multiple of 32 and <= 512")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit
    def kernel(
        raw_selected: pl.Tensor[[capacity, topk], pl.INT32],
        positions: pl.Tensor[[capacity], pl.INT32],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        with pl.spmd(active, name_hint="dsv41_a2a3_prepare_indexer_indices"):
            token = pl.tile.get_block_idx()
            position = pl.max(pl.read(positions, [token]), 0)
            visible = pl.min((position + 1) // compress_ratio, topk)
            # sort32 is descending.  Negated logical IDs therefore produce
            # ascending chronological IDs, while the very negative sentinel
            # remains at the tail.
            scores = pl.tile.full([1, topk], dtype=pl.FP32, value=NEGATIVE_INF)
            ids = pl.tile.full([1, topk], dtype=pl.INT32, value=-1)
            for lane in pl.range(topk):
                index = pl.read(raw_selected, [token, lane])
                if index >= 0:
                    if index < visible:
                        pl.tile.write(scores, [0, lane], -pl.cast(index, pl.FP32))
                        pl.tile.write(ids, [0, lane], index)
            pairs = pl.tile.sort32(scores, pl.reinterpret_view(ids, pl.UINT32))
            if topk >= 128:
                pairs = pl.tile.mrgsort(pairs, block_len=64)
                pairs = pl.tile.mrgsort(pairs, block_len=256)
            ordered = pl.tile.slice(pairs, [1, 2 * topk], [0, 0])
            output = pl.tile.gather_mask(
                ordered, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32
            )
            pl.store(output, [token, 0], indices)
        return indices

    return kernel


__all__ = [
    "INDEXER_TOPK",
    "make_indexer_prepare",
    "prepare_indexer_indices_reference",
]
