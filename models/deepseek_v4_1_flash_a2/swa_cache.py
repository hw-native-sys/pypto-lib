"""BF16 sliding-window cache publication and gather."""

from __future__ import annotations

import torch
import pypto.language as pl

from .operator_config import CONTRACT

BLOCK_SIZE = 128

def write_swa_cache_reference(
    cache: torch.Tensor,
    values: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> torch.Tensor:
    """Write BF16 rows into a ``[blocks, 128, 1, head_dim]`` cache."""

    if cache.dtype != torch.bfloat16 or values.dtype != torch.bfloat16:
        raise TypeError("A2/A3 SWA cache and values must be BF16")
    result = cache.clone()
    flat = result.view(-1, result.shape[-1])
    for row, slot in zip(values, slot_mapping.reshape(-1, 2).tolist()):
        page, offset = slot
        if page >= 0:
            flat[page * CONTRACT.swa_block_size + offset].copy_(row)
    return result


def gather_swa_cache_reference(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather physical cache rows; negative indices produce zero rows."""

    flat = cache.reshape(-1, cache.shape[-1])
    result = torch.zeros(*indices.shape, cache.shape[-1], dtype=cache.dtype, device=cache.device)
    valid = indices >= 0
    if valid.any():
        result[valid] = flat[indices[valid].long()]
    return result


def make_swa_cache_publish(capacity: int, cache_blocks: int, head_dim: int):
    cache_rows = cache_blocks * BLOCK_SIZE

    @pl.jit.inline
    def publish(
        values: pl.Tensor[[capacity, head_dim], pl.BF16],
        slots: pl.Tensor[[capacity, 2], pl.INT64],
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for token in pl.spmd(capacity, name_hint="dsv41_a2a3_swa_publish"):
            if token < active:
                page = pl.read(slots, [token, 0])
                offset = pl.read(slots, [token, 1])
                if page >= 0:
                    physical = pl.cast(page * 128 + offset, pl.INDEX)
                    flat[physical:physical + 1, :] = values[token:token + 1, :]
        return cache

    return publish


def make_swa_cache_gather(capacity: int, window: int, cache_blocks: int, head_dim: int):
    cache_rows = cache_blocks * BLOCK_SIZE

    @pl.jit.inline
    def gather(
        cache: pl.Tensor[[cache_blocks, 128, 1, head_dim], pl.BF16],
        indices: pl.Tensor[[capacity, window], pl.INT32],
        output: pl.Tensor[[capacity * window, head_dim], pl.BF16],
        active: pl.Scalar[pl.INT32],
    ):
        flat = pl.reshape(cache, [cache_rows, head_dim])
        for block in pl.spmd(capacity * window, name_hint="dsv41_a2a3_swa_gather"):
            token = block // window
            column = block % window
            if token < active:
                index = pl.read(indices, [token, column])
                if index >= 0:
                    physical = pl.cast(index, pl.INDEX)
                    output[block:block + 1, :] = flat[physical:physical + 1, :]
                else:
                    output[block:block + 1, :] = pl.full([1, head_dim], dtype=pl.BF16, value=0.0)
        return output

    return gather


__all__ = [
    "gather_swa_cache_reference",
    "make_swa_cache_gather",
    "make_swa_cache_publish",
    "write_swa_cache_reference",
]
