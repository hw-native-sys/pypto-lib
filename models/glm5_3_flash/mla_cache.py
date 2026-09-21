# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The paged 512-wide latent cache the 12 MLA layers share.

One row per token per layer: ``kv_a_layernorm(kv_a_proj_with_mqa(x))``, width
``kv_lora_rank = 512``, with ``num_kv_heads = 1``, so the cache is **not** TP
sharded — every rank holds the same latent. Pages are ``BLOCK_SIZE = 128`` tokens.

The write is a scatter by ``ForwardMetadata.mla_slots``; the read is a paged gather
driven by the indexer's selected rows, which is why the two live in separate files.

At BF16 this is ``12 layers x 512 x 2`` = 12.3 KB per token per rank, and it is the
larger half of the hybrid cache budget.

A slot of ``-1`` marks a row that owns no cache position — a padded row in a packed
decode batch, where a request commits fewer than ``DECODE_ROWS_PER_REQUEST`` tokens —
and is skipped. Every sibling scatter in ``models/`` carries the same guard; without
it a negative slot would be a wrapped index into the tail of the pool. The scatter
assumes distinct slots: two rows sharing one slot resolve by dispatch order, which is
not defined here.

One ``KV_LORA``-wide BF16 row is 1,024 B, a multiple of the 512 B a2a3 GM access
granularity, so each row moves as whole bursts and the store needs no alignment
padding. The kernel is pure data movement; if a profile shows it dispatch-bound
rather than MTE-bound, tiling several tokens into one block is the first lever.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, KV_LORA, TABLE_DYN, T_DYN


def golden_mla_cache_write(
    cache: torch.Tensor,
    latent: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's latent rows into the paged pool, skipping ``-1`` slots.

    Args:
        cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` pool holding the prior state.
        latent: ``[T, KV_LORA]`` rows from :func:`models.glm5_3_flash.mla_prolog.golden_mla_prolog`.
        slots: ``[T]`` physical rows from ``ForwardMetadata.mla_slots``, ``-1`` for a row
            that owns no cache position.

    Returns:
        A new pool with the selected rows replaced; the input is left untouched.
    """
    updated = cache.clone()
    written = slots >= 0
    updated[slots.to(torch.long)[written]] = latent.to(cache.dtype)[written]
    return updated


@pl.jit.inline
def mla_cache_write(
    latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
):
    """Scatter one step's latent rows into the paged pool, one block per row.

    Returns the region's TaskId so a later gather can order against the write.
    """
    tokens = pl.tensor.dim(latent, 0)
    with pl.spmd(tokens, name_hint="mla_cache_write") as write_tid:
        token = pl.tile.get_block_idx()
        slot_i32 = pl.read(slots, [token])
        if slot_i32 >= 0:
            slot = pl.cast(slot_i32, pl.INDEX)
            pl.store(pl.load(latent, [token, 0], [1, KV_LORA]), [slot, 0], cache)
    return write_tid


@pl.jit
def mla_cache_write_test(
    latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16]],
):
    """Run one scatter for golden.run validation."""
    latent.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    mla_cache_write(latent, slots, cache)
    return cache


def build_mla_cache_tensor_specs(tokens: int = 24, pages: int = 4):
    """Build one deterministic scatter: distinct slots, a padded row, a spare page.

    The slots are distinct and deliberately unordered across pages, and row 3 carries
    ``-1``. ``cache`` is ``InOut`` so its untouched rows are uploaded and can be
    asserted, rather than read back as allocator residue.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(53)
    cache_rows = pages * BLOCK_SIZE

    def init_latent():
        return torch.randn(tokens, KV_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_slots():
        # Distinct rows, drawn out of order.
        chosen = torch.randperm(cache_rows, generator=generator)[:tokens]
        slots = chosen.to(torch.int32)
        slots[3] = -1
        return slots

    return [
        TensorSpec("latent", [tokens, KV_LORA], torch.bfloat16, init_value=init_latent),
        TensorSpec("slots", [tokens], torch.int32, init_value=init_slots),
        TensorSpec("cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
    ]


def golden_mla_cache_case(tensors):
    """Fill the expected pool state for :func:`build_mla_cache_tensor_specs`."""
    tensors["cache"][:] = golden_mla_cache_write(
        tensors["cache"], tensors["latent"], tensors["slots"]
    )


def main():
    """Prove the golden on CPU, then validate the scatter on device.

    The device case runs at zero tolerance: a scatter either moves the bytes or is
    wrong, so there is no precision budget to spend.
    """
    import argparse

    from golden import run
    from models.glm5_3_flash._golden_smoke import run_mla_cache_golden

    run_mla_cache_golden(golden_mla_cache_write)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run(
        fn=mla_cache_write_test,
        specs=build_mla_cache_tensor_specs(args.tokens, args.pages),
        golden_fn=golden_mla_cache_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=0.0,
        atol=0.0,
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "build_mla_cache_tensor_specs",
    "golden_mla_cache_case",
    "golden_mla_cache_write",
    "mla_cache_write",
    "mla_cache_write_test",
]


if __name__ == "__main__":
    main()
