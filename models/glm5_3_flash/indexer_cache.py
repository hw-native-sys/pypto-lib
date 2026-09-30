# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The indexer's own paged state cache: 256 values per token per DSA layer.

``Glm5NextTextIndexer.forward`` packs ``[key(128), gate_scores(128), valid(1)]``
into one row and pushes it through ``past_key_values.update_indexer``. The valid
channel is the cached form of the attention mask, and it exists only to let pooling
start at the first real token of a **left-padded dense** batch. A paged cache has no
left padding, so this ABI drops it and derives validity from the request length —
which is also what vLLM Ascend does (``AscendIndexerKPoolStateSpec`` has
``head_size = 2 * 128``). A kernel that ever sees a padded dense layout must put it
back.

vLLM Ascend splits this into two specs — an ``AscendMLAAttentionSpec`` with
``head_size=128`` for the keys and an ``AscendIndexerKPoolStateSpec`` with
``block_size=4``, ``head_size=256`` and ``dtype=float32`` for the pooled state — and
excludes both from the generic page-size alignment so they form their own page
class. Deciding whether to keep that split or store the raw 257-wide row is the
first thing the assignee has to settle with the cache manager.

The ABI below stores FP32, which is what vLLM Ascend enforces for the pooled state.
The a2a3 sibling port quantizes its own indexer cache to INT8 with a per-row FP32
scale (``docs/models/deepseek_v4_flash_mtp/index.md``), which would take this from
about 3 KB to 0.8 KB per token per rank. Treat that as a follow-up, not a default:
the pooled key feeds a ``relu``-gated score whose sensitivity to INT8 has not been
measured for this checkpoint.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, INDEX_DIM, INDEX_STATE_WIDTH
from models.glm5_3_flash.config import TABLE_DYN, T_DYN


def golden_indexer_cache_write(
    cache: torch.Tensor,
    index_k: torch.Tensor,
    gate_scores: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's packed indexer state, skipping ``-1`` slots.

    A slot of ``-1`` marks a row that owns no cache position — a padded row in a
    packed decode batch — and is skipped, exactly as
    :func:`models.glm5_3_flash.mla_cache.golden_mla_cache_write` skips it for the
    latent pool. ``index_slots`` and ``mla_slots`` are the same kind of quantity, both
    built by :func:`models.glm5_3_flash.metadata.paged_slots`, so without the guard a
    negative slot would be a wrapped index into the tail of the pool. The kernel body
    owes the same guard.
    """
    packed = torch.cat([index_k, gate_scores], dim=-1)
    updated = cache.clone()
    written = slots >= 0
    updated[slots.to(torch.long)[written]] = packed.to(cache.dtype)[written]
    return updated


@pl.jit.inline
def indexer_cache_write(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
):
    """Scatter one step's packed indexer state, one block per row.

    The row is ``[key(INDEX_DIM), gate_scores(INDEX_DIM)]``, so the two halves are
    stored separately rather than concatenated on core: the key arrives BF16 and is
    widened to the cache's FP32, while the gate half is already FP32 and moves as a
    plain copy. Each half is ``INDEX_DIM * 4`` = 512 B, exactly the a2a3 GM access
    granularity, so both stores are whole bursts and neither needs padding.

    A slot of ``-1`` owns no cache row and is skipped, as in
    :func:`models.glm5_3_flash.mla_cache.mla_cache_write`. The scatter assumes
    distinct slots; two rows sharing one slot resolve by dispatch order, which is not
    defined here.

    Returns the region's TaskId so a later read can order against the write.
    """
    tokens = pl.tensor.dim(index_k, 0)
    with pl.spmd(tokens, name_hint="indexer_cache_write") as write_tid:
        token = pl.tile.get_block_idx()
        slot_i32 = pl.read(slots, [token])
        if slot_i32 >= 0:
            slot = pl.cast(slot_i32, pl.INDEX)
            key = pl.cast(
                pl.load(index_k, [token, 0], [1, INDEX_DIM]), target_type=pl.FP32
            )
            pl.store(key, [slot, 0], cache)
            pl.store(pl.load(gate_scores, [token, 0], [1, INDEX_DIM]), [slot, INDEX_DIM], cache)
    return write_tid


@pl.jit
def indexer_cache_write_test(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32]],
):
    """Run one scatter for golden.run validation."""
    index_k.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    indexer_cache_write(index_k, gate_scores, slots, cache)
    return cache


def build_indexer_cache_tensor_specs(tokens: int = 24, pages: int = 4):
    """Build one deterministic scatter: distinct slots, a padded row, a spare page.

    The slots are distinct and deliberately unordered across pages, and row 3 carries
    ``-1``. ``cache`` is ``InOut`` so its untouched rows are uploaded and can be
    asserted, rather than read back as allocator residue.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(67)
    cache_rows = pages * BLOCK_SIZE

    def init_index_k():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32).bfloat16()

    def init_gate_scores():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32)

    def init_cache():
        return torch.randn(cache_rows, INDEX_STATE_WIDTH, generator=generator, dtype=torch.float32)

    def init_slots():
        # Distinct rows, drawn out of order.
        chosen = torch.randperm(cache_rows, generator=generator)[:tokens]
        slots = chosen.to(torch.int32)
        slots[3] = -1
        return slots

    return [
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16, init_value=init_index_k),
        TensorSpec(
            "gate_scores", [tokens, INDEX_DIM], torch.float32, init_value=init_gate_scores
        ),
        TensorSpec("slots", [tokens], torch.int32, init_value=init_slots),
        TensorSpec(
            "cache", [cache_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_cache
        ),
    ]


def golden_indexer_cache_case(tensors):
    """Fill the expected cache state for :func:`build_indexer_cache_tensor_specs`."""
    tensors["cache"][:] = golden_indexer_cache_write(
        tensors["cache"], tensors["index_k"], tensors["gate_scores"], tensors["slots"]
    )


def main():
    """Prove the golden on CPU, then validate the scatter on device.

    The key half is widened BF16 to FP32, which is exact, and the gate half is copied,
    so the device case runs at zero tolerance: the scatter either moves the bytes or is
    wrong.
    """
    import argparse

    from golden import run
    from models.glm5_3_flash._golden_smoke import run_indexer_cache_golden

    run_indexer_cache_golden(golden_indexer_cache_write)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    result = run(
        fn=indexer_cache_write_test,
        specs=build_indexer_cache_tensor_specs(args.tokens, args.pages),
        golden_fn=golden_indexer_cache_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=0.0,
        atol=0.0,
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "build_indexer_cache_tensor_specs",
    "golden_indexer_cache_case",
    "golden_indexer_cache_write",
    "indexer_cache_write",
    "indexer_cache_write_test",
]


if __name__ == "__main__":
    main()
