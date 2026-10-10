# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The indexer's own paged state cache: raw support rows plus a pooled key table.

Two tables, written by two kernels, make up the DSA indexer's persistent state:

* the **raw support table** — one FP32 row per token, ``[key(128), gate_scores(128)]``
  (:data:`INDEX_STATE_WIDTH` wide), paged at :data:`BLOCK_SIZE` like the MLA latent
  pool. :func:`indexer_cache_write` scatters one dispatch's rows into it. It is the
  authoritative record the pooled table is derived from: scoring never reads it, but
  every pool that closes is compressed from these rows, so re-pooling is always
  possible and the gate projections are never recomputed.
* the **pooled key table** — one INT8 :data:`INDEX_DIM`-wide row per closed pool
  plus one FP32 dequant scale per row (:data:`POOL_SCALE_WIDTH`-lane padded), paged
  at :data:`INDEX_STATE_BLOCK_SIZE` pools per block. :func:`indexer_pool_write`
  closes pools: for one pool event it reads the four member raw rows, takes the
  learned per-dimension softmax ``softmax(gate_scores + ape)`` over the four lanes —
  the reference ``Glm5NextTextIndexer.get_pooled_states`` does this in FP32 — and
  stores the weighted key rotated into the scorer's Hadamard basis, quantized to
  INT8 with its scale, written once at close.
  The score then reads this table directly and never re-quantizes it, which is the
  whole point of the layout: a query scans one pooled row per four tokens instead of
  pooling on the fly, and the quantization cost is paid once per pool rather than
  once per scoring pass.

A pool closes exactly when its fourth token is cached, so the write side is driven by
**pool-close events**: prefill emits one event per pool the chunk completes, a decode
dispatch of ``DECODE_ROWS_PER_REQUEST`` = 4 rows per request completes exactly one
pool per request. The incomplete tail pool is never stored —
``index_kpool_always_select_tail`` force-selects its raw positions in the expansion,
so it is not a scoring candidate and needs no pooled row. This is the same shape the
a2a3 sibling port arrives at (``deepseek_v4_flash_mtp`` keeps a compressed
per-four-token table and a rolling tail) and the shape vLLM Ascend's page classes
describe; what has no donor is the pooling math itself, because the gate scores here
come from the cached projection rather than an in-kernel recompute.

``Glm5NextTextIndexer.forward`` packs a ``valid`` channel next to key and gate. That
channel only exists to let pooling start at the first real token of a left-padded
dense batch; a paged cache has no left padding, so this ABI drops it the way
``AscendIndexerKPoolStateSpec`` does, and a pool event whose member slot is ``-1``
(padded row) is rejected through :func:`indexer_pool_write`'s ``pool_valid`` output
instead.

FP32 keys and gates cost 1 KB per token per rank per DSA layer on the raw side and
about 33 B per token on the pooled side (INT8 key plus the padded scale lane). The
INT8 pooled row follows the a2a3 sibling's C8 shape exactly — its scorer proved the
quantized pooled key is accuracy-neutral for the relu-gated score — and the INT8
quantization runs per closed pool at write time, so the per-row work is identical
to quantizing at score time while the per-pass cost disappears.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import BLOCK_SIZE, INDEX_BLOCKS_DYN, INDEX_DIM
from models.glm5_3_flash.config import INDEX_KPOOL, INDEX_STATE_BLOCK_SIZE
from models.glm5_3_flash.config import INDEX_STATE_WIDTH, POOLS_DYN, TABLE_DYN, T_DYN
from models.glm5_3_flash.quantization import INT8_AMAX_EPS, INT8_SCALE_MAX
from models.glm5_3_flash.quantization import quantize_per_token_int8


# The per-event validity flag is one INT32 lane padded to the 32-byte minimum of
# an a2a3 vector tile; the flag sits in column 0 and columns 1-7 stay zero.
POOL_VALID_WIDTH = 8

# The pooled table stores INT8 keys plus one FP32 dequant scale per row, written
# once at pool close and read directly by the scorer. The scale lane is padded to
# the same 32-byte minimum; the writer replicates the scale across all lanes and
# the scorer reads column 0.
POOL_SCALE_WIDTH = 8

# Event rows quantized and scattered per block of the pool-write epilogue.
POOL_QUANT_TILE = 64


def sylvester_hadamard(head_dim: int = INDEX_DIM) -> torch.Tensor:
    """The scaled Sylvester Hadamard the load-time converter builds.

    Lives here rather than in :mod:`models.glm5_3_flash.prefill_indexer` because
    the pool writer consumes it too, and prefill imports this module.
    """
    matrix = torch.ones((1, 1))
    while matrix.shape[0] < head_dim:
        matrix = torch.cat([torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0)
    return (matrix * (head_dim**-0.5)).to(torch.bfloat16)


def golden_indexer_cache_write(
    cache: torch.Tensor,
    index_k: torch.Tensor,
    gate_scores: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's raw rows, skipping ``-1`` slots.

    The guard is the ABI, not politeness: an unmasked ``-1`` indexes the last row of
    the pool through Python's negative indexing and silently corrupts it.
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
    """Scatter one step's raw indexer rows into the paged pool, one block per row.

    A slot of ``-1`` marks a row that owns no cache position — a padded row in a
    packed batch — and is skipped, so a negative slot cannot wrap into the tail of
    the pool. The key half is widened to FP32 on the way in: the pooled-table writer
    re-reads these rows in FP32, and the storage ABI is FP32 end to end.

    Returns the region's TaskId so a later reader can order against the write.
    """
    tokens = pl.tensor.dim(index_k, 0)
    with pl.spmd(tokens, name_hint="indexer_cache_write") as write_tid:
        token = pl.tile.get_block_idx()
        slot_i32 = pl.read(slots, [token])
        if slot_i32 >= 0:
            slot = pl.cast(slot_i32, pl.INDEX)
            key_f32 = pl.cast(pl.load(index_k, [token, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
            pl.store(key_f32, [slot, 0], cache)
            gate_f32 = pl.load(gate_scores, [token, 0], [1, INDEX_DIM])
            pl.store(gate_f32, [slot, INDEX_DIM], cache)
    return write_tid


def golden_indexer_pool_write(
    pool_cache: torch.Tensor,
    pool_scale: torch.Tensor,
    raw_cache: torch.Tensor,
    compress_ape: torch.Tensor,
    hadamard: torch.Tensor,
    pool_token_slots: torch.Tensor,
    pool_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Close each pool event: per-dimension softmax over its four gate lanes.

    The pooled key is rotated into the scorer's Hadamard basis (with the same
    BF16 round the kernel's operand takes) and quantized to INT8 plus a per-row
    dequant scale at write time, exactly the chain the kernel runs, so both
    outputs compare exactly.

    Args:
        pool_cache: ``[POOL_TABLE * INDEX_STATE_BLOCK_SIZE, INDEX_DIM]`` INT8
            pooled-key table holding the prior state.
        pool_scale: ``[POOL_TABLE * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH]``
            FP32 per-row dequant scales holding the prior state.
        raw_cache: ``[TABLE * BLOCK_SIZE, INDEX_STATE_WIDTH]`` raw support table.
        compress_ape: ``[INDEX_KPOOL, INDEX_DIM]`` learned pool-lane bias.
        hadamard: ``[INDEX_DIM, INDEX_DIM]`` BF16 scorer-basis rotation.
        pool_token_slots: ``[events, INDEX_KPOOL]`` physical raw row of each member
            token, ``-1`` for a padded member.
        pool_slots: ``[events]`` destination pooled row of each event.

    Returns:
        The updated INT8 table, the updated scale table and the per-event
        validity (all four members present).
    """
    updated = pool_cache.clone()
    updated_scale = pool_scale.clone()
    valid = torch.zeros(pool_slots.numel(), POOL_VALID_WIDTH, dtype=torch.int32)
    for event in range(pool_slots.numel()):
        members = pool_token_slots[event].to(torch.long)
        destination = int(pool_slots[event])
        present = bool((members >= 0).all())
        if present:
            keys = raw_cache[members, 0:INDEX_DIM].float()
            gates = raw_cache[members, INDEX_DIM:].float()
            probabilities = torch.softmax(gates + compress_ape.float(), dim=0)
            pooled = (probabilities * keys).sum(dim=0, keepdim=True)
        else:
            pooled = torch.zeros(1, INDEX_DIM)
        pooled = pooled.to(torch.bfloat16).float() @ hadamard.float()
        key_i8, key_scale = quantize_per_token_int8(pooled)
        updated[destination] = key_i8[0]
        updated_scale[destination, :] = key_scale[0, 0]
        valid[event, 0] = 1 if present else 0
    return updated, updated_scale, valid


@pl.jit.incore
def _indexer_pool_write_event(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_stage: pl.Tensor[[POOLS_DYN, INDEX_DIM], pl.FP32],
    pool_valid: pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32],
) -> None:
    """Close one pool event; the caller gives this scope one block per event.

    The softmax runs per dimension over the four pool lanes in FP32, matching the
    reference numerics: ``weights = softmax(gate + ape)`` along the lane axis, then
    ``pool_key = sum(weights * key)``. The FP32 pooled key lands in the
    contiguous ``pool_stage`` row of its event; a second scope quantizes the
    staged rows to INT8 plus a dequant scale and scatters them to the paged
    destinations. An event with a ``-1`` member slot is a padded pool: it stages a
    zero row and reports ``pool_valid = 0`` so the scorer can drop it. The staged
    row is always written, which keeps the epilogue's input fully defined.
    """
    event = pl.tile.get_block_idx()
    slot0 = pl.read(pool_token_slots, [event, 0])
    slot1 = pl.read(pool_token_slots, [event, 1])
    slot2 = pl.read(pool_token_slots, [event, 2])
    slot3 = pl.read(pool_token_slots, [event, 3])
    members_present = pl.min(pl.min(slot0, slot1), pl.min(slot2, slot3))
    # 0/1 flag in column 0 of a [1, 8] INT32 tile, the narrowest store the
    # 32-byte vector alignment accepts. Slot 0 is a legal raw row, so the flag
    # comes from the branch condition, not from clamping the slot value.
    flag_tile = pl.tile.full([1, POOL_VALID_WIDTH], dtype=pl.INT32, value=0)
    if members_present >= 0:
        # Per-lane logits and keys, kept as [1, INDEX_DIM] rows so every step is
        # an elementwise row operation; the softmax reduces over the four lanes.
        key0 = pl.load(raw_cache, [pl.cast(slot0, pl.INDEX), 0], [1, INDEX_DIM])
        key1 = pl.load(raw_cache, [pl.cast(slot1, pl.INDEX), 0], [1, INDEX_DIM])
        key2 = pl.load(raw_cache, [pl.cast(slot2, pl.INDEX), 0], [1, INDEX_DIM])
        key3 = pl.load(raw_cache, [pl.cast(slot3, pl.INDEX), 0], [1, INDEX_DIM])
        gate0 = pl.load(raw_cache, [pl.cast(slot0, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate1 = pl.load(raw_cache, [pl.cast(slot1, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate2 = pl.load(raw_cache, [pl.cast(slot2, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        gate3 = pl.load(raw_cache, [pl.cast(slot3, pl.INDEX), INDEX_DIM], [1, INDEX_DIM])
        ape0 = pl.cast(pl.load(compress_ape, [0, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape1 = pl.cast(pl.load(compress_ape, [1, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape2 = pl.cast(pl.load(compress_ape, [2, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        ape3 = pl.cast(pl.load(compress_ape, [3, 0], [1, INDEX_DIM]), pl.FP32, mode="none")
        logit0 = pl.add(gate0, ape0)
        logit1 = pl.add(gate1, ape1)
        logit2 = pl.add(gate2, ape2)
        logit3 = pl.add(gate3, ape3)
        lane_max = pl.maximum(pl.maximum(logit0, logit1), pl.maximum(logit2, logit3))
        weight0 = pl.exp(pl.sub(logit0, lane_max))
        weight1 = pl.exp(pl.sub(logit1, lane_max))
        weight2 = pl.exp(pl.sub(logit2, lane_max))
        weight3 = pl.exp(pl.sub(logit3, lane_max))
        denominator = pl.add(pl.add(weight0, weight1), pl.add(weight2, weight3))
        weighted_sum = pl.mul(weight0, key0)
        weighted_sum = pl.add(weighted_sum, pl.mul(weight1, key1))
        weighted_sum = pl.add(weighted_sum, pl.mul(weight2, key2))
        weighted_sum = pl.add(weighted_sum, pl.mul(weight3, key3))
        pool_key = pl.div(weighted_sum, denominator)
        pl.store(pool_key, [event, 0], pool_stage)
        pl.tile.write(flag_tile, [0, 0], pl.const(1, pl.INT32))
    else:
        zero_row = pl.tile.full([1, INDEX_DIM], dtype=pl.FP32, value=0.0)
        pl.store(zero_row, [event, 0], pool_stage)
    pl.store(flag_tile, [event, 0], pool_valid)


@pl.jit.inline
def indexer_pool_write(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_cache: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8],
    pool_scale: pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32],
    pool_valid: pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32],
):
    """Write one quantized pooled key row per pool-close event.

    The pooled table stores INT8 keys plus one FP32 dequant scale per row, both
    written once here at pool close — the scorer reads them directly and never
    re-quantizes the table (donor C8 shape). The softmax runs per dimension over
    the four pool lanes in FP32; each event stages its FP32 pooled key
    contiguously, then a block-parallel epilogue rotates rows of
    :data:`POOL_QUANT_TILE` events into the scorer's Hadamard basis, quantizes
    them and scatters INT8, scale and validity to the paged destinations. A
    padded event quantizes a zero row and reports ``pool_valid = 0``; the
    destination row itself is always written, which keeps the outputs fully
    defined for the golden harness.

    Returns the region's TaskId so the scorer can order against the pool writes.
    """
    events = pl.tensor.dim(pool_token_slots, 0)
    stage_rows = ((events + POOL_QUANT_TILE - 1) // POOL_QUANT_TILE) * POOL_QUANT_TILE
    pool_stage = pl.create_tensor([stage_rows, INDEX_DIM], dtype=pl.FP32)
    with pl.spmd(events, name_hint="indexer_pool_write") as pool_tid:
        _indexer_pool_write_event(raw_cache, compress_ape, pool_token_slots, pool_stage, pool_valid)
    for blk in pl.spmd((events + POOL_QUANT_TILE - 1) // POOL_QUANT_TILE, name_hint="indexer_pool_quant"):
        # Slicing wants INDEX offsets; the INT32 twin feeds the arithmetic.
        e0 = pl.cast(blk, pl.INT32) * POOL_QUANT_TILE
        e0_idx = pl.cast(e0, pl.INDEX)
        staged = pool_stage[e0_idx : e0_idx + POOL_QUANT_TILE, :]
        # Rotate into the scorer's Hadamard basis before quantization, so the
        # INT8 dot product computes (qH)·(kH) = q·k — the reference score. The
        # BF16 round mirrors the query side's BF16 ABI (and the donor's
        # rotate-then-quantize chain).
        staged_rot = pl.matmul(pl.cast(staged, pl.BF16, mode="rint"), hadamard, out_dtype=pl.FP32)
        staged_abs = pl.maximum(staged_rot, pl.neg(staged_rot))
        amax = pl.full([1, POOL_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        amax = pl.maximum(amax, pl.reshape(pl.row_max(staged_abs), [1, POOL_QUANT_TILE]))
        scale_quant_row = pl.div(pl.full([1, POOL_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), amax)
        staged_scaled = pl.row_expand_mul(staged_rot, pl.reshape(scale_quant_row, [POOL_QUANT_TILE, 1]))
        staged_i32 = pl.cast(staged_scaled, target_type=pl.INT32, mode="rint")
        staged_half = pl.cast(staged_i32, target_type=pl.FP16, mode="round")
        staged_i8 = pl.cast(staged_half, target_type=pl.INT8, mode="trunc")
        scale_dq_col = pl.reshape(pl.recip(scale_quant_row), [POOL_QUANT_TILE, 1])
        scale_pad = pl.row_expand_mul(
            pl.full([POOL_QUANT_TILE, POOL_SCALE_WIDTH], dtype=pl.FP32, value=1.0), scale_dq_col
        )
        for e in pl.range(POOL_QUANT_TILE):
            ei = pl.cast(e, pl.INT32)
            if e0 + ei < events:
                ei_idx = pl.cast(ei, pl.INDEX)
                dest = pl.cast(pl.read(pool_slots, [e0 + ei]), pl.INDEX)
                pool_cache[dest : dest + 1, :] = staged_i8[ei_idx : ei_idx + 1, :]
                pool_scale[dest : dest + 1, :] = scale_pad[ei_idx : ei_idx + 1, :]
    return pool_tid


@pl.jit
def indexer_cache_write_test(
    index_k: pl.Tensor[[T_DYN, INDEX_DIM], pl.BF16],
    gate_scores: pl.Tensor[[T_DYN, INDEX_DIM], pl.FP32],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32]],
):
    """Run one raw scatter for golden.run validation."""
    index_k.bind_dynamic(0, T_DYN)
    gate_scores.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    indexer_cache_write(index_k, gate_scores, slots, cache)
    return cache


@pl.jit
def indexer_pool_write_test(
    raw_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, INDEX_STATE_WIDTH], pl.FP32],
    compress_ape: pl.Tensor[[INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    pool_token_slots: pl.Tensor[[POOLS_DYN, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[POOLS_DYN], pl.INT32],
    pool_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, INDEX_DIM], pl.INT8]],
    pool_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN * INDEX_STATE_BLOCK_SIZE, POOL_SCALE_WIDTH], pl.FP32]],
    pool_valid: pl.Out[pl.Tensor[[POOLS_DYN, POOL_VALID_WIDTH], pl.INT32]],
):
    """Close every pool event of one dispatch for golden.run validation."""
    pool_token_slots.bind_dynamic(0, POOLS_DYN)
    pool_slots.bind_dynamic(0, POOLS_DYN)
    pool_valid.bind_dynamic(0, POOLS_DYN)
    indexer_pool_write(
        raw_cache, compress_ape, hadamard, pool_token_slots, pool_slots, pool_cache, pool_scale, pool_valid
    )
    return pool_cache, pool_scale, pool_valid


def build_indexer_cache_write_specs(tokens: int = 24, pages: int = 4):
    """Build one deterministic raw scatter: unordered distinct slots and a padded row.

    Row 3 carries ``-1``, and the destination rows are deliberately out of order
    across pages so the write cannot pass by walking the pool linearly. ``cache`` is
    ``InOut`` so its untouched rows are uploaded and can be asserted, rather than
    read back as allocator residue.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(59)
    cache_rows = pages * BLOCK_SIZE

    def init_index_k():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32).bfloat16()

    def init_gate_scores():
        return torch.randn(tokens, INDEX_DIM, generator=generator, dtype=torch.float32)

    def init_cache():
        return torch.randn(cache_rows, INDEX_STATE_WIDTH, generator=generator, dtype=torch.float32)

    def init_slots():
        chosen = torch.randperm(cache_rows, generator=generator)[:tokens]
        slots = chosen.to(torch.int32)
        slots[3] = -1
        return slots

    return [
        TensorSpec("index_k", [tokens, INDEX_DIM], torch.bfloat16, init_value=init_index_k),
        TensorSpec("gate_scores", [tokens, INDEX_DIM], torch.float32, init_value=init_gate_scores),
        TensorSpec("slots", [tokens], torch.int32, init_value=init_slots),
        TensorSpec("cache", [cache_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_cache),
    ]


def build_indexer_pool_write_specs(events: int = 6, raw_pages: int = 2, pool_pages: int = 8):
    """Build one deterministic pool-close batch: fresh events and one padded pool.

    Member slots are grouped four-per-event out of consecutive raw rows so the
    golden's pooling window is unambiguous, the destinations are a permutation of
    pooled rows, event 0's members are forced onto physical rows 0-3 (a slot of 0
    is valid — the regression corner of the validity flag) and event 4 carries a
    ``-1`` member to exercise the padded-pool rejection path. The raw table is
    built as the cache write would leave it: BF16 keys widened to FP32 beside
    FP32 gates.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(61)
    raw_rows = raw_pages * BLOCK_SIZE
    pool_rows = pool_pages * INDEX_STATE_BLOCK_SIZE
    member_rows = events * INDEX_KPOOL
    if member_rows > raw_rows or events > pool_rows:
        raise ValueError("fixture sizes exceed the requested pool geometry")

    def init_raw_cache():
        keys = torch.randn(raw_rows, INDEX_DIM, generator=generator)
        keys = keys.bfloat16().float()
        gates = torch.randn(raw_rows, INDEX_DIM, generator=generator)
        return torch.cat([keys, gates], dim=-1)

    def init_compress_ape():
        return torch.randn(INDEX_KPOOL, INDEX_DIM, generator=generator).bfloat16()

    def init_pool_token_slots():
        base = torch.randperm(raw_rows - INDEX_KPOOL + 1, generator=generator)[:events]
        offsets = torch.arange(INDEX_KPOOL)
        slots = (base.unsqueeze(1) + offsets).to(torch.int32)
        # Event 0's members are physical rows 0-3: row 0 is a legal slot, and a
        # flag derived by clamping the slot value would report it invalid.
        slots[0] = torch.arange(INDEX_KPOOL, dtype=torch.int32)
        slots[4, 3] = -1
        return slots

    def init_pool_slots():
        chosen = torch.randperm(pool_rows, generator=generator)[:events]
        return chosen.to(torch.int32)

    # One draw feeds both tables so the prior INT8 rows and their scales match;
    # the rows read as already Hadamard-rotated prior state, the way earlier
    # pool writes would have left them.
    prior_i8, prior_scale_col = quantize_per_token_int8(
        torch.randn(pool_rows, INDEX_DIM, generator=generator, dtype=torch.float32)
    )

    def init_pool_cache():
        return prior_i8

    def init_pool_scale():
        scale = torch.zeros(pool_rows, POOL_SCALE_WIDTH)
        scale[:, 0] = prior_scale_col[:, 0]
        return scale

    return [
        TensorSpec("raw_cache", [raw_rows, INDEX_STATE_WIDTH], torch.float32, init_value=init_raw_cache),
        TensorSpec("compress_ape", [INDEX_KPOOL, INDEX_DIM], torch.bfloat16, init_value=init_compress_ape),
        TensorSpec("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16, init_value=sylvester_hadamard),
        TensorSpec(
            "pool_token_slots",
            [events, INDEX_KPOOL],
            torch.int32,
            init_value=init_pool_token_slots,
        ),
        TensorSpec("pool_slots", [events], torch.int32, init_value=init_pool_slots),
        TensorSpec("pool_cache", [pool_rows, INDEX_DIM], torch.int8, init_value=init_pool_cache),
        TensorSpec("pool_scale", [pool_rows, POOL_SCALE_WIDTH], torch.float32, init_value=init_pool_scale),
        TensorSpec("pool_valid", [events, POOL_VALID_WIDTH], torch.int32),
    ]


def golden_indexer_cache_write_case(tensors):
    """Fill the expected raw pool state for :func:`build_indexer_cache_write_specs`."""
    tensors["cache"][:] = golden_indexer_cache_write(
        tensors["cache"], tensors["index_k"], tensors["gate_scores"], tensors["slots"]
    )


def golden_indexer_pool_write_case(tensors):
    """Fill the expected pooled tables and validity for :func:`build_indexer_pool_write_specs`."""
    pool_cache, pool_scale, pool_valid = golden_indexer_pool_write(
        tensors["pool_cache"],
        tensors["pool_scale"],
        tensors["raw_cache"],
        tensors["compress_ape"],
        tensors["hadamard"],
        tensors["pool_token_slots"],
        tensors["pool_slots"],
    )
    tensors["pool_cache"][:] = pool_cache
    tensors["pool_scale"][:] = pool_scale
    tensors["pool_valid"][:] = pool_valid


def _self_check() -> None:
    """Prove both goldens on CPU with inlined fixtures, before any device run.

    Kept local rather than in ``_golden_smoke.py`` because that module is shared
    across the whole model directory.
    """
    cache = torch.randn(4 * BLOCK_SIZE, INDEX_STATE_WIDTH)
    index_k = torch.randn(8, INDEX_DIM).bfloat16()
    gates = torch.randn(8, INDEX_DIM)
    slots = torch.tensor([5, 0, -1, 11, 300, 128, 257, 3], dtype=torch.int32)
    updated = golden_indexer_cache_write(cache, index_k, gates, slots)
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            expected = torch.cat([index_k[row].float(), gates[row]], dim=-1)
            assert torch.equal(updated[slot], expected), f"slot {slot} did not receive its row"
    assert torch.equal(updated[-1], cache[-1]), "a -1 slot wrapped into the last row"

    pool_i8, pool_scale_col = quantize_per_token_int8(torch.randn(8, INDEX_DIM))
    pool_cache = pool_i8
    pool_scale = torch.zeros(8, POOL_SCALE_WIDTH)
    pool_scale[:, 0] = pool_scale_col[:, 0]
    raw_cache = torch.cat(
        [
            torch.randn(32, INDEX_DIM).bfloat16().float(),
            torch.randn(32, INDEX_DIM),
        ],
        dim=-1,
    )
    ape = torch.randn(INDEX_KPOOL, INDEX_DIM).bfloat16()
    hadamard = sylvester_hadamard()
    pool_token_slots = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, -1, 15], [16, 17, 18, 19]],
        dtype=torch.int32,
    )
    pool_slots = torch.tensor([7, 3, 0, 5, 2], dtype=torch.int32)
    new_cache, new_scale, valid = golden_indexer_pool_write(
        pool_cache, pool_scale, raw_cache, ape, hadamard, pool_token_slots, pool_slots
    )
    assert valid[:, 0].tolist() == [1, 1, 1, 0, 1], valid
    members = pool_token_slots[0].long()
    weights = torch.softmax(raw_cache[members, INDEX_DIM:].float() + ape.float(), dim=0)
    expected = (weights * raw_cache[members, 0:INDEX_DIM].float()).sum(dim=0, keepdim=True)
    expected = expected.to(torch.bfloat16).float() @ hadamard.float()
    expected_i8, expected_scale = quantize_per_token_int8(expected)
    assert torch.equal(new_cache[7], expected_i8[0]), "pool 0 INT8 bytes differ"
    assert (new_scale[7] == expected_scale[0, 0]).all(), "pool 0 scale lanes differ"
    zero_i8, zero_scale = quantize_per_token_int8(torch.zeros(1, INDEX_DIM))
    assert torch.equal(new_cache[5], zero_i8[0].to(torch.int8)), "padded pool is not quantized zero"
    assert (new_scale[5] == zero_scale[0, 0]).all(), "padded pool scale lanes differ"
    assert torch.equal(new_cache[1], pool_cache[1]), "an untouched pooled row changed"
    assert torch.equal(new_scale[1], pool_scale[1]), "an untouched scale row changed"
    print("[GOLDEN] PASS indexer_cache self-check")


def main():
    """Prove the goldens on CPU, then validate both writers on device.

    The raw scatter is pure data movement, so it runs at zero tolerance. The
    pooled key is an FP32 softmax quantized once to INT8 plus a dequant scale;
    device and reference differ only in the softmax transcendentals, which
    almost always lands on the same quantized byte — the budget allows a tiny
    tail of one-quantum flips at rint boundaries.
    """
    import argparse

    from golden import ratio_allclose, run

    _self_check()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", default="both", choices=["write", "pool", "both"])
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--events", type=int, default=6)
    parser.add_argument("--raw-pages", type=int, default=2)
    parser.add_argument("--pool-pages", type=int, default=8)
    parser.add_argument(
        "--bench",
        action="store_true",
        help="skip golden validation; shapes come from --tokens/--pages/--events, timing from PYPTO_BENCH=1",
    )
    args = parser.parse_args()

    def print_bench(label: str, result) -> None:
        stats = result.bench
        if stats is None:
            print(f"[BENCH] {label}: no timing (run with PYPTO_BENCH=1)")
            return
        print(
            f"[BENCH] {label}: device_us median={stats.device_us_median:.1f} "
            f"min={stats.device_us_min:.1f} mean={stats.device_us_mean:.1f} "
            f"max={stats.device_us_max:.1f} rounds={stats.rounds}"
        )

    results = []
    if args.case in ("write", "both"):
        results.append(
            run(
                fn=indexer_cache_write_test,
                specs=build_indexer_cache_write_specs(args.tokens, args.pages),
                golden_fn=None if args.bench else golden_indexer_cache_write_case,
                config={"platform": args.platform, "device_id": args.device},
                rtol=0.0,
                atol=0.0,
                compile_only=args.compile_only,
            )
        )
        if args.bench:
            print_bench(f"cache_write tokens={args.tokens} pages={args.pages}", results[-1])
    if args.case in ("pool", "both"):
        results.append(
            run(
                fn=indexer_pool_write_test,
                specs=build_indexer_pool_write_specs(args.events, args.raw_pages, args.pool_pages),
                golden_fn=None if args.bench else golden_indexer_pool_write_case,
                config={"platform": args.platform, "device_id": args.device},
                rtol=1.0 / 64,
                atol=1e-3,
                # The quantization chain itself is deterministic, but the pooled
                # FP32 key differs from the reference by one exp() ulp, which can
                # flip a rint boundary: allow a tiny tail of one-quantum flips.
                compare_fn=None
                if args.bench
                else {
                    "pool_cache": ratio_allclose(atol=1.0, rtol=0.0, max_error_ratio=0.005),
                    "pool_scale": ratio_allclose(atol=1e-6, rtol=1e-3, max_error_ratio=0.005),
                },
                compile_only=args.compile_only,
            )
        )
        if args.bench:
            print_bench(
                f"pool_write events={args.events} raw_pages={args.raw_pages} pool_pages={args.pool_pages}",
                results[-1],
            )
    for result in results:
        print(result)
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "build_indexer_cache_write_specs",
    "build_indexer_pool_write_specs",
    "golden_indexer_cache_write",
    "golden_indexer_cache_write_case",
    "golden_indexer_pool_write",
    "golden_indexer_pool_write_case",
    "indexer_cache_write",
    "indexer_cache_write_test",
    "indexer_pool_write",
    "indexer_pool_write_test",
    "sylvester_hadamard",
]


if __name__ == "__main__":
    main()
