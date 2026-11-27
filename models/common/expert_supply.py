# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-side cache planning and mapped packed-field transfer for routed experts."""

from pathlib import Path

import pypto.language as pl

_NATIVE = Path(__file__).parent


def mapped_copy_workers(field_bytes: int) -> int:
    """Size one field's SPMD copy group, keeping small scales on a single AIV."""
    return max(1, min(16, (field_bytes + (256 << 10) - 1) // (256 << 10)))


@pl.jit.extern(core_type="aiv", source=str(_NATIVE / "mapped_expert_plan.cpp"))
def mapped_expert_plan(
    counts: pl.Tensor,
    state: pl.InOut[pl.Tensor],
    slot_map: pl.Out[pl.Tensor],
    misses: pl.Out[pl.Tensor],
    layer: pl.Scalar[pl.INDEX],
    first: pl.Scalar[pl.INDEX],
    stop: pl.Scalar[pl.INDEX],
) -> tuple[pl.Tensor, pl.Tensor, pl.Tensor]:
    """Plan one wave whose positive local counts fit in the slot pool.

    State is initialized by initial_expert_cache: an LRU sentinel, slot rows
    (layer, local expert, previous, next, hits, misses), then directory words.
    The map and miss destinations are int32 expert vectors for [first, stop).
    Submit with dependencies on the preceding wave's final weight readers.
    """
    return state, slot_map, misses


@pl.jit.extern(core_type="aiv", source=str(_NATIVE / "mapped_expert_copy.cpp"))
def mapped_expert_copy(
    field_desc: pl.Tensor,
    misses: pl.Tensor,
    bank: pl.InOut[pl.Tensor],
    first: pl.Scalar[pl.INDEX],
    stop: pl.Scalar[pl.INDEX],
    bank_origin: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Supply one typed bank; pass this result to every weight reader.

    Descriptor int64 words are the mapped source base, source expert stride,
    packed field bytes, and destination slot stride. Extents are byte counts.
    Pass zero for a full bank or its first physical slot for a typed subview.
    For a single expert, set stop to -1 and first to the expert index.
    Use pl.spmd_submit with mapped_copy_workers(field_bytes) for large fields.
    """
    return bank


@pl.jit.inline(auto_scope=False)
def supply_packed_fields(
    descriptors: pl.Tensor, misses: pl.Tensor,
    w1: pl.Tensor, s1: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor,
    layer: pl.Scalar[pl.INDEX], first: pl.Scalar[pl.INDEX], stop: pl.Scalar[pl.INDEX],
    origin: pl.Scalar[pl.INDEX], planned: pl.Scalar[pl.TASK_ID],
    weight_workers: pl.constexpr, scale_workers: pl.constexpr,
):
    """Publish three packed weight/scale pairs in descriptor order to their typed bank views."""
    w1, c1 = pl.spmd_submit(mapped_expert_copy, descriptors[layer, 0], misses, w1, first, stop, origin,
                            core_num=weight_workers, deps=[planned])
    s1, c2 = pl.spmd_submit(mapped_expert_copy, descriptors[layer, 1], misses, s1, first, stop, origin,
                            core_num=scale_workers, deps=[planned])
    w2, c3 = pl.spmd_submit(mapped_expert_copy, descriptors[layer, 2], misses, w2, first, stop, origin,
                            core_num=weight_workers, deps=[planned])
    s2, c4 = pl.spmd_submit(mapped_expert_copy, descriptors[layer, 3], misses, s2, first, stop, origin,
                            core_num=scale_workers, deps=[planned])
    w3, c5 = pl.spmd_submit(mapped_expert_copy, descriptors[layer, 4], misses, w3, first, stop, origin,
                            core_num=weight_workers, deps=[planned])
    s3, c6 = pl.spmd_submit(mapped_expert_copy, descriptors[layer, 5], misses, s3, first, stop, origin,
                            core_num=scale_workers, deps=[planned])
    ready = pl.system.task_dummy(deps=[c1, c2, c3, c4, c5, c6])
    return w1, s1, w2, s2, w3, s3, ready


@pl.jit.inline(auto_scope=False)
def supply_packed_wave(
    counts: pl.Tensor, state: pl.Tensor, descriptors: pl.Tensor, slots: pl.Tensor, misses: pl.Tensor,
    w1: pl.Tensor, s1: pl.Tensor, w2: pl.Tensor, s2: pl.Tensor, w3: pl.Tensor, s3: pl.Tensor,
    layer: pl.Scalar[pl.INDEX], first: pl.Scalar[pl.INDEX], stop: pl.Scalar[pl.INDEX],
    inputs_ready: pl.Scalar[pl.TASK_ID], weight_workers: pl.constexpr, scale_workers: pl.constexpr,
):
    """Plan and supply a local expert range after its counts and previous weight readers are ready."""
    (state, slots, misses), planned = pl.submit(
        mapped_expert_plan, counts, state, slots, misses, layer, first, stop, deps=[inputs_ready])
    w1, s1, w2, s2, w3, s3, ready = supply_packed_fields(
        descriptors, misses, w1, s1, w2, s2, w3, s3, layer, first, stop, 0,
        planned, weight_workers, scale_workers)
    return slots, w1, s1, w2, s2, w3, s3, ready
