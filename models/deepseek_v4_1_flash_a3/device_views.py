# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Typed views into shared resident device allocations."""


def _device_view(allocation, value, base_offset=0):
    """Bind a contiguous typed view to the allocation's owning Buffer."""
    from pypto.runtime import DeviceTensor

    offset = value.storage_offset() * value.element_size() - base_offset
    if offset < 0:
        raise ValueError("Resident tensor view begins before its allocation")
    if offset + value.numel() * value.element_size() > allocation.nbytes:
        raise ValueError("Resident tensor view exceeds its allocation")
    return DeviceTensor(allocation.data_ptr + offset, value.shape, value.dtype,
                        buffer=allocation.buffer, byte_offset=allocation.byte_offset + offset)


def resident_graph_argument(shards, worker_ids, parameter):
    """Bind a rank stack or a rank-major NZ bank without copying its shards."""
    from pypto.runtime import StackedDeviceTensor

    stacked = StackedDeviceTensor(shards, (len(shards), *shards[0].shape), worker_ids)
    physical_shape = tuple(parameter.shape)
    if parameter.layout != "NZ" or physical_shape == stacked.full_shape:
        return stacked

    ranks, matrices, *tiles = stacked.full_shape
    if len(tiles) != 4 or physical_shape != (ranks * matrices, *tiles):
        raise ValueError(f"Cannot bind resident NZ shape {stacked.full_shape} to graph shape {physical_shape}")

    class RankMajorNZBank(StackedDeviceTensor):
        def __getitem__(self, index):
            if not isinstance(index, tuple) or len(index) != 5:
                raise IndexError(f"Expected a whole NZ bank slice, got {index!r}")
            head, *tail = index
            if (not isinstance(head, slice) or head.step not in (None, 1) or head.start is None
                    or head.start < 0 or head.start % matrices or head.stop != head.start + matrices
                    or head.stop > ranks * matrices
                    or any(part not in (slice(None), slice(0, dim)) for part, dim in zip(tail, tiles, strict=True))):
                raise IndexError(f"Expected an aligned NZ slice for one rank, got {index!r}")
            return self.shards[head.start // matrices]

    bank = RankMajorNZBank(stacked.shards, stacked.full_shape, stacked.worker_ids)
    # The runner validates the logical shape; orchestration indexes the flat physical bank.
    n_blocks, m_blocks, m0, n0 = tiles
    object.__setattr__(bank, "full_shape", (ranks, matrices, m_blocks * m0, n_blocks * n0))
    return bank
