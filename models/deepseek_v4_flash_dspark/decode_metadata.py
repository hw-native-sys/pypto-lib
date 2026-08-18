# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-side metadata lowering for DeepSeek-V4 decode.

``utils`` holds the host-side torch counterpart used by the per-kernel test
fixtures.
"""


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

try:
    import pypto.language as pl  # device lowering; only available on Serving-B
except ImportError:  # pragma: no cover - host-side contract gate has no pypto
    # Pure-Python host reference path (Phase A ragged metadata, contract test)
    # must run without a device toolchain. The fallback makes the device-only
    # decorators and tensor-annotation symbols no-ops; the decorated device
    # kernels are never invoked on the host. Only the Phase A pure-Python
    # ragged-metadata builders / cases are reachable in this mode.
    class _NoPl:
        class _Jit:
            """Decorator stub: ``@pl.jit.inline`` and ``@pl.jit`` both work."""

            @staticmethod
            def inline(fn):
                return fn

            def __call__(self, fn):  # pragma: no cover
                return fn

        jit = _Jit()
        del _Jit

        class _Annotation:
            """Accepts ``pl.Tensor[[shape], dtype]`` and ``pl.Out[...]`` usage
            as annotations without constructing anything."""

            def __getitem__(self, _):
                return self

        Tensor = _Annotation()
        Out = _Annotation()
        INT32 = INT64 = INT8 = FP32 = BF16 = None
        INDEX = None
        dynamic = staticmethod(lambda _name: 1)
        spmd = staticmethod(lambda *a, **k: None)
        read = staticmethod(lambda *a, **k: None)
        write = staticmethod(lambda *a, **k: None)
        cast = staticmethod(lambda *a, **k: None)
        min = staticmethod(lambda *a, **k: None)
        max = staticmethod(lambda *a, **k: None)
        create_tensor = staticmethod(lambda *a, **k: None)
        full = staticmethod(lambda *a, **k: None)
        range = staticmethod(lambda *a, **k: None)

    pl = _NoPl()  # type: ignore[assignment]

from config import (
    BLOCK_SIZE,
    CACHE_BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    CSA_CANDIDATES_PER_LEAF,
    CSA_COMPRESS_RATIO,
    CSA_INNER_STATE_BLOCK_SIZE,
    CSA_INNER_STATE_PAGES_PER_REQUEST,
    CSA_INNER_STATE_ROWS_PER_REQUEST,
    CSA_MAX_LEAVES_PER_QUERY,
    CSA_MAX_NODES_PER_QUERY,
    CSA_MAX_QUERIES,
    CSA_MAX_TOPK_TASKS,
    CSA_MERGE_ARITY,
    CSA_STATE_BLOCK_SIZE,
    CSA_STATE_PAGES_PER_REQUEST,
    CSA_STATE_ROWS_PER_REQUEST,
    CSA_TOPK_INVALID_TASK_SLOT,
    CSA_TOPK_READY_FRONTIER_W,
    DECODE_LOCAL_REQUESTS,
    DECODE_SEQ,
    FLASH as M,
    HCA_COMPRESS_RATIO,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_PAGES_PER_REQUEST,
    HCA_STATE_ROWS_PER_REQUEST,
    HCA_ROWS_PER_SHARD,
    IDX_CACHE_MAX_BLOCKS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_MAX_BLOCKS,
    MAX_CSA_CANDIDATES,
    MAX_HCA_ROWS,
    SWA_PERSISTENT_PAGES_PER_REQUEST,
    SWA_PERSISTENT_ROWS_PER_REQUEST,
    SWA_SOURCE_INVALID,
    SWA_SOURCE_INT32_MAX,
    SWA_SOURCE_MAX_OVERLAY_QUERY,
    SWA_SOURCE_OVERLAY_BASE,
    SWA_WINDOW_ROWS,
    encode_swa_overlay_source,
)


B = DECODE_LOCAL_REQUESTS
S = DECODE_SEQ
T = B * S
assert S == 8
WIN = M.sliding_window
ORI_MAX_BLOCKS = KV_ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS
IDX_MAX_BLOCKS = IDX_CACHE_MAX_BLOCKS
HCA_STATE_MAX_BLOCKS = 2048
CSA_STATE_MAX_BLOCKS = 4096
CSA_INNER_STATE_MAX_BLOCKS = 4096

GROUP_ORI = 0
GROUP_CMP = 1
GROUP_IDX = 2
GROUP_HCA_STATE = 3
GROUP_CSA_STATE = 4
GROUP_CSA_INNER_STATE = 5
N_CACHE_GROUPS = 6


@pl.jit.inline
def build_swa_metadata(
    # Inputs: bare Tensor parameters have PyPTO's default In direction.
    position_ids: pl.Tensor[[T], pl.INT32],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    # Outputs.
    swa_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_indices: pl.Out[pl.Tensor[[T, WIN], pl.INT32]],
    swa_lens: pl.Out[pl.Tensor[[T], pl.INT32]],
):
    """Lower paged write slots and visible SWA rows for each decode token."""
    for token in pl.spmd(T, name_hint="decode_build_swa_metadata"):
        request = token // S
        position = pl.read(position_ids, [token])
        valid_len = pl.min(position + 1, WIN)
        start = position - valid_len + 1
        index_row = pl.create_tensor([1, WIN], dtype=pl.INT32)
        index_row[:, :] = pl.full([1, WIN], dtype=pl.INT32, value=-1)
        for offset in pl.range(WIN):
            if offset < valid_len:
                visible_position = start + offset
                visible_block = visible_position // BLOCK_SIZE
                visible_offset = visible_position % BLOCK_SIZE
                visible_physical_block = pl.read(
                    ori_block_table,
                    [request, pl.cast(visible_block, pl.INDEX)],
                )
                pl.write(
                    index_row,
                    [0, offset],
                    pl.cast(
                        visible_physical_block * BLOCK_SIZE + visible_offset,
                        pl.INT32,
                    ),
                )
        swa_indices[token : token + 1, :] = index_row

    for metadata_core in pl.spmd(1, name_hint="decode_build_swa_scalar_metadata"):
        for token in pl.range(metadata_core, T):
            request = token // S
            position = pl.read(position_ids, [token])
            logical_block = position // BLOCK_SIZE
            block_offset = position % BLOCK_SIZE
            physical_block = pl.read(
                ori_block_table,
                [request, pl.cast(logical_block, pl.INDEX)],
            )
            pl.write(
                swa_slot_mapping,
                [token],
                pl.cast(physical_block * BLOCK_SIZE + block_offset, pl.INT64),
            )
            pl.write(
                swa_lens,
                [token],
                pl.cast(pl.min(position + 1, WIN), pl.INT32),
            )


@pl.jit.inline
def build_decode_metadata(
    # Inputs: bare Tensor parameters have PyPTO's default In direction.
    position_ids: pl.Tensor[[T], pl.INT32],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[B, IDX_MAX_BLOCKS], pl.INT32],
    hca_state_block_table: pl.Tensor[[B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_state_block_table: pl.Tensor[[B, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_state_block_table: pl.Tensor[
        [B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    block_counts: pl.Tensor[[B, N_CACHE_GROUPS], pl.INT32],
    # Outputs.
    ori_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_indices: pl.Out[pl.Tensor[[T, WIN], pl.INT32]],
    swa_lens: pl.Out[pl.Tensor[[T], pl.INT32]],
    hca_cmp_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    hca_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_cmp_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_idx_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_inner_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
):
    """Build every position-dependent metadata tensor the decode path consumes."""
    build_swa_metadata(
        position_ids,
        ori_block_table,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
    )
    for metadata_core in pl.spmd(1, name_hint="decode_build_cache_metadata"):
        for token in pl.range(metadata_core, T):
            request = token // S
            position = pl.read(position_ids, [token])
            logical_block = position // BLOCK_SIZE
            block_offset = position % BLOCK_SIZE
            ori_physical_block = pl.read(
                ori_block_table,
                [request, pl.cast(logical_block, pl.INDEX)],
            )
            pl.write(
                ori_slot_mapping,
                [token],
                pl.cast(ori_physical_block * BLOCK_SIZE + block_offset, pl.INT64),
            )

            hca_cmp_slot = pl.cast(-1, pl.INT64)
            if (position + 1) % 128 == 0:
                logical = position // 128
                count = pl.read(block_counts, [request, GROUP_CMP])
                physical_block = pl.read(
                    cmp_block_table,
                    [
                        request,
                        pl.cast(logical // BLOCK_SIZE % count, pl.INDEX),
                    ],
                )
                hca_cmp_slot = pl.cast(
                    physical_block * BLOCK_SIZE + logical % BLOCK_SIZE,
                    pl.INT64,
                )
            pl.write(hca_cmp_slot_mapping, [token], hca_cmp_slot)

            csa_cmp_slot = pl.cast(-1, pl.INT64)
            csa_idx_slot = pl.cast(-1, pl.INT64)
            if (position + 1) % 4 == 0:
                logical = position // 4
                cmp_count = pl.read(block_counts, [request, GROUP_CMP])
                cmp_physical_block = pl.read(
                    cmp_block_table,
                    [request, pl.cast(logical // BLOCK_SIZE % cmp_count, pl.INDEX)],
                )
                csa_cmp_slot = pl.cast(
                    cmp_physical_block * BLOCK_SIZE + logical % BLOCK_SIZE,
                    pl.INT64,
                )
                idx_count = pl.read(block_counts, [request, GROUP_IDX])
                idx_physical_block = pl.read(
                    idx_block_table,
                    [request, pl.cast(logical // BLOCK_SIZE % idx_count, pl.INDEX)],
                )
                csa_idx_slot = pl.cast(
                    idx_physical_block * BLOCK_SIZE + logical % BLOCK_SIZE,
                    pl.INT64,
                )
            pl.write(csa_cmp_slot_mapping, [token], csa_cmp_slot)
            pl.write(csa_idx_slot_mapping, [token], csa_idx_slot)

            hca_state_logical = position // C128_COMPRESSOR_BLOCK_SIZE
            hca_state_count = pl.read(block_counts, [request, GROUP_HCA_STATE])
            hca_state_physical_block = pl.read(
                hca_state_block_table,
                [
                    request,
                    pl.cast(hca_state_logical % hca_state_count, pl.INDEX),
                ],
            )
            pl.write(
                hca_state_slot_mapping,
                [token],
                pl.cast(
                    hca_state_physical_block * C128_COMPRESSOR_BLOCK_SIZE
                    + position % C128_COMPRESSOR_BLOCK_SIZE,
                    pl.INT64,
                ),
            )

            csa_state_logical = position // C4A_COMPRESSOR_BLOCK_SIZE
            csa_state_count = pl.read(block_counts, [request, GROUP_CSA_STATE])
            csa_state_physical_block = pl.read(
                csa_state_block_table,
                [
                    request,
                    pl.cast(csa_state_logical % csa_state_count, pl.INDEX),
                ],
            )
            pl.write(
                csa_state_slot_mapping,
                [token],
                pl.cast(
                    csa_state_physical_block * C4A_COMPRESSOR_BLOCK_SIZE
                    + position % C4A_COMPRESSOR_BLOCK_SIZE,
                    pl.INT64,
                ),
            )

            inner_state_count = pl.read(
                block_counts,
                [request, GROUP_CSA_INNER_STATE],
            )
            inner_state_physical_block = pl.read(
                csa_inner_state_block_table,
                [
                    request,
                    pl.cast(csa_state_logical % inner_state_count, pl.INDEX),
                ],
            )
            pl.write(
                csa_inner_state_slot_mapping,
                [token],
                pl.cast(
                    inner_state_physical_block * C4A_COMPRESSOR_BLOCK_SIZE
                    + position % C4A_COMPRESSOR_BLOCK_SIZE,
                    pl.INT64,
                ),
            )
    return (
        ori_slot_mapping,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
    )


@pl.jit
def decode_metadata(
    position_ids: pl.Tensor[[T], pl.INT32],
    ori_block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[B, IDX_MAX_BLOCKS], pl.INT32],
    hca_state_block_table: pl.Tensor[[B, HCA_STATE_MAX_BLOCKS], pl.INT32],
    csa_state_block_table: pl.Tensor[[B, CSA_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_state_block_table: pl.Tensor[
        [B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    block_counts: pl.Tensor[[B, N_CACHE_GROUPS], pl.INT32],
    ori_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    swa_indices: pl.Out[pl.Tensor[[T, WIN], pl.INT32]],
    swa_lens: pl.Out[pl.Tensor[[T], pl.INT32]],
    hca_cmp_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    hca_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_cmp_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_idx_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
    csa_inner_state_slot_mapping: pl.Out[pl.Tensor[[T], pl.INT64]],
):
    """Standalone validation entry for device metadata lowering."""
    return build_decode_metadata(
        position_ids,
        ori_block_table,
        cmp_block_table,
        idx_block_table,
        hca_state_block_table,
        csa_state_block_table,
        csa_inner_state_block_table,
        block_counts,
        ori_slot_mapping,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
    )


def _test_inputs():
    import torch

    # Four request regimes of S=2 consecutive positions, tiled up to B requests.
    position_regimes = torch.tensor(
        [126, 127, 3, 4, 8191, 8192, 16382, 16383],
        dtype=torch.int32,
    )
    positions = position_regimes.repeat((T + position_regimes.numel() - 1) // position_regimes.numel())[:T]
    count_regimes = torch.tensor(
        [
            [2, 3, 4, 5, 6, 7],
            [3, 4, 5, 6, 7, 8],
            [4, 5, 6, 7, 8, 9],
            [5, 6, 7, 8, 9, 10],
        ],
        dtype=torch.int32,
    )
    counts = count_regimes.repeat((B + count_regimes.shape[0] - 1) // count_regimes.shape[0], 1)[:B]

    def table(width, group, *, repeat):
        out = torch.zeros((B, width), dtype=torch.int32)
        for request in range(B):
            count = int(counts[request, group])
            ids = torch.arange(count, dtype=torch.int32) + 1000 * (group + 1) + 100 * request
            if repeat:
                out[request] = ids.repeat((width + count - 1) // count)[:width]
            else:
                out[request, :count] = ids
        return out

    return {
        "position_ids": positions,
        "ori_block_table": table(ORI_MAX_BLOCKS, GROUP_ORI, repeat=True),
        "cmp_block_table": table(CMP_MAX_BLOCKS, GROUP_CMP, repeat=False),
        "idx_block_table": table(IDX_MAX_BLOCKS, GROUP_IDX, repeat=False),
        "hca_state_block_table": table(
            HCA_STATE_MAX_BLOCKS,
            GROUP_HCA_STATE,
            repeat=True,
        ),
        "csa_state_block_table": table(
            CSA_STATE_MAX_BLOCKS,
            GROUP_CSA_STATE,
            repeat=True,
        ),
        "csa_inner_state_block_table": table(
            CSA_INNER_STATE_MAX_BLOCKS,
            GROUP_CSA_INNER_STATE,
            repeat=True,
        ),
        "block_counts": counts,
    }


def golden_decode_metadata(tensors):
    positions = tensors["position_ids"]
    ori_table = tensors["ori_block_table"]
    cmp_table = tensors["cmp_block_table"]
    idx_table = tensors["idx_block_table"]
    hca_state_table = tensors["hca_state_block_table"]
    csa_state_table = tensors["csa_state_block_table"]
    inner_state_table = tensors["csa_inner_state_block_table"]
    counts = tensors["block_counts"]

    tensors["swa_indices"].fill_(-1)
    for token in range(T):
        request = token // S
        position = int(positions[token])
        logical_block, block_offset = divmod(position, BLOCK_SIZE)
        tensors["swa_slot_mapping"][token] = (
            int(ori_table[request, logical_block]) * BLOCK_SIZE + block_offset
        )
        valid_len = min(position + 1, WIN)
        start = position - valid_len + 1
        tensors["swa_lens"][token] = valid_len
        for offset, visible_position in enumerate(range(start, position + 1)):
            visible_block, visible_offset = divmod(visible_position, BLOCK_SIZE)
            tensors["swa_indices"][token, offset] = (
                int(ori_table[request, visible_block]) * BLOCK_SIZE + visible_offset
            )

        tensors["ori_slot_mapping"][token] = (
            int(ori_table[request, logical_block]) * BLOCK_SIZE + block_offset
        )
        tensors["hca_cmp_slot_mapping"][token] = -1
        if (position + 1) % 128 == 0:
            logical = position // 128
            count = int(counts[request, GROUP_CMP])
            block_index, offset = divmod(logical, BLOCK_SIZE)
            tensors["hca_cmp_slot_mapping"][token] = (
                int(cmp_table[request, block_index % count]) * BLOCK_SIZE + offset
            )

        tensors["csa_cmp_slot_mapping"][token] = -1
        tensors["csa_idx_slot_mapping"][token] = -1
        if (position + 1) % 4 == 0:
            logical = position // 4
            block_index, offset = divmod(logical, BLOCK_SIZE)
            cmp_count = int(counts[request, GROUP_CMP])
            idx_count = int(counts[request, GROUP_IDX])
            tensors["csa_cmp_slot_mapping"][token] = (
                int(cmp_table[request, block_index % cmp_count]) * BLOCK_SIZE + offset
            )
            tensors["csa_idx_slot_mapping"][token] = (
                int(idx_table[request, block_index % idx_count]) * BLOCK_SIZE + offset
            )

        hca_block, hca_offset = divmod(position, C128_COMPRESSOR_BLOCK_SIZE)
        hca_count = int(counts[request, GROUP_HCA_STATE])
        tensors["hca_state_slot_mapping"][token] = (
            int(hca_state_table[request, hca_block % hca_count])
            * C128_COMPRESSOR_BLOCK_SIZE
            + hca_offset
        )
        csa_block, csa_offset = divmod(position, C4A_COMPRESSOR_BLOCK_SIZE)
        csa_count = int(counts[request, GROUP_CSA_STATE])
        inner_count = int(counts[request, GROUP_CSA_INNER_STATE])
        tensors["csa_state_slot_mapping"][token] = (
            int(csa_state_table[request, csa_block % csa_count])
            * C4A_COMPRESSOR_BLOCK_SIZE
            + csa_offset
        )
        tensors["csa_inner_state_slot_mapping"][token] = (
            int(inner_state_table[request, csa_block % inner_count])
            * C4A_COMPRESSOR_BLOCK_SIZE
            + csa_offset
        )


def build_tensor_specs():
    import torch
    from golden import TensorSpec

    inputs = _test_inputs()
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in inputs.items()
    ]
    for name, shape, dtype in (
        ("ori_slot_mapping", [T], torch.int64),
        ("swa_slot_mapping", [T], torch.int64),
        ("swa_indices", [T, WIN], torch.int32),
        ("swa_lens", [T], torch.int32),
        ("hca_cmp_slot_mapping", [T], torch.int64),
        ("hca_state_slot_mapping", [T], torch.int64),
        ("csa_cmp_slot_mapping", [T], torch.int64),
        ("csa_idx_slot_mapping", [T], torch.int64),
        ("csa_state_slot_mapping", [T], torch.int64),
        ("csa_inner_state_slot_mapping", [T], torch.int64),
    ):
        specs.append(TensorSpec(name, shape, dtype, is_output=True))
    return specs


# Dynamic Phase A ragged-metadata ABI. Cache-group page records are packed as
# ``[physical_page_id, epoch]`` rows. ``page_offsets[group, request]`` gives
# each request's flat range, and ``page_windows[group, request]`` stores
# ``[valid_begin, valid_end, head]``.
PHASE_A_INVALID_SLOT = -1
PHASE_A_REQUESTS_DYN = pl.dynamic("PHASE_A_REQUESTS_DYN")
PHASE_A_OFFSETS_DYN = pl.dynamic("PHASE_A_OFFSETS_DYN")
PHASE_A_QUERIES_DYN = pl.dynamic("PHASE_A_QUERIES_DYN")
PHASE_A_PAGES_DYN = pl.dynamic("PHASE_A_PAGES_DYN")

PHASE_A_GROUP_HCA = 0
PHASE_A_GROUP_CSA = 1
PHASE_A_GROUP_CSA_IDX = 2
PHASE_A_GROUP_RAW = 3
PHASE_A_GROUP_HCA_STATE = 4
PHASE_A_GROUP_CSA_STATE = 5
PHASE_A_GROUP_CSA_INNER_STATE = 6
PHASE_A_CACHE_GROUPS = 7

PHASE_A_GEOM_VISIBLE_TOKENS = 0
PHASE_A_GEOM_VISIBLE_SWA_ROWS = 1
PHASE_A_GEOM_VISIBLE_HCA_ROWS = 2
PHASE_A_GEOM_NUM_HCA_SHARDS = 3
PHASE_A_GEOM_HCA_TAIL_ROWS = 4
PHASE_A_GEOM_VISIBLE_CSA_CANDIDATES = 5
PHASE_A_GEOM_NUM_CSA_LEAVES = 6
PHASE_A_GEOM_CSA_TAIL_CANDIDATES = 7
PHASE_A_GEOMETRY_FIELDS = 8


@pl.jit.inline
def _phase_a_write_slot(
    pages: pl.Tensor[[PHASE_A_PAGES_DYN, 2], pl.INT32],
    page_offsets: pl.Tensor[
        [PHASE_A_CACHE_GROUPS, PHASE_A_OFFSETS_DYN], pl.INT32
    ],
    page_windows: pl.Tensor[
        [PHASE_A_CACHE_GROUPS, PHASE_A_REQUESTS_DYN, 3], pl.INT32
    ],
    slots: pl.Tensor[
        [PHASE_A_QUERIES_DYN, PHASE_A_CACHE_GROUPS], pl.INT64
    ],
    query: pl.Scalar[pl.INDEX],
    request_index: pl.Scalar[pl.INDEX],
    position: pl.Scalar[pl.INT32],
    active: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
    group: pl.Scalar[pl.INT32],
    compress: pl.Scalar[pl.INT32],
    slot_block_size: pl.Scalar[pl.INT32],
):
    slot = pl.cast(PHASE_A_INVALID_SLOT, pl.INT64)
    if active != 0:
        if (position + 1) % compress == 0:
            logical_slot = position // compress
            valid_begin = pl.read(page_windows, [group, request_index, 0])
            valid_end = pl.read(page_windows, [group, request_index, 1])
            if logical_slot >= valid_begin:
                if logical_slot < valid_end:
                    page_begin = pl.read(page_offsets, [group, request_index])
                    page_end = pl.read(
                        page_offsets, [group, request_index + 1]
                    )
                    page_count = page_end - page_begin
                    if page_count > 0:
                        head = pl.read(page_windows, [group, request_index, 2])
                        if head >= 0:
                            if head < page_count:
                                logical_page_base = valid_begin // slot_block_size
                                relative_page = (
                                    logical_slot // slot_block_size
                                    - logical_page_base
                                )
                                if relative_page >= 0:
                                    if relative_page < page_count:
                                        page_index = page_begin + (
                                            head + relative_page
                                        ) % page_count
                                        page_id = pl.read(
                                            pages,
                                            [pl.cast(page_index, pl.INDEX), 0],
                                        )
                                        page_epoch = pl.read(
                                            pages,
                                            [pl.cast(page_index, pl.INDEX), 1],
                                        )
                                        if page_id >= 0:
                                            if page_epoch == epoch:
                                                slot = (
                                                    pl.cast(page_id, pl.INT64)
                                                    * pl.cast(
                                                        slot_block_size, pl.INT64
                                                    )
                                                    + pl.cast(
                                                        logical_slot
                                                        % slot_block_size,
                                                        pl.INT64,
                                                    )
                                                )
    pl.write(slots, [query, group], slot)
    return slots


@pl.jit
def phase_a_decode_metadata(
    request_state: pl.Tensor[[PHASE_A_REQUESTS_DYN, 3], pl.INT32],
    query_request_ids: pl.Tensor[[PHASE_A_QUERIES_DYN], pl.INT32],
    position_ids: pl.Tensor[[PHASE_A_QUERIES_DYN], pl.INT32],
    pages: pl.Tensor[[PHASE_A_PAGES_DYN, 2], pl.INT32],
    page_offsets: pl.Tensor[[PHASE_A_CACHE_GROUPS, PHASE_A_OFFSETS_DYN], pl.INT32],
    page_windows: pl.Tensor[[PHASE_A_CACHE_GROUPS, PHASE_A_REQUESTS_DYN, 3], pl.INT32],
    geometry: pl.Out[pl.Tensor[[PHASE_A_QUERIES_DYN, PHASE_A_GEOMETRY_FIELDS], pl.INT32]],
    slots: pl.Out[pl.Tensor[[PHASE_A_QUERIES_DYN, PHASE_A_CACHE_GROUPS], pl.INT64]],
):
    """Lower the 1M-capable ragged descriptor on device."""
    request_state.bind_dynamic(0, PHASE_A_REQUESTS_DYN)
    query_request_ids.bind_dynamic(0, PHASE_A_QUERIES_DYN)
    position_ids.bind_dynamic(0, PHASE_A_QUERIES_DYN)
    pages.bind_dynamic(0, PHASE_A_PAGES_DYN)
    page_offsets.bind_dynamic(1, PHASE_A_OFFSETS_DYN)
    page_windows.bind_dynamic(1, PHASE_A_REQUESTS_DYN)
    geometry.bind_dynamic(0, PHASE_A_QUERIES_DYN)
    slots.bind_dynamic(0, PHASE_A_QUERIES_DYN)

    query_count = pl.tensor.dim(position_ids, 0)
    for metadata_core in pl.spmd(1, name_hint="decode_phase_a_ragged_metadata"):
        for query in pl.range(metadata_core, query_count):
            request = pl.read(query_request_ids, [query])
            request_index = pl.cast(request, pl.INDEX)
            position = pl.read(position_ids, [query])
            committed = pl.read(request_state, [request_index, 0])
            active = pl.read(request_state, [request_index, 1])
            epoch = pl.read(request_state, [request_index, 2])

            visible = pl.cast(0, pl.INT32)
            if active != 0:
                visible = pl.cast(
                    pl.min(pl.min(committed, position + 1), M.max_position_embeddings),
                    pl.INT32,
                )
            visible_swa = pl.min(visible, SWA_WINDOW_ROWS)

            hca_page_begin = pl.read(
                page_offsets, [PHASE_A_GROUP_HCA, request_index]
            )
            hca_page_end = pl.read(
                page_offsets, [PHASE_A_GROUP_HCA, request_index + 1]
            )
            hca_allocated = (hca_page_end - hca_page_begin) * CACHE_BLOCK_SIZE
            hca_valid_begin = pl.read(
                page_windows, [PHASE_A_GROUP_HCA, request_index, 0]
            )
            hca_valid_end = pl.read(
                page_windows, [PHASE_A_GROUP_HCA, request_index, 1]
            )
            hca_committed = visible // HCA_COMPRESS_RATIO
            hca_rows = pl.max(
                0, pl.min(hca_committed, hca_valid_end) - hca_valid_begin
            )
            hca_rows = pl.min(hca_rows, hca_allocated)
            hca_shards = (hca_rows + HCA_ROWS_PER_SHARD - 1) // HCA_ROWS_PER_SHARD
            hca_tail = pl.cast(0, pl.INT32)
            if hca_rows > 0:
                hca_tail = pl.cast(hca_rows % HCA_ROWS_PER_SHARD, pl.INT32)
                if hca_tail == 0:
                    hca_tail = pl.cast(HCA_ROWS_PER_SHARD, pl.INT32)

            csa_page_begin = pl.read(
                page_offsets, [PHASE_A_GROUP_CSA, request_index]
            )
            csa_page_end = pl.read(
                page_offsets, [PHASE_A_GROUP_CSA, request_index + 1]
            )
            csa_allocated = (csa_page_end - csa_page_begin) * CACHE_BLOCK_SIZE
            csa_valid_begin = pl.read(
                page_windows, [PHASE_A_GROUP_CSA, request_index, 0]
            )
            csa_valid_end = pl.read(
                page_windows, [PHASE_A_GROUP_CSA, request_index, 1]
            )
            csa_committed = visible // CSA_COMPRESS_RATIO
            csa_candidates = pl.max(
                0, pl.min(csa_committed, csa_valid_end) - csa_valid_begin
            )
            csa_candidates = pl.min(csa_candidates, csa_allocated)
            csa_leaves = (
                csa_candidates + CSA_CANDIDATES_PER_LEAF - 1
            ) // CSA_CANDIDATES_PER_LEAF
            csa_tail = pl.cast(0, pl.INT32)
            if csa_candidates > 0:
                csa_tail = pl.cast(
                    csa_candidates % CSA_CANDIDATES_PER_LEAF, pl.INT32
                )
                if csa_tail == 0:
                    csa_tail = pl.cast(CSA_CANDIDATES_PER_LEAF, pl.INT32)

            pl.write(
                geometry, [query, PHASE_A_GEOM_VISIBLE_TOKENS],
                pl.cast(visible, pl.INT32),
            )
            pl.write(
                geometry, [query, PHASE_A_GEOM_VISIBLE_SWA_ROWS],
                pl.cast(visible_swa, pl.INT32),
            )
            pl.write(
                geometry, [query, PHASE_A_GEOM_VISIBLE_HCA_ROWS],
                pl.cast(hca_rows, pl.INT32),
            )
            pl.write(
                geometry, [query, PHASE_A_GEOM_NUM_HCA_SHARDS],
                pl.cast(hca_shards, pl.INT32),
            )
            pl.write(
                geometry, [query, PHASE_A_GEOM_HCA_TAIL_ROWS],
                pl.cast(hca_tail, pl.INT32),
            )
            pl.write(
                geometry,
                [query, PHASE_A_GEOM_VISIBLE_CSA_CANDIDATES],
                pl.cast(csa_candidates, pl.INT32),
            )
            pl.write(
                geometry, [query, PHASE_A_GEOM_NUM_CSA_LEAVES],
                pl.cast(csa_leaves, pl.INT32),
            )
            pl.write(
                geometry,
                [query, PHASE_A_GEOM_CSA_TAIL_CANDIDATES],
                pl.cast(csa_tail, pl.INT32),
            )

            slots = _phase_a_write_slot(
                pages, page_offsets, page_windows, slots, query,
                request_index, position, active, epoch, PHASE_A_GROUP_HCA,
                HCA_COMPRESS_RATIO, CACHE_BLOCK_SIZE,
            )
            slots = _phase_a_write_slot(
                pages, page_offsets, page_windows, slots, query,
                request_index, position, active, epoch, PHASE_A_GROUP_CSA,
                CSA_COMPRESS_RATIO, CACHE_BLOCK_SIZE,
            )
            slots = _phase_a_write_slot(
                pages, page_offsets, page_windows, slots, query,
                request_index, position, active, epoch, PHASE_A_GROUP_CSA_IDX,
                CSA_COMPRESS_RATIO, CACHE_BLOCK_SIZE,
            )
            slots = _phase_a_write_slot(
                pages, page_offsets, page_windows, slots, query,
                request_index, position, active, epoch, PHASE_A_GROUP_RAW,
                1, CACHE_BLOCK_SIZE,
            )
            slots = _phase_a_write_slot(
                pages, page_offsets, page_windows, slots, query,
                request_index, position, active, epoch, PHASE_A_GROUP_HCA_STATE,
                1, C128_COMPRESSOR_BLOCK_SIZE,
            )
            slots = _phase_a_write_slot(
                pages, page_offsets, page_windows, slots, query,
                request_index, position, active, epoch, PHASE_A_GROUP_CSA_STATE,
                1, C4A_COMPRESSOR_BLOCK_SIZE,
            )
            slots = _phase_a_write_slot(
                pages, page_offsets, page_windows, slots, query,
                request_index, position, active, epoch, PHASE_A_GROUP_CSA_INNER_STATE,
                1, C4A_COMPRESSOR_BLOCK_SIZE,
            )
    return geometry, slots


# ===========================================================================
# Phase A (Run 055): ragged page-map / per-request descriptor metadata.
#
# The legacy ``build_decode_metadata`` above retains the dense
# ``[B, *_MAX_BLOCKS]`` + ``block_counts`` ABI. The ragged descriptor below is
# the canonical 1M-capable metadata lowering for SWA, HCA, and CSA.
#
# Semantics that differ from the legacy dense path:
#   - per-request ``committed_tokens`` / ``allocated_*`` / ``valid_*`` /
#     ``epoch`` are explicit inputs, not implicit from ``block_counts``;
#   - token -> request mapping is explicit via ``request_ids`` (no reliance on
#     a fixed ``token // S`` ABI for the long context);
#   - each cache group takes a flat ``page_ids`` + ragged
#     ``request_page_offsets`` (one past-the-end range per request), replacing
#     the dense ``[B, max_logical_pages]`` + ``% block_count`` modulo mapping;
#   - an absolute logical slot is mapped to a physical page only after passing
#     ``valid_begin`` / ``valid_end`` / ``epoch`` checks; stale pages are NOT
#     revived by modulo arithmetic on the physical pool;
#   - inactive queries output invalid sentinels (``-1``) and all active counts
#     are 0 (0 work, no cache access, no tasks).
# ===========================================================================

# The dataclasses and builder below are the pure-Python reference for the
# packed device lowering above.
from context_geometry import (  # noqa: E402
    HCA_COMPRESS_RATIO,
    HCA_ROWS_PER_SHARD,
    MAX_CSA_CANDIDATES,
    MAX_HCA_ROWS,
    CSA_CANDIDATES_PER_LEAF,
    CSA_COMPRESS_RATIO,
    SWA_WINDOW_ROWS,
    QueryGeometry,
    RequestGeometry,
    csa_next_inner_state_valid_range,
    csa_next_state_valid_range,
    csa_boundary_event,
    derive_query_geometry,
    hca_boundary_event,
    hca_next_state_valid_range,
    swa_next_valid_range,
    swa_ring_row,
    swa_window_sources_reference,
    swa_write_row,
    validate_swa_raw_descriptor,
)

def _page_index_for_slot(
    logical_slot: int,
    *,
    page_ids: list[int],
    request_page_offsets: list[int],
    request_id: int,
    block_size: int,
    valid_begin: int,
    valid_end: int,
    head: int,
    epoch: int,
    page_epoch: list[int],
) -> int:
    """Map one absolute logical slot through a ragged page map with validity.

    Returns the flattened physical row, or ``PHASE_A_INVALID_SLOT`` if the slot
    is outside the request's valid window, the page epoch is stale, or the
    page id is a sentinel. The modulo-on-physical-pool revival of stale pages
    is structurally impossible here: a page is read only by indexing the
    request's own flat ``page_ids`` range, never by ``slot % pool_size``.
    """
    if logical_slot < valid_begin or logical_slot >= valid_end:
        return PHASE_A_INVALID_SLOT
    logical_page = logical_slot // block_size
    logical_page_base = valid_begin // block_size
    intra = logical_slot % block_size
    base = request_page_offsets[request_id]
    end = request_page_offsets[request_id + 1]
    n_pages = end - base
    if n_pages == 0:
        return PHASE_A_INVALID_SLOT
    if head < 0 or head >= n_pages:
        return PHASE_A_INVALID_SLOT
    relative_page = logical_page - logical_page_base
    if relative_page < 0 or relative_page >= n_pages:
        return PHASE_A_INVALID_SLOT
    page_index = base + (head + relative_page) % n_pages
    page_id = page_ids[page_index]
    if page_id < 0:
        return PHASE_A_INVALID_SLOT
    if page_epoch is not None and page_epoch[page_index] != epoch:
        return PHASE_A_INVALID_SLOT
    return page_id * block_size + intra


@dataclass  # noqa: E402
class PhaseARaggedRequest:
    """One request's Phase A ragged descriptor (host reference)."""

    committed_tokens: int
    active: bool
    epoch: int
    # HCA compressed-row cache group.
    hca_valid_begin: int
    hca_valid_end: int
    hca_page_ids: list[int]
    # CSA main compressed cache group.
    csa_valid_begin: int
    csa_valid_end: int
    csa_page_ids: list[int]
    # CSA index cache group (parallel to CSA main).
    csa_idx_valid_begin: int
    csa_idx_valid_end: int
    csa_idx_page_ids: list[int]
    # Ring index that maps the page containing valid_begin to page_ids[head].
    hca_head: int = 0
    csa_head: int = 0
    csa_idx_head: int = 0
    # Per-page epoch for the cache groups above (one epoch per page id, in the
    # same order). Stale pages carry an older epoch and are rejected.
    hca_page_epochs: list[int] | None = None
    csa_page_epochs: list[int] | None = None
    csa_idx_page_epochs: list[int] | None = None
    # Raw/SWA and recurrent compressor-state cache groups.
    raw_valid_begin: int = 0
    raw_valid_end: int = 0
    raw_page_ids: list[int] = field(default_factory=list)
    raw_head: int = 0
    raw_page_epochs: list[int] | None = None
    hca_state_valid_begin: int = 0
    hca_state_valid_end: int = 0
    hca_state_page_ids: list[int] = field(default_factory=list)
    hca_state_head: int = 0
    hca_state_page_epochs: list[int] | None = None
    csa_state_valid_begin: int = 0
    csa_state_valid_end: int = 0
    csa_state_page_ids: list[int] = field(default_factory=list)
    csa_state_head: int = 0
    csa_state_page_epochs: list[int] | None = None
    csa_inner_state_valid_begin: int = 0
    csa_inner_state_valid_end: int = 0
    csa_inner_state_page_ids: list[int] = field(default_factory=list)
    csa_inner_state_head: int = 0
    csa_inner_state_page_epochs: list[int] | None = None


@dataclass  # noqa: E402
class PhaseAQuery:
    """One active decode query mapped to a request."""

    request_id: int
    position: int  # absolute logical position


@dataclass  # noqa: E402
class PhaseAMetadataResult:
    """Output of the Phase A ragged metadata builder for one query."""

    visible_tokens: int
    visible_swa_rows: int
    visible_hca_rows: int
    num_hca_shards: int
    hca_tail_valid_rows: int
    visible_csa_candidates: int
    num_csa_leaves: int
    csa_tail_valid_candidates: int
    # Slot mappings (absolute logical -> physical row); -1 when invalid.
    hca_slot: int
    csa_slot: int
    csa_idx_slot: int
    raw_slot: int = PHASE_A_INVALID_SLOT
    hca_state_slot: int = PHASE_A_INVALID_SLOT
    csa_state_slot: int = PHASE_A_INVALID_SLOT
    csa_inner_state_slot: int = PHASE_A_INVALID_SLOT


def _validate_page_group(
    *,
    name: str,
    valid_begin: int,
    valid_end: int,
    head: int,
    block_size: int,
    page_ids: list[int],
    page_epochs: list[int] | None,
) -> None:
    if valid_begin < 0 or valid_end < valid_begin:
        raise ValueError(f"{name} valid range [{valid_begin}, {valid_end}) is invalid")
    if page_epochs is not None and len(page_epochs) != len(page_ids):
        raise ValueError(
            f"{name} page_epochs has {len(page_epochs)} entries for "
            f"{len(page_ids)} page_ids"
        )
    if not page_ids:
        if valid_begin != valid_end or head != 0:
            raise ValueError(f"empty {name} page map must have an empty range and head=0")
        return
    if head < 0 or head >= len(page_ids):
        raise ValueError(f"{name} head {head} out of [0, {len(page_ids)})")
    first_page = valid_begin // block_size
    last_page = (valid_end - 1) // block_size if valid_end > valid_begin else first_page - 1
    required_pages = max(0, last_page - first_page + 1)
    if required_pages > len(page_ids):
        raise ValueError(
            f"{name} valid range needs {required_pages} pages but only "
            f"{len(page_ids)} were allocated"
        )


def _validate_phase_a_request(req: PhaseARaggedRequest) -> None:
    if req.committed_tokens < 0 or req.committed_tokens > M.max_position_embeddings:
        raise ValueError(
            f"committed_tokens {req.committed_tokens} out of "
            f"[0, {M.max_position_embeddings}]"
        )
    if req.epoch < 0:
        raise ValueError(f"epoch must be non-negative, got {req.epoch}")
    if req.hca_valid_end > MAX_HCA_ROWS:
        raise ValueError(
            f"hca_valid_end {req.hca_valid_end} exceeds {MAX_HCA_ROWS}"
        )
    if req.csa_valid_end > MAX_CSA_CANDIDATES:
        raise ValueError(
            f"csa_valid_end {req.csa_valid_end} exceeds {MAX_CSA_CANDIDATES}"
        )
    if req.csa_idx_valid_end > MAX_CSA_CANDIDATES:
        raise ValueError(
            f"csa_idx_valid_end {req.csa_idx_valid_end} exceeds "
            f"{MAX_CSA_CANDIDATES}"
        )
    for name, valid_end in (
        ("raw", req.raw_valid_end),
        ("hca_state", req.hca_state_valid_end),
        ("csa_state", req.csa_state_valid_end),
        ("csa_inner_state", req.csa_inner_state_valid_end),
    ):
        if valid_end > M.max_position_embeddings:
            raise ValueError(
                f"{name}_valid_end {valid_end} exceeds {M.max_position_embeddings}"
            )
    for args in (
        (
            "hca", req.hca_valid_begin, req.hca_valid_end, req.hca_head,
            CACHE_BLOCK_SIZE, req.hca_page_ids, req.hca_page_epochs,
        ),
        (
            "csa", req.csa_valid_begin, req.csa_valid_end, req.csa_head,
            CACHE_BLOCK_SIZE, req.csa_page_ids, req.csa_page_epochs,
        ),
        (
            "csa_idx", req.csa_idx_valid_begin, req.csa_idx_valid_end,
            req.csa_idx_head, CACHE_BLOCK_SIZE, req.csa_idx_page_ids,
            req.csa_idx_page_epochs,
        ),
        (
            "raw", req.raw_valid_begin, req.raw_valid_end, req.raw_head,
            CACHE_BLOCK_SIZE, req.raw_page_ids, req.raw_page_epochs,
        ),
        (
            "hca_state", req.hca_state_valid_begin, req.hca_state_valid_end,
            req.hca_state_head, C128_COMPRESSOR_BLOCK_SIZE,
            req.hca_state_page_ids, req.hca_state_page_epochs,
        ),
        (
            "csa_state", req.csa_state_valid_begin, req.csa_state_valid_end,
            req.csa_state_head, C4A_COMPRESSOR_BLOCK_SIZE,
            req.csa_state_page_ids, req.csa_state_page_epochs,
        ),
        (
            "csa_inner_state", req.csa_inner_state_valid_begin,
            req.csa_inner_state_valid_end, req.csa_inner_state_head,
            C4A_COMPRESSOR_BLOCK_SIZE, req.csa_inner_state_page_ids,
            req.csa_inner_state_page_epochs,
        ),
    ):
        _validate_page_group(
            name=args[0],
            valid_begin=args[1],
            valid_end=args[2],
            head=args[3],
            block_size=args[4],
            page_ids=args[5],
            page_epochs=args[6],
        )


def _request_geometry_from_phase_a(req: PhaseARaggedRequest) -> RequestGeometry:
    n_hca_pages = len(req.hca_page_ids)
    n_csa_pages = len(req.csa_page_ids)
    n_csa_idx_pages = len(req.csa_idx_page_ids)
    return RequestGeometry(
        committed_tokens=req.committed_tokens,
        raw_allocated_rows=len(req.raw_page_ids) * CACHE_BLOCK_SIZE,
        raw_valid_begin=req.raw_valid_begin,
        raw_valid_end=req.raw_valid_end,
        hca_allocated_rows=n_hca_pages * BLOCK_SIZE,
        hca_valid_begin=req.hca_valid_begin,
        hca_valid_end=req.hca_valid_end,
        csa_allocated_candidates=n_csa_pages * BLOCK_SIZE,
        csa_valid_begin=req.csa_valid_begin,
        csa_valid_end=req.csa_valid_end,
        active=req.active,
    )


def build_phase_a_metadata(
    requests: list[PhaseARaggedRequest],
    queries: list[PhaseAQuery],
    *,
    block_size: int = BLOCK_SIZE,
) -> list[PhaseAMetadataResult]:
    """Lower a ragged, per-request descriptor batch into per-query metadata.

    Returns one :class:`PhaseAMetadataResult` per query, in input order. The
    arithmetic mirrors :mod:`context_geometry` exactly; the slot mappings are
    lowered through the ragged page map with explicit validity/epoch checks.
    """
    for req in requests:
        _validate_phase_a_request(req)
    for query in queries:
        if query.request_id < 0 or query.request_id >= len(requests):
            raise ValueError(
                f"query request_id {query.request_id} out of [0, {len(requests)})"
            )
    out: list[PhaseAMetadataResult] = []
    # Flat page-id arrays + ragged offsets per cache group.
    hca_page_ids: list[int] = []
    hca_offsets = [0]
    csa_page_ids: list[int] = []
    csa_offsets = [0]
    csa_idx_page_ids: list[int] = []
    csa_idx_offsets = [0]
    hca_epochs: list[int] = []
    csa_epochs: list[int] = []
    csa_idx_epochs: list[int] = []
    for r in requests:
        hca_page_ids.extend(r.hca_page_ids)
        hca_offsets.append(len(hca_page_ids))
        csa_page_ids.extend(r.csa_page_ids)
        csa_offsets.append(len(csa_page_ids))
        csa_idx_page_ids.extend(r.csa_idx_page_ids)
        csa_idx_offsets.append(len(csa_idx_page_ids))
        if r.hca_page_epochs is not None:
            hca_epochs.extend(r.hca_page_epochs)
        else:
            hca_epochs.extend([r.epoch] * len(r.hca_page_ids))
        if r.csa_page_epochs is not None:
            csa_epochs.extend(r.csa_page_epochs)
        else:
            csa_epochs.extend([r.epoch] * len(r.csa_page_ids))
        if r.csa_idx_page_epochs is not None:
            csa_idx_epochs.extend(r.csa_idx_page_epochs)
        else:
            csa_idx_epochs.extend([r.epoch] * len(r.csa_idx_page_ids))

    for q in queries:
        req = requests[q.request_id]
        rg = _request_geometry_from_phase_a(req)
        qg = QueryGeometry(request=rg, position=q.position)
        geom = derive_query_geometry(qg)

        # HCA slot: only every HCA_COMPRESS_RATIO-th position has a complete
        # compressed row; otherwise no slot is written this step.
        hca_slot = PHASE_A_INVALID_SLOT
        if geom.visible_hca_rows > 0 and (q.position + 1) % HCA_COMPRESS_RATIO == 0:
            hca_row = q.position // HCA_COMPRESS_RATIO
            hca_slot = _page_index_for_slot(
                hca_row,
                page_ids=hca_page_ids,
                request_page_offsets=hca_offsets,
                request_id=q.request_id,
                block_size=BLOCK_SIZE,
                valid_begin=req.hca_valid_begin,
                valid_end=req.hca_valid_end,
                head=req.hca_head,
                epoch=req.epoch,
                page_epoch=hca_epochs,
            )
        # CSA main slot: only every CSA_COMPRESS_RATIO-th position.
        csa_slot = PHASE_A_INVALID_SLOT
        if geom.visible_csa_candidates > 0 and (q.position + 1) % CSA_COMPRESS_RATIO == 0:
            csa_cand = q.position // CSA_COMPRESS_RATIO
            csa_slot = _page_index_for_slot(
                csa_cand,
                page_ids=csa_page_ids,
                request_page_offsets=csa_offsets,
                request_id=q.request_id,
                block_size=BLOCK_SIZE,
                valid_begin=req.csa_valid_begin,
                valid_end=req.csa_valid_end,
                head=req.csa_head,
                epoch=req.epoch,
                page_epoch=csa_epochs,
            )
        # CSA index slot (parallel).
        csa_idx_slot = PHASE_A_INVALID_SLOT
        if geom.visible_csa_candidates > 0 and (q.position + 1) % CSA_COMPRESS_RATIO == 0:
            csa_cand = q.position // CSA_COMPRESS_RATIO
            csa_idx_slot = _page_index_for_slot(
                csa_cand,
                page_ids=csa_idx_page_ids,
                request_page_offsets=csa_idx_offsets,
                request_id=q.request_id,
                block_size=BLOCK_SIZE,
                valid_begin=req.csa_idx_valid_begin,
                valid_end=req.csa_idx_valid_end,
                head=req.csa_idx_head,
                epoch=req.epoch,
                page_epoch=csa_idx_epochs,
            )

        def state_slot(
            *,
            page_ids: list[int],
            page_epochs: list[int] | None,
            valid_begin: int,
            valid_end: int,
            head: int,
            block_size: int,
        ) -> int:
            if not req.active:
                return PHASE_A_INVALID_SLOT
            epochs = page_epochs
            if epochs is None:
                epochs = [req.epoch] * len(page_ids)
            return _page_index_for_slot(
                q.position,
                page_ids=page_ids,
                request_page_offsets=[0, len(page_ids)],
                request_id=0,
                block_size=block_size,
                valid_begin=valid_begin,
                valid_end=valid_end,
                head=head,
                epoch=req.epoch,
                page_epoch=epochs,
            )

        raw_slot = state_slot(
            page_ids=req.raw_page_ids,
            page_epochs=req.raw_page_epochs,
            valid_begin=req.raw_valid_begin,
            valid_end=req.raw_valid_end,
            head=req.raw_head,
            block_size=CACHE_BLOCK_SIZE,
        )
        hca_state_slot = state_slot(
            page_ids=req.hca_state_page_ids,
            page_epochs=req.hca_state_page_epochs,
            valid_begin=req.hca_state_valid_begin,
            valid_end=req.hca_state_valid_end,
            head=req.hca_state_head,
            block_size=C128_COMPRESSOR_BLOCK_SIZE,
        )
        csa_state_slot = state_slot(
            page_ids=req.csa_state_page_ids,
            page_epochs=req.csa_state_page_epochs,
            valid_begin=req.csa_state_valid_begin,
            valid_end=req.csa_state_valid_end,
            head=req.csa_state_head,
            block_size=C4A_COMPRESSOR_BLOCK_SIZE,
        )
        csa_inner_state_slot = state_slot(
            page_ids=req.csa_inner_state_page_ids,
            page_epochs=req.csa_inner_state_page_epochs,
            valid_begin=req.csa_inner_state_valid_begin,
            valid_end=req.csa_inner_state_valid_end,
            head=req.csa_inner_state_head,
            block_size=C4A_COMPRESSOR_BLOCK_SIZE,
        )
        out.append(
            PhaseAMetadataResult(
                visible_tokens=geom.visible_tokens,
                visible_swa_rows=geom.visible_swa_rows,
                visible_hca_rows=geom.visible_hca_rows,
                num_hca_shards=geom.num_hca_shards,
                hca_tail_valid_rows=geom.hca_tail_valid_rows,
                visible_csa_candidates=geom.visible_csa_candidates,
                num_csa_leaves=geom.num_csa_leaves,
                csa_tail_valid_candidates=geom.csa_tail_valid_candidates,
                hca_slot=hca_slot,
                csa_slot=csa_slot,
                csa_idx_slot=csa_idx_slot,
                raw_slot=raw_slot,
                hca_state_slot=hca_state_slot,
                csa_state_slot=csa_state_slot,
                csa_inner_state_slot=csa_inner_state_slot,
            )
        )
    return out


_PHASE_A_GROUP_ATTRS = (
    (
        "hca_page_ids", "hca_page_epochs", "hca_valid_begin",
        "hca_valid_end", "hca_head",
    ),
    (
        "csa_page_ids", "csa_page_epochs", "csa_valid_begin",
        "csa_valid_end", "csa_head",
    ),
    (
        "csa_idx_page_ids", "csa_idx_page_epochs", "csa_idx_valid_begin",
        "csa_idx_valid_end", "csa_idx_head",
    ),
    (
        "raw_page_ids", "raw_page_epochs", "raw_valid_begin",
        "raw_valid_end", "raw_head",
    ),
    (
        "hca_state_page_ids", "hca_state_page_epochs", "hca_state_valid_begin",
        "hca_state_valid_end", "hca_state_head",
    ),
    (
        "csa_state_page_ids", "csa_state_page_epochs", "csa_state_valid_begin",
        "csa_state_valid_end", "csa_state_head",
    ),
    (
        "csa_inner_state_page_ids", "csa_inner_state_page_epochs",
        "csa_inner_state_valid_begin", "csa_inner_state_valid_end",
        "csa_inner_state_head",
    ),
)


def _pack_phase_a_inputs(
    requests: list[PhaseARaggedRequest],
    queries: list[PhaseAQuery],
):
    """Pack host descriptors into the device ragged ABI."""
    import torch

    for req in requests:
        _validate_phase_a_request(req)
    request_state = torch.tensor(
        [[r.committed_tokens, int(r.active), r.epoch] for r in requests],
        dtype=torch.int32,
    )
    page_offsets = torch.empty(
        (PHASE_A_CACHE_GROUPS, len(requests) + 1), dtype=torch.int32
    )
    page_windows = torch.empty(
        (PHASE_A_CACHE_GROUPS, len(requests), 3), dtype=torch.int32
    )
    page_records: list[list[int]] = []
    for group, attrs in enumerate(_PHASE_A_GROUP_ATTRS):
        ids_attr, epochs_attr, begin_attr, end_attr, head_attr = attrs
        page_offsets[group, 0] = len(page_records)
        for request_id, req in enumerate(requests):
            page_ids = getattr(req, ids_attr)
            page_epochs = getattr(req, epochs_attr)
            if page_epochs is None:
                page_epochs = [req.epoch] * len(page_ids)
            page_records.extend(
                [[int(page_id), int(page_epoch)] for page_id, page_epoch in zip(page_ids, page_epochs)]
            )
            page_offsets[group, request_id + 1] = len(page_records)
            page_windows[group, request_id] = torch.tensor(
                [getattr(req, begin_attr), getattr(req, end_attr), getattr(req, head_attr)],
                dtype=torch.int32,
            )
    pages = torch.tensor(page_records, dtype=torch.int32).reshape(-1, 2)
    query_request_ids = torch.tensor(
        [q.request_id for q in queries], dtype=torch.int32
    )
    position_ids = torch.tensor([q.position for q in queries], dtype=torch.int32)
    return {
        "request_state": request_state,
        "query_request_ids": query_request_ids,
        "position_ids": position_ids,
        "pages": pages,
        "page_offsets": page_offsets,
        "page_windows": page_windows,
    }


def golden_phase_a_decode_metadata(tensors):
    """Independent torch reference for :func:`phase_a_decode_metadata`."""
    request_state = tensors["request_state"]
    request_ids = tensors["query_request_ids"]
    positions = tensors["position_ids"]
    pages = tensors["pages"]
    page_offsets = tensors["page_offsets"]
    page_windows = tensors["page_windows"]
    geometry = tensors["geometry"]
    slots = tensors["slots"]
    geometry.zero_()
    slots.fill_(PHASE_A_INVALID_SLOT)

    for query in range(int(positions.shape[0])):
        request = int(request_ids[query])
        position = int(positions[query])
        committed = int(request_state[request, 0])
        active = bool(int(request_state[request, 1]))
        epoch = int(request_state[request, 2])
        visible = min(committed, position + 1, M.max_position_embeddings) if active else 0

        hca_begin = int(page_windows[PHASE_A_GROUP_HCA, request, 0])
        hca_end = int(page_windows[PHASE_A_GROUP_HCA, request, 1])
        hca_pages = int(
            page_offsets[PHASE_A_GROUP_HCA, request + 1]
            - page_offsets[PHASE_A_GROUP_HCA, request]
        )
        hca_rows = min(
            max(0, min(visible // HCA_COMPRESS_RATIO, hca_end) - hca_begin),
            hca_pages * CACHE_BLOCK_SIZE,
        )
        csa_begin = int(page_windows[PHASE_A_GROUP_CSA, request, 0])
        csa_end = int(page_windows[PHASE_A_GROUP_CSA, request, 1])
        csa_pages = int(
            page_offsets[PHASE_A_GROUP_CSA, request + 1]
            - page_offsets[PHASE_A_GROUP_CSA, request]
        )
        csa_candidates = min(
            max(0, min(visible // CSA_COMPRESS_RATIO, csa_end) - csa_begin),
            csa_pages * CACHE_BLOCK_SIZE,
        )
        hca_shards = (hca_rows + HCA_ROWS_PER_SHARD - 1) // HCA_ROWS_PER_SHARD
        csa_leaves = (
            csa_candidates + CSA_CANDIDATES_PER_LEAF - 1
        ) // CSA_CANDIDATES_PER_LEAF
        geometry[query] = geometry.new_tensor(
            [
                visible,
                min(visible, SWA_WINDOW_ROWS),
                hca_rows,
                hca_shards,
                _tail(hca_rows, HCA_ROWS_PER_SHARD),
                csa_candidates,
                csa_leaves,
                _tail(csa_candidates, CSA_CANDIDATES_PER_LEAF),
            ]
        )

        for group in range(PHASE_A_CACHE_GROUPS):
            compress = 1
            block_size = C4A_COMPRESSOR_BLOCK_SIZE
            if group == PHASE_A_GROUP_HCA:
                compress = HCA_COMPRESS_RATIO
                block_size = CACHE_BLOCK_SIZE
            elif group in (PHASE_A_GROUP_CSA, PHASE_A_GROUP_CSA_IDX):
                compress = CSA_COMPRESS_RATIO
                block_size = CACHE_BLOCK_SIZE
            elif group == PHASE_A_GROUP_RAW:
                block_size = CACHE_BLOCK_SIZE
            elif group == PHASE_A_GROUP_HCA_STATE:
                block_size = C128_COMPRESSOR_BLOCK_SIZE
            if not active or (position + 1) % compress != 0:
                continue
            logical_slot = position // compress
            valid_begin = int(page_windows[group, request, 0])
            valid_end = int(page_windows[group, request, 1])
            if logical_slot < valid_begin or logical_slot >= valid_end:
                continue
            page_begin = int(page_offsets[group, request])
            page_end = int(page_offsets[group, request + 1])
            page_count = page_end - page_begin
            head = int(page_windows[group, request, 2])
            relative_page = (
                logical_slot // block_size
                - valid_begin // block_size
            )
            if page_count <= 0 or head < 0 or head >= page_count:
                continue
            if relative_page < 0 or relative_page >= page_count:
                continue
            page_index = page_begin + (head + relative_page) % page_count
            page_id = int(pages[page_index, 0])
            page_epoch = int(pages[page_index, 1])
            if page_id < 0 or page_epoch != epoch:
                continue
            slots[query, group] = (
                page_id * block_size + logical_slot % block_size
            )


def build_phase_a_tensor_specs(case: str):
    """Build a concrete device fixture for one named Phase A case."""
    import torch
    from golden import TensorSpec

    if case not in _PHASE_A_CASES:
        raise ValueError(f"unknown Phase A case: {case!r}")
    requests, queries, _ = _PHASE_A_CASES[case]()
    inputs = _pack_phase_a_inputs(requests, queries)
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in inputs.items()
    ]
    specs.extend(
        [
            TensorSpec(
                "geometry",
                [len(queries), PHASE_A_GEOMETRY_FIELDS],
                torch.int32,
                is_output=True,
            ),
            TensorSpec(
                "slots",
                [len(queries), PHASE_A_CACHE_GROUPS],
                torch.int64,
                is_output=True,
            ),
        ]
    )
    return specs


# ---------------------------------------------------------------------------
# Phase A standalone cases (run_055 plan §5 A3 / §6.3). Each case builds a
# ragged descriptor batch, lowers it, and compares against an inline golden.
# These run as pure Python (host reference) so the contract gate and the
# device harness can both assert them; the device harness additionally runs
# the legacy dense ``decode_metadata`` as a migration-fence regression.
# ---------------------------------------------------------------------------


def _case_geometry_boundaries() -> tuple[list[PhaseARaggedRequest], list[PhaseAQuery], list[PhaseAMetadataResult]]:
    """Mixed-length requests covering 1 / 128 / 12K / 16K / 1M tails.

    All requests share the full-capacity allocator so the case isolates the
    runtime length derivation (run_055 §6.1 matrix packed into one batch).
    Page counts are KV storage pages (BLOCK_SIZE rows/page); valid ranges are
    in row/candidate units.
    """
    lengths = [1, 128, 12288, 16384, M.max_position_embeddings]
    requests: list[PhaseARaggedRequest] = []
    queries: list[PhaseAQuery] = []
    for rid, n in enumerate(lengths):
        n_hca_rows = n // HCA_COMPRESS_RATIO
        n_hca_pages = (n_hca_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
        n_hca_pages = max(1, n_hca_pages)
        n_csa_cands = n // CSA_COMPRESS_RATIO
        n_csa_pages = (n_csa_cands + BLOCK_SIZE - 1) // BLOCK_SIZE
        n_csa_pages = max(1, n_csa_pages)
        requests.append(
            PhaseARaggedRequest(
                committed_tokens=n,
                active=True,
                epoch=1,
                hca_valid_begin=0,
                hca_valid_end=n_hca_rows,
                hca_page_ids=[1000 + rid * 100 + i for i in range(n_hca_pages)],
                csa_valid_begin=0,
                csa_valid_end=n_csa_cands,
                csa_page_ids=[2000 + rid * 100 + i for i in range(n_csa_pages)],
                csa_idx_valid_begin=0,
                csa_idx_valid_end=n_csa_cands,
                csa_idx_page_ids=[3000 + rid * 100 + i for i in range(n_csa_pages)],
            )
        )
        queries.append(PhaseAQuery(request_id=rid, position=n - 1))

    # Inline golden — derived directly from the fixture, independent of the
    # lowering. (vt, swa, hca_rows, hca_sh, csa_c, csa_l) per request.
    expected_rows = [
        (1, 1, 0, 0, 0, 0),
        (128, 128, 1, 1, 32, 1),
        (12288, 128, 96, 1, 3072, 2),
        (16384, 128, 128, 1, 4096, 2),
        (M.max_position_embeddings, 128, 8192, 64, 262144, 128),
    ]
    expected: list[PhaseAMetadataResult] = []
    for rid, (vt, swa, hr, hsh, cc, cl) in enumerate(expected_rows):
        expected.append(
            PhaseAMetadataResult(
                visible_tokens=vt,
                visible_swa_rows=swa,
                visible_hca_rows=hr,
                num_hca_shards=hsh,
                hca_tail_valid_rows=_tail(hr, HCA_ROWS_PER_SHARD),
                visible_csa_candidates=cc,
                num_csa_leaves=cl,
                csa_tail_valid_candidates=_tail(cc, CSA_CANDIDATES_PER_LEAF),
                # positions are n-1; HCA slot valid iff (n-1+1) % 128 == 0,
                # i.e. n is a multiple of 128. n=1 -> invalid; others valid.
                hca_slot=_expected_slot(
                    requests[rid].hca_page_ids, queries[rid].position,
                    compress=HCA_COMPRESS_RATIO, valid_end=requests[rid].hca_valid_end,
                ),
                csa_slot=_expected_slot(
                    requests[rid].csa_page_ids, queries[rid].position,
                    compress=CSA_COMPRESS_RATIO, valid_end=requests[rid].csa_valid_end,
                ),
                csa_idx_slot=_expected_slot(
                    requests[rid].csa_idx_page_ids, queries[rid].position,
                    compress=CSA_COMPRESS_RATIO, valid_end=requests[rid].csa_idx_valid_end,
                ),
            )
        )
    return requests, queries, expected


def _case_ragged_page_permutation() -> tuple[list[PhaseARaggedRequest], list[PhaseAQuery], list[PhaseAMetadataResult]]:
    """Permutation / rollover of physical page ids; logical slot still
    resolves to the page the allocator assigned, not by modulo-on-pool."""
    pos = HCA_COMPRESS_RATIO - 1  # 127 — last token of the first HCA row
    req0 = PhaseARaggedRequest(
        committed_tokens=HCA_COMPRESS_RATIO * HCA_ROWS_PER_SHARD,
        active=True, epoch=1,
        hca_valid_begin=0, hca_valid_end=HCA_ROWS_PER_SHARD, hca_page_ids=[7, 3, 19, 23],
        csa_valid_begin=0, csa_valid_end=2 * BLOCK_SIZE, csa_page_ids=[11, 5],
        csa_idx_valid_begin=0, csa_idx_valid_end=2 * BLOCK_SIZE, csa_idx_page_ids=[13, 9],
    )
    req1 = PhaseARaggedRequest(
        committed_tokens=HCA_COMPRESS_RATIO * HCA_ROWS_PER_SHARD,
        active=True, epoch=1,
        hca_valid_begin=0, hca_valid_end=HCA_ROWS_PER_SHARD, hca_page_ids=[4, 8, 16, 20],
        csa_valid_begin=0, csa_valid_end=2 * BLOCK_SIZE, csa_page_ids=[6, 12],
        csa_idx_valid_begin=0, csa_idx_valid_end=2 * BLOCK_SIZE, csa_idx_page_ids=[10, 14],
    )
    requests = [req0, req1]
    queries = [PhaseAQuery(request_id=0, position=pos), PhaseAQuery(request_id=1, position=pos)]
    expected = [
        PhaseAMetadataResult(
            visible_tokens=HCA_COMPRESS_RATIO,
            visible_swa_rows=min(HCA_COMPRESS_RATIO, SWA_WINDOW_ROWS),
            visible_hca_rows=1, num_hca_shards=1,
            hca_tail_valid_rows=_tail(1, HCA_ROWS_PER_SHARD),
            visible_csa_candidates=HCA_COMPRESS_RATIO // CSA_COMPRESS_RATIO,
            num_csa_leaves=1,
            csa_tail_valid_candidates=_tail(HCA_COMPRESS_RATIO // CSA_COMPRESS_RATIO, CSA_CANDIDATES_PER_LEAF),
            hca_slot=_expected_slot(req0.hca_page_ids, pos, HCA_COMPRESS_RATIO, req0.hca_valid_end),
            csa_slot=_expected_slot(req0.csa_page_ids, pos, CSA_COMPRESS_RATIO, req0.csa_valid_end),
            csa_idx_slot=_expected_slot(req0.csa_idx_page_ids, pos, CSA_COMPRESS_RATIO, req0.csa_idx_valid_end),
        ),
        PhaseAMetadataResult(
            visible_tokens=HCA_COMPRESS_RATIO,
            visible_swa_rows=min(HCA_COMPRESS_RATIO, SWA_WINDOW_ROWS),
            visible_hca_rows=1, num_hca_shards=1,
            hca_tail_valid_rows=_tail(1, HCA_ROWS_PER_SHARD),
            visible_csa_candidates=HCA_COMPRESS_RATIO // CSA_COMPRESS_RATIO,
            num_csa_leaves=1,
            csa_tail_valid_candidates=_tail(HCA_COMPRESS_RATIO // CSA_COMPRESS_RATIO, CSA_CANDIDATES_PER_LEAF),
            hca_slot=_expected_slot(req1.hca_page_ids, pos, HCA_COMPRESS_RATIO, req1.hca_valid_end),
            csa_slot=_expected_slot(req1.csa_page_ids, pos, CSA_COMPRESS_RATIO, req1.csa_valid_end),
            csa_idx_slot=_expected_slot(req1.csa_idx_page_ids, pos, CSA_COMPRESS_RATIO, req1.csa_idx_valid_end),
        ),
    ]
    return requests, queries, expected


def _case_rolling_head() -> tuple[list[PhaseARaggedRequest], list[PhaseAQuery], list[PhaseAMetadataResult]]:
    """Compact nonzero valid ranges with rotated page-array heads."""
    position = 32767
    req = PhaseARaggedRequest(
        committed_tokens=position + 1,
        active=True,
        epoch=7,
        hca_valid_begin=128,
        hca_valid_end=256,
        hca_page_ids=[4, 8, 16, 20],
        csa_valid_begin=7936,
        csa_valid_end=8192,
        csa_page_ids=[6, 12, 18, 24, 30, 36, 42, 48],
        csa_idx_valid_begin=7936,
        csa_idx_valid_end=8192,
        csa_idx_page_ids=[10, 14, 22, 26, 34, 38, 46, 50],
        hca_head=1,
        csa_head=1,
        csa_idx_head=1,
        raw_valid_begin=32640,
        raw_valid_end=32768,
        raw_page_ids=[20, 21, 22, 23],
        hca_state_valid_begin=32760,
        hca_state_valid_end=32768,
        hca_state_page_ids=[21],
        csa_state_valid_begin=32764,
        csa_state_valid_end=32768,
        csa_state_page_ids=[24, 25],
        csa_inner_state_valid_begin=32764,
        csa_inner_state_valid_end=32768,
        csa_inner_state_page_ids=[26, 27],
    )
    expected = [
        PhaseAMetadataResult(
            visible_tokens=position + 1,
            visible_swa_rows=SWA_WINDOW_ROWS,
            visible_hca_rows=128,
            num_hca_shards=1,
            hca_tail_valid_rows=128,
            visible_csa_candidates=256,
            num_csa_leaves=1,
            csa_tail_valid_candidates=256,
            hca_slot=_expected_slot(
                req.hca_page_ids,
                position,
                HCA_COMPRESS_RATIO,
                req.hca_valid_end,
                req.hca_valid_begin,
                req.hca_head,
            ),
            csa_slot=_expected_slot(
                req.csa_page_ids,
                position,
                CSA_COMPRESS_RATIO,
                req.csa_valid_end,
                req.csa_valid_begin,
                req.csa_head,
            ),
            csa_idx_slot=_expected_slot(
                req.csa_idx_page_ids,
                position,
                CSA_COMPRESS_RATIO,
                req.csa_idx_valid_end,
                req.csa_idx_valid_begin,
                req.csa_idx_head,
            ),
            raw_slot=_expected_slot(
                req.raw_page_ids,
                position,
                1,
                req.raw_valid_end,
                req.raw_valid_begin,
                req.raw_head,
                CACHE_BLOCK_SIZE,
            ),
            hca_state_slot=_expected_slot(
                req.hca_state_page_ids,
                position,
                1,
                req.hca_state_valid_end,
                req.hca_state_valid_begin,
                req.hca_state_head,
                C128_COMPRESSOR_BLOCK_SIZE,
            ),
            csa_state_slot=_expected_slot(
                req.csa_state_page_ids,
                position,
                1,
                req.csa_state_valid_end,
                req.csa_state_valid_begin,
                req.csa_state_head,
                C4A_COMPRESSOR_BLOCK_SIZE,
            ),
            csa_inner_state_slot=_expected_slot(
                req.csa_inner_state_page_ids,
                position,
                1,
                req.csa_inner_state_valid_end,
                req.csa_inner_state_valid_begin,
                req.csa_inner_state_head,
                C4A_COMPRESSOR_BLOCK_SIZE,
            ),
        )
    ]
    return [req], [PhaseAQuery(request_id=0, position=position)], expected


def _case_inactive_lane() -> tuple[list[PhaseARaggedRequest], list[PhaseAQuery], list[PhaseAMetadataResult]]:
    """One active, one inactive request in the same batch. Inactive outputs
    all-zero counts and invalid sentinels."""
    active = PhaseARaggedRequest(
        committed_tokens=128, active=True, epoch=1,
        hca_valid_begin=0, hca_valid_end=HCA_ROWS_PER_SHARD, hca_page_ids=[100, 101, 102, 103],
        csa_valid_begin=0, csa_valid_end=BLOCK_SIZE, csa_page_ids=[200],
        csa_idx_valid_begin=0, csa_idx_valid_end=BLOCK_SIZE, csa_idx_page_ids=[300],
    )
    inactive = PhaseARaggedRequest(
        committed_tokens=0, active=False, epoch=0,
        hca_valid_begin=0, hca_valid_end=0, hca_page_ids=[],
        csa_valid_begin=0, csa_valid_end=0, csa_page_ids=[],
        csa_idx_valid_begin=0, csa_idx_valid_end=0, csa_idx_page_ids=[],
    )
    requests = [active, inactive]
    queries = [PhaseAQuery(request_id=0, position=127), PhaseAQuery(request_id=1, position=0)]
    expected = [
        PhaseAMetadataResult(
            visible_tokens=128, visible_swa_rows=128,
            visible_hca_rows=1, num_hca_shards=1,
            hca_tail_valid_rows=_tail(1, HCA_ROWS_PER_SHARD),
            visible_csa_candidates=32, num_csa_leaves=1,
            csa_tail_valid_candidates=_tail(32, CSA_CANDIDATES_PER_LEAF),
            hca_slot=_expected_slot(active.hca_page_ids, 127, HCA_COMPRESS_RATIO, active.hca_valid_end),
            csa_slot=_expected_slot(active.csa_page_ids, 127, CSA_COMPRESS_RATIO, active.csa_valid_end),
            csa_idx_slot=_expected_slot(active.csa_idx_page_ids, 127, CSA_COMPRESS_RATIO, active.csa_idx_valid_end),
        ),
        PhaseAMetadataResult(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1),
    ]
    return requests, queries, expected


def _case_stale_epoch_rejected() -> tuple[list[PhaseARaggedRequest], list[PhaseAQuery], list[PhaseAMetadataResult]]:
    """A page whose stored epoch is older than the request's current epoch
    must NOT be mapped, even though the logical slot is in-range. The legacy
    modulo path would silently revive it; Phase A rejects it."""
    pos = HCA_COMPRESS_RATIO - 1
    stale = PhaseARaggedRequest(
        committed_tokens=HCA_COMPRESS_RATIO, active=True, epoch=2,
        hca_valid_begin=0, hca_valid_end=HCA_ROWS_PER_SHARD,
        hca_page_ids=[42, 45, 48, 51], hca_page_epochs=[1, 1, 1, 1],
        csa_valid_begin=0, csa_valid_end=BLOCK_SIZE,
        csa_page_ids=[43], csa_page_epochs=[1],
        csa_idx_valid_begin=0, csa_idx_valid_end=BLOCK_SIZE,
        csa_idx_page_ids=[44], csa_idx_page_epochs=[1],
    )
    requests = [stale]
    queries = [PhaseAQuery(request_id=0, position=pos)]
    # Geometry is computed; but every slot is rejected by the stale epoch.
    expected = [
        PhaseAMetadataResult(
            visible_tokens=HCA_COMPRESS_RATIO,
            visible_swa_rows=min(HCA_COMPRESS_RATIO, SWA_WINDOW_ROWS),
            visible_hca_rows=1, num_hca_shards=1,
            hca_tail_valid_rows=_tail(1, HCA_ROWS_PER_SHARD),
            visible_csa_candidates=HCA_COMPRESS_RATIO // CSA_COMPRESS_RATIO,
            num_csa_leaves=1,
            csa_tail_valid_candidates=_tail(HCA_COMPRESS_RATIO // CSA_COMPRESS_RATIO, CSA_CANDIDATES_PER_LEAF),
            hca_slot=PHASE_A_INVALID_SLOT,   # rejected by epoch
            csa_slot=PHASE_A_INVALID_SLOT,
            csa_idx_slot=PHASE_A_INVALID_SLOT,
        )
    ]
    return requests, queries, expected


def _case_one_m_tail() -> tuple[list[PhaseARaggedRequest], list[PhaseAQuery], list[PhaseAMetadataResult]]:
    """A single 1M request: HCA 64 shards / 8192 rows, CSA 128 leaves /
    262144 candidates. The tail fills the last shard/leaf exactly."""
    n = M.max_position_embeddings
    n_hca_rows = n // HCA_COMPRESS_RATIO          # 8192
    n_hca_pages = n_hca_rows // BLOCK_SIZE         # 256
    n_csa_cands = n // CSA_COMPRESS_RATIO          # 262144
    n_csa_pages = n_csa_cands // BLOCK_SIZE        # 8192
    req = PhaseARaggedRequest(
        committed_tokens=n, active=True, epoch=1,
        hca_valid_begin=0, hca_valid_end=n_hca_rows,
        hca_page_ids=[1000 + i for i in range(n_hca_pages)],
        csa_valid_begin=0, csa_valid_end=n_csa_cands,
        csa_page_ids=[2000 + i for i in range(n_csa_pages)],
        csa_idx_valid_begin=0, csa_idx_valid_end=n_csa_cands,
        csa_idx_page_ids=[3000 + i for i in range(n_csa_pages)],
    )
    requests = [req]
    queries = [PhaseAQuery(request_id=0, position=n - 1)]
    expected = [
        PhaseAMetadataResult(
            visible_tokens=n, visible_swa_rows=SWA_WINDOW_ROWS,
            visible_hca_rows=MAX_HCA_ROWS,
            num_hca_shards=MAX_HCA_ROWS // HCA_ROWS_PER_SHARD,
            hca_tail_valid_rows=_tail(MAX_HCA_ROWS, HCA_ROWS_PER_SHARD),
            visible_csa_candidates=MAX_CSA_CANDIDATES,
            num_csa_leaves=MAX_CSA_CANDIDATES // CSA_CANDIDATES_PER_LEAF,
            csa_tail_valid_candidates=_tail(MAX_CSA_CANDIDATES, CSA_CANDIDATES_PER_LEAF),
            hca_slot=_expected_slot(req.hca_page_ids, n - 1, HCA_COMPRESS_RATIO, req.hca_valid_end),
            csa_slot=_expected_slot(req.csa_page_ids, n - 1, CSA_COMPRESS_RATIO, req.csa_valid_end),
            csa_idx_slot=_expected_slot(req.csa_idx_page_ids, n - 1, CSA_COMPRESS_RATIO, req.csa_idx_valid_end),
        )
    ]
    return requests, queries, expected


def _tail(n: int, shard: int) -> int:
    """Valid items in the last active shard/leaf (independent golden formula).

    0 items -> 0 tail; otherwise the remainder, or the full shard when the
    items fill it exactly.
    """
    if n == 0:
        return 0
    rem = n % shard
    return rem if rem != 0 else shard


def _expected_slot(
    page_ids: list[int],
    position: int,
    compress: int,
    valid_end: int,
    valid_begin: int = 0,
    head: int = 0,
    block_size: int = BLOCK_SIZE,
) -> int:
    """Direct golden slot mapping, independent of the ragged lowering.

    Returns ``page_ids[idx] * BLOCK_SIZE + intra`` when the position lies on a
    compression boundary and the row/candidate is inside the valid window and
    page range; otherwise ``PHASE_A_INVALID_SLOT``. This mirrors what the
    allocator contract says, not what the lowering code does.
    """
    if (position + 1) % compress != 0:
        return PHASE_A_INVALID_SLOT
    row = position // compress
    if row < valid_begin or row >= valid_end:
        return PHASE_A_INVALID_SLOT
    if not page_ids or head < 0 or head >= len(page_ids):
        return PHASE_A_INVALID_SLOT
    logical_page_base = valid_begin // block_size
    relative_page = row // block_size - logical_page_base
    intra = row % block_size
    if relative_page < 0 or relative_page >= len(page_ids):
        return PHASE_A_INVALID_SLOT
    page_idx = (head + relative_page) % len(page_ids)
    if page_ids[page_idx] < 0:
        return PHASE_A_INVALID_SLOT
    return page_ids[page_idx] * block_size + intra


_PHASE_A_CASES = {
    "geometry_boundaries": _case_geometry_boundaries,
    "ragged_page_permutation": _case_ragged_page_permutation,
    "rolling_head": _case_rolling_head,
    "inactive_lane": _case_inactive_lane,
    "stale_epoch_rejected": _case_stale_epoch_rejected,
    "one_m_tail": _case_one_m_tail,
}


def run_phase_a_case(case: str) -> bool:
    """Run one Phase A standalone case and compare its lowering to the
    inline golden returned alongside the fixture."""
    if case not in _PHASE_A_CASES:
        raise ValueError(f"unknown Phase A case: {case!r}")
    requests, queries, expected = _PHASE_A_CASES[case]()
    got = build_phase_a_metadata(requests, queries)
    if len(got) != len(expected):
        print(f"[case={case}] length mismatch: got {len(got)}, want {len(expected)}")
        return False
    ok = True
    for i, (g, w) in enumerate(zip(got, expected)):
        if g != w:
            print(f"[case={case} query={i}] mismatch:")
            print(f"  got  = {g}")
            print(f"  want = {w}")
            ok = False
    return ok


PHASE_B_REQUESTS_DYN = pl.dynamic("PHASE_B_REQUESTS_DYN")
PHASE_B_REQUEST_OFFSETS_DYN = pl.dynamic("PHASE_B_REQUEST_OFFSETS_DYN")
PHASE_B_QUERIES_DYN = pl.dynamic("PHASE_B_QUERIES_DYN")
PHASE_B_PAGES_DYN = pl.dynamic("PHASE_B_PAGES_DYN")
PHASE_B_RAW_GROUP = PHASE_A_GROUP_RAW
PHASE_B_SOURCE_WIDTH = SWA_WINDOW_ROWS


@pl.jit
def phase_b_swa_metadata(
    request_state: pl.Tensor[[PHASE_B_REQUESTS_DYN, 3], pl.INT32],
    query_request_ids: pl.Tensor[[PHASE_B_QUERIES_DYN], pl.INT32],
    request_query_offsets: pl.Tensor[[PHASE_B_REQUEST_OFFSETS_DYN], pl.INT32],
    position_ids: pl.Tensor[[PHASE_B_QUERIES_DYN], pl.INT32],
    pages: pl.Tensor[[PHASE_B_PAGES_DYN, 2], pl.INT32],
    page_offsets: pl.Tensor[
        [PHASE_A_CACHE_GROUPS, PHASE_B_REQUEST_OFFSETS_DYN], pl.INT32
    ],
    page_windows: pl.Tensor[
        [PHASE_A_CACHE_GROUPS, PHASE_B_REQUESTS_DYN, 3], pl.INT32
    ],
    swa_write_slots: pl.Out[pl.Tensor[[PHASE_B_QUERIES_DYN], pl.INT64]],
    swa_sources: pl.Out[
        pl.Tensor[[PHASE_B_QUERIES_DYN, PHASE_B_SOURCE_WIDTH], pl.INT32]
    ],
    swa_lens: pl.Out[pl.Tensor[[PHASE_B_QUERIES_DYN], pl.INT32]],
    next_raw_valid_ranges: pl.Out[
        pl.Tensor[[PHASE_B_REQUESTS_DYN, 2], pl.INT32]
    ],
):
    """Lower a four-page raw/SWA ring from the Phase A ragged ABI."""
    request_state.bind_dynamic(0, PHASE_B_REQUESTS_DYN)
    query_request_ids.bind_dynamic(0, PHASE_B_QUERIES_DYN)
    request_query_offsets.bind_dynamic(0, PHASE_B_REQUEST_OFFSETS_DYN)
    position_ids.bind_dynamic(0, PHASE_B_QUERIES_DYN)
    pages.bind_dynamic(0, PHASE_B_PAGES_DYN)
    page_offsets.bind_dynamic(1, PHASE_B_REQUEST_OFFSETS_DYN)
    page_windows.bind_dynamic(1, PHASE_B_REQUESTS_DYN)
    swa_write_slots.bind_dynamic(0, PHASE_B_QUERIES_DYN)
    swa_sources.bind_dynamic(0, PHASE_B_QUERIES_DYN)
    swa_lens.bind_dynamic(0, PHASE_B_QUERIES_DYN)
    next_raw_valid_ranges.bind_dynamic(0, PHASE_B_REQUESTS_DYN)

    request_count = pl.tensor.dim(request_state, 0)
    query_count = pl.tensor.dim(position_ids, 0)
    offset_count = pl.tensor.dim(request_query_offsets, 0)
    page_count = pl.tensor.dim(pages, 0)

    for metadata_core in pl.spmd(1, name_hint="decode_phase_b_swa_metadata"):
        for query in pl.range(metadata_core, query_count):
            pl.write(
                swa_write_slots,
                [query],
                pl.cast(SWA_SOURCE_INVALID, pl.INT64),
            )
            pl.write(swa_lens, [query], pl.cast(0, pl.INT32))
            for source_column in pl.range(PHASE_B_SOURCE_WIDTH):
                pl.write(
                    swa_sources,
                    [query, source_column],
                    pl.cast(SWA_SOURCE_INVALID, pl.INT32),
                )

        for request in pl.range(metadata_core, request_count):
            pl.write(
                next_raw_valid_ranges,
                [request, 0],
                pl.cast(0, pl.INT32),
            )
            pl.write(
                next_raw_valid_ranges,
                [request, 1],
                pl.cast(0, pl.INT32),
            )

        offsets_match_requests = pl.cast(0, pl.INT32)
        if offset_count == request_count + 1:
            offsets_match_requests = pl.cast(1, pl.INT32)

        if offsets_match_requests != 0:
            for request in pl.range(metadata_core, request_count):
                request_index = pl.cast(request, pl.INDEX)
                active = pl.read(request_state, [request_index, 1])
                committed = pl.read(request_state, [request_index, 0])
                epoch = pl.read(request_state, [request_index, 2])
                request_begin = pl.read(request_query_offsets, [request])
                request_end = pl.read(request_query_offsets, [request + 1])

                query_range_valid = pl.cast(1, pl.INT32)
                first_position = pl.cast(0, pl.INT32)
                previous_position = pl.cast(0, pl.INT32)
                if request_begin < 0:
                    query_range_valid = pl.cast(0, pl.INT32)
                if request_end < request_begin:
                    query_range_valid = pl.cast(0, pl.INT32)
                if request_end > query_count:
                    query_range_valid = pl.cast(0, pl.INT32)
                if query_range_valid != 0:
                    for query in pl.range(request_begin, request_end):
                        query_request = pl.read(query_request_ids, [query])
                        query_position = pl.read(position_ids, [query])
                        if query_request != request:
                            query_range_valid = pl.cast(0, pl.INT32)
                        if query_position < 0:
                            query_range_valid = pl.cast(0, pl.INT32)
                        if query_position >= M.max_position_embeddings:
                            query_range_valid = pl.cast(0, pl.INT32)
                        if query == request_begin:
                            first_position = query_position
                        if query > request_begin:
                            if query_position != previous_position + 1:
                                query_range_valid = pl.cast(0, pl.INT32)
                        previous_position = query_position

                raw_descriptor_valid = pl.cast(0, pl.INT32)
                raw_valid_begin = pl.cast(0, pl.INT32)
                raw_valid_end = pl.cast(0, pl.INT32)
                # Keep the raw page-table base in request scope.  It is only
                # consumed when ``raw_descriptor_valid`` is true, but defining
                # it here gives the SSA lowering an explicit value on every
                # control-flow path (including inactive/stale descriptors).
                raw_page_begin = pl.cast(0, pl.INT32)
                if active != 0:
                    if committed >= 0:
                        if committed <= M.max_position_embeddings:
                            if epoch >= 0:
                                if query_range_valid != 0:
                                    raw_valid_begin = pl.read(
                                        page_windows,
                                        [PHASE_B_RAW_GROUP, request_index, 0],
                                    )
                                    raw_valid_end = pl.read(
                                        page_windows,
                                        [PHASE_B_RAW_GROUP, request_index, 1],
                                    )
                                    raw_head = pl.read(
                                        page_windows,
                                        [PHASE_B_RAW_GROUP, request_index, 2],
                                    )
                                    raw_page_begin = pl.read(
                                        page_offsets,
                                        [PHASE_B_RAW_GROUP, request_index],
                                    )
                                    raw_page_end = pl.read(
                                        page_offsets,
                                        [PHASE_B_RAW_GROUP, request_index + 1],
                                    )
                                    raw_page_total = raw_page_end - raw_page_begin
                                    if raw_page_total == SWA_PERSISTENT_PAGES_PER_REQUEST:
                                        if raw_head == 0:
                                            if raw_page_begin >= 0:
                                                if raw_page_end <= page_count:
                                                    if raw_valid_begin >= 0:
                                                        if raw_valid_end >= raw_valid_begin:
                                                            if raw_valid_end <= committed:
                                                                if raw_valid_end <= M.max_position_embeddings:
                                                                    if (
                                                                        raw_valid_end
                                                                        - raw_valid_begin
                                                                        <= SWA_PERSISTENT_ROWS_PER_REQUEST
                                                                    ):
                                                                        raw_descriptor_valid = pl.cast(
                                                                            1,
                                                                            pl.INT32,
                                                                        )
                                                                        for raw_relative_page in pl.range(
                                                                            SWA_PERSISTENT_PAGES_PER_REQUEST
                                                                        ):
                                                                            raw_page_entry = pl.cast(
                                                                                raw_page_begin
                                                                                + raw_relative_page,
                                                                                pl.INDEX,
                                                                            )
                                                                            raw_page_id = pl.read(
                                                                                pages,
                                                                                [raw_page_entry, 0],
                                                                            )
                                                                            raw_page_epoch = pl.read(
                                                                                pages,
                                                                                [raw_page_entry, 1],
                                                                            )
                                                                            if raw_page_id < 0:
                                                                                raw_descriptor_valid = pl.cast(
                                                                                    0,
                                                                                    pl.INT32,
                                                                                )
                                                                            if (
                                                                                raw_page_id
                                                                                > SWA_SOURCE_INT32_MAX
                                                                                // CACHE_BLOCK_SIZE
                                                                            ):
                                                                                raw_descriptor_valid = pl.cast(
                                                                                    0,
                                                                                    pl.INT32,
                                                                                )
                                                                            if raw_page_epoch != epoch:
                                                                                raw_descriptor_valid = pl.cast(
                                                                                    0,
                                                                                    pl.INT32,
                                                                                )

                if raw_descriptor_valid != 0:
                    next_begin = pl.cast(
                        raw_valid_end - SWA_PERSISTENT_ROWS_PER_REQUEST,
                        pl.INT32,
                    )
                    if next_begin < 0:
                        next_begin = pl.cast(0, pl.INT32)
                    next_end = raw_valid_end
                    if request_end > request_begin:
                        last_position = pl.read(
                            position_ids,
                            [request_end - 1],
                        )
                        if last_position + 1 > next_end:
                            next_end = pl.cast(last_position + 1, pl.INT32)
                        next_begin = pl.cast(
                            next_end - SWA_PERSISTENT_ROWS_PER_REQUEST,
                            pl.INT32,
                        )
                        if next_begin < 0:
                            next_begin = pl.cast(0, pl.INT32)
                    pl.write(next_raw_valid_ranges, [request, 0], next_begin)
                    pl.write(next_raw_valid_ranges, [request, 1], next_end)

                if query_range_valid != 0:
                    if active != 0:
                        if committed >= 0:
                            if committed <= M.max_position_embeddings:
                                for query in pl.range(request_begin, request_end):
                                    position = pl.read(position_ids, [query])
                                    visible_end = committed
                                    if position + 1 < visible_end:
                                        visible_end = pl.cast(
                                            position + 1,
                                            pl.INT32,
                                        )
                                    if visible_end > M.max_position_embeddings:
                                        visible_end = pl.cast(
                                            M.max_position_embeddings,
                                            pl.INT32,
                                        )
                                    if visible_end < 0:
                                        visible_end = pl.cast(0, pl.INT32)
                                    window_begin = pl.cast(
                                        visible_end - PHASE_B_SOURCE_WIDTH,
                                        pl.INT32,
                                    )
                                    if window_begin < 0:
                                        window_begin = pl.cast(0, pl.INT32)

                                    persistent_begin = window_begin
                                    if raw_valid_begin > persistent_begin:
                                        persistent_begin = raw_valid_begin
                                    persistent_end = visible_end
                                    if raw_valid_end < persistent_end:
                                        persistent_end = raw_valid_end
                                    if first_position < persistent_end:
                                        persistent_end = first_position
                                    persistent_len = pl.cast(0, pl.INT32)
                                    if raw_descriptor_valid != 0:
                                        if persistent_end > persistent_begin:
                                            persistent_len = pl.cast(
                                                persistent_end - persistent_begin,
                                                pl.INT32,
                                            )

                                    overlay_begin = window_begin
                                    if first_position > overlay_begin:
                                        overlay_begin = first_position
                                    overlay_end = visible_end
                                    if position + 1 < overlay_end:
                                        overlay_end = pl.cast(
                                            position + 1,
                                            pl.INT32,
                                        )
                                    overlay_len = pl.cast(0, pl.INT32)
                                    if overlay_end > overlay_begin:
                                        overlay_len = pl.cast(
                                            overlay_end - overlay_begin,
                                            pl.INT32,
                                        )
                                    valid_sources = pl.cast(
                                        persistent_len + overlay_len,
                                        pl.INT32,
                                    )

                                    for source_column in pl.range(
                                        PHASE_B_SOURCE_WIDTH
                                    ):
                                        source = pl.cast(
                                            SWA_SOURCE_INVALID,
                                            pl.INT32,
                                        )
                                        if source_column < persistent_len:
                                            logical_position = (
                                                persistent_begin + source_column
                                            )
                                            ring_offset = (
                                                logical_position
                                                % SWA_PERSISTENT_ROWS_PER_REQUEST
                                            )
                                            raw_relative_page = (
                                                ring_offset // CACHE_BLOCK_SIZE
                                            )
                                            raw_page_entry = pl.cast(
                                                raw_page_begin + raw_relative_page,
                                                pl.INDEX,
                                            )
                                            raw_page_id = pl.read(
                                                pages, [raw_page_entry, 0]
                                            )
                                            source = pl.cast(
                                                raw_page_id
                                                * CACHE_BLOCK_SIZE
                                                + ring_offset % CACHE_BLOCK_SIZE,
                                                pl.INT32,
                                            )
                                        else:
                                            overlay_column = (
                                                source_column - persistent_len
                                            )
                                            if overlay_column < overlay_len:
                                                logical_position = (
                                                    overlay_begin + overlay_column
                                                )
                                                overlay_query = (
                                                    request_begin
                                                    + logical_position
                                                    - first_position
                                                )
                                                if (
                                                    overlay_query
                                                    <= SWA_SOURCE_MAX_OVERLAY_QUERY
                                                ):
                                                    source = pl.cast(
                                                        SWA_SOURCE_OVERLAY_BASE
                                                        - overlay_query,
                                                        pl.INT32,
                                                    )
                                        pl.write(
                                            swa_sources,
                                            [query, source_column],
                                            source,
                                        )

                                    if raw_descriptor_valid != 0:
                                        ring_offset = (
                                            position
                                            % SWA_PERSISTENT_ROWS_PER_REQUEST
                                        )
                                        raw_relative_page = (
                                            ring_offset // CACHE_BLOCK_SIZE
                                        )
                                        raw_page_entry = pl.cast(
                                            raw_page_begin + raw_relative_page,
                                            pl.INDEX,
                                        )
                                        raw_page_id = pl.read(
                                            pages, [raw_page_entry, 0]
                                        )
                                        pl.write(
                                            swa_write_slots,
                                            [query],
                                            pl.cast(raw_page_id, pl.INT64)
                                            * pl.cast(
                                                CACHE_BLOCK_SIZE,
                                                pl.INT64,
                                            )
                                            + pl.cast(
                                                ring_offset % CACHE_BLOCK_SIZE,
                                                pl.INT64,
                                            ),
                                        )
                                    pl.write(swa_lens, [query], valid_sources)
    return (
        swa_write_slots,
        swa_sources,
        swa_lens,
        next_raw_valid_ranges,
    )


@dataclass(frozen=True)
class PhaseBSwaRequest:
    """One request's pre-step four-page raw/SWA ring descriptor.

    ``raw_page_id`` remains a fixture convenience for a contiguous four-page
    span. Production metadata consumes ``raw_page_ids`` from the ragged page
    allocator and permits arbitrary physical-page permutations.
    """

    committed_tokens: int
    active: bool
    epoch: int
    raw_valid_begin: int
    raw_valid_end: int
    raw_page_id: int = SWA_SOURCE_INVALID
    raw_page_epoch: int | None = None
    raw_page_count: int = SWA_PERSISTENT_PAGES_PER_REQUEST
    raw_head: int = 0
    raw_page_ids: tuple[int, ...] = ()
    raw_page_epochs: tuple[int, ...] | None = None

    @property
    def page_ids(self) -> tuple[int, ...]:
        """Return allocator page IDs, expanding the fixture base if needed."""
        if self.raw_page_ids:
            return self.raw_page_ids
        if self.raw_page_id < 0:
            return ()
        return tuple(
            self.raw_page_id + relative
            for relative in range(self.raw_page_count)
        )

    @property
    def page_epochs(self) -> tuple[int, ...]:
        """Return per-page epochs, defaulting every page to the request epoch."""
        if self.raw_page_epochs is not None:
            return self.raw_page_epochs
        epoch = self.epoch if self.raw_page_epoch is None else self.raw_page_epoch
        return (epoch,) * self.raw_page_count


@dataclass(frozen=True)
class PhaseBSwaQuery:
    """One packed current-step query used by the SWA metadata ABI."""

    request_id: int
    position: int


@dataclass(frozen=True)
class PhaseBSwaMetadataResult:
    """Host reference outputs for the Phase B SWA metadata entry."""

    request_query_offsets: tuple[int, ...]
    swa_write_slots: tuple[int, ...]
    swa_sources: tuple[tuple[int, ...], ...]
    swa_lens: tuple[int, ...]
    next_raw_valid_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _PhaseBSwaFixture:
    requests: tuple[PhaseBSwaRequest, ...]
    queries: tuple[PhaseBSwaQuery, ...]
    request_query_offsets: tuple[int, ...]


def _phase_b_request_query_offsets(
    requests: list[PhaseBSwaRequest] | tuple[PhaseBSwaRequest, ...],
    queries: list[PhaseBSwaQuery] | tuple[PhaseBSwaQuery, ...],
) -> tuple[int, ...]:
    """Validate grouped, contiguous current queries and return their offsets."""
    offsets = [0]
    cursor = 0
    for request_id in range(len(requests)):
        previous_position: int | None = None
        while cursor < len(queries) and queries[cursor].request_id == request_id:
            position = queries[cursor].position
            if position < 0 or position >= M.max_position_embeddings:
                raise ValueError(
                    f"query position {position} out of [0, {M.max_position_embeddings})"
                )
            if previous_position is not None and position != previous_position + 1:
                raise ValueError(
                    "queries for each request must have contiguous increasing "
                    "positions"
                )
            previous_position = position
            cursor += 1
        offsets.append(cursor)
    if cursor != len(queries):
        raise ValueError("queries must be grouped by ascending request id")
    return tuple(offsets)


def _phase_b_raw_descriptor_is_usable(request: PhaseBSwaRequest) -> bool:
    """Whether a raw descriptor can safely read or receive persistent rows."""
    if not request.active:
        return False
    if request.committed_tokens < 0 or request.committed_tokens > M.max_position_embeddings:
        return False
    if request.epoch < 0:
        return False
    try:
        validate_swa_raw_descriptor(
            request.page_ids,
            request.raw_valid_begin,
            request.raw_valid_end,
            request.page_epochs,
            request.epoch,
            active=True,
            raw_page_count=request.raw_page_count,
            raw_head=request.raw_head,
        )
    except ValueError:
        return False
    if request.raw_valid_end > request.committed_tokens:
        return False
    if any(
        page > SWA_SOURCE_INT32_MAX // CACHE_BLOCK_SIZE
        for page in request.page_ids
    ):
        return False
    return all(epoch == request.epoch for epoch in request.page_epochs)


def build_phase_b_swa_metadata(
    requests: list[PhaseBSwaRequest] | tuple[PhaseBSwaRequest, ...],
    queries: list[PhaseBSwaQuery] | tuple[PhaseBSwaQuery, ...],
    *,
    request_query_offsets: list[int] | tuple[int, ...] | None = None,
) -> PhaseBSwaMetadataResult:
    """Pure-Python Phase B reference for the four-page SWA ring metadata.

    ``committed_tokens`` is the logical visibility target. The raw descriptor
    remains the pre-step persistent history, so current-step positions are
    selected from the causal negative-source overlay and never pre-written to
    the persistent ring.
    """
    for request in requests:
        if request.committed_tokens < 0 or request.committed_tokens > M.max_position_embeddings:
            raise ValueError(
                "committed_tokens must lie in the context ceiling, got "
                f"{request.committed_tokens}"
            )
        if request.epoch < 0:
            raise ValueError(f"request epoch must be non-negative, got {request.epoch}")

    derived_offsets = _phase_b_request_query_offsets(requests, queries)
    if request_query_offsets is not None:
        supplied_offsets = tuple(int(offset) for offset in request_query_offsets)
        if supplied_offsets != derived_offsets:
            raise ValueError(
                "request_query_offsets does not match grouped query layout: "
                f"got {supplied_offsets}, expected {derived_offsets}"
            )

    write_slots = [SWA_SOURCE_INVALID] * len(queries)
    sources = [
        [SWA_SOURCE_INVALID] * PHASE_B_SOURCE_WIDTH for _ in queries
    ]
    lens = [0] * len(queries)
    next_ranges = [(0, 0)] * len(requests)

    for request_id, request in enumerate(requests):
        request_begin = derived_offsets[request_id]
        request_end = derived_offsets[request_id + 1]
        overlay_by_position = {
            queries[query].position: query
            for query in range(request_begin, request_end)
        }
        raw_usable = _phase_b_raw_descriptor_is_usable(request)
        raw_page_ids = request.page_ids if raw_usable else ()
        raw_page_epochs = request.page_epochs if raw_usable else ()

        if raw_usable:
            if request_end > request_begin:
                next_ranges[request_id] = swa_next_valid_range(
                    request.raw_valid_end,
                    queries[request_end - 1].position,
                )
            elif request.raw_valid_end > 0:
                next_ranges[request_id] = swa_next_valid_range(
                    request.raw_valid_end,
                    request.raw_valid_end - 1,
                )

        if not request.active:
            continue
        for query in range(request_begin, request_end):
            target = queries[query]
            source_row, source_len = swa_window_sources_reference(
                committed_tokens=request.committed_tokens,
                target_position=target.position,
                raw_page_ids=raw_page_ids,
                raw_valid_begin=request.raw_valid_begin,
                raw_valid_end=request.raw_valid_end,
                page_epochs=raw_page_epochs,
                request_epoch=request.epoch,
                overlay_query_by_position=overlay_by_position,
                active=True,
            )
            sources[query] = source_row
            lens[query] = source_len
            if raw_usable:
                write_slots[query] = swa_write_row(
                    request.page_ids,
                    target.position,
                    True,
                    request.page_epochs,
                    request.epoch,
                )

    return PhaseBSwaMetadataResult(
        request_query_offsets=derived_offsets,
        swa_write_slots=tuple(write_slots),
        swa_sources=tuple(tuple(row) for row in sources),
        swa_lens=tuple(lens),
        next_raw_valid_ranges=tuple(next_ranges),
    )


def _phase_b_fixture_from_lengths(
    lengths: list[int],
    *,
    page_ids: list[int],
) -> _PhaseBSwaFixture:
    requests: list[PhaseBSwaRequest] = []
    queries: list[PhaseBSwaQuery] = []
    for request_id, (length, page_id) in enumerate(zip(lengths, page_ids)):
        if length <= 0 or length > M.max_position_embeddings:
            raise ValueError(f"fixture length {length} is invalid")
        position = length - 1
        raw_valid_end = position
        raw_valid_begin = max(0, raw_valid_end - SWA_PERSISTENT_ROWS_PER_REQUEST)
        requests.append(
            PhaseBSwaRequest(
                committed_tokens=length,
                active=True,
                epoch=request_id + 3,
                raw_valid_begin=raw_valid_begin,
                raw_valid_end=raw_valid_end,
                raw_page_id=page_id,
            )
        )
        queries.append(PhaseBSwaQuery(request_id=request_id, position=position))
    offsets = _phase_b_request_query_offsets(requests, queries)
    return _PhaseBSwaFixture(tuple(requests), tuple(queries), offsets)


def _case_swa_length_matrix() -> _PhaseBSwaFixture:
    return _phase_b_fixture_from_lengths(
        [1, 127, 128, 129, 12288, 16384, M.max_position_embeddings],
        page_ids=[0, 4, 8, 12, 16, 20, 24],
    )


def _case_swa_ring_wrap() -> _PhaseBSwaFixture:
    requests = (
        PhaseBSwaRequest(129, True, 5, 0, 127, raw_page_id=19),
        PhaseBSwaRequest(130, True, 9, 0, 128, raw_page_id=5),
    )
    queries = (
        PhaseBSwaQuery(0, 127),
        PhaseBSwaQuery(0, 128),
        PhaseBSwaQuery(1, 128),
        PhaseBSwaQuery(1, 129),
    )
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


def _case_swa_two_step_rollover() -> _PhaseBSwaFixture:
    """Second S=2 step after positions 127/128 rolled the ring to [1, 129)."""
    requests = (
        PhaseBSwaRequest(131, True, 7, 1, 129, raw_page_id=19),
    )
    queries = (PhaseBSwaQuery(0, 129), PhaseBSwaQuery(0, 130))
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


def _case_swa_causal_overlay() -> _PhaseBSwaFixture:
    requests = (
        PhaseBSwaRequest(129, True, 6, 0, 127, raw_page_id=13),
    )
    queries = (PhaseBSwaQuery(0, 127), PhaseBSwaQuery(0, 128))
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


def _case_swa_page_permutation() -> _PhaseBSwaFixture:
    requests = (
        PhaseBSwaRequest(130, True, 4, 1, 129, raw_page_ids=(29, 101, 7, 55)),
        PhaseBSwaRequest(130, True, 5, 1, 129, raw_page_ids=(3, 88, 17, 64)),
        PhaseBSwaRequest(130, True, 6, 1, 129, raw_page_ids=(117, 5, 92, 41)),
    )
    queries = (
        PhaseBSwaQuery(0, 129),
        PhaseBSwaQuery(1, 129),
        PhaseBSwaQuery(2, 129),
    )
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


def _case_swa_inactive_lane() -> _PhaseBSwaFixture:
    requests = (
        PhaseBSwaRequest(128, True, 3, 0, 127, raw_page_id=9),
        PhaseBSwaRequest(
            1,
            False,
            4,
            0,
            0,
            raw_page_id=SWA_SOURCE_INVALID,
            raw_page_count=0,
        ),
    )
    queries = (PhaseBSwaQuery(0, 127), PhaseBSwaQuery(1, 0))
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


def _case_swa_stale_epoch_rejected() -> _PhaseBSwaFixture:
    requests = (
        PhaseBSwaRequest(
            129,
            True,
            8,
            0,
            128,
            raw_page_id=7,
            raw_page_epoch=6,
        ),
    )
    queries = (PhaseBSwaQuery(0, 128),)
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


def _case_swa_heterogeneous_lengths() -> _PhaseBSwaFixture:
    return _phase_b_fixture_from_lengths(
        [1, 129, 12288, M.max_position_embeddings],
        page_ids=[0, 4, 8, 12],
    )


def _case_swa_one_m_tail() -> _PhaseBSwaFixture:
    return _phase_b_fixture_from_lengths(
        [M.max_position_embeddings],
        page_ids=[37],
    )


def _case_swa_missing_page() -> _PhaseBSwaFixture:
    requests = (
        PhaseBSwaRequest(
            129,
            True,
            2,
            0,
            128,
            raw_page_id=SWA_SOURCE_INVALID,
            raw_page_count=0,
        ),
    )
    queries = (PhaseBSwaQuery(0, 128),)
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


def _case_swa_mixed_validity() -> _PhaseBSwaFixture:
    """One grouped ragged batch mixing valid, inactive, stale, and missing rings.

    The first two requests deliberately use different physical page ids (the
    logical window is identical), while the remaining requests exercise the
    rejection paths.  Keeping all requests in one fixture catches accidental
    cross-request state or page-table reuse in the metadata builder.
    """
    requests = (
        PhaseBSwaRequest(130, True, 4, 1, 129, raw_page_ids=(29, 101, 7, 55)),
        PhaseBSwaRequest(130, True, 5, 1, 129, raw_page_ids=(3, 88, 17, 64)),
        PhaseBSwaRequest(
            1,
            False,
            6,
            0,
            0,
            raw_page_id=SWA_SOURCE_INVALID,
            raw_page_count=0,
        ),
        PhaseBSwaRequest(
            129,
            True,
            7,
            0,
            128,
            raw_page_id=11,
            raw_page_epoch=6,
        ),
        PhaseBSwaRequest(
            129,
            True,
            8,
            0,
            128,
            raw_page_id=SWA_SOURCE_INVALID,
            raw_page_count=0,
        ),
    )
    queries = (
        PhaseBSwaQuery(0, 128),
        PhaseBSwaQuery(0, 129),
        PhaseBSwaQuery(1, 129),
        PhaseBSwaQuery(2, 0),
        PhaseBSwaQuery(3, 128),
        PhaseBSwaQuery(4, 128),
    )
    return _PhaseBSwaFixture(
        requests,
        queries,
        _phase_b_request_query_offsets(requests, queries),
    )


_PHASE_B_CASES = {
    "swa_length_matrix": _case_swa_length_matrix,
    "swa_ring_wrap": _case_swa_ring_wrap,
    "swa_two_step_rollover": _case_swa_two_step_rollover,
    "swa_causal_overlay": _case_swa_causal_overlay,
    "swa_page_permutation": _case_swa_page_permutation,
    "swa_inactive_lane": _case_swa_inactive_lane,
    "swa_stale_epoch_rejected": _case_swa_stale_epoch_rejected,
    "swa_heterogeneous_lengths": _case_swa_heterogeneous_lengths,
    "swa_one_m_tail": _case_swa_one_m_tail,
    "swa_missing_page": _case_swa_missing_page,
    "swa_mixed_validity": _case_swa_mixed_validity,
}


def _assert_phase_b_source_layout(result: PhaseBSwaMetadataResult) -> None:
    for source_row, source_len in zip(result.swa_sources, result.swa_lens):
        assert len(source_row) == PHASE_B_SOURCE_WIDTH
        assert 0 <= source_len <= PHASE_B_SOURCE_WIDTH
        assert all(source != SWA_SOURCE_INVALID for source in source_row[:source_len])
        assert all(source == SWA_SOURCE_INVALID for source in source_row[source_len:])
        assert all(
            source >= 0 or source <= SWA_SOURCE_OVERLAY_BASE
            for source in source_row[:source_len]
        )


def _phase_b_fixture_ring_row(request: PhaseBSwaRequest, position: int) -> int:
    ring_offset = position % SWA_PERSISTENT_ROWS_PER_REQUEST
    relative_page = ring_offset // CACHE_BLOCK_SIZE
    return (
        request.page_ids[relative_page] * CACHE_BLOCK_SIZE
        + ring_offset % CACHE_BLOCK_SIZE
    )


def _assert_phase_b_case(
    case: str,
    fixture: _PhaseBSwaFixture,
    result: PhaseBSwaMetadataResult,
) -> None:
    _assert_phase_b_source_layout(result)
    assert result.request_query_offsets == fixture.request_query_offsets

    if case in ("swa_length_matrix", "swa_heterogeneous_lengths"):
        for query, request in enumerate(fixture.requests):
            position = fixture.queries[query].position
            expected_len = min(request.committed_tokens, PHASE_B_SOURCE_WIDTH)
            assert result.swa_lens[query] == expected_len
            assert result.swa_write_slots[query] == _phase_b_fixture_ring_row(
                request, position
            )
            assert result.swa_sources[query][expected_len - 1] == encode_swa_overlay_source(query)
            assert result.next_raw_valid_ranges[query] == (
                max(0, request.committed_tokens - PHASE_B_SOURCE_WIDTH),
                request.committed_tokens,
            )
        return

    if case == "swa_ring_wrap":
        assert result.swa_write_slots == (22 * 32 + 31, 19 * 32, 5 * 32, 5 * 32 + 1)
        assert result.swa_lens == (128, 128, 128, 128)
        assert encode_swa_overlay_source(1) not in result.swa_sources[0][:128]
        assert result.swa_sources[1][126:128] == (-2, -3)
        assert encode_swa_overlay_source(3) not in result.swa_sources[2][:128]
        assert result.swa_sources[3][126:128] == (-4, -5)
        assert result.next_raw_valid_ranges == ((1, 129), (2, 130))
        return

    if case == "swa_two_step_rollover":
        first_step = build_phase_b_swa_metadata(
            (PhaseBSwaRequest(129, True, 7, 0, 127, raw_page_id=19),),
            (PhaseBSwaQuery(0, 127), PhaseBSwaQuery(0, 128)),
        )
        assert first_step.next_raw_valid_ranges == ((1, 129),)
        assert first_step.next_raw_valid_ranges[0] == (
            fixture.requests[0].raw_valid_begin,
            fixture.requests[0].raw_valid_end,
        )
        assert result.swa_write_slots == (19 * 32 + 1, 19 * 32 + 2)
        assert result.swa_lens == (128, 128)
        assert result.swa_sources[0][0] == 19 * 32 + 2
        assert result.swa_sources[0][-1] == encode_swa_overlay_source(0)
        assert result.swa_sources[1][-2:] == (
            encode_swa_overlay_source(0),
            encode_swa_overlay_source(1),
        )
        assert result.next_raw_valid_ranges == ((3, 131),)
        return

    if case == "swa_causal_overlay":
        assert result.swa_lens == (128, 128)
        assert encode_swa_overlay_source(1) not in result.swa_sources[0][:128]
        assert result.swa_sources[0][127] == encode_swa_overlay_source(0)
        assert result.swa_sources[1][126:128] == (
            encode_swa_overlay_source(0),
            encode_swa_overlay_source(1),
        )
        return

    if case == "swa_page_permutation":
        assert [request.page_ids for request in fixture.requests] == [
            (29, 101, 7, 55),
            (3, 88, 17, 64),
            (117, 5, 92, 41),
        ]
        for query, request in enumerate(fixture.requests):
            for source in result.swa_sources[query][:result.swa_lens[query]]:
                if source >= 0:
                    assert source // CACHE_BLOCK_SIZE in request.page_ids
        return

    if case == "swa_inactive_lane":
        assert result.swa_lens[1] == 0
        assert result.swa_write_slots[1] == SWA_SOURCE_INVALID
        assert result.swa_sources[1] == (SWA_SOURCE_INVALID,) * PHASE_B_SOURCE_WIDTH
        assert result.next_raw_valid_ranges[1] == (0, 0)
        return

    if case in ("swa_stale_epoch_rejected", "swa_missing_page"):
        assert result.swa_write_slots == (SWA_SOURCE_INVALID,)
        assert result.swa_lens == (1,)
        assert result.swa_sources[0][0] == encode_swa_overlay_source(0)
        assert result.next_raw_valid_ranges == ((0, 0),)
        return

    if case == "swa_mixed_validity":
        # Requests 0/1 are independent valid rings despite their permuted
        # physical page ids.  Their positive sources must stay request-local.
        assert result.swa_write_slots[0:3] == (
            29 * CACHE_BLOCK_SIZE + 0,
            29 * CACHE_BLOCK_SIZE + 1,
            3 * CACHE_BLOCK_SIZE + 1,
        )
        for query, request_id in ((0, 0), (1, 0), (2, 1)):
            assert all(
                source < 0
                or source // CACHE_BLOCK_SIZE
                in fixture.requests[request_id].page_ids
                for source in result.swa_sources[query][: result.swa_lens[query]]
            )
        # The inactive, stale, and missing requests have no persistent write
        # slot or next-range, but their own current token remains a causal
        # overlay source.  No invalid request may affect a valid neighbour.
        assert result.swa_lens[3:] == (0, 1, 1)
        assert result.swa_write_slots[3:] == (
            SWA_SOURCE_INVALID,
            SWA_SOURCE_INVALID,
            SWA_SOURCE_INVALID,
        )
        assert result.swa_sources[3] == (SWA_SOURCE_INVALID,) * PHASE_B_SOURCE_WIDTH
        assert result.swa_sources[4][0] == encode_swa_overlay_source(4)
        assert result.swa_sources[5][0] == encode_swa_overlay_source(5)
        assert result.next_raw_valid_ranges[2:] == ((0, 0), (0, 0), (0, 0))
        return

    if case == "swa_one_m_tail":
        assert result.swa_lens == (PHASE_B_SOURCE_WIDTH,)
        assert result.swa_write_slots == (40 * 32 + 31,)
        assert result.swa_sources[0][-1] == encode_swa_overlay_source(0)
        assert result.next_raw_valid_ranges == (
            (M.max_position_embeddings - PHASE_B_SOURCE_WIDTH, M.max_position_embeddings),
        )
        return

    raise AssertionError(f"no checker for Phase B case {case!r}")


def run_phase_b_case(case: str) -> bool:
    """Run one pure-Python Phase B fixture against explicit contract checks."""
    if case not in _PHASE_B_CASES:
        raise ValueError(f"unknown Phase B case: {case!r}")
    fixture = _PHASE_B_CASES[case]()
    try:
        result = build_phase_b_swa_metadata(
            fixture.requests,
            fixture.queries,
            request_query_offsets=fixture.request_query_offsets,
        )
        _assert_phase_b_case(case, fixture, result)
    except AssertionError as error:
        print(f"[Phase B host case={case}] FAIL: {error}")
        return False
    return True


def _pack_phase_b_swa_inputs(fixture: _PhaseBSwaFixture):
    """Pack four-page raw rings into the existing Phase A ragged page layout."""
    import torch

    from utils import swa_request_query_offsets

    request_count = len(fixture.requests)
    query_request_ids = torch.tensor(
        [query.request_id for query in fixture.queries],
        dtype=torch.int32,
    )
    derived_offsets = swa_request_query_offsets(
        query_request_ids,
        request_count=request_count,
    )
    if tuple(int(offset) for offset in derived_offsets.tolist()) != fixture.request_query_offsets:
        raise ValueError("fixture request_query_offsets is not grouped")

    page_offsets = torch.zeros(
        (PHASE_A_CACHE_GROUPS, request_count + 1),
        dtype=torch.int32,
    )
    page_windows = torch.zeros(
        (PHASE_A_CACHE_GROUPS, request_count, 3),
        dtype=torch.int32,
    )
    page_records: list[list[int]] = []
    page_offsets[PHASE_B_RAW_GROUP, 0] = 0
    for request_index, request in enumerate(fixture.requests):
        page_records.extend(
            [int(page), int(epoch)]
            for page, epoch in zip(request.page_ids, request.page_epochs)
        )
        page_offsets[PHASE_B_RAW_GROUP, request_index + 1] = len(page_records)
        page_windows[PHASE_B_RAW_GROUP, request_index, 0:2] = torch.tensor(
            (request.raw_valid_begin, request.raw_valid_end), dtype=torch.int32
        )
        page_windows[PHASE_B_RAW_GROUP, request_index, 2] = request.raw_head
    if not page_records:
        page_records.append([SWA_SOURCE_INVALID, 0])

    return {
        "request_state": torch.tensor(
            [
                [request.committed_tokens, int(request.active), request.epoch]
                for request in fixture.requests
            ],
            dtype=torch.int32,
        ),
        "query_request_ids": query_request_ids,
        "request_query_offsets": torch.tensor(
            fixture.request_query_offsets,
            dtype=torch.int32,
        ),
        "position_ids": torch.tensor(
            [query.position for query in fixture.queries],
            dtype=torch.int32,
        ),
        "pages": torch.tensor(page_records, dtype=torch.int32),
        "page_offsets": page_offsets,
        "page_windows": page_windows,
    }


def golden_phase_b_swa_metadata(tensors):
    """Independent torch golden for :func:`phase_b_swa_metadata`."""
    request_state = tensors["request_state"]
    request_ids = tensors["query_request_ids"]
    request_offsets = tensors["request_query_offsets"]
    positions = tensors["position_ids"]
    pages = tensors["pages"]
    page_offsets = tensors["page_offsets"]
    page_windows = tensors["page_windows"]
    write_slots = tensors["swa_write_slots"]
    sources = tensors["swa_sources"]
    lens = tensors["swa_lens"]
    next_ranges = tensors["next_raw_valid_ranges"]

    request_count = int(request_state.shape[0])
    query_count = int(positions.shape[0])
    write_slots.fill_(SWA_SOURCE_INVALID)
    sources.fill_(SWA_SOURCE_INVALID)
    lens.zero_()
    next_ranges.zero_()
    if int(request_offsets.shape[0]) != request_count + 1:
        return

    for request in range(request_count):
        request_begin = int(request_offsets[request])
        request_end = int(request_offsets[request + 1])
        query_range_valid = (
            0 <= request_begin <= request_end <= query_count
        )
        first_position = 0
        previous_position = 0
        if query_range_valid:
            for query in range(request_begin, request_end):
                query_position = int(positions[query])
                if int(request_ids[query]) != request:
                    query_range_valid = False
                if query_position < 0 or query_position >= M.max_position_embeddings:
                    query_range_valid = False
                if query == request_begin:
                    first_position = query_position
                elif query_position != previous_position + 1:
                    query_range_valid = False
                previous_position = query_position
        if not query_range_valid:
            continue

        committed = int(request_state[request, 0])
        active = bool(int(request_state[request, 1]))
        epoch = int(request_state[request, 2])
        raw_valid_begin = int(page_windows[PHASE_B_RAW_GROUP, request, 0])
        raw_valid_end = int(page_windows[PHASE_B_RAW_GROUP, request, 1])
        raw_head = int(page_windows[PHASE_B_RAW_GROUP, request, 2])
        raw_page_begin = int(page_offsets[PHASE_B_RAW_GROUP, request])
        raw_page_end = int(page_offsets[PHASE_B_RAW_GROUP, request + 1])
        raw_page_total = raw_page_end - raw_page_begin
        raw_usable = False
        raw_page_ids: list[int] = []
        raw_page_epochs: list[int] = []
        if active and 0 <= committed <= M.max_position_embeddings and epoch >= 0:
            if raw_page_total == SWA_PERSISTENT_PAGES_PER_REQUEST and raw_head == 0:
                if 0 <= raw_page_begin < raw_page_end <= int(pages.shape[0]):
                    if (
                        0 <= raw_valid_begin <= raw_valid_end <= committed
                        and raw_valid_end <= M.max_position_embeddings
                        and raw_valid_end - raw_valid_begin
                        <= SWA_PERSISTENT_ROWS_PER_REQUEST
                    ):
                        raw_page_ids = [
                            int(pages[raw_page_begin + relative, 0])
                            for relative in range(SWA_PERSISTENT_PAGES_PER_REQUEST)
                        ]
                        raw_page_epochs = [
                            int(pages[raw_page_begin + relative, 1])
                            for relative in range(SWA_PERSISTENT_PAGES_PER_REQUEST)
                        ]
                        if all(
                            0 <= page <= SWA_SOURCE_INT32_MAX // CACHE_BLOCK_SIZE
                            for page in raw_page_ids
                        ) and all(page_epoch == epoch for page_epoch in raw_page_epochs):
                            raw_usable = True

        if raw_usable:
            if request_end > request_begin:
                last_position = int(positions[request_end - 1])
                step_end = max(raw_valid_end, last_position + 1)
            else:
                step_end = raw_valid_end
            next_ranges[request, 0] = max(
                0,
                step_end - SWA_PERSISTENT_ROWS_PER_REQUEST,
            )
            next_ranges[request, 1] = step_end

        if not active or not (0 <= committed <= M.max_position_embeddings):
            continue
        for query in range(request_begin, request_end):
            position = int(positions[query])
            visible_end = min(committed, position + 1, M.max_position_embeddings)
            window_begin = max(0, visible_end - PHASE_B_SOURCE_WIDTH)
            valid_sources = 0
            for logical_position in range(window_begin, visible_end):
                source = SWA_SOURCE_INVALID
                overlay_query = request_begin + logical_position - first_position
                if request_begin <= overlay_query < request_end and overlay_query <= query:
                    if overlay_query <= SWA_SOURCE_MAX_OVERLAY_QUERY:
                        if (
                            int(request_ids[overlay_query]) == request
                            and int(positions[overlay_query]) == logical_position
                            and int(positions[overlay_query]) <= position
                        ):
                            source = SWA_SOURCE_OVERLAY_BASE - overlay_query
                if (
                    source == SWA_SOURCE_INVALID
                    and raw_usable
                    and raw_valid_begin <= logical_position < raw_valid_end
                ):
                    ring_offset = logical_position % SWA_PERSISTENT_ROWS_PER_REQUEST
                    relative_page = ring_offset // CACHE_BLOCK_SIZE
                    source = (
                        raw_page_ids[relative_page] * CACHE_BLOCK_SIZE
                        + ring_offset % CACHE_BLOCK_SIZE
                    )
                if source != SWA_SOURCE_INVALID:
                    sources[query, valid_sources] = source
                    valid_sources += 1
            lens[query] = valid_sources
            if raw_usable:
                ring_offset = position % SWA_PERSISTENT_ROWS_PER_REQUEST
                relative_page = ring_offset // CACHE_BLOCK_SIZE
                write_slots[query] = (
                    raw_page_ids[relative_page] * CACHE_BLOCK_SIZE
                    + ring_offset % CACHE_BLOCK_SIZE
                )


def build_phase_b_swa_tensor_specs(case: str):
    """Build a real-device metadata fixture for a named Phase B case."""
    import torch
    from golden import TensorSpec

    if case not in _PHASE_B_CASES:
        raise ValueError(f"unknown Phase B case: {case!r}")
    fixture = _PHASE_B_CASES[case]()
    inputs = _pack_phase_b_swa_inputs(fixture)
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in inputs.items()
    ]
    query_count = len(fixture.queries)
    request_count = len(fixture.requests)
    specs.extend(
        [
            TensorSpec(
                "swa_write_slots",
                [query_count],
                torch.int64,
                is_output=True,
            ),
            TensorSpec(
                "swa_sources",
                [query_count, PHASE_B_SOURCE_WIDTH],
                torch.int32,
                is_output=True,
            ),
            TensorSpec(
                "swa_lens",
                [query_count],
                torch.int32,
                is_output=True,
            ),
            TensorSpec(
                "next_raw_valid_ranges",
                [request_count, 2],
                torch.int32,
                is_output=True,
            ),
        ]
    )
    return specs


# Phase C uses an exact packed-work ABI.  The dynamic dimensions below are
# runtime tensor extents; the 128-row shard granularity remains compile time.
PHASE_C_REQUESTS_DYN = pl.dynamic("PHASE_C_REQUESTS_DYN")
PHASE_C_REQUEST_OFFSETS_DYN = pl.dynamic("PHASE_C_REQUEST_OFFSETS_DYN")
PHASE_C_QUERIES_DYN = pl.dynamic("PHASE_C_QUERIES_DYN")
PHASE_C_QUERY_OFFSETS_DYN = pl.dynamic("PHASE_C_QUERY_OFFSETS_DYN")
PHASE_C_MAIN_PAGES_DYN = pl.dynamic("PHASE_C_MAIN_PAGES_DYN")
PHASE_C_WORK_DYN = pl.dynamic("PHASE_C_WORK_DYN")
PHASE_C_EVENTS_DYN = pl.dynamic("PHASE_C_EVENTS_DYN")


@pl.jit
def phase_c_hca_metadata(
    request_state: pl.Tensor[[PHASE_C_REQUESTS_DYN, 3], pl.INT32],
    query_request_ids: pl.Tensor[[PHASE_C_QUERIES_DYN], pl.INT32],
    request_query_offsets: pl.Tensor[[PHASE_C_REQUEST_OFFSETS_DYN], pl.INT32],
    position_ids: pl.Tensor[[PHASE_C_QUERIES_DYN], pl.INT32],
    hca_pages: pl.Tensor[[PHASE_C_MAIN_PAGES_DYN, 2], pl.INT32],
    hca_page_offsets: pl.Tensor[[PHASE_C_REQUEST_OFFSETS_DYN], pl.INT32],
    hca_windows: pl.Tensor[[PHASE_C_REQUESTS_DYN, 3], pl.INT32],
    state_pages: pl.Tensor[
        [PHASE_C_REQUESTS_DYN, HCA_STATE_PAGES_PER_REQUEST, 2], pl.INT32
    ],
    state_valid_ranges: pl.Tensor[[PHASE_C_REQUESTS_DYN, 2], pl.INT32],
    hca_visible_rows: pl.Out[pl.Tensor[[PHASE_C_QUERIES_DYN], pl.INT32]],
    hca_query_work_offsets: pl.Out[
        pl.Tensor[[PHASE_C_QUERY_OFFSETS_DYN], pl.INT32]
    ],
    hca_work_query_ids: pl.Out[pl.Tensor[[PHASE_C_WORK_DYN], pl.INT32]],
    hca_work_row_begin: pl.Out[pl.Tensor[[PHASE_C_WORK_DYN], pl.INT32]],
    hca_work_valid_rows: pl.Out[pl.Tensor[[PHASE_C_WORK_DYN], pl.INT32]],
    hca_event_query_ids: pl.Out[pl.Tensor[[PHASE_C_EVENTS_DYN], pl.INT32]],
    hca_event_rows: pl.Out[pl.Tensor[[PHASE_C_EVENTS_DYN], pl.INT32]],
    hca_event_write_slots: pl.Out[pl.Tensor[[PHASE_C_EVENTS_DYN], pl.INT64]],
    hca_request_event_indices: pl.Out[
        pl.Tensor[[PHASE_C_REQUESTS_DYN], pl.INT32]
    ],
    hca_state_write_slots: pl.Out[pl.Tensor[[PHASE_C_QUERIES_DYN], pl.INT64]],
    next_hca_valid_ranges: pl.Out[
        pl.Tensor[[PHASE_C_REQUESTS_DYN, 2], pl.INT32]
    ],
    next_hca_state_valid_ranges: pl.Out[
        pl.Tensor[[PHASE_C_REQUESTS_DYN, 2], pl.INT32]
    ],
):
    """Lower HCA main/state descriptors to exact runtime packed work."""
    request_state.bind_dynamic(0, PHASE_C_REQUESTS_DYN)
    query_request_ids.bind_dynamic(0, PHASE_C_QUERIES_DYN)
    request_query_offsets.bind_dynamic(0, PHASE_C_REQUEST_OFFSETS_DYN)
    position_ids.bind_dynamic(0, PHASE_C_QUERIES_DYN)
    hca_pages.bind_dynamic(0, PHASE_C_MAIN_PAGES_DYN)
    hca_page_offsets.bind_dynamic(0, PHASE_C_REQUEST_OFFSETS_DYN)
    hca_windows.bind_dynamic(0, PHASE_C_REQUESTS_DYN)
    state_pages.bind_dynamic(0, PHASE_C_REQUESTS_DYN)
    state_valid_ranges.bind_dynamic(0, PHASE_C_REQUESTS_DYN)
    hca_visible_rows.bind_dynamic(0, PHASE_C_QUERIES_DYN)
    hca_query_work_offsets.bind_dynamic(0, PHASE_C_QUERY_OFFSETS_DYN)
    hca_work_query_ids.bind_dynamic(0, PHASE_C_WORK_DYN)
    hca_work_row_begin.bind_dynamic(0, PHASE_C_WORK_DYN)
    hca_work_valid_rows.bind_dynamic(0, PHASE_C_WORK_DYN)
    hca_event_query_ids.bind_dynamic(0, PHASE_C_EVENTS_DYN)
    hca_event_rows.bind_dynamic(0, PHASE_C_EVENTS_DYN)
    hca_event_write_slots.bind_dynamic(0, PHASE_C_EVENTS_DYN)
    hca_request_event_indices.bind_dynamic(0, PHASE_C_REQUESTS_DYN)
    hca_state_write_slots.bind_dynamic(0, PHASE_C_QUERIES_DYN)
    next_hca_valid_ranges.bind_dynamic(0, PHASE_C_REQUESTS_DYN)
    next_hca_state_valid_ranges.bind_dynamic(0, PHASE_C_REQUESTS_DYN)

    request_count = pl.tensor.dim(request_state, 0)
    query_count = pl.tensor.dim(position_ids, 0)
    offset_count = pl.tensor.dim(request_query_offsets, 0)
    page_count = pl.tensor.dim(hca_pages, 0)
    work_capacity = pl.tensor.dim(hca_work_query_ids, 0)
    event_capacity = pl.tensor.dim(hca_event_query_ids, 0)

    for core in pl.spmd(1, name_hint="decode_phase_c_hca_metadata"):
        for query in pl.range(core, query_count):
            pl.write(hca_visible_rows, [query], pl.cast(0, pl.INT32))
            pl.write(
                hca_state_write_slots, [query], pl.cast(PHASE_A_INVALID_SLOT, pl.INT64)
            )
        for item in pl.range(core, work_capacity):
            pl.write(hca_work_query_ids, [item], pl.cast(-1, pl.INT32))
            pl.write(hca_work_row_begin, [item], pl.cast(0, pl.INT32))
            pl.write(hca_work_valid_rows, [item], pl.cast(0, pl.INT32))
        for event in pl.range(core, event_capacity):
            pl.write(hca_event_query_ids, [event], pl.cast(-1, pl.INT32))
            pl.write(hca_event_rows, [event], pl.cast(-1, pl.INT32))
            pl.write(
                hca_event_write_slots,
                [event],
                pl.cast(PHASE_A_INVALID_SLOT, pl.INT64),
            )
        for offset in pl.range(core, pl.tensor.dim(hca_query_work_offsets, 0)):
            pl.write(hca_query_work_offsets, [offset], pl.cast(0, pl.INT32))
        for request in pl.range(core, request_count):
            pl.write(
                hca_request_event_indices,
                [request],
                pl.cast(-1, pl.INT32),
            )
            pl.write(next_hca_valid_ranges, [request, 0], pl.cast(0, pl.INT32))
            pl.write(next_hca_valid_ranges, [request, 1], pl.cast(0, pl.INT32))
            pl.write(
                next_hca_state_valid_ranges, [request, 0], pl.cast(0, pl.INT32)
            )
            pl.write(
                next_hca_state_valid_ranges, [request, 1], pl.cast(0, pl.INT32)
            )

        work_cursor = pl.cast(0, pl.INT32)
        event_cursor = pl.cast(0, pl.INT32)
        offsets_valid = pl.cast(0, pl.INT32)
        if offset_count == request_count + 1:
            if pl.tensor.dim(hca_query_work_offsets, 0) == query_count + 1:
                offsets_valid = pl.cast(1, pl.INT32)

        if offsets_valid != 0:
            for request in pl.range(core, request_count):
                request_index = pl.cast(request, pl.INDEX)
                committed = pl.read(request_state, [request_index, 0])
                active = pl.read(request_state, [request_index, 1])
                epoch = pl.read(request_state, [request_index, 2])
                query_begin = pl.read(request_query_offsets, [request])
                query_end = pl.read(request_query_offsets, [request + 1])

                query_range_valid = pl.cast(1, pl.INT32)
                previous_position = pl.cast(-1, pl.INT32)
                if query_begin < 0:
                    query_range_valid = pl.cast(0, pl.INT32)
                if query_end < query_begin:
                    query_range_valid = pl.cast(0, pl.INT32)
                if query_end > query_count:
                    query_range_valid = pl.cast(0, pl.INT32)
                if query_range_valid != 0:
                    for query in pl.range(query_begin, query_end):
                        position = pl.read(position_ids, [query])
                        if pl.read(query_request_ids, [query]) != request:
                            query_range_valid = pl.cast(0, pl.INT32)
                        if position < 0:
                            query_range_valid = pl.cast(0, pl.INT32)
                        if position >= M.max_position_embeddings:
                            query_range_valid = pl.cast(0, pl.INT32)
                        if query > query_begin:
                            if position != previous_position + 1:
                                query_range_valid = pl.cast(0, pl.INT32)
                        previous_position = position

                main_valid = pl.cast(0, pl.INT32)
                main_begin = pl.cast(0, pl.INT32)
                main_end = pl.cast(0, pl.INT32)
                main_head = pl.cast(0, pl.INT32)
                main_page_begin = pl.cast(0, pl.INT32)
                main_page_total = pl.cast(0, pl.INT32)
                if active != 0:
                    if committed >= 0:
                        if committed <= M.max_position_embeddings:
                            if epoch >= 0:
                                if query_range_valid != 0:
                                    main_begin = pl.read(hca_windows, [request_index, 0])
                                    main_end = pl.read(hca_windows, [request_index, 1])
                                    main_head = pl.read(hca_windows, [request_index, 2])
                                    main_page_begin = pl.read(
                                        hca_page_offsets, [request_index]
                                    )
                                    main_page_end = pl.read(
                                        hca_page_offsets, [request_index + 1]
                                    )
                                    main_page_total = main_page_end - main_page_begin
                                    main_valid = pl.cast(1, pl.INT32)
                                    if main_page_total <= 0:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_page_begin < 0:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_page_end > page_count:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_begin < 0:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_end < main_begin:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_end > MAX_HCA_ROWS:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_head < 0:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_head >= main_page_total:
                                        main_valid = pl.cast(0, pl.INT32)
                                    allocated_end = (
                                        main_begin // CACHE_BLOCK_SIZE
                                        + main_page_total
                                    ) * CACHE_BLOCK_SIZE
                                    if main_end > allocated_end:
                                        main_valid = pl.cast(0, pl.INT32)
                                    if main_valid != 0:
                                        for page_offset in pl.range(main_page_total):
                                            stored_epoch = pl.read(
                                                hca_pages,
                                                [main_page_begin + page_offset, 1],
                                            )
                                            page_id = pl.read(
                                                hca_pages,
                                                [main_page_begin + page_offset, 0],
                                            )
                                            if stored_epoch != epoch:
                                                main_valid = pl.cast(0, pl.INT32)
                                            if page_id < 0:
                                                main_valid = pl.cast(0, pl.INT32)

                state_valid = pl.cast(0, pl.INT32)
                state_begin = pl.cast(0, pl.INT32)
                state_end = pl.cast(0, pl.INT32)
                if active != 0:
                    state_begin = pl.read(state_valid_ranges, [request_index, 0])
                    state_end = pl.read(state_valid_ranges, [request_index, 1])
                    state_valid = pl.cast(1, pl.INT32)
                    for state_relative_page in pl.range(HCA_STATE_PAGES_PER_REQUEST):
                        state_page_id = pl.read(
                            state_pages,
                            [request_index, state_relative_page, 0],
                        )
                        state_page_epoch = pl.read(
                            state_pages,
                            [request_index, state_relative_page, 1],
                        )
                        if state_page_id < 0:
                            state_valid = pl.cast(0, pl.INT32)
                        if state_page_epoch != epoch:
                            state_valid = pl.cast(0, pl.INT32)
                    if state_begin < 0:
                        state_valid = pl.cast(0, pl.INT32)
                    if state_end < state_begin:
                        state_valid = pl.cast(0, pl.INT32)
                    if state_end - state_begin >= HCA_STATE_ROWS_PER_REQUEST:
                        state_valid = pl.cast(0, pl.INT32)
                    if state_end > M.max_position_embeddings:
                        state_valid = pl.cast(0, pl.INT32)
                    if state_end > state_begin:
                        if state_begin // HCA_COMPRESS_RATIO != (
                            state_end - 1
                        ) // HCA_COMPRESS_RATIO:
                            state_valid = pl.cast(0, pl.INT32)

                if main_valid != 0:
                    pl.write(next_hca_valid_ranges, [request, 0], main_begin)
                    next_main_end = main_end
                    if query_end > query_begin:
                        step_end = pl.cast(
                            pl.read(position_ids, [query_end - 1]) + 1,
                            pl.INT32,
                        )
                        event_end = step_end // HCA_COMPRESS_RATIO
                        if event_end > next_main_end:
                            next_main_end = pl.cast(event_end, pl.INT32)
                    pl.write(next_hca_valid_ranges, [request, 1], next_main_end)

                if state_valid != 0:
                    next_state_begin = state_begin
                    next_state_end = state_end
                    if query_end > query_begin:
                        next_state_end = pl.cast(
                            pl.read(position_ids, [query_end - 1]) + 1,
                            pl.INT32,
                        )
                        next_state_begin = pl.cast(
                            (next_state_end // HCA_COMPRESS_RATIO)
                            * HCA_COMPRESS_RATIO,
                            pl.INT32,
                        )
                    pl.write(
                        next_hca_state_valid_ranges,
                        [request, 0],
                        next_state_begin,
                    )
                    pl.write(
                        next_hca_state_valid_ranges,
                        [request, 1],
                        next_state_end,
                    )

                if query_range_valid != 0:
                    for query in pl.range(query_begin, query_end):
                        pl.write(hca_query_work_offsets, [query], work_cursor)
                        position = pl.read(position_ids, [query])
                        if state_valid != 0:
                            state_ring_row = position % HCA_STATE_ROWS_PER_REQUEST
                            state_relative_page = (
                                state_ring_row // HCA_STATE_BLOCK_SIZE
                            )
                            state_page_id = pl.read(
                                state_pages,
                                [request_index, state_relative_page, 0],
                            )
                            pl.write(
                                hca_state_write_slots,
                                [query],
                                pl.cast(state_page_id, pl.INT64)
                                * pl.cast(HCA_STATE_BLOCK_SIZE, pl.INT64)
                                + pl.cast(
                                    state_ring_row % HCA_STATE_BLOCK_SIZE,
                                    pl.INT64,
                                ),
                            )

                        visible_begin = main_begin
                        visible_end = main_begin
                        if main_valid != 0:
                            causal_tokens = committed
                            if position + 1 < causal_tokens:
                                causal_tokens = pl.cast(position + 1, pl.INT32)
                            causal_rows = causal_tokens // HCA_COMPRESS_RATIO
                            visible_end = main_end
                            if causal_rows < visible_end:
                                visible_end = pl.cast(causal_rows, pl.INT32)
                            if visible_end < visible_begin:
                                visible_end = visible_begin

                            for overlay_query in pl.range(query_begin, query + 1):
                                overlay_position = pl.read(
                                    position_ids, [overlay_query]
                                )
                                if (
                                    (overlay_position + 1) % HCA_COMPRESS_RATIO
                                    == 0
                                ):
                                    overlay_row = overlay_position // HCA_COMPRESS_RATIO
                                    allocated_begin = (
                                        main_begin // CACHE_BLOCK_SIZE
                                    ) * CACHE_BLOCK_SIZE
                                    allocated_end = (
                                        main_begin // CACHE_BLOCK_SIZE
                                        + main_page_total
                                    ) * CACHE_BLOCK_SIZE
                                    if overlay_row >= allocated_begin:
                                        if overlay_row < allocated_end:
                                            if overlay_row == visible_end:
                                                visible_end = pl.cast(
                                                    visible_end + 1, pl.INT32
                                                )

                            if (position + 1) % HCA_COMPRESS_RATIO == 0:
                                event_row = position // HCA_COMPRESS_RATIO
                                allocated_base = main_begin // CACHE_BLOCK_SIZE
                                logical_page = (
                                    event_row // CACHE_BLOCK_SIZE
                                    - allocated_base
                                )
                                if logical_page >= 0:
                                    if logical_page < main_page_total:
                                        physical_page_offset = (
                                            main_head + logical_page
                                        ) % main_page_total
                                        physical_page = pl.read(
                                            hca_pages,
                                            [
                                                main_page_begin
                                                + physical_page_offset,
                                                0,
                                            ],
                                        )
                                        if event_cursor < event_capacity:
                                            pl.write(
                                                hca_request_event_indices,
                                                [request],
                                                event_cursor,
                                            )
                                            pl.write(
                                                hca_event_query_ids,
                                                [event_cursor],
                                                pl.cast(query, pl.INT32),
                                            )
                                            pl.write(
                                                hca_event_rows,
                                                [event_cursor],
                                                pl.cast(event_row, pl.INT32),
                                            )
                                            pl.write(
                                                hca_event_write_slots,
                                                [event_cursor],
                                                pl.cast(physical_page, pl.INT64)
                                                * pl.cast(
                                                    CACHE_BLOCK_SIZE, pl.INT64
                                                )
                                                + pl.cast(
                                                    event_row
                                                    % CACHE_BLOCK_SIZE,
                                                    pl.INT64,
                                                ),
                                            )
                                        event_cursor = pl.cast(
                                            event_cursor + 1, pl.INT32
                                        )

                        visible_rows = visible_end - visible_begin
                        pl.write(hca_visible_rows, [query], visible_rows)
                        shard_count = (
                            visible_rows + HCA_ROWS_PER_SHARD - 1
                        ) // HCA_ROWS_PER_SHARD
                        for shard in pl.range(shard_count):
                            row_begin = pl.cast(
                                visible_begin + shard * HCA_ROWS_PER_SHARD,
                                pl.INT32,
                            )
                            valid_rows = pl.cast(
                                visible_end - row_begin, pl.INT32
                            )
                            if valid_rows > HCA_ROWS_PER_SHARD:
                                valid_rows = pl.cast(HCA_ROWS_PER_SHARD, pl.INT32)
                            if work_cursor < work_capacity:
                                pl.write(
                                    hca_work_query_ids,
                                    [work_cursor],
                                    pl.cast(query, pl.INT32),
                                )
                                pl.write(
                                    hca_work_row_begin,
                                    [work_cursor],
                                    pl.cast(row_begin, pl.INT32),
                                )
                                pl.write(
                                    hca_work_valid_rows,
                                    [work_cursor],
                                    pl.cast(valid_rows, pl.INT32),
                                )
                            work_cursor = pl.cast(work_cursor + 1, pl.INT32)
                        pl.write(
                            hca_query_work_offsets, [query + 1], work_cursor
                        )

    return (
        hca_visible_rows,
        hca_query_work_offsets,
        hca_work_query_ids,
        hca_work_row_begin,
        hca_work_valid_rows,
        hca_event_query_ids,
        hca_event_rows,
        hca_event_write_slots,
        hca_request_event_indices,
        hca_state_write_slots,
        next_hca_valid_ranges,
        next_hca_state_valid_ranges,
    )


@dataclass(frozen=True)
class PhaseCHcaRequest:
    """One request's pre-step HCA main and compressor-state descriptors."""

    committed_tokens: int
    active: bool
    epoch: int
    hca_valid_begin: int
    hca_valid_end: int
    hca_page_ids: tuple[int, ...]
    hca_page_epochs: tuple[int, ...]
    state_page_id: int
    state_page_epoch: int
    state_valid_begin: int
    state_valid_end: int
    hca_head: int = 0
    state_page_ids: tuple[int, ...] = ()
    state_page_epochs: tuple[int, ...] | None = None

    @property
    def state_pages(self) -> tuple[int, ...]:
        if self.state_page_ids:
            return self.state_page_ids
        if self.state_page_id < 0:
            return ()
        return tuple(
            self.state_page_id + relative
            for relative in range(HCA_STATE_PAGES_PER_REQUEST)
        )

    @property
    def state_epochs(self) -> tuple[int, ...]:
        if self.state_page_epochs is not None:
            return self.state_page_epochs
        return (self.state_page_epoch,) * len(self.state_pages)


@dataclass(frozen=True)
class PhaseCHcaQuery:
    request_id: int
    position: int


@dataclass(frozen=True)
class PhaseCHcaMetadataResult:
    request_query_offsets: tuple[int, ...]
    hca_visible_rows: tuple[int, ...]
    hca_query_work_offsets: tuple[int, ...]
    hca_work_query_ids: tuple[int, ...]
    hca_work_row_begin: tuple[int, ...]
    hca_work_valid_rows: tuple[int, ...]
    hca_event_query_ids: tuple[int, ...]
    hca_event_rows: tuple[int, ...]
    hca_event_write_slots: tuple[int, ...]
    hca_request_event_indices: tuple[int, ...]
    hca_state_write_slots: tuple[int, ...]
    next_hca_valid_ranges: tuple[tuple[int, int], ...]
    next_hca_state_valid_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _PhaseCHcaFixture:
    requests: tuple[PhaseCHcaRequest, ...]
    queries: tuple[PhaseCHcaQuery, ...]
    request_query_offsets: tuple[int, ...]


def _phase_c_offsets(request_count: int, queries: tuple[PhaseCHcaQuery, ...]):
    offsets = [0]
    cursor = 0
    for request in range(request_count):
        while cursor < len(queries) and queries[cursor].request_id == request:
            cursor += 1
        offsets.append(cursor)
    if cursor != len(queries):
        raise ValueError("Phase C queries must be grouped by request")
    return tuple(offsets)


def _phase_c_main_descriptor_valid(request: PhaseCHcaRequest) -> bool:
    pages = len(request.hca_page_ids)
    if not request.active or not 0 <= request.committed_tokens <= M.max_position_embeddings:
        return False
    if request.epoch < 0 or pages == 0 or pages != len(request.hca_page_epochs):
        return False
    if not 0 <= request.hca_head < pages:
        return False
    if not 0 <= request.hca_valid_begin <= request.hca_valid_end <= MAX_HCA_ROWS:
        return False
    allocated_end = (
        request.hca_valid_begin // CACHE_BLOCK_SIZE + pages
    ) * CACHE_BLOCK_SIZE
    if request.hca_valid_end > allocated_end:
        return False
    return all(
        page_id >= 0 and page_epoch == request.epoch
        for page_id, page_epoch in zip(
            request.hca_page_ids, request.hca_page_epochs
        )
    )


def _phase_c_state_descriptor_valid(request: PhaseCHcaRequest) -> bool:
    if not request.active:
        return False
    if len(request.state_pages) != HCA_STATE_PAGES_PER_REQUEST:
        return False
    if len(request.state_epochs) != len(request.state_pages):
        return False
    if any(page < 0 for page in request.state_pages):
        return False
    if any(page_epoch != request.epoch for page_epoch in request.state_epochs):
        return False
    begin, end = request.state_valid_begin, request.state_valid_end
    if not 0 <= begin <= end <= M.max_position_embeddings:
        return False
    if end - begin >= HCA_STATE_ROWS_PER_REQUEST:
        return False
    return end == begin or begin // HCA_COMPRESS_RATIO == (end - 1) // HCA_COMPRESS_RATIO


def _phase_c_event_slot(request: PhaseCHcaRequest, event_row: int) -> int:
    if not _phase_c_main_descriptor_valid(request):
        return PHASE_A_INVALID_SLOT
    base_page = request.hca_valid_begin // CACHE_BLOCK_SIZE
    logical_page = event_row // CACHE_BLOCK_SIZE - base_page
    if not 0 <= logical_page < len(request.hca_page_ids):
        return PHASE_A_INVALID_SLOT
    page_index = (request.hca_head + logical_page) % len(request.hca_page_ids)
    return (
        request.hca_page_ids[page_index] * CACHE_BLOCK_SIZE
        + event_row % CACHE_BLOCK_SIZE
    )


def build_phase_c_hca_metadata(
    requests: tuple[PhaseCHcaRequest, ...],
    queries: tuple[PhaseCHcaQuery, ...],
    request_query_offsets: tuple[int, ...] | None = None,
) -> PhaseCHcaMetadataResult:
    """Pure-Python reference for Phase C packed HCA metadata."""
    offsets = _phase_c_offsets(len(requests), queries)
    if request_query_offsets is not None and tuple(request_query_offsets) != offsets:
        raise ValueError("request_query_offsets does not match grouped queries")
    visible_rows_out: list[int] = []
    work_offsets = [0]
    work_query_ids: list[int] = []
    work_row_begin: list[int] = []
    work_valid_rows: list[int] = []
    event_query_ids: list[int] = []
    event_rows: list[int] = []
    event_slots: list[int] = []
    request_event_indices: list[int] = [-1] * len(requests)
    state_slots: list[int] = [PHASE_A_INVALID_SLOT] * len(queries)
    next_main: list[tuple[int, int]] = []
    next_state: list[tuple[int, int]] = []

    for request_id, request in enumerate(requests):
        begin, end = offsets[request_id], offsets[request_id + 1]
        positions = [queries[index].position for index in range(begin, end)]
        if any(
            queries[index].request_id != request_id
            or not 0 <= queries[index].position < M.max_position_embeddings
            or (index > begin and queries[index].position != queries[index - 1].position + 1)
            for index in range(begin, end)
        ):
            raise ValueError("Phase C active queries must be contiguous and in range")
        main_valid = _phase_c_main_descriptor_valid(request)
        state_valid = _phase_c_state_descriptor_valid(request)
        if main_valid:
            step_end = positions[-1] + 1 if positions else request.hca_valid_end * 128
            next_main.append(
                (
                    request.hca_valid_begin,
                    max(request.hca_valid_end, step_end // HCA_COMPRESS_RATIO),
                )
            )
        else:
            next_main.append((0, 0))
        if state_valid:
            step_end = positions[-1] + 1 if positions else request.state_valid_end
            next_state.append(
                (
                    step_end // HCA_COMPRESS_RATIO * HCA_COMPRESS_RATIO,
                    step_end,
                )
            )
        else:
            next_state.append((0, 0))

        causal_events: list[int] = []
        for query_index in range(begin, end):
            position = queries[query_index].position
            if state_valid:
                state_ring_row = position % HCA_STATE_ROWS_PER_REQUEST
                state_relative_page = state_ring_row // HCA_STATE_BLOCK_SIZE
                state_slots[query_index] = (
                    request.state_pages[state_relative_page] * HCA_STATE_BLOCK_SIZE
                    + state_ring_row % HCA_STATE_BLOCK_SIZE
                )
            if (position + 1) % HCA_COMPRESS_RATIO == 0:
                row = position // HCA_COMPRESS_RATIO
                slot = _phase_c_event_slot(request, row)
                if slot != PHASE_A_INVALID_SLOT:
                    request_event_indices[request_id] = len(event_rows)
                    event_query_ids.append(query_index)
                    event_rows.append(row)
                    event_slots.append(slot)
                    causal_events.append(row)

            visible_begin = request.hca_valid_begin if main_valid else 0
            visible_end = visible_begin
            if main_valid:
                causal_rows = min(request.committed_tokens, position + 1) // HCA_COMPRESS_RATIO
                visible_end = max(
                    visible_begin, min(request.hca_valid_end, causal_rows)
                )
                for row in causal_events:
                    if row == visible_end:
                        visible_end += 1
            rows = visible_end - visible_begin
            visible_rows_out.append(rows)
            shard_count = (rows + HCA_ROWS_PER_SHARD - 1) // HCA_ROWS_PER_SHARD
            for shard in range(shard_count):
                row_begin = visible_begin + shard * HCA_ROWS_PER_SHARD
                work_query_ids.append(query_index)
                work_row_begin.append(row_begin)
                work_valid_rows.append(
                    min(HCA_ROWS_PER_SHARD, visible_end - row_begin)
                )
            work_offsets.append(len(work_query_ids))

    return PhaseCHcaMetadataResult(
        request_query_offsets=offsets,
        hca_visible_rows=tuple(visible_rows_out),
        hca_query_work_offsets=tuple(work_offsets),
        hca_work_query_ids=tuple(work_query_ids),
        hca_work_row_begin=tuple(work_row_begin),
        hca_work_valid_rows=tuple(work_valid_rows),
        hca_event_query_ids=tuple(event_query_ids),
        hca_event_rows=tuple(event_rows),
        hca_event_write_slots=tuple(event_slots),
        hca_request_event_indices=tuple(request_event_indices),
        hca_state_write_slots=tuple(state_slots),
        next_hca_valid_ranges=tuple(next_main),
        next_hca_state_valid_ranges=tuple(next_state),
    )


def _phase_c_request_for_length(length: int, request_id: int) -> PhaseCHcaRequest:
    position = length - 1
    total_rows = length // HCA_COMPRESS_RATIO
    current_event = int(length % HCA_COMPRESS_RATIO == 0)
    pre_rows = total_rows - current_event
    page_count = max(1, (total_rows + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE)
    state_begin = position // HCA_COMPRESS_RATIO * HCA_COMPRESS_RATIO
    return PhaseCHcaRequest(
        committed_tokens=length,
        active=True,
        epoch=7 + request_id,
        hca_valid_begin=0,
        hca_valid_end=pre_rows,
        hca_page_ids=tuple(1000 + request_id * 10_000 + page for page in range(page_count)),
        hca_page_epochs=tuple(7 + request_id for _ in range(page_count)),
        state_page_id=200 + request_id * HCA_STATE_PAGES_PER_REQUEST,
        state_page_epoch=7 + request_id,
        state_valid_begin=state_begin,
        state_valid_end=position,
    )


def _phase_c_fixture_from_lengths(lengths: tuple[int, ...]) -> _PhaseCHcaFixture:
    requests = tuple(
        _phase_c_request_for_length(length, request_id)
        for request_id, length in enumerate(lengths)
    )
    queries = tuple(
        PhaseCHcaQuery(request_id, length - 1)
        for request_id, length in enumerate(lengths)
    )
    return _PhaseCHcaFixture(requests, queries, _phase_c_offsets(len(requests), queries))


def _case_hca_length_matrix():
    return _phase_c_fixture_from_lengths(
        (1, 127, 128, 129, 12_288, 16_384, 16_385, 32_768, 1_048_575, 1_048_576)
    )


def _case_hca_shard_tail():
    return _phase_c_fixture_from_lengths((128, 12_288, 16_384, 32_768))


def _case_hca_boundary_event():
    request = _phase_c_request_for_length(128, 0)
    queries = (PhaseCHcaQuery(0, 126), PhaseCHcaQuery(0, 127))
    request = PhaseCHcaRequest(
        **{**request.__dict__, "state_valid_begin": 0, "state_valid_end": 126}
    )
    return _PhaseCHcaFixture((request,), queries, (0, 2))


def _case_hca_state_rollover():
    request = _phase_c_request_for_length(129, 0)
    request = PhaseCHcaRequest(
        **{
            **request.__dict__,
            "hca_valid_end": 0,
            "state_valid_begin": 0,
            "state_valid_end": 127,
        }
    )
    queries = (PhaseCHcaQuery(0, 127), PhaseCHcaQuery(0, 128))
    return _PhaseCHcaFixture((request,), queries, (0, 2))


def _case_hca_page_permutation():
    request = PhaseCHcaRequest(
        16_512, True, 4, 0, 129,
        (11, 7, 23, 5, 19), (4, 4, 4, 4, 4),
        9, 4, 128, 129, 1
    )
    queries = (PhaseCHcaQuery(0, 16_511),)
    return _PhaseCHcaFixture((request,), queries, (0, 1))


def _case_hca_cross_page_shard():
    request = PhaseCHcaRequest(
        16_640, True, 3, 1, 129,
        (17, 5, 29, 11, 23), (3, 3, 3, 3, 3),
        8, 3, 128, 129
    )
    queries = (PhaseCHcaQuery(0, 16_638),)
    return _PhaseCHcaFixture((request,), queries, (0, 1))


def _case_hca_stale_epoch_rejected():
    request = PhaseCHcaRequest(
        16_384, True, 8, 0, 128,
        (4, 8, 12, 16), (7, 7, 7, 7),
        6, 7, 16_256, 16_383
    )
    queries = (PhaseCHcaQuery(0, 16_383),)
    return _PhaseCHcaFixture((request,), queries, (0, 1))


def _case_hca_missing_page_rejected():
    """A request with no HCA page or state allocation must produce no work."""
    request = PhaseCHcaRequest(
        129,
        True,
        9,
        0,
        1,
        (),
        (),
        -1,
        -1,
        0,
        0,
    )
    queries = (PhaseCHcaQuery(0, 128),)
    return _PhaseCHcaFixture((request,), queries, (0, 1))


def _case_hca_mixed_validity():
    """Valid long/permuted request beside inactive, stale, and missing ones."""
    valid = PhaseCHcaRequest(
        16_512,
        True,
        4,
        0,
        129,
        (11, 7, 23, 5, 19),
        (4, 4, 4, 4, 4),
        9,
        4,
        128,
        129,
        1,
    )
    inactive = PhaseCHcaRequest(
        128,
        False,
        5,
        0,
        1,
        (),
        (),
        -1,
        -1,
        0,
        0,
    )
    stale = PhaseCHcaRequest(
        16_384,
        True,
        8,
        0,
        128,
        (4, 8, 12, 16),
        (7, 7, 7, 7),
        6,
        7,
        16_256,
        16_383,
    )
    missing = PhaseCHcaRequest(
        129,
        True,
        9,
        0,
        1,
        (),
        (),
        -1,
        -1,
        0,
        0,
    )
    requests = (valid, inactive, stale, missing)
    queries = (
        PhaseCHcaQuery(0, 16_511),
        PhaseCHcaQuery(1, 127),
        PhaseCHcaQuery(2, 16_383),
        PhaseCHcaQuery(3, 128),
    )
    return _PhaseCHcaFixture(requests, queries, (0, 1, 2, 3, 4))


def _case_hca_heterogeneous_lengths():
    return _phase_c_fixture_from_lengths((127, 128, 32_768, 1_048_576))


def _case_hca_one_m_tail():
    return _phase_c_fixture_from_lengths((1_048_575, 1_048_576))


_PHASE_C_CASES = {
    "hca_length_matrix": _case_hca_length_matrix,
    "hca_shard_tail": _case_hca_shard_tail,
    "hca_boundary_event": _case_hca_boundary_event,
    "hca_state_rollover": _case_hca_state_rollover,
    "hca_page_permutation": _case_hca_page_permutation,
    "hca_cross_page_shard": _case_hca_cross_page_shard,
    "hca_stale_epoch_rejected": _case_hca_stale_epoch_rejected,
    "hca_missing_page_rejected": _case_hca_missing_page_rejected,
    "hca_mixed_validity": _case_hca_mixed_validity,
    "hca_heterogeneous_lengths": _case_hca_heterogeneous_lengths,
    "hca_one_m_tail": _case_hca_one_m_tail,
}


def run_phase_c_case(case: str) -> bool:
    fixture = _PHASE_C_CASES[case]()
    result = build_phase_c_hca_metadata(
        fixture.requests, fixture.queries, fixture.request_query_offsets
    )
    if result.hca_query_work_offsets[-1] != len(result.hca_work_query_ids):
        return False
    if len(result.hca_event_rows) != len(result.hca_event_write_slots):
        return False
    if case == "hca_heterogeneous_lengths":
        shards = [
            result.hca_query_work_offsets[index + 1]
            - result.hca_query_work_offsets[index]
            for index in range(len(fixture.queries))
        ]
        if shards != [0, 1, 2, 64]:
            print(f"[Phase C case={case}] shard mismatch: {shards}")
            return False

    if case == "hca_missing_page_rejected":
        assert result.hca_visible_rows == (0,)
        assert result.hca_query_work_offsets == (0, 0)
        assert result.hca_work_query_ids == ()
        assert result.hca_event_query_ids == ()
        assert result.hca_event_rows == ()
        assert result.hca_event_write_slots == ()
        assert result.hca_request_event_indices == (-1,)
        assert result.hca_state_write_slots == (PHASE_A_INVALID_SLOT,)
        assert result.next_hca_valid_ranges == ((0, 0),)
        assert result.next_hca_state_valid_ranges == ((0, 0),)

    if case == "hca_mixed_validity":
        # Only request 0 contributes shards/events/state writes.  The invalid
        # neighbours stay zeroed and cannot contaminate the valid request.
        assert result.hca_visible_rows[0] == 129
        assert result.hca_visible_rows[1:] == (0, 0, 0)
        assert result.hca_query_work_offsets == (0, 2, 2, 2, 2)
        assert result.hca_work_query_ids == (0, 0)
        assert result.hca_work_row_begin == (0, 128)
        assert result.hca_work_valid_rows == (128, 1)
        assert result.hca_event_query_ids == (0,)
        assert result.hca_event_rows == (128,)
        assert result.hca_event_write_slots == (11 * CACHE_BLOCK_SIZE,)
        assert result.hca_request_event_indices == (0, -1, -1, -1)
        assert result.hca_state_write_slots[0] != PHASE_A_INVALID_SLOT
        assert result.hca_state_write_slots[1:] == (
            PHASE_A_INVALID_SLOT,
            PHASE_A_INVALID_SLOT,
            PHASE_A_INVALID_SLOT,
        )
        assert result.next_hca_valid_ranges[0] != (0, 0)
        assert result.next_hca_valid_ranges[1:] == ((0, 0), (0, 0), (0, 0))
        assert result.next_hca_state_valid_ranges[1:] == (
            (0, 0),
            (0, 0),
            (0, 0),
        )
    print(
        f"[Phase C host case={case}] PASS "
        f"Q={len(fixture.queries)} N_work={len(result.hca_work_query_ids)} "
        f"N_event={len(result.hca_event_rows)}"
    )
    return True


def _pack_phase_c_inputs(fixture: _PhaseCHcaFixture):
    import torch

    requests = fixture.requests
    queries = fixture.queries
    pages: list[tuple[int, int]] = []
    page_offsets = [0]
    for request in requests:
        pages.extend(zip(request.hca_page_ids, request.hca_page_epochs))
        page_offsets.append(len(pages))
    return {
        "request_state": torch.tensor(
            [[r.committed_tokens, int(r.active), r.epoch] for r in requests],
            dtype=torch.int32,
        ),
        "query_request_ids": torch.tensor(
            [q.request_id for q in queries], dtype=torch.int32
        ),
        "request_query_offsets": torch.tensor(
            fixture.request_query_offsets, dtype=torch.int32
        ),
        "position_ids": torch.tensor([q.position for q in queries], dtype=torch.int32),
        "hca_pages": torch.tensor(pages, dtype=torch.int32).reshape(-1, 2),
        "hca_page_offsets": torch.tensor(page_offsets, dtype=torch.int32),
        "hca_windows": torch.tensor(
            [[r.hca_valid_begin, r.hca_valid_end, r.hca_head] for r in requests],
            dtype=torch.int32,
        ),
        "state_pages": torch.tensor(
            [
                list(zip(r.state_pages, r.state_epochs))
                if len(r.state_pages) == HCA_STATE_PAGES_PER_REQUEST
                else [(-1, -1)] * HCA_STATE_PAGES_PER_REQUEST
                for r in requests
            ],
            dtype=torch.int32,
        ),
        "state_valid_ranges": torch.tensor(
            [[r.state_valid_begin, r.state_valid_end] for r in requests],
            dtype=torch.int32,
        ),
    }


def golden_phase_c_hca_metadata(tensors):
    requests = tuple(
        PhaseCHcaRequest(
            committed_tokens=int(tensors["request_state"][r, 0]),
            active=bool(int(tensors["request_state"][r, 1])),
            epoch=int(tensors["request_state"][r, 2]),
            hca_valid_begin=int(tensors["hca_windows"][r, 0]),
            hca_valid_end=int(tensors["hca_windows"][r, 1]),
            hca_head=int(tensors["hca_windows"][r, 2]),
            hca_page_ids=tuple(
                int(tensors["hca_pages"][p, 0])
                for p in range(
                    int(tensors["hca_page_offsets"][r]),
                    int(tensors["hca_page_offsets"][r + 1]),
                )
            ),
            hca_page_epochs=tuple(
                int(tensors["hca_pages"][p, 1])
                for p in range(
                    int(tensors["hca_page_offsets"][r]),
                    int(tensors["hca_page_offsets"][r + 1]),
                )
            ),
            state_page_id=-1,
            state_page_epoch=-1,
            state_page_ids=tuple(
                int(tensors["state_pages"][r, page, 0])
                for page in range(HCA_STATE_PAGES_PER_REQUEST)
            ),
            state_page_epochs=tuple(
                int(tensors["state_pages"][r, page, 1])
                for page in range(HCA_STATE_PAGES_PER_REQUEST)
            ),
            state_valid_begin=int(tensors["state_valid_ranges"][r, 0]),
            state_valid_end=int(tensors["state_valid_ranges"][r, 1]),
        )
        for r in range(int(tensors["request_state"].shape[0]))
    )
    queries = tuple(
        PhaseCHcaQuery(
            int(tensors["query_request_ids"][q]),
            int(tensors["position_ids"][q]),
        )
        for q in range(int(tensors["position_ids"].shape[0]))
    )
    offsets = tuple(int(v) for v in tensors["request_query_offsets"])
    result = build_phase_c_hca_metadata(requests, queries, offsets)
    names = {
        "hca_visible_rows": result.hca_visible_rows,
        "hca_query_work_offsets": result.hca_query_work_offsets,
        "hca_work_query_ids": result.hca_work_query_ids,
        "hca_work_row_begin": result.hca_work_row_begin,
        "hca_work_valid_rows": result.hca_work_valid_rows,
        "hca_event_query_ids": result.hca_event_query_ids,
        "hca_event_rows": result.hca_event_rows,
        "hca_event_write_slots": result.hca_event_write_slots,
        "hca_request_event_indices": result.hca_request_event_indices,
        "hca_state_write_slots": result.hca_state_write_slots,
        "next_hca_valid_ranges": result.next_hca_valid_ranges,
        "next_hca_state_valid_ranges": result.next_hca_state_valid_ranges,
    }
    for name, value in names.items():
        tensors[name].copy_(tensors[name].new_tensor(value))


def build_phase_c_hca_tensor_specs(case: str):
    import torch
    from golden import TensorSpec

    fixture = _PHASE_C_CASES[case]()
    result = build_phase_c_hca_metadata(
        fixture.requests, fixture.queries, fixture.request_query_offsets
    )
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in _pack_phase_c_inputs(fixture).items()
    ]
    shapes = {
        "hca_visible_rows": ([len(fixture.queries)], torch.int32),
        "hca_query_work_offsets": ([len(fixture.queries) + 1], torch.int32),
        "hca_work_query_ids": ([len(result.hca_work_query_ids)], torch.int32),
        "hca_work_row_begin": ([len(result.hca_work_query_ids)], torch.int32),
        "hca_work_valid_rows": ([len(result.hca_work_query_ids)], torch.int32),
        "hca_event_query_ids": ([len(result.hca_event_rows)], torch.int32),
        "hca_event_rows": ([len(result.hca_event_rows)], torch.int32),
        "hca_event_write_slots": ([len(result.hca_event_rows)], torch.int64),
        "hca_request_event_indices": ([len(fixture.requests)], torch.int32),
        "hca_state_write_slots": ([len(fixture.queries)], torch.int64),
        "next_hca_valid_ranges": ([len(fixture.requests), 2], torch.int32),
        "next_hca_state_valid_ranges": ([len(fixture.requests), 2], torch.int32),
    }
    specs.extend(
        TensorSpec(name, shape, dtype, is_output=True)
        for name, (shape, dtype) in shapes.items()
    )
    return specs


# Phase D lowers paired main/index pages, two eight-row state rings, exact
# ratio-4 events, packed 2K leaves, and the branch-free Top-512 forest schedule
# consumed by ``decode_indexer_topk.active_topk_forest``. The accepted Phase
# A/B/C entries above remain independent migration-fence ABIs.
PHASE_D_REQUESTS_DYN = pl.dynamic("PHASE_D_REQUESTS_DYN")
PHASE_D_REQUEST_OFFSETS_DYN = pl.dynamic("PHASE_D_REQUEST_OFFSETS_DYN")
PHASE_D_QUERIES_DYN = pl.dynamic("PHASE_D_QUERIES_DYN")
PHASE_D_QUERY_OFFSETS_DYN = pl.dynamic("PHASE_D_QUERY_OFFSETS_DYN")
PHASE_D_MAIN_PAGES_DYN = pl.dynamic("PHASE_D_MAIN_PAGES_DYN")
PHASE_D_INDEX_PAGES_DYN = pl.dynamic("PHASE_D_INDEX_PAGES_DYN")
PHASE_D_LEAVES_DYN = pl.dynamic("PHASE_D_LEAVES_DYN")
PHASE_D_MERGES_DYN = pl.dynamic("PHASE_D_MERGES_DYN")
PHASE_D_EVENTS_DYN = pl.dynamic("PHASE_D_EVENTS_DYN")
PHASE_D_PAIR_GROUPS_DYN = pl.dynamic("PHASE_D_PAIR_GROUPS_DYN")
PHASE_D_SINGLETONS_DYN = pl.dynamic("PHASE_D_SINGLETONS_DYN")
PHASE_D_UPPER_MERGES_DYN = pl.dynamic("PHASE_D_UPPER_MERGES_DYN")


@dataclass(frozen=True)
class PhaseDCsaRequest:
    """One request's pre-step paired CSA and state-ring descriptors."""

    committed_tokens: int
    active: bool
    epoch: int
    csa_valid_begin: int
    csa_valid_end: int
    main_page_ids: tuple[int, ...]
    main_page_epochs: tuple[int, ...]
    index_page_ids: tuple[int, ...]
    index_page_epochs: tuple[int, ...]
    main_state_page_id: int
    main_state_page_epoch: int
    main_state_valid_begin: int
    main_state_valid_end: int
    inner_state_page_id: int
    inner_state_page_epoch: int
    inner_state_valid_begin: int
    inner_state_valid_end: int
    main_head: int = 0
    index_head: int = 0
    index_valid_begin: int | None = None
    index_valid_end: int | None = None
    main_state_page_ids: tuple[int, ...] = ()
    main_state_page_epochs: tuple[int, ...] | None = None
    inner_state_page_ids: tuple[int, ...] = ()
    inner_state_page_epochs: tuple[int, ...] | None = None

    @property
    def main_state_pages(self) -> tuple[int, ...]:
        if self.main_state_page_ids:
            return self.main_state_page_ids
        if self.main_state_page_id < 0:
            return ()
        return tuple(
            self.main_state_page_id + relative
            for relative in range(CSA_STATE_PAGES_PER_REQUEST)
        )

    @property
    def main_state_epochs(self) -> tuple[int, ...]:
        if self.main_state_page_epochs is not None:
            return self.main_state_page_epochs
        return (self.main_state_page_epoch,) * len(self.main_state_pages)

    @property
    def inner_state_pages(self) -> tuple[int, ...]:
        if self.inner_state_page_ids:
            return self.inner_state_page_ids
        if self.inner_state_page_id < 0:
            return ()
        return tuple(
            self.inner_state_page_id + relative
            for relative in range(CSA_INNER_STATE_PAGES_PER_REQUEST)
        )

    @property
    def inner_state_epochs(self) -> tuple[int, ...]:
        if self.inner_state_page_epochs is not None:
            return self.inner_state_page_epochs
        return (self.inner_state_page_epoch,) * len(self.inner_state_pages)


@dataclass(frozen=True)
class PhaseDCsaQuery:
    request_id: int
    position: int


@dataclass(frozen=True)
class PhaseDCsaMetadataResult:
    """Exact actual-shape Phase D metadata and branch-free forest schedule."""

    request_query_offsets: tuple[int, ...]
    csa_visible_candidates: tuple[int, ...]
    csa_query_leaf_offsets: tuple[int, ...]
    csa_leaf_query_ids: tuple[int, ...]
    csa_leaf_candidate_begin: tuple[int, ...]
    csa_leaf_valid_candidates: tuple[int, ...]
    csa_leaf_output_slots: tuple[int, ...]
    csa_leaf_credit_predecessors: tuple[int, ...]
    csa_query_node_offsets: tuple[int, ...]
    csa_query_merge_offsets: tuple[int, ...]
    csa_query_pair_group_offsets: tuple[int, ...]
    csa_merge_query_ids: tuple[int, ...]
    csa_merge_levels: tuple[int, ...]
    csa_merge_left_slots: tuple[int, ...]
    csa_merge_right_slots: tuple[int, ...]
    csa_merge_output_slots: tuple[int, ...]
    csa_pair_left_leaf_ids: tuple[int, ...]
    csa_pair_right_leaf_ids: tuple[int, ...]
    csa_pair_left_slots: tuple[int, ...]
    csa_pair_right_slots: tuple[int, ...]
    csa_pair_output_slots: tuple[int, ...]
    csa_pair_credit_slots: tuple[int, ...]
    csa_singleton_leaf_ids: tuple[int, ...]
    csa_singleton_slots: tuple[int, ...]
    csa_singleton_credit_slots: tuple[int, ...]
    csa_upper_left_slots: tuple[int, ...]
    csa_upper_right_slots: tuple[int, ...]
    csa_upper_output_slots: tuple[int, ...]
    csa_root_slots: tuple[int, ...]
    csa_root_dependency_slots: tuple[int, ...]
    csa_credit_predecessors: tuple[int, ...]
    csa_event_query_ids: tuple[int, ...]
    csa_event_rows: tuple[int, ...]
    csa_main_event_write_slots: tuple[int, ...]
    csa_idx_event_write_slots: tuple[int, ...]
    csa_request_event_indices: tuple[int, ...]
    csa_state_write_slots: tuple[int, ...]
    csa_inner_state_write_slots: tuple[int, ...]
    next_csa_valid_ranges: tuple[tuple[int, int], ...]
    next_csa_state_valid_ranges: tuple[tuple[int, int], ...]
    next_csa_inner_state_valid_ranges: tuple[tuple[int, int], ...]

    @property
    def n_leaves(self) -> int:
        return len(self.csa_leaf_query_ids)

    @property
    def n_merges(self) -> int:
        return len(self.csa_merge_output_slots)

    @property
    def n_nodes(self) -> int:
        return self.n_leaves + self.n_merges


@dataclass(frozen=True)
class _PhaseDCsaFixture:
    requests: tuple[PhaseDCsaRequest, ...]
    queries: tuple[PhaseDCsaQuery, ...]
    request_query_offsets: tuple[int, ...]


def _phase_d_offsets(
    request_count: int,
    queries: tuple[PhaseDCsaQuery, ...],
) -> tuple[int, ...]:
    offsets = [0]
    cursor = 0
    for request in range(request_count):
        while cursor < len(queries) and queries[cursor].request_id == request:
            cursor += 1
        offsets.append(cursor)
    if cursor != len(queries):
        raise ValueError("Phase D queries must be grouped by ascending request id")
    return tuple(offsets)


def _phase_d_page_span_valid(
    page_ids: tuple[int, ...],
    page_epochs: tuple[int, ...],
    *,
    valid_begin: int,
    valid_end: int,
    head: int,
    epoch: int,
) -> bool:
    if len(page_ids) != len(page_epochs) or len(page_ids) == 0:
        return False
    if len(page_ids) > MAX_CSA_CANDIDATES // CACHE_BLOCK_SIZE:
        return False
    if not 0 <= head < len(page_ids):
        return False
    if not 0 <= valid_begin <= valid_end <= MAX_CSA_CANDIDATES:
        return False
    span_end = (valid_begin // CACHE_BLOCK_SIZE + len(page_ids)) * CACHE_BLOCK_SIZE
    if valid_end > span_end:
        return False
    return all(
        page_id >= 0 and page_epoch == epoch
        for page_id, page_epoch in zip(page_ids, page_epochs)
    )


def _phase_d_state_valid(
    page_ids: tuple[int, ...],
    page_epochs: tuple[int, ...],
    valid_begin: int,
    valid_end: int,
    *,
    request: PhaseDCsaRequest,
) -> bool:
    return (
        request.active
        and len(page_ids) == CSA_STATE_PAGES_PER_REQUEST
        and len(page_epochs) == len(page_ids)
        and all(page_id >= 0 for page_id in page_ids)
        and all(page_epoch == request.epoch for page_epoch in page_epochs)
        and 0 <= valid_begin <= valid_end <= M.max_position_embeddings
        and valid_end - valid_begin <= CSA_STATE_ROWS_PER_REQUEST
    )


def _phase_d_request_valid(request: PhaseDCsaRequest) -> bool:
    index_begin = (
        request.csa_valid_begin
        if request.index_valid_begin is None
        else request.index_valid_begin
    )
    index_end = (
        request.csa_valid_end
        if request.index_valid_end is None
        else request.index_valid_end
    )
    if (
        not request.active
        or not 0 <= request.committed_tokens <= M.max_position_embeddings
        or request.epoch < 0
        or index_begin != request.csa_valid_begin
        or index_end != request.csa_valid_end
    ):
        return False
    main_valid = _phase_d_page_span_valid(
        request.main_page_ids,
        request.main_page_epochs,
        valid_begin=request.csa_valid_begin,
        valid_end=request.csa_valid_end,
        head=request.main_head,
        epoch=request.epoch,
    )
    index_valid = _phase_d_page_span_valid(
        request.index_page_ids,
        request.index_page_epochs,
        valid_begin=index_begin,
        valid_end=index_end,
        head=request.index_head,
        epoch=request.epoch,
    )
    main_state_valid = _phase_d_state_valid(
        request.main_state_pages,
        request.main_state_epochs,
        request.main_state_valid_begin,
        request.main_state_valid_end,
        request=request,
    )
    inner_state_valid = _phase_d_state_valid(
        request.inner_state_pages,
        request.inner_state_epochs,
        request.inner_state_valid_begin,
        request.inner_state_valid_end,
        request=request,
    )
    return main_valid and index_valid and main_state_valid and inner_state_valid


def _phase_d_write_slot(
    page_ids: tuple[int, ...],
    *,
    valid_begin: int,
    head: int,
    logical_candidate: int,
) -> int:
    relative_page = (
        logical_candidate // CACHE_BLOCK_SIZE - valid_begin // CACHE_BLOCK_SIZE
    )
    if not 0 <= relative_page < len(page_ids):
        return PHASE_A_INVALID_SLOT
    page_index = (head + relative_page) % len(page_ids)
    return (
        page_ids[page_index] * CACHE_BLOCK_SIZE
        + logical_candidate % CACHE_BLOCK_SIZE
    )


def _phase_d_next_state_range(position: int | None, *, active: bool) -> tuple[int, int]:
    if not active or position is None:
        return 0, 0
    step_end = position + 1
    block_begin = position // CSA_COMPRESS_RATIO * CSA_COMPRESS_RATIO
    if step_end % CSA_COMPRESS_RATIO == 0:
        return block_begin, step_end
    return max(0, block_begin - CSA_COMPRESS_RATIO), step_end


def build_phase_d_csa_metadata(
    requests: tuple[PhaseDCsaRequest, ...],
    queries: tuple[PhaseDCsaQuery, ...],
    request_query_offsets: tuple[int, ...] | None = None,
) -> PhaseDCsaMetadataResult:
    """Build exact Phase D metadata without max-query or max-context padding."""
    if len(queries) > CSA_MAX_QUERIES:
        raise ValueError("Phase D query count exceeds the local Top-K task bound")
    offsets = _phase_d_offsets(len(requests), queries)
    if request_query_offsets is not None and tuple(request_query_offsets) != offsets:
        raise ValueError("request_query_offsets does not match grouped queries")

    visible_candidates: list[int] = []
    leaf_offsets = [0]
    leaf_query_ids: list[int] = []
    leaf_begins: list[int] = []
    leaf_valid: list[int] = []
    leaf_slots: list[int] = []
    leaf_credits: list[int] = []
    node_offsets = [0]
    merge_offsets = [0]
    pair_offsets = [0]
    merge_query_ids: list[int] = []
    merge_levels: list[int] = []
    merge_left: list[int] = []
    merge_right: list[int] = []
    merge_output: list[int] = []
    pair_left_leaf_ids: list[int] = []
    pair_right_leaf_ids: list[int] = []
    pair_left_slots: list[int] = []
    pair_right_slots: list[int] = []
    pair_output_slots: list[int] = []
    pair_credit_slots: list[int] = []
    singleton_leaf_ids: list[int] = []
    singleton_slots: list[int] = []
    singleton_credit_slots: list[int] = []
    upper_left_slots: list[int] = []
    upper_right_slots: list[int] = []
    upper_output_slots: list[int] = []
    root_slots: list[int] = []
    root_dependency_slots: list[int] = []
    credit_predecessors: list[int] = []
    event_query_ids: list[int] = []
    event_rows: list[int] = []
    main_event_slots: list[int] = []
    index_event_slots: list[int] = []
    request_event_indices = [-1] * len(requests)
    state_slots = [PHASE_A_INVALID_SLOT] * len(queries)
    inner_state_slots = [PHASE_A_INVALID_SLOT] * len(queries)
    next_ranges: list[tuple[int, int]] = []
    next_state_ranges: list[tuple[int, int]] = []
    next_inner_ranges: list[tuple[int, int]] = []
    node_cursor = 0

    for request_id, request in enumerate(requests):
        query_begin, query_end = offsets[request_id], offsets[request_id + 1]
        positions = [queries[q].position for q in range(query_begin, query_end)]
        if any(
            queries[q].request_id != request_id
            or not 0 <= queries[q].position < M.max_position_embeddings
            or (q > query_begin and queries[q].position != queries[q - 1].position + 1)
            for q in range(query_begin, query_end)
        ):
            raise ValueError("Phase D active queries must be contiguous and in range")

        transaction_valid = _phase_d_request_valid(request)
        causal_event_rows: list[int] = []
        request_event_begin = len(event_rows)
        if transaction_valid:
            next_end = request.csa_valid_end
            for query_index in range(query_begin, query_end):
                position = queries[query_index].position
                main_state_ring_row = position % CSA_STATE_ROWS_PER_REQUEST
                main_state_relative_page = (
                    main_state_ring_row // CSA_STATE_BLOCK_SIZE
                )
                state_slots[query_index] = (
                    request.main_state_pages[main_state_relative_page]
                    * CSA_STATE_BLOCK_SIZE
                    + main_state_ring_row % CSA_STATE_BLOCK_SIZE
                )
                inner_state_ring_row = (
                    position % CSA_INNER_STATE_ROWS_PER_REQUEST
                )
                inner_state_relative_page = (
                    inner_state_ring_row // CSA_INNER_STATE_BLOCK_SIZE
                )
                inner_state_slots[query_index] = (
                    request.inner_state_pages[inner_state_relative_page]
                    * CSA_INNER_STATE_BLOCK_SIZE
                    + inner_state_ring_row % CSA_INNER_STATE_BLOCK_SIZE
                )
                if (position + 1) % CSA_COMPRESS_RATIO == 0:
                    row = position // CSA_COMPRESS_RATIO
                    main_slot = _phase_d_write_slot(
                        request.main_page_ids,
                        valid_begin=request.csa_valid_begin,
                        head=request.main_head,
                        logical_candidate=row,
                    )
                    index_slot = _phase_d_write_slot(
                        request.index_page_ids,
                        valid_begin=request.csa_valid_begin,
                        head=request.index_head,
                        logical_candidate=row,
                    )
                    if main_slot < 0 or index_slot < 0:
                        transaction_valid = False
                        break
                    request_event_indices[request_id] = len(event_rows)
                    event_query_ids.append(query_index)
                    event_rows.append(row)
                    main_event_slots.append(main_slot)
                    index_event_slots.append(index_slot)
                    causal_event_rows.append(row)
                    next_end = max(next_end, row + 1)
            if transaction_valid:
                next_ranges.append((request.csa_valid_begin, next_end))
                last_position = positions[-1] if positions else None
                next_state_ranges.append(
                    _phase_d_next_state_range(last_position, active=True)
                )
                next_inner_ranges.append(
                    _phase_d_next_state_range(last_position, active=True)
                )
            else:
                del event_query_ids[request_event_begin:]
                del event_rows[request_event_begin:]
                del main_event_slots[request_event_begin:]
                del index_event_slots[request_event_begin:]
                request_event_indices[request_id] = -1
                for query_index in range(query_begin, query_end):
                    state_slots[query_index] = PHASE_A_INVALID_SLOT
                    inner_state_slots[query_index] = PHASE_A_INVALID_SLOT
                next_ranges.append((0, 0))
                next_state_ranges.append((0, 0))
                next_inner_ranges.append((0, 0))
        else:
            next_ranges.append((0, 0))
            next_state_ranges.append((0, 0))
            next_inner_ranges.append((0, 0))

        request_events = (
            tuple(event_rows[request_event_begin:]) if transaction_valid else ()
        )
        for query_index in range(query_begin, query_end):
            query = queries[query_index]
            visible_begin = request.csa_valid_begin if transaction_valid else 0
            visible_end = visible_begin
            if transaction_valid:
                causal_end = min(request.committed_tokens, query.position + 1) // CSA_COMPRESS_RATIO
                visible_end = max(
                    visible_begin,
                    min(request.csa_valid_end, causal_end),
                )
                for event_row in request_events:
                    event_position = event_row * CSA_COMPRESS_RATIO + CSA_COMPRESS_RATIO - 1
                    if event_position <= query.position and event_row == visible_end:
                        visible_end += 1
            candidates = visible_end - visible_begin
            visible_candidates.append(candidates)
            leaf_count = (
                candidates + CSA_CANDIDATES_PER_LEAF - 1
            ) // CSA_CANDIDATES_PER_LEAF
            if leaf_count > CSA_MAX_LEAVES_PER_QUERY:
                raise ValueError("Phase D query exceeds the 1M leaf ceiling")

            query_leaf_base = len(leaf_query_ids)
            query_pair_outputs: list[int] = []
            for group in range(leaf_count // CSA_MERGE_ARITY):
                credit = CSA_TOPK_INVALID_TASK_SLOT
                if group >= CSA_TOPK_READY_FRONTIER_W:
                    credit = query_pair_outputs[group - CSA_TOPK_READY_FRONTIER_W]
                left_leaf = len(leaf_query_ids)
                left_local = 2 * group
                left_begin = visible_begin + left_local * CSA_CANDIDATES_PER_LEAF
                left_valid = min(CSA_CANDIDATES_PER_LEAF, visible_end - left_begin)
                left_slot = node_cursor
                node_cursor += 1
                leaf_query_ids.append(query_index)
                leaf_begins.append(left_begin)
                leaf_valid.append(left_valid)
                leaf_slots.append(left_slot)
                leaf_credits.append(credit)

                right_leaf = len(leaf_query_ids)
                right_local = left_local + 1
                right_begin = visible_begin + right_local * CSA_CANDIDATES_PER_LEAF
                right_valid = min(CSA_CANDIDATES_PER_LEAF, visible_end - right_begin)
                right_slot = node_cursor
                node_cursor += 1
                leaf_query_ids.append(query_index)
                leaf_begins.append(right_begin)
                leaf_valid.append(right_valid)
                leaf_slots.append(right_slot)
                leaf_credits.append(credit)

                output_slot = node_cursor
                node_cursor += 1
                query_pair_outputs.append(output_slot)
                pair_left_leaf_ids.append(left_leaf)
                pair_right_leaf_ids.append(right_leaf)
                pair_left_slots.append(left_slot)
                pair_right_slots.append(right_slot)
                pair_output_slots.append(output_slot)
                pair_credit_slots.append(credit)
                credit_predecessors.append(credit)
                merge_query_ids.append(query_index)
                merge_levels.append(0)
                merge_left.append(left_slot)
                merge_right.append(right_slot)
                merge_output.append(output_slot)

            frontier = list(query_pair_outputs)
            if leaf_count % CSA_MERGE_ARITY:
                local_leaf = leaf_count - 1
                credit = CSA_TOPK_INVALID_TASK_SLOT
                if len(query_pair_outputs) >= CSA_TOPK_READY_FRONTIER_W:
                    credit = query_pair_outputs[-CSA_TOPK_READY_FRONTIER_W]
                leaf_id = len(leaf_query_ids)
                begin = visible_begin + local_leaf * CSA_CANDIDATES_PER_LEAF
                valid = min(CSA_CANDIDATES_PER_LEAF, visible_end - begin)
                slot = node_cursor
                node_cursor += 1
                leaf_query_ids.append(query_index)
                leaf_begins.append(begin)
                leaf_valid.append(valid)
                leaf_slots.append(slot)
                leaf_credits.append(credit)
                singleton_leaf_ids.append(leaf_id)
                singleton_slots.append(slot)
                singleton_credit_slots.append(credit)
                frontier.append(slot)

            level = 1
            while len(frontier) > 1:
                next_frontier: list[int] = []
                for pair in range(len(frontier) // CSA_MERGE_ARITY):
                    left_slot = frontier[2 * pair]
                    right_slot = frontier[2 * pair + 1]
                    output_slot = node_cursor
                    node_cursor += 1
                    upper_left_slots.append(left_slot)
                    upper_right_slots.append(right_slot)
                    upper_output_slots.append(output_slot)
                    merge_query_ids.append(query_index)
                    merge_levels.append(level)
                    merge_left.append(left_slot)
                    merge_right.append(right_slot)
                    merge_output.append(output_slot)
                    next_frontier.append(output_slot)
                if len(frontier) % CSA_MERGE_ARITY:
                    next_frontier.append(frontier[-1])
                frontier = next_frontier
                level += 1

            root = frontier[0] if frontier else -1
            root_slots.append(root)
            root_dependency_slots.append(
                root if root >= 0 else CSA_TOPK_INVALID_TASK_SLOT
            )
            leaf_offsets.append(len(leaf_query_ids))
            node_offsets.append(node_cursor)
            merge_offsets.append(len(merge_output))
            pair_offsets.append(len(pair_output_slots))
            if len(leaf_query_ids) - query_leaf_base != leaf_count:
                raise AssertionError("Phase D leaf schedule lost an active leaf")

    if node_cursor > CSA_MAX_TOPK_TASKS:
        raise ValueError("Phase D active forest exceeds the local task capacity")
    return PhaseDCsaMetadataResult(
        request_query_offsets=offsets,
        csa_visible_candidates=tuple(visible_candidates),
        csa_query_leaf_offsets=tuple(leaf_offsets),
        csa_leaf_query_ids=tuple(leaf_query_ids),
        csa_leaf_candidate_begin=tuple(leaf_begins),
        csa_leaf_valid_candidates=tuple(leaf_valid),
        csa_leaf_output_slots=tuple(leaf_slots),
        csa_leaf_credit_predecessors=tuple(leaf_credits),
        csa_query_node_offsets=tuple(node_offsets),
        csa_query_merge_offsets=tuple(merge_offsets),
        csa_query_pair_group_offsets=tuple(pair_offsets),
        csa_merge_query_ids=tuple(merge_query_ids),
        csa_merge_levels=tuple(merge_levels),
        csa_merge_left_slots=tuple(merge_left),
        csa_merge_right_slots=tuple(merge_right),
        csa_merge_output_slots=tuple(merge_output),
        csa_pair_left_leaf_ids=tuple(pair_left_leaf_ids),
        csa_pair_right_leaf_ids=tuple(pair_right_leaf_ids),
        csa_pair_left_slots=tuple(pair_left_slots),
        csa_pair_right_slots=tuple(pair_right_slots),
        csa_pair_output_slots=tuple(pair_output_slots),
        csa_pair_credit_slots=tuple(pair_credit_slots),
        csa_singleton_leaf_ids=tuple(singleton_leaf_ids),
        csa_singleton_slots=tuple(singleton_slots),
        csa_singleton_credit_slots=tuple(singleton_credit_slots),
        csa_upper_left_slots=tuple(upper_left_slots),
        csa_upper_right_slots=tuple(upper_right_slots),
        csa_upper_output_slots=tuple(upper_output_slots),
        csa_root_slots=tuple(root_slots),
        csa_root_dependency_slots=tuple(root_dependency_slots),
        csa_credit_predecessors=tuple(credit_predecessors),
        csa_event_query_ids=tuple(event_query_ids),
        csa_event_rows=tuple(event_rows),
        csa_main_event_write_slots=tuple(main_event_slots),
        csa_idx_event_write_slots=tuple(index_event_slots),
        csa_request_event_indices=tuple(request_event_indices),
        csa_state_write_slots=tuple(state_slots),
        csa_inner_state_write_slots=tuple(inner_state_slots),
        next_csa_valid_ranges=tuple(next_ranges),
        next_csa_state_valid_ranges=tuple(next_state_ranges),
        next_csa_inner_state_valid_ranges=tuple(next_inner_ranges),
    )


# Packed device ABI field indices. Packing keeps the entry below PyPTO's
# 32-tensor callable limit while preserving exact actual row counts.
PHASE_D_QUERY_OFFSET_LEAF = 0
PHASE_D_QUERY_OFFSET_NODE = 1
PHASE_D_QUERY_OFFSET_MERGE = 2
PHASE_D_QUERY_OFFSET_PAIR_GROUP = 3
PHASE_D_QUERY_OFFSET_FIELDS = 4
PHASE_D_LEAF_QUERY = 0
PHASE_D_LEAF_BEGIN = 1
PHASE_D_LEAF_VALID = 2
PHASE_D_LEAF_OUTPUT_SLOT = 3
PHASE_D_LEAF_CREDIT_SLOT = 4
PHASE_D_LEAF_FIELDS = 5
PHASE_D_MERGE_QUERY = 0
PHASE_D_MERGE_LEVEL = 1
PHASE_D_MERGE_LEFT_SLOT = 2
PHASE_D_MERGE_RIGHT_SLOT = 3
PHASE_D_MERGE_OUTPUT_SLOT = 4
PHASE_D_MERGE_FIELDS = 5
PHASE_D_PAIR_LEFT_LEAF = 0
PHASE_D_PAIR_RIGHT_LEAF = 1
PHASE_D_PAIR_LEFT_SLOT = 2
PHASE_D_PAIR_RIGHT_SLOT = 3
PHASE_D_PAIR_OUTPUT_SLOT = 4
PHASE_D_PAIR_CREDIT_SLOT = 5
PHASE_D_PAIR_FIELDS = 6
PHASE_D_SINGLETON_LEAF = 0
PHASE_D_SINGLETON_SLOT = 1
PHASE_D_SINGLETON_CREDIT_SLOT = 2
PHASE_D_SINGLETON_FIELDS = 3
PHASE_D_UPPER_LEFT_SLOT = 0
PHASE_D_UPPER_RIGHT_SLOT = 1
PHASE_D_UPPER_OUTPUT_SLOT = 2
PHASE_D_UPPER_FIELDS = 3
PHASE_D_ROOT_SLOT = 0
PHASE_D_ROOT_DEPENDENCY_SLOT = 1
PHASE_D_ROOT_FIELDS = 2
PHASE_D_EVENT_QUERY = 0
PHASE_D_EVENT_ROW = 1
PHASE_D_EVENT_MAIN_SLOT = 2
PHASE_D_EVENT_IDX_SLOT = 3
PHASE_D_EVENT_FIELDS = 4
PHASE_D_STATE_MAIN_SLOT = 0
PHASE_D_STATE_INNER_SLOT = 1
PHASE_D_STATE_SLOT_FIELDS = 2
PHASE_D_NEXT_MAIN_BEGIN = 0
PHASE_D_NEXT_MAIN_END = 1
PHASE_D_NEXT_STATE_BEGIN = 2
PHASE_D_NEXT_STATE_END = 3
PHASE_D_NEXT_INNER_BEGIN = 4
PHASE_D_NEXT_INNER_END = 5
PHASE_D_NEXT_FIELDS = 6
PHASE_D_STATE_PAGE_ID_BASE = 0
PHASE_D_STATE_PAGE_EPOCH_BASE = PHASE_D_STATE_PAGE_ID_BASE + CSA_STATE_PAGES_PER_REQUEST
PHASE_D_STATE_VALID_BEGIN = PHASE_D_STATE_PAGE_EPOCH_BASE + CSA_STATE_PAGES_PER_REQUEST
PHASE_D_STATE_VALID_END = PHASE_D_STATE_VALID_BEGIN + 1
PHASE_D_STATE_DESCRIPTOR_FIELDS = PHASE_D_STATE_VALID_END + 1


@pl.jit
def phase_d_csa_metadata(
    request_state: pl.Tensor[[PHASE_D_REQUESTS_DYN, 3], pl.INT32],
    query_request_ids: pl.Tensor[[PHASE_D_QUERIES_DYN], pl.INT32],
    request_query_offsets: pl.Tensor[[PHASE_D_REQUEST_OFFSETS_DYN], pl.INT32],
    position_ids: pl.Tensor[[PHASE_D_QUERIES_DYN], pl.INT32],
    main_pages: pl.Tensor[[PHASE_D_MAIN_PAGES_DYN, 2], pl.INT32],
    main_page_offsets: pl.Tensor[[PHASE_D_REQUEST_OFFSETS_DYN], pl.INT32],
    main_windows: pl.Tensor[[PHASE_D_REQUESTS_DYN, 3], pl.INT32],
    idx_pages: pl.Tensor[[PHASE_D_INDEX_PAGES_DYN, 2], pl.INT32],
    idx_page_offsets: pl.Tensor[[PHASE_D_REQUEST_OFFSETS_DYN], pl.INT32],
    idx_windows: pl.Tensor[[PHASE_D_REQUESTS_DYN, 3], pl.INT32],
    state_descriptors: pl.Tensor[
        [PHASE_D_REQUESTS_DYN, PHASE_D_STATE_DESCRIPTOR_FIELDS], pl.INT32
    ],
    inner_state_descriptors: pl.Tensor[
        [PHASE_D_REQUESTS_DYN, PHASE_D_STATE_DESCRIPTOR_FIELDS], pl.INT32
    ],
    csa_visible_candidates: pl.Out[
        pl.Tensor[[PHASE_D_QUERIES_DYN], pl.INT32]
    ],
    csa_query_offsets: pl.Out[
        pl.Tensor[[PHASE_D_QUERY_OFFSETS_DYN, PHASE_D_QUERY_OFFSET_FIELDS], pl.INT32]
    ],
    csa_leaf_descriptors: pl.Out[
        pl.Tensor[[PHASE_D_LEAVES_DYN, PHASE_D_LEAF_FIELDS], pl.INT32]
    ],
    csa_merge_descriptors: pl.Out[
        pl.Tensor[[PHASE_D_MERGES_DYN, PHASE_D_MERGE_FIELDS], pl.INT32]
    ],
    csa_pair_descriptors: pl.Out[
        pl.Tensor[[PHASE_D_PAIR_GROUPS_DYN, PHASE_D_PAIR_FIELDS], pl.INT32]
    ],
    csa_singleton_descriptors: pl.Out[
        pl.Tensor[[PHASE_D_SINGLETONS_DYN, PHASE_D_SINGLETON_FIELDS], pl.INT32]
    ],
    csa_upper_merge_descriptors: pl.Out[
        pl.Tensor[[PHASE_D_UPPER_MERGES_DYN, PHASE_D_UPPER_FIELDS], pl.INT32]
    ],
    csa_root_descriptors: pl.Out[
        pl.Tensor[[PHASE_D_QUERIES_DYN, PHASE_D_ROOT_FIELDS], pl.INT32]
    ],
    csa_event_descriptors: pl.Out[
        pl.Tensor[[PHASE_D_EVENTS_DYN, PHASE_D_EVENT_FIELDS], pl.INT64]
    ],
    csa_request_event_indices: pl.Out[
        pl.Tensor[[PHASE_D_REQUESTS_DYN], pl.INT32]
    ],
    csa_state_write_slots: pl.Out[
        pl.Tensor[[PHASE_D_QUERIES_DYN, PHASE_D_STATE_SLOT_FIELDS], pl.INT64]
    ],
    next_csa_ranges: pl.Out[
        pl.Tensor[[PHASE_D_REQUESTS_DYN, PHASE_D_NEXT_FIELDS], pl.INT32]
    ],
):
    """Lower paired CSA metadata using an exact packed, 24-tensor ABI.

    Descriptor columns are named by the ``PHASE_D_*`` field constants above.
    In particular, leaf rows contain query id, absolute logical candidate
    begin, valid count, output slot, and ready-frontier credit slot. Physical
    index-cache pages remain in the ragged input descriptor and are resolved
    by the score leaf instead of being duplicated for every leaf.
    """
    request_state.bind_dynamic(0, PHASE_D_REQUESTS_DYN)
    query_request_ids.bind_dynamic(0, PHASE_D_QUERIES_DYN)
    request_query_offsets.bind_dynamic(0, PHASE_D_REQUEST_OFFSETS_DYN)
    position_ids.bind_dynamic(0, PHASE_D_QUERIES_DYN)
    main_pages.bind_dynamic(0, PHASE_D_MAIN_PAGES_DYN)
    main_page_offsets.bind_dynamic(0, PHASE_D_REQUEST_OFFSETS_DYN)
    main_windows.bind_dynamic(0, PHASE_D_REQUESTS_DYN)
    idx_pages.bind_dynamic(0, PHASE_D_INDEX_PAGES_DYN)
    idx_page_offsets.bind_dynamic(0, PHASE_D_REQUEST_OFFSETS_DYN)
    idx_windows.bind_dynamic(0, PHASE_D_REQUESTS_DYN)
    state_descriptors.bind_dynamic(0, PHASE_D_REQUESTS_DYN)
    inner_state_descriptors.bind_dynamic(0, PHASE_D_REQUESTS_DYN)
    csa_visible_candidates.bind_dynamic(0, PHASE_D_QUERIES_DYN)
    csa_query_offsets.bind_dynamic(0, PHASE_D_QUERY_OFFSETS_DYN)
    csa_leaf_descriptors.bind_dynamic(0, PHASE_D_LEAVES_DYN)
    csa_merge_descriptors.bind_dynamic(0, PHASE_D_MERGES_DYN)
    csa_pair_descriptors.bind_dynamic(0, PHASE_D_PAIR_GROUPS_DYN)
    csa_singleton_descriptors.bind_dynamic(0, PHASE_D_SINGLETONS_DYN)
    csa_upper_merge_descriptors.bind_dynamic(0, PHASE_D_UPPER_MERGES_DYN)
    csa_root_descriptors.bind_dynamic(0, PHASE_D_QUERIES_DYN)
    csa_event_descriptors.bind_dynamic(0, PHASE_D_EVENTS_DYN)
    csa_request_event_indices.bind_dynamic(0, PHASE_D_REQUESTS_DYN)
    csa_state_write_slots.bind_dynamic(0, PHASE_D_QUERIES_DYN)
    next_csa_ranges.bind_dynamic(0, PHASE_D_REQUESTS_DYN)

    request_count = pl.tensor.dim(request_state, 0)
    query_count = pl.tensor.dim(position_ids, 0)
    main_page_count = pl.tensor.dim(main_pages, 0)
    idx_page_count = pl.tensor.dim(idx_pages, 0)
    for core in pl.spmd(1, name_hint="decode_phase_d_csa_metadata"):
        for query in pl.range(core, query_count):
            pl.write(csa_visible_candidates, [query], pl.cast(0, pl.INT32))
            pl.write(csa_root_descriptors, [query, PHASE_D_ROOT_SLOT], pl.cast(-1, pl.INT32))
            pl.write(
                csa_root_descriptors,
                [query, PHASE_D_ROOT_DEPENDENCY_SLOT],
                pl.cast(CSA_TOPK_INVALID_TASK_SLOT, pl.INT32),
            )
            for field in pl.range(PHASE_D_STATE_SLOT_FIELDS):
                pl.write(
                    csa_state_write_slots,
                    [query, field],
                    pl.cast(PHASE_A_INVALID_SLOT, pl.INT64),
                )
        for offset in pl.range(core, pl.tensor.dim(csa_query_offsets, 0)):
            for field in pl.range(PHASE_D_QUERY_OFFSET_FIELDS):
                pl.write(csa_query_offsets, [offset, field], pl.cast(0, pl.INT32))
        for request in pl.range(core, request_count):
            pl.write(csa_request_event_indices, [request], pl.cast(-1, pl.INT32))
            for field in pl.range(PHASE_D_NEXT_FIELDS):
                pl.write(next_csa_ranges, [request, field], pl.cast(0, pl.INT32))

        leaf_cursor = pl.cast(0, pl.INT32)
        node_cursor = pl.cast(0, pl.INT32)
        merge_cursor = pl.cast(0, pl.INT32)
        pair_cursor = pl.cast(0, pl.INT32)
        singleton_cursor = pl.cast(0, pl.INT32)
        upper_cursor = pl.cast(0, pl.INT32)
        event_cursor = pl.cast(0, pl.INT32)
        frontier = pl.array.create(CSA_MAX_LEAVES_PER_QUERY, pl.INT32)
        offsets_valid = pl.cast(0, pl.INT32)
        if pl.tensor.dim(request_query_offsets, 0) == request_count + 1:
            if pl.tensor.dim(csa_query_offsets, 0) == query_count + 1:
                offsets_valid = pl.cast(1, pl.INT32)

        if offsets_valid != 0:
            for request in pl.range(core, request_count):
                request_index = pl.cast(request, pl.INDEX)
                committed = pl.read(request_state, [request_index, 0])
                active = pl.read(request_state, [request_index, 1])
                epoch = pl.read(request_state, [request_index, 2])
                query_begin = pl.read(request_query_offsets, [request])
                query_end = pl.read(request_query_offsets, [request + 1])
                valid = pl.cast(1, pl.INT32)
                if active == 0 or committed < 0 or committed > M.max_position_embeddings:
                    valid = pl.cast(0, pl.INT32)
                if epoch < 0 or query_begin < 0 or query_end < query_begin:
                    valid = pl.cast(0, pl.INT32)
                if query_end > query_count:
                    valid = pl.cast(0, pl.INT32)

                main_begin = pl.read(main_windows, [request_index, 0])
                main_end = pl.read(main_windows, [request_index, 1])
                main_head = pl.read(main_windows, [request_index, 2])
                idx_begin = pl.read(idx_windows, [request_index, 0])
                idx_end = pl.read(idx_windows, [request_index, 1])
                idx_head = pl.read(idx_windows, [request_index, 2])
                main_page_begin = pl.read(main_page_offsets, [request])
                main_page_end = pl.read(main_page_offsets, [request + 1])
                idx_page_begin = pl.read(idx_page_offsets, [request])
                idx_page_end = pl.read(idx_page_offsets, [request + 1])
                main_page_total = main_page_end - main_page_begin
                idx_page_total = idx_page_end - idx_page_begin
                if main_begin != idx_begin or main_end != idx_end:
                    valid = pl.cast(0, pl.INT32)
                if main_begin < 0 or main_end < main_begin or main_end > MAX_CSA_CANDIDATES:
                    valid = pl.cast(0, pl.INT32)
                if main_page_total <= 0 or idx_page_total <= 0:
                    valid = pl.cast(0, pl.INT32)
                if main_page_begin < 0 or main_page_end > main_page_count:
                    valid = pl.cast(0, pl.INT32)
                if idx_page_begin < 0 or idx_page_end > idx_page_count:
                    valid = pl.cast(0, pl.INT32)
                if main_head < 0 or main_head >= main_page_total:
                    valid = pl.cast(0, pl.INT32)
                if idx_head < 0 or idx_head >= idx_page_total:
                    valid = pl.cast(0, pl.INT32)
                main_span_end = (
                    main_begin // CACHE_BLOCK_SIZE + main_page_total
                ) * CACHE_BLOCK_SIZE
                idx_span_end = (
                    idx_begin // CACHE_BLOCK_SIZE + idx_page_total
                ) * CACHE_BLOCK_SIZE
                if main_end > main_span_end or idx_end > idx_span_end:
                    valid = pl.cast(0, pl.INT32)
                if valid != 0:
                    for page in pl.range(main_page_begin, main_page_end):
                        if pl.read(main_pages, [page, 0]) < 0:
                            valid = pl.cast(0, pl.INT32)
                        if pl.read(main_pages, [page, 1]) != epoch:
                            valid = pl.cast(0, pl.INT32)
                    for page in pl.range(idx_page_begin, idx_page_end):
                        if pl.read(idx_pages, [page, 0]) < 0:
                            valid = pl.cast(0, pl.INT32)
                        if pl.read(idx_pages, [page, 1]) != epoch:
                            valid = pl.cast(0, pl.INT32)

                state_begin = pl.read(
                    state_descriptors, [request_index, PHASE_D_STATE_VALID_BEGIN]
                )
                state_end = pl.read(
                    state_descriptors, [request_index, PHASE_D_STATE_VALID_END]
                )
                inner_begin = pl.read(
                    inner_state_descriptors,
                    [request_index, PHASE_D_STATE_VALID_BEGIN],
                )
                inner_end = pl.read(
                    inner_state_descriptors,
                    [request_index, PHASE_D_STATE_VALID_END],
                )
                for state_relative_page in pl.range(CSA_STATE_PAGES_PER_REQUEST):
                    state_page_id = pl.read(
                        state_descriptors,
                        [request_index, PHASE_D_STATE_PAGE_ID_BASE + state_relative_page],
                    )
                    state_epoch = pl.read(
                        state_descriptors,
                        [request_index, PHASE_D_STATE_PAGE_EPOCH_BASE + state_relative_page],
                    )
                    inner_page_id = pl.read(
                        inner_state_descriptors,
                        [request_index, PHASE_D_STATE_PAGE_ID_BASE + state_relative_page],
                    )
                    inner_epoch = pl.read(
                        inner_state_descriptors,
                        [request_index, PHASE_D_STATE_PAGE_EPOCH_BASE + state_relative_page],
                    )
                    if state_page_id < 0 or state_epoch != epoch:
                        valid = pl.cast(0, pl.INT32)
                    if inner_page_id < 0 or inner_epoch != epoch:
                        valid = pl.cast(0, pl.INT32)
                if state_begin < 0 or state_end < state_begin:
                    valid = pl.cast(0, pl.INT32)
                if state_end > M.max_position_embeddings:
                    valid = pl.cast(0, pl.INT32)
                if state_end - state_begin > CSA_STATE_ROWS_PER_REQUEST:
                    valid = pl.cast(0, pl.INT32)
                if inner_begin < 0 or inner_end < inner_begin:
                    valid = pl.cast(0, pl.INT32)
                if inner_end > M.max_position_embeddings:
                    valid = pl.cast(0, pl.INT32)
                if inner_end - inner_begin > CSA_INNER_STATE_ROWS_PER_REQUEST:
                    valid = pl.cast(0, pl.INT32)

                previous_position = pl.cast(-1, pl.INT32)
                if valid != 0:
                    for query in pl.range(query_begin, query_end):
                        position = pl.read(position_ids, [query])
                        if pl.read(query_request_ids, [query]) != request:
                            valid = pl.cast(0, pl.INT32)
                        if position < 0 or position >= M.max_position_embeddings:
                            valid = pl.cast(0, pl.INT32)
                        if query > query_begin:
                            if position != previous_position + 1:
                                valid = pl.cast(0, pl.INT32)
                        if (position + 1) % CSA_COMPRESS_RATIO == 0:
                            if position // CSA_COMPRESS_RATIO < main_begin:
                                valid = pl.cast(0, pl.INT32)
                        previous_position = position
                step_candidate_end = main_end
                if query_end > query_begin:
                    last_position = pl.read(position_ids, [query_end - 1])
                    step_candidate_end = pl.cast(
                        pl.max(
                            step_candidate_end,
                            (last_position + 1) // CSA_COMPRESS_RATIO,
                        ),
                        pl.INT32,
                    )
                if step_candidate_end > main_span_end or step_candidate_end > idx_span_end:
                    valid = pl.cast(0, pl.INT32)

                if valid != 0:
                    pl.write(next_csa_ranges, [request, PHASE_D_NEXT_MAIN_BEGIN], main_begin)
                    pl.write(next_csa_ranges, [request, PHASE_D_NEXT_MAIN_END], step_candidate_end)
                    if query_end > query_begin:
                        last_position = pl.read(position_ids, [query_end - 1])
                        state_next_end = pl.cast(last_position + 1, pl.INT32)
                        state_next_begin = pl.cast(
                            last_position // CSA_COMPRESS_RATIO * CSA_COMPRESS_RATIO,
                            pl.INT32,
                        )
                        if state_next_end % CSA_COMPRESS_RATIO != 0:
                            state_next_begin = pl.cast(
                                pl.max(0, state_next_begin - CSA_COMPRESS_RATIO),
                                pl.INT32,
                            )
                        pl.write(next_csa_ranges, [request, PHASE_D_NEXT_STATE_BEGIN], state_next_begin)
                        pl.write(next_csa_ranges, [request, PHASE_D_NEXT_STATE_END], state_next_end)
                        pl.write(next_csa_ranges, [request, PHASE_D_NEXT_INNER_BEGIN], state_next_begin)
                        pl.write(next_csa_ranges, [request, PHASE_D_NEXT_INNER_END], state_next_end)

                for query in pl.range(query_begin, query_end):
                    position = pl.read(position_ids, [query])
                    if valid != 0:
                        state_ring_row = position % CSA_STATE_ROWS_PER_REQUEST
                        state_relative_page = state_ring_row // CSA_STATE_BLOCK_SIZE
                        state_page_id = pl.read(
                            state_descriptors,
                            [
                                request_index,
                                PHASE_D_STATE_PAGE_ID_BASE + state_relative_page,
                            ],
                        )
                        pl.write(
                            csa_state_write_slots,
                            [query, PHASE_D_STATE_MAIN_SLOT],
                            pl.cast(state_page_id, pl.INT64) * CSA_STATE_BLOCK_SIZE
                            + pl.cast(state_ring_row % CSA_STATE_BLOCK_SIZE, pl.INT64),
                        )
                        inner_state_ring_row = (
                            position % CSA_INNER_STATE_ROWS_PER_REQUEST
                        )
                        inner_state_relative_page = (
                            inner_state_ring_row // CSA_INNER_STATE_BLOCK_SIZE
                        )
                        inner_page_id = pl.read(
                            inner_state_descriptors,
                            [
                                request_index,
                                PHASE_D_STATE_PAGE_ID_BASE
                                + inner_state_relative_page,
                            ],
                        )
                        pl.write(
                            csa_state_write_slots,
                            [query, PHASE_D_STATE_INNER_SLOT],
                            pl.cast(inner_page_id, pl.INT64)
                            * CSA_INNER_STATE_BLOCK_SIZE
                            + pl.cast(
                                inner_state_ring_row % CSA_INNER_STATE_BLOCK_SIZE,
                                pl.INT64,
                            ),
                        )
                        if (position + 1) % CSA_COMPRESS_RATIO == 0:
                            event_row = pl.cast(position // CSA_COMPRESS_RATIO, pl.INT32)
                            relative_page = (
                                event_row // CACHE_BLOCK_SIZE
                                - main_begin // CACHE_BLOCK_SIZE
                            )
                            main_page_offset = (main_head + relative_page) % main_page_total
                            idx_page_offset = (idx_head + relative_page) % idx_page_total
                            main_page = pl.read(main_pages, [main_page_begin + main_page_offset, 0])
                            idx_page = pl.read(idx_pages, [idx_page_begin + idx_page_offset, 0])
                            pl.write(
                                csa_event_descriptors,
                                [event_cursor, PHASE_D_EVENT_QUERY],
                                pl.cast(query, pl.INT64),
                            )
                            pl.write(
                                csa_event_descriptors,
                                [event_cursor, PHASE_D_EVENT_ROW],
                                pl.cast(event_row, pl.INT64),
                            )
                            pl.write(
                                csa_event_descriptors,
                                [event_cursor, PHASE_D_EVENT_MAIN_SLOT],
                                pl.cast(main_page, pl.INT64) * CACHE_BLOCK_SIZE
                                + pl.cast(event_row % CACHE_BLOCK_SIZE, pl.INT64),
                            )
                            pl.write(
                                csa_event_descriptors,
                                [event_cursor, PHASE_D_EVENT_IDX_SLOT],
                                pl.cast(idx_page, pl.INT64) * CACHE_BLOCK_SIZE
                                + pl.cast(event_row % CACHE_BLOCK_SIZE, pl.INT64),
                            )
                            pl.write(csa_request_event_indices, [request], event_cursor)
                            event_cursor = pl.cast(event_cursor + 1, pl.INT32)

                    visible_begin = pl.cast(0, pl.INT32)
                    visible_end = pl.cast(0, pl.INT32)
                    if valid != 0:
                        visible_begin = main_begin
                        causal_end = pl.min(committed, position + 1) // CSA_COMPRESS_RATIO
                        visible_end = pl.cast(
                            pl.max(visible_begin, pl.min(main_end, causal_end)),
                            pl.INT32,
                        )
                        for overlay_query in pl.range(query_begin, query + 1):
                            overlay_position = pl.read(position_ids, [overlay_query])
                            if (overlay_position + 1) % CSA_COMPRESS_RATIO == 0:
                                if overlay_position // CSA_COMPRESS_RATIO == visible_end:
                                    visible_end = pl.cast(visible_end + 1, pl.INT32)
                    candidates = pl.cast(visible_end - visible_begin, pl.INT32)
                    pl.write(csa_visible_candidates, [query], candidates)
                    pl.write(csa_query_offsets, [query, PHASE_D_QUERY_OFFSET_LEAF], leaf_cursor)
                    pl.write(csa_query_offsets, [query, PHASE_D_QUERY_OFFSET_NODE], node_cursor)
                    pl.write(csa_query_offsets, [query, PHASE_D_QUERY_OFFSET_MERGE], merge_cursor)
                    pl.write(csa_query_offsets, [query, PHASE_D_QUERY_OFFSET_PAIR_GROUP], pair_cursor)
                    leaf_count = (
                        candidates + CSA_CANDIDATES_PER_LEAF - 1
                    ) // CSA_CANDIDATES_PER_LEAF
                    pair_count = leaf_count // CSA_MERGE_ARITY
                    for group in pl.range(pair_count):
                        credit = pl.cast(CSA_TOPK_INVALID_TASK_SLOT, pl.INT32)
                        if group >= CSA_TOPK_READY_FRONTIER_W:
                            credit = frontier[group - CSA_TOPK_READY_FRONTIER_W]
                        left_leaf = leaf_cursor
                        left_begin = pl.cast(
                            visible_begin + 2 * group * CSA_CANDIDATES_PER_LEAF,
                            pl.INT32,
                        )
                        left_valid = pl.cast(
                            pl.min(CSA_CANDIDATES_PER_LEAF, visible_end - left_begin),
                            pl.INT32,
                        )
                        left_slot = node_cursor
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_QUERY], pl.cast(query, pl.INT32))
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_BEGIN], left_begin)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_VALID], left_valid)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_OUTPUT_SLOT], left_slot)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_CREDIT_SLOT], credit)
                        leaf_cursor = pl.cast(leaf_cursor + 1, pl.INT32)
                        node_cursor = pl.cast(node_cursor + 1, pl.INT32)

                        right_leaf = leaf_cursor
                        right_begin = pl.cast(left_begin + CSA_CANDIDATES_PER_LEAF, pl.INT32)
                        right_valid = pl.cast(
                            pl.min(CSA_CANDIDATES_PER_LEAF, visible_end - right_begin),
                            pl.INT32,
                        )
                        right_slot = node_cursor
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_QUERY], pl.cast(query, pl.INT32))
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_BEGIN], right_begin)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_VALID], right_valid)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_OUTPUT_SLOT], right_slot)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_CREDIT_SLOT], credit)
                        leaf_cursor = pl.cast(leaf_cursor + 1, pl.INT32)
                        node_cursor = pl.cast(node_cursor + 1, pl.INT32)

                        output_slot = node_cursor
                        node_cursor = pl.cast(node_cursor + 1, pl.INT32)
                        frontier[group] = output_slot
                        pl.write(csa_pair_descriptors, [pair_cursor, PHASE_D_PAIR_LEFT_LEAF], left_leaf)
                        pl.write(csa_pair_descriptors, [pair_cursor, PHASE_D_PAIR_RIGHT_LEAF], right_leaf)
                        pl.write(csa_pair_descriptors, [pair_cursor, PHASE_D_PAIR_LEFT_SLOT], left_slot)
                        pl.write(csa_pair_descriptors, [pair_cursor, PHASE_D_PAIR_RIGHT_SLOT], right_slot)
                        pl.write(csa_pair_descriptors, [pair_cursor, PHASE_D_PAIR_OUTPUT_SLOT], output_slot)
                        pl.write(csa_pair_descriptors, [pair_cursor, PHASE_D_PAIR_CREDIT_SLOT], credit)
                        pair_cursor = pl.cast(pair_cursor + 1, pl.INT32)
                        pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_QUERY], pl.cast(query, pl.INT32))
                        pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_LEVEL], pl.cast(0, pl.INT32))
                        pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_LEFT_SLOT], left_slot)
                        pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_RIGHT_SLOT], right_slot)
                        pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_OUTPUT_SLOT], output_slot)
                        merge_cursor = pl.cast(merge_cursor + 1, pl.INT32)

                    frontier_count = pair_count
                    if leaf_count % CSA_MERGE_ARITY != 0:
                        credit = pl.cast(CSA_TOPK_INVALID_TASK_SLOT, pl.INT32)
                        if pair_count >= CSA_TOPK_READY_FRONTIER_W:
                            credit = frontier[pair_count - CSA_TOPK_READY_FRONTIER_W]
                        begin = pl.cast(
                            visible_begin + (leaf_count - 1) * CSA_CANDIDATES_PER_LEAF,
                            pl.INT32,
                        )
                        valid_tail = pl.cast(
                            pl.min(CSA_CANDIDATES_PER_LEAF, visible_end - begin),
                            pl.INT32,
                        )
                        leaf_id = leaf_cursor
                        output_slot = node_cursor
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_QUERY], pl.cast(query, pl.INT32))
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_BEGIN], begin)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_VALID], valid_tail)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_OUTPUT_SLOT], output_slot)
                        pl.write(csa_leaf_descriptors, [leaf_cursor, PHASE_D_LEAF_CREDIT_SLOT], credit)
                        pl.write(csa_singleton_descriptors, [singleton_cursor, PHASE_D_SINGLETON_LEAF], leaf_id)
                        pl.write(csa_singleton_descriptors, [singleton_cursor, PHASE_D_SINGLETON_SLOT], output_slot)
                        pl.write(csa_singleton_descriptors, [singleton_cursor, PHASE_D_SINGLETON_CREDIT_SLOT], credit)
                        frontier[frontier_count] = output_slot
                        leaf_cursor = pl.cast(leaf_cursor + 1, pl.INT32)
                        node_cursor = pl.cast(node_cursor + 1, pl.INT32)
                        singleton_cursor = pl.cast(singleton_cursor + 1, pl.INT32)
                        frontier_count = frontier_count + 1

                    for upper_level in pl.range(7):
                        if frontier_count > 1:
                            next_count = pl.cast(0, pl.INDEX)
                            for pair in pl.range(frontier_count // CSA_MERGE_ARITY):
                                left_slot = frontier[2 * pair]
                                right_slot = frontier[2 * pair + 1]
                                output_slot = node_cursor
                                node_cursor = pl.cast(node_cursor + 1, pl.INT32)
                                pl.write(csa_upper_merge_descriptors, [upper_cursor, PHASE_D_UPPER_LEFT_SLOT], left_slot)
                                pl.write(csa_upper_merge_descriptors, [upper_cursor, PHASE_D_UPPER_RIGHT_SLOT], right_slot)
                                pl.write(csa_upper_merge_descriptors, [upper_cursor, PHASE_D_UPPER_OUTPUT_SLOT], output_slot)
                                upper_cursor = pl.cast(upper_cursor + 1, pl.INT32)
                                pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_QUERY], pl.cast(query, pl.INT32))
                                pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_LEVEL], pl.cast(upper_level + 1, pl.INT32))
                                pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_LEFT_SLOT], left_slot)
                                pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_RIGHT_SLOT], right_slot)
                                pl.write(csa_merge_descriptors, [merge_cursor, PHASE_D_MERGE_OUTPUT_SLOT], output_slot)
                                merge_cursor = pl.cast(merge_cursor + 1, pl.INT32)
                                frontier[next_count] = output_slot
                                next_count = next_count + 1
                            if frontier_count % CSA_MERGE_ARITY != 0:
                                frontier[next_count] = frontier[frontier_count - 1]
                                next_count = next_count + 1
                            frontier_count = next_count

                    root = pl.cast(-1, pl.INT32)
                    root_dependency = pl.cast(CSA_TOPK_INVALID_TASK_SLOT, pl.INT32)
                    if frontier_count > 0:
                        root = frontier[0]
                        root_dependency = root
                    pl.write(csa_root_descriptors, [query, PHASE_D_ROOT_SLOT], root)
                    pl.write(csa_root_descriptors, [query, PHASE_D_ROOT_DEPENDENCY_SLOT], root_dependency)
                    pl.write(csa_query_offsets, [query + 1, PHASE_D_QUERY_OFFSET_LEAF], leaf_cursor)
                    pl.write(csa_query_offsets, [query + 1, PHASE_D_QUERY_OFFSET_NODE], node_cursor)
                    pl.write(csa_query_offsets, [query + 1, PHASE_D_QUERY_OFFSET_MERGE], merge_cursor)
                    pl.write(csa_query_offsets, [query + 1, PHASE_D_QUERY_OFFSET_PAIR_GROUP], pair_cursor)

    return (
        csa_visible_candidates,
        csa_query_offsets,
        csa_leaf_descriptors,
        csa_merge_descriptors,
        csa_pair_descriptors,
        csa_singleton_descriptors,
        csa_upper_merge_descriptors,
        csa_root_descriptors,
        csa_event_descriptors,
        csa_request_event_indices,
        csa_state_write_slots,
        next_csa_ranges,
    )


def _phase_d_request_for_length(
    length: int,
    request_id: int,
) -> PhaseDCsaRequest:
    if not 1 <= length <= M.max_position_embeddings:
        raise ValueError("Phase D fixture lengths must lie in [1, 1M]")
    position = length - 1
    candidates = length // CSA_COMPRESS_RATIO
    current_event = int(length % CSA_COMPRESS_RATIO == 0)
    pre_candidates = candidates - current_event
    page_count = max(
        1,
        (candidates + CACHE_BLOCK_SIZE - 1) // CACHE_BLOCK_SIZE,
    )
    epoch = 17 + request_id
    state_end = position
    state_begin = max(0, state_end - (CSA_STATE_ROWS_PER_REQUEST - 1))
    return PhaseDCsaRequest(
        committed_tokens=length,
        active=True,
        epoch=epoch,
        csa_valid_begin=0,
        csa_valid_end=pre_candidates,
        main_page_ids=tuple(
            10_000 + request_id * 20_000 + page for page in range(page_count)
        ),
        main_page_epochs=(epoch,) * page_count,
        index_page_ids=tuple(
            100_000 + request_id * 20_000 + page for page in range(page_count)
        ),
        index_page_epochs=(epoch,) * page_count,
        main_state_page_id=500 + request_id * CSA_STATE_PAGES_PER_REQUEST,
        main_state_page_epoch=epoch,
        main_state_valid_begin=state_begin,
        main_state_valid_end=state_end,
        inner_state_page_id=900 + request_id * CSA_INNER_STATE_PAGES_PER_REQUEST,
        inner_state_page_epoch=epoch,
        inner_state_valid_begin=state_begin,
        inner_state_valid_end=state_end,
    )


def _phase_d_fixture_from_lengths(
    lengths: tuple[int, ...],
) -> _PhaseDCsaFixture:
    requests = tuple(
        _phase_d_request_for_length(length, request_id)
        for request_id, length in enumerate(lengths)
    )
    queries = tuple(
        PhaseDCsaQuery(request_id, length - 1)
        for request_id, length in enumerate(lengths)
    )
    return _PhaseDCsaFixture(
        requests,
        queries,
        _phase_d_offsets(len(requests), queries),
    )


def _phase_d_two_query_fixture(
    positions: tuple[int, int],
) -> _PhaseDCsaFixture:
    request = _phase_d_request_for_length(positions[-1] + 1, 0)
    first_state = max(0, positions[0] - (CSA_STATE_ROWS_PER_REQUEST - 1))
    request = PhaseDCsaRequest(
        **{
            **request.__dict__,
            "main_state_valid_begin": first_state,
            "main_state_valid_end": positions[0],
            "inner_state_valid_begin": first_state,
            "inner_state_valid_end": positions[0],
        }
    )
    queries = (PhaseDCsaQuery(0, positions[0]), PhaseDCsaQuery(0, positions[1]))
    return _PhaseDCsaFixture((request,), queries, (0, 2))


def _case_csa_length_matrix() -> _PhaseDCsaFixture:
    return _phase_d_fixture_from_lengths(
        (1, 3, 4, 128, 8192, 12_288, 16_384, 16_388, 65_536, 1_048_572, 1_048_576)
    )


def _case_csa_candidate_boundaries() -> _PhaseDCsaFixture:
    lengths = tuple(
        3 if candidates == 0 else candidates * CSA_COMPRESS_RATIO
        for candidates in (
            0, 1, 511, 512, 513, 2047, 2048, 2049, 3072, 4096,
            4097, 16383, 16384, MAX_CSA_CANDIDATES - 1,
            MAX_CSA_CANDIDATES,
        )
    )
    return _phase_d_fixture_from_lengths(lengths)


def _case_csa_no_event() -> _PhaseDCsaFixture:
    return _phase_d_two_query_fixture((1, 2))


def _case_csa_boundary_2_3() -> _PhaseDCsaFixture:
    return _phase_d_two_query_fixture((2, 3))


def _case_csa_rollover_3_4() -> _PhaseDCsaFixture:
    return _phase_d_two_query_fixture((3, 4))


def _case_csa_second_boundary_6_7() -> _PhaseDCsaFixture:
    return _phase_d_two_query_fixture((6, 7))


def _case_csa_second_rollover_7_8() -> _PhaseDCsaFixture:
    return _phase_d_two_query_fixture((7, 8))


def _case_csa_paired_slot_permutation() -> _PhaseDCsaFixture:
    request = PhaseDCsaRequest(
        committed_tokens=516,
        active=True,
        epoch=6,
        csa_valid_begin=127,
        csa_valid_end=129,
        main_page_ids=(41, 9),
        main_page_epochs=(6, 6),
        index_page_ids=(31, 7),
        index_page_epochs=(6, 6),
        main_state_page_id=5,
        main_state_page_epoch=6,
        main_state_valid_begin=508,
        main_state_valid_end=515,
        inner_state_page_id=13,
        inner_state_page_epoch=6,
        inner_state_valid_begin=508,
        inner_state_valid_end=515,
        main_head=1,
        index_head=0,
    )
    queries = (PhaseDCsaQuery(0, 515),)
    return _PhaseDCsaFixture((request,), queries, (0, 1))


def _case_csa_cross_page_leaf() -> _PhaseDCsaFixture:
    epoch = 4
    pages = tuple(range(65))
    request = PhaseDCsaRequest(
        committed_tokens=8196,
        active=True,
        epoch=epoch,
        csa_valid_begin=1,
        csa_valid_end=2048,
        main_page_ids=tuple(200 + page for page in pages),
        main_page_epochs=(epoch,) * 65,
        index_page_ids=tuple(500 + page for page in pages),
        index_page_epochs=(epoch,) * 65,
        main_state_page_id=7,
        main_state_page_epoch=epoch,
        main_state_valid_begin=8188,
        main_state_valid_end=8195,
        inner_state_page_id=8,
        inner_state_page_epoch=epoch,
        inner_state_valid_begin=8188,
        inner_state_valid_end=8195,
        main_head=5,
        index_head=11,
    )
    return _PhaseDCsaFixture((request,), (PhaseDCsaQuery(0, 8195),), (0, 1))


def _case_csa_odd_leaf_carry() -> _PhaseDCsaFixture:
    return _phase_d_fixture_from_lengths((4097 * CSA_COMPRESS_RATIO,))


def _case_csa_stale_epoch_rejected() -> _PhaseDCsaFixture:
    fixture = _phase_d_fixture_from_lengths((16_384,))
    request = fixture.requests[0]
    stale = PhaseDCsaRequest(
        **{
            **request.__dict__,
            "index_page_epochs": tuple(request.epoch - 1 for _ in request.index_page_ids),
        }
    )
    return _PhaseDCsaFixture((stale,), fixture.queries, fixture.request_query_offsets)


def _case_csa_missing_page_rejected() -> _PhaseDCsaFixture:
    fixture = _phase_d_fixture_from_lengths((128,))
    request = fixture.requests[0]
    missing = PhaseDCsaRequest(
        **{
            **request.__dict__,
            "main_page_ids": (),
            "main_page_epochs": (),
        }
    )
    return _PhaseDCsaFixture((missing,), fixture.queries, fixture.request_query_offsets)


def _case_csa_mismatched_range_rejected() -> _PhaseDCsaFixture:
    fixture = _phase_d_fixture_from_lengths((8192,))
    request = fixture.requests[0]
    mismatch = PhaseDCsaRequest(
        **{
            **request.__dict__,
            "index_valid_begin": request.csa_valid_begin,
            "index_valid_end": request.csa_valid_end - 1,
        }
    )
    return _PhaseDCsaFixture((mismatch,), fixture.queries, fixture.request_query_offsets)


def _case_csa_inactive_boundary() -> _PhaseDCsaFixture:
    fixture = _phase_d_fixture_from_lengths((4,))
    request = fixture.requests[0]
    inactive = PhaseDCsaRequest(**{**request.__dict__, "active": False})
    return _PhaseDCsaFixture((inactive,), fixture.queries, fixture.request_query_offsets)


def _case_csa_heterogeneous_lengths() -> _PhaseDCsaFixture:
    return _phase_d_fixture_from_lengths((3, 128, 16_384, M.max_position_embeddings))


def _case_csa_one_m_tail() -> _PhaseDCsaFixture:
    return _phase_d_fixture_from_lengths((M.max_position_embeddings - 4, M.max_position_embeddings))


def _case_csa_query_capacity() -> _PhaseDCsaFixture:
    """Exercise the 16-query admission ceiling with one aggregate 1M history."""
    return _phase_d_fixture_from_lengths(
        (M.max_position_embeddings,) + (3,) * (CSA_MAX_QUERIES - 1)
    )


_PHASE_D_CASES = {
    "csa_length_matrix": _case_csa_length_matrix,
    "csa_candidate_boundaries": _case_csa_candidate_boundaries,
    "csa_no_event": _case_csa_no_event,
    "csa_boundary_2_3": _case_csa_boundary_2_3,
    "csa_rollover_3_4": _case_csa_rollover_3_4,
    "csa_second_boundary_6_7": _case_csa_second_boundary_6_7,
    "csa_second_rollover_7_8": _case_csa_second_rollover_7_8,
    "csa_paired_slot_permutation": _case_csa_paired_slot_permutation,
    "csa_cross_page_leaf": _case_csa_cross_page_leaf,
    "csa_odd_leaf_carry": _case_csa_odd_leaf_carry,
    "csa_stale_epoch_rejected": _case_csa_stale_epoch_rejected,
    "csa_missing_page_rejected": _case_csa_missing_page_rejected,
    "csa_mismatched_range_rejected": _case_csa_mismatched_range_rejected,
    "csa_inactive_boundary": _case_csa_inactive_boundary,
    "csa_heterogeneous_lengths": _case_csa_heterogeneous_lengths,
    "csa_one_m_tail": _case_csa_one_m_tail,
    "csa_query_capacity": _case_csa_query_capacity,
}


def run_phase_d_case(case: str) -> bool:
    fixture = _PHASE_D_CASES[case]()
    result = build_phase_d_csa_metadata(
        fixture.requests,
        fixture.queries,
        fixture.request_query_offsets,
    )
    leaf_counts = [
        result.csa_query_leaf_offsets[q + 1] - result.csa_query_leaf_offsets[q]
        for q in range(len(fixture.queries))
    ]
    if result.n_nodes != sum(max(2 * leaves - 1, 0) for leaves in leaf_counts):
        return False
    if len(result.csa_pair_output_slots) != sum(leaves // 2 for leaves in leaf_counts):
        return False
    if len(result.csa_singleton_slots) != sum(leaves % 2 for leaves in leaf_counts):
        return False
    if case == "csa_heterogeneous_lengths" and leaf_counts != [0, 1, 2, 128]:
        print(f"[Phase D case={case}] leaf mismatch: {leaf_counts}")
        return False
    if case in {
        "csa_stale_epoch_rejected",
        "csa_missing_page_rejected",
        "csa_mismatched_range_rejected",
        "csa_inactive_boundary",
    }:
        if any(leaf_counts) or result.csa_event_rows:
            return False
    if case == "csa_cross_page_leaf":
        if result.csa_leaf_candidate_begin != (1,):
            return False
        if result.csa_leaf_valid_candidates != (2048,):
            return False
    print(
        f"[Phase D host case={case}] PASS Q={len(fixture.queries)} "
        f"N_leaf={result.n_leaves} N_merge={result.n_merges} "
        f"N_event={len(result.csa_event_rows)}"
    )
    return True


def _pack_phase_d_inputs(fixture: _PhaseDCsaFixture):
    import torch

    main_pages: list[tuple[int, int]] = []
    main_offsets = [0]
    idx_pages: list[tuple[int, int]] = []
    idx_offsets = [0]
    for request in fixture.requests:
        main_pages.extend(zip(request.main_page_ids, request.main_page_epochs))
        main_offsets.append(len(main_pages))
        idx_pages.extend(zip(request.index_page_ids, request.index_page_epochs))
        idx_offsets.append(len(idx_pages))
    return {
        "request_state": torch.tensor(
            [[r.committed_tokens, int(r.active), r.epoch] for r in fixture.requests],
            dtype=torch.int32,
        ),
        "query_request_ids": torch.tensor(
            [q.request_id for q in fixture.queries], dtype=torch.int32
        ),
        "request_query_offsets": torch.tensor(
            fixture.request_query_offsets, dtype=torch.int32
        ),
        "position_ids": torch.tensor(
            [q.position for q in fixture.queries], dtype=torch.int32
        ),
        "main_pages": torch.tensor(main_pages, dtype=torch.int32).reshape(-1, 2),
        "main_page_offsets": torch.tensor(main_offsets, dtype=torch.int32),
        "main_windows": torch.tensor(
            [
                [r.csa_valid_begin, r.csa_valid_end, r.main_head]
                for r in fixture.requests
            ],
            dtype=torch.int32,
        ),
        "idx_pages": torch.tensor(idx_pages, dtype=torch.int32).reshape(-1, 2),
        "idx_page_offsets": torch.tensor(idx_offsets, dtype=torch.int32),
        "idx_windows": torch.tensor(
            [
                [
                    r.csa_valid_begin if r.index_valid_begin is None else r.index_valid_begin,
                    r.csa_valid_end if r.index_valid_end is None else r.index_valid_end,
                    r.index_head,
                ]
                for r in fixture.requests
            ],
            dtype=torch.int32,
        ),
        "state_descriptors": torch.tensor(
            [
                list(r.main_state_pages)
                + list(r.main_state_epochs)
                + [r.main_state_valid_begin, r.main_state_valid_end]
                if len(r.main_state_pages) == CSA_STATE_PAGES_PER_REQUEST
                else [-1] * (2 * CSA_STATE_PAGES_PER_REQUEST)
                + [r.main_state_valid_begin, r.main_state_valid_end]
                for r in fixture.requests
            ],
            dtype=torch.int32,
        ),
        "inner_state_descriptors": torch.tensor(
            [
                list(r.inner_state_pages)
                + list(r.inner_state_epochs)
                + [r.inner_state_valid_begin, r.inner_state_valid_end]
                if len(r.inner_state_pages) == CSA_INNER_STATE_PAGES_PER_REQUEST
                else [-1] * (2 * CSA_INNER_STATE_PAGES_PER_REQUEST)
                + [r.inner_state_valid_begin, r.inner_state_valid_end]
                for r in fixture.requests
            ],
            dtype=torch.int32,
        ),
    }


def _phase_d_requests_from_tensors(tensors) -> tuple[PhaseDCsaRequest, ...]:
    requests: list[PhaseDCsaRequest] = []
    for request in range(int(tensors["request_state"].shape[0])):
        main_begin = int(tensors["main_page_offsets"][request])
        main_end = int(tensors["main_page_offsets"][request + 1])
        idx_begin = int(tensors["idx_page_offsets"][request])
        idx_end = int(tensors["idx_page_offsets"][request + 1])
        requests.append(
            PhaseDCsaRequest(
                committed_tokens=int(tensors["request_state"][request, 0]),
                active=bool(int(tensors["request_state"][request, 1])),
                epoch=int(tensors["request_state"][request, 2]),
                csa_valid_begin=int(tensors["main_windows"][request, 0]),
                csa_valid_end=int(tensors["main_windows"][request, 1]),
                main_head=int(tensors["main_windows"][request, 2]),
                main_page_ids=tuple(
                    int(tensors["main_pages"][page, 0])
                    for page in range(main_begin, main_end)
                ),
                main_page_epochs=tuple(
                    int(tensors["main_pages"][page, 1])
                    for page in range(main_begin, main_end)
                ),
                index_valid_begin=int(tensors["idx_windows"][request, 0]),
                index_valid_end=int(tensors["idx_windows"][request, 1]),
                index_head=int(tensors["idx_windows"][request, 2]),
                index_page_ids=tuple(
                    int(tensors["idx_pages"][page, 0])
                    for page in range(idx_begin, idx_end)
                ),
                index_page_epochs=tuple(
                    int(tensors["idx_pages"][page, 1])
                    for page in range(idx_begin, idx_end)
                ),
                main_state_page_id=-1,
                main_state_page_epoch=-1,
                main_state_valid_begin=int(
                    tensors["state_descriptors"][request, PHASE_D_STATE_VALID_BEGIN]
                ),
                main_state_valid_end=int(
                    tensors["state_descriptors"][request, PHASE_D_STATE_VALID_END]
                ),
                main_state_page_ids=tuple(
                    int(tensors["state_descriptors"][request, PHASE_D_STATE_PAGE_ID_BASE + page])
                    for page in range(CSA_STATE_PAGES_PER_REQUEST)
                ),
                main_state_page_epochs=tuple(
                    int(tensors["state_descriptors"][request, PHASE_D_STATE_PAGE_EPOCH_BASE + page])
                    for page in range(CSA_STATE_PAGES_PER_REQUEST)
                ),
                inner_state_page_id=-1,
                inner_state_page_epoch=-1,
                inner_state_valid_begin=int(
                    tensors["inner_state_descriptors"][request, PHASE_D_STATE_VALID_BEGIN]
                ),
                inner_state_valid_end=int(
                    tensors["inner_state_descriptors"][request, PHASE_D_STATE_VALID_END]
                ),
                inner_state_page_ids=tuple(
                    int(tensors["inner_state_descriptors"][request, PHASE_D_STATE_PAGE_ID_BASE + page])
                    for page in range(CSA_INNER_STATE_PAGES_PER_REQUEST)
                ),
                inner_state_page_epochs=tuple(
                    int(tensors["inner_state_descriptors"][request, PHASE_D_STATE_PAGE_EPOCH_BASE + page])
                    for page in range(CSA_INNER_STATE_PAGES_PER_REQUEST)
                ),
            )
        )
    return tuple(requests)


def golden_phase_d_csa_metadata(tensors):
    requests = _phase_d_requests_from_tensors(tensors)
    queries = tuple(
        PhaseDCsaQuery(
            int(tensors["query_request_ids"][query]),
            int(tensors["position_ids"][query]),
        )
        for query in range(int(tensors["position_ids"].shape[0]))
    )
    offsets = tuple(int(value) for value in tensors["request_query_offsets"])
    result = build_phase_d_csa_metadata(requests, queries, offsets)
    packed = {
        "csa_visible_candidates": result.csa_visible_candidates,
        "csa_query_offsets": tuple(
            zip(
                result.csa_query_leaf_offsets,
                result.csa_query_node_offsets,
                result.csa_query_merge_offsets,
                result.csa_query_pair_group_offsets,
            )
        ),
        "csa_leaf_descriptors": tuple(
            zip(
                result.csa_leaf_query_ids,
                result.csa_leaf_candidate_begin,
                result.csa_leaf_valid_candidates,
                result.csa_leaf_output_slots,
                result.csa_leaf_credit_predecessors,
            )
        ),
        "csa_merge_descriptors": tuple(
            zip(
                result.csa_merge_query_ids,
                result.csa_merge_levels,
                result.csa_merge_left_slots,
                result.csa_merge_right_slots,
                result.csa_merge_output_slots,
            )
        ),
        "csa_pair_descriptors": tuple(
            zip(
                result.csa_pair_left_leaf_ids,
                result.csa_pair_right_leaf_ids,
                result.csa_pair_left_slots,
                result.csa_pair_right_slots,
                result.csa_pair_output_slots,
                result.csa_pair_credit_slots,
            )
        ),
        "csa_singleton_descriptors": tuple(
            zip(
                result.csa_singleton_leaf_ids,
                result.csa_singleton_slots,
                result.csa_singleton_credit_slots,
            )
        ),
        "csa_upper_merge_descriptors": tuple(
            zip(
                result.csa_upper_left_slots,
                result.csa_upper_right_slots,
                result.csa_upper_output_slots,
            )
        ),
        "csa_root_descriptors": tuple(
            zip(result.csa_root_slots, result.csa_root_dependency_slots)
        ),
        "csa_event_descriptors": tuple(
            zip(
                result.csa_event_query_ids,
                result.csa_event_rows,
                result.csa_main_event_write_slots,
                result.csa_idx_event_write_slots,
            )
        ),
        "csa_request_event_indices": result.csa_request_event_indices,
        "csa_state_write_slots": tuple(
            zip(result.csa_state_write_slots, result.csa_inner_state_write_slots)
        ),
        "next_csa_ranges": tuple(
            main + state + inner
            for main, state, inner in zip(
                result.next_csa_valid_ranges,
                result.next_csa_state_valid_ranges,
                result.next_csa_inner_state_valid_ranges,
            )
        ),
    }
    for name, value in packed.items():
        tensors[name].copy_(tensors[name].new_tensor(value).reshape_as(tensors[name]))


def build_phase_d_csa_tensor_specs(case: str):
    import torch
    from golden import TensorSpec

    fixture = _PHASE_D_CASES[case]()
    result = build_phase_d_csa_metadata(
        fixture.requests,
        fixture.queries,
        fixture.request_query_offsets,
    )
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in _pack_phase_d_inputs(fixture).items()
    ]
    q = len(fixture.queries)
    r = len(fixture.requests)
    pair_count = len(result.csa_pair_left_leaf_ids)
    singleton_count = len(result.csa_singleton_leaf_ids)
    upper_count = len(result.csa_upper_left_slots)
    event_count = len(result.csa_event_rows)
    shapes = {
        "csa_visible_candidates": ([q], torch.int32),
        "csa_query_offsets": ([q + 1, PHASE_D_QUERY_OFFSET_FIELDS], torch.int32),
        "csa_leaf_descriptors": ([result.n_leaves, PHASE_D_LEAF_FIELDS], torch.int32),
        "csa_merge_descriptors": ([result.n_merges, PHASE_D_MERGE_FIELDS], torch.int32),
        "csa_pair_descriptors": ([pair_count, PHASE_D_PAIR_FIELDS], torch.int32),
        "csa_singleton_descriptors": (
            [singleton_count, PHASE_D_SINGLETON_FIELDS],
            torch.int32,
        ),
        "csa_upper_merge_descriptors": (
            [upper_count, PHASE_D_UPPER_FIELDS],
            torch.int32,
        ),
        "csa_root_descriptors": ([q, PHASE_D_ROOT_FIELDS], torch.int32),
        "csa_event_descriptors": ([event_count, PHASE_D_EVENT_FIELDS], torch.int64),
        "csa_request_event_indices": ([r], torch.int32),
        "csa_state_write_slots": ([q, PHASE_D_STATE_SLOT_FIELDS], torch.int64),
        "next_csa_ranges": ([r, PHASE_D_NEXT_FIELDS], torch.int32),
    }
    specs.extend(
        TensorSpec(name, shape, dtype, is_output=True)
        for name, (shape, dtype) in shapes.items()
    )
    return specs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="Run the pure-Python reference for --case without compiling.",
    )
    parser.add_argument(
        "--case",
        choices=(
            list(_PHASE_A_CASES.keys())
            + list(_PHASE_B_CASES.keys())
            + list(_PHASE_C_CASES.keys())
            + list(_PHASE_D_CASES.keys())
        ),
        default=None,
        help="Run one Phase A, Phase B, Phase C, or Phase D metadata case. When "
        "omitted, run the legacy dense migration-fence regression.",
    )
    args = parser.parse_args()

    if args.case is not None:
        if args.case in _PHASE_A_CASES:
            if args.host_only:
                ok = run_phase_a_case(args.case)
                if ok:
                    print(f"[Phase A host case={args.case}] PASS")
                    raise SystemExit(0)
                raise SystemExit(1)

            from golden import run_jit

            result = run_jit(
                fn=phase_a_decode_metadata,
                specs=build_phase_a_tensor_specs(args.case),
                golden_fn=golden_phase_a_decode_metadata,
                compile_only=args.compile_only,
                runtime_cfg={"platform": args.platform, "device_id": args.device},
            )
        elif args.case in _PHASE_B_CASES:
            if args.host_only:
                ok = run_phase_b_case(args.case)
                if ok:
                    print(f"[Phase B host case={args.case}] PASS")
                    raise SystemExit(0)
                raise SystemExit(1)

            from golden import run_jit

            result = run_jit(
                fn=phase_b_swa_metadata,
                specs=build_phase_b_swa_tensor_specs(args.case),
                golden_fn=golden_phase_b_swa_metadata,
                compile_only=args.compile_only,
                runtime_cfg={"platform": args.platform, "device_id": args.device},
            )
        elif args.case in _PHASE_C_CASES:
            if args.host_only:
                ok = run_phase_c_case(args.case)
                if ok:
                    raise SystemExit(0)
                raise SystemExit(1)

            from golden import run_jit

            result = run_jit(
                fn=phase_c_hca_metadata,
                specs=build_phase_c_hca_tensor_specs(args.case),
                golden_fn=golden_phase_c_hca_metadata,
                compile_only=args.compile_only,
                runtime_cfg={"platform": args.platform, "device_id": args.device},
            )
        else:
            if args.host_only:
                ok = run_phase_d_case(args.case)
                if ok:
                    raise SystemExit(0)
                raise SystemExit(1)

            from golden import run_jit

            result = run_jit(
                fn=phase_d_csa_metadata,
                specs=build_phase_d_csa_tensor_specs(args.case),
                golden_fn=golden_phase_d_csa_metadata,
                compile_only=args.compile_only,
                runtime_cfg={"platform": args.platform, "device_id": args.device},
            )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
        raise SystemExit(0)

    from golden import run_jit

    result = run_jit(
        fn=decode_metadata,
        specs=build_tensor_specs(),
        golden_fn=golden_decode_metadata,
        compile_only=args.compile_only,
        runtime_cfg={"platform": args.platform, "device_id": args.device},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
