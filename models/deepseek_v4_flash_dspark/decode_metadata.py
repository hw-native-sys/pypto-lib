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

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    C4A_COMPRESSOR_BLOCK_SIZE,
    C128_COMPRESSOR_BLOCK_SIZE,
    DECODE_BATCH,
    DECODE_SEQ,
    FLASH as M,
    IDX_CACHE_MAX_BLOCKS,
    KV_CMP_MAX_BLOCKS,
    KV_ORI_TABLE_MAX_BLOCKS,
)


B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
WIN = M.sliding_window
ORI_TABLE_MAX_BLOCKS = KV_ORI_TABLE_MAX_BLOCKS
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
    ori_block_table: pl.Tensor[[B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
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
    ori_block_table: pl.Tensor[[B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
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
    ori_block_table: pl.Tensor[[B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
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
        "ori_block_table": table(ORI_TABLE_MAX_BLOCKS, GROUP_ORI, repeat=True),
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


if __name__ == "__main__":
    import argparse

    from golden import run_jit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-p",
        "--platform",
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

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
