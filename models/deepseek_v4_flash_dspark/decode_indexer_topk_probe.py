# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone exact Top-K leaf and dynamic forest probes for the CSA indexer."""

import pypto.language as pl

from decode_indexer import (
    FP32_NEG_INF,
    IDX_TOPK,
    TOPK_CANDIDATES_PER_LEAF,
    TOPK_MAX_CANDIDATES,
    TOPK_MAX_NODES,
    TOPK_LEVEL1_BASE,
    TOPK_LEVEL2_BASE,
    TOPK_LEVEL3_BASE,
    TOPK_LEVEL4_BASE,
    TOPK_LEVEL5_BASE,
    TOPK_LEVEL6_BASE,
    TOPK_ROOT_BASE,
    TOPK_PAIR_WIDTH,
    merge_topk_level_pairs,
    select_2k_top512_pairs,
)


LEAF_DYN = pl.dynamic("TOPK_PROBE_LEAF_DYN")
QUERY_DYN = pl.dynamic("TOPK_PROBE_QUERY_DYN")
PROBE_QUERY_CAP = 16
TOPK_ARENA_ROWS = PROBE_QUERY_CAP * TOPK_MAX_NODES


@pl.jit.incore
def _topk_leaf_probe_body(
    scores: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    indices: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.INT32],
    valid_counts: pl.Tensor[[LEAF_DYN], pl.INT32],
    topk_scores: pl.Tensor[[LEAF_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[LEAF_DYN, IDX_TOPK], pl.INT32],
) -> None:
    leaf = pl.tile.get_block_idx()
    valid_count = pl.cast(pl.read(valid_counts, [leaf]), pl.INDEX)
    leaf_pairs = select_2k_top512_pairs(
        scores, indices, leaf
    )
    topk_scores[leaf : leaf + 1, :] = pl.gather(
        leaf_pairs,
        mask_pattern=pl.tile.MaskPattern.P0101,
        output_dtype=pl.FP32,
    )
    leaf_topk_indices = pl.gather(
        leaf_pairs,
        mask_pattern=pl.tile.MaskPattern.P1010,
        output_dtype=pl.INT32,
    )
    output_indices = pl.tile.full(
        [1, IDX_TOPK], dtype=pl.INT32, value=-1
    )
    valid_topk = pl.min(valid_count, IDX_TOPK)
    for lane in pl.range(valid_topk):
        pl.tile.write(
            output_indices,
            [0, lane],
            pl.read(leaf_topk_indices, [0, lane]),
        )
    pl.store(output_indices, [leaf, 0], topk_indices)


@pl.jit
def topk_leaf_probe(
    scores: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    indices: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.INT32],
    valid_counts: pl.Tensor[[LEAF_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[LEAF_DYN, IDX_TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[LEAF_DYN, IDX_TOPK], pl.INT32]],
):
    scores.bind_dynamic(0, LEAF_DYN)
    indices.bind_dynamic(0, LEAF_DYN)
    valid_counts.bind_dynamic(0, LEAF_DYN)
    topk_scores.bind_dynamic(0, LEAF_DYN)
    topk_indices.bind_dynamic(0, LEAF_DYN)

    leaf_count = pl.tensor.dim(scores, 0)
    with pl.spmd(leaf_count, name_hint="topk_leaf_probe"):
        _topk_leaf_probe_body(
            scores,
            indices,
            valid_counts,
            topk_scores,
            topk_indices,
        )
    return topk_scores, topk_indices


@pl.jit.incore
def _topk_forest_probe_body(
    leaf_scores: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    leaf_indices: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.INT32],
    leaf_offsets: pl.Tensor[[QUERY_DYN], pl.INT32],
    candidate_counts: pl.Tensor[[QUERY_DYN], pl.INT32],
    pair_arena: pl.Tensor[[TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], pl.FP32],
    topk_scores: pl.Tensor[[QUERY_DYN, IDX_TOPK], pl.FP32],
    topk_indices: pl.Tensor[[QUERY_DYN, IDX_TOPK], pl.INT32],
) -> None:
    query = pl.tile.get_block_idx()
    candidate_count = pl.cast(pl.read(candidate_counts, [query]), pl.INDEX)
    leaf_offset = pl.cast(pl.read(leaf_offsets, [query]), pl.INDEX)
    leaf_count = (
        candidate_count + TOPK_CANDIDATES_PER_LEAF - 1
    ) // TOPK_CANDIDATES_PER_LEAF
    arena_base = query * TOPK_MAX_NODES
    topk_scores[query : query + 1, :] = pl.full(
        [1, IDX_TOPK], dtype=pl.FP32, value=FP32_NEG_INF
    )
    topk_indices[query : query + 1, :] = pl.full(
        [1, IDX_TOPK], dtype=pl.INT32, value=-1
    )

    if leaf_count > 0:
        for leaf in pl.range(leaf_count):
            valid_count = pl.min(
                TOPK_CANDIDATES_PER_LEAF,
                candidate_count - leaf * TOPK_CANDIDATES_PER_LEAF,
            )
            leaf_id = leaf_offset + leaf
            leaf_pairs = select_2k_top512_pairs(
                leaf_scores,
                leaf_indices,
                leaf_id,
            )
            pair_arena[
                arena_base + leaf : arena_base + leaf + 1, :
            ] = leaf_pairs

        level1_count = (leaf_count + 1) // 2
        merge_topk_level_pairs(
            pair_arena, arena_base, leaf_count, 0, TOPK_LEVEL1_BASE
        )
        level2_count = (level1_count + 1) // 2
        merge_topk_level_pairs(
            pair_arena,
            arena_base,
            level1_count,
            TOPK_LEVEL1_BASE,
            TOPK_LEVEL2_BASE,
        )
        level3_count = (level2_count + 1) // 2
        merge_topk_level_pairs(
            pair_arena,
            arena_base,
            level2_count,
            TOPK_LEVEL2_BASE,
            TOPK_LEVEL3_BASE,
        )
        level4_count = (level3_count + 1) // 2
        merge_topk_level_pairs(
            pair_arena,
            arena_base,
            level3_count,
            TOPK_LEVEL3_BASE,
            TOPK_LEVEL4_BASE,
        )
        level5_count = (level4_count + 1) // 2
        merge_topk_level_pairs(
            pair_arena,
            arena_base,
            level4_count,
            TOPK_LEVEL4_BASE,
            TOPK_LEVEL5_BASE,
        )
        level6_count = (level5_count + 1) // 2
        merge_topk_level_pairs(
            pair_arena,
            arena_base,
            level5_count,
            TOPK_LEVEL5_BASE,
            TOPK_LEVEL6_BASE,
        )
        merge_topk_level_pairs(
            pair_arena,
            arena_base,
            level6_count,
            TOPK_LEVEL6_BASE,
            TOPK_ROOT_BASE,
        )

        root_slot = arena_base + TOPK_ROOT_BASE
        root_pairs = pair_arena[root_slot : root_slot + 1, :]
        topk_scores[query : query + 1, :] = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P0101,
            output_dtype=pl.FP32,
        )
        root_indices = pl.gather(
            root_pairs,
            mask_pattern=pl.tile.MaskPattern.P1010,
            output_dtype=pl.INT32,
        )
        output_indices = pl.tile.full(
            [1, IDX_TOPK], dtype=pl.INT32, value=-1
        )
        valid_topk = pl.min(candidate_count, IDX_TOPK)
        for lane in pl.range(valid_topk):
            pl.tile.write(
                output_indices,
                [0, lane],
                pl.read(root_indices, [0, lane]),
            )
        pl.store(output_indices, [query, 0], topk_indices)


@pl.jit
def topk_forest_probe(
    leaf_scores: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.FP32],
    leaf_indices: pl.Tensor[[LEAF_DYN, TOPK_CANDIDATES_PER_LEAF], pl.INT32],
    leaf_offsets: pl.Tensor[[QUERY_DYN], pl.INT32],
    candidate_counts: pl.Tensor[[QUERY_DYN], pl.INT32],
    topk_scores: pl.Out[pl.Tensor[[QUERY_DYN, IDX_TOPK], pl.FP32]],
    topk_indices: pl.Out[pl.Tensor[[QUERY_DYN, IDX_TOPK], pl.INT32]],
):
    leaf_scores.bind_dynamic(0, LEAF_DYN)
    leaf_indices.bind_dynamic(0, LEAF_DYN)
    leaf_offsets.bind_dynamic(0, QUERY_DYN)
    candidate_counts.bind_dynamic(0, QUERY_DYN)
    topk_scores.bind_dynamic(0, QUERY_DYN)
    topk_indices.bind_dynamic(0, QUERY_DYN)

    query_count = pl.tensor.dim(candidate_counts, 0)
    pair_arena = pl.create_tensor(
        [TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], dtype=pl.FP32
    )
    with pl.spmd(query_count, name_hint="topk_forest_probe"):
        _topk_forest_probe_body(
            leaf_scores,
            leaf_indices,
            leaf_offsets,
            candidate_counts,
            pair_arena,
            topk_scores,
            topk_indices,
        )
    return topk_scores, topk_indices


def _topk_with_padding(scores, indices=None):
    import torch

    output_scores = torch.full((IDX_TOPK,), FP32_NEG_INF, dtype=torch.float32)
    output_indices = torch.full((IDX_TOPK,), -1, dtype=torch.int32)
    if scores.numel() == 0:
        return output_scores, output_indices
    count = min(IDX_TOPK, scores.numel())
    selected_scores, selected_positions = torch.topk(scores, count)
    output_scores[:count] = selected_scores
    if indices is None:
        output_indices[:count] = selected_positions.to(torch.int32)
    else:
        output_indices[:count] = indices[selected_positions].to(torch.int32)
    return output_scores, output_indices


def build_leaf_probe_specs():
    import torch
    from golden import TensorSpec

    valid_counts = torch.tensor([1, 511, 512, 513, 2047, 2048], dtype=torch.int32)
    logical_begins = torch.arange(valid_counts.numel(), dtype=torch.int32) * 4096
    indices = logical_begins[:, None] + torch.arange(
        TOPK_CANDIDATES_PER_LEAF, dtype=torch.int32
    )[None, :]
    generator = torch.Generator().manual_seed(20260821)
    scores = torch.randn(
        valid_counts.numel(), TOPK_CANDIDATES_PER_LEAF, generator=generator
    )
    for leaf, valid_count in enumerate(valid_counts.tolist()):
        scores[leaf, valid_count:] = FP32_NEG_INF
    return [
        TensorSpec(
            "scores", list(scores.shape), torch.float32, init_value=lambda: scores
        ),
        TensorSpec(
            "indices",
            list(indices.shape),
            torch.int32,
            init_value=lambda: indices,
        ),
        TensorSpec(
            "valid_counts",
            list(valid_counts.shape),
            torch.int32,
            init_value=lambda: valid_counts,
        ),
        TensorSpec(
            "topk_scores",
            [valid_counts.numel(), IDX_TOPK],
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "topk_indices",
            [valid_counts.numel(), IDX_TOPK],
            torch.int32,
            is_output=True,
        ),
    ]


def golden_leaf_probe(tensors):
    for leaf, valid_count in enumerate(tensors["valid_counts"].tolist()):
        scores, indices = _topk_with_padding(
            tensors["scores"][leaf, :valid_count],
            tensors["indices"][leaf, :valid_count],
        )
        tensors["topk_scores"][leaf] = scores
        tensors["topk_indices"][leaf] = indices


def build_forest_probe_specs():
    import torch
    from golden import TensorSpec

    candidate_counts = torch.tensor(
        [
            0,
            1,
            511,
            512,
            513,
            2048,
            2049,
            4096,
            4097,
            16384,
            127 * TOPK_CANDIDATES_PER_LEAF,
            TOPK_MAX_CANDIDATES,
        ],
        dtype=torch.int32,
    )
    assert candidate_counts.numel() <= PROBE_QUERY_CAP
    leaf_counts = (
        candidate_counts + TOPK_CANDIDATES_PER_LEAF - 1
    ) // TOPK_CANDIDATES_PER_LEAF
    leaf_offsets = torch.zeros_like(leaf_counts)
    if leaf_counts.numel() > 1:
        leaf_offsets[1:] = torch.cumsum(leaf_counts[:-1], dim=0)
    generator = torch.Generator().manual_seed(20260821)
    leaf_scores = torch.randn(
        int(leaf_counts.sum()), TOPK_CANDIDATES_PER_LEAF, generator=generator
    )
    leaf_indices = torch.empty_like(leaf_scores, dtype=torch.int32)
    for query, candidate_count in enumerate(candidate_counts.tolist()):
        if candidate_count == 0:
            continue
        query_leaf_offset = int(leaf_offsets[query])
        query_leaf_count = int(leaf_counts[query])
        for leaf in range(query_leaf_count):
            logical_begin = leaf * TOPK_CANDIDATES_PER_LEAF
            leaf_indices[query_leaf_offset + leaf] = torch.arange(
                logical_begin,
                logical_begin + TOPK_CANDIDATES_PER_LEAF,
                dtype=torch.int32,
            )
        last_leaf = int(leaf_offsets[query] + leaf_counts[query] - 1)
        tail = candidate_count % TOPK_CANDIDATES_PER_LEAF
        if tail != 0:
            leaf_scores[last_leaf, tail:] = FP32_NEG_INF
    return [
        TensorSpec(
            "leaf_scores",
            list(leaf_scores.shape),
            torch.float32,
            init_value=lambda: leaf_scores,
        ),
        TensorSpec(
            "leaf_indices",
            list(leaf_indices.shape),
            torch.int32,
            init_value=lambda: leaf_indices,
        ),
        TensorSpec(
            "leaf_offsets",
            list(leaf_offsets.shape),
            torch.int32,
            init_value=lambda: leaf_offsets,
        ),
        TensorSpec(
            "candidate_counts",
            list(candidate_counts.shape),
            torch.int32,
            init_value=lambda: candidate_counts,
        ),
        TensorSpec(
            "topk_scores",
            [candidate_counts.numel(), IDX_TOPK],
            torch.float32,
            is_output=True,
        ),
        TensorSpec(
            "topk_indices",
            [candidate_counts.numel(), IDX_TOPK],
            torch.int32,
            is_output=True,
        ),
    ]


def golden_forest_probe(tensors):
    for query, candidate_count in enumerate(tensors["candidate_counts"].tolist()):
        leaf_offset = int(tensors["leaf_offsets"][query])
        leaf_count = (
            candidate_count + TOPK_CANDIDATES_PER_LEAF - 1
        ) // TOPK_CANDIDATES_PER_LEAF
        if leaf_count == 0:
            scores = tensors["leaf_scores"].new_empty((0,))
            indices = tensors["leaf_indices"].new_empty((0,))
        else:
            scores = tensors["leaf_scores"][
                leaf_offset : leaf_offset + leaf_count
            ].reshape(-1)[:candidate_count]
            indices = tensors["leaf_indices"][
                leaf_offset : leaf_offset + leaf_count
            ].reshape(-1)[:candidate_count]
        output_scores, output_indices = _topk_with_padding(scores, indices)
        tensors["topk_scores"][query] = output_scores
        tensors["topk_indices"][query] = output_indices


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", choices=["leaf", "forest"], default="leaf")
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.probe == "leaf":
        fn = topk_leaf_probe
        specs = build_leaf_probe_specs()
        golden_fn = golden_leaf_probe
    else:
        fn = topk_forest_probe
        specs = build_forest_probe_specs()
        golden_fn = golden_forest_probe
    result = run_jit(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
