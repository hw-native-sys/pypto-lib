# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4.1 lightning-indexer scores — paged, variable-length (B=1, H=32, D=128, page=128), bf16; the a2a3-runnable V4.1 DSA path (the official pypto-lib implementation targets a5 only)."""
import pypto.language as pl

DEFAULT_SHAPE = {"M": 64, "N": 1024}


H = 32
D = 128
PAGE = 128
COL_TILE = 128
M = pl.dynamic("M")
N = pl.dynamic("N")
P = pl.dynamic("P")
NB = pl.dynamic("NB")
MH = pl.dynamic("MH")
QUERY_GROUP = 4
PAGE_PARTS = 2
LANE_QUERIES = 2
LANE_COLS = 128

# cross_core_slot keeps its portable form: local_slot_num is A2/A3-only and
# pl.spmd requires a literal optimizations list, so it is intentionally omitted.


@pl.jit
def grouped_score(
    q: pl.Tensor[[M, H * D], pl.BF16],
    cache: pl.Tensor[[NB, PAGE, D], pl.BF16],
    block_table: pl.Tensor[[1, P], pl.INT32],
    lengths: pl.Tensor[[M], pl.INT32],
    w: pl.Tensor[[M, H], pl.FP32],
    dots: pl.InOut[pl.Tensor[[MH, N], pl.FP32]],
    score: pl.InOut[pl.Tensor[[M, N], pl.FP32]],
):
    cache_flat = pl.reshape(cache, [NB * PAGE, D])
    query_flat = pl.reshape(q, [M * H, D])
    for block_idx in pl.spmd(M // QUERY_GROUP * PAGE_PARTS,
            name_hint="persistent_grouped_score",
            optimizations=[pl.cross_core_slot(slot_num=2)]):
        query_base = (block_idx // PAGE_PARTS) * QUERY_GROUP
        pages_per_partition = (P + PAGE_PARTS - 1) // PAGE_PARTS
        page_base = (block_idx % PAGE_PARTS) * pages_per_partition
        query_tile = pl.load(query_flat, [query_base * H, 0], [QUERY_GROUP * H, D])
        for metadata_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            lane_query_base = query_base + metadata_lane * LANE_QUERIES
            grouped_weights = pl.reshape(
                pl.load(w, [lane_query_base, 0], [LANE_QUERIES, H]), [LANE_QUERIES * H, 1])
            negative_inf = pl.reinterpret_view(
                pl.tile.full([1, LANE_COLS], dtype=pl.INT32, value=-8388608), pl.FP32)
            position_offset = pl.cast(
                pl.tile.arange(0, [1, LANE_COLS], dtype=pl.INT32), pl.FP32)
        for page_idx in pl.range(page_base, pl.min(page_base + pages_per_partition, P)):
            physical_page = pl.read(block_table, [0, page_idx])
            cache_row = pl.cast(physical_page, pl.INDEX) * PAGE
            key_tile = pl.load(cache_flat, [cache_row, 0], [COL_TILE, D])
            logical_start = page_idx * PAGE
            grouped_dots = pl.matmul(
                query_tile, pl.tile.transpose_view(key_tile), out_dtype=pl.FP32)
            for transfer_lane in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                sharded_dots = pl.aiv_shard(grouped_dots)
            for vector_lane in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                lane_dots = pl.tile.slice(sharded_dots, [LANE_QUERIES * H, LANE_COLS], [0, 0])
                output_query_base = query_base + vector_lane * LANE_QUERIES
                output_col = logical_start
                pl.store(lane_dots, [output_query_base * H, output_col], dots)
                positions = pl.add(position_offset,
                    pl.cast(pl.cast(output_col, pl.INT32), pl.FP32))
                grouped_activated = pl.maximum(lane_dots, 0.0)
                grouped_weighted = pl.row_expand_mul(grouped_activated, grouped_weights)
                for local_query in pl.range(LANE_QUERIES):
                    query_idx = output_query_base + local_query
                    dot_tile = pl.tile.slice(grouped_weighted,
                        [H, LANE_COLS], [local_query * H, 0])
                    visible = pl.min(pl.max(pl.read(lengths, [query_idx]), 0), N)
                    visible_float = pl.cast(pl.cast(visible, pl.INT32), pl.FP32)
                    reduced = pl.col_sum(dot_tile)
                    masked = pl.tile.select(
                        pl.tile.cmps(positions, visible_float, cmp_type=2), reduced, negative_inf)
                    pl.store(masked, [query_idx, output_col], score)
    return score


def build(shape):
    import torch
    from golden import TensorSpec

    M, N = shape["M"], shape["N"]
    P = N // PAGE
    NB = P + 8

    specs = [
        TensorSpec("q", [M, H * D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("cache", [NB, PAGE, D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("block_table", [1, P], torch.int32,
            init_value=lambda: torch.randperm(NB, dtype=torch.int64)[:P]
                .to(torch.int32).reshape(1, P)),
        TensorSpec("lengths", [M], torch.int32,
            init_value=lambda: torch.randint(1, N + 1, (M,), dtype=torch.int32)),
        TensorSpec("w", [M, H], torch.float32,
            init_value=lambda: torch.randn(M, H) * (D ** -0.5 * H ** -0.5)),
        TensorSpec("dots", [M * H, N], torch.float32),
        TensorSpec("score", [M, N], torch.float32),
    ]

    def golden(tensors):
        physical = (tensors["block_table"][0].long().unsqueeze(1) * PAGE
                    + torch.arange(PAGE).unsqueeze(0)).reshape(-1)
        logical_keys = tensors["cache"].reshape(NB * PAGE, D).float()[physical]
        all_dots = torch.einsum("mhd,nd->mhn",
            tensors["q"].reshape(M, H, D).float(), logical_keys)
        tensors["dots"][:] = all_dots.reshape(M * H, N)
        raw_score = (torch.relu(all_dots) * tensors["w"].unsqueeze(-1)).sum(dim=1)
        visible = tensors["lengths"].long().clamp(0, N)
        invalid = torch.arange(N).unsqueeze(0) >= visible.unsqueeze(1)
        tensors["score"][:] = raw_score.masked_fill(invalid, float("-inf"))

    return grouped_score, specs, golden, 2e-2, 2e-2

if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    fn, specs, golden_fn, rtol, atol = build(DEFAULT_SHAPE)
    result = run(
        fn=fn,
        specs=specs,
        golden_fn=golden_fn,
        config=dict(
            backend_type=args.backend_type,
            execution_mode=args.execution_mode,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=rtol,
        atol=atol,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
