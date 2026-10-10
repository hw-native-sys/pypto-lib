# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""DeepSeek V4.1 DSpark main projection and query metadata."""

from dataclasses import dataclass
import pypto.language as pl
import torch

from models.deepseek_v4_1_flash.attention_ops import make_bf16_projection, make_norm
from models.deepseek_v4_1_flash.config import BLOCK_SIZE, B_DYN, D, FLASH, T_DYN, TABLE_DYN
from models.deepseek_v4_1_flash.golden import rms_norm
from models.deepseek_v4_1_flash.metadata import paged_slots
from models.deepseek_v4_1_flash_dspark.config import (
    DSPARK_DRAFT_LAYERS,
    DSPARK_NOISE_TOKEN_ID,
    DSPARK_QUERY_WIDTH,
    DSPARK_SWA_INDEX_WIDTH,
)


MAIN_IN = DSPARK_DRAFT_LAYERS * D
DSPARK_WINDOW = FLASH.sliding_window
_project = make_bf16_projection(MAIN_IN, D, name_hint="dspark_main_proj")
_normalize = make_norm(D, name_hint="dspark_main_norm")


def golden_main_projection(
    target_hidden: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
) -> torch.Tensor:
    """Combine checkpoint-selected target layers in their declared order."""
    if target_hidden.ndim != 3 or target_hidden.shape[1] != DSPARK_DRAFT_LAYERS:
        raise ValueError("target_hidden must be [tokens, three target layers, hidden]")
    flat = target_hidden.flatten(-2)
    if weight.shape != (flat.shape[-1], target_hidden.shape[-1]):
        raise ValueError("main projection weight must be [3 * hidden, hidden]")
    projected = torch.matmul(flat.float(), weight.float()).to(flat.dtype)
    return rms_norm(projected, norm_weight)


def golden_query_block(
    anchor_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    noise_token_id: int,
    num_speculative_tokens: int = DSPARK_QUERY_WIDTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Anchor-first block: every query row predicts one draft token."""
    if anchor_ids.ndim != 1 or anchor_positions.shape != anchor_ids.shape:
        raise ValueError("anchors and positions must be one row per request")
    if num_speculative_tokens <= 0:
        raise ValueError("num_speculative_tokens must be positive")
    ids = anchor_ids[:, None].expand(-1, num_speculative_tokens).clone()
    ids[:, 1:] = noise_token_id
    positions = anchor_positions[:, None] + torch.arange(
        num_speculative_tokens, device=anchor_positions.device
    )
    return ids, positions


@dataclass(frozen=True)
class DSparkQueryMetadata:
    token_ids: torch.Tensor
    positions: torch.Tensor
    query_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor


def golden_query_metadata(
    anchor_ids: torch.Tensor,
    first_query_positions: torch.Tensor,
    block_tables: torch.Tensor,
    noise_token_id: int,
    window_size: int = FLASH.sliding_window,
) -> DSparkQueryMetadata:
    """Build per-layer paged slots and non-causal visibility for one draft block."""
    if block_tables.ndim != 3 or block_tables.shape[:2] != (DSPARK_DRAFT_LAYERS, anchor_ids.numel()):
        raise ValueError("block tables must be [three draft layers, batch, pages]")
    if first_query_positions.shape != anchor_ids.shape or bool((first_query_positions < 0).any()):
        raise ValueError("first query positions must be nonnegative and match anchors")
    if window_size <= 0:
        raise ValueError("window size must be positive")
    token_ids, positions = golden_query_block(anchor_ids, first_query_positions, noise_token_id)
    batch, width = positions.shape
    index_width = (window_size + width + 63) // 64 * 64
    query_slots = torch.empty(DSPARK_DRAFT_LAYERS, batch, width, dtype=torch.int64, device=positions.device)
    window_indices = torch.full(
        (DSPARK_DRAFT_LAYERS, batch, width, index_width),
        -1,
        dtype=torch.int32,
        device=positions.device,
    )
    window_lens = torch.empty(DSPARK_DRAFT_LAYERS, batch, dtype=torch.int32, device=positions.device)
    for layer in range(DSPARK_DRAFT_LAYERS):
        table = block_tables[layer]
        for request in range(batch):
            prefix_end = int(first_query_positions[request])
            context_positions = torch.arange(
                max(0, prefix_end - window_size), prefix_end, device=positions.device
            )
            visible_positions = torch.cat((context_positions, positions[request]))
            request_ids = torch.full_like(visible_positions, request)
            visible_slots = paged_slots(visible_positions, request_ids, table)
            window_lens[layer, request] = visible_slots.numel()
            window_indices[layer, request, :, : visible_slots.numel()] = visible_slots.to(torch.int32)
            query_slots[layer, request] = visible_slots[-width:]
    return DSparkQueryMetadata(token_ids, positions, query_slots, window_indices, window_lens)


@pl.jit.inline(auto_scope=False)
def dspark_proj(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    main_x: pl.Tensor[[T_DYN, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project the ordered target taps and normalize only active token rows."""
    tokens = pl.tensor.dim(target_hidden, 0)
    projected = pl.create_tensor([tokens, D], dtype=pl.BF16)
    _project(target_hidden, main_proj_weight, projected, num_tokens)
    _normalize(projected, main_norm_weight, main_x, num_tokens)
    return main_x


@pl.jit
def dspark_proj_test(
    target_hidden: pl.Tensor[[T_DYN, MAIN_IN], pl.BF16],
    main_proj_weight: pl.Tensor[[MAIN_IN, D], pl.BF16],
    main_norm_weight: pl.Tensor[[D], pl.BF16],
    main_x: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    target_hidden.bind_dynamic(0, T_DYN)
    main_x.bind_dynamic(0, T_DYN)
    return dspark_proj(target_hidden, main_proj_weight, main_norm_weight, main_x, num_tokens)


def run_projection_case(*, platform: str = "a5sim", device_id: int = 0, compile_only: bool = False):
    """Compile or validate the BF16 projection with six active draft rows."""
    import torch

    from golden import ScalarSpec, TensorSpec, run

    torch.manual_seed(41)
    tokens = 6
    hidden = torch.randn(tokens, MAIN_IN, dtype=torch.bfloat16) * 0.1
    weight = (torch.randn(MAIN_IN, D) * 0.01).to(torch.bfloat16)

    def golden(values):
        values["main_x"].copy_(
            golden_main_projection(
                values["target_hidden"].reshape(tokens, DSPARK_DRAFT_LAYERS, D),
                values["main_proj_weight"],
                values["main_norm_weight"],
            )
        )

    specs = [
        TensorSpec("target_hidden", [tokens, MAIN_IN], torch.bfloat16, init_value=hidden),
        TensorSpec("main_proj_weight", [MAIN_IN, D], torch.bfloat16, init_value=weight),
        TensorSpec("main_norm_weight", [D], torch.bfloat16, init_value=1.0),
        TensorSpec("main_x", [tokens, D], torch.bfloat16),
        ScalarSpec("num_tokens", torch.int32, tokens, compile_runtime=True),
    ]
    return run(
        fn=dspark_proj_test,
        specs=specs,
        golden_fn=golden,
        config={"platform": platform, "device_id": device_id},
        rtol=0.05,
        atol=0.05,
        compile_only=compile_only,
    )


@pl.jit.inline(auto_scope=False)
def dspark_metadata(
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    query_ids: pl.Tensor[[T_DYN], pl.INT64],
    query_positions: pl.Tensor[[T_DYN], pl.INT32],
    query_slots: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32],
    window_lens: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN], pl.INT32],
):
    batch = pl.tensor.dim(anchor_ids, 0)
    for worker in pl.spmd(1, name_hint="dspark_query_metadata"):
        for request_offset in pl.range(batch):
            request = worker + request_offset
            first = pl.read(first_query_positions, [request])
            anchor = pl.read(anchor_ids, [request])
            prefix_start = pl.max(first - DSPARK_WINDOW, 0)
            context_len = first - prefix_start
            visible_len = context_len + DSPARK_QUERY_WIDTH
            for offset in pl.range(DSPARK_QUERY_WIDTH):
                token = request * DSPARK_QUERY_WIDTH + offset
                token_id = pl.cast(DSPARK_NOISE_TOKEN_ID, pl.INT64)
                if offset == 0:
                    token_id = anchor
                pl.write(query_ids, [token], token_id)
                pl.write(query_positions, [token], pl.cast(first + offset, pl.INT32))
            for layer in pl.range(DSPARK_DRAFT_LAYERS):
                pl.write(window_lens, [layer, request], pl.cast(visible_len, pl.INT32))
                for offset in pl.range(DSPARK_SWA_INDEX_WIDTH):
                    slot = pl.cast(-1, pl.INT32)
                    if offset < visible_len:
                        position = prefix_start + offset
                        logical_block = position // BLOCK_SIZE
                        physical_block = pl.read(block_tables, [layer, request, pl.cast(logical_block, pl.INDEX)])
                        slot = pl.cast(physical_block * BLOCK_SIZE + position % BLOCK_SIZE, pl.INT32)
                    for query in pl.range(DSPARK_QUERY_WIDTH):
                        token = request * DSPARK_QUERY_WIDTH + query
                        pl.write(window_indices, [layer, token, offset], slot)
                        if offset == context_len + query:
                            pl.write(query_slots, [layer, token], pl.cast(slot, pl.INT64))
    return query_ids, query_positions, query_slots, window_indices, window_lens


@pl.jit
def dspark_metadata_test(
    anchor_ids: pl.Tensor[[B_DYN], pl.INT64],
    first_query_positions: pl.Tensor[[B_DYN], pl.INT32],
    block_tables: pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN, TABLE_DYN], pl.INT32],
    query_ids: pl.Out[pl.Tensor[[T_DYN], pl.INT64]],
    query_positions: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
    query_slots: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN], pl.INT64]],
    window_indices: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, T_DYN, DSPARK_SWA_INDEX_WIDTH], pl.INT32]],
    window_lens: pl.Out[pl.Tensor[[DSPARK_DRAFT_LAYERS, B_DYN], pl.INT32]],
):
    anchor_ids.bind_dynamic(0, B_DYN)
    first_query_positions.bind_dynamic(0, B_DYN)
    block_tables.bind_dynamic(1, B_DYN)
    block_tables.bind_dynamic(2, TABLE_DYN)
    query_ids.bind_dynamic(0, T_DYN)
    query_positions.bind_dynamic(0, T_DYN)
    query_slots.bind_dynamic(1, T_DYN)
    window_indices.bind_dynamic(1, T_DYN)
    window_lens.bind_dynamic(1, B_DYN)
    return dspark_metadata(
        anchor_ids,
        first_query_positions,
        block_tables,
        query_ids,
        query_positions,
        query_slots,
        window_indices,
        window_lens,
    )


def run_metadata_case(*, platform: str = "a5sim", device_id: int = 0, compile_only: bool = False):
    """Compare all published metadata against the paged Torch reference."""
    import torch

    from golden import TensorSpec, run

    batch = 1
    tokens = batch * DSPARK_QUERY_WIDTH
    anchors = torch.tensor([7], dtype=torch.int64)
    positions = torch.tensor([125], dtype=torch.int32)
    tables = torch.tensor(
        [[[4, 5]], [[14, 15]], [[24, 25]]],
        dtype=torch.int32,
    )

    def golden(values):
        expected = golden_query_metadata(
            values["anchor_ids"],
            values["first_query_positions"],
            values["block_tables"],
            DSPARK_NOISE_TOKEN_ID,
        )
        values["query_ids"].copy_(expected.token_ids.reshape(-1))
        values["query_positions"].copy_(expected.positions.reshape(-1))
        values["query_slots"].copy_(expected.query_slots.reshape(DSPARK_DRAFT_LAYERS, tokens))
        values["window_indices"].copy_(
            expected.window_indices.reshape(DSPARK_DRAFT_LAYERS, tokens, DSPARK_SWA_INDEX_WIDTH)
        )
        values["window_lens"].copy_(expected.window_lens)

    specs = [
        TensorSpec("anchor_ids", [batch], torch.int64, init_value=anchors),
        TensorSpec("first_query_positions", [batch], torch.int32, init_value=positions),
        TensorSpec("block_tables", [DSPARK_DRAFT_LAYERS, batch, 2], torch.int32, init_value=tables),
        TensorSpec("query_ids", [tokens], torch.int64),
        TensorSpec("query_positions", [tokens], torch.int32),
        TensorSpec("query_slots", [DSPARK_DRAFT_LAYERS, tokens], torch.int64),
        TensorSpec("window_indices", [DSPARK_DRAFT_LAYERS, tokens, DSPARK_SWA_INDEX_WIDTH], torch.int32),
        TensorSpec("window_lens", [DSPARK_DRAFT_LAYERS, batch], torch.int32),
    ]
    return run(
        fn=dspark_metadata_test,
        specs=specs,
        golden_fn=golden,
        config={"platform": platform, "device_id": device_id},
        compile_only=compile_only,
    )


if __name__ == "__" + "main__":
    import argparse

    parser = argparse.ArgumentParser(description="V4.1 DSpark drafter validation")
    parser.add_argument("--mode", choices=("projection", "metadata"), required=True)
    parser.add_argument("-p", "--platform", choices=("a5", "a5sim"), default="a5sim")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    run_case = run_projection_case if args.mode == "projection" else run_metadata_case
    result = run_case(platform=args.platform, device_id=args.device, compile_only=args.compile_only)
    print(result)
    if not result.passed:
        raise SystemExit(1)
