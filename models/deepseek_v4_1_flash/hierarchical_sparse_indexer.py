# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""First-level candidate-block selection for the decoder hierarchical sparse indexer."""

from collections.abc import Callable

import pypto.language as pl
import torch
import torch.nn.functional as F

from config import FLASH as M, CMP_POSITIONS_DYN, T_DYN


from qkv_proj_rope import golden_rope_interleave
from quantization import mxfp8_linear
from rmsnorm import golden_rms_norm_reference


# model config
INDEX_H = M.index_n_heads
INDEX_TOPK = M.index_topk


CANDIDATE_BLOCK_SIZE = M.candidate_block_size
CANDIDATE_TOPK_BLOCKS = M.candidate_topk_blocks
CANDIDATE_PAIR_WIDTH = 2 * CANDIDATE_TOPK_BLOCKS
CANDIDATE_SCORE_TILE = 1024
CANDIDATE_LEAF = 8192
CANDIDATE_SHORT_LEAF = 2048
CANDIDATE_MAX_BLOCKS = M.max_position_embeddings // CANDIDATE_BLOCK_SIZE
CANDIDATE_MAX_LEAVES = (CANDIDATE_MAX_BLOCKS + CANDIDATE_LEAF - 1) // CANDIDATE_LEAF
INDEX_SCORE_POSITIONS_DYN = pl.dynamic("V41_INDEX_SCORE_POSITIONS_DYN")


from attention_ops import prefill_einsum
from attention_ops import prefill_linear
from attention_ops import prefill_matmul
from qkv_proj_rope import prefill_rotate
from quantization import prefill_quantize_dequantize
from quantization import prefill_round_activation


def _prefill_paged_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    request_ids: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    compressed_lens: torch.Tensor,
    wq_b: torch.Tensor,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    candidates: torch.Tensor | None = None,
    topk: int = M.index_topk,
    *,
    precision="fp32",
    formats=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score request-local paged index keys and return scores plus physical rows."""
    if compressed_lens.ndim != 1 or compressed_lens.shape[0] != x.shape[0]:
        raise ValueError("compressed_lens must contain one causal length per query token")
    max_len = int(compressed_lens.max().item()) if compressed_lens.numel() else 0
    scores = x.new_full((x.shape[0], max_len), -torch.inf, dtype=torch.float32)
    physical = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    if max_len == 0:
        return scores, physical

    positions = torch.arange(max_len, device=x.device)
    request_rows = request_ids.to(torch.long).unsqueeze(-1)
    blocks = torch.div(positions, 128, rounding_mode="floor")
    offsets = positions.remainder(128)
    physical_rows = block_table[request_rows, blocks] * 128 + offsets
    flat_keys = index_cache.flatten(0, 1).squeeze(-2)
    keys = flat_keys[physical_rows.to(torch.long)]

    index_dim = index_cache.shape[-1]
    index_heads = weights_proj.shape[-1]
    if precision == "official":
        formats = formats or {}
        q = prefill_linear(qr, wq_b.T, precision=precision, format=formats["index_wq_b"])
        q = q.unflatten(-1, (index_heads, index_dim))
        q = prefill_quantize_dequantize(prefill_rotate(q, cos, sin, precision=precision), "index_fp4")
        weights = prefill_linear(x, weights_proj.T, precision=precision, format=formats["index_weights_proj"])
        weights = prefill_round_activation(weights * (index_dim**-0.5 * index_heads**-0.5), precision)
        dots = prefill_round_activation(prefill_einsum("thd,tkd->thk", q, keys), precision)
        weighted = prefill_round_activation(dots.relu() * weights.unsqueeze(-1), precision)
        scores = prefill_round_activation(weighted.sum(dim=-2), precision)
    else:
        q = prefill_matmul(qr, wq_b).unflatten(-1, (index_heads, index_dim))
        rd = cos.shape[-1] * 2
        q = torch.cat((q[..., :-rd], golden_rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
        weights = prefill_matmul(x.float(), weights_proj.float()).to(x.dtype)
        weights = (weights.float() * (index_dim**-0.5 * index_heads**-0.5)).to(x.dtype)
        dots = prefill_einsum("thd,tkd->thk", q.float(), keys.float()).to(q.dtype)
        weighted = (dots.float().relu() * weights.float().unsqueeze(-1)).to(q.dtype)
        scores = weighted.float().sum(dim=-2).to(q.dtype).float()
    valid_positions = positions.unsqueeze(0) < compressed_lens.to(torch.long).unsqueeze(-1)
    scores = scores.masked_fill(~valid_positions, -torch.inf)
    if candidates is not None:
        scores = scores.masked_fill(~candidates[..., :max_len].to(torch.bool), -torch.inf)

    count = min(topk, max_len)
    logical = scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected_scores = scores.gather(-1, logical)
    selected_rows = physical_rows.gather(-1, logical)
    for token in range(x.shape[0]):
        valid_rows = selected_rows[token, torch.isfinite(selected_scores[token])].to(torch.int32)
        physical[token, : valid_rows.numel()] = valid_rows
    return scores, physical


def golden_index_key(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Project an unrotated compressor latent into a rotated index key."""
    projected = torch.matmul(latent.float(), wk.float()).to(latent.dtype)
    key = golden_rms_norm_reference(projected, norm_weight)
    rd = cos.shape[-1] * 2
    return torch.cat((key[..., :-rd], golden_rope_interleave(key[..., -rd:], cos, sin)), dim=-1)


def golden_paged_indexer(
    x: torch.Tensor,
    qr: torch.Tensor,
    request_ids: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    compressed_lens: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    weights_proj: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    candidates: torch.Tensor | None = None,
    topk: int = M.index_topk,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score request-local paged index keys and return scores plus physical rows."""
    if compressed_lens.ndim != 1 or compressed_lens.shape[0] != x.shape[0]:
        raise ValueError("compressed_lens must contain one causal length per query token")
    max_len = int(compressed_lens.max().item()) if compressed_lens.numel() else 0
    scores = x.new_full((x.shape[0], max_len), -torch.inf, dtype=torch.float32)
    physical = torch.full((x.shape[0], topk), -1, dtype=torch.int32, device=x.device)
    if max_len == 0:
        return scores, physical

    positions = torch.arange(max_len, device=x.device)
    request_rows = request_ids.to(torch.long).unsqueeze(-1)
    blocks = torch.div(positions, 128, rounding_mode="floor")
    offsets = positions.remainder(128)
    physical_rows = block_table[request_rows, blocks] * 128 + offsets
    flat_keys = index_cache.flatten(0, 1).squeeze(-2)
    keys = flat_keys[physical_rows.to(torch.long)]

    index_dim = index_cache.shape[-1]
    index_heads = weights_proj.shape[-1]
    q = mxfp8_linear(qr, wq_b, wq_b_scale).unflatten(-1, (index_heads, index_dim))
    rd = cos.shape[-1] * 2
    q = torch.cat((q[..., :-rd], golden_rope_interleave(q[..., -rd:], cos, sin)), dim=-1)
    weights = torch.matmul(x.float(), weights_proj.float()).to(x.dtype)
    weights = (weights.float() * (index_dim**-0.5 * index_heads**-0.5)).to(x.dtype)
    dots = torch.einsum("thd,tkd->thk", q.float(), keys.float()).to(q.dtype)
    weighted = (dots.float().relu() * weights.float().unsqueeze(-1)).to(q.dtype)
    scores = weighted.float().sum(dim=-2).to(q.dtype).float()
    valid_positions = positions.unsqueeze(0) < compressed_lens.to(torch.long).unsqueeze(-1)
    scores = scores.masked_fill(~valid_positions, -torch.inf)
    if candidates is not None:
        scores = scores.masked_fill(~candidates[..., :max_len].to(torch.bool), -torch.inf)

    count = min(topk, max_len)
    logical = scores.topk(count, dim=-1, sorted=False).indices.sort(dim=-1).values
    selected_scores = scores.gather(-1, logical)
    selected_rows = physical_rows.gather(-1, logical)
    for token in range(x.shape[0]):
        valid_rows = selected_rows[token, torch.isfinite(selected_scores[token])].to(torch.int32)
        physical[token, :valid_rows.numel()] = valid_rows
    return scores, physical


def select_candidate_blocks(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = M.candidate_topk_blocks,
    block_size: int = M.candidate_block_size,
) -> torch.Tensor:
    """Return the first-level candidate mask used by late index-source layers."""
    width = logits.shape[-1]
    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last = (torch.as_tensor(compressed_lens, device=logits.device) - 1) // block_size
    while last.ndim < scores.ndim:
        last = last.unsqueeze(-1)
    block_ids = torch.arange(scores.shape[-1], device=logits.device)
    scores = scores.masked_fill(block_ids == last, torch.inf)
    top = scores.topk(min(topk_blocks, scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def golden_hierarchical_sparse_indexer(
    index_scores: torch.Tensor,
    compressed_lens: torch.Tensor,
) -> torch.Tensor:
    return select_candidate_blocks(
        index_scores, compressed_lens, M.candidate_topk_blocks, M.candidate_block_size
    )


@pl.jit.inline
def _merge_candidate_pairs(
    arena: pl.Tensor,
    left: pl.Scalar[pl.INDEX],
    right: pl.Scalar[pl.INDEX],
    output: pl.Scalar[pl.INDEX],
):
    left_pairs = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
    right_pairs = pl.load(arena, [right, 0], [1, CANDIDATE_PAIR_WIDTH])
    temporary = pl.tile.create([1, 2 * CANDIDATE_PAIR_WIDTH], dtype=pl.FP32)
    merged_all = pl.tile.mrgsort(left_pairs, right_pairs, tmp=temporary)
    merged = pl.tile.slice(merged_all, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(merged, [output, 0], arena)


@pl.jit.inline
def _merge_candidate_level(
    arena: pl.Tensor,
    base: pl.Scalar[pl.INDEX],
    count: pl.Scalar[pl.INDEX],
):
    output_count = (count + 1) // 2
    for output in pl.range(output_count):
        left = base + 2 * output
        right = left + 1
        if right < base + count:
            _merge_candidate_pairs(arena, left, right, base + output)
        else:
            forwarded = pl.load(arena, [left, 0], [1, CANDIDATE_PAIR_WIDTH])
            pl.store(forwarded, [base + output, 0], arena)
    return output_count


@pl.jit.inline
def _sort_candidate_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    block_begin: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(
        scores,
        [token, block_begin],
        [1, CANDIDATE_LEAF],
        valid_shape=[1, valid_count],
    )
    padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    floor = pl.tile.full([1, CANDIDATE_LEAF], dtype=pl.FP32, value=-1e30)
    values = pl.maximum(padded, floor)
    block_begin_i32 = pl.cast(block_begin, pl.INT32)
    indices = pl.add(
        pl.tile.arange(0, [1, CANDIDATE_LEAF], dtype=pl.INT32),
        block_begin_i32,
    )
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    pairs = pl.tile.mrgsort(pairs, block_len=4096)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline
def _sort_candidate_short_leaf(
    scores: pl.Tensor,
    arena: pl.Tensor,
    token: pl.Scalar[pl.INDEX],
    valid_count: pl.Scalar[pl.INDEX],
    output_slot: pl.Scalar[pl.INDEX],
):
    raw = pl.load(
        scores,
        [token, 0],
        [1, CANDIDATE_SHORT_LEAF],
        valid_shape=[1, valid_count],
    )
    values = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
    values = pl.maximum(values, -1e30)
    indices = pl.tile.arange(0, [1, CANDIDATE_SHORT_LEAF], dtype=pl.INT32)
    pairs = pl.tile.sort32(values, pl.reinterpret_view(indices, pl.UINT32))
    pairs = pl.tile.mrgsort(pairs, block_len=64)
    pairs = pl.tile.mrgsort(pairs, block_len=256)
    pairs = pl.tile.mrgsort(pairs, block_len=1024)
    top_pairs = pl.tile.slice(pairs, [1, CANDIDATE_PAIR_WIDTH], [0, 0])
    pl.store(top_pairs, [output_slot, 0], arena)


@pl.jit.inline(auto_scope=False)
def _hierarchical_sparse_indexer(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
):
    """Write the selected-block mask and return its publication TaskId."""
    tokens = pl.tensor.dim(index_scores, 0)
    positions = pl.tensor.dim(candidate_mask, 1)
    block_count = (positions + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    padded_block_count = (block_count + CANDIDATE_LEAF - 1) // CANDIDATE_LEAF * CANDIDATE_LEAF
    mask_tiles = (positions + 127) // 128
    padded_positions = mask_tiles * 128
    block_scores = pl.create_tensor([tokens, padded_block_count], dtype=pl.FP32)
    aligned_mask = pl.create_tensor([tokens, padded_positions], dtype=pl.UINT8)
    score_tiles = (block_count + CANDIDATE_SCORE_TILE - 1) // CANDIDATE_SCORE_TILE
    with pl.spmd(tokens * score_tiles, name_hint="c1a_candidate_block_scores") as score_tid:
        work = pl.tile.get_block_idx()
        token = work // score_tiles
        score_tile = work % score_tiles
        block_begin = score_tile * CANDIDATE_SCORE_TILE
        position_begin = block_begin * CANDIDATE_BLOCK_SIZE
        valid_blocks = pl.min(CANDIDATE_SCORE_TILE, block_count - block_begin)
        valid_positions = pl.min(
            CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE,
            positions - position_begin,
        )
        raw = pl.load(
            index_scores,
            [token, position_begin],
            [1, CANDIDATE_SCORE_TILE * CANDIDATE_BLOCK_SIZE],
            valid_shape=[1, valid_positions],
        )
        padded = pl.tile.fillpad(raw, pad_value=pl.PadValue.min)
        grouped = pl.reshape(padded, [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE])
        reduce_tmp = pl.create_tile(
            [CANDIDATE_SCORE_TILE, CANDIDATE_BLOCK_SIZE],
            dtype=pl.FP32,
        )
        maxima = pl.reshape(pl.row_max(grouped, tmp_tile=reduce_tmp), [1, CANDIDATE_SCORE_TILE])
        visible = pl.read(compressed_lens, [token])
        if visible > 0:
            last_block = (visible - 1) // CANDIDATE_BLOCK_SIZE
            if last_block >= block_begin and last_block < block_begin + valid_blocks:
                pl.tile.write(maxima, [0, last_block - block_begin], 1e30)
        maxima = pl.tile.set_validshape(maxima, 1, valid_blocks)
        pl.store(maxima, [token, block_begin], block_scores)

    with pl.spmd(tokens * mask_tiles, name_hint="c1a_candidate_clear") as clear_tid:
        work = pl.tile.get_block_idx()
        token = work // mask_tiles
        tile = work % mask_tiles
        position = tile * 128
        empty_i32 = pl.tile.full([1, 128], dtype=pl.INT32, value=0)
        empty = pl.cast(empty_i32, pl.UINT8)
        pl.store(empty, [token, position], aligned_mask)

    arena = pl.create_tensor(
        [tokens * CANDIDATE_MAX_LEAVES, CANDIDATE_PAIR_WIDTH],
        dtype=pl.FP32,
    )
    with pl.spmd(tokens, name_hint="c1a_candidate_topk", deps=[score_tid, clear_tid]) as topk_tid:
        token = pl.tile.get_block_idx()
        visible = pl.read(compressed_lens, [token])
        visible_blocks = (visible + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
        arena_base = token * CANDIDATE_MAX_LEAVES
        count = 0
        if visible_blocks > 0 and visible_blocks <= CANDIDATE_SHORT_LEAF:
            _sort_candidate_short_leaf(block_scores, arena, token, visible_blocks, arena_base)
            count = 1
        elif visible_blocks > 0:
            leaf_count = (visible_blocks + CANDIDATE_LEAF - 1) // CANDIDATE_LEAF
            for leaf in pl.range(leaf_count):
                block_begin = leaf * CANDIDATE_LEAF
                valid_count = pl.min(CANDIDATE_LEAF, visible_blocks - block_begin)
                _sort_candidate_leaf(
                    block_scores,
                    arena,
                    token,
                    block_begin,
                    valid_count,
                    arena_base + leaf,
                )
            count = leaf_count
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)
        if count > 1:
            count = _merge_candidate_level(arena, arena_base, count)

        if visible_blocks > 0:
            root = pl.load(arena, [arena_base, 0], [1, CANDIDATE_PAIR_WIDTH])
            selected = pl.tile.gather_mask(
                root,
                mask_pattern=pl.tile.MaskPattern.P1010,
                output_dtype=pl.INT32,
            )
            selected_count = pl.min(visible_blocks, CANDIDATE_TOPK_BLOCKS)
            for selected_lane in pl.range(selected_count):
                selected_block = pl.tile.read(selected, [0, selected_lane])
                position_begin = selected_block * CANDIDATE_BLOCK_SIZE
                for offset in pl.unroll(CANDIDATE_BLOCK_SIZE):
                    position = position_begin + offset
                    if position < positions:
                        # Each token owns a distinct 128-byte-aligned row.
                        pl.write(aligned_mask, [token, position], pl.cast(1, pl.UINT8))

    with pl.spmd(tokens * mask_tiles, name_hint="c1a_candidate_publish", deps=[topk_tid]) as publish_tid:
        work = pl.tile.get_block_idx()
        token = work // mask_tiles
        tile = work % mask_tiles
        position = tile * 128
        valid = pl.min(128, positions - position)
        mask = pl.load(aligned_mask, [token, position], [1, 128])
        mask = pl.tile.set_validshape(mask, 1, valid)
        pl.store(mask, [token, position], candidate_mask)
    return publish_tid


@pl.jit.inline(auto_scope=False)
def hierarchical_sparse_indexer(
    index_scores: pl.Tensor[[T_DYN, INDEX_SCORE_POSITIONS_DYN], pl.FP32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    candidate_mask: pl.Tensor[[T_DYN, CMP_POSITIONS_DYN], pl.UINT8],
):
    """Select candidate blocks from scores with optional trailing row padding."""
    _hierarchical_sparse_indexer(index_scores, compressed_lens, candidate_mask)
    return candidate_mask


__all__ = ["golden_hierarchical_sparse_indexer", "hierarchical_sparse_indexer"]


def run_hierarchical_indexer_golden(golden_fn: Callable[..., torch.Tensor]) -> None:
    width = 2050 * 8
    scores = -torch.arange(width, dtype=torch.float32).reshape(1, width)
    candidate_mask = golden_fn(scores, torch.tensor([width]))
    if not bool(candidate_mask[..., -8:].all()) or int(candidate_mask.sum()) > 2048 * 8:
        raise RuntimeError("candidate selector did not pin the newest partial block")
    print(f"[GOLDEN] PASS {golden_fn.__name__} output={tuple(candidate_mask.shape)}")


if __name__ == "__main__":
    run_hierarchical_indexer_golden(golden_hierarchical_sparse_indexer)
