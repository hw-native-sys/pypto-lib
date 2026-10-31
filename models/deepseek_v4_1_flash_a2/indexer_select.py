"""A2/A3 INT8 QLI scoring and Top-K selection.

The A2/A3 DeepSeek V4.1 indexer does not have the A5 FP8/MX path.  Its
selection ABI is nevertheless simple and stable: quantized per-head query
rows are compared with a shared INT8 index-key cache, the two FP16 scales are
applied in FP32, and the positive head scores are multiplied by the FP16
per-head weights.  This module implements that boundary directly in PyPTO.

``indices`` intentionally keeps the score order returned by QLI.  vLLM's
``prepare_indexer_indices`` performs the later visibility filter and
chronological sort; keeping that step separate makes the score/index boundary
independently checkable on A2/A3 and lets the sparse-attention consumer use
either ordering (attention is permutation invariant).
"""

from __future__ import annotations

import torch
import pypto.language as pl


INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
KEY_CAPACITY = 512
KEY_TILE = 32
HEAD_TILE = 16
# A2/A3 sort32 is most reliable at this width.  The public kernel signature
# currently fixes the candidate row to this size.
SORT_LEAF = 512
NEGATIVE_INF = -3.0e38


def indexer_select_reference(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
) -> dict[str, torch.Tensor]:
    """CPU oracle for the A2/A3 QLI score and score-order Top-K contract.

    ``query`` is ``[T, H, 128]`` INT8, ``key`` is ``[K, 128]`` INT8, and the
    two scale tensors are FP16.  ``positions`` is the raw token position used
    by vLLM to derive the visible compressed-key prefix.
    """

    if query.ndim != 3 or tuple(query.shape[1:]) != (INDEX_HEADS, INDEX_HEAD_DIM):
        raise ValueError("query must have shape [T, 32, 128]")
    if key.ndim != 2 or key.shape[-1] != INDEX_HEAD_DIM:
        raise ValueError("key must have shape [K, 128]")
    tokens, heads, _ = query.shape
    key_rows = key.shape[0]
    if query_scale.shape != (tokens, heads) or weights.shape != (tokens, heads):
        raise ValueError("query_scale and weights must have shape [T, 32]")
    if key_scale.numel() != key_rows or positions.shape != (tokens,):
        raise ValueError("key_scale must be [K] and positions must be [T]")
    if not 1 <= topk <= key_rows:
        raise ValueError("topk must be in [1, key.shape[0]]")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    dots = torch.einsum("thd,kd->thk", query.to(torch.int32), key.to(torch.int32)).float()
    dots.mul_(query_scale.float().unsqueeze(-1))
    dots.mul_(key_scale.float().reshape(1, 1, key_rows))
    scores = torch.relu(dots)
    scores.mul_(weights.float().unsqueeze(-1))
    scores = scores.sum(dim=1)

    indices = torch.full((tokens, topk), -1, dtype=torch.int32)
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=key_rows)
    for token in range(tokens):
        count = min(int(visible[token]), topk)
        if count:
            # QLI returns score order.  ``sorted=False`` matches the native
            # operator's contract; ties are not expected for random fixtures.
            picked = torch.topk(scores[token, : int(visible[token])], count, sorted=False).indices
            indices[token, :count] = picked.to(torch.int32)
    return {"scores": scores, "indices": indices}


def make_indexer_select(
    capacity: int,
    key_capacity: int,
    *,
    heads: int = INDEX_HEADS,
    head_dim: int = INDEX_HEAD_DIM,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 2,
):
    """Build a fixed-bucket PyPTO QLI score/Top-K kernel."""

    if capacity <= 0 or capacity % HEAD_TILE:
        raise ValueError("capacity must be a positive multiple of 16")
    if heads != INDEX_HEADS or head_dim != INDEX_HEAD_DIM:
        raise ValueError("A2/A3 QLI uses 32 heads x 128")
    # The native A2/A3 QLI handoff allocates one 512-entry candidate row.
    # Smaller visible prefixes are represented by score padding and are
    # filtered by ``prepare_indexer_indices`` in the next boundary.  The
    # fixed signature is deliberate: larger or variable rows are not stable
    # on the current A2/A3 sort32/mrgsort backend.
    if key_capacity != KEY_CAPACITY:
        raise ValueError(f"A2/A3 QLI uses fixed key_capacity={KEY_CAPACITY}")
    if topk != INDEX_TOPK:
        raise ValueError(f"A2/A3 QLI uses fixed topk={INDEX_TOPK}")
    if compress_ratio not in (1, 2):
        raise ValueError("compress_ratio must be 1 or 2")

    @pl.jit.incore
    def select_one(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        running_pairs: pl.Tensor[[capacity, 2 * topk], pl.FP32],
        scores: pl.Tensor[[capacity, key_capacity], pl.FP32],
        indices: pl.Tensor[[capacity, topk], pl.INT32],
    ) -> None:
        token = pl.tile.get_block_idx()
        # The A2/A3 QLI cache ABI is one fixed 512-entry candidate row.  Sort
        # the complete row here; ``prepare_indexer_indices`` applies the
        # request-specific visible-prefix filter before cache gather.  Keeping
        # this boundary fixed avoids dynamic tile tail writes on A2/A3 while
        # preserving the final selected set.
        visible = key_capacity
        # QLI writes a score tile for every visible key.  Invalid rows never
        # enter the sort, and score padding stays at the hardware minimum.
        # Materialize that floor explicitly: dynamic ``valid_shape`` on a
        # sort32 input is not reliable on the A2/A3 backend for partial rows.
        pl.store(
            pl.tile.full([1, key_capacity], dtype=pl.FP32, value=NEGATIVE_INF),
            [token, 0],
            scores,
        )
        for k0 in pl.range(0, visible, KEY_TILE):
            valid = pl.min(KEY_TILE, visible - k0)
            key_tile = key[k0:k0 + KEY_TILE, :]
            scale_tile = pl.reshape(pl.cast(key_scale[k0:k0 + KEY_TILE], pl.FP32), [1, KEY_TILE])
            total = pl.full([1, KEY_TILE], dtype=pl.FP32, value=0.0)
            for h0 in pl.range(0, heads, HEAD_TILE):
                q_tile = pl.reshape(query[token:token + 1, h0:h0 + HEAD_TILE, :], [HEAD_TILE, head_dim])
                dots = pl.matmul(q_tile, key_tile, b_trans=True, out_dtype=pl.INT32)
                dots = pl.cast(dots, pl.FP32, mode="none")
                q_scale_tile = pl.reshape(
                    pl.cast(query_scale[token, h0:h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                weight_tile = pl.reshape(
                    pl.cast(weights[token, h0:h0 + HEAD_TILE], pl.FP32), [HEAD_TILE, 1]
                )
                dots = pl.row_expand_mul(dots, q_scale_tile)
                dots = pl.col_expand_mul(dots, scale_tile)
                dots = pl.maximum(dots, 0.0)
                dots = pl.row_expand_mul(dots, weight_tile)
                total = pl.add(total, pl.reshape(pl.col_sum(dots), [1, KEY_TILE]))
            scores[token:token + 1, k0:k0 + KEY_TILE] = pl.fillpad(
                pl.set_validshape(total, 1, valid), pad_value=pl.PadValue.min
            )

        # Match QLI's score-order result with a bounded merge sort.  The
        # payload carries absolute key positions, so no host-side offset is
        # needed for the fixed flattened cache used by the A2/A3 path.
        # The A2/A3 ABI uses one 512-entry QLI candidate row.  Always execute
        # the fixed sort once and represent a short visible prefix with the
        # explicit score floor above; a dynamic ``pl.range(leaf_count)`` leaves
        # the pair arena uninitialized for partial prefixes on A2/A3.
        leaf0 = 0
        row_raw = pl.load(scores, [token, leaf0], [1, SORT_LEAF])
        row = pl.tile.fillpad(row_raw, pad_value=pl.PadValue.min)
        row = pl.maximum(row, pl.tile.full([1, SORT_LEAF], dtype=pl.FP32, value=NEGATIVE_INF))
        ramp = pl.tile.arange(0, [1, SORT_LEAF], dtype=pl.INT32)
        leaf_indices = pl.tile.add(ramp, pl.cast(leaf0, pl.INT32))
        sort_tmp = pl.tile.create([1, 2 * SORT_LEAF], dtype=pl.FP32)
        leaf_pairs = pl.tile.sort32(
            row, pl.reinterpret_view(leaf_indices, pl.UINT32), tmp=sort_tmp
        )
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=64)
        leaf_pairs = pl.tile.mrgsort(leaf_pairs, block_len=256)
        leaf_pairs = pl.tile.slice(leaf_pairs, [1, 2 * topk], [0, 0])
        pl.store(leaf_pairs, [token, 0], running_pairs)

        # Materialize the complete padding row in one UB tile.  Explicitly
        # initialize every lane because the backend may preserve stale pair
        # lanes when only a dynamic prefix is written.
        selected = pl.tile.full([1, topk], dtype=pl.INT32, value=-1)
        if visible > 0:
            pairs = pl.load(running_pairs, [token, 0], [1, 2 * topk])
            picked = pl.tile.gather_mask(pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
            selected = picked
        pl.store(selected, [token, 0], indices)

    @pl.jit
    def kernel(
        query: pl.Tensor[[capacity, heads, head_dim], pl.INT8],
        query_scale: pl.Tensor[[capacity, heads], pl.FP16],
        key: pl.Tensor[[key_capacity, head_dim], pl.INT8],
        key_scale: pl.Tensor[[key_capacity], pl.FP16],
        weights: pl.Tensor[[capacity, heads], pl.FP16],
        positions: pl.Tensor[[capacity], pl.INT32],
        scores: pl.Out[pl.Tensor[[capacity, key_capacity], pl.FP32]],
        indices: pl.Out[pl.Tensor[[capacity, topk], pl.INT32]],
        active: pl.Scalar[pl.INT32],
    ):
        running_pairs = pl.create_tensor([capacity, 2 * topk], dtype=pl.FP32)
        with pl.spmd(active, name_hint="dsv41_a2a3_qli_select"):
            select_one(
                query, query_scale, key, key_scale, weights, positions,
                running_pairs, scores, indices,
            )
        return scores, indices

    return kernel


__all__ = ["INDEX_HEADS", "INDEX_HEAD_DIM", "INDEX_TOPK", "indexer_select_reference", "make_indexer_select"]
