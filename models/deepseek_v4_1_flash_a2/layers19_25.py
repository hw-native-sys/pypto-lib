# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A2/A3 attention stages for DeepSeek V4.1 layers 19, 20, 21, 24 and 25.

The A5 decoder has five distinct schedules in this range:

* L19 is a ratio-2 cache consumer (C2A reuse) of the L14 source;
* L20 is the ratio-1 source (C1A full);
* L21 and L25 reuse the ratio-1 cache;
* L24 refreshes only the index selection (C1A reindex).

This module keeps those ownership rules explicit and uses the A2/A3 ABI at
every boundary: INT8 W8A8 projections with FP32 scales, BF16 cache/value
rows, INT8 index keys with FP16 scales, and BF16 output projections.  The
reference functions are intentionally stage shaped so a vLLM capture can be
compared before composing a complete decoder block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from .operator_config import A2A3_DTYPE_CONTRACT
from .candidate_pool import make_candidate_block_pool
from .layer14_15 import index_key_reference, make_index_key_publish_rope
from .sparse_attention import make_sparse_attention, sparse_attention_reference
from .indexer_prepare import (
    make_indexer_prepare,
    prepare_indexer_indices_reference,
)
from .indexer_query import (
    indexer_query_reference,
    make_l3_indexer_query,
)
from .indexer_select import (
    indexer_select_reference,
    make_indexer_select,
)
from .norm_rope import make_rope, make_rmsnorm, rms_norm_reference, rope_reference
from .output_proj import (
    make_output_projection_bf16,
    output_projection_bf16_reference,
)
from .qkv import make_qkv_w8a8, qkv_reference
from .staged import make_bf16_linear


HIDDEN = 5120
HEADS = 64
HEAD_DIM = 512
ROPE_DIM = 64
Q_LORA = 1280
INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_WIDTH = INDEX_HEADS * INDEX_DIM
INDEX_TOPK = 512
CANDIDATE_TOPK_BLOCKS = 2048
CANDIDATE_BLOCK_SIZE = 8
GROUPS = 8
OUTPUT_RANK = 1024
WINDOW = 128
EPS = 1.0e-20


@dataclass(frozen=True)
class DecoderLayerA2A3Contract:
    """Cache ownership and dtype contract for one migrated layer."""

    layer_id: int
    compression_ratio: int
    mode: str
    kv_source_layer_id: int
    index_source_layer_id: int
    has_compressor: bool
    has_index_key_writer: bool
    dtype_contract: Mapping[str, str] = field(
        default_factory=lambda: dict(A2A3_DTYPE_CONTRACT)
    )


LAYER19 = DecoderLayerA2A3Contract(19, 2, "c2a_reuse", 14, 14, False, False)
LAYER20 = DecoderLayerA2A3Contract(20, 1, "c1a_full", 20, 20, True, True)
LAYER21 = DecoderLayerA2A3Contract(21, 1, "c1a_reuse", 20, 20, False, False)
LAYER24 = DecoderLayerA2A3Contract(24, 1, "c1a_reindex", 20, 24, False, False)
LAYER25 = DecoderLayerA2A3Contract(25, 1, "c1a_reuse", 20, 24, False, False)

TARGET_LAYERS = (LAYER19, LAYER20, LAYER21, LAYER24, LAYER25)
_CONTRACT_BY_LAYER = {item.layer_id: item for item in TARGET_LAYERS}


def layer_contract(layer_id: int) -> DecoderLayerA2A3Contract:
    """Return the migrated contract and reject accidental schedule drift."""

    try:
        return _CONTRACT_BY_LAYER[int(layer_id)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("supported migrated layers are 19, 20, 21, 24 and 25") from error


def validate_layer19_20_21_24_25_contracts() -> tuple[DecoderLayerA2A3Contract, ...]:
    """Validate the five layer schedule and the shared A2/A3 dtype ABI."""

    for contract in TARGET_LAYERS:
        if contract.compression_ratio not in (1, 2):
            raise AssertionError(f"unsupported compression ratio on L{contract.layer_id}")
        if contract.mode.startswith("c2a") and contract.compression_ratio != 2:
            raise AssertionError("C2A layers must use ratio 2")
        if contract.mode.startswith("c1a") and contract.compression_ratio != 1:
            raise AssertionError("C1A layers must use ratio 1")
        if contract.dtype_contract["hidden"] != "bfloat16":
            raise AssertionError("hidden state must remain BF16 on A2/A3")
        if contract.dtype_contract["index_cache"] != "int8":
            raise AssertionError("index cache must remain INT8 on A2/A3")
        if contract.dtype_contract["index_cache_scale"] != "float16":
            raise AssertionError("index cache scale must remain FP16 on A2/A3")
        if contract.dtype_contract["compressed_kv_cache"] != "bfloat16":
            raise AssertionError("compressed KV cache must remain BF16 on A2/A3")
    if LAYER24.index_source_layer_id != 24 or LAYER25.index_source_layer_id != 24:
        raise AssertionError("L24 must publish the index source reused by L25")
    return TARGET_LAYERS


def _canonical_weight(weight: torch.Tensor, output_width: int, input_width: int) -> torch.Tensor:
    """Normalize a checkpoint matrix to the A2/A3 ``[K, N]`` layout."""

    if weight.ndim != 2:
        raise ValueError(f"linear weight must be rank 2, got {tuple(weight.shape)}")
    if tuple(weight.shape) == (input_width, output_width):
        return weight
    if tuple(weight.shape) == (output_width, input_width):
        return weight.transpose(0, 1).contiguous()
    raise ValueError(
        f"weight shape {tuple(weight.shape)} does not match [{input_width}, {output_width}]"
    )


def _qkv_weights(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept either loader output or the compact canonical test mapping."""

    result = dict(weights)
    for name in ("wq_a", "wq_b", "wkv"):
        result.setdefault(name, result.get(f"{name}_canonical"))
        if result.get(name) is None:
            raise KeyError(f"missing {name} or {name}_canonical")
    result.setdefault("q_norm_weight", result.get("q_norm"))
    result.setdefault("kv_norm_weight", result.get("kv_norm"))
    if result.get("q_norm_weight") is None or result.get("kv_norm_weight") is None:
        raise KeyError("missing q_norm/kv_norm weights")
    return result


def _ratio1_compressor_reference(
    value: torch.Tensor,
    weight: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    eps: float = EPS,
) -> torch.Tensor:
    """Project and normalize the C1A latent in the vLLM BF16 order."""

    if value.ndim != 2 or value.shape[-1] != HIDDEN:
        raise ValueError("compressor input must have shape [T, 5120]")
    canonical = _canonical_weight(weight, HEAD_DIM, HIDDEN)
    projected = torch.matmul(value.float(), canonical.float()).to(torch.bfloat16)
    return rms_norm_reference(projected, norm_weight, eps=eps)


def _index_weights(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute the FP16 per-head index weights consumed by A2/A3 QLI."""

    canonical = _canonical_weight(weight, INDEX_HEADS, HIDDEN)
    projected = torch.matmul(value.float(), canonical.float()).to(torch.bfloat16)
    return (projected.float() * (INDEX_WIDTH ** -0.5)).to(torch.float16)


def _pad_index_cache(payload: torch.Tensor, scale: torch.Tensor, capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    if payload.ndim != 2 or payload.shape[-1] != INDEX_DIM:
        raise ValueError("index payload must have shape [rows, 128]")
    if scale.numel() != payload.shape[0]:
        raise ValueError("index scale must have one value per key row")
    if capacity < payload.shape[0]:
        raise ValueError("index cache capacity is smaller than the published rows")
    if capacity == payload.shape[0]:
        return payload, scale.reshape(-1)
    pad = capacity - payload.shape[0]
    return (
        torch.cat((payload, torch.zeros(pad, INDEX_DIM, dtype=payload.dtype, device=payload.device))),
        torch.cat((scale.reshape(-1), torch.ones(pad, dtype=scale.dtype, device=scale.device))),
    )


def select_candidate_block_pool_reference(
    scores: torch.Tensor,
    compressed_lengths: torch.Tensor | int,
    *,
    topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    block_size: int = CANDIDATE_BLOCK_SIZE,
    num_tokens: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build the vLLM C1A candidate pool from dense QLI scores."""

    if scores.ndim != 2:
        raise ValueError("scores must have shape [T, compressed_rows]")
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("candidate pool dimensions must be positive")
    tokens, width = scores.shape
    lengths = torch.as_tensor(compressed_lengths, device=scores.device, dtype=torch.int64)
    if lengths.ndim == 0:
        lengths = lengths.expand(tokens)
    if lengths.shape != (tokens,):
        raise ValueError("compressed_lengths must provide one length per score row")
    active = tokens if num_tokens is None else max(0, min(tokens, int(num_tokens)))
    blocks = max(1, (width + block_size - 1) // block_size)
    capacity = min(topk_blocks, max(32, ((blocks + 31) // 32) * 32))
    ids = torch.full((tokens, capacity), -1, dtype=torch.int32, device=scores.device)
    counts = torch.zeros((tokens,), dtype=torch.int32, device=scores.device)
    if width == 0:
        return {"pool_block_ids": ids, "pool_block_counts": counts}
    padded_width = ((width + block_size - 1) // block_size) * block_size
    if padded_width != width:
        padded = torch.full(
            (tokens, padded_width), -torch.inf, dtype=scores.dtype, device=scores.device
        )
        padded[:, :width] = scores
    else:
        padded = scores
    block_scores = padded.reshape(tokens, -1, block_size).amax(dim=-1)
    for token in range(active):
        visible = max(0, min(int(lengths[token]), width))
        visible_blocks = (visible + block_size - 1) // block_size
        if visible_blocks == 0:
            continue
        count = min(visible_blocks, topk_blocks, capacity)
        ranked = block_scores[token, :visible_blocks].clone().clamp_min(-1.0e30)
        ranked[visible_blocks - 1] = 1.0e30
        chosen = torch.argsort(ranked, descending=True, stable=True)[:count]
        ids[token, :count] = chosen.sort().values.to(torch.int32)
        counts[token] = count
    return {"pool_block_ids": ids, "pool_block_counts": counts}


def _select_index_reference(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    key: torch.Tensor,
    key_scale: torch.Tensor,
    weights: torch.Tensor,
    positions: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    compress_ratio: int = 1,
    candidate_blocks: torch.Tensor | None = None,
    candidate_block_counts: torch.Tensor | None = None,
    candidate_block_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Score QLI rows, optionally restricting L24 to candidate blocks."""

    if candidate_blocks is None:
        selected = indexer_select_reference(
            query, query_scale, key, key_scale, weights, positions,
            topk=topk, compress_ratio=compress_ratio,
        )
        return prepare_indexer_indices_reference(
            selected["indices"], positions, compress_ratio=compress_ratio
        ) | {"scores": selected["scores"], "raw_indices": selected["indices"]}

    if candidate_blocks.ndim == 3:
        if candidate_blocks.shape[1] != 1:
            raise ValueError("candidate_blocks rank-3 layout must have shape [T, 1, blocks]")
        candidate_blocks = candidate_blocks[:, 0, :]
    if candidate_blocks.ndim != 2 or candidate_blocks.shape[0] != query.shape[0]:
        raise ValueError("candidate_blocks must have shape [T, blocks] or [T, 1, blocks]")
    if candidate_block_counts is None:
        candidate_block_counts = candidate_blocks.ge(0).sum(dim=-1).to(torch.int32)
    elif candidate_block_counts.ndim == 2:
        if candidate_block_counts.shape[1] != 1:
            raise ValueError("candidate_block_counts rank-2 layout must have shape [T, 1]")
        candidate_block_counts = candidate_block_counts[:, 0]
    if candidate_block_counts.shape != (query.shape[0],):
        raise ValueError("candidate_block_counts must have shape [T] or [T, 1]")
    dots = torch.einsum("thd,kd->thk", query.to(torch.int32), key.to(torch.int32)).float()
    dots.mul_(query_scale.float().unsqueeze(-1))
    dots.mul_(key_scale.float().reshape(1, 1, -1))
    scores = torch.relu(dots).mul_(weights.float().unsqueeze(-1)).sum(dim=1)
    visible = ((positions.to(torch.int64) + 1) // compress_ratio).clamp(min=0, max=key.shape[0])
    allowed = torch.zeros_like(scores, dtype=torch.bool)
    for token in range(query.shape[0]):
        block_count = max(0, min(int(candidate_block_counts[token]), candidate_blocks.shape[1]))
        blocks = candidate_blocks[token, :block_count]
        blocks = blocks[blocks >= 0].to(torch.int64)
        if blocks.numel():
            starts = blocks[:, None] * candidate_block_size
            cols = torch.arange(candidate_block_size, device=key.device)[None, :]
            rows = (starts + cols).reshape(-1)
            rows = rows[(rows < visible[token]) & (rows < key.shape[0])]
            allowed[token, rows] = True
    scores = scores.masked_fill(~allowed, -torch.inf)
    indices = torch.full((query.shape[0], topk), -1, dtype=torch.int32, device=query.device)
    for token in range(query.shape[0]):
        count = min(int(visible[token]), topk, int(allowed[token].sum()))
        if count <= 0:
            continue
        candidates = torch.where(allowed[token])[0]
        order = torch.argsort(scores[token, candidates], descending=True, stable=True)[:count]
        indices[token, :count] = candidates[order].to(torch.int32)
    prepared = prepare_indexer_indices_reference(indices, positions, compress_ratio=compress_ratio)
    return prepared | {"scores": scores, "raw_indices": indices}


def c2a_reuse_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cosine: torch.Tensor,
    sine: torch.Tensor,
    window_rows: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_rows: torch.Tensor,
    compressed_indices: torch.Tensor,
    sink: torch.Tensor,
    *,
    active: int | None = None,
) -> dict[str, torch.Tensor]:
    """Run L19's QKV, BF16 cache gather and C2A sparse attention."""

    qkv = qkv_reference(hidden, _qkv_weights(weights), cosine, sine, _QKV_CONFIG)
    attended = sparse_attention_reference(
        qkv["query"].reshape(hidden.shape[0], HEADS, HEAD_DIM),
        window_rows,
        window_indices,
        compressed_rows,
        compressed_indices,
        sink,
        active=active,
    )
    projected = output_projection_bf16_reference(
        attended.reshape(hidden.shape[0], HEADS * HEAD_DIM),
        weights["wo_a"], weights["wo_b"], cosine, sine,
        HEADS, HEAD_DIM, ROPE_DIM,
    )
    return {
        **qkv,
        "compressed_indices": compressed_indices,
        "attended": attended,
        **{f"output_{key}": value for key, value in projected.items()},
    }


def c1a_full_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    position_ids: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    compressed_cosine: torch.Tensor,
    compressed_sine: torch.Tensor,
    *,
    index_cache_capacity: int = INDEX_TOPK,
    topk: int = INDEX_TOPK,
    window_rows: torch.Tensor | None = None,
    window_indices: torch.Tensor | None = None,
    compressed_rows: torch.Tensor | None = None,
    sink: torch.Tensor | None = None,
    candidate_topk_blocks: int = CANDIDATE_TOPK_BLOCKS,
    candidate_block_size: int = CANDIDATE_BLOCK_SIZE,
) -> dict[str, torch.Tensor]:
    """Run L20's C1A full source path and publish BF16/INT8 cache rows."""

    qkv = qkv_reference(hidden, _qkv_weights(weights), cosine, sine, _QKV_CONFIG)
    latent = _ratio1_compressor_reference(
        hidden, weights["compressor_wkv"], weights["compressor_norm"], eps=EPS
    )
    rotated_latent = rope_reference(latent, compressed_cosine, compressed_sine, 1, HEAD_DIM, ROPE_DIM)
    key = index_key_reference(
        latent, weights["index_wk"], weights["index_k_norm"],
        cosine=compressed_cosine, sine=compressed_sine,
    )
    key_payload, key_scale = _pad_index_cache(key["payload"], key["scale"], max(index_cache_capacity, key["payload"].shape[0], topk))
    query = indexer_query_reference(
        qkv["query_latent"], weights["index_wq_b"], weights["index_wq_b_scale"],
        cosine, sine,
    )
    index_weights = _index_weights(hidden, weights["index_weights_proj"])
    selected = _select_index_reference(
        query["payload"], query["scales"], key_payload, key_scale,
        index_weights, position_ids, topk=topk, compress_ratio=1,
    )
    candidate = select_candidate_block_pool_reference(
        selected["scores"], position_ids.to(torch.int64) + 1,
        topk_blocks=candidate_topk_blocks, block_size=candidate_block_size,
    )
    result: dict[str, torch.Tensor] = {
        **qkv,
        "compressor_latent": latent,
        "compressed": rotated_latent,
        "index_key": key["payload"],
        "index_key_scale": key["scale"],
        "index_query": query["query"],
        "index_query_payload": query["payload"],
        "index_query_scale": query["scales"],
        "index_weights": index_weights,
        "qli_scores": selected["scores"],
        "qli_indices": selected["raw_indices"],
        "prepared_indices": selected["indices"],
        "prepared_lengths": selected["lengths"],
        **candidate,
    }
    if window_rows is not None or window_indices is not None or compressed_rows is not None or sink is not None:
        if any(item is None for item in (window_rows, window_indices, compressed_rows, sink)):
            raise ValueError("window_rows, window_indices, compressed_rows and sink must be supplied together")
        attended = sparse_attention_reference(
            qkv["query"].reshape(hidden.shape[0], HEADS, HEAD_DIM),
            window_rows, window_indices, compressed_rows, selected["indices"], sink,
        )
        projected = output_projection_bf16_reference(
            attended.reshape(hidden.shape[0], HEADS * HEAD_DIM),
            weights["wo_a"], weights["wo_b"], cosine, sine,
            HEADS, HEAD_DIM, ROPE_DIM,
        )
        result["attended"] = attended
        result.update({f"output_{key}": value for key, value in projected.items()})
    return result


def c1a_reuse_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    cosine: torch.Tensor,
    sine: torch.Tensor,
    prepared_indices: torch.Tensor,
    window_rows: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_rows: torch.Tensor,
    sink: torch.Tensor,
    *,
    active: int | None = None,
) -> dict[str, torch.Tensor]:
    """Run the C1A reuse path used by L21 and L25."""

    qkv = qkv_reference(hidden, _qkv_weights(weights), cosine, sine, _QKV_CONFIG)
    attended = sparse_attention_reference(
        qkv["query"].reshape(hidden.shape[0], HEADS, HEAD_DIM),
        window_rows, window_indices, compressed_rows, prepared_indices, sink, active=active,
    )
    projected = output_projection_bf16_reference(
        attended.reshape(hidden.shape[0], HEADS * HEAD_DIM),
        weights["wo_a"], weights["wo_b"], cosine, sine,
        HEADS, HEAD_DIM, ROPE_DIM,
    )
    return {
        **qkv,
        "prepared_indices": prepared_indices,
        "attended": attended,
        **{f"output_{key}": value for key, value in projected.items()},
    }


def c1a_reindex_reference(
    hidden: torch.Tensor,
    weights: Mapping[str, torch.Tensor],
    position_ids: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    candidate_blocks: torch.Tensor,
    window_rows: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_rows: torch.Tensor,
    sink: torch.Tensor,
    *,
    topk: int = INDEX_TOPK,
    candidate_block_size: int = 8,
    active: int | None = None,
    candidate_block_counts: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Run L24's index re-selection against the L20 cache."""

    qkv = qkv_reference(hidden, _qkv_weights(weights), cosine, sine, _QKV_CONFIG)
    query = indexer_query_reference(
        qkv["query_latent"], weights["index_wq_b"], weights["index_wq_b_scale"], cosine, sine,
    )
    key = index_cache.reshape(-1, INDEX_DIM).to(torch.int8)
    scales = index_cache_scale.reshape(-1).to(torch.float16)
    index_weights = _index_weights(hidden, weights["index_weights_proj"])
    selected = _select_index_reference(
        query["payload"], query["scales"], key, scales, index_weights, position_ids,
        topk=topk, compress_ratio=1, candidate_blocks=candidate_blocks,
        candidate_block_counts=candidate_block_counts, candidate_block_size=candidate_block_size,
    )
    attended = sparse_attention_reference(
        qkv["query"].reshape(hidden.shape[0], HEADS, HEAD_DIM),
        window_rows, window_indices, compressed_rows, selected["indices"], sink, active=active,
    )
    projected = output_projection_bf16_reference(
        attended.reshape(hidden.shape[0], HEADS * HEAD_DIM),
        weights["wo_a"], weights["wo_b"], cosine, sine,
        HEADS, HEAD_DIM, ROPE_DIM,
    )
    return {
        **qkv,
        "index_query": query["query"],
        "index_query_payload": query["payload"],
        "index_query_scale": query["scales"],
        "index_weights": index_weights,
        "qli_scores": selected["scores"],
        "qli_indices": selected["raw_indices"],
        "prepared_indices": selected["indices"],
        "prepared_lengths": selected["lengths"],
        "attended": attended,
        **{f"output_{key}": value for key, value in projected.items()},
    }


_QKV_CONFIG = {
    "hidden_size": HIDDEN,
    "q_lora_rank": Q_LORA,
    "num_attention_heads": HEADS,
    "head_dim": HEAD_DIM,
    "qk_rope_head_dim": ROPE_DIM,
    "rms_norm_eps": EPS,
}


def make_decoder_layer_stages(layer_id: int, capacity: int) -> dict[str, Any]:
    """Build independently compilable A2/A3 stages for one target layer."""

    contract = layer_contract(layer_id)
    validate_layer19_20_21_24_25_contracts()
    stages: dict[str, Any] = {
        "contract": contract,
        "qkv": make_qkv_w8a8(capacity, _QKV_CONFIG),
        "sparse_attention": make_sparse_attention(capacity, HEADS, HEAD_DIM, WINDOW, INDEX_TOPK),
        "output_projection_bf16": make_output_projection_bf16(
            capacity, HEADS, HEAD_DIM, ROPE_DIM, GROUPS, OUTPUT_RANK, HIDDEN,
        ),
        "dtype_contract": A2A3_DTYPE_CONTRACT,
    }
    if contract.mode == "c1a_full":
        stages.update(
            compressor=make_bf16_linear(capacity, HIDDEN, HEAD_DIM),
            compressor_norm=make_rmsnorm(capacity, HEAD_DIM, EPS),
            compressed_rope=make_rope(capacity, 1, HEAD_DIM, ROPE_DIM),
            index_key=make_index_key_publish_rope(capacity, HEAD_DIM, ROPE_DIM),
            index_query=make_l3_indexer_query(capacity),
            index_select=make_indexer_select(capacity, INDEX_TOPK, topk=INDEX_TOPK, compress_ratio=1),
            index_prepare=make_indexer_prepare(capacity, topk=INDEX_TOPK, compress_ratio=1),
            candidate_pool=make_candidate_block_pool(capacity),
        )
    elif contract.mode == "c1a_reindex":
        stages.update(
            index_query=make_l3_indexer_query(capacity),
            index_select=make_indexer_select(capacity, INDEX_TOPK, topk=INDEX_TOPK, compress_ratio=1),
            index_prepare=make_indexer_prepare(capacity, topk=INDEX_TOPK, compress_ratio=1),
        )
    else:
        stages["index_prepare"] = make_indexer_prepare(capacity, topk=INDEX_TOPK, compress_ratio=contract.compression_ratio)
    return stages


__all__ = [
    "DecoderLayerA2A3Contract",
    "LAYER19",
    "LAYER20",
    "LAYER21",
    "LAYER24",
    "LAYER25",
    "TARGET_LAYERS",
    "c1a_full_reference",
    "c1a_reindex_reference",
    "c1a_reuse_reference",
    "c2a_reuse_reference",
    "select_candidate_block_pool_reference",
    "layer_contract",
    "make_decoder_layer_stages",
    "validate_layer19_20_21_24_25_contracts",
]
