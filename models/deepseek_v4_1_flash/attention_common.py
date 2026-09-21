# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Torch references and cache validation shared by the V4.1 attention modes."""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from models.deepseek_v4_1_flash.config import FLASH, AttentionMode
from models.deepseek_v4_1_flash.golden import (
    compressor_ratio1,
    compressor_ratio2_paged,
    index_key,
    merge_attention_stats,
    paged_indexer,
    paged_sparse_attention,
    paged_sparse_attention_stats,
    qkv_proj_rope,
    rope_interleave,
    select_candidate_blocks,
)
from models.deepseek_v4_1_flash.quantization import (
    dequantize_mxfp4_cache,
    dequantize_mxfp8_cache,
    mxfp8_linear,
    quantize_mxfp4_cache,
    quantize_mxfp8_cache,
)


@dataclass(frozen=True)
class AttentionGoldenResult:
    """Attention output and every mutable or published state produced by a mode."""

    output: torch.Tensor
    window_cache: torch.Tensor
    window_cache_scale: torch.Tensor
    compressed_cache: torch.Tensor | None
    compressed_cache_scale: torch.Tensor | None
    index_cache: torch.Tensor | None
    index_cache_scale: torch.Tensor | None
    state_cache: torch.Tensor | None
    topk_indices: torch.Tensor | None
    candidate_mask: torch.Tensor | None


def _project_output(
    attended: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    rope_dim = cos.shape[-1] * 2
    tail = rope_interleave(attended[..., -rope_dim:], cos, sin, inverse=True)
    attended = torch.cat((attended[..., :-rope_dim], tail), dim=-1)
    groups = wo_a.shape[0]
    grouped = attended.flatten(-2).unflatten(-1, (groups, -1))
    latent = torch.einsum("tgd,grd->tgr", grouped.float(), wo_a.float()).to(attended.dtype)
    return mxfp8_linear(latent.flatten(-2), wo_b, wo_b_scale, output_dtype=output_dtype)


def _copy_cache_rows(destination: torch.Tensor, source: torch.Tensor, rows: torch.Tensor) -> None:
    destination_rows = destination.flatten(0, 1).view(torch.uint8)
    source_rows = source.contiguous().view(torch.uint8).reshape_as(destination_rows[rows])
    destination_rows[rows] = source_rows


def _publish_quantized_cache(
    payload: torch.Tensor,
    scale: torch.Tensor,
    values: torch.Tensor,
    slots: torch.Tensor,
    quantize: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize and replace only mapped rows in a packed cache pair."""
    updated_payload = payload.clone()
    updated_scale = scale.clone()
    valid = slots >= 0
    rows = slots[valid].to(torch.int64)
    if rows.numel():
        row_payload, row_scale = quantize(values[valid])
        _copy_cache_rows(updated_payload, row_payload, rows)
        _copy_cache_rows(updated_scale, row_scale, rows)
    return updated_payload, updated_scale


def quantized_cache_compare(
    payload_name: str,
    scale_name: str,
    mapping_name: str,
    max_relative_l2: float,
    group_size: int | None = None,
    scale_format: str | None = None,
) -> Callable:
    """Validate mapped cache values and exact storage ownership elsewhere."""

    def compare(
        _actual: torch.Tensor,
        _expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        del rtol, atol
        if group_size is None:
            actual_value = dequantize_mxfp8_cache(
                actual_outputs[payload_name],
                actual_outputs[scale_name],
            )
            expected_value = dequantize_mxfp8_cache(
                expected_outputs[payload_name],
                expected_outputs[scale_name],
            )
        else:
            actual_value = dequantize_mxfp4_cache(
                actual_outputs[payload_name],
                actual_outputs[scale_name],
                group_size=group_size,
                scale_format=scale_format,
            )
            expected_value = dequantize_mxfp4_cache(
                expected_outputs[payload_name],
                expected_outputs[scale_name],
                group_size=group_size,
                scale_format=scale_format,
            )
        physical_rows = actual_value.shape[1] * actual_value.shape[2]
        actual_rows = actual_value.flatten(1, 2)
        expected_rows = expected_value.flatten(1, 2)
        active_actual = []
        active_expected = []
        for rank in range(actual_value.shape[0]):
            active = torch.unique(inputs[mapping_name][rank].flatten().to(torch.int64), sorted=True)
            active = active[active >= 0]
            if active.numel() and int(active[-1]) >= physical_rows:
                return False, f"    {mapping_name} row {int(active[-1])} exceeds {physical_rows}"
            active_actual.append(actual_rows[rank, active])
            active_expected.append(expected_rows[rank, active])
            inactive = torch.ones(physical_rows, dtype=torch.bool)
            inactive[active] = False
            for name in (payload_name, scale_name):
                actual_storage = actual_outputs[name][rank].contiguous().view(torch.uint8).reshape(physical_rows, -1)
                expected_storage = expected_outputs[name][rank].contiguous().view(torch.uint8).reshape(
                    physical_rows,
                    -1,
                )
                if not torch.equal(actual_storage[inactive], expected_storage[inactive]):
                    return False, f"    {name} modified inactive {mapping_name} rows"

        actual_active = torch.cat(active_actual).float()
        expected_active = torch.cat(active_expected).float()
        if not torch.isfinite(actual_active).all() or not torch.isfinite(expected_active).all():
            return False, f"    active {mapping_name} rows contain non-finite values"
        relative_l2 = (actual_active - expected_active).norm() / expected_active.norm().clamp_min(1e-12)
        print(f"[PRECISION] {payload_name} active_value_rel_l2={relative_l2.item():.6g}")
        return (
            bool(relative_l2 <= max_relative_l2),
            f"    active {mapping_name} rows must stay within {max_relative_l2:.0%} relative L2",
        )

    compare.__name__ = f"dequantized_{payload_name}"
    return compare


def golden_swa_attention(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> AttentionGoldenResult:
    """Evaluate SWA and publish the current token KV rows."""
    query, window_kv, _ = qkv_proj_rope(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
    )
    window_payload, updated_window_scale = _publish_quantized_cache(
        window_cache,
        window_cache_scale,
        window_kv,
        window_slots,
        quantize_mxfp8_cache,
    )
    quantized_window = dequantize_mxfp8_cache(window_payload, updated_window_scale).to(query.dtype)
    window_stats = paged_sparse_attention_stats(query, quantized_window, window_indices)
    attended = merge_attention_stats((window_stats,), attn_sink).to(query.dtype)
    output = _project_output(attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale)
    return AttentionGoldenResult(
        output, window_payload, updated_window_scale, None, None, None, None, None, None, None
    )


def golden_compressed_attention(
    mode: AttentionMode,
    ratio: int,
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor | None,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    compressed_indices: torch.Tensor | None,
    compressor_wkv: torch.Tensor | None,
    compressor_wgate: torch.Tensor | None,
    compressor_norm_weight: torch.Tensor | None,
    state_block_table: torch.Tensor | None,
    state_cache: torch.Tensor | None,
    compressed_slots: torch.Tensor | None,
    position_ids: torch.Tensor | None,
    compressed_lens: torch.Tensor | None,
    compressed_rope_cos: torch.Tensor | None,
    compressed_rope_sin: torch.Tensor | None,
    index_wk: torch.Tensor | None,
    index_norm_weight: torch.Tensor | None,
    index_wq_b: torch.Tensor | None,
    index_wq_b_scale: torch.Tensor | None,
    index_weights_proj: torch.Tensor | None,
    index_cache: torch.Tensor | None,
    index_cache_scale: torch.Tensor | None,
    index_block_table: torch.Tensor | None,
    request_ids: torch.Tensor | None,
    candidate_mask: torch.Tensor | None,
    attention_fn: Callable | None = None,
    output_dtype: torch.dtype | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> AttentionGoldenResult:
    """Evaluate C2A/C1A full, reindex, or reuse with paged cache state."""
    query, window_kv, query_latent = qkv_proj_rope(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        rope_cos,
        rope_sin,
    )
    window_payload, updated_window_scale = _publish_quantized_cache(
        window_cache,
        window_cache_scale,
        window_kv,
        window_slots,
        quantize_mxfp8_cache,
    )
    quantized_window = dequantize_mxfp8_cache(window_payload, updated_window_scale).to(query.dtype)
    compressed_payload = compressed_cache
    updated_compressed_scale = compressed_cache_scale
    quantized_compressed = dequantize_mxfp4_cache(
        compressed_cache, compressed_cache_scale, group_size=16, scale_format="e4m3"
    ).to(query.dtype)
    index_payload = index_cache
    updated_index_scale = index_cache_scale
    quantized_index = None
    if index_cache is not None and index_cache_scale is not None:
        quantized_index = dequantize_mxfp4_cache(
            index_cache, index_cache_scale, group_size=32, scale_format="e8m0"
        ).to(query.dtype)
    updated_state = None if state_cache is None else state_cache.clone()
    topk_indices = compressed_indices
    candidates = candidate_mask

    if mode is AttentionMode.FULL:
        if compressor_wkv is None or compressor_norm_weight is None or compressed_slots is None:
            raise ValueError("full mode requires compressor weights and compressed slots")
        if compressed_rope_cos is None or compressed_rope_sin is None or quantized_index is None:
            raise ValueError("full mode requires compressed RoPE rows and an index cache")
        if ratio == 1:
            latent = compressor_ratio1(x, compressor_wkv, compressor_norm_weight)
            publish_mask = compressed_slots >= 0
        elif ratio == 2:
            if compressor_wgate is None or state_block_table is None or updated_state is None:
                raise ValueError("ratio-2 full mode requires gate weights and recurrent state")
            if position_ids is None or query_start_loc is None or request_ids is None:
                raise ValueError("ratio-2 full mode requires positions, query starts and token-to-request indices")
            latent, publish_mask = compressor_ratio2_paged(
                x,
                query_start_loc,
                position_ids,
                request_ids,
                state_block_table,
                updated_state,
                compressor_wkv,
                compressor_wgate,
                compressor_norm_weight,
            )
        else:
            raise ValueError(f"unsupported compression ratio {ratio}")
        latent_tail = rope_interleave(
            latent[..., -compressed_rope_cos.shape[-1] * 2 :], compressed_rope_cos, compressed_rope_sin
        )
        rotated_latent = torch.cat((latent[..., : -compressed_rope_cos.shape[-1] * 2], latent_tail), dim=-1)
        publish_slots = compressed_slots.masked_fill(~publish_mask, -1)
        compressed_payload, updated_compressed_scale = _publish_quantized_cache(
            compressed_cache,
            compressed_cache_scale,
            rotated_latent,
            publish_slots,
            lambda value: quantize_mxfp4_cache(value, group_size=16, scale_format="e4m3"),
        )
        quantized_compressed = dequantize_mxfp4_cache(
            compressed_payload,
            updated_compressed_scale,
            group_size=16,
            scale_format="e4m3",
        ).to(query.dtype)
        if index_wk is None or index_norm_weight is None:
            raise ValueError("full mode requires index-key weights")
        keys = index_key(
            latent,
            index_wk,
            index_norm_weight,
            compressed_rope_cos,
            compressed_rope_sin,
        )
        index_payload, updated_index_scale = _publish_quantized_cache(
            index_cache,
            index_cache_scale,
            keys,
            publish_slots,
            lambda value: quantize_mxfp4_cache(value, group_size=32, scale_format="e8m0"),
        )
        quantized_index = dequantize_mxfp4_cache(
            index_payload,
            updated_index_scale,
            group_size=32,
            scale_format="e8m0",
        ).to(query.dtype)

    if mode in (AttentionMode.FULL, AttentionMode.REINDEX):
        if index_wq_b is None or index_weights_proj is None:
            raise ValueError("indexing modes require query and score weights")
        if (
            quantized_index is None
            or index_block_table is None
            or request_ids is None
            or compressed_lens is None
        ):
            raise ValueError("indexing modes require cache addressing and causal compressed lengths")
        scores, topk_indices = paged_indexer(
            x,
            query_latent,
            request_ids,
            quantized_index,
            index_block_table,
            compressed_lens,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            rope_cos,
            rope_sin,
            candidates=candidate_mask,
            topk=FLASH.index_topk,
        )
        if ratio == 1 and mode is AttentionMode.FULL:
            candidates = select_candidate_blocks(
                scores,
                compressed_lens,
                FLASH.candidate_topk_blocks,
                FLASH.candidate_block_size,
            )
    if topk_indices is None:
        raise ValueError("reuse mode requires published compressed Top-K indices")

    attend = attention_fn or paged_sparse_attention
    attended = attend(
        query, quantized_window, window_indices, quantized_compressed, topk_indices, attn_sink
    )
    output = _project_output(
        attended, rope_cos, rope_sin, wo_a, wo_b, wo_b_scale, output_dtype=output_dtype
    )
    return AttentionGoldenResult(
        output,
        window_payload,
        updated_window_scale,
        compressed_payload,
        updated_compressed_scale,
        index_payload,
        updated_index_scale,
        updated_state,
        topk_indices,
        candidates,
    )
