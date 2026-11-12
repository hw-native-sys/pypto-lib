# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Registered native QLI custom-op boundary for actual PyPTO operands."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QLIContract:
    """Source-equivalent A2 QLI selection configuration."""

    topk: int = 512
    compress_ratio: int = 2
    candidate_mode: int = 3
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8


def run_qli_native_boundary(
    query_i8, weights_fp16, query_scale_fp16,
    key_i8, key_scale_fp16, source_metadata,
    *, contract=QLIContract(), candidates=None,
):
    """Dispatch the compiled QLI operator using product-generated operands.

    The caller supplies eager-completed vLLM scheduling metadata and executes
    PyPTO causal sorting/paging after this call. Tensor views here preserve the
    native paged strides. No model forward or Torch arithmetic is executed.
    """
    if query_i8.ndim != 3 or tuple(query_i8.shape[1:]) != (32, 128):
        raise ValueError("query_i8 must have all 32 index heads and width 128")
    active = query_i8.shape[0]
    expected = ((query_i8, torch.int8), (weights_fp16, torch.float16), (query_scale_fp16, torch.float16))
    expected += ((key_i8, torch.int8), (key_scale_fp16, torch.float16))
    if any(value.dtype != dtype for value, dtype in expected):
        raise ValueError("QLI payloads and scales must match the INT8/FP16 A2 ABI")
    if tuple(weights_fp16.shape) != (active, 32) or tuple(query_scale_fp16.shape) != (active, 32):
        raise ValueError("query scales and weights must have shape [active, 32]")
    if key_i8.ndim != 4 or tuple(key_i8.shape[2:]) != (1, 128):
        raise ValueError("key_i8 must preserve native [pages, storage_rows, 1, 128]")
    if tuple(key_scale_fp16.shape) != tuple(key_i8.shape[:3]) + (1,):
        raise ValueError("key scales must have one FP16 scale per physical row")
    if key_i8.shape[1] not in (64, 128):
        raise ValueError("cache storage rows must be the actual 64/128-row A2 ABI")
    if any(value.device != query_i8.device for value, _ in expected):
        raise ValueError("all QLI operands must be on the same device")
    if query_i8.device.type != "npu":
        raise ValueError("native QLI acceptance requires the real A2 device")
    if not all(value.is_contiguous() for value in (query_i8, weights_fp16, query_scale_fp16)):
        raise ValueError("active query operands must be contiguous")
    if source_metadata.max_cache_seq_len <= 0:
        raise ValueError("empty-cache sentinel publication must use the product control kernel")
    if contract.topk != 512 or contract.compress_ratio not in (1, 2):
        raise ValueError("the declared source contract requires 512 sparse slots and ratio 1 or 2")
    if contract.candidate_mode not in (1, 2, 3):
        raise ValueError("invalid native candidate mode")
    if contract.candidate_mode == 2 and candidates is None:
        raise ValueError("candidate-filter consumers require actual source candidate IDs")
    operator = torch.ops._C_ascend.npu_quant_lightning_indexer_v3
    selected, scores, candidate_out = operator(
        query_i8, key_i8, weights_fp16, query_scale_fp16, key_scale_fp16.squeeze(-1),
        contract.topk, 2,
        block_table=source_metadata.block_table,
        metadata=source_metadata.qli_metadata,
        candidate_topk_index=candidates if contract.candidate_mode == 2 else None,
        candidate_mode=contract.candidate_mode,
        candidate_topk_blocks=contract.candidate_topk_blocks,
        candidate_block_size=contract.candidate_block_size,
        cu_seqlens_q=source_metadata.query_start_loc,
        seqused_k=source_metadata.cache_seq_lens,
        cmp_residual_k=source_metadata.cmp_residual,
        max_seqlen_q=source_metadata.max_query_len,
        layout_q="TND", layout_k="PA_BBND", mask_mode=3, cmp_ratio=contract.compress_ratio,
    )
    return selected.squeeze(1), scores, candidate_out
