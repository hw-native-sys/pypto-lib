# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Concrete PrivateUse1 registration for full-head ordinary BF16 paged SWA."""

from .scheduler_utils import require
from .swa_scheduler import HEAD_DIM, HEADS, PAGE_ROWS, from_controls

NAMESPACE = 'pypto_dsv41_swa_bf16_v1'
QUALIFIED = NAMESPACE + '::attention'
SCHEMA = ('attention(Tensor query, Tensor published_cache, Tensor sink, Tensor block_table, '
          'Tensor query_start_loc, Tensor seq_lens, int num_requests, int max_query_len, '
          'int max_seq_len) -> (Tensor, Tensor)')
METADATA_SIZE = 1024  # Upstream SMLA_METADATA_SIZE, sparse_flash_mla_torch_adpt.h.
_REGISTRATIONS = []


def validate_descriptors(torch_api, query, cache, sink, block_table, starts, seq_lens,
                         num_requests, max_query_len, max_seq_len):
    require(query.ndim == 3 and query.shape[0] > 0
            and tuple(query.shape[1:]) == (HEADS, HEAD_DIM), 'Require all64 BF16 heads of dimension512')
    require(cache.ndim == 4 and cache.shape[0] > 0
            and tuple(cache.shape[1:]) == (PAGE_ROWS, 1, HEAD_DIM), 'Require own complete PA_BBND cache')
    require(tuple(sink.shape) == (HEADS,), 'Require one original F32 sink for every query head')
    for value in (num_requests, max_query_len, max_seq_len):
        require(type(value) is int and 0 < value < (1 << 31), 'Exact positive scheduler scalar required')
    require(block_table.ndim == 2 and block_table.shape[0] == num_requests
            and block_table.shape[1] > 0 and tuple(starts.shape) == (num_requests + 1,)
            and tuple(seq_lens.shape) == (num_requests,), 'Complete own scheduler tensor shapes required')
    values = (query, cache, sink, block_table, starts, seq_lens)
    expected = (torch_api.bfloat16, torch_api.bfloat16, torch_api.float32,
                torch_api.int32, torch_api.int32, torch_api.int32)
    require(all(value.dtype == dtype for value, dtype in zip(values, expected, strict=True)),
            'BF16 data/F32 sinks/INT32 scheduler ports required')
    require(query.device.type == 'npu' and all(value.device == query.device for value in values),
            'Registered leaf requires actual operands on one NPU')
    require(all(value.is_contiguous() for value in values), 'Contiguous original leaf layouts required')
    return values


def implementation(torch_api):
    def attention(query, published_cache, sink, block_table, query_start_loc, seq_lens,
                  num_requests, max_query_len, max_seq_len):
        validate_descriptors(torch_api, query, published_cache, sink, block_table,
                             query_start_loc, seq_lens, num_requests, max_query_len, max_seq_len)
        # These readbacks validate only own integer control data; no attention arithmetic.
        schedule = from_controls(query_start_loc.detach().cpu().tolist(), seq_lens.detach().cpu().tolist(),
                                 block_table.detach().cpu().tolist(), cache_blocks=published_cache.shape[0],
                                 capacity=query.shape[0])
        require(schedule['active'] == query.shape[0] and schedule['num_reqs'] == num_requests
                and schedule['max_query_len'] == max_query_len and schedule['max_seq_len'] == max_seq_len,
                'Own actual scheduler tensors must agree with every scalar')
        metadata = torch_api.ops._C_ascend.npu_sparse_flash_mla_metadata(
            HEADS, 1, HEAD_DIM, cu_seqlens_q=query_start_loc, seqused_ori_kv=seq_lens,
            seqused_cmp_kv=None, cmp_residual_kv=None, batch_size=num_requests,
            max_seqlen_q=max_query_len, max_seqlen_ori_kv=max_seq_len, max_seqlen_cmp_kv=0,
            ori_topk=0, ori_topk_length=None, cmp_topk=0, cmp_ratio=0,
            ori_mask_mode=4, cmp_mask_mode=0, ori_win_left=127, ori_win_right=0,
            layout_q='TND', layout_kv='PA_BBND', has_ori_kv=True, has_cmp_kv=False)
        require(tuple(metadata.shape) == (METADATA_SIZE,) and metadata.dtype == torch_api.int32
                and metadata.device == query.device and metadata.is_contiguous(),
                'Fresh operator-generated INT32[1024] metadata required')
        output, lse = torch_api.ops._C_ascend.npu_sparse_flash_mla(
            query, ori_kv=published_cache, cmp_kv=None, ori_sparse_indices=None,
            ori_topk_length=None, cmp_sparse_indices=None, ori_block_table=block_table,
            cmp_block_table=None, cu_seqlens_q=query_start_loc, seqused_ori_kv=seq_lens,
            seqused_cmp_kv=None, cmp_residual_kv=None, sinks=sink, metadata=metadata,
            softmax_scale=HEAD_DIM**-0.5, cmp_ratio=0, ori_mask_mode=4, cmp_mask_mode=0,
            ori_win_left=127, ori_win_right=0, layout_q='TND', layout_kv='PA_BBND',
            topk_value_mode=1, return_softmax_lse=False)
        require(tuple(output.shape) == tuple(query.shape) and output.dtype == torch_api.bfloat16
                and output.device == query.device and output.is_contiguous()
                and lse.numel() == 0, 'Original BF16 output and absent LSE ABI required')
        # Metadata is an output receipt; it is never a functional input.
        return output, metadata
    return attention


def register(torch_api):
    """Install an actual custom op; libraries stay alive for the runtime lifetime."""
    require(not _REGISTRATIONS, 'This fresh interpreter must register the namespace exactly once')
    definition = torch_api.library.Library(NAMESPACE, 'DEF')
    definition.define(SCHEMA)
    dispatch = torch_api.library.Library(NAMESPACE, 'IMPL', 'PrivateUse1')
    dispatch.impl('attention', implementation(torch_api))
    _REGISTRATIONS.extend((definition, dispatch))
    return dict(namespace=NAMESPACE, operator=QUALIFIED, schema=SCHEMA,
                dispatch='PrivateUse1', CPU_implementation=False, fallback=False,
                fresh_metadata_inside_leaf=True, full_query_heads=HEADS, head_dim=HEAD_DIM)
