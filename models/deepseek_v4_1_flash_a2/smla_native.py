# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Full64 head BF16 ratio1 paged SMLA with its own fresh metadata."""

from .scheduler_utils import require
from .smla_scheduler import HEAD_DIM, HEADS, PAGE_ROWS, TOPK, from_controls

NAMESPACE = 'pypto_dsv41_ratio1_smla_bf16_v1'
QUALIFIED = NAMESPACE + '::attention'
SCHEMA = ('attention(Tensor query, Tensor ori_cache, Tensor cmp_cache, Tensor sink, '
          'Tensor ori_block_table, Tensor cmp_block_table, Tensor selected_logical, '
          'Tensor query_start_loc, Tensor seq_lens, int num_requests, int max_query_len, '
          'int max_seq_len) -> (Tensor, Tensor)')
METADATA_SIZE = 1024
_REGISTRATIONS = []


def validate_descriptors(torch_api, query, ori_cache, cmp_cache, sink, ori_table, cmp_table,
                         selected, starts, seq_lens, num_requests, max_query_len, max_seq_len):
    require(query.ndim == 3 and query.shape[0] > 0
            and tuple(query.shape[1:]) == (HEADS, HEAD_DIM), 'Complete BF16 TND64x512 query required')
    for cache in (ori_cache, cmp_cache):
        require(cache.ndim == 4 and cache.shape[0] > 0 and tuple(cache.shape[1:]) ==
                (PAGE_ROWS, 1, HEAD_DIM), 'Separate complete BF16 PA_BBND own cache lanes required')
    for value in (num_requests, max_query_len, max_seq_len):
        require(type(value) is int and 0 < value < (1 << 31), 'Exact own positive INT32 scalar required')
    require(tuple(sink.shape) == (HEADS,) and tuple(selected.shape) == (query.shape[0], 1, TOPK)
            and tuple(starts.shape) == (num_requests + 1,) and tuple(seq_lens.shape) == (num_requests,),
            'Full original source sink, logical compressed selection and scheduler shapes required')
    for table in (ori_table, cmp_table):
        require(table.ndim == 2 and table.shape[0] == num_requests and table.shape[1] > 0,
                'Independent complete own page tables required')
    values = (query, ori_cache, cmp_cache, sink, ori_table, cmp_table, selected, starts, seq_lens)
    expected = (torch_api.bfloat16,) * 3 + (torch_api.float32,) + (torch_api.int32,) * 5
    require(all(value.dtype == dtype for value, dtype in zip(values, expected, strict=True))
            and query.device.type == 'npu' and all(value.device == query.device for value in values)
            and all(value.is_contiguous() for value in values), 'One NPU, contiguous BF16/F32/INT32 ports required')


def implementation(torch_api):
    def attention(query, ori_cache, cmp_cache, sink, ori_block_table, cmp_block_table,
                  selected_logical, query_start_loc, seq_lens, num_requests, max_query_len, max_seq_len):
        validate_descriptors(torch_api, query, ori_cache, cmp_cache, sink, ori_block_table,
            cmp_block_table, selected_logical, query_start_loc, seq_lens,
            num_requests, max_query_len, max_seq_len)
        # Only own integer control readbacks; no floating point attention arithmetic.
        controls = [value.detach().cpu().tolist() for value in
                    (query_start_loc, seq_lens, ori_block_table, cmp_block_table)]
        schedule = from_controls(*controls, [row[0] for row in selected_logical.detach().cpu().tolist()],
            ori_blocks=ori_cache.shape[0], cmp_blocks=cmp_cache.shape[0], capacity=query.shape[0])
        require((schedule['active'], schedule['num_reqs'], schedule['max_query_len'], schedule['max_seq_len'])
                == (query.shape[0], num_requests, max_query_len, max_seq_len),
                'Actual own integer controls and all scalars must agree')
        metadata = torch_api.ops._C_ascend.npu_sparse_flash_mla_metadata(
            HEADS, 1, HEAD_DIM, cu_seqlens_q=query_start_loc, seqused_ori_kv=seq_lens,
            seqused_cmp_kv=seq_lens, cmp_residual_kv=None, batch_size=num_requests,
            max_seqlen_q=max_query_len, max_seqlen_ori_kv=max_seq_len, max_seqlen_cmp_kv=max_seq_len,
            ori_topk=0, ori_topk_length=None, cmp_topk=TOPK, cmp_ratio=1,
            ori_mask_mode=4, cmp_mask_mode=3, ori_win_left=127, ori_win_right=0,
            layout_q='TND', layout_kv='PA_BBND', has_ori_kv=True, has_cmp_kv=True)
        require(tuple(metadata.shape) == (METADATA_SIZE,) and metadata.dtype == torch_api.int32
                and metadata.device == query.device and metadata.is_contiguous(), 'Fresh generated INT32[1024] metadata required')
        output, lse = torch_api.ops._C_ascend.npu_sparse_flash_mla(
            query, ori_kv=ori_cache, cmp_kv=cmp_cache, ori_sparse_indices=None,
            ori_topk_length=None, cmp_sparse_indices=selected_logical, ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table, cu_seqlens_q=query_start_loc, seqused_ori_kv=seq_lens,
            seqused_cmp_kv=seq_lens, cmp_residual_kv=None, sinks=sink, metadata=metadata,
            softmax_scale=HEAD_DIM**-0.5, cmp_ratio=1, ori_mask_mode=4, cmp_mask_mode=3,
            ori_win_left=127, ori_win_right=0, layout_q='TND', layout_kv='PA_BBND',
            topk_value_mode=1, return_softmax_lse=False)
        require(tuple(output.shape) == tuple(query.shape) and output.dtype == torch_api.bfloat16
                and output.device == query.device and output.is_contiguous() and lse.numel() == 0,
                'Original BF16 SMLA output and absent LSE ABI required')
        return output, metadata
    return attention


def register(torch_api):
    require(not _REGISTRATIONS, 'Fresh interpreter must register exactly once')
    definition = torch_api.library.Library(NAMESPACE, 'DEF')
    definition.define(SCHEMA)
    dispatch = torch_api.library.Library(NAMESPACE, 'IMPL', 'PrivateUse1')
    dispatch.impl('attention', implementation(torch_api))
    _REGISTRATIONS.extend((definition, dispatch))
    return dict(namespace=NAMESPACE, operator=QUALIFIED, schema=SCHEMA, dispatch='PrivateUse1',
        CPU_implementation=False, fallback=False, fresh_metadata_inside_leaf=True,
        full_query_heads=HEADS, head_dim=HEAD_DIM, cmp_ratio=1, compressed_indices='own logical IDs')
