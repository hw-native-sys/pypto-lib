# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill C1A full attention wired through delayed mHC mixing."""

from collections.abc import Callable
import config as C
from prefill_indexer import index_pool_block_capacity
import functools
import inspect
import math
import os
import pypto.language as pl
import pypto.language.distributed as pld
import torch
from golden import ScalarSpec, TensorSpec, ratio_allclose, run, mapped_pool_ratio_allclose
from pypto.ir import DistributedConfig
from config import (
    FLASH as M, TP_SIZE, T_DYN, ORI_BLOCKS_DYN, CMP_BLOCKS_DYN, INDEX_BLOCKS_DYN, B_DYN, TABLE_DYN,
    INDEX_POOL_BLOCKS_DYN, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP, PREFILL_MAX_TOKENS, SP_T_DYN,
    WINDOW_CACHE_GROUP, BLOCK_SIZE,
)
from hc import golden_mhc_mixes
from hc import golden_hc_post, hc_post, prefill_sp_post
from hc import golden_hc_collapse, hc_pre
from rmsnorm import golden_rms_norm, rms_norm_tiled, rms_norm
from sparse_attention import (
    attend_packed, golden_compressed_attention, golden_prefill_c1a_attention,
    prefill_publish_window,
)
from hc import HC_INPUT_NAMES, make_hc_inputs
from utils import token_shards
from prefill_cp_allgather import prefill_sp_input_allgather
from o_proj import retire_prefill_signals, prefill_sp_output_reduce_scatter, o_proj, prefill_tp_output_all_reduce
from qkv_proj_rope import quantize_projection_input, q_proj_rope, kv_proj_rope
from compressor import c1a_compressor_project, compressor_rope, c1a_compressor_publish
from prefill_indexer import (
    prefill_indexer, publish_c1a_index_key, golden_prefill_indexer,
    golden_index_key, golden_select_candidate_block_pool,
    golden_prefill_index_scores as golden_index_scores,
)


CASE_DEFAULT = 'multi_1k'
CASE_MAX_TOKENS = C.PREFILL_MAX_TOKENS
CASE_PAGE = 128
CASE_NAMES = (CASE_DEFAULT, 'causal', 'mixed', 'long')
COMMON_INPUT_NAMES = ('x', 'wq_a', 'wq_a_scale', 'q_norm_weight', 'wq_b', 'wq_b_scale', 'wkv', 'wkv_scale', 'kv_norm_weight', 'attn_sink', 'wo_a', 'wo_a_scale', 'wo_b', 'wo_b_scale', 'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'window_cache', 'window_cache_scale', 'compressed_cache', 'compressed_cache_scale')


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
INDEX_TOPK = M.index_topk


PREFILL_ATTN_RING_HEAP = (1024 * 1024 * 1024,) * 4


LEAF_NAMES = ('wq_a', 'wq_a_scale', 'q_norm_weight', 'wq_b', 'wq_b_scale', 'wkv', 'wkv_scale', 'kv_norm_weight', 'attn_sink', 'wo_a', 'wo_a_scale', 'wo_b', 'wo_b_scale', 'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'window_cache', 'window_cache_scale', 'compressed_cache', 'compressed_cache_scale', 'token_to_req_indices', 'compressed_lens', 'index_cache', 'index_cache_scale', 'index_block_table', 'compressed_rope_cos', 'compressed_rope_sin', 'compressor_wkv', 'compressor_norm_weight', 'compressed_slots', 'index_wk', 'index_norm_weight', 'index_wq_b', 'index_wq_b_scale', 'index_weights_proj', 'topk_indices', "pool_block_ids", "pool_block_counts", 'compressed_indices')

FULL_INPUT_NAMES = COMMON_INPUT_NAMES + (
    "token_to_req_indices",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "compressed_rope_cos",
    "compressed_rope_sin",
    "compressor_wkv",
    "compressor_norm_weight",
    "compressed_slots",
    "index_wk",
    "index_norm_weight",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
)


@pl.jit.inline(auto_scope=False)
def prefill_c1a_full_attention(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
    scatter_output: pl.constexpr = False,
):
    """Publish ratio-1 caches, select sparse rows, and compute packed C1A."""
    tokens = pl.tensor.dim(x, 0)
    partial = pl.create_tensor([tokens, D], dtype=pl.BF16)
    query_latent = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    query = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
    proj_payload, proj_scales = quantize_projection_input(x, num_tokens)
    publication = pl.array.create(1, pl.TASK_ID)
    publication[0] = pl.system.task_dummy(deps=[])
    if num_tokens > 0:
        published = prefill_publish_decoder(
            x, compressor_wkv, compressor_norm_weight, index_wk, index_norm_weight,
            compressed_rope_cos, compressed_rope_sin, compressed_slots,
            compressed_cache, compressed_cache_scale, index_cache, index_cache_scale, num_tokens,
        )
        q_proj_rope(
            proj_payload, proj_scales,
            wq_a, wq_a_scale, q_norm_weight,
            wq_b, wq_b_scale,
            rope_cos, rope_sin,
            query_latent, query,
            num_tokens,
        )
        publication[0] = published
    publication_ready = publication[0]
    indexer_completion = prefill_indexer(
        x, query_latent, token_to_req_indices, compressed_lens, index_cache, index_cache_scale, index_block_table,
        rope_cos, rope_sin, index_wq_b, index_wq_b_scale, index_weights_proj,
        pool_block_ids, pool_block_counts, topk_indices, num_tokens, publication_ready, pool=False, c2a=False,
    )
    if num_tokens > 0:
        source_ready = pl.system.task_dummy(deps=[publication_ready, indexer_completion])
        window_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_proj_rope(
            proj_payload, proj_scales,
            wkv, wkv_scale, kv_norm_weight,
            rope_cos, rope_sin,
            kv_projection, window_kv,
            num_tokens,
        )
        window_ready = prefill_publish_window(window_kv, window_slots, window_cache, window_cache_scale, num_tokens, source_ready)

        attended = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        attend_packed(
            query,
            window_indices, window_cache, window_cache_scale,
            topk_indices, compressed_cache, compressed_cache_scale,
            attn_sink, attended,
            num_tokens, window_ready,
        )
        o_proj(
            attended, wo_a, wo_a_scale, wo_b,
            wo_b_scale, rope_cos, rope_sin,
            partial, num_tokens,
        )
    if scatter_output:
        prefill_sp_output_reduce_scatter(
            partial, output_window, output_arrived, output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    else:
        prefill_tp_output_all_reduce(
            partial, output_window, output_arrived, output,
            group_base, tp_rank, num_tokens, attention_epoch,
        )
    return output


def _ranked(value: torch.Tensor) -> torch.Tensor:
    from config import TP_SIZE
    return value.unsqueeze(0).repeat(TP_SIZE, *[1] * value.ndim).contiguous()


def _fp8_weight(input_width: int, output_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    import torch
    from quant import pack_mx_b_scale
    weight = torch.randn(input_width, output_width).mul_(input_width ** (-0.5))
    weight = weight.clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    scale_codes = torch.full((input_width // 32, output_width), 127, dtype=torch.uint8)
    scale = pack_mx_b_scale(scale_codes).view(torch.float8_e8m0fnu)
    return (weight, scale)


def _ranked_fp8_weight(input_width: int, output_width: int, shared: bool) -> tuple[torch.Tensor, torch.Tensor]:
    import torch
    from config import TP_SIZE
    pairs = [_fp8_weight(input_width, output_width) for _ in range(1 if shared else TP_SIZE)]
    if shared:
        weight = _ranked(pairs[0][0])
        scale = _ranked(pairs[0][1])
    else:
        weight = torch.stack([pair[0] for pair in pairs])
        scale = torch.stack([pair[1] for pair in pairs])
    return (weight, scale)


def _case_metadata(token_count: int, case_name: str) -> tuple[torch.Tensor, ...]:
    import torch
    if case_name == CASE_DEFAULT:
        token_to_req_indices = torch.arange(3, dtype=torch.int32)
        compressed_lens = torch.full((3,), 1024, dtype=torch.int32)
    elif case_name == 'causal':
        if not 1 <= token_count <= CASE_MAX_TOKENS:
            raise ValueError(f'token_count must be in [1, {CASE_MAX_TOKENS}], got {token_count}')
        token_to_req_indices = torch.zeros(token_count, dtype=torch.int32)
        compressed_lens = torch.arange(1, token_count + 1, dtype=torch.int32)
    elif case_name == 'mixed':
        token_to_req_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
        compressed_lens = torch.tensor([1, 2, 3, 127, 128, 511, 512, 513], dtype=torch.int32)
    elif case_name == 'long':
        token_to_req_indices = torch.arange(3, dtype=torch.int32)
        compressed_lens = torch.tensor([1024, 4096, 8192], dtype=torch.int32)
    else:
        raise ValueError(f'unknown case {case_name!r}; expected one of {CASE_NAMES}')
    request_count = int(token_to_req_indices.max().item()) + 1
    table_pages = (int(compressed_lens.max().item()) + CASE_PAGE - 1) // CASE_PAGE
    block_table = torch.arange(request_count * table_pages, dtype=torch.int32).reshape(request_count, table_pages)
    positions = compressed_lens.to(torch.int64) - 1
    pages = torch.div(positions, CASE_PAGE, rounding_mode='floor')
    offsets = positions.remainder(CASE_PAGE)
    slots = block_table[token_to_req_indices.to(torch.long), pages] * CASE_PAGE + offsets
    window_indices = torch.full((token_to_req_indices.numel(), CASE_PAGE), -1, dtype=torch.int32)
    compressed_indices = torch.full((token_to_req_indices.numel(), INDEX_TOPK), -1, dtype=torch.int32)
    candidate_mask = torch.zeros(token_to_req_indices.numel(), table_pages * CASE_PAGE, dtype=torch.uint8)
    for token in range(token_to_req_indices.numel()):
        request = int(token_to_req_indices[token])
        visible = int(compressed_lens[token])
        logical = torch.arange(visible, dtype=torch.int64)
        physical = block_table[request, logical // CASE_PAGE] * CASE_PAGE + logical % CASE_PAGE
        window_count = min(visible, CASE_PAGE)
        window_indices[token, :window_count] = physical[-window_count:].to(torch.int32)
        compressed_count = min(visible, INDEX_TOPK)
        compressed_indices[token, :compressed_count] = physical[:compressed_count].to(torch.int32)
        candidate_mask[token, :visible] = 1
    return (token_to_req_indices, compressed_lens, block_table, slots.to(torch.int64), window_indices, compressed_indices, candidate_mask)


def fixture_token_count(token_count: int, case_name: str) -> int:
    """Return the token rows a case fixture provides; only the causal case follows ``token_count``."""
    return int(_case_metadata(token_count, case_name)[0].numel())


def make_fixture_values(token_count: int, case_name: str) -> dict[str, torch.Tensor]:
    import torch
    from config import TP_SIZE, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP
    from quant import pack_bf16_nz, pack_fp8_nz, quantize_grouped_mxfp8_weight, quantize_mxfp4_cache, quantize_mxfp8_cache
    torch.manual_seed(41)
    token_to_req_indices, compressed_lens, block_table, slots, window_indices, compressed_indices, candidate_mask = _case_metadata(token_count, case_name)
    token_count = token_to_req_indices.numel()
    cache_blocks = block_table.numel()
    wq_a, wq_a_scale = _ranked_fp8_weight(D, Q_LORA, shared=True)
    wq_b, wq_b_scale = _ranked_fp8_weight(Q_LORA, LOCAL_H * HEAD_DIM, shared=False)
    wkv, wkv_scale = _ranked_fp8_weight(D, HEAD_DIM, shared=True)
    wo_b, wo_b_scale = _ranked_fp8_weight(LOCAL_O_WIDTH, D, shared=False)
    wq_a = pack_fp8_nz(wq_a)
    wq_b = pack_fp8_nz(wq_b)
    wkv = pack_fp8_nz(wkv)
    wo_b = pack_fp8_nz(wo_b)
    index_wq_b, index_wq_b_scale = _ranked_fp8_weight(Q_LORA, INDEX_H * INDEX_DIM, shared=True)
    index_wq_b = pack_fp8_nz(index_wq_b)
    window_value = torch.randn(cache_blocks, CASE_PAGE, 1, HEAD_DIM).mul_(0.05)
    window_cache, window_cache_scale = quantize_mxfp8_cache(window_value)
    compressed_value = torch.randn(cache_blocks, CASE_PAGE, 1, HEAD_DIM).mul_(0.05)
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(compressed_value, group_size=COMPRESSED_CACHE_GROUP, scale_format='bf16')
    index_value = torch.randn(cache_blocks, CASE_PAGE, 1, INDEX_DIM).mul_(0.05)
    index_cache, index_cache_scale = quantize_mxfp4_cache(index_value, group_size=INDEX_CACHE_GROUP, scale_format='e8m0')
    index_cache_scale = index_cache_scale.view(torch.float8_e8m0fnu)
    angles = (compressed_lens - 1).to(torch.float32).unsqueeze(1)
    angles = angles * torch.arange(ROPE_DIM // 2, dtype=torch.float32).unsqueeze(0) * 0.001
    values = {'x': _ranked(torch.randn(token_count, D).to(torch.bfloat16)), 'wq_a': wq_a, 'wq_a_scale': wq_a_scale, 'q_norm_weight': _ranked(torch.ones(Q_LORA, dtype=torch.bfloat16)), 'wq_b': wq_b, 'wq_b_scale': wq_b_scale, 'wkv': wkv, 'wkv_scale': wkv_scale, 'kv_norm_weight': _ranked(torch.ones(HEAD_DIM, dtype=torch.bfloat16)), 'attn_sink': torch.zeros(TP_SIZE, LOCAL_H, dtype=torch.float32), 'wo_a': torch.randn(TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16).mul_(O_GROUP_IN ** (-0.5)), 'wo_b': wo_b, 'wo_b_scale': wo_b_scale, 'rope_cos': _ranked(torch.cos(angles)), 'rope_sin': _ranked(torch.sin(angles)), 'window_slots': _ranked(slots), 'window_indices': _ranked(window_indices), 'window_cache': _ranked(window_cache), 'window_cache_scale': _ranked(window_cache_scale), 'compressed_cache': _ranked(compressed_cache), 'compressed_cache_scale': _ranked(compressed_cache_scale), 'compressed_indices': _ranked(compressed_indices), 'token_to_req_indices': _ranked(token_to_req_indices), 'compressed_lens': _ranked(compressed_lens), 'index_cache': _ranked(index_cache), 'index_cache_scale': _ranked(index_cache_scale), 'index_block_table': _ranked(block_table), 'candidate_mask': _ranked(candidate_mask), 'compressed_rope_cos': _ranked(torch.cos(angles)), 'compressed_rope_sin': _ranked(torch.sin(angles)), 'compressor_wkv': _ranked(torch.randn(D, HEAD_DIM).mul_(D ** (-0.5)).to(torch.bfloat16)), 'compressor_norm_weight': _ranked(torch.ones(HEAD_DIM, dtype=torch.bfloat16)), 'compressed_slots': _ranked(slots), 'index_wk': _ranked(torch.randn(HEAD_DIM, INDEX_DIM).mul_(HEAD_DIM ** (-0.5)).to(torch.bfloat16)), 'index_norm_weight': _ranked(torch.ones(INDEX_DIM, dtype=torch.bfloat16)), 'index_wq_b': index_wq_b, 'index_wq_b_scale': index_wq_b_scale, 'index_weights_proj': _ranked(torch.randn(D, INDEX_H).mul_(D ** (-0.5)).to(torch.bfloat16))}
    for name in ("compressor_wkv", "index_wk", "index_weights_proj"):
        values[name] = pack_bf16_nz(values[name])
    values['attn_norm_weight'] = _ranked(torch.randn(D).mul_(0.1).add_(1).to(torch.bfloat16))
    values["wo_a"], values["wo_a_scale"] = quantize_grouped_mxfp8_weight(values["wo_a"])
    pool_width = index_pool_block_capacity(candidate_mask.shape[1])
    ids = torch.full((token_count, pool_width), -1, dtype=torch.int32)
    counts = torch.zeros(token_count, dtype=torch.int32)
    for token in range(token_count):
        blocks = torch.nonzero(candidate_mask[token].reshape(-1, 8).any(-1)).flatten()
        count = min(blocks.numel(), pool_width)
        ids[token, :count] = blocks[-count:].int()
        counts[token] = count
    values['pool_block_ids'] = _ranked(ids)
    values['pool_block_counts'] = _ranked(counts)
    return values


def _copy_storage(destination: torch.Tensor, source: torch.Tensor) -> None:
    import torch
    destination.view(torch.uint8).copy_(source.contiguous().view(torch.uint8))


def _reduce_tp_partials(partials: list[torch.Tensor]) -> torch.Tensor:
    """Sum rank partials in rank order and round the completed reduction to BF16."""
    import torch
    reduced = torch.zeros_like(partials[0], dtype=torch.float32)
    for partial in partials:
        reduced.add_(partial.float())
    return reduced.to(torch.bfloat16)


def apply_distributed_golden(mode: str, golden_fn: Callable, tensors: dict[str, torch.Tensor], active_tokens: int | None=None) -> None:
    from config import TP_SIZE
    parameter_names = tuple(name for name, parameter in inspect.signature(golden_fn).parameters.items()
                            if parameter.default is inspect.Parameter.empty)
    token_input_names = {'x', 'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'token_to_req_indices', 'compressed_lens', 'compressed_rope_cos', 'compressed_rope_sin', 'compressed_slots', 'pool_block_ids', 'pool_block_counts', 'compressed_indices'}
    results = []
    for rank in range(TP_SIZE):
        kwargs = {}
        for name in parameter_names:
            value = tensors[name][rank]
            if active_tokens is not None and name in token_input_names and (value.ndim > 0):
                value = value[:active_tokens]
            kwargs[name] = value
        results.append(golden_fn(**kwargs))
    reduced = _reduce_tp_partials([result["output"] for result in results])
    if active_tokens is None:
        tensors['output'][:] = reduced.unsqueeze(0)
    else:
        tensors['output'][:, :active_tokens] = reduced.unsqueeze(0)
    for rank, result in enumerate(results):
        tensors['window_cache'][rank].copy_(result["window_cache"])
        _copy_storage(tensors['window_cache_scale'][rank], result["window_cache_scale"])
        if mode == 'full':
            tensors['compressed_cache'][rank].copy_(result["compressed_cache"])
            tensors['compressed_cache_scale'][rank].copy_(result["compressed_cache_scale"])
            tensors['index_cache'][rank].copy_(result["index_cache"])
            _copy_storage(tensors['index_cache_scale'][rank], result["index_cache_scale"])
            ids = tensors['pool_block_ids'][rank]
            counts = tensors['pool_block_counts'][rank]
            ids.fill_(-1)
            counts.zero_()
            rows = result["pool_block_ids"].shape[0]
            ids[:rows].copy_(result["pool_block_ids"])
            counts[:rows].copy_(result["pool_block_counts"])
        if mode in ('full', 'reindex'):
            if active_tokens is None:
                tensors['topk_indices'][rank].copy_(result["topk_indices"])
            else:
                tensors['topk_indices'][rank, :active_tokens].copy_(result["topk_indices"])


def _physical_to_logical(inputs: dict[str, torch.Tensor], rank: int, token: int) -> dict[int, int]:
    import torch
    request = int(inputs['token_to_req_indices'][rank, token])
    visible = int(inputs['compressed_lens'][rank, token])
    logical = torch.arange(visible, dtype=torch.int64)
    table = inputs['index_block_table'][rank, request]
    physical = table[logical // CASE_PAGE].to(torch.int64) * CASE_PAGE + logical % CASE_PAGE
    return {int(row): index for index, row in enumerate(physical.tolist())}


def topk_indices_compare(mode: str, score_fn: Callable) -> Callable:
    """Accept device/CPU substitutions only at a numerically tied top-k cutoff."""
    import torch

    def compare(actual: torch.Tensor, expected: torch.Tensor, *, actual_outputs: dict[str, torch.Tensor], expected_outputs: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor], rtol: float, atol: float) -> tuple[bool, str]:
        del rtol, atol
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return (False, '    Top-K shape and dtype must match the reference')
        active = min(int(inputs.get('num_tokens', actual.shape[1])), actual.shape[1])
        actual_active = actual[:, :active]
        expected_active = expected[:, :active]
        if bool((actual_active < -1).any()):
            return (False, '    Top-K padding must use -1')
        valid = actual_active >= 0
        if bool((valid[..., 1:] & ~valid[..., :-1]).any()):
            return (False, '    Top-K padding must follow all valid indices')
        if torch.equal(actual_active, expected_active):
            return (True, '')
        for rank in range(actual_active.shape[0]):
            scores = None
            for token in range(actual_active.shape[1]):
                actual_row = actual_active[rank, token]
                expected_row = expected_active[rank, token]
                actual_valid = actual_row[actual_row >= 0].to(torch.int64)
                expected_valid = expected_row[expected_row >= 0].to(torch.int64)
                if actual_valid.numel() != expected_valid.numel():
                    return (False, f'    rank {rank} token {token} has {actual_valid.numel()} valid indices; expected {expected_valid.numel()}')
                if torch.unique(actual_valid).numel() != actual_valid.numel():
                    return (False, f'    rank {rank} token {token} contains duplicate indices')
                physical_to_logical = _physical_to_logical(inputs, rank, token)
                if any((row not in physical_to_logical for row in actual_valid.tolist())):
                    return (False, f'    rank {rank} token {token} selected a row outside its request')
                positions = [physical_to_logical[row] for row in actual_valid.tolist()]
                if any((left >= right for left, right in zip(positions, positions[1:]))):
                    return (False, f'    rank {rank} token {token} indices must follow logical position order')
                actual_set = set(actual_valid.tolist())
                expected_set = set(expected_valid.tolist())
                if actual_set == expected_set:
                    continue
                missing = sorted(expected_set - actual_set)
                extra = sorted(actual_set - expected_set)
                if scores is None:
                    score_outputs = actual_outputs if mode == 'full' else expected_outputs
                    scores = score_fn(mode, rank, score_outputs, inputs)
                missing_logical = torch.tensor([physical_to_logical[row] for row in missing], dtype=torch.int64)
                extra_logical = torch.tensor([physical_to_logical[row] for row in extra], dtype=torch.int64)
                missing_scores = torch.sort(scores[token, missing_logical], descending=True).values
                extra_scores = torch.sort(scores[token, extra_logical], descending=True).values
                tolerance = 5e-05 + 0.015 * missing_scores.abs()
                score_gap = missing_scores - extra_scores
                if torch.all(score_gap <= tolerance):
                    continue
                worst = int(torch.argmax(score_gap - tolerance))
                return (False, f'    rank {rank} token {token} top-k cutoff differs: missing score {float(missing_scores[worst]):.8g}, replacement score {float(extra_scores[worst]):.8g}, tolerance {float(tolerance[worst]):.8g}')
        return (True, '')
    compare.__name__ = 'numerical_cutoff_topk_compare'
    return compare


def wrap_attention_compare(compare):
    """Present HC workspace names through the attention-only comparator ABI."""

    def compare_hc(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
        attention_inputs = {**inputs, "x": expected_outputs["attn_input"]}
        attention_actual = {**actual_outputs, "output": actual_outputs["attn_out"]}
        attention_expected = {**expected_outputs, "output": expected_outputs["attn_out"]}
        return compare(
            actual,
            expected,
            inputs=attention_inputs,
            actual_outputs=attention_actual,
            expected_outputs=attention_expected,
            **kwargs,
        )

    return compare_hc


def golden_prefill_attn_c1a_full(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_a_scale: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_cache_scale: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    compressed_lens: torch.Tensor,
    index_cache: torch.Tensor,
    index_cache_scale: torch.Tensor,
    index_block_table: torch.Tensor,
    compressed_rope_cos: torch.Tensor,
    compressed_rope_sin: torch.Tensor,
    compressor_wkv: torch.Tensor,
    compressor_norm_weight: torch.Tensor,
    compressed_slots: torch.Tensor,
    index_wk: torch.Tensor,
    index_norm_weight: torch.Tensor,
    index_wq_b: torch.Tensor,
    index_wq_b_scale: torch.Tensor,
    index_weights_proj: torch.Tensor,
) -> dict:
    """Return one rank's FP32 partial output and reference cache state."""
    return golden_compressed_attention(
        mode='full', ratio=1,
        x=x,
        wq_a=wq_a, wq_a_scale=wq_a_scale, q_norm_weight=q_norm_weight, wq_b=wq_b, wq_b_scale=wq_b_scale,
        wkv=wkv, wkv_scale=wkv_scale, kv_norm_weight=kv_norm_weight, attn_sink=attn_sink,
        wo_a=wo_a, wo_a_scale=wo_a_scale, wo_b=wo_b, wo_b_scale=wo_b_scale,
        rope_cos=rope_cos, rope_sin=rope_sin,
        window_slots=window_slots, window_indices=window_indices, window_cache=window_cache, window_cache_scale=window_cache_scale,
        compressed_cache=compressed_cache, compressed_cache_scale=compressed_cache_scale, compressed_indices=None,
        compressor_wkv=compressor_wkv, compressor_wgate=None, compressor_norm_weight=compressor_norm_weight,
        state_block_table=None, state_cache=None,
        compressed_slots=compressed_slots, position_ids=None, compressed_lens=compressed_lens, compressed_rope_cos=compressed_rope_cos, compressed_rope_sin=compressed_rope_sin,
        index_wk=index_wk, index_norm_weight=index_norm_weight, index_wq_b=index_wq_b, index_wq_b_scale=index_wq_b_scale, index_weights_proj=index_weights_proj,
        index_cache=index_cache, index_cache_scale=index_cache_scale, index_block_table=index_block_table, token_to_req_indices=token_to_req_indices, candidate_mask=None,
        indexer_fn=golden_prefill_indexer, index_key_fn=golden_index_key,
        pool_selector_fn=golden_select_candidate_block_pool,
        attention_fn=golden_prefill_c1a_attention, sequential_query=True, round_cache_to_even=True, output_dtype=torch.float32,
    )


@pl.jit.inline(auto_scope=False)
def prefill_publish_decoder(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    num_tokens: pl.Scalar[pl.INT32],
):
    tokens = pl.tensor.dim(x, 0)
    compressed_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    projection_ready = pl.system.task_dummy(deps=[])
    compressed_projection, projection_done = c1a_compressor_project(
        x, compressor_wkv, compressed_projection, num_tokens, projection_ready,
    )
    compressed_latent = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    norm_blocks = (num_tokens + 7) // 8
    for norm_worker in pl.spmd(pl.max(1, norm_blocks), name_hint="attention_rmsnorm", deps=[projection_done]):
        if norm_worker < norm_blocks:
            rms_norm(compressed_projection, compressor_norm_weight, compressed_latent, norm_worker, num_tokens, width=HEAD_DIM)
    compressed_rotated = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    compressed_rotated = compressor_rope(
        compressed_latent, compressed_rope_cos, compressed_rope_sin, compressed_rotated, num_tokens,
    )
    compressed_ready = c1a_compressor_publish(
        compressed_rotated, compressed_slots, compressed_cache, compressed_cache_scale,
        num_tokens, projection_ready,
    )
    index_ready = publish_c1a_index_key(
        compressed_latent, index_wk, index_norm_weight, compressed_rope_cos, compressed_rope_sin,
        compressed_slots, index_cache, index_cache_scale, num_tokens,
    )
    ready = pl.system.task_dummy(deps=[compressed_ready, index_ready])
    return ready


RANKS_DYN = pl.dynamic("RANKS_DYN")


@pl.jit
def l2_prefill_c1a_full_sharded(
    x_hc: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.InOut[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.InOut[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[T_DYN], pl.INT32]],
    compressed_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32]],
    input_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    input_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    num_tokens: pl.Tensor[[1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, SP_T_DYN)
    pre_mix.bind_dynamic(0, SP_T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    window_slots.bind_dynamic(0, T_DYN)
    window_indices.bind_dynamic(0, T_DYN)
    window_cache.bind_dynamic(0, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(0, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(0, CMP_BLOCKS_DYN)
    compressed_cache_scale.bind_dynamic(0, CMP_BLOCKS_DYN)
    token_to_req_indices.bind_dynamic(0, T_DYN)
    compressed_lens.bind_dynamic(0, T_DYN)
    index_cache.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_cache_scale.bind_dynamic(0, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(0, B_DYN)
    index_block_table.bind_dynamic(1, TABLE_DYN)
    compressed_rope_cos.bind_dynamic(0, T_DYN)
    compressed_rope_sin.bind_dynamic(0, T_DYN)
    compressed_slots.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    pool_block_ids.bind_dynamic(0, T_DYN)
    pool_block_counts.bind_dynamic(0, T_DYN)
    pool_block_ids.bind_dynamic(1, INDEX_POOL_BLOCKS_DYN)
    compressed_indices.bind_dynamic(0, T_DYN)
    attn_input.bind_dynamic(0, T_DYN)
    attn_output.bind_dynamic(0, SP_T_DYN)
    next_pre_mix.bind_dynamic(0, SP_T_DYN)
    x_hc_out.bind_dynamic(0, SP_T_DYN)
    active = pl.read(num_tokens, [0])
    tokens = pl.tensor.dim(x_hc, 0)
    group_tokens = pl.tensor.dim(attn_input, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    local_hidden = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hc_collapsed, next_pre_mix, post_mix, residual_mix,
    )
    hc_norm_tokens = pl.tensor.dim(hc_collapsed, 0)
    for hc_norm_block in pl.spmd((hc_norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(hc_collapsed, attn_norm_weight, local_hidden, hc_norm_block, hc_norm_tokens, output_fp32=False)
    prefill_sp_input_allgather(
        local_hidden, input_window, input_arrived, attn_input, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE,
        active, attention_epoch,
    )
    for row in pl.spmd(group_tokens, name_hint='c1a_sp_inactive_topk'):
        if row >= active:
            topk_indices[row:row + 1, :] = pl.full([1, INDEX_TOPK], dtype=pl.INT32, value=-1)
    prefill_c1a_full_attention(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm_weight,
        attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin, window_slots, window_indices, window_cache,
        window_cache_scale, compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos, compressed_rope_sin,
        compressor_wkv, compressor_norm_weight, compressed_slots, index_wk, index_norm_weight, index_wq_b,
        index_wq_b_scale, index_weights_proj, topk_indices, pool_block_ids, pool_block_counts, output_window, output_arrived,
        attn_output, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE, active, attention_epoch,
        scatter_output=True,
    )
    prefill_sp_post(attn_output, x_hc, post_mix, residual_mix, x_hc_out, next_pre_mix, rank % TP_SIZE, active)
    retire_prefill_signals(x_hc_out, input_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    retire_prefill_signals(x_hc_out, output_arrived, rank // TP_SIZE * TP_SIZE, rank % TP_SIZE)
    return x_hc_out


@pl.jit.host
def l3_prefill_c1a_full_sharded(
    x_hc: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[RANKS_DYN, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[RANKS_DYN, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[RANKS_DYN, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[RANKS_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[RANKS_DYN, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[RANKS_DYN, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[RANKS_DYN, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[RANKS_DYN, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[RANKS_DYN, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[RANKS_DYN, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[RANKS_DYN, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]],
    compressed_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16]],
    token_to_req_indices: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32],
    index_cache: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]],
    index_cache_scale: pl.InOut[pl.Tensor[[RANKS_DYN, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0]],
    index_block_table: pl.Tensor[[RANKS_DYN, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[RANKS_DYN, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[RANKS_DYN, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[RANKS_DYN, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[RANKS_DYN, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[RANKS_DYN, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[RANKS_DYN, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[RANKS_DYN, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[RANKS_DYN, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0],
    index_weights_proj: pl.Tensor[[RANKS_DYN, D, INDEX_H], pl.BF16],
    topk_indices: pl.InOut[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.InOut[pl.Tensor[[RANKS_DYN, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.InOut[pl.Tensor[[RANKS_DYN, T_DYN], pl.INT32]],
    compressed_indices: pl.Tensor[[RANKS_DYN, T_DYN, INDEX_TOPK], pl.INT32],
    attn_input: pl.Out[pl.Tensor[[RANKS_DYN, T_DYN, D], pl.BF16]],
    attn_output: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, D], pl.BF16]],
    next_pre_mix: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT], pl.FP32]],
    x_hc_out: pl.Out[pl.Tensor[[RANKS_DYN, SP_T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Tensor[[RANKS_DYN, 1], pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    x_hc.bind_dynamic(0, RANKS_DYN)
    x_hc.bind_dynamic(1, SP_T_DYN)
    pre_mix.bind_dynamic(1, SP_T_DYN)
    rope_cos.bind_dynamic(1, T_DYN)
    rope_sin.bind_dynamic(1, T_DYN)
    window_slots.bind_dynamic(1, T_DYN)
    window_indices.bind_dynamic(1, T_DYN)
    window_cache.bind_dynamic(1, ORI_BLOCKS_DYN)
    window_cache_scale.bind_dynamic(1, ORI_BLOCKS_DYN)
    compressed_cache.bind_dynamic(1, CMP_BLOCKS_DYN)
    compressed_cache_scale.bind_dynamic(1, CMP_BLOCKS_DYN)
    token_to_req_indices.bind_dynamic(1, T_DYN)
    compressed_lens.bind_dynamic(1, T_DYN)
    index_cache.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_cache_scale.bind_dynamic(1, INDEX_BLOCKS_DYN)
    index_block_table.bind_dynamic(1, B_DYN)
    index_block_table.bind_dynamic(2, TABLE_DYN)
    compressed_rope_cos.bind_dynamic(1, T_DYN)
    compressed_rope_sin.bind_dynamic(1, T_DYN)
    compressed_slots.bind_dynamic(1, T_DYN)
    topk_indices.bind_dynamic(1, T_DYN)
    pool_block_ids.bind_dynamic(1, T_DYN)
    pool_block_counts.bind_dynamic(1, T_DYN)
    pool_block_ids.bind_dynamic(2, INDEX_POOL_BLOCKS_DYN)
    compressed_indices.bind_dynamic(1, T_DYN)
    attn_input.bind_dynamic(1, T_DYN)
    attn_output.bind_dynamic(1, SP_T_DYN)
    next_pre_mix.bind_dynamic(1, SP_T_DYN)
    x_hc_out.bind_dynamic(1, SP_T_DYN)
    input_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    input_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    output_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    output_signal_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        input_window = pld.window(input_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        input_arrived = pld.window(input_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        output_window = pld.window(output_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        output_arrived = pld.window(output_signal_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = index_wq_b_scale[rank]
        l2_prefill_c1a_full_sharded(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank], wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank], wo_b[rank],
            wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank], window_indices[rank],
            window_cache[rank], window_cache_scale[rank], compressed_cache[rank], compressed_cache_scale[rank],
            token_to_req_indices[rank], compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank], index_wk[rank],
            index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r, index_weights_proj[rank],
            topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank], compressed_indices[rank], attn_input[rank],
            attn_output[rank], next_pre_mix[rank], x_hc_out[rank], input_window, input_arrived, output_window,
            output_arrived, num_tokens[rank], rank, attention_epoch, device=rank,
        )


STATE_NAMES = {'full': ('window_cache', 'window_cache_scale', 'compressed_cache', 'compressed_cache_scale', 'index_cache', 'index_cache_scale', 'topk_indices', "pool_block_ids", "pool_block_counts"), 'reindex': ('window_cache', 'window_cache_scale', 'topk_indices'), 'reuse': ('window_cache', 'window_cache_scale')}

TOKEN_NAMES = {'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'compressed_indices', 'token_to_req_indices', 'compressed_lens', "pool_block_ids", "pool_block_counts", 'compressed_rope_cos', 'compressed_rope_sin', 'compressed_slots'}

def c1a_sp_reference_attention(mode, tensors, x, ranks, *, reference):
    """Run the independent C1A reference in full token order before sharding."""
    names = tuple(name for name, parameter in inspect.signature(reference).parameters.items()
                  if parameter.default is inspect.Parameter.empty)
    active = int(tensors['num_tokens'][ranks[0], 0])
    partials, results = ([], [])
    for rank in ranks:
        values = {name: x[:active] if name == 'x' else tensors[name][rank] for name in names}
        for name in TOKEN_NAMES.intersection(values):
            values[name] = values[name][:active]
        result = {name: tensors[name][rank].clone() for name in STATE_NAMES[mode]}
        partial = torch.zeros_like(x, dtype=torch.float32)
        if mode != 'reuse':
            result['topk_indices'][active:] = -1
        if mode == 'full':
            result['pool_block_ids'].fill_(-1)
            result['pool_block_counts'].zero_()
        if active:
            computed = reference(**values)
            partial[:active] = computed["output"]
            for name in STATE_NAMES[mode]:
                if name == 'topk_indices':
                    result[name][:active] = computed["topk_indices"]
                elif name in ('pool_block_ids', 'pool_block_counts'):
                    result[name][:active] = computed[name]
                else:
                    result[name] = computed[name]
        partials.append(partial)
        results.append(result)
    return (sum(partials).bfloat16(), results)

def build_c1a_sp_specs(args, initial_state):
    """Make distinct DP requests, contiguous local residuals, and global-order metadata."""

    fixture = make_fixture_values(args.tokens, args.case)
    tokens = fixture['x'].shape[1]
    world = TP_SIZE * args.dp
    local = (tokens + TP_SIZE - 1) // TP_SIZE
    if any((count < 0 or count > tokens for count in args.dp_tokens)):
        raise ValueError(f'active counts must be in [0, {tokens}]')
    values = {}
    for name in LEAF_NAMES:
        if name == 'topk_indices':
            value = torch.full((TP_SIZE, tokens, INDEX_TOPK), -1, dtype=torch.int32)
        else:
            value = fixture[name]
        values[name] = torch.cat([value.view(torch.uint8)] * args.dp).view(value.dtype)
    hc_keys = HC_INPUT_NAMES
    seed = getattr(args, "seed", 17)
    groups = [make_hc_inputs(tokens, seed + group * 7919, 'mixed') for group in range(args.dp)]
    for name in hc_keys:
        if name in ('x_hc', 'pre_mix'):
            values[name] = torch.cat([token_shards(group[name], local) for group in groups])
        else:
            values[name] = torch.stack([group[name] for group in groups]).repeat_interleave(TP_SIZE, 0)
    for rank in range(world):
        active = args.dp_tokens[rank // TP_SIZE]
        valid = max(0, min(local, active - rank % TP_SIZE * local))
        values['x_hc'][rank, valid:] = 17
        for name in ('window_slots', 'compressed_slots', 'window_indices', 'compressed_indices'):
            values[name][rank, active:] = -1
        values['pool_block_ids'][rank, active:] = -1
        values['pool_block_counts'][rank, active:] = 0
        values['compressed_lens'][rank, active:] = 0
    initial_state.update({name: values[name].clone() for name in set().union(*STATE_NAMES.values())})
    specs = [TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name], resident='stacked') for name in hc_keys + LEAF_NAMES]
    for name, shape, dtype in (('attn_input', [tokens, D], torch.bfloat16), ('attn_output', [local, D], torch.bfloat16), ('next_pre_mix', [local, HC_MULT], torch.float32), ('x_hc_out', [local, HC_MULT, D], torch.float32)):
        specs.append(TensorSpec(name, [world, *shape], dtype, resident='stacked'))
    counts = torch.tensor(args.dp_tokens, dtype=torch.int32).repeat_interleave(TP_SIZE).reshape(-1, 1)
    specs.append(TensorSpec('num_tokens', [world, 1], torch.int32, init_value=counts, resident='stacked'))
    specs.append(ScalarSpec(
        'attention_epoch', torch.int32, 1, compile_runtime=True,
        benchmark_step=None,
    ))
    return specs

def make_c1a_sp_compare(mode, initial_state, world_size, tokens):
    """Compare every shard's outputs and cache side effects with the default comparators."""
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)

    def group_values(values, base, active):
        grouped = {}
        for name, value in values.items():
            if name == 'num_tokens':
                grouped[name] = active
                continue
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                grouped[name] = value
                continue
            value = value[base:base + TP_SIZE]
            if name in TOKEN_NAMES or name in ('x', 'topk_indices'):
                value = value[:, :active]
            grouped[name] = value
        return grouped

    def selected_topk_compare(actual, expected, *, inputs, actual_outputs, expected_outputs, **kwargs):
        """Apply the tied-cutoff Top-K contract to every DP group's active rows."""
        check = topk_indices_compare(mode, golden_index_scores)
        values = {**initial_state, **inputs, "x": actual_outputs["attn_input"]}
        for base in range(0, actual.shape[0], TP_SIZE):
            active = int(inputs["num_tokens"][base, 0])
            group = slice(base, base + TP_SIZE)
            if not bool((actual[group, active:] == -1).all()):
                return False, f"DP group {base // TP_SIZE} has invalid Top-K padding"
            if not active:
                continue
            passed, detail = check(
                actual[group, :active],
                expected[group, :active],
                inputs=group_values(values, base, active),
                actual_outputs=group_values(actual_outputs, base, active),
                expected_outputs=group_values(expected_outputs, base, active),
                rtol=kwargs.get("rtol", 0),
                atol=kwargs.get("atol", 0),
            )
            if not passed:
                return False, f"DP group {base // TP_SIZE}: {detail}"
        return True, "tied-cutoff Top-K per DP group; exact inactive padding"

    compares = {
        "attn_input": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
        "attn_output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
        "x_hc_out": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
        "window_cache": mapped_pool_ratio_allclose(
            "window_slots", pool_name="window_cache",
            mapping_shape=(world_size, tokens),
            block_size=BLOCK_SIZE, leading_rank_axis=True,
            atol=0.0, rtol=0.0, max_error_ratio=0.0,
        ),
        "window_cache_scale": mapped_pool_ratio_allclose(
            "window_slots", pool_name="window_cache_scale",
            mapping_shape=(world_size, tokens),
            block_size=BLOCK_SIZE, leading_rank_axis=True,
            atol=0.0, rtol=0.0, max_error_ratio=0.0,
        ),
    }
    if mode == "full":
        compares.update({
            "compressed_cache": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="compressed_cache",
                mapping_shape=(world_size, tokens),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "compressed_cache_scale": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="compressed_cache_scale",
                mapping_shape=(world_size, tokens),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "index_cache": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="index_cache",
                mapping_shape=(world_size, tokens),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "index_cache_scale": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="index_cache_scale",
                mapping_shape=(world_size, tokens),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
        })
    for name in STATE_NAMES[mode]:
        if name == "topk_indices":
            compares[name] = selected_topk_compare
        elif "cache" not in name:
            compares[name] = exact
    for name in ("compressed_cache", "compressed_cache_scale", "index_cache", "index_cache_scale", "topk_indices", "pool_block_ids", "pool_block_counts"):
        if name not in STATE_NAMES[mode]:
            compares[name] = exact
    return compares


def golden_c1a_sp(tensors, *, mode, reference):
    """Evaluate attention and mHC outputs with per-rank cache state."""
    local = tensors["x_hc"].shape[1]
    tokens = tensors["attn_input"].shape[1]
    for base in range(0, tensors["x_hc"].shape[0], TP_SIZE):
        ranks = range(base, base + TP_SIZE)
        group = slice(base, base + TP_SIZE)
        active = int(tensors["num_tokens"][base, 0])
        x_hc = tensors["x_hc"][group].flatten(0, 1)[:tokens].clone()
        x_hc[active:] = 0
        incoming = tensors["pre_mix"][group].flatten(0, 1)[:tokens]
        pre, post, comb = golden_mhc_mixes(
            x_hc, tensors["hc_attn_fn"][base], tensors["hc_attn_scale"][base], tensors["hc_attn_base"][base]
        )
        collapsed = golden_hc_collapse(x_hc, incoming).to(torch.bfloat16)
        x = golden_rms_norm(collapsed, tensors["attn_norm_weight"][base])
        x[active:] = 0
        attention, results = c1a_sp_reference_attention(mode, tensors, x, ranks, reference=reference)
        for rank, result in zip(ranks, results):
            for name in STATE_NAMES[mode]:
                tensors[name][rank].copy_(result[name])
        output = golden_hc_post(attention, x_hc, post, comb)
        output[active:] = 0
        pre[active:] = 0
        tensors["attn_input"][group].copy_(x.expand_as(tensors["attn_input"][group]))
        for name, value in (("attn_output", attention), ("next_pre_mix", pre), ("x_hc_out", output)):
            tensors[name][group].copy_(token_shards(value, local))


def validate_c1a_sp(mode, args, fn, reference):
    if args.tp != TP_SIZE:
        raise ValueError(f"--tp was parsed as TP{TP_SIZE}, got --tp {args.tp}")

    args.mode = mode
    args.bench = os.environ.get("PYPTO_BENCH", "0") == "1"
    fixture_tokens = {"mixed": 8, "multi_1k": 3, "long": 3}.get(args.case, args.tokens)
    active = getattr(args, "active_tokens", None)
    active = fixture_tokens if active is None else active
    args.dp_tokens = ([int(value) for value in args.dp_tokens.split(",")] if args.dp_tokens else [active] * args.dp)
    if len(args.dp_tokens) != args.dp:
        raise ValueError("provide one active count per DP group")
    if not 1 <= args.tokens <= CASE_MAX_TOKENS:
        raise ValueError(f"tokens must be in [1, {CASE_MAX_TOKENS}]")
    devices = [int(value) for value in args.device.split(",")]
    if len(devices) != TP_SIZE * args.dp or len(set(devices)) != len(devices):
        raise ValueError("device IDs must name TP*DP distinct devices")
    initial_state = {}
    specs = build_c1a_sp_specs(args, initial_state)
    return run(
        fn=fn,
        specs=specs,
        golden_data=getattr(args, "golden_data", None),
        runtime_dir=getattr(args, "runtime_dir", None),
        save_data=getattr(args, "save_data", False),
        golden_fn=functools.partial(golden_c1a_sp, mode=mode, reference=reference),
        compare_fn=make_c1a_sp_compare(mode, initial_state, len(devices), fixture_tokens),
        compile_only=args.compile_only,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0)),
    )


@pl.jit.inline(auto_scope=False)
def prefill_c1a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    window_cache_scale: pl.Tensor[
        [ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16
    ],
    compressed_cache: pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8],
    compressed_cache_scale: pl.Tensor[
        [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8],
    index_cache_scale: pl.Tensor[
        [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    hidden: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_input: pl.Tensor[[T_DYN, D], pl.BF16],
    attn_out: pl.Tensor[[T_DYN, D], pl.BF16],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    """Run official mHC mixes, collapse, attention RMSNorm, C1A, then mHC post."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        hidden, next_pre_mix, post_mix, residual_mix,
    )
    norm_tokens = pl.tensor.dim(hidden, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(hidden, attn_norm_weight, attn_input, norm_block, norm_tokens, output_fp32=False)
    prefill_c1a_full_attention(
        attn_input, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos,
        compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, pool_block_ids, pool_block_counts, output_window, output_arrived, attn_out,
        group_base, tp_rank, num_tokens, attention_epoch,
        scatter_output=False,
    )
    hc_post_deps = pl.array.create(2, pl.TASK_ID)
    hc_post(attn_out, x_hc, post_mix, residual_mix, output, deps=hc_post_deps)
    return output


@pl.jit
def l2_prefill_c1a_full(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[T_DYN], pl.INT64],
    window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]],
    window_cache_scale: pl.InOut[
        pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[
        pl.Tensor[[CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[
            [CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ]
    ],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    index_cache: pl.InOut[
        pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[
            [INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
        ]
    ],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    compressor_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN, pl.NZ],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    topk_indices: pl.Out[pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]],
    pool_block_counts: pl.Out[pl.Tensor[[T_DYN], pl.INT32]],
    output_window: pld.DistributedTensor[[PREFILL_MAX_TOKENS, D], pl.BF16],
    output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    next_pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run one TP rank of the delayed-mix full C1A path."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    prefill_c1a_full(
        x_hc, pre_mix, hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_weight,
        wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale, wkv, wkv_scale,
        kv_norm_weight, attn_sink, wo_a, wo_a_scale, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
        compressed_cache, compressed_cache_scale, token_to_req_indices, compressed_lens,
        index_cache, index_cache_scale, index_block_table, compressed_rope_cos,
        compressed_rope_sin, compressor_wkv, compressor_norm_weight, compressed_slots,
        index_wk, index_norm_weight, index_wq_b, index_wq_b_scale, index_weights_proj,
        topk_indices, pool_block_ids, pool_block_counts, output_window, output_arrived,
        hidden, attn_input, attn_out, output, next_pre_mix, 0, tp_rank, num_tokens, 1,
    )
    retire_prefill_signals(output, output_arrived, 0, tp_rank)
    return output


@pl.jit.host
def l3_prefill_c1a_full(
    x_hc: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[TP_SIZE, D // 32, Q_LORA], pl.FP8E8M0],
    q_norm_weight: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[TP_SIZE, Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[TP_SIZE, D // 32, HEAD_DIM], pl.FP8E8M0],
    kv_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[TP_SIZE, LOCAL_H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN, O_LORA], pl.FP8E4M3FN],
    wo_a_scale: pl.Tensor[[TP_SIZE, LOCAL_O_GROUPS, O_GROUP_IN // 32, O_LORA], pl.FP8E8M0],
    wo_b: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
    wo_b_scale: pl.Tensor[[TP_SIZE, LOCAL_O_WIDTH // 32, D], pl.FP8E8M0],
    rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    window_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    window_indices: pl.Tensor[[TP_SIZE, T_DYN, 128], pl.INT32],
    window_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN]
    ],
    window_cache_scale: pl.InOut[
        pl.Tensor[[TP_SIZE, ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], pl.BF16]
    ],
    compressed_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // 2], pl.UINT8]
    ],
    compressed_cache_scale: pl.InOut[
        pl.Tensor[
            [TP_SIZE, CMP_BLOCKS_DYN, 128, 1, HEAD_DIM // COMPRESSED_CACHE_GROUP], pl.BF16
        ]
    ],
    token_to_req_indices: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[TP_SIZE, T_DYN], pl.INT32],
    index_cache: pl.InOut[
        pl.Tensor[[TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // 2], pl.UINT8]
    ],
    index_cache_scale: pl.InOut[
        pl.Tensor[
            [TP_SIZE, INDEX_BLOCKS_DYN, 128, 1, INDEX_DIM // INDEX_CACHE_GROUP], pl.FP8E8M0
        ]
    ],
    index_block_table: pl.Tensor[[TP_SIZE, B_DYN, TABLE_DYN], pl.INT32],
    compressed_rope_cos: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[TP_SIZE, T_DYN, ROPE_DIM // 2], pl.FP32],
    compressor_wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    compressor_norm_weight: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compressed_slots: pl.Tensor[[TP_SIZE, T_DYN], pl.INT64],
    index_wk: pl.Tensor[[TP_SIZE, HEAD_DIM, INDEX_DIM], pl.BF16],
    index_norm_weight: pl.Tensor[[TP_SIZE, INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[TP_SIZE, Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [TP_SIZE, Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0
    ],
    index_weights_proj: pl.Tensor[[TP_SIZE, D, INDEX_H], pl.BF16],
    topk_indices: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, INDEX_TOPK], pl.INT32]],
    pool_block_ids: pl.Out[
        pl.Tensor[[TP_SIZE, T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32]
    ],
    pool_block_counts: pl.Out[
        pl.Tensor[[TP_SIZE, T_DYN], pl.INT32]
    ],
    next_pre_mix: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT], pl.FP32]],
    hidden: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    attn_input: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    attn_out: pl.InOut[pl.Tensor[[TP_SIZE, T_DYN, D], pl.BF16]],
    output: pl.Out[pl.Tensor[[TP_SIZE, T_DYN, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    output_window_buf = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
    output_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        output_window = pld.window(output_window_buf, [PREFILL_MAX_TOKENS, D], dtype=pl.BF16)
        output_arrived = pld.window(output_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
        wq_a_scale_r: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = wq_a_scale[rank]
        wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = wq_b_scale[rank]
        wkv_scale_r: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = wkv_scale[rank]
        wo_b_scale_r: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = wo_b_scale[rank]
        index_wq_b_scale_r: pl.Tensor[
            [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ] = index_wq_b_scale[rank]
        l2_prefill_c1a_full(
            x_hc[rank], pre_mix[rank], hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_weight[rank],
            wq_a[rank], wq_a_scale_r, q_norm_weight[rank], wq_b[rank], wq_b_scale_r,
            wkv[rank], wkv_scale_r, kv_norm_weight[rank], attn_sink[rank], wo_a[rank], wo_a_scale[rank],
            wo_b[rank], wo_b_scale_r, rope_cos[rank], rope_sin[rank], window_slots[rank],
            window_indices[rank], window_cache[rank], window_cache_scale[rank],
            compressed_cache[rank], compressed_cache_scale[rank], token_to_req_indices[rank],
            compressed_lens[rank], index_cache[rank], index_cache_scale[rank],
            index_block_table[rank], compressed_rope_cos[rank], compressed_rope_sin[rank],
            compressor_wkv[rank], compressor_norm_weight[rank], compressed_slots[rank],
            index_wk[rank], index_norm_weight[rank], index_wq_b[rank], index_wq_b_scale_r,
            index_weights_proj[rank], topk_indices[rank], pool_block_ids[rank], pool_block_counts[rank], output_window,
            output_arrived, next_pre_mix[rank], hidden[rank], attn_input[rank], attn_out[rank], output[rank],
            rank, num_tokens, device=rank,
        )


def build_hc_tensor_specs(token_count=32, case_name=CASE_DEFAULT, active_tokens=None):
    """Build C1A inputs plus a pre-mix produced by the preceding mHC block."""
    values = make_fixture_values(token_count, case_name)
    tokens = values["x"].shape[1]
    active_tokens = tokens if active_tokens is None else active_tokens
    if not 0 <= active_tokens <= tokens:
        raise ValueError(f"active_tokens must be in [0, {tokens}], got {active_tokens}")
    generator = torch.Generator().manual_seed(2026)
    x_hc = torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16().float()
    previous_x_hc = torch.randn(tokens, HC_MULT, D, generator=generator).bfloat16().float()
    hc_attn_fn = torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)
    hc_attn_scale = torch.randn(3, generator=generator)
    hc_attn_base = torch.randn(MIX_HC, generator=generator)
    pre_mix, _, _ = golden_mhc_mixes(previous_x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    hc_values = {
        "x_hc": x_hc.unsqueeze(0).repeat(TP_SIZE, 1, 1, 1),
        "pre_mix": pre_mix.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        "hc_attn_fn": hc_attn_fn.unsqueeze(0).repeat(TP_SIZE, 1, 1),
        "hc_attn_scale": hc_attn_scale.unsqueeze(0).repeat(TP_SIZE, 1),
        "hc_attn_base": hc_attn_base.unsqueeze(0).repeat(TP_SIZE, 1),
        "attn_norm_weight": values["attn_norm_weight"],
    }
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value)
        for name, value in hc_values.items()
    ]
    specs += [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
        for name in FULL_INPUT_NAMES if name != "x"
    ]
    specs += [
        TensorSpec("topk_indices", [TP_SIZE, tokens, INDEX_TOPK], torch.int32),
        TensorSpec("pool_block_ids", list(values["pool_block_ids"].shape), torch.int32),
        TensorSpec("pool_block_counts", list(values["pool_block_counts"].shape), torch.int32),
        TensorSpec("next_pre_mix", [TP_SIZE, tokens, HC_MULT], torch.float32),
        TensorSpec("hidden", [TP_SIZE, tokens, D], torch.bfloat16),
        TensorSpec("attn_input", [TP_SIZE, tokens, D], torch.bfloat16),
        TensorSpec(
            "attn_out", [TP_SIZE, tokens, D], torch.bfloat16,
            init_value=torch.zeros(TP_SIZE, tokens, D, dtype=torch.bfloat16),
        ),
        TensorSpec("output", [TP_SIZE, tokens, HC_MULT, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, active_tokens),
    ]
    return specs


def golden_prefill_c1a_full_case(tensors):
    """Reference delayed pre-mix, current coefficient generation, attention, and post-mix."""
    next_pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"][0], tensors["hc_attn_fn"][0],
        tensors["hc_attn_scale"][0], tensors["hc_attn_base"][0],
    )
    hidden = golden_hc_collapse(tensors["x_hc"][0], tensors["pre_mix"][0]).to(torch.bfloat16)
    attn_input = golden_rms_norm(hidden.to(torch.bfloat16), tensors["attn_norm_weight"][0])
    active = int(tensors["num_tokens"])
    output = tensors["output"]
    tensors["x"] = attn_input.unsqueeze(0).expand(TP_SIZE, -1, -1)
    tensors["output"] = tensors["attn_out"]
    apply_distributed_golden("full", golden_prefill_attn_c1a_full, tensors, active)
    tensors["attn_out"][:, active:].zero_()
    tensors["attn_input"][:] = attn_input.unsqueeze(0)
    hc_output = golden_hc_post(tensors["attn_out"][0], tensors["x_hc"][0], post_mix, residual_mix)
    tensors["output"] = output
    tensors["next_pre_mix"][:] = next_pre_mix.unsqueeze(0)
    tensors["hidden"][:] = hidden.unsqueeze(0)
    tensors["output"][:] = hc_output.unsqueeze(0)


if __name__ == "__main__":
    import argparse

    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 prefill C1A full with mHC validation")
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a5"], default="a5")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=(1, 2, 4))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dp", type=int, default=1, choices=(1, 2))
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--case", default=CASE_DEFAULT, choices=CASE_NAMES)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--golden-data")
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--runtime-dir")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--wiring", choices=("replicated", "sharded"), default="replicated")
    parser.add_argument("--dp-tokens")
    args = parser.parse_args()
    if args.tp != TP_SIZE:
        parser.error(f"--tp was parsed as TP{TP_SIZE}, got --tp {args.tp}")
    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(TP_SIZE * args.dp))
    if args.wiring == "sharded":
        result = validate_c1a_sp("full", args, l3_prefill_c1a_full_sharded, golden_prefill_attn_c1a_full)
        if not result.passed:
            raise SystemExit(result.error or 1)
        raise SystemExit(0)
    if args.dp != 1:
        parser.error("replicated wiring requires --dp 1")
    if args.case == "causal" and not 1 <= args.tokens <= CASE_MAX_TOKENS:
        parser.error(f"--tokens must be in [1, {CASE_MAX_TOKENS}]")
    if args.active_tokens is not None and not 0 <= args.active_tokens <= args.tokens:
        parser.error(f"--active-tokens must be in [0, {args.tokens}]")
    devices = [int(device) for device in args.device.split(",")]
    if len(devices) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {devices}")
    result = run(
        fn=l3_prefill_c1a_full,
        specs=build_hc_tensor_specs(args.tokens, args.case, args.active_tokens),
        golden_fn=golden_prefill_c1a_full_case,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        config={
            "backend_type": args.backend_type, "execution_mode": args.execution_mode,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
            "dump_passes": args.dump_passes,
            "enable_chip_swimlane": args.enable_chip_swimlane,
            "ring_heap": PREFILL_ATTN_RING_HEAP,
        },
        compare_fn={
            "next_pre_mix": ratio_allclose(atol=2.5e-5, rtol=5e-3),
            "hidden": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "attn_input": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "attn_out": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "output": ratio_allclose(atol=1e-2, rtol=1.0 / 128),
            "window_cache": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "window_cache_scale": mapped_pool_ratio_allclose(
                "window_slots", pool_name="window_cache_scale",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "compressed_cache": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="compressed_cache",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "compressed_cache_scale": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="compressed_cache_scale",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "index_cache": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="index_cache",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "index_cache_scale": mapped_pool_ratio_allclose(
                "compressed_slots", pool_name="index_cache_scale",
                mapping_shape=(len(devices), fixture_token_count(args.tokens, args.case)),
                block_size=BLOCK_SIZE, leading_rank_axis=True,
                atol=0.0, rtol=0.0, max_error_ratio=0.0,
            ),
            "topk_indices": wrap_attention_compare(topk_indices_compare("full", golden_index_scores)),
        },
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
