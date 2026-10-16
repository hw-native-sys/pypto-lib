# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Metadata lowering and shared validation helpers for prefill and decode."""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, replace
import json
import math
from numbers import Integral
from pathlib import Path
import struct
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import pypto.language as pl
import torch


import config as C
from config import BLOCK_SIZE, FLASH as M, TP_SIZE
from golden import ScalarSpec, TensorSpec, ratio_allclose
from rope_tables import precompute_rope_tables


# Model config.
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc


@dataclass(frozen=True)
class ForwardMetadata:
    """Canonical token-major metadata consumed by all layer kernels."""

    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    logit_row_indices: torch.Tensor
    token_to_req_indices: torch.Tensor
    moe_token_owners: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    new_kv_seq_lens: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor
    compressed_slots: Mapping[int, torch.Tensor]
    index_slots: Mapping[int, torch.Tensor]
    state_block_tables: Mapping[int, torch.Tensor]
    compressed_seq_lens: Mapping[int, torch.Tensor]
    compressed_seq_remainders: Mapping[int, torch.Tensor]
    compressed_lens: Mapping[int, torch.Tensor]
    compressor_output_start_loc: Mapping[int, torch.Tensor]
    compressor_source_token_indices: Mapping[int, torch.Tensor]
    compressor_position_ids: Mapping[int, torch.Tensor]
    compressed_rope_position_ids: Mapping[int, torch.Tensor]


LOCAL_H = M.num_attention_heads // TP_SIZE

LOCAL_O_GROUPS = M.o_groups // TP_SIZE

LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

Q_LORA = M.q_lora_rank

O_LORA = M.o_lora_rank

O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups

INDEX_H = M.index_n_heads

INDEX_DIM = M.index_head_dim

INDEX_TOPK = M.index_topk

CASE_DEFAULT = 'multi_1k'

CASE_TOKENS = 64

CASE_MAX_TOKENS = 128

CASE_PAGE = 128

CASE_NAMES = (CASE_DEFAULT, 'causal', 'mixed', 'long')

TOPK_SCORE_ATOL = 5e-05

TOPK_SCORE_RTOL = 0.015

COMMON_INPUT_NAMES = ('x', 'wq_a', 'wq_a_scale', 'q_norm_weight', 'wq_b', 'wq_b_scale', 'wkv', 'wkv_scale', 'kv_norm_weight', 'attn_sink', 'wo_a', 'wo_b', 'wo_b_scale', 'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'window_cache', 'window_cache_scale', 'compressed_cache', 'compressed_cache_scale')


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
        request_ids = torch.arange(3, dtype=torch.int32)
        compressed_lens = torch.full((3,), 1024, dtype=torch.int32)
    elif case_name == 'causal':
        if not 1 <= token_count <= CASE_MAX_TOKENS:
            raise ValueError(f'token_count must be in [1, {CASE_MAX_TOKENS}], got {token_count}')
        request_ids = torch.zeros(token_count, dtype=torch.int32)
        compressed_lens = torch.arange(1, token_count + 1, dtype=torch.int32)
    elif case_name == 'mixed':
        request_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
        compressed_lens = torch.tensor([1, 2, 3, 127, 128, 511, 512, 513], dtype=torch.int32)
    elif case_name == 'long':
        request_ids = torch.arange(3, dtype=torch.int32)
        compressed_lens = torch.tensor([1024, 4096, 8192], dtype=torch.int32)
    else:
        raise ValueError(f'unknown case {case_name!r}; expected one of {CASE_NAMES}')
    request_count = int(request_ids.max().item()) + 1
    table_pages = (int(compressed_lens.max().item()) + CASE_PAGE - 1) // CASE_PAGE
    block_table = torch.arange(request_count * table_pages, dtype=torch.int32).reshape(request_count, table_pages)
    positions = compressed_lens.to(torch.int64) - 1
    pages = torch.div(positions, CASE_PAGE, rounding_mode='floor')
    offsets = positions.remainder(CASE_PAGE)
    slots = block_table[request_ids.to(torch.long), pages] * CASE_PAGE + offsets
    window_indices = torch.full((request_ids.numel(), CASE_PAGE), -1, dtype=torch.int32)
    compressed_indices = torch.full((request_ids.numel(), INDEX_TOPK), -1, dtype=torch.int32)
    candidate_mask = torch.zeros(request_ids.numel(), table_pages * CASE_PAGE, dtype=torch.uint8)
    for token in range(request_ids.numel()):
        request = int(request_ids[token])
        visible = int(compressed_lens[token])
        logical = torch.arange(visible, dtype=torch.int64)
        physical = block_table[request, logical // CASE_PAGE] * CASE_PAGE + logical % CASE_PAGE
        window_count = min(visible, CASE_PAGE)
        window_indices[token, :window_count] = physical[-window_count:].to(torch.int32)
        compressed_count = min(visible, INDEX_TOPK)
        compressed_indices[token, :compressed_count] = physical[:compressed_count].to(torch.int32)
        candidate_mask[token, :visible] = 1
    return (request_ids, compressed_lens, block_table, slots.to(torch.int64), window_indices, compressed_indices, candidate_mask)

def fixture_token_count(token_count: int, case_name: str) -> int:
    """Return the token rows a case fixture provides; only the causal case follows ``token_count``."""
    return int(_case_metadata(token_count, case_name)[0].numel())

def make_fixture_values(token_count: int, case_name: str) -> dict[str, torch.Tensor]:
    import torch
    from config import TP_SIZE, COMPRESSED_CACHE_GROUP, INDEX_CACHE_GROUP
    from quant import quantize_mxfp4_cache, quantize_mxfp8_cache
    torch.manual_seed(41)
    request_ids, compressed_lens, block_table, slots, window_indices, compressed_indices, candidate_mask = _case_metadata(token_count, case_name)
    token_count = request_ids.numel()
    cache_blocks = block_table.numel()
    wq_a, wq_a_scale = _ranked_fp8_weight(D, Q_LORA, shared=True)
    wq_b, wq_b_scale = _ranked_fp8_weight(Q_LORA, LOCAL_H * HEAD_DIM, shared=False)
    wkv, wkv_scale = _ranked_fp8_weight(D, HEAD_DIM, shared=True)
    wo_b, wo_b_scale = _ranked_fp8_weight(LOCAL_O_WIDTH, D, shared=False)
    index_wq_b, index_wq_b_scale = _ranked_fp8_weight(Q_LORA, INDEX_H * INDEX_DIM, shared=True)
    window_value = torch.randn(cache_blocks, CASE_PAGE, 1, HEAD_DIM).mul_(0.05)
    window_cache, window_cache_scale = quantize_mxfp8_cache(window_value)
    compressed_value = torch.randn(cache_blocks, CASE_PAGE, 1, HEAD_DIM).mul_(0.05)
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(compressed_value, group_size=COMPRESSED_CACHE_GROUP, scale_format='bf16')
    index_value = torch.randn(cache_blocks, CASE_PAGE, 1, INDEX_DIM).mul_(0.05)
    index_cache, index_cache_scale = quantize_mxfp4_cache(index_value, group_size=INDEX_CACHE_GROUP, scale_format='e8m0')
    index_cache_scale = index_cache_scale.view(torch.float8_e8m0fnu)
    angles = (compressed_lens - 1).to(torch.float32).unsqueeze(1)
    angles = angles * torch.arange(ROPE_DIM // 2, dtype=torch.float32).unsqueeze(0) * 0.001
    values = {'x': _ranked(torch.randn(token_count, D).to(torch.bfloat16)), 'wq_a': wq_a, 'wq_a_scale': wq_a_scale, 'q_norm_weight': _ranked(torch.ones(Q_LORA, dtype=torch.bfloat16)), 'wq_b': wq_b, 'wq_b_scale': wq_b_scale, 'wkv': wkv, 'wkv_scale': wkv_scale, 'kv_norm_weight': _ranked(torch.ones(HEAD_DIM, dtype=torch.bfloat16)), 'attn_sink': torch.zeros(TP_SIZE, LOCAL_H, dtype=torch.float32), 'wo_a': torch.randn(TP_SIZE, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN, dtype=torch.bfloat16).mul_(O_GROUP_IN ** (-0.5)), 'wo_b': wo_b, 'wo_b_scale': wo_b_scale, 'rope_cos': _ranked(torch.cos(angles)), 'rope_sin': _ranked(torch.sin(angles)), 'window_slots': _ranked(slots), 'window_indices': _ranked(window_indices), 'window_cache': _ranked(window_cache), 'window_cache_scale': _ranked(window_cache_scale), 'compressed_cache': _ranked(compressed_cache), 'compressed_cache_scale': _ranked(compressed_cache_scale), 'compressed_indices': _ranked(compressed_indices), 'request_ids': _ranked(request_ids), 'compressed_lens': _ranked(compressed_lens), 'index_cache': _ranked(index_cache), 'index_cache_scale': _ranked(index_cache_scale), 'index_block_table': _ranked(block_table), 'candidate_mask': _ranked(candidate_mask), 'compressed_rope_cos': _ranked(torch.cos(angles)), 'compressed_rope_sin': _ranked(torch.sin(angles)), 'compressor_wkv': _ranked(torch.randn(D, HEAD_DIM).mul_(D ** (-0.5)).to(torch.bfloat16)), 'compressor_norm_weight': _ranked(torch.ones(HEAD_DIM, dtype=torch.bfloat16)), 'compressed_slots': _ranked(slots), 'index_wk': _ranked(torch.randn(HEAD_DIM, INDEX_DIM).mul_(HEAD_DIM ** (-0.5)).to(torch.bfloat16)), 'index_norm_weight': _ranked(torch.ones(INDEX_DIM, dtype=torch.bfloat16)), 'index_wq_b': index_wq_b, 'index_wq_b_scale': index_wq_b_scale, 'index_weights_proj': _ranked(torch.randn(D, INDEX_H).mul_(D ** (-0.5)).to(torch.bfloat16))}
    values['attn_norm_weight'] = _ranked(torch.randn(D).mul_(0.1).add_(1).to(torch.bfloat16))
    return values

def make_tensor_specs(input_names: tuple[str, ...], output_names: tuple[str, ...], token_count: int=CASE_TOKENS, case_name: str=CASE_DEFAULT) -> list[TensorSpec | ScalarSpec]:
    import torch
    from golden import ScalarSpec, TensorSpec
    from config import TP_SIZE
    values = make_fixture_values(token_count, case_name)
    token_count = values['x'].shape[1]
    specs = [TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name]) for name in input_names]
    for name in output_names:
        if name == 'topk_indices':
            shape = [TP_SIZE, token_count, INDEX_TOPK]
            dtype = torch.int32
        elif name == 'candidate_mask':
            shape = list(values['candidate_mask'].shape)
            dtype = torch.uint8
        else:
            shape = [TP_SIZE, token_count, D]
            dtype = torch.bfloat16
        specs.append(TensorSpec(name, shape, dtype))
    specs.append(ScalarSpec('num_tokens', torch.int32, token_count))
    return specs

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

def golden_prefill_tp_attention(attention_fn: Callable, hidden: torch.Tensor, attention_args: tuple[torch.Tensor, ...], tp_size: int) -> tuple[torch.Tensor, object]:
    """Run one prefill attention golden per TP rank and reduce its FP32 partials.

    The exported mHC helpers accept either one rank's attention arguments or rank-stacked
    arguments (the first dimension is ``tp_size``).  Rank-stacked arguments reproduce the
    sharded invocation; the local form remains available for CPU-side smoke checks.
    """
    import torch
    ranked_args = bool(attention_args and isinstance(attention_args[0], torch.Tensor) and (attention_args[0].ndim >= 2) and (attention_args[0].shape[0] == tp_size))
    results = []
    for rank in range(tp_size):
        rank_hidden = hidden
        if hidden.ndim > 0 and hidden.shape[0] == tp_size and (hidden.ndim == 3):
            rank_hidden = hidden[rank]
        rank_args = attention_args
        if ranked_args:
            rank_args = tuple((argument[rank] for argument in attention_args))
        result = attention_fn(rank_hidden, *rank_args)
        results.append(result)
        output = result.output
        if output.ndim == rank_hidden.ndim + 1 and output.shape[0] == tp_size:
            partials = [output[peer] for peer in range(tp_size)]
            return (_reduce_tp_partials(partials), result)
    return (_reduce_tp_partials([result.output for result in results]), results[0])

def apply_distributed_golden(mode: str, golden_fn: Callable, tensors: dict[str, torch.Tensor], active_tokens: int | None=None) -> None:
    import torch
    from config import FLASH as M, TP_SIZE
    parameter_names = tuple(inspect.signature(golden_fn).parameters)
    token_input_names = {'x', 'rope_cos', 'rope_sin', 'window_slots', 'window_indices', 'request_ids', 'compressed_lens', 'compressed_rope_cos', 'compressed_rope_sin', 'compressed_slots', 'candidate_mask', 'compressed_indices'}
    results = []
    for rank in range(TP_SIZE):
        kwargs = {}
        for name in parameter_names:
            value = tensors[name][rank]
            if active_tokens is not None and name in token_input_names and (value.ndim > 0):
                value = value[:active_tokens]
            kwargs[name] = value
        results.append(golden_fn(**kwargs))
    reduced = _reduce_tp_partials([result.output for result in results])
    if active_tokens is None:
        tensors['output'][:] = reduced.unsqueeze(0)
    else:
        tensors['output'][:, :active_tokens] = reduced.unsqueeze(0)
    for rank, result in enumerate(results):
        tensors['window_cache'][rank].copy_(result.window_cache)
        _copy_storage(tensors['window_cache_scale'][rank], result.window_cache_scale)
        if mode == 'full':
            tensors['compressed_cache'][rank].copy_(result.compressed_cache)
            tensors['compressed_cache_scale'][rank].copy_(result.compressed_cache_scale)
            tensors['index_cache'][rank].copy_(result.index_cache)
            _copy_storage(tensors['index_cache_scale'][rank], result.index_cache_scale)
            candidate_mask = tensors['candidate_mask'][rank]
            candidate_mask.zero_()
            block_size = M.candidate_block_size
            for token in range(result.candidate_mask.shape[0]):
                selected = torch.nonzero(result.candidate_mask[token], as_tuple=False).flatten()
                for block in torch.unique(selected // block_size):
                    begin = int(block) * block_size
                    end = min(begin + block_size, candidate_mask.shape[1])
                    candidate_mask[token, begin:end] = 1
        if mode in ('full', 'reindex'):
            if active_tokens is None:
                tensors['topk_indices'][rank].copy_(result.topk_indices)
            else:
                tensors['topk_indices'][rank, :active_tokens].copy_(result.topk_indices)

def _golden_index_scores(mode: str, rank: int, cache_outputs: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    import torch
    from config import INDEX_CACHE_GROUP
    from hierarchical_sparse_indexer import golden_paged_indexer as paged_indexer
    from qkv_proj_rope import golden_qkv_proj_rope as qkv_proj_rope
    from quant import dequantize_mxfp4_cache
    index_payload = cache_outputs['index_cache'] if mode == 'full' else inputs['index_cache']
    index_scale = cache_outputs['index_cache_scale'] if mode == 'full' else inputs['index_cache_scale']
    index_value = dequantize_mxfp4_cache(index_payload[rank], index_scale[rank], group_size=INDEX_CACHE_GROUP, scale_format='e8m0').to(torch.bfloat16)
    _, _, query_latent = qkv_proj_rope(inputs['x'][rank], inputs['wq_a'][rank], inputs['wq_a_scale'][rank], inputs['q_norm_weight'][rank], inputs['wq_b'][rank], inputs['wq_b_scale'][rank], inputs['wkv'][rank], inputs['wkv_scale'][rank], inputs['kv_norm_weight'][rank], inputs['rope_cos'][rank], inputs['rope_sin'][rank])
    candidates = inputs['candidate_mask'][rank] if mode == 'reindex' else None
    scores, _ = paged_indexer(inputs['x'][rank], query_latent, inputs['request_ids'][rank], index_value, inputs['index_block_table'][rank], inputs['compressed_lens'][rank], inputs['index_wq_b'][rank], inputs['index_wq_b_scale'][rank], inputs['index_weights_proj'][rank], inputs['rope_cos'][rank], inputs['rope_sin'][rank], candidates=candidates, topk=INDEX_TOPK)
    return scores

def _physical_to_logical(inputs: dict[str, torch.Tensor], rank: int, token: int) -> dict[int, int]:
    import torch
    request = int(inputs['request_ids'][rank, token])
    visible = int(inputs['compressed_lens'][rank, token])
    logical = torch.arange(visible, dtype=torch.int64)
    table = inputs['index_block_table'][rank, request]
    physical = table[logical // CASE_PAGE].to(torch.int64) * CASE_PAGE + logical % CASE_PAGE
    return {int(row): index for index, row in enumerate(physical.tolist())}

def topk_indices_compare(mode: str) -> Callable:
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
                    scores = _golden_index_scores(mode, rank, score_outputs, inputs)
                missing_logical = torch.tensor([physical_to_logical[row] for row in missing], dtype=torch.int64)
                extra_logical = torch.tensor([physical_to_logical[row] for row in extra], dtype=torch.int64)
                missing_scores = torch.sort(scores[token, missing_logical], descending=True).values
                extra_scores = torch.sort(scores[token, extra_logical], descending=True).values
                tolerance = TOPK_SCORE_ATOL + TOPK_SCORE_RTOL * missing_scores.abs()
                score_gap = missing_scores - extra_scores
                if torch.all(score_gap <= tolerance):
                    continue
                worst = int(torch.argmax(score_gap - tolerance))
                return (False, f'    rank {rank} token {token} top-k cutoff differs: missing score {float(missing_scores[worst]):.8g}, replacement score {float(extra_scores[worst]):.8g}, tolerance {float(tolerance[worst]):.8g}')
        return (True, '')
    compare.__name__ = 'numerical_cutoff_topk_compare'
    return compare


def request_ids_from_starts(query_start_loc: torch.Tensor) -> torch.Tensor:
    """Expand packed cumulative query lengths into one request id per token."""
    lengths = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    return torch.repeat_interleave(torch.arange(lengths.numel(), device=lengths.device), lengths).to(
        torch.int32
    )


def paged_slots(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    storage_block_size: int = BLOCK_SIZE,
    logical_divisor: int = 1,
    publish_only_complete: bool = False,
) -> torch.Tensor:
    """Map logical token positions to flattened physical cache rows."""
    if positions.ndim != 1 or request_ids.shape != positions.shape or block_table.ndim != 2:
        raise ValueError("positions/request_ids must be matching vectors and block_table must be a matrix")
    if storage_block_size <= 0 or logical_divisor <= 0:
        raise ValueError("storage_block_size and logical_divisor must be positive")
    if any(value.dtype not in (torch.int32, torch.int64) for value in (positions, request_ids, block_table)):
        raise ValueError("positions, request_ids and block_table must contain integer indices")
    publish = torch.ones_like(positions, dtype=torch.bool)
    if publish_only_complete and logical_divisor > 1:
        publish = (positions + 1).remainder(logical_divisor) == 0
    slots = torch.full_like(positions, -1, dtype=torch.int64)
    logical = torch.div(positions[publish], logical_divisor, rounding_mode="floor")
    logical_block = torch.div(logical, storage_block_size, rounding_mode="floor")
    offset = logical.remainder(storage_block_size)
    active_requests = request_ids[publish]
    if bool(((active_requests < 0) | (active_requests >= block_table.shape[0])).any()):
        raise ValueError("published request id is outside the block table")
    if bool(((logical_block < 0) | (logical_block >= block_table.shape[1])).any()):
        raise ValueError("block table does not cover a required logical page")
    physical = block_table[request_ids[publish].to(torch.long), logical_block.to(torch.long)]
    if bool((physical < 0).any()):
        raise ValueError("required logical page has no physical allocation")
    slots[publish] = physical.to(torch.int64) * storage_block_size + offset.to(torch.int64)
    return slots


def compressor_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    compression_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build ragged compressor output starts, source-token rows, and group-first positions."""
    query_lens = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    old_groups = torch.div(kv_seq_lens.to(torch.int64), compression_ratio, rounding_mode="floor")
    new_groups = torch.div(kv_seq_lens.to(torch.int64) + query_lens, compression_ratio, rounding_mode="floor")
    output_lens = new_groups - old_groups
    output_starts = torch.cat(
        (torch.zeros(1, dtype=torch.int64, device=query_start_loc.device), output_lens.cumsum(0))
    )
    source_rows: list[torch.Tensor] = []
    output_positions: list[torch.Tensor] = []
    for request in range(query_lens.numel()):
        groups = torch.arange(old_groups[request], new_groups[request], device=query_start_loc.device)
        completed_positions = (groups + 1) * compression_ratio - 1
        local_rows = completed_positions - kv_seq_lens[request].to(torch.int64)
        source_rows.append(query_start_loc[request].to(torch.int64) + local_rows)
        output_positions.append(groups * compression_ratio)
    empty = torch.empty(0, dtype=torch.int64, device=query_start_loc.device)
    return (
        output_starts.to(torch.int32),
        torch.cat(source_rows).to(torch.int32) if source_rows else empty.to(torch.int32),
        torch.cat(output_positions).to(torch.int32) if output_positions else empty.to(torch.int32),
    )


def window_metadata(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    window: int = M.sliding_window,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build paged write slots plus causal indices for the visible sliding window."""
    slots = paged_slots(positions, request_ids, block_table)
    lens = torch.minimum(positions + 1, torch.full_like(positions, window)).to(torch.int32)
    offsets = torch.arange(window, device=positions.device)
    starts = positions - lens.to(positions.dtype) + 1
    visible = starts.unsqueeze(-1) + offsets
    valid = offsets.unsqueeze(0) < lens.unsqueeze(-1)
    indices = torch.full_like(visible, -1, dtype=torch.int64)
    visible_requests = request_ids.unsqueeze(-1).expand_as(visible)
    indices[valid] = paged_slots(visible[valid], visible_requests[valid], block_table)
    return slots, indices.to(torch.int32), lens


def build_forward_metadata(
    query_start_loc: torch.Tensor,
    kv_seq_lens: torch.Tensor,
    window_block_table: torch.Tensor,
    compressed_block_tables: Mapping[int, torch.Tensor],
    state_block_tables: Mapping[int, torch.Tensor],
) -> ForwardMetadata:
    """Lower engine inputs for packed prefill or one-token-per-request decode."""
    if query_start_loc.ndim != 1 or kv_seq_lens.ndim != 1:
        raise ValueError("query_start_loc and kv_seq_lens must be one-dimensional")
    if query_start_loc.numel() != kv_seq_lens.numel() + 1:
        raise ValueError("query_start_loc must contain one more element than kv_seq_lens")
    query_lens = query_start_loc[1:].to(torch.int64) - query_start_loc[:-1].to(torch.int64)
    if bool((query_lens < 0).any()) or int(query_start_loc[0]) != 0:
        raise ValueError("query_start_loc must be nondecreasing and start at zero")
    if bool((kv_seq_lens < 0).any()):
        raise ValueError("kv_seq_lens must be non-negative")
    tables_by_ratio: dict[int, torch.Tensor] = {}
    for source in M.kv_source_layer_ids:
        if source not in compressed_block_tables:
            raise ValueError(f"missing compressed block table for source layer {source}")
        ratio = M.compress_ratios[source]
        previous = tables_by_ratio.setdefault(ratio, compressed_block_tables[source])
        if not torch.equal(previous, compressed_block_tables[source]):
            raise ValueError(f"compression ratio {ratio} sources must share one compressed block table")
    request_ids = request_ids_from_starts(query_start_loc)
    logit_rows = torch.where(query_lens > 0, query_start_loc[1:].to(torch.int64) - 1, -1).to(torch.int32)
    local_offsets = torch.arange(request_ids.numel(), device=request_ids.device)
    local_offsets -= query_start_loc[request_ids.to(torch.long)].to(local_offsets.dtype)
    positions = kv_seq_lens[request_ids.to(torch.long)].to(torch.int64) + local_offsets
    window_slots, window_indices, window_lens = window_metadata(positions, request_ids, window_block_table)
    compressed_slots: dict[int, torch.Tensor] = {}
    index_slots: dict[int, torch.Tensor] = {}
    state_tables: dict[int, torch.Tensor] = {}
    compressed_seq_lens: dict[int, torch.Tensor] = {}
    compressed_seq_remainders: dict[int, torch.Tensor] = {}
    compressed_lens: dict[int, torch.Tensor] = {}
    compressor_output_starts: dict[int, torch.Tensor] = {}
    compressor_source_rows: dict[int, torch.Tensor] = {}
    compressor_positions: dict[int, torch.Tensor] = {}
    compressed_rope_positions: dict[int, torch.Tensor] = {}
    new_kv_seq_lens = kv_seq_lens.to(torch.int64) + query_lens
    for source in M.kv_source_layer_ids:
        ratio = M.compress_ratios[source]
        storage_rows = BLOCK_SIZE
        table = compressed_block_tables[source]
        if table.ndim != 2 or table.shape[0] != query_lens.numel():
            raise ValueError("compressed block tables must have one row per request")
        # Sparse attention can read the entire compressed history of an active request.
        for request in range(query_lens.numel()):
            if int(query_lens[request]) == 0:
                continue
            visible_rows = int(new_kv_seq_lens[request]) // ratio
            pages = (visible_rows + storage_rows - 1) // storage_rows
            if pages > table.shape[1] or bool((table[request, :pages] < 0).any()):
                raise ValueError(f"source layer {source} has missing visible compressed pages")
        compressed_slots[source] = paged_slots(
            positions, request_ids, compressed_block_tables[source], storage_rows, ratio, True
        )
        index_slots[source] = paged_slots(
            positions, request_ids, compressed_block_tables[source], storage_rows, ratio, True
        )
        compressed_seq_lens[source] = torch.div(new_kv_seq_lens, ratio, rounding_mode="floor").to(torch.int32)
        compressed_seq_remainders[source] = new_kv_seq_lens.remainder(ratio).to(torch.int32)
        compressed_lens[source] = torch.div(positions + 1, ratio, rounding_mode="floor").to(torch.int32)
        output_starts, source_rows, output_positions = compressor_metadata(
            query_start_loc, kv_seq_lens, ratio
        )
        compressor_output_starts[source] = output_starts
        compressor_source_rows[source] = source_rows
        compressor_positions[source] = output_positions
        complete = (positions + 1).remainder(ratio) == 0
        compressed_rope_positions[source] = torch.where(complete, positions + 1 - ratio, -1).to(torch.int32)
        if ratio > 1:
            if source not in state_block_tables:
                raise ValueError(f"missing state block table for source layer {source}")
            table = state_block_tables[source]
            if table.shape != (kv_seq_lens.numel(), 1) or table.dtype != torch.int32:
                raise ValueError("state block tables must be INT32 [requests, 1]")
            allocated = table[table >= 0]
            if allocated.unique().numel() != allocated.numel():
                raise ValueError("live requests must own distinct state blocks")
            if bool((table < -1).any()):
                raise ValueError("inactive requests use state block -1")
            state_tables[source] = table
            inactive = table[request_ids.to(torch.long), 0] < 0
            compressed_slots[source] = compressed_slots[source].masked_fill(inactive, -1)
            index_slots[source] = index_slots[source].masked_fill(inactive, -1)
    return ForwardMetadata(
        query_start_loc=query_start_loc.to(torch.int32),
        query_lens=query_lens.to(torch.int32),
        logit_row_indices=logit_rows,
        token_to_req_indices=request_ids,
        moe_token_owners=torch.arange(
            request_ids.numel(), device=request_ids.device, dtype=torch.int32
        ).remainder(TP_SIZE),
        position_ids=positions.to(torch.int32),
        kv_seq_lens=kv_seq_lens.to(torch.int32),
        new_kv_seq_lens=new_kv_seq_lens.to(torch.int32),
        window_slots=window_slots,
        window_indices=window_indices,
        window_lens=window_lens,
        compressed_slots=compressed_slots,
        index_slots=index_slots,
        state_block_tables=state_tables,
        compressed_seq_lens=compressed_seq_lens,
        compressed_seq_remainders=compressed_seq_remainders,
        compressed_lens=compressed_lens,
        compressor_output_start_loc=compressor_output_starts,
        compressor_source_token_indices=compressor_source_rows,
        compressor_position_ids=compressor_positions,
        compressed_rope_position_ids=compressed_rope_positions,
    )


# Fresh-prompt CED replay metadata.


@dataclass(frozen=True)
class DecoderReplay:
    """Request-local positions and physical cache rows of the selected encoder tail."""

    source_rows: torch.Tensor
    query_start_loc: torch.Tensor
    query_lens: torch.Tensor
    replay_starts: torch.Tensor
    position_ids: torch.Tensor
    token_to_req_indices: torch.Tensor
    compressed_lens: torch.Tensor
    window_slots: torch.Tensor
    window_indices: torch.Tensor
    window_lens: torch.Tensor

    @property
    def logit_row_indices(self) -> torch.Tensor:
        """Last replay row per request, or -1 when that request has no rows."""
        return torch.where(self.query_lens > 0, self.query_start_loc[1:] - 1, -1)


def select_decoder_replay(
    metadata: ForwardMetadata,
    window_block_table: torch.Tensor,
    *,
    window: int = M.sliding_window,
) -> DecoderReplay:
    """Select at most 128 final rows per request and truncate SWA to that replay."""
    starts = metadata.query_start_loc.to(torch.int64)
    if window <= 0 or starts.ndim != 1 or starts.numel() < 2 or int(starts[0]) != 0:
        raise ValueError("positive window and packed query starts are required")
    lengths = starts[1:] - starts[:-1]
    if bool((lengths < 0).any()):
        raise ValueError("query_start_loc must be nondecreasing")
    if metadata.position_ids.numel() != int(starts[-1]) or metadata.token_to_req_indices.numel() != int(starts[-1]):
        raise ValueError("encoder metadata does not cover the packed query rows")
    if metadata.kv_seq_lens.numel() != lengths.numel() or bool((metadata.kv_seq_lens != 0).any()):
        raise ValueError("fresh decoder prefill does not accept cached prefixes")
    if window_block_table.ndim != 2 or window_block_table.shape[0] != lengths.numel():
        raise ValueError("window block table must have one row per request")
    selected, positions, requests = [], [], []
    replay_lengths = torch.minimum(lengths, torch.full_like(lengths, window))
    replay_starts = lengths - replay_lengths
    for request, length in enumerate(lengths.tolist()):
        first, last = int(starts[request]), int(starts[request + 1])
        expected = torch.arange(length, device=starts.device)
        if not torch.equal(metadata.position_ids[first:last].to(torch.int64), expected):
            raise ValueError(f"request {request} has noncontiguous encoder positions")
        if not bool((metadata.token_to_req_indices[first:last] == request).all()):
            raise ValueError(f"request {request} has incorrect encoder request indices")
        count = int(replay_lengths[request])
        selected.append(torch.arange(last - count, last, device=starts.device))
        positions.append(torch.arange(length - count, length, device=starts.device))
        requests.append(torch.full((count,), request, dtype=torch.int32, device=starts.device))
    source_rows = torch.cat(selected).long()
    position_ids = torch.cat(positions).int()
    request_ids = torch.cat(requests)
    query_lens = replay_lengths.int()
    query_start_loc = torch.cat((torch.zeros(1, dtype=torch.int32, device=starts.device), query_lens.cumsum(0)))
    window_lens = position_ids - replay_starts[request_ids.long()] + 1
    offsets = torch.arange(window, device=starts.device)
    visible = position_ids[:, None] - window_lens[:, None] + 1 + offsets
    valid = offsets[None] < window_lens[:, None]
    window_indices = torch.full_like(visible, -1, dtype=torch.int32)
    window_indices[valid] = paged_slots(
        visible[valid], request_ids[:, None].expand_as(visible)[valid], window_block_table
    ).int()
    slots = paged_slots(position_ids, request_ids, window_block_table)
    return DecoderReplay(
        source_rows, query_start_loc, query_lens, replay_starts.int(), position_ids,
        request_ids, position_ids + 1, slots, window_indices, window_lens.int(),
    )


_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
}


def decode_checkpoint_linear(weight: torch.Tensor, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Decode stored weights without activation quantization or requantizing weights.

    The returned matrix is output-major [N,K], matching the checkpoint and
    torch.nn.functional.linear. MXFP8 uses 32x32 scale blocks; MXFP4 packs the
    even input column in the low nibble and uses one scale per 32 input columns.
    """
    if weight.dtype in (torch.float32, torch.bfloat16):
        if scale is not None:
            raise ValueError("unquantized checkpoint weights must not have a scale")
        return weight.float().contiguous()
    if scale is None or scale.dtype != torch.float8_e8m0fnu:
        raise ValueError("quantized checkpoint weights require E8M0 scales")
    codes = scale.contiguous().view(torch.uint8).int()
    if bool((codes == 255).any()):
        raise ValueError("checkpoint E8M0 scales must be finite")
    factors = torch.exp2(codes.float() - 127)
    if weight.dtype == torch.float8_e4m3fn:
        if weight.ndim != 2 or tuple(weight.shape) != (scale.shape[0] * 32, scale.shape[1] * 32):
            raise ValueError("MXFP8 checkpoint weight/scale shapes disagree")
        result = weight.float() * factors.repeat_interleave(32, 0).repeat_interleave(32, 1)
    elif weight.dtype == torch.int8:
        if (
            weight.ndim != 2
            or weight.shape[0] != scale.shape[0]
            or weight.shape[1] * 2 != scale.shape[1] * 32
        ):
            raise ValueError("MXFP4 checkpoint weight/scale shapes disagree")
        packed = weight.contiguous().view(torch.uint8)
        nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        magnitude = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=weight.device)
        values = magnitude[(nibbles & 7).long()]
        values = torch.where((nibbles & 8) != 0, -values, values)
        result = values * factors.repeat_interleave(32, -1)
    else:
        raise ValueError(f"unsupported checkpoint linear dtype {weight.dtype}")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("decoded checkpoint linear contains non-finite values")
    return result.contiguous()


class PrefillCheckpoint:
    """Read requested original tensors and decode linear matrices as output-major FP32."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        config = json.loads((self.root / "config.json").read_text())
        text = config["text_config"]
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]:
            raise ValueError("checkpoint must use the official block-32 FP8/FP4 format")
        required = {
            "hidden_size": M.hidden_size,
            "num_hidden_layers": M.num_hidden_layers,
            "num_attention_heads": M.num_attention_heads,
            "head_dim": M.head_dim,
            "sliding_window": M.sliding_window,
            "kv_source_layer_ids": list(M.kv_source_layer_ids),
            "index_source_layer_ids": list(M.index_source_layer_ids),
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"checkpoint {name}={text.get(name)!r}, expected {expected!r}")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def tensor(self, name: str) -> torch.Tensor:
        """Return an owned CPU tensor without loading the rest of its shard."""
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype = _DTYPES[entry["dtype"]]
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if start < 0 or end - start != expected_bytes or 8 + header_size + end > path.stat().st_size:
                raise ValueError(f"invalid safetensors range for {name}")
            file.seek(8 + header_size + start)
            payload = bytearray(file.read(expected_bytes))
        if len(payload) != expected_bytes:
            raise ValueError(f"truncated safetensors payload for {name}")
        return torch.frombuffer(payload, dtype=dtype).reshape(shape)

    def tensor_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Read selected table rows without loading the complete tensor."""
        if rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("table row ids must have an integer dtype")
        ids = rows.to(dtype=torch.int64, device="cpu").flatten()
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
            if len(shape) != 2 or bool(((ids < 0) | (ids >= shape[0])).any()):
                raise ValueError(f"invalid table rows for {name}")
            row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
            start, end = entry["data_offsets"]
            if (
                start < 0
                or end - start != shape[0] * row_bytes
                or 8 + header_size + end > path.stat().st_size
            ):
                raise ValueError(f"invalid safetensors range for {name}")
            unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
            payload = bytearray(unique.numel() * row_bytes)
            for index, row in enumerate(unique.tolist()):
                file.seek(8 + header_size + start + row * row_bytes)
                data = file.read(row_bytes)
                if len(data) != row_bytes:
                    raise ValueError(f"truncated safetensors row for {name}")
                payload[index * row_bytes : (index + 1) * row_bytes] = data
        if not ids.numel():
            return torch.empty(*rows.shape, shape[1], dtype=dtype)
        table = torch.frombuffer(payload, dtype=dtype).reshape(-1, shape[1])
        # CPU indexing is unavailable for FP8; index its byte representation.
        return (
            table.view(torch.uint8)
            .reshape(unique.numel(), row_bytes)[inverse]
            .contiguous()
            .view(dtype)
            .reshape(*rows.shape, shape[1])
        )


    def linear(self, stem: str) -> torch.Tensor:
        weight = self.tensor(stem + ".weight")
        scale = self.tensor(stem + ".scale") if weight.dtype in (torch.int8, torch.float8_e4m3fn) else None
        return decode_checkpoint_linear(weight, scale)


class ResidentPrefillWeights:
    """Pack Engram-disabled prefill weights into TP1/EP4 or TP1/EP8 layouts.

    Routed weights retain their FP4 nibbles; MXFP8 payloads retain their bytes.
    Scale bytes are reordered into the native MX_B_NN layout. Only projections
    whose released inference dtype is BF16 or FP32 are decoded on the host.
    Offsets count elements in their named bank. Large byte banks use 128-column
    rows so runtime shapes and slice coordinates fit their uint32 ABI.
    """

    bank_columns = 128
    fp8_names = (
        "attn.wq_a",
        "attn.wq_b",
        "attn.wkv",
        "attn.wo_b",
        "attn.indexer.wq_b",
        "ffn.shared_experts.w1",
        "ffn.shared_experts.w3",
        "ffn.shared_experts.w2",
    )
    bf16_names = (
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn.wo_a.weight",
        "attn.compressor.wkv.weight",
        "attn.compressor.norm.weight",
        "attn.indexer.wk.weight",
        "attn.indexer.k_norm.weight",
        "attn.indexer.weights_proj.weight",
    )
    dense_names = (
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "attn.attn_sink",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "attn.compressor.wkv.weight",
        "attn.compressor.wgate.weight",
    )
    expert_names = ("w1", "w3", "w2")

    def __init__(self, checkpoint: PrefillCheckpoint, *, ep_size: int = 4):
        if isinstance(ep_size, bool) or not isinstance(ep_size, Integral) or ep_size not in (4, 8):
            raise ValueError("resident prefill requires EP4 or EP8")
        self.ranks = int(ep_size)
        self.local_experts = M.n_routed_experts // self.ranks
        self.checkpoint = checkpoint
        layers = M.num_hidden_layers
        self.fp8_offsets = torch.full((layers, len(self.fp8_names)), -1, dtype=torch.int64)
        self.fp8_scale_offsets = torch.full_like(self.fp8_offsets, -1)
        self.bf16_offsets = torch.full((layers, len(self.bf16_names)), -1, dtype=torch.int64)
        self.dense_offsets = torch.full((layers, len(self.dense_names)), -1, dtype=torch.int64)
        self.expert_offsets = torch.empty((layers, 3), dtype=torch.int64)
        self.expert_scale_offsets = torch.empty_like(self.expert_offsets)
        self.fp8_entries, self.bf16_entries, self.dense_entries = [], [], []
        self.fp8_values = self.scale_values = self.bf16_values = self.dense_values = 0
        for layer in range(layers):
            for column, suffix in enumerate(self.fp8_names):
                stem = f"layers.{layer}.{suffix}"
                if stem + ".weight" not in checkpoint.weight_map:
                    continue
                header = self._header(stem + ".weight")
                n, k = header["shape"]
                scale = self._header(stem + ".scale")
                if (
                    header["dtype"] != "F8_E4M3"
                    or n % 32
                    or k % 64
                    or scale["dtype"] != "F8_E8M0"
                    or scale["shape"] != [n // 32, k // 32]
                ):
                    raise ValueError(f"unsupported native MXFP8 weight layout: {stem}")
                assert n * k % self.bank_columns == 0
                assert (n * k // 32) % self.bank_columns == 0
                self.fp8_offsets[layer, column] = self.fp8_values
                self.fp8_scale_offsets[layer, column] = self.scale_values
                self.fp8_entries.append((stem, self.fp8_values, self.scale_values))
                self.fp8_values += n * k
                self.scale_values += n * k // 32
            for column, suffix in enumerate(self.bf16_names):
                name = f"layers.{layer}.{suffix}"
                if name not in checkpoint.weight_map:
                    continue
                if column == 5 and M.compress_ratios[layer] != 1:
                    continue
                header = self._header(name)
                if header["dtype"] not in ("F32", "BF16", "F8_E4M3"):
                    raise ValueError(f"unsupported BF16 inference weight: {name}")
                if header["dtype"] == "F8_E4M3" and column != 4:
                    raise ValueError(f"unexpected quantized BF16 inference weight: {name}")
                self.bf16_offsets[layer, column] = self.bf16_values
                self.bf16_entries.append((name, column, self.bf16_values))
                self.bf16_values += math.prod(header["shape"])
            for column, suffix in enumerate(self.dense_names):
                name = f"layers.{layer}.{suffix}"
                if name not in checkpoint.weight_map:
                    continue
                if column in (9, 10) and M.compress_ratios[layer] != 2:
                    continue
                header = self._header(name)
                if header["dtype"] not in ("F32", "BF16"):
                    raise ValueError(f"unsupported FP32 inference weight: {name}")
                # Keep native matrix bases aligned after small mHC scalar vectors.
                self.dense_values = (self.dense_values + 127) // 128 * 128
                self.dense_offsets[layer, column] = self.dense_values
                self.dense_entries.append((name, column, self.dense_values))
                self.dense_values += math.prod(header["shape"])
        self.fp4_values = 0
        for layer in range(layers):
            for column in range(3):
                self.expert_offsets[layer, column] = self.fp4_values
                self.expert_scale_offsets[layer, column] = self.scale_values
                self.fp4_values += self.local_experts * M.hidden_size * M.moe_intermediate_size // 2
                self.scale_values += self.local_experts * M.hidden_size * M.moe_intermediate_size // 32
        assert (M.hidden_size * M.moe_intermediate_size // 2) % self.bank_columns == 0
        assert (M.hidden_size * M.moe_intermediate_size // 32) % self.bank_columns == 0
        for values in (self.fp4_values, self.fp8_values, self.scale_values):
            assert values % self.bank_columns == 0
            assert values // self.bank_columns < 2**32
        for offsets in (
            self.fp8_offsets, self.fp8_scale_offsets, self.expert_offsets, self.expert_scale_offsets,
        ):
            assert bool((offsets[offsets >= 0] % self.bank_columns == 0).all())
        assert self.bf16_values * 2 < 2**32
        assert self.dense_values * 4 < 2**32

    def _header(self, name: str) -> dict[str, Any]:
        checkpoint = self.checkpoint
        shard = checkpoint.weight_map[name]
        if shard not in checkpoint._headers:
            path = (checkpoint.root / shard).resolve()
            if not path.is_relative_to(checkpoint.root):
                raise ValueError(f"checkpoint shard leaves its root: {shard}")
            with path.open("rb") as file:
                size = struct.unpack("<Q", file.read(8))[0]
                if size > 64 * 1024 * 1024:
                    raise ValueError(f"safetensors header is too large: {shard}")
                checkpoint._headers[shard] = json.loads(file.read(size))
        return checkpoint._headers[shard][name]

    @property
    def tensor_layouts(self):
        """Return contiguous bank allocation shapes and exact storage dtypes."""
        return {
            "bank4": ((self.fp4_values // self.bank_columns, self.bank_columns), torch.uint8),
            "bank8": ((self.fp8_values // self.bank_columns, self.bank_columns), torch.float8_e4m3fn),
            "scales8": ((self.scale_values // self.bank_columns, self.bank_columns), torch.float8_e8m0fnu),
            "bf16": ((self.bf16_values,), torch.bfloat16),
            "dense": ((self.dense_values,), torch.float32),
        }

    @property
    def bytes_per_rank(self):
        return sum(
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in self.tensor_layouts.values()
        )

    @property
    def offset_tensors(self):
        return {
            name: getattr(self, name)
            for name in (
                "fp8_offsets",
                "fp8_scale_offsets",
                "bf16_offsets",
                "dense_offsets",
                "expert_offsets",
                "expert_scale_offsets",
            )
        }

    def upload(self, runtime, rank: int, handles):
        """Upload a rank's persistent banks with one projection staged at a time."""
        from quant import pack_mx_b_scale, pack_mxfp4_weight_tiles

        if rank not in range(self.ranks):
            raise ValueError(f"expert rank must be in [0, {self.ranks})")
        checkpoint = self.checkpoint

        def copy(bank, data, offset):
            data = data.contiguous()
            if bank in ("bank4", "bank8", "scales8"):
                assert int(offset) % self.bank_columns == 0
                assert data.numel() % self.bank_columns == 0
            runtime.copy_to(
                handles[bank].data_ptr,
                data.data_ptr(),
                data.numel() * data.element_size(),
                dst_offset=int(offset) * data.element_size(),
                worker_id=rank,
            )

        for stem, offset, scale_offset in self.fp8_entries:
            weight = checkpoint.tensor(stem + ".weight")
            codes = checkpoint.tensor(stem + ".scale").view(torch.uint8)
            logical_scale = codes.repeat_interleave(32, dim=0).t().contiguous()
            copy("bank8", weight.t(), offset)
            copy("scales8", pack_mx_b_scale(logical_scale).view(torch.float8_e8m0fnu), scale_offset)
        for name, column, offset in self.bf16_entries:
            if column == 4:
                value = checkpoint.linear(name.removesuffix(".weight")).to(torch.bfloat16)
            else:
                value = checkpoint.tensor(name).to(torch.bfloat16)
            if column in (5, 7, 9):
                value = value.t()
            copy("bf16", value, offset)
        for name, column, offset in self.dense_entries:
            value = checkpoint.tensor(name).float()
            if column in (9, 10):
                value = value.t()
            copy("dense", value, offset)
        for layer in range(M.num_hidden_layers):
            for column, projection in enumerate(self.expert_names):
                k, n = (M.moe_intermediate_size, M.hidden_size) if projection == "w2" else (M.hidden_size, M.moe_intermediate_size)
                logical_scales = torch.empty((self.local_experts * (k // 32), n), dtype=torch.uint8)
                offset = int(self.expert_offsets[layer, column])
                for local in range(self.local_experts):
                    expert = rank * self.local_experts + local
                    stem = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                    weight = checkpoint.tensor(stem + ".weight")
                    scale = checkpoint.tensor(stem + ".scale")
                    if (
                        weight.dtype != torch.int8
                        or tuple(weight.shape) != (n, k // 2)
                        or scale.dtype != torch.float8_e8m0fnu
                        or tuple(scale.shape) != (n, k // 32)
                    ):
                        raise ValueError(f"unsupported native MXFP4 weight layout: {stem}")
                    packed = pack_mxfp4_weight_tiles(weight, 256, 256)
                    copy("bank4", packed, offset + local * (n * k // 2))
                    logical_scales[local * (k // 32) : (local + 1) * (k // 32)] = scale.view(torch.uint8).t()
                copy(
                    "scales8",
                    pack_mx_b_scale(logical_scales).view(torch.float8_e8m0fnu),
                    self.expert_scale_offsets[layer, column],
                )

    def lookup_rows(self, token_ids: torch.Tensor):
        """Read only the prompt's BF16 token embeddings."""
        return self.checkpoint.tensor_rows("embed.weight", token_ids).to(torch.bfloat16)


def prepare_prefill_inputs(embedding: torch.Tensor, capacity: int) -> dict[str, torch.Tensor]:
    """Prepare one rank's prompt and causal replay with Engram disabled."""
    if embedding.ndim != 2 or embedding.shape[1] != M.hidden_size:
        raise ValueError("embedding must have shape [prompt tokens, hidden size]")
    count = embedding.shape[0]
    if capacity % 16 or not 1 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("a nonempty prompt must fit a prefill capacity aligned to 16")
    data_pages = (capacity + C.BLOCK_SIZE - 1) // C.BLOCK_SIZE
    pages = data_pages + 1
    table = torch.arange(data_pages, dtype=torch.int32).reshape(1, -1)
    compressed_tables = {source: table for source in M.kv_source_layer_ids}
    state_table = torch.zeros((1, 1), dtype=torch.int32)
    state_tables = {
        source: state_table
        for source in M.kv_source_layer_ids
        if M.compress_ratios[source] == 2
    }
    encoder = build_forward_metadata(
        torch.tensor([0, count], dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        table,
        compressed_tables,
        state_tables,
    )
    decoder = select_decoder_replay(encoder, table)
    decoder_count = decoder.source_rows.numel()
    c2_source = next(source for source in M.kv_source_layer_ids if M.compress_ratios[source] == 2)
    publisher_source = M.num_hidden_layers // 2
    plain_cos, plain_sin = precompute_rope_tables(count, False)
    compressed_cos, compressed_sin = precompute_rope_tables(count, True)

    def pad(value, fill=0):
        result = torch.full((capacity, *value.shape[1:]), fill, dtype=value.dtype)
        result[: value.shape[0]] = value
        return result

    def rope_rows(cosine, sine, positions):
        cos = torch.ones((capacity, M.qk_rope_head_dim // 2), dtype=torch.float32)
        sin = torch.zeros_like(cos)
        active = positions >= 0
        rows = torch.arange(positions.numel())[active]
        cos[rows] = cosine[positions[active].long()]
        sin[rows] = sine[positions[active].long()]
        return cos, sin

    encoder_plain = rope_rows(plain_cos, plain_sin, encoder.position_ids)
    encoder_compressed = rope_rows(compressed_cos, compressed_sin, encoder.position_ids)
    decoder_compressed = rope_rows(compressed_cos, compressed_sin, decoder.position_ids)
    pair_cos, pair_sin = rope_rows(
        compressed_cos, compressed_sin, encoder.compressed_rope_position_ids[c2_source]
    )
    initial_pre = torch.zeros((capacity, M.hc_mult), dtype=torch.float32)
    initial_pre[:, 0] = 1.0
    index_tables = torch.full((2, 1, pages), -1, dtype=torch.int32)
    index_tables[:, :, :data_pages] = table
    logit_rows = torch.full((16,), -1, dtype=torch.int32)
    logit_rows[0] = decoder_count - 1
    window_shape = (M.num_hidden_layers, pages, C.BLOCK_SIZE, 1)
    source_shape = (len(M.kv_source_layer_ids), pages, C.BLOCK_SIZE, 1)
    values = {
        "embedding": pad(embedding),
        "embedding_ids": torch.arange(capacity, dtype=torch.int64),
        "initial_pre": initial_pre,
        "identity_rows": pad(torch.arange(count, dtype=torch.int32), -1),
        "counts": torch.tensor([count, decoder_count], dtype=torch.int32),
        "replay_rows": pad(decoder.source_rows.int(), -1),
        "position_ids": torch.stack((pad(encoder.position_ids, -1), pad(decoder.position_ids, -1))),
        "rope_cos": torch.stack((encoder_plain[0], encoder_compressed[0], decoder_compressed[0])),
        "rope_sin": torch.stack((encoder_plain[1], encoder_compressed[1], decoder_compressed[1])),
        "compressed_cos": pair_cos,
        "compressed_sin": pair_sin,
        "publisher_cos": encoder_compressed[0],
        "publisher_sin": encoder_compressed[1],
        "window_slots": torch.stack((pad(encoder.window_slots, -1), pad(decoder.window_slots, -1))),
        "window_indices": torch.stack((pad(encoder.window_indices, -1), pad(decoder.window_indices, -1))),
        "compressed_slots": pad(encoder.compressed_slots[c2_source], -1),
        "publisher_slots": pad(encoder.compressed_slots[publisher_source], -1),
        "token_requests": torch.stack((pad(encoder.token_to_req_indices, -1), pad(decoder.token_to_req_indices, -1))),
        "compressed_lens": torch.stack((pad(encoder.compressed_lens[c2_source]), pad(decoder.compressed_lens))),
        "query_starts": torch.stack((encoder.query_start_loc, decoder.query_start_loc.int())),
        "index_block_tables": index_tables,
        "state_block_table": state_table,
        "window_cache": torch.zeros((*window_shape, M.head_dim), dtype=torch.uint8).view(torch.float8_e4m3fn),
        "window_scales": torch.full(
            (*window_shape, M.head_dim // C.WINDOW_CACHE_GROUP), 1.0, dtype=torch.bfloat16
        ),
        "compressed_cache": torch.zeros((*source_shape, M.head_dim // 2), dtype=torch.uint8),
        "compressed_scales": torch.full(
            (*source_shape, M.head_dim // C.COMPRESSED_CACHE_GROUP), 1.0, dtype=torch.bfloat16
        ),
        "index_cache": torch.zeros((*source_shape, M.index_head_dim // 2), dtype=torch.uint8),
        "index_scales": torch.full(
            (*source_shape, M.index_head_dim // C.INDEX_CACHE_GROUP), 127, dtype=torch.uint8
        ).view(torch.float8_e8m0fnu),
        "state_cache": torch.zeros((len(state_tables), 2, C.STATE_CAPACITY, (2 * M.head_dim)), dtype=torch.float32),
        "logit_rows": logit_rows,
    }
    return values


def prepare_resident_inputs(
    checkpoint: PrefillCheckpoint,
    weights: ResidentPrefillWeights,
    token_ids: torch.Tensor | Sequence[int],
    capacity: int,
) -> dict[str, torch.Tensor]:
    """Prepare checkpoint embeddings, weights and causal replay for one prefill rank."""
    ids = torch.as_tensor(token_ids, device="cpu")
    if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("token_ids must be a one-dimensional integer sequence")
    if not isinstance(capacity, Integral) or isinstance(capacity, bool) or capacity % 16:
        raise ValueError("prefill capacity must be an integer multiple of 16")
    count = ids.numel()
    if not 1 <= count <= capacity <= C.PREFILL_MAX_TOKENS:
        raise ValueError("a nonempty prompt must fit the configured prefill capacity")
    if bool(((ids < 0) | (ids >= M.vocab_size)).any()):
        raise ValueError("token id is outside the checkpoint vocabulary")
    ids = ids.long()
    embedding = weights.lookup_rows(ids)

    values = prepare_prefill_inputs(embedding, capacity)
    if not hasattr(weights, "_output_weights"):
        head = checkpoint.tensor("head.weight")
        if head.dtype != torch.bfloat16 or tuple(head.shape) != (M.vocab_size, M.hidden_size):
            raise ValueError("the native LM head requires the released BF16 checkpoint matrix")
        weights._output_weights = {
            "head": head,
            "final_norm": checkpoint.tensor("norm.weight").to(torch.bfloat16),
        }
    values.update(weights._output_weights)
    values.update(weights.offset_tensors)
    return values


def run_forward_smoke(fn, specs, *, config, compile_only=False, replicated=()):
    """Compile a forward smoke case and upload each rank's inputs in bounded chunks."""
    from golden import run
    from pypto.ir import ParamDirection
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    from pypto.runtime import RunConfig, StackedDeviceTensor

    result = run(fn=fn, specs=specs, config=config, compile_only=True)
    if not result.passed or compile_only:
        return result
    cfg = RunConfig(**{name: value for name, value in config.items() if name != "log_level"})
    compiled = DistributedCompiledProgram.from_dir(result.work_dir, platform=cfg.platform)
    parameters = compiled._get_metadata()[0]
    ranks = len(cfg.distributed_config.device_ids)
    chunk_bytes = 64 * 1024**2
    host_inputs = {spec.name: spec.create_tensor().share_memory_() for spec in specs if spec.name in replicated}
    with compiled.prepare(cfg, persistent=False, inherited_host_tensors=list(host_inputs.values())) as runtime:
        arguments = []
        for spec, parameter in zip(specs, parameters, strict=True):
            if isinstance(spec, ScalarSpec):
                arguments.append(spec.to_python())
                continue
            if spec.name in host_inputs:
                arguments.append(host_inputs[spec.name])
                continue
            shape = spec.shape[1:]
            if spec.shape[0] != ranks:
                raise ValueError(f"{spec.name} needs a leading rank axis of size {ranks}")
            shards = []
            for rank in range(ranks):
                handle = runtime.alloc_tensor(shape, spec.dtype, worker_id=rank)
                shards.append(handle)
                if parameter.direction == ParamDirection.Out:
                    continue
                if isinstance(spec.init_value, torch.Tensor):
                    if list(spec.init_value.shape) != spec.shape:
                        raise ValueError(f"{spec.name} initialization must match its tensor spec")
                    value = spec.init_value[rank].contiguous()
                else:
                    fill = 0 if spec.init_value is None else spec.init_value
                    element_bytes = torch.empty((), dtype=spec.dtype).element_size()
                    elements = min(math.prod(shape), chunk_bytes // element_bytes)
                    if spec.dtype == torch.float8_e8m0fnu:
                        code = torch.tensor(float(fill)).to(spec.dtype).view(torch.uint8).item()
                        value = torch.full((elements,), code, dtype=torch.uint8).view(spec.dtype)
                    else:
                        value = torch.full((elements,), fill, dtype=spec.dtype)
                for offset in range(0, handle.nbytes, chunk_bytes):
                    size = min(chunk_bytes, handle.nbytes - offset)
                    source_offset = offset if isinstance(spec.init_value, torch.Tensor) else 0
                    source = value.data_ptr() + source_offset
                    runtime.copy_to(handle.data_ptr, source, size, dst_offset=offset, worker_id=rank)
            arguments.append(StackedDeviceTensor(shards, tuple(spec.shape), tuple(range(ranks))))
        runtime(*arguments, config=cfg)
    print("[FWD] Execution passed; numerical accuracy was not validated.", flush=True)
    return result


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


BOUNDARY_PREFIX_NAMES = (
    "x_hc",
    "incoming_pre_mix",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_weight",
)


BOUNDARY_OUTPUT_NAMES = (
    "attention_output",
    "attention_hidden",
    "attention_pre_mix",
)


SCALAR_NAMES = ("num_tokens", "attention_epoch")


def owner_rows(actual, rank: int, active: int, *, tp_size: int | None = None) -> tuple[int, int, int]:
    """Return ``(first, count, width)`` of *rank* inside a per-rank slab tensor.

    The CPU mirror of :func:`slab_owner` for goldens and comparators: ``actual`` is
    a ``[ranks, slab, ...]`` tensor, so its second extent is the physical slab and
    its first one the number of ranks that participate (``tp_size`` when several
    DP groups share the tensor).
    """
    width = actual.shape[1]
    world = tp_size or actual.shape[0]
    first, count, _ = sequence_parallel_bounds(active, world, rank % world, capacity=width * world)
    return first, count, width


def make_boundary_specs(args, sharded=False):
    """Create replicated mHC inputs, visible boundaries, and runtime scalars.

    With ``sharded`` the per-token boundaries keep the rank's physical slab.
    ``local_tokens`` supplies a padded slab extent when present; otherwise it
    is ``ceil(tokens / tp)``. The gather scratch keeps the full Attention extent.
    """
    # Keep one uniform physical slab per rank.  The last rank(s) may contain
    # padding when the active token count is not divisible by TP; the runtime
    # scalar ``num_tokens`` and the collectives' first/count calculation keep
    # those rows out of the gather and reduce-scatter.
    local = args.tokens
    if sharded:
        local = getattr(args, "local_tokens", (args.tokens + args.tp - 1) // args.tp)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    specs = []
    shapes = {
        "x_hc": [local, HC_MULT, D],
        "incoming_pre_mix": [local, HC_MULT],
        "hc_attn_fn": [MIX_HC, HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [MIX_HC],
        "attn_norm_weight": [D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        elif name == "incoming_pre_mix":
            value = torch.softmax(value, dim=-1)
        if sharded and name in ("x_hc", "incoming_pre_mix"):
            # Sequence-parallel ranks own different slices of the batch, so the token
            # inputs must differ per rank: with replicated values a gather or
            # ReduceScatter that mis-orders the slabs would still compare equal.
            stacked = torch.randn([args.tp, *shape], generator=generator)
            if name == "incoming_pre_mix":
                stacked = torch.softmax(stacked, dim=-1)
        else:
            stacked = value.unsqueeze(0).repeat(args.tp, *([1] * len(shape)))
        specs.append(
            TensorSpec(name, [args.tp, *shape], value.dtype, init_value=stacked, resident="stacked")
        )
    for name, shape, dtype in (
        ("attention_output", [local, D], torch.bfloat16),
        ("attention_hidden", [local, HC_MULT, D], torch.float32),
        ("attention_pre_mix", [local, HC_MULT], torch.float32),
    ):
        sentinel = 13.0 if name == "attention_output" else 0.0
        specs.append(TensorSpec(name, [args.tp, *shape], dtype, init_value=sentinel, resident="stacked"))
    if sharded:
        specs.append(
            TensorSpec(
                "gathered", [args.tp, args.tokens, D], torch.bfloat16,
                init_value=13.0, resident="stacked",
            )
        )
    specs.append(ScalarSpec("num_tokens", torch.int32, args.active_tokens))
    specs.append(
        ScalarSpec(
            "attention_epoch",
            torch.int32,
            1,
            compile_runtime=True,
            benchmark_step=args.epochs if args.bench else None,
        )
    )
    return specs


def assemble_specs(args, leaf_specs, spec_names, aliases=None, sharded=False):
    """Combine one leaf's exact specs with the stable half-layer boundary specs."""
    aliases = aliases or {}
    specs = []
    for spec in leaf_specs:
        name = aliases.get(spec.name, spec.name)
        if name in ("x", "output", *SCALAR_NAMES):
            continue
        specs.append(spec if name == spec.name else replace(spec, name=name))
    specs.extend(make_boundary_specs(args, sharded))
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in spec_names if name not in by_name]
    if missing:
        raise ValueError(f"missing Attention specs: {missing}")
    return [by_name[name] for name in spec_names]


def check_program_specs(world_size, specs, spec_names):
    """Validate host ordering and return tensor shapes/dtypes keyed by name."""
    if world_size != TP_SIZE:
        raise ValueError("Attention world size must match TP_SIZE")
    if tuple(spec.name for spec in specs) != tuple(spec_names):
        raise ValueError("Attention specs must match the host parameter names and order")
    dtypes = {
        torch.bfloat16: pl.BF16,
        torch.float32: pl.FP32,
        torch.float8_e4m3fn: pl.FP8E4M3FN,
        torch.float8_e8m0fnu: pl.FP8E8M0,
        torch.int32: pl.INT32,
        torch.int64: pl.INT64,
        torch.uint8: pl.UINT8,
    }
    tensors = [spec for spec in specs if isinstance(spec, TensorSpec)]
    return SimpleNamespace(
        **{spec.name: SimpleNamespace(shape=spec.shape, dtype=dtypes[spec.dtype]) for spec in tensors}
    )


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


def make_boundary_comparisons(compare_output):
    """Build comparisons shared by every mHC-Attention composition boundary."""
    return {
        "attention_hidden": compare_output,
        "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
    }


def wiring_replay_dir(base: str | None, wiring: str) -> str | None:
    """Namespace a replay directory per wiring.

    The wiring name is the subdirectory, so ``--golden-data``/``--runtime-dir``
    can never hand one ABI the artifacts of the other: wiring ``W`` always reads
    ``<base>/W``, whether one wiring or both run in this invocation.
    """
    if base is None:
        return None
    if wiring not in ("replicated", "sharded"):
        raise ValueError(f"a replay directory needs a concrete wiring, got {wiring!r}")
    return str(Path(base) / wiring)


@dataclass(frozen=True)
class DecodeTokenShard:
    """One TP rank's fixed-capacity token slab and its request metadata."""

    hidden: torch.Tensor
    token_ids: torch.Tensor
    position_ids: torch.Tensor
    valid_mask: torch.Tensor


def sequence_parallel_bounds(
    num_tokens: int, tp_size: int, tp_rank: int, capacity: int | None = None
) -> tuple[int, int, int]:
    """Return ``(first, count, width)`` for one contiguous TP slab.

    ``width`` is the *physical* slab every rank publishes, which the device takes
    from the tensor extent (``ceil(capacity / tp)``) rather than from the active row
    count.  ``capacity`` defaults to ``num_tokens`` for the common case where every
    capacity row is active; pass it explicitly whenever a padded batch or ``T < TP``
    makes the two differ, otherwise the rank mapping would disagree with the
    AllGather and ReduceScatter.
    """
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if tp_size <= 0:
        raise ValueError("tp_size must be positive")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")
    physical = num_tokens if capacity is None else capacity
    if physical < num_tokens:
        raise ValueError(f"capacity {physical} cannot be smaller than {num_tokens} active tokens")
    width = (physical + tp_size - 1) // tp_size if physical else 0
    first = min(tp_rank * width, num_tokens)
    count = max(0, min(width, num_tokens - first))
    return first, count, width


def shard_decode_batch(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    position_ids: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    tp_size: int,
    capacity: int | None = None,
) -> tuple[DecodeTokenShard, ...]:
    """Split a global Decode batch into fixed-width owner slabs.

    Padding rows are physically present so every rank can participate in the
    same collective.  They carry ``token_id=-1`` and ``valid_mask=False`` and
    are never included by :func:`gather_decode_batch`.
    """
    if hidden.ndim < 1:
        raise ValueError("hidden must have a token dimension")
    num_tokens = hidden.shape[0]
    if token_ids.shape != (num_tokens,) or position_ids.shape != (num_tokens,):
        raise ValueError("token_ids and position_ids must have one entry per token")
    if valid_mask is None:
        valid_mask = torch.ones(num_tokens, dtype=torch.bool, device=hidden.device)
    if valid_mask.shape != (num_tokens,) or valid_mask.dtype is not torch.bool:
        raise ValueError("valid_mask must be a bool vector with one entry per token")
    if not bool(valid_mask.all()):
        raise ValueError("sequence-parallel ownership requires a contiguous valid token prefix")

    shards: list[DecodeTokenShard] = []
    for rank in range(tp_size):
        first, count, width = sequence_parallel_bounds(num_tokens, tp_size, rank, capacity)
        local_hidden = hidden.new_zeros((width, *hidden.shape[1:]))
        local_tokens = torch.full((width,), -1, dtype=token_ids.dtype, device=token_ids.device)
        local_positions = torch.full((width,), -1, dtype=position_ids.dtype, device=position_ids.device)
        local_valid = torch.zeros(width, dtype=torch.bool, device=hidden.device)
        if count:
            local_hidden[:count].copy_(hidden[first : first + count])
            local_tokens[:count].copy_(token_ids[first : first + count])
            local_positions[:count].copy_(position_ids[first : first + count])
            local_valid[:count] = True
        shards.append(DecodeTokenShard(local_hidden, local_tokens, local_positions, local_valid))
    return tuple(shards)


def gather_decode_batch(shards: tuple[DecodeTokenShard, ...] | list[DecodeTokenShard], num_tokens: int) -> DecodeTokenShard:
    """Restore global token order from owner-local slabs after Attention.

    The slab width carried by the shards is authoritative: it is the physical
    capacity the collectives published, so a caller does not have to restate it.
    """
    if not shards:
        raise ValueError("at least one TP shard is required")
    tp_size = len(shards)
    slab_width = shards[0].hidden.shape[0]
    pieces = []
    token_pieces = []
    position_pieces = []
    mask_pieces = []
    for rank, shard in enumerate(shards):
        if shard.hidden.shape[0] != slab_width:
            raise ValueError(
                f"rank {rank} has slab width {shard.hidden.shape[0]}, expected {slab_width}"
            )
        first, count, _ = sequence_parallel_bounds(
            num_tokens, tp_size, rank, capacity=slab_width * tp_size
        )
        if count:
            pieces.append(shard.hidden[:count])
            token_pieces.append(shard.token_ids[:count])
            position_pieces.append(shard.position_ids[:count])
            mask_pieces.append(shard.valid_mask[:count])
    if pieces:
        hidden = torch.cat(pieces, dim=0)
        token_ids = torch.cat(token_pieces, dim=0)
        position_ids = torch.cat(position_pieces, dim=0)
        valid_mask = torch.cat(mask_pieces, dim=0)
    else:
        template = shards[0].hidden
        hidden = template[:0]
        token_ids = shards[0].token_ids[:0]
        position_ids = shards[0].position_ids[:0]
        valid_mask = shards[0].valid_mask[:0]
    if hidden.shape[0] != num_tokens or not bool(valid_mask.all()):
        raise RuntimeError("sequence-parallel gather did not restore every active token")
    return DecodeTokenShard(hidden, token_ids, position_ids, valid_mask)
