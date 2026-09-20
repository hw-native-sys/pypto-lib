# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared fixture primitives for packed-prefill C1A validation."""

import inspect
from collections.abc import Callable

import torch

from golden import ScalarSpec, TensorSpec
from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import _project_output
from models.deepseek_v4_1_flash.golden import paged_indexer, qkv_proj_rope
from models.deepseek_v4_1_flash.quantization import (
    dequantize_mxfp4_cache,
    dequantize_mxfp8_cache,
    pack_mx_b_scale,
    quantize_mxfp4_cache,
    quantize_mxfp8_cache,
)


CASE_DEFAULT = "multi_1k"
CASE_TOKENS = 64
CASE_MAX_TOKENS = 128
CASE_PAGE = 128
CASE_NAMES = (CASE_DEFAULT, "causal", "mixed", "long")

OUTPUT_RMS_ATOL = 1e-6
OUTPUT_RMS_RTOL = 0.01
OUTPUT_PEAK_ATOL = 1e-5
OUTPUT_PEAK_RMS_RATIO = 0.05
CACHE_MAX_RELATIVE_L2 = 0.01
MXFP4_CACHE_MAX_RELATIVE_L2 = 0.04
TOPK_SCORE_ATOL = 5e-5
TOPK_SCORE_RTOL = 1.5e-2

COMMON_INPUT_NAMES = (
    "x",
    "wq_a",
    "wq_a_scale",
    "q_norm_weight",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "wkv_scale",
    "kv_norm_weight",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "rope_cos",
    "rope_sin",
    "window_slots",
    "window_indices",
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
)


def _ranked(value: torch.Tensor) -> torch.Tensor:
    return value.unsqueeze(0).repeat(C.TP_SIZE, *([1] * value.ndim)).contiguous()


def _fp8_weight(input_width: int, output_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randn(input_width, output_width).mul_(input_width**-0.5)
    weight = weight.clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
    scale_codes = torch.full((input_width // 32, output_width), 127, dtype=torch.uint8)
    scale = pack_mx_b_scale(scale_codes).view(torch.float8_e8m0fnu)
    return weight, scale


def _ranked_fp8_weight(input_width: int, output_width: int, shared: bool) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = [_fp8_weight(input_width, output_width) for _ in range(1 if shared else C.TP_SIZE)]
    if shared:
        weight = _ranked(pairs[0][0])
        scale = _ranked(pairs[0][1])
    else:
        weight = torch.stack([pair[0] for pair in pairs])
        scale = torch.stack([pair[1] for pair in pairs])
    return weight, scale


def _case_metadata(token_count: int, case_name: str) -> tuple[torch.Tensor, ...]:
    if case_name == CASE_DEFAULT:
        request_ids = torch.arange(3, dtype=torch.int32)
        compressed_lens = torch.full((3,), 1024, dtype=torch.int32)
    elif case_name == "causal":
        if not 1 <= token_count <= CASE_MAX_TOKENS:
            raise ValueError(f"token_count must be in [1, {CASE_MAX_TOKENS}], got {token_count}")
        request_ids = torch.zeros(token_count, dtype=torch.int32)
        compressed_lens = torch.arange(1, token_count + 1, dtype=torch.int32)
    elif case_name == "mixed":
        request_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2], dtype=torch.int32)
        compressed_lens = torch.tensor([1, 2, 3, 127, 128, 511, 512, 513], dtype=torch.int32)
    elif case_name == "long":
        request_ids = torch.arange(3, dtype=torch.int32)
        compressed_lens = torch.tensor([1024, 4096, 8192], dtype=torch.int32)
    else:
        raise ValueError(f"unknown case {case_name!r}; expected one of {CASE_NAMES}")

    request_count = int(request_ids.max().item()) + 1
    table_pages = (int(compressed_lens.max().item()) + CASE_PAGE - 1) // CASE_PAGE
    block_table = torch.arange(request_count * table_pages, dtype=torch.int32).reshape(
        request_count,
        table_pages,
    )
    positions = compressed_lens.to(torch.int64) - 1
    pages = torch.div(positions, CASE_PAGE, rounding_mode="floor")
    offsets = positions.remainder(CASE_PAGE)
    slots = block_table[request_ids.to(torch.long), pages] * CASE_PAGE + offsets

    window_indices = torch.full((request_ids.numel(), CASE_PAGE), -1, dtype=torch.int32)
    compressed_indices = torch.full((request_ids.numel(), C.INDEX_TOPK), -1, dtype=torch.int32)
    candidate_mask = torch.zeros(request_ids.numel(), table_pages * CASE_PAGE, dtype=torch.uint8)
    for token in range(request_ids.numel()):
        request = int(request_ids[token])
        visible = int(compressed_lens[token])
        logical = torch.arange(visible, dtype=torch.int64)
        physical = block_table[request, logical // CASE_PAGE] * CASE_PAGE + logical % CASE_PAGE
        window_count = min(visible, CASE_PAGE)
        window_indices[token, :window_count] = physical[-window_count:].to(torch.int32)
        compressed_count = min(visible, C.INDEX_TOPK)
        compressed_indices[token, :compressed_count] = physical[:compressed_count].to(torch.int32)
        candidate_mask[token, :visible] = 1

    return (
        request_ids,
        compressed_lens,
        block_table,
        slots.to(torch.int64),
        window_indices,
        compressed_indices,
        candidate_mask,
    )


def make_fixture_values(token_count: int, case_name: str) -> dict[str, torch.Tensor]:
    torch.manual_seed(41)
    (
        request_ids,
        compressed_lens,
        block_table,
        slots,
        window_indices,
        compressed_indices,
        candidate_mask,
    ) = _case_metadata(token_count, case_name)
    token_count = request_ids.numel()
    cache_blocks = block_table.numel()
    wq_a, wq_a_scale = _ranked_fp8_weight(C.D, C.Q_LORA, shared=True)
    wq_b, wq_b_scale = _ranked_fp8_weight(
        C.Q_LORA,
        C.LOCAL_H * C.HEAD_DIM,
        shared=False,
    )
    wkv, wkv_scale = _ranked_fp8_weight(C.D, C.HEAD_DIM, shared=True)
    wo_b, wo_b_scale = _ranked_fp8_weight(C.LOCAL_O_WIDTH, C.D, shared=False)
    index_wq_b, index_wq_b_scale = _ranked_fp8_weight(
        C.Q_LORA,
        C.INDEX_H * C.INDEX_DIM,
        shared=True,
    )

    window_value = torch.randn(cache_blocks, CASE_PAGE, 1, C.HEAD_DIM).mul_(0.05)
    window_cache, window_cache_scale = quantize_mxfp8_cache(window_value)
    window_cache_scale = window_cache_scale.view(torch.float8_e8m0fnu)
    compressed_value = torch.randn(cache_blocks, CASE_PAGE, 1, C.HEAD_DIM).mul_(0.05)
    compressed_cache, compressed_cache_scale = quantize_mxfp4_cache(
        compressed_value,
        group_size=C.COMPRESSED_CACHE_GROUP,
        scale_format="e4m3",
    )
    index_value = torch.randn(cache_blocks, CASE_PAGE, 1, C.INDEX_DIM).mul_(0.05)
    index_cache, index_cache_scale = quantize_mxfp4_cache(
        index_value,
        group_size=C.INDEX_CACHE_GROUP,
        scale_format="e8m0",
    )
    index_cache_scale = index_cache_scale.view(torch.float8_e8m0fnu)

    angles = (compressed_lens - 1).to(torch.float32).unsqueeze(1)
    angles = angles * torch.arange(C.ROPE_DIM // 2, dtype=torch.float32).unsqueeze(0) * 0.001
    values = {
        "x": _ranked(torch.randn(token_count, C.D).to(torch.bfloat16)),
        "wq_a": wq_a,
        "wq_a_scale": wq_a_scale,
        "q_norm_weight": _ranked(torch.ones(C.Q_LORA, dtype=torch.bfloat16)),
        "wq_b": wq_b,
        "wq_b_scale": wq_b_scale,
        "wkv": wkv,
        "wkv_scale": wkv_scale,
        "kv_norm_weight": _ranked(torch.ones(C.HEAD_DIM, dtype=torch.bfloat16)),
        "attn_sink": torch.zeros(C.TP_SIZE, C.LOCAL_H, dtype=torch.float32),
        "wo_a": torch.randn(
            C.TP_SIZE,
            C.LOCAL_O_GROUPS,
            C.O_LORA,
            C.O_GROUP_IN,
            dtype=torch.bfloat16,
        ).mul_(C.O_GROUP_IN**-0.5),
        "wo_b": wo_b,
        "wo_b_scale": wo_b_scale,
        "rope_cos": _ranked(torch.cos(angles)),
        "rope_sin": _ranked(torch.sin(angles)),
        "window_slots": _ranked(slots),
        "window_indices": _ranked(window_indices),
        "window_cache": _ranked(window_cache),
        "window_cache_scale": _ranked(window_cache_scale),
        "compressed_cache": _ranked(compressed_cache),
        "compressed_cache_scale": _ranked(compressed_cache_scale),
        "compressed_indices": _ranked(compressed_indices),
        "request_ids": _ranked(request_ids),
        "compressed_lens": _ranked(compressed_lens),
        "index_cache": _ranked(index_cache),
        "index_cache_scale": _ranked(index_cache_scale),
        "index_block_table": _ranked(block_table),
        "candidate_mask": _ranked(candidate_mask),
        "compressed_rope_cos": _ranked(torch.cos(angles)),
        "compressed_rope_sin": _ranked(torch.sin(angles)),
        "compressor_wkv": _ranked(
            torch.randn(C.D, C.HEAD_DIM).mul_(C.D**-0.5).to(torch.bfloat16)
        ),
        "compressor_norm_weight": _ranked(torch.ones(C.HEAD_DIM, dtype=torch.bfloat16)),
        "compressed_slots": _ranked(slots),
        "index_wk": _ranked(
            torch.randn(C.HEAD_DIM, C.INDEX_DIM).mul_(C.HEAD_DIM**-0.5).to(torch.bfloat16)
        ),
        "index_norm_weight": _ranked(torch.ones(C.INDEX_DIM, dtype=torch.bfloat16)),
        "index_wq_b": index_wq_b,
        "index_wq_b_scale": index_wq_b_scale,
        "index_weights_proj": _ranked(
            torch.randn(C.D, C.INDEX_H).mul_(C.D**-0.5).to(torch.bfloat16)
        ),
    }
    values["attn_norm_weight"] = _ranked((torch.randn(C.D).mul_(0.1).add_(1)).to(torch.bfloat16))
    return values


def make_tensor_specs(
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    token_count: int = CASE_TOKENS,
    case_name: str = CASE_DEFAULT,
) -> list[TensorSpec | ScalarSpec]:
    values = make_fixture_values(token_count, case_name)
    token_count = values["x"].shape[1]
    specs = [
        TensorSpec(name, list(values[name].shape), values[name].dtype, init_value=values[name])
        for name in input_names
    ]
    for name in output_names:
        if name == "topk_indices":
            shape = [C.TP_SIZE, token_count, C.INDEX_TOPK]
            dtype = torch.int32
        elif name == "candidate_mask":
            shape = list(values["candidate_mask"].shape)
            dtype = torch.uint8
        else:
            shape = [C.TP_SIZE, token_count, C.D]
            dtype = torch.bfloat16
        specs.append(TensorSpec(name, shape, dtype))
    specs.append(ScalarSpec("num_tokens", torch.int32, token_count))
    return specs


def golden_c1a_attention_input(hidden: torch.Tensor, attn_norm_weight: torch.Tensor) -> torch.Tensor:
    """Apply the official attention RMSNorm after the BF16 mHC collapse."""
    from models.deepseek_v4_1_flash.golden import rms_norm

    return rms_norm(hidden.to(torch.bfloat16), attn_norm_weight)


def _copy_storage(destination: torch.Tensor, source: torch.Tensor) -> None:
    destination.view(torch.uint8).copy_(source.contiguous().view(torch.uint8))


def golden_prefill_c1a_attention(
    query: torch.Tensor,
    window_cache: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_indices: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Mirror C1A tiled online softmax, BF16 PV, and its FP32 first-vector patch."""
    from models.deepseek_v4_1_flash.prefill_c1a_common import ATTENTION_TILE, M_TILE

    output = torch.empty_like(query)
    width = query.shape[-1]
    for token in range(query.shape[0]):
        q = query[token].float()
        maximum = q.new_full((q.shape[0],), -1e30)
        denominator = q.new_zeros(q.shape[0])
        numerator = torch.zeros_like(q)
        for cache, indices in ((window_cache, window_indices), (compressed_cache, compressed_indices)):
            flat_cache = cache.flatten(0, 1).squeeze(-2)
            for begin in range(0, indices.shape[-1], ATTENTION_TILE):
                rows = indices[token, begin:begin + ATTENTION_TILE].long()
                valid = rows >= 0
                if not bool(valid.any()):
                    continue
                kv = flat_cache[rows.clamp_min(0)].clone()
                kv[~valid] = 0
                scores = torch.matmul(q, kv.float().T) * width**-0.5
                scores = scores + (valid.float() - 1).unsqueeze(0) * 1e30
                next_maximum = torch.maximum(maximum, scores.amax(dim=-1))
                correction = torch.exp(maximum - next_maximum)
                probability = torch.exp(scores - next_maximum.unsqueeze(-1)) * valid.unsqueeze(0)
                denominator = denominator * correction + probability.sum(dim=-1)
                weighted = torch.matmul(probability.to(torch.bfloat16).float(), kv.float())
                # The first head in each M_TILE group uses the Vec PV correction for 16 columns.
                weighted[::M_TILE, :16] = (
                    probability[::M_TILE].unsqueeze(-1) * kv[:, :16].float().unsqueeze(0)
                ).sum(dim=1)
                numerator = numerator * correction.unsqueeze(-1) + weighted
                maximum = next_maximum
        final_maximum = torch.maximum(maximum, sink.float())
        correction = torch.exp(maximum - final_maximum)
        denominator = denominator * correction + torch.exp(sink.float() - final_maximum)
        output[token] = (numerator * (correction / denominator).unsqueeze(-1)).to(query.dtype)
    return output


def _reduce_tp_partials(partials: list[torch.Tensor]) -> torch.Tensor:
    """Sum rank partials in rank order and round the completed reduction to BF16."""
    reduced = torch.zeros_like(partials[0], dtype=torch.float32)
    for partial in partials:
        reduced.add_(partial.float())
    return reduced.to(torch.bfloat16)


def golden_prefill_tp_attention(
    attention_fn: Callable,
    hidden: torch.Tensor,
    attention_args: tuple[torch.Tensor, ...],
    tp_size: int,
) -> tuple[torch.Tensor, object]:
    """Run one prefill attention golden per TP rank and reduce its FP32 partials.

    The exported mHC helpers accept either one rank's attention arguments or rank-stacked
    arguments (the first dimension is ``tp_size``).  Rank-stacked arguments reproduce the
    sharded invocation; the local form remains available for CPU-side smoke checks.
    """
    ranked_args = bool(
        attention_args
        and isinstance(attention_args[0], torch.Tensor)
        and attention_args[0].ndim >= 2
        and attention_args[0].shape[0] == tp_size
    )
    results = []
    for rank in range(tp_size):
        rank_hidden = hidden
        if hidden.ndim > 0 and hidden.shape[0] == tp_size and hidden.ndim == 3:
            rank_hidden = hidden[rank]
        rank_args = attention_args
        if ranked_args:
            rank_args = tuple(argument[rank] for argument in attention_args)
        result = attention_fn(rank_hidden, *rank_args)
        results.append(result)
        output = result.output
        if output.ndim == rank_hidden.ndim + 1 and output.shape[0] == tp_size:
            partials = [output[peer] for peer in range(tp_size)]
            return _reduce_tp_partials(partials), result

    return _reduce_tp_partials([result.output for result in results]), results[0]


def apply_distributed_golden(
    mode: str,
    golden_fn: Callable,
    tensors: dict[str, torch.Tensor],
    active_tokens: int | None = None,
) -> None:
    parameter_names = tuple(inspect.signature(golden_fn).parameters)
    token_input_names = {
        "x",
        "rope_cos",
        "rope_sin",
        "window_slots",
        "window_indices",
        "request_ids",
        "compressed_lens",
        "compressed_rope_cos",
        "compressed_rope_sin",
        "compressed_slots",
        "candidate_mask",
        "compressed_indices",
    }
    results = []
    for rank in range(C.TP_SIZE):
        kwargs = {}
        for name in parameter_names:
            value = tensors[name][rank]
            if active_tokens is not None and name in token_input_names and value.ndim > 0:
                value = value[:active_tokens]
            kwargs[name] = value
        results.append(golden_fn(**kwargs))

    reduced = _reduce_tp_partials([result.output for result in results])
    if active_tokens is None:
        tensors["output"][:] = reduced.unsqueeze(0)
    else:
        tensors["output"][:, :active_tokens] = reduced.unsqueeze(0)
    for rank, result in enumerate(results):
        tensors["window_cache"][rank].copy_(result.window_cache)
        _copy_storage(tensors["window_cache_scale"][rank], result.window_cache_scale)
        if mode == "full":
            tensors["compressed_cache"][rank].copy_(result.compressed_cache)
            tensors["compressed_cache_scale"][rank].copy_(result.compressed_cache_scale)
            tensors["index_cache"][rank].copy_(result.index_cache)
            _copy_storage(tensors["index_cache_scale"][rank], result.index_cache_scale)
            candidate_mask = tensors["candidate_mask"][rank]
            candidate_mask.zero_()
            block_size = C.FLASH.candidate_block_size
            for token in range(result.candidate_mask.shape[0]):
                selected = torch.nonzero(result.candidate_mask[token], as_tuple=False).flatten()
                for block in torch.unique(selected // block_size):
                    begin = int(block) * block_size
                    end = min(begin + block_size, candidate_mask.shape[1])
                    candidate_mask[token, begin:end] = 1
        if mode in ("full", "reindex"):
            if active_tokens is None:
                tensors["topk_indices"][rank].copy_(result.topk_indices)
            else:
                tensors["topk_indices"][rank, :active_tokens].copy_(result.topk_indices)


def _golden_index_scores(
    mode: str,
    rank: int,
    cache_outputs: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    index_payload = cache_outputs["index_cache"] if mode == "full" else inputs["index_cache"]
    index_scale = cache_outputs["index_cache_scale"] if mode == "full" else inputs["index_cache_scale"]
    index_value = dequantize_mxfp4_cache(
        index_payload[rank], index_scale[rank], group_size=C.INDEX_CACHE_GROUP, scale_format="e8m0"
    ).to(torch.bfloat16)
    _, _, query_latent = qkv_proj_rope(
        inputs["x"][rank], inputs["wq_a"][rank], inputs["wq_a_scale"][rank],
        inputs["q_norm_weight"][rank], inputs["wq_b"][rank], inputs["wq_b_scale"][rank],
        inputs["wkv"][rank], inputs["wkv_scale"][rank], inputs["kv_norm_weight"][rank],
        inputs["rope_cos"][rank], inputs["rope_sin"][rank],
    )
    candidates = inputs["candidate_mask"][rank] if mode == "reindex" else None
    scores, _ = paged_indexer(
        inputs["x"][rank], query_latent, inputs["request_ids"][rank],
        index_value, inputs["index_block_table"][rank], inputs["compressed_lens"][rank],
        inputs["index_wq_b"][rank], inputs["index_wq_b_scale"][rank],
        inputs["index_weights_proj"][rank], inputs["rope_cos"][rank], inputs["rope_sin"][rank],
        candidates=candidates, topk=C.INDEX_TOPK,
    )
    return scores


def _physical_to_logical(inputs: dict[str, torch.Tensor], rank: int, token: int) -> dict[int, int]:
    request = int(inputs["request_ids"][rank, token])
    visible = int(inputs["compressed_lens"][rank, token])
    logical = torch.arange(visible, dtype=torch.int64)
    table = inputs["index_block_table"][rank, request]
    physical = table[logical // CASE_PAGE].to(torch.int64) * CASE_PAGE + logical % CASE_PAGE
    return {int(row): index for index, row in enumerate(physical.tolist())}


def topk_indices_compare(mode: str) -> Callable:
    """Accept device/CPU substitutions only at a numerically tied top-k cutoff."""

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        del rtol, atol
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return False, "    Top-K shape and dtype must match the reference"
        active = min(int(inputs.get("num_tokens", actual.shape[1])), actual.shape[1])
        actual_active = actual[:, :active]
        expected_active = expected[:, :active]
        if bool((actual_active < -1).any()):
            return False, "    Top-K padding must use -1"
        valid = actual_active >= 0
        if bool((valid[..., 1:] & ~valid[..., :-1]).any()):
            return False, "    Top-K padding must follow all valid indices"
        if torch.equal(actual_active, expected_active):
            return True, ""

        for rank in range(actual_active.shape[0]):
            scores = None
            for token in range(actual_active.shape[1]):
                actual_row = actual_active[rank, token]
                expected_row = expected_active[rank, token]
                actual_valid = actual_row[actual_row >= 0].to(torch.int64)
                expected_valid = expected_row[expected_row >= 0].to(torch.int64)
                if actual_valid.numel() != expected_valid.numel():
                    return False, (
                        f"    rank {rank} token {token} has {actual_valid.numel()} valid indices; "
                        f"expected {expected_valid.numel()}"
                    )
                if torch.unique(actual_valid).numel() != actual_valid.numel():
                    return False, f"    rank {rank} token {token} contains duplicate indices"

                physical_to_logical = _physical_to_logical(inputs, rank, token)
                if any(row not in physical_to_logical for row in actual_valid.tolist()):
                    return False, f"    rank {rank} token {token} selected a row outside its request"
                positions = [physical_to_logical[row] for row in actual_valid.tolist()]
                if any(left >= right for left, right in zip(positions, positions[1:])):
                    return False, f"    rank {rank} token {token} indices must follow logical position order"
                actual_set = set(actual_valid.tolist())
                expected_set = set(expected_valid.tolist())
                if actual_set == expected_set:
                    continue
                missing = sorted(expected_set - actual_set)
                extra = sorted(actual_set - expected_set)
                if scores is None:
                    score_outputs = actual_outputs if mode == "full" else expected_outputs
                    scores = _golden_index_scores(mode, rank, score_outputs, inputs)
                missing_logical = torch.tensor(
                    [physical_to_logical[row] for row in missing],
                    dtype=torch.int64,
                )
                extra_logical = torch.tensor(
                    [physical_to_logical[row] for row in extra],
                    dtype=torch.int64,
                )
                missing_scores = torch.sort(scores[token, missing_logical], descending=True).values
                extra_scores = torch.sort(scores[token, extra_logical], descending=True).values
                tolerance = TOPK_SCORE_ATOL + TOPK_SCORE_RTOL * missing_scores.abs()
                score_gap = missing_scores - extra_scores
                if torch.all(score_gap <= tolerance):
                    continue
                worst = int(torch.argmax(score_gap - tolerance))
                return False, (
                    f"    rank {rank} token {token} top-k cutoff differs: "
                    f"missing score {float(missing_scores[worst]):.8g}, "
                    f"replacement score {float(extra_scores[worst]):.8g}, "
                    f"tolerance {float(tolerance[worst]):.8g}"
                )
        return True, ""

    compare.__name__ = "numerical_cutoff_topk_compare"
    return compare


def _selected_output_reference(
    inputs: dict[str, torch.Tensor],
    expected_outputs: dict[str, torch.Tensor],
    selected: torch.Tensor,
) -> torch.Tensor:
    """Evaluate validated selections against reference cache values and original weights."""
    active = selected.shape[1]
    partials = []
    for rank in range(selected.shape[0]):
        def tensor(name: str) -> torch.Tensor:
            source = expected_outputs if name in expected_outputs else inputs
            return source[name][rank]

        query, _, _ = qkv_proj_rope(
            tensor("x")[:active], tensor("wq_a"), tensor("wq_a_scale"), tensor("q_norm_weight"),
            tensor("wq_b"), tensor("wq_b_scale"), tensor("wkv"), tensor("wkv_scale"),
            tensor("kv_norm_weight"), tensor("rope_cos")[:active], tensor("rope_sin")[:active],
        )
        window = dequantize_mxfp8_cache(tensor("window_cache"), tensor("window_cache_scale"))
        compressed = dequantize_mxfp4_cache(
            tensor("compressed_cache"), tensor("compressed_cache_scale"),
            group_size=C.COMPRESSED_CACHE_GROUP, scale_format="e4m3",
        )
        attended = golden_prefill_c1a_attention(
            query, window.to(query.dtype), tensor("window_indices")[:active],
            compressed.to(query.dtype), selected[rank], tensor("attn_sink"),
        )
        partials.append(_project_output(
            attended, tensor("rope_cos")[:active], tensor("rope_sin")[:active], tensor("wo_a"),
            tensor("wo_b"), tensor("wo_b_scale"), output_dtype=torch.float32,
        ))
    return _reduce_tp_partials(partials).unsqueeze(0).expand(selected.shape[0], -1, -1)


def _compare_attention_rows(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, str]:
    """Bound RMS and peak errors separately for every rank/token row."""
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.numel() == 0:
        return False, "    output shapes and dtypes must match and contain at least one row"
    actual = actual.double()
    expected = expected.double()
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        return False, "    output and reference must be finite"
    error = actual - expected
    reference_rms = expected.square().mean(dim=-1).sqrt()
    error_rms = error.square().mean(dim=-1).sqrt()
    error_peak = error.abs().amax(dim=-1)
    rms_limit = OUTPUT_RMS_ATOL + OUTPUT_RMS_RTOL * reference_rms
    peak_limit = OUTPUT_PEAK_ATOL + OUTPUT_PEAK_RMS_RATIO * reference_rms
    bad = (error_rms > rms_limit) | (error_peak > peak_limit)
    denominator = reference_rms.clamp_min(OUTPUT_RMS_ATOL)
    print(
        f"[PRECISION] output max_row_rel_l2={float((error_rms / denominator).max()):.8g} "
        f"max_peak_over_rms={float((error_peak / denominator).max()):.8g} "
        f"bad_rows={int(bad.sum())}/{bad.numel()}"
    )
    if not bool(bad.any()):
        return True, ""
    index = tuple(int(value) for value in torch.nonzero(bad, as_tuple=False)[0])
    return False, (
        f"    rank/token {index}: error RMS {float(error_rms[index]):.8g} "
        f"exceeds limit {float(rms_limit[index]):.8g} or peak {float(error_peak[index]):.8g} "
        f"exceeds limit {float(peak_limit[index]):.8g}; "
        f"reference RMS={float(reference_rms[index]):.8g}"
    )


def attention_output_compare(mode: str) -> Callable:
    """Validate selection quality before comparing the selected output with strict row bounds."""
    if mode not in ("full", "reindex", "reuse"):
        raise ValueError(f"unsupported C1A attention mode: {mode}")

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        actual_outputs: dict[str, torch.Tensor],
        expected_outputs: dict[str, torch.Tensor],
        inputs: dict[str, torch.Tensor],
        rtol: float,
        atol: float,
    ) -> tuple[bool, str]:
        active = min(int(inputs.get("num_tokens", actual.shape[1])), actual.shape[1])
        reference = expected
        if mode != "reuse":
            selected = actual_outputs["topk_indices"]
            nominal = expected_outputs["topk_indices"]
            valid, detail = topk_indices_compare(mode)(
                selected, nominal, actual_outputs=actual_outputs,
                expected_outputs=expected_outputs, inputs=inputs, rtol=rtol, atol=atol,
            )
            if not valid:
                return False, f"    output reference rejected invalid Top-K: {detail.strip()}"
            if not torch.equal(selected[:, :active], nominal[:, :active]):
                reference = _selected_output_reference(inputs, expected_outputs, selected[:, :active])
        return _compare_attention_rows(actual[:, :active], reference[:, :active])

    compare.__name__ = "c1a_output_row_rms_and_peak"
    return compare


HC_ROW_BUDGET = 2.0**-8


def _report_hc_stage(name: str, actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Report global and worst token-row relative L2 for one HC stage."""
    diff = actual.double() - expected.double()
    reference = expected.double()
    rel_l2 = (diff.norm() / reference.norm().clamp_min(1e-12)).item()
    rows = diff.flatten(1).norm(dim=-1) / reference.flatten(1).norm(dim=-1).clamp_min(1e-12)
    worst_row = rows.max().item()
    print(f"[PRECISION] {name} rel_l2={rel_l2:.8g} max_row_rel_l2={worst_row:.8g}")
    return rel_l2, worst_row


def _tp_replicated(actual: torch.Tensor) -> bool:
    return all(torch.equal(actual[0], actual[rank]) for rank in range(1, C.TP_SIZE))


def hc_hidden_compare() -> Callable:
    """Validate mHC collapse precision per token row and TP replication."""
    from golden import ratio_allclose

    close = ratio_allclose(atol=1e-4, rtol=1.0 / 128)

    def compare(actual: torch.Tensor, expected: torch.Tensor, **kwargs) -> tuple[bool, str]:
        _, worst_row = _report_hc_stage("hidden", actual[0], expected[0])
        elementwise, _ = close(actual[0], expected[0], **kwargs)
        passed = elementwise and worst_row <= HC_ROW_BUDGET and _tp_replicated(actual)
        return passed, f"hc_pre every token row <= {HC_ROW_BUDGET:.3g} rel L2; TP replicas identical"

    return compare


def attn_input_compare() -> Callable:
    """Validate the post-collapse attention RMSNorm boundary per token row."""
    from golden import ratio_allclose

    close = ratio_allclose(atol=1e-4, rtol=1.0 / 128)

    def compare(actual: torch.Tensor, expected: torch.Tensor, **kwargs) -> tuple[bool, str]:
        _, worst_row = _report_hc_stage("attn_input", actual[0], expected[0])
        elementwise, _ = close(actual[0], expected[0], **kwargs)
        passed = elementwise and worst_row <= HC_ROW_BUDGET and _tp_replicated(actual)
        return passed, f"attention RMSNorm every token row <= {HC_ROW_BUDGET:.3g} rel L2; TP replicas identical"

    return compare


def hc_output_compare() -> Callable:
    """Replay mHC post on device attention output and bound full-chain relative L2."""
    from golden import ratio_allclose
    from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes
    from models.deepseek_v4_1_flash.hc_post import golden_mhc_post

    close = ratio_allclose(atol=1e-4, rtol=1.0 / 128)

    def compare(
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        inputs: dict[str, torch.Tensor],
        actual_outputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> tuple[bool, str]:
        _, post_mix, residual_mix = golden_mhc_mixes(
            inputs["x_hc"][0], inputs["hc_attn_fn"][0],
            inputs["hc_attn_scale"][0], inputs["hc_attn_base"][0],
        )
        replay = golden_mhc_post(
            actual_outputs["attn_out"][0], inputs["x_hc"][0], post_mix, residual_mix
        )
        _, worst_row = _report_hc_stage("output(hc_post replay)", actual[0], replay)
        elementwise, _ = close(actual[0], replay, inputs=inputs, actual_outputs=actual_outputs, **kwargs)
        end_to_end, _ = _report_hc_stage("output(end-to-end)", actual[0], expected[0])
        passed = (
            elementwise
            and worst_row <= HC_ROW_BUDGET
            and end_to_end <= 0.01
            and _tp_replicated(actual)
        )
        return passed, "hc_post replay within row budget; end-to-end rel L2 <= 1%; TP replicas identical"

    return compare
