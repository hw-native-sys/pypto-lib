# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim    # A2/A3-only driver; a2a3sim remains available explicitly with --compile-only.
"""Dynamic correctness, codegen, and raw-performance driver for PyPTO Qwen PA.

The handwritten CCE backend and its existing validation remain available. This
component driver focuses on the checks needed to validate the PyPTO
implementation used by production decode:

* dynamic batch and sequence-length shapes;
* page-boundary, ragged, remapped, shared-prefix, and packed-layer addressing;
* Q/K head norm, RoPE, V scaling, current-token cache append, and PA output;
* Torch numerical comparison on a real NPU, including cache canaries;
* full wrapper codegen on ``a2a3sim``;
* optional raw timing samples, with no performance pass/fail threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pypto.language as pl
import torch

from golden import ScalarSpec, TensorSpec, run_jit
from paged_attention_pypto import (
    BATCH,
    BLOCK_SIZE,
    EPS,
    FFTS_WORKSPACE_ELEMENTS,
    GROUP,
    HALF_DIM,
    HEAD_DIM,
    HEAD_DIM_INV,
    HIDDEN,
    KV_HIDDEN,
    NUM_HEADS,
    NUM_KV_HEADS,
    PRE_LAUNCH,
    SCALE,
    STACK_PAGES,
    STACK_TOKENS,
    TRANSFER_ROWS,
    TRANSFER_SLOTS,
    paged_attention_pypto_swpipe,
)

MAX_CONTEXT_LENGTH = 4096
EXTRA_PHYSICAL_PAGES = 17
PAGE_MAPPINGS = ("identity", "random", "noncontiguous", "reverse", "shared-prefix")
DATA_PATTERNS = ("random", "zero-query", "kv-head-id", "dominant-score")
MATRIX_SCHEMA = "qwen3-fused-paged-attention-pypto-matrix-v2"

PAGE_BOUNDARY_SEQ_LENS = (
    127,
    128,
    129,
    255,
    256,
    257,
    383,
    384,
    385,
    511,
    512,
    513,
    3967,
    3968,
    3969,
    4096,
)
RAGGED_SEQ_LENS = (
    1,
    2,
    127,
    128,
    129,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    1536,
    2048,
    3584,
    4095,
    4096,
)


@dataclass(frozen=True)
class DynamicCase:
    """One deterministic dynamic paged-attention workload."""

    name: str
    seq_lens: tuple[int, ...]
    capacity: int = MAX_CONTEXT_LENGTH
    physical_pages: int | None = None
    page_mapping: str = "identity"
    pattern: str = "random"
    seed: int = 1234
    cache_layers: int = 1
    layer_idx: int = 0
    canary_salt: int = 0

    @property
    def batch(self) -> int:
        return len(self.seq_lens)

    @property
    def max_blocks_per_seq(self) -> int:
        return math.ceil(self.capacity / BLOCK_SIZE)

    @property
    def resolved_physical_pages(self) -> int:
        if self.physical_pages is not None:
            return self.physical_pages
        return self.batch * self.max_blocks_per_seq + EXTRA_PHYSICAL_PAGES

    @property
    def layer_cache_token_rows(self) -> int:
        return self.resolved_physical_pages * BLOCK_SIZE

    @property
    def layer_cache_base_token_rows(self) -> int:
        return self.layer_idx * self.layer_cache_token_rows

    def summary(self) -> dict[str, object]:
        return {
            "name": self.name,
            "batch": self.batch,
            "seq_lens": list(self.seq_lens),
            "capacity": self.capacity,
            "max_blocks_per_seq": self.max_blocks_per_seq,
            "physical_pages": self.resolved_physical_pages,
            "page_mapping": self.page_mapping,
            "pattern": self.pattern,
            "seed": self.seed,
            "cache_layers": self.cache_layers,
            "layer_idx": self.layer_idx,
            "layer_cache_base_token_rows": self.layer_cache_base_token_rows,
        }


@pl.jit
def paged_attention_pypto_dynamic(
    key_cache: pl.InOut[pl.Tensor],
    value_cache: pl.InOut[pl.Tensor],
    block_table: pl.Tensor,
    seq_lens: pl.Tensor,
    inv_rms_states: pl.Tensor,
    slot_mapping: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    q_proj: pl.Tensor,
    k_proj: pl.Tensor,
    v_proj: pl.Tensor,
    q_norm_w: pl.Tensor,
    k_norm_w: pl.Tensor,
    layer_cache_base_token_rows: pl.Scalar[pl.INDEX],
    out: pl.Out[pl.Tensor],
) -> pl.Tensor:
    """Allocate fused Phase-0/PA scratch and launch the PyPTO helper."""
    active_batch = pl.tensor.dim(seq_lens, 0)
    q_tnd_flat = pl.create_tensor([active_batch * NUM_HEADS, HEAD_DIM], dtype=pl.BF16)
    score_transfer = pl.create_tensor([TRANSFER_ROWS, STACK_TOKENS], dtype=pl.FP32)
    probability_transfer = pl.create_tensor([TRANSFER_ROWS, STACK_TOKENS], dtype=pl.BF16)
    pv_transfer = pl.create_tensor([TRANSFER_ROWS, HEAD_DIM], dtype=pl.FP32)
    ffts_workspace = pl.create_tensor([FFTS_WORKSPACE_ELEMENTS], dtype=pl.INT64)
    q_proj_tid = pl.system.task_dummy(deps=[])
    k_proj_tid = pl.system.task_dummy(deps=[])
    v_proj_tid = pl.system.task_dummy(deps=[])
    rms_tid = pl.system.task_dummy(deps=[])
    attn_out_seed_tid = pl.system.task_dummy(deps=[])
    mlp_out_seed_tid = pl.system.task_dummy(deps=[])
    scratch_ready_tid = pl.system.task_dummy(deps=[])
    paged_attention_pypto_swpipe(
        q_tnd_flat,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        inv_rms_states,
        slot_mapping,
        rope_cos,
        rope_sin,
        q_proj,
        k_proj,
        v_proj,
        q_norm_w,
        k_norm_w,
        layer_cache_base_token_rows,
        out,
        score_transfer,
        probability_transfer,
        pv_transfer,
        ffts_workspace,
        q_proj_tid,
        k_proj_tid,
        v_proj_tid,
        rms_tid,
        attn_out_seed_tid,
        mlp_out_seed_tid,
        scratch_ready_tid,
    )
    return out


def _validate_case(case: DynamicCase) -> None:
    if not case.name:
        raise ValueError("case name must not be empty")
    if not 1 <= case.batch <= BATCH:
        raise ValueError(f"batch must be in [1, {BATCH}]")
    if not 1 <= case.capacity <= MAX_CONTEXT_LENGTH:
        raise ValueError(f"capacity must be in [1, {MAX_CONTEXT_LENGTH}]")
    if any(not 1 <= seq_len <= case.capacity for seq_len in case.seq_lens):
        raise ValueError("every sequence length must be in [1, capacity]")
    if case.page_mapping not in PAGE_MAPPINGS:
        raise ValueError(f"unsupported page mapping: {case.page_mapping}")
    if case.pattern not in DATA_PATTERNS:
        raise ValueError(f"unsupported data pattern: {case.pattern}")
    if case.cache_layers < 1 or not 0 <= case.layer_idx < case.cache_layers:
        raise ValueError("layer_idx must select one supplied cache layer")
    if case.canary_salt < 0:
        raise ValueError("canary_salt must be non-negative")


def _coprime_step(size: int) -> int:
    step = max(3, size // 2) | 1
    while math.gcd(step, size) != 1:
        step += 2
    return step


def _page_order(case: DynamicCase) -> list[int]:
    pages = case.resolved_physical_pages
    if case.page_mapping == "identity":
        return list(range(pages))
    if case.page_mapping == "reverse":
        return list(range(pages - 1, -1, -1))
    if case.page_mapping == "noncontiguous":
        step = _coprime_step(pages)
        offset = (case.seed * 17 + 3) % pages
        return [(offset + index * step) % pages for index in range(pages)]
    generator = torch.Generator().manual_seed(case.seed ^ 0x5A17)
    return torch.randperm(pages, generator=generator, dtype=torch.int64).tolist()


def _make_page_layout(case: DynamicCase) -> tuple[torch.Tensor, tuple[tuple[int, ...], ...]]:
    """Create a flat table and the active physical-page rows it describes."""
    _validate_case(case)
    order = _page_order(case)
    page_counts = [math.ceil(seq_len / BLOCK_SIZE) for seq_len in case.seq_lens]
    active_rows: list[tuple[int, ...]] = []
    cursor = 0
    if case.page_mapping == "shared-prefix":
        if min(page_counts) < 2:
            raise ValueError("shared-prefix requires at least two pages per request")
        shared_count = min(4, min(page_counts) - 1)
        prefix = tuple(order[:shared_count])
        cursor = shared_count
        for page_count in page_counts:
            private_count = page_count - shared_count
            private = tuple(order[cursor : cursor + private_count])
            cursor += private_count
            active_rows.append((*prefix, *private))
    else:
        for page_count in page_counts:
            row = tuple(order[cursor : cursor + page_count])
            cursor += page_count
            active_rows.append(row)

    if any(len(row) != count for row, count in zip(active_rows, page_counts, strict=True)):
        raise ValueError("physical page pool is too small for active logical pages")

    active_pages = {page for row in active_rows for page in row}
    unused_pages = [page for page in order if page not in active_pages] or [0]
    table = torch.empty([case.batch, case.max_blocks_per_seq], dtype=torch.int32)
    for batch_idx, row in enumerate(active_rows):
        for logical_page in range(case.max_blocks_per_seq):
            index = (batch_idx * case.max_blocks_per_seq + logical_page + case.canary_salt) % len(
                unused_pages
            )
            table[batch_idx, logical_page] = unused_pages[index]
        table[batch_idx, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return table.reshape(-1).contiguous(), tuple(active_rows)


def _cache_view(value: torch.Tensor, case: DynamicCase) -> torch.Tensor:
    return value.view(
        case.cache_layers,
        case.resolved_physical_pages,
        BLOCK_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
    )


def _apply_pattern(
    fixture: dict[str, torch.Tensor],
    case: DynamicCase,
    active_rows: tuple[tuple[int, ...], ...],
) -> None:
    key = _cache_view(fixture["key_cache"], case)[case.layer_idx]
    value = _cache_view(fixture["value_cache"], case)[case.layer_idx]
    active_pages = sorted({page for row in active_rows for page in row})
    if case.pattern == "random":
        return
    if case.pattern == "zero-query":
        fixture["q_proj"].zero_()
        return
    if case.pattern == "kv-head-id":
        key[active_pages] = 0.0
        for kv_head in range(NUM_KV_HEADS):
            value[active_pages, :, kv_head] = float(kv_head + 1)
            col = slice(kv_head * HEAD_DIM, (kv_head + 1) * HEAD_DIM)
            fixture["v_proj"][:, col] = float(kv_head + 1) / fixture["inv_rms_states"]
        return
    if case.pattern == "dominant-score":
        if case.page_mapping == "shared-prefix":
            raise ValueError("dominant-score requires private active pages")
        generator = torch.Generator().manual_seed(case.seed ^ 0xD011)
        bases = torch.empty([case.batch, NUM_KV_HEADS, HEAD_DIM], dtype=torch.float32).normal_(
            generator=generator
        )
        fixture["q_norm_w"].fill_(1.0)
        fixture["k_norm_w"].fill_(1.0)
        fixture["q_proj"].view(case.batch, NUM_KV_HEADS, GROUP, HEAD_DIM).copy_(
            bases.unsqueeze(2).expand(-1, -1, GROUP, -1)
        )
        fixture["k_proj"].view(case.batch, NUM_KV_HEADS, HEAD_DIM).copy_(bases)
        query, _, _ = _phase0_rows(fixture, case)
        for batch_idx, row in enumerate(active_rows):
            key[list(row)] = -query[batch_idx, ::GROUP].view(1, 1, NUM_KV_HEADS, HEAD_DIM)
        return
    raise AssertionError(f"unhandled pattern: {case.pattern}")


def _rope_tables(capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(capacity, dtype=torch.float32).view(-1, 1)
    frequencies = torch.pow(
        torch.tensor(10000.0, dtype=torch.float32),
        -torch.arange(HALF_DIM, dtype=torch.float32) / HALF_DIM,
    ).view(1, -1)
    angles = positions * frequencies
    return (
        torch.cat((angles.cos(), angles.cos()), dim=1).contiguous(),
        torch.cat((angles.sin(), angles.sin()), dim=1).contiguous(),
    )


def _phase0_rows(
    values: dict[str, torch.Tensor],
    case: DynamicCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Torch oracle for the fused Q/K head norm, RoPE, and V scaling producer."""
    inv_rms = values["inv_rms_states"].float().view(case.batch, 1, 1)
    q = values["q_proj"].float().view(case.batch, NUM_HEADS, HEAD_DIM) * inv_rms
    k = values["k_proj"].float().view(case.batch, NUM_KV_HEADS, HEAD_DIM) * inv_rms
    v = values["v_proj"].float().view(case.batch, NUM_KV_HEADS, HEAD_DIM) * inv_rms
    q = q * torch.rsqrt(q.square().sum(dim=-1, keepdim=True) * HEAD_DIM_INV + EPS)
    k = k * torch.rsqrt(k.square().sum(dim=-1, keepdim=True) * HEAD_DIM_INV + EPS)
    q = q * values["q_norm_w"].float().view(1, 1, HEAD_DIM)
    k = k * values["k_norm_w"].float().view(1, 1, HEAD_DIM)
    positions = values["seq_lens"].long() - 1
    cos = values["rope_cos"][positions].float()
    sin = values["rope_sin"][positions].float()

    def rotate(rows: torch.Tensor) -> torch.Tensor:
        lo, hi = rows[..., :HALF_DIM], rows[..., HALF_DIM:]
        cos_lo, cos_hi = cos[:, None, :HALF_DIM], cos[:, None, HALF_DIM:]
        sin_lo, sin_hi = sin[:, None, :HALF_DIM], sin[:, None, HALF_DIM:]
        return torch.cat((lo * cos_lo - hi * sin_lo, hi * cos_hi + lo * sin_hi), dim=-1)

    return rotate(q).to(torch.bfloat16), rotate(k).to(torch.bfloat16), v.to(torch.bfloat16)


def make_fixture(case: DynamicCase) -> dict[str, torch.Tensor]:
    table, active_rows = _make_page_layout(case)
    generator = torch.Generator().manual_seed(case.seed)
    rope_cos, rope_sin = _rope_tables(case.capacity)
    cache_shape = [
        case.cache_layers * case.resolved_physical_pages * BLOCK_SIZE,
        KV_HIDDEN,
    ]
    fixture = {
        "key_cache": torch.empty(cache_shape, dtype=torch.bfloat16).normal_(generator=generator).mul_(0.1),
        "value_cache": torch.empty(cache_shape, dtype=torch.bfloat16).normal_(generator=generator).mul_(0.2),
        "block_table": table,
        "seq_lens": torch.tensor(case.seq_lens, dtype=torch.int32),
        "inv_rms_states": torch.empty([case.batch, 1], dtype=torch.float32).uniform_(
            0.75, 1.25, generator=generator
        ),
        "slot_mapping": torch.tensor(
            [
                row[(seq_len - 1) // BLOCK_SIZE] * BLOCK_SIZE + (seq_len - 1) % BLOCK_SIZE
                for seq_len, row in zip(case.seq_lens, active_rows, strict=True)
            ],
            dtype=torch.int32,
        ),
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "q_proj": torch.empty([case.batch, HIDDEN], dtype=torch.float32).normal_(generator=generator),
        "k_proj": torch.empty([case.batch, KV_HIDDEN], dtype=torch.float32).normal_(generator=generator),
        "v_proj": torch.empty([case.batch, KV_HIDDEN], dtype=torch.float32).normal_(generator=generator),
        "q_norm_w": torch.empty([1, HEAD_DIM], dtype=torch.float32).uniform_(0.8, 1.2, generator=generator),
        "k_norm_w": torch.empty([1, HEAD_DIM], dtype=torch.float32).uniform_(0.8, 1.2, generator=generator),
    }
    key_layers = _cache_view(fixture["key_cache"], case)
    value_layers = _cache_view(fixture["value_cache"], case)
    for layer_idx in range(case.cache_layers):
        if layer_idx != case.layer_idx:
            key_layers[layer_idx].fill_(17.0 + layer_idx)
            value_layers[layer_idx].fill_(-19.0 - layer_idx)
    _apply_pattern(fixture, case, active_rows)
    validate_fixture(fixture, case)
    return fixture


def validate_fixture(fixture: dict[str, torch.Tensor], case: DynamicCase) -> None:
    expected = {
        "key_cache",
        "value_cache",
        "block_table",
        "seq_lens",
        "inv_rms_states",
        "slot_mapping",
        "rope_cos",
        "rope_sin",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm_w",
        "k_norm_w",
    }
    if set(fixture) != expected:
        raise ValueError(f"fixture tensor names differ: {sorted(fixture)}")
    key_cache = fixture["key_cache"]
    value_cache = fixture["value_cache"]
    table = fixture["block_table"]
    seq_lens = fixture["seq_lens"]
    if any(not value.is_contiguous() for value in fixture.values()):
        raise ValueError("every fixture tensor must be contiguous")
    expected_cache_shape = (
        case.cache_layers * case.resolved_physical_pages * BLOCK_SIZE,
        KV_HIDDEN,
    )
    if key_cache.dtype != torch.bfloat16 or tuple(key_cache.shape) != expected_cache_shape:
        raise ValueError("key cache shape/dtype differs from the case")
    if value_cache.dtype != torch.bfloat16 or value_cache.shape != key_cache.shape:
        raise ValueError("value cache shape/dtype differs from key cache")
    if table.dtype != torch.int32 or table.numel() != case.batch * case.max_blocks_per_seq:
        raise ValueError("block table must be flat INT32 [batch * row_stride]")
    if seq_lens.dtype != torch.int32 or tuple(seq_lens.shape) != (case.batch,):
        raise ValueError("seq_lens must contain one INT32 value per request")
    if torch.any(table < 0) or torch.any(table >= case.resolved_physical_pages):
        raise ValueError("block table contains a page outside the selected layer")
    if fixture["inv_rms_states"].dtype != torch.float32 or tuple(fixture["inv_rms_states"].shape) != (
        case.batch,
        1,
    ):
        raise ValueError("inv_rms_states must be FP32 [batch, 1]")
    if fixture["slot_mapping"].dtype != torch.int32 or tuple(fixture["slot_mapping"].shape) != (case.batch,):
        raise ValueError("slot_mapping must be INT32 [batch]")
    for name in ("rope_cos", "rope_sin"):
        if fixture[name].dtype != torch.float32 or tuple(fixture[name].shape) != (
            case.capacity,
            HEAD_DIM,
        ):
            raise ValueError(f"{name} must be FP32 [capacity, head_dim]")
    for name, width in (("q_proj", HIDDEN), ("k_proj", KV_HIDDEN), ("v_proj", KV_HIDDEN)):
        if fixture[name].dtype != torch.float32 or tuple(fixture[name].shape) != (
            case.batch,
            width,
        ):
            raise ValueError(f"{name} shape/dtype differs from the case")
    for name in ("q_norm_w", "k_norm_w"):
        if fixture[name].dtype != torch.float32 or tuple(fixture[name].shape) != (1, HEAD_DIM):
            raise ValueError(f"{name} must be FP32 [1, head_dim]")


def golden_attention(values: dict[str, torch.Tensor], case: DynamicCase) -> None:
    """Run fused Phase 0 then ragged GQA attention against the paged cache ABI."""
    query_bf16, current_key, current_value = _phase0_rows(values, case)
    query = query_bf16.float()
    key_cache = _cache_view(values["key_cache"], case)[case.layer_idx]
    value_cache = _cache_view(values["value_cache"], case)[case.layer_idx]
    table = values["block_table"].view(case.batch, case.max_blocks_per_seq).long()
    out = values["out"]
    for batch_idx, slot in enumerate(values["slot_mapping"].long()):
        page = int(slot.item()) // BLOCK_SIZE
        offset = int(slot.item()) % BLOCK_SIZE
        key_cache[page, offset] = current_key[batch_idx]
        value_cache[page, offset] = current_value[batch_idx]
    for batch_idx, seq_len in enumerate(case.seq_lens):
        pages = table[batch_idx, : math.ceil(seq_len / BLOCK_SIZE)]
        for kv_head in range(NUM_KV_HEADS):
            key = key_cache[pages, :, kv_head].reshape(-1, HEAD_DIM)[:seq_len].float()
            value = value_cache[pages, :, kv_head].reshape(-1, HEAD_DIM)[:seq_len].float()
            q_start = kv_head * GROUP
            q_group = query[batch_idx, q_start : q_start + GROUP]
            probability = torch.softmax((q_group @ key.t()) * SCALE, dim=-1)
            out[batch_idx, q_start : q_start + GROUP] = (probability @ value).to(torch.bfloat16)


def _compare_cache(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    case: DynamicCase,
    **_: Any,
) -> tuple[bool, str]:
    """Require exact canaries outside current-token rows and close Phase-0 writes."""
    target = torch.zeros(actual.shape[0], dtype=torch.bool)
    _, rows = _make_page_layout(case)
    for seq_len, row in zip(case.seq_lens, rows, strict=True):
        position = seq_len - 1
        slot = row[position // BLOCK_SIZE] * BLOCK_SIZE + position % BLOCK_SIZE
        target[case.layer_cache_base_token_rows + slot] = True
    if not torch.equal(actual[~target], expected[~target]):
        changed = int(torch.count_nonzero(actual[~target] != expected[~target]).item())
        return False, f"non-target cache canary changed at {changed} elements"
    if not torch.allclose(actual[target], expected[target], rtol=5.0e-3, atol=2.0e-2):
        max_abs = float((actual[target].float() - expected[target].float()).abs().max().item())
        return False, f"current-token cache append differs: max_abs={max_abs}"
    return True, ""


def _compare_output(actual: torch.Tensor, expected: torch.Tensor, **_: Any) -> tuple[bool, str]:
    if not torch.isfinite(actual.float()).all():
        return False, "attention output contains NaN or Inf"
    if not torch.allclose(actual, expected, rtol=5.0e-3, atol=2.0e-2):
        max_abs = float((actual.float() - expected.float()).abs().max().item())
        return False, f"attention output differs from Torch oracle: max_abs={max_abs}"
    return True, ""


def build_specs(
    case: DynamicCase,
    fixture: dict[str, torch.Tensor] | None,
) -> list[TensorSpec | ScalarSpec]:
    cache_shape = [
        case.cache_layers * case.resolved_physical_pages * BLOCK_SIZE,
        KV_HIDDEN,
    ]

    def initializer(name: str):
        return None if fixture is None else lambda: fixture[name].clone()

    return [
        TensorSpec(
            "key_cache",
            cache_shape,
            torch.bfloat16,
            init_value=initializer("key_cache"),
            is_output=True,
        ),
        TensorSpec(
            "value_cache",
            cache_shape,
            torch.bfloat16,
            init_value=initializer("value_cache"),
            is_output=True,
        ),
        TensorSpec(
            "block_table",
            [case.batch * case.max_blocks_per_seq],
            torch.int32,
            init_value=initializer("block_table"),
        ),
        TensorSpec(
            "seq_lens",
            [case.batch],
            torch.int32,
            init_value=initializer("seq_lens"),
        ),
        TensorSpec(
            "inv_rms_states",
            [case.batch, 1],
            torch.float32,
            init_value=initializer("inv_rms_states"),
        ),
        TensorSpec(
            "slot_mapping",
            [case.batch],
            torch.int32,
            init_value=initializer("slot_mapping"),
        ),
        TensorSpec(
            "rope_cos",
            [case.capacity, HEAD_DIM],
            torch.float32,
            init_value=initializer("rope_cos"),
        ),
        TensorSpec(
            "rope_sin",
            [case.capacity, HEAD_DIM],
            torch.float32,
            init_value=initializer("rope_sin"),
        ),
        TensorSpec(
            "q_proj",
            [case.batch, HIDDEN],
            torch.float32,
            init_value=initializer("q_proj"),
        ),
        TensorSpec(
            "k_proj",
            [case.batch, KV_HIDDEN],
            torch.float32,
            init_value=initializer("k_proj"),
        ),
        TensorSpec(
            "v_proj",
            [case.batch, KV_HIDDEN],
            torch.float32,
            init_value=initializer("v_proj"),
        ),
        TensorSpec(
            "q_norm_w",
            [1, HEAD_DIM],
            torch.float32,
            init_value=initializer("q_norm_w"),
        ),
        TensorSpec(
            "k_norm_w",
            [1, HEAD_DIM],
            torch.float32,
            init_value=initializer("k_norm_w"),
        ),
        ScalarSpec(
            "layer_cache_base_token_rows",
            torch.int64,
            case.layer_cache_base_token_rows,
        ),
        TensorSpec(
            "out",
            [case.batch, NUM_HEADS, HEAD_DIM],
            torch.bfloat16,
            is_output=True,
        ),
    ]


def pr_cases() -> tuple[DynamicCase, ...]:
    """Small PR gate covering the distinct correctness dimensions."""
    return (
        DynamicCase("b1-s1-gqa", (1,), page_mapping="reverse", pattern="kv-head-id", seed=0),
        DynamicCase("b2-page-edge", (127, 129), page_mapping="random", seed=1),
        DynamicCase("b8-s257-dominant", (257,) * 8, page_mapping="noncontiguous", pattern="dominant-score"),
        DynamicCase("b16-page-boundary", PAGE_BOUNDARY_SEQ_LENS, page_mapping="random"),
        DynamicCase("b16-ragged", RAGGED_SEQ_LENS, page_mapping="reverse", seed=2026),
        DynamicCase("b16-s4096-shared", (4096,) * 16, page_mapping="shared-prefix", seed=2026),
        DynamicCase(
            "b2-packed-layer-middle",
            (383, 513),
            capacity=640,
            page_mapping="noncontiguous",
            cache_layers=3,
            layer_idx=1,
            seed=2026,
        ),
    )


def full_cases() -> tuple[DynamicCase, ...]:
    """Focused pairwise nightly matrix instead of the old 375-case Cartesian product."""
    batches = (1, 8, 16)
    lengths = (1, 127, 128, 129, 511, 512, 513, 4095, 4096)
    mappings = ("identity", "random", "noncontiguous", "reverse")
    patterns = ("random", "zero-query", "kv-head-id", "dominant-score")
    core = tuple(
        DynamicCase(
            f"full-b{batch}-s{seq_len}",
            (seq_len,) * batch,
            page_mapping=mappings[(batch + seq_len) % len(mappings)],
            pattern=patterns[(batch * 3 + seq_len) % len(patterns)],
            seed=1000 + batch * 17 + seq_len,
        )
        for batch in batches
        for seq_len in lengths
    )
    names = {case.name for case in core}
    return (*core, *(case for case in pr_cases() if case.name not in names))


def matrix_cases(name: str) -> tuple[DynamicCase, ...]:
    if name == "pr":
        return pr_cases()
    if name == "full":
        return full_cases()
    raise ValueError(f"unsupported matrix: {name}")


def _compile_signature(case: DynamicCase) -> tuple[int, ...]:
    return (
        case.batch,
        case.capacity,
        case.cache_layers,
        case.resolved_physical_pages,
        case.max_blocks_per_seq,
        case.layer_cache_base_token_rows,
    )


def _resolve_pass_dump(work_dir: Path, semantic_name: str) -> Path:
    passes_dir = work_dir / "passes_dump"
    matches = sorted(passes_dir.glob(f"*_{semantic_name}.py"))
    if len(matches) == 1:
        return matches[0]
    available = sorted(path.name for path in passes_dir.glob("*.py"))
    raise RuntimeError(
        f"expected one '*_{semantic_name}.py' in {passes_dir}, "
        f"found {[path.name for path in matches]}; available={available}"
    )


def assert_codegen_artifact(work_dir: Path) -> dict[str, object]:
    """Check stable semantic artifacts without pinning compiler pass ordinals."""
    frontend = _resolve_pass_dump(work_dir, "frontend")
    split = _resolve_pass_dump(work_dir, "after_ExpandMixedKernel")
    allocation = _resolve_pass_dump(work_dir, "after_AllocateMemoryAddr")
    pto = work_dir / "ptoas" / "attn_swpipe.pto"
    aiv_kernel = work_dir / "kernels" / "aiv" / "attn_swpipe_aiv.cpp"
    orchestration = sorted((work_dir / "orchestration").glob("*.cpp"))
    errors = work_dir / "report" / "codegen_errors.txt"
    for path in (frontend, split, allocation, pto, aiv_kernel):
        if not path.is_file():
            raise RuntimeError(f"required PA codegen artifact is missing: {path}")
    if len(orchestration) != 1:
        raise RuntimeError("PA build must contain one orchestration source")
    if errors.is_file() and errors.read_text().strip():
        raise RuntimeError(f"PA codegen reported errors: {errors}")
    frontend_text = frontend.read_text()
    allocation_text = allocation.read_text()
    if frontend_text.count("pl.tensor.create(") != 5:
        raise RuntimeError("fused PA must allocate q_tnd plus one four-tensor scratch set")
    for name in (
        "q_tnd_flat",
        "score_transfer",
        "probability_transfer",
        "pv_transfer",
        "ffts_workspace",
    ):
        if name not in allocation_text:
            raise RuntimeError(f"allocated PA artifact is missing scratch tensor {name}")
    source_text = frontend_text + split.read_text() + pto.read_text()
    if "paged_attention_cce" in source_text or "fa_fused" in source_text:
        raise RuntimeError("PyPTO PA artifact contains a legacy CCE marker")
    if "rope_qkv" in source_text:
        raise RuntimeError("fused PA artifact still contains a standalone rope task")
    pto_source = pto.read_text()
    aic_fence = pto_source.find("pto.fence.barrier_all #pto.fence_scope<gm>")
    aic_syncall = pto_source.find("pto.syncall()", aic_fence)
    aiv_fence = pto_source.find("pto.fence.barrier_all #pto.fence_scope<gm>", aic_syncall)
    aiv_syncall = pto_source.find("pto.syncall()", aiv_fence)
    if min(aic_fence, aic_syncall, aiv_fence, aiv_syncall) < 0 or "sync_core_type<mix>" not in pto_source:
        raise RuntimeError("fused PA artifact is missing the Phase-0 GM fence + hard mixed-core barrier")
    return {
        "frontend": frontend.name,
        "post_split": split.name,
        "post_allocation": allocation.name,
        "pto": str(pto.relative_to(work_dir)),
    }


def _summary(samples: Sequence[float]) -> dict[str, float]:
    return {
        "min": min(samples),
        "median": statistics.median(samples),
        "mean": statistics.fmean(samples),
        "max": max(samples),
    }


def _raw_benchmark(result: Any, case: DynamicCase, platform: str, device: int) -> dict[str, object]:
    stats = result.bench
    if stats is None:
        raise RuntimeError("--benchmark-json requires PYPTO_BENCH=1 and runtime timing support")
    device_wall = [float(value) for value in stats.device_wall_us]
    host_wall = [float(value) for value in stats.host_wall_us]
    effective = [float(value) for value in stats.per_round("effective")]
    return {
        "schema": "qwen3-fused-paged-attention-pypto-raw-benchmark-v2",
        "backend": "pypto",
        "platform": platform,
        "device": device,
        "case": case.summary(),
        "warmup": int(stats.warmup),
        "rounds": int(stats.rounds),
        "samples_us": {
            "device_wall": device_wall,
            "host_wall": host_wall,
            "effective": effective,
        },
        "summary_us": {
            "device_wall": _summary(device_wall),
            "host_wall": _summary(host_wall),
            "effective": _summary(effective),
        },
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    temporary.replace(path)


def _run_case(case: DynamicCase, args: argparse.Namespace) -> dict[str, object]:
    _validate_case(case)
    compile_only = args.compile_only or args.platform.endswith("sim")
    fixture = None if compile_only else make_fixture(case)
    result = run_jit(
        fn=paged_attention_pypto_dynamic,
        specs=build_specs(case, fixture),
        golden_fn=None if fixture is None else lambda values: golden_attention(values, case),
        compile_cfg={"dump_passes": True},
        runtime_cfg={"platform": args.platform, "device_id": args.device},
        compile_only=compile_only,
        rtol=5.0e-3,
        atol=2.0e-2,
        compare_fn=None
        if fixture is None
        else {
            "out": _compare_output,
            "key_cache": lambda actual, expected, **kwargs: _compare_cache(
                actual, expected, case=case, **kwargs
            ),
            "value_cache": lambda actual, expected, **kwargs: _compare_cache(
                actual, expected, case=case, **kwargs
            ),
        },
        save_data=False,
    )
    if not result.passed:
        raise RuntimeError(result.error or f"case failed: {case.name}")
    if result.work_dir is None:
        raise RuntimeError("successful PA compile did not return its build directory")
    report: dict[str, object] = {
        "case": case.summary(),
        "status": "compile-only" if compile_only else "passed",
        "compile_only": compile_only,
        "work_dir": str(result.work_dir),
        "artifact": assert_codegen_artifact(Path(result.work_dir)),
        "execution_time_seconds": result.execution_time,
    }
    if args.benchmark_json is not None:
        report["benchmark"] = _raw_benchmark(result, case, args.platform, args.device)
        _write_json_atomic(args.benchmark_json, report["benchmark"])
    print(
        f"[test_paged_attention_pypto] case={case.name} status={report['status']} work_dir={result.work_dir}"
    )
    return report


def _run_matrix(args: argparse.Namespace) -> dict[str, object]:
    cases = matrix_cases(args.matrix)
    compile_only = args.compile_only or args.platform.endswith("sim")
    report: dict[str, object] = {
        "schema": MATRIX_SCHEMA,
        "matrix": args.matrix,
        "platform": args.platform,
        "device": args.device,
        "compile_only": compile_only,
        "declared_case_count": len(cases),
        "cases": [],
        "passed": False,
    }
    entries = report["cases"]
    assert isinstance(entries, list)
    compiled_signatures: dict[tuple[int, ...], str] = {}
    try:
        for case in cases:
            _make_page_layout(case)
            signature = _compile_signature(case)
            if compile_only and signature in compiled_signatures:
                entries.append(
                    {
                        "case": case.summary(),
                        "status": "covered-by-compile-signature",
                        "covered_by": compiled_signatures[signature],
                    }
                )
            else:
                entries.append(_run_case(case, args))
                compiled_signatures[signature] = case.name
            if args.matrix_json is not None:
                _write_json_atomic(args.matrix_json, report)
        report["compiled_signature_count"] = len(compiled_signatures)
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if args.matrix_json is not None:
            _write_json_atomic(args.matrix_json, report)
        raise
    if args.matrix_json is not None:
        _write_json_atomic(args.matrix_json, report)
    print(
        f"[test_paged_attention_pypto] matrix={args.matrix} passed=True "
        f"cases={len(cases)} compiled_signatures={len(compiled_signatures)}"
    )
    return report


def _parse_seq_lens(text: str | None, batch: int, context_len: int) -> tuple[int, ...]:
    if text is None:
        return (context_len,) * batch
    try:
        values = tuple(int(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise ValueError("--seq-lens must be a comma-separated integer list") from error
    if len(values) != batch:
        raise ValueError(f"--seq-lens contains {len(values)} values, expected --batch={batch}")
    return values


def _case_from_args(args: argparse.Namespace) -> DynamicCase:
    return DynamicCase(
        name=f"cli-b{args.batch}",
        seq_lens=_parse_seq_lens(args.seq_lens, args.batch, args.context_len),
        capacity=args.capacity,
        physical_pages=args.physical_pages,
        page_mapping=args.page_mapping,
        pattern=args.pattern,
        seed=args.seed,
        cache_layers=args.cache_layers,
        layer_idx=args.layer_index,
        canary_salt=args.canary_salt,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", choices=("a2a3", "a2a3sim"), default="a2a3")
    parser.add_argument("-d", "--device", type=int, default=2)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--matrix", choices=("pr", "full"))
    parser.add_argument("--matrix-json", type=Path)
    parser.add_argument("--benchmark-json", type=Path)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seq-lens")
    parser.add_argument("--capacity", type=int, default=MAX_CONTEXT_LENGTH)
    parser.add_argument("--physical-pages", type=int)
    parser.add_argument("--page-mapping", choices=PAGE_MAPPINGS, default="identity")
    parser.add_argument("--pattern", choices=DATA_PATTERNS, default="random")
    parser.add_argument("--cache-layers", type=int, default=1)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--canary-salt", type=int, default=0)
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error("--device must be non-negative")
    if not 1 <= args.batch <= BATCH:
        parser.error(f"--batch must be in [1, {BATCH}]")
    if not 1 <= args.context_len <= MAX_CONTEXT_LENGTH:
        parser.error(f"--context-len must be in [1, {MAX_CONTEXT_LENGTH}]")
    if not 1 <= args.capacity <= MAX_CONTEXT_LENGTH:
        parser.error(f"--capacity must be in [1, {MAX_CONTEXT_LENGTH}]")
    if args.physical_pages is not None and args.physical_pages < 1:
        parser.error("--physical-pages must be positive")
    if args.cache_layers < 1 or not 0 <= args.layer_index < args.cache_layers:
        parser.error("--layer-index must select one supplied cache layer")
    if args.canary_salt < 0:
        parser.error("--canary-salt must be non-negative")
    if args.matrix_json is not None and args.matrix is None:
        parser.error("--matrix-json requires --matrix")
    if args.benchmark_json is not None:
        if args.matrix is not None or args.compile_only or args.platform.endswith("sim"):
            parser.error("--benchmark-json requires one real-device case")
        if os.environ.get("PYPTO_BENCH", "").strip() in ("", "0", "false", "False"):
            parser.error("--benchmark-json requires PYPTO_BENCH=1")
    if args.matrix is None:
        try:
            _validate_case(_case_from_args(args))
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.matrix is not None:
        _run_matrix(args)
    else:
        _run_case(_case_from_args(args), args)


def test_matrix_is_focused_and_covers_product_boundaries() -> None:
    pr = pr_cases()
    full = full_cases()
    assert len(pr) == 7
    assert 20 <= len(full) <= 40
    assert {case.batch for case in full} == {1, 2, 8, 16}
    assert {1, 127, 128, 129, 511, 512, 513, 1024, 1025, 1536, 4095, 4096} <= {
        seq_len for case in full for seq_len in case.seq_lens
    }
    assert any(len(set(case.seq_lens)) > 1 for case in pr)
    assert any(case.page_mapping == "shared-prefix" for case in pr)
    assert any(case.layer_cache_base_token_rows > 0 for case in pr)
    assert len({case.name for case in full}) == len(full)


def test_every_declared_case_has_a_valid_page_layout() -> None:
    for case in full_cases():
        table, rows = _make_page_layout(case)
        assert table.numel() == case.batch * case.max_blocks_per_seq
        assert len(rows) == case.batch


def test_seq_len_one_golden_is_selected_value() -> None:
    case = DynamicCase("unit-s1", (1,), page_mapping="reverse", pattern="kv-head-id", seed=0)
    fixture = make_fixture(case)
    values = {
        **{name: tensor.clone() for name, tensor in fixture.items()},
        "out": torch.empty([1, NUM_HEADS, HEAD_DIM], dtype=torch.bfloat16),
    }
    key_before = values["key_cache"].clone()
    value_before = values["value_cache"].clone()
    _, expected_key, expected_value = _phase0_rows(values, case)
    golden_attention(values, case)
    slot = int(fixture["slot_mapping"][0].item())
    page, offset = divmod(slot, BLOCK_SIZE)
    key = _cache_view(values["key_cache"], case)[0]
    value = _cache_view(values["value_cache"], case)[0]
    assert torch.equal(key[page, offset], expected_key[0])
    assert torch.equal(value[page, offset], expected_value[0])
    for q_head in range(NUM_HEADS):
        kv_head = q_head // GROUP
        assert torch.equal(values["out"][0, q_head], value[page, offset, kv_head])

    target_row = case.layer_cache_base_token_rows + slot
    key_before[target_row] = values["key_cache"][target_row]
    value_before[target_row] = values["value_cache"][target_row]
    assert torch.equal(values["key_cache"], key_before)
    assert torch.equal(values["value_cache"], value_before)


def test_pass_dump_resolution_ignores_numeric_ordinals(tmp_path: Path) -> None:
    passes = tmp_path / "passes_dump"
    passes.mkdir()
    expected = passes / "77_after_ExpandMixedKernel.py"
    expected.write_text("pass", encoding="utf-8")
    assert _resolve_pass_dump(tmp_path, "after_ExpandMixedKernel") == expected
    (passes / "91_after_ExpandMixedKernel.py").write_text("duplicate", encoding="utf-8")
    try:
        _resolve_pass_dump(tmp_path, "after_ExpandMixedKernel")
    except RuntimeError as error:
        assert "expected one" in str(error)
    else:
        raise AssertionError("duplicate pass dumps must be rejected")


def test_pypto_pa_is_fused_and_exposes_block_table_dimensions_separately() -> None:
    source = Path(__file__).with_name("paged_attention_pypto.py").read_text()
    assert "block_table_2d = pl.reshape(block_table, [active_batch, max_blocks_per_seq])" in source
    assert "pl.tensor.read(block_table_2d, [batch," in source
    assert "base = batch * max_blocks_per_seq" not in source
    first_cacheinvalid = source.index("pl.system.cacheinvalid()")
    pipe_drain = source.index("pl.system.fence()")
    syncall = source.index("pl.system.syncall(core_type=pl.KernelType.MIX)")
    second_cacheinvalid = source.index("pl.system.cacheinvalid()", first_cacheinvalid + 1)
    assert first_cacheinvalid < pipe_drain < syncall < second_cacheinvalid
    assert "allow_early_resolve=True" in source
    assert "def rope_qkv_pypto(" not in source


def test_pypto_pa_clamps_warmup_slot_before_layer_base() -> None:
    source = Path(__file__).with_name("paged_attention_pypto.py").read_text()
    slot_clamp = source.index("write_slot = pl.max(")
    cache_row = source.index("cache_row = layer_cache_base_token_rows + write_slot")
    assert slot_clamp < cache_row


def test_decode_uses_pypto_pa_and_retains_cce_sources() -> None:
    model_dir = Path(__file__).parent
    decode_source = (model_dir / "decode_fwd.py").read_text()
    assert "from paged_attention_pypto import (" in decode_source
    assert "paged_attention_pypto_swpipe(" in decode_source
    assert "from paged_attention_cce import" not in decode_source
    assert "paged_attention_rope_cce(" not in decode_source
    assert (model_dir / "paged_attention_cce.py").is_file()
    assert (model_dir / "kernels" / "paged_attention_cce").is_dir()
    assert (model_dir / "rope_qkv_regen.py").is_file()
    assert (model_dir / "test_paged_attention_cce.py").is_file()


def test_pypto_pa_preserves_two_stack_prelaunch_skew() -> None:
    source = Path(__file__).with_name("paged_attention_pypto.py").read_text()
    assert PRE_LAUNCH == 2
    assert TRANSFER_SLOTS == PRE_LAUNCH + 1
    assert STACK_TOKENS == 4 * BLOCK_SIZE
    assert "stack_count = (seq_len + STACK_TOKENS - 1) // STACK_TOKENS" in source
    assert "for qk_page_offset in pl.pipeline(STACK_PAGES, stage=2):" in source
    assert 'pl.MemRef("pv_v_l1", slots=STACK_PAGES)' in source
    assert 'pl.MemRef("pv_p_l1", slots=2)' in source
    consume = source.index("if tick >= PRE_LAUNCH:")
    full_stack = source.index("if consume_page + STACK_PAGES <= page_count:", consume)
    v_pages = [source.index(f"v{page}: pl.Tile", full_stack) for page in range(STACK_PAGES)]
    softmax_wait = source.index("SOFTMAX_READY_EVENT", v_pages[-1])
    p_pages = [source.index(f"probability{page}: pl.Tile", softmax_wait) for page in range(STACK_PAGES)]
    assert v_pages == sorted(v_pages)
    assert v_pages[-1] < softmax_wait < p_pages[0]
    assert p_pages == sorted(p_pages)
    assert 'pl.MemRef("pv_v_prefetch' not in source
    assert "tail_pages = page_count - consume_page" in source
    assert source.count("if tail_pages == ") == 3
    assert "qk_ph1 = qk_ph0" not in source
    assert "pv_ph1 = pv_ph0" not in source

    for page_count in range(1, 17):
        stack_count = (page_count + 3) // 4
        produced: list[int] = []
        consumed: list[int] = []
        for tick in range(stack_count + PRE_LAUNCH):
            if tick < stack_count:
                produced.append(tick)
            if tick >= PRE_LAUNCH:
                consumed.append(tick - PRE_LAUNCH)
                if tick < stack_count:
                    assert tick % TRANSFER_SLOTS != (tick - PRE_LAUNCH) % TRANSFER_SLOTS
        assert produced == list(range(stack_count))
        assert consumed == list(range(stack_count))


def test_cli_accepts_scheduler_selected_devices() -> None:
    assert _parse_args(["-p", "a2a3", "-d", "13", "--matrix", "pr"]).device == 13
    assert _parse_args(["-p", "a2a3", "-d", "0", "--matrix", "pr"]).device == 0
    try:
        _parse_args(["-d", "-1"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("negative device ids must be rejected")


if __name__ == "__main__":
    main()
