# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared topology and deterministic routing fixtures for MoE placement benchmarks."""

from __future__ import annotations

import sys


EPLB_EP_SIZE = 8
EPLB_TP_SIZE = 4
EPLB_EXPERTS_PER_RANK = 16
EPLB_NUM_EXPERTS = EPLB_EP_SIZE * EPLB_EXPERTS_PER_RANK
STATS_PLACEMENT_EXPERTS_PER_RANK = 32
STATS_PLACEMENT_NUM_EXPERTS = EPLB_EP_SIZE * STATS_PLACEMENT_EXPERTS_PER_RANK
STATS_PLACEMENT_FLAG = "--stats-placement"
EXPERT_PLACEMENT_FLAG = "--expert-placement"
PLACEMENT_MANIFEST_FLAG = "--placement-manifest"
CONTIGUOUS_PLACEMENT = "contiguous"
STATIC_STATS_PLACEMENT = "stats"
EPLB_PLACEMENT = "eplb"
EXPERT_PLACEMENT_CHOICES = (CONTIGUOUS_PLACEMENT, STATIC_STATS_PLACEMENT, EPLB_PLACEMENT)
EPLB_TOKENS = 8
EPLB_TOPK = 6
EPLB_START_POS = 8192
EPLB_COMPRESSED_WRITE_START_POS = EPLB_START_POS - 1
EPLB_START_POS_CHOICES = (EPLB_START_POS, EPLB_COMPRESSED_WRITE_START_POS)
COMPRESSED_CACHE_ACTIVE_WRITE_FLAG = "--compressed-cache-active-write"
EPLB_ROUTES_PER_EXPERT = EPLB_EP_SIZE * EPLB_TOKENS * EPLB_TOPK // EPLB_NUM_EXPERTS

def _read_int_argv(argv: list[str], name: str) -> int | None:
    for index, token in enumerate(argv):
        if token == name:
            if index + 1 >= len(argv):
                raise ValueError(f"{name} requires an integer value")
            return int(argv[index + 1])
        if token.startswith(f"{name}="):
            return int(token.split("=", 1)[1])
    return None


def stats_placement_enabled(argv: list[str] | None = None) -> bool:
    """Return whether the static stats-placement profile is selected."""
    target = sys.argv if argv is None else argv
    return STATS_PLACEMENT_FLAG in target


def benchmark_experts_per_rank(argv: list[str] | None = None) -> int:
    """Return the local expert count for the selected benchmark profile."""
    if stats_placement_enabled(argv):
        return STATS_PLACEMENT_EXPERTS_PER_RANK
    return EPLB_EXPERTS_PER_RANK


def benchmark_num_experts(argv: list[str] | None = None) -> int:
    """Return the global expert count for the selected benchmark profile."""
    return EPLB_EP_SIZE * benchmark_experts_per_rank(argv)


def configure_eplb_argv(argv: list[str] | None = None) -> list[str]:
    """Install and validate the selected benchmark topology before model imports."""
    target = sys.argv if argv is None else argv
    fixed_argv = {
        "--ep": EPLB_EP_SIZE,
        "--tp": EPLB_TP_SIZE,
        "--experts-per-rank": benchmark_experts_per_rank(target),
    }
    for name, expected in fixed_argv.items():
        actual = _read_int_argv(target, name)
        if actual is None:
            target.extend([name, str(expected)])
        elif actual != expected:
            raise ValueError(f"EPLB benchmark requires {name}={expected}, got {actual}")
    return target


def validate_eplb_topology(
    *,
    ep_size: int,
    tp_size: int,
    experts_per_rank: int,
    num_experts: int,
    tokens: int,
    topk: int,
    start_pos: int = EPLB_START_POS,
) -> None:
    """Validate the selected benchmark workload dimensions and route balance."""
    actual = {
        "ep_size": ep_size,
        "tp_size": tp_size,
        "experts_per_rank": experts_per_rank,
        "num_experts": num_experts,
        "tokens": tokens,
        "topk": topk,
        "start_pos": start_pos,
    }
    expected_experts_per_rank = benchmark_experts_per_rank()
    expected = {
        "ep_size": EPLB_EP_SIZE,
        "tp_size": EPLB_TP_SIZE,
        "experts_per_rank": expected_experts_per_rank,
        "num_experts": EPLB_EP_SIZE * expected_experts_per_rank,
        "tokens": EPLB_TOKENS,
        "topk": EPLB_TOPK,
    }
    mismatches = []
    for name, value in expected.items():
        if actual[name] != value:
            mismatches.append(f"{name}={actual[name]} (expected {value})")
    allowed_start_positions = (
        EPLB_START_POS_CHOICES
        if stats_placement_enabled()
        else (EPLB_START_POS,)
    )
    if start_pos not in allowed_start_positions:
        mismatches.append(
            f"start_pos={start_pos} (expected one of {allowed_start_positions})"
        )
    if mismatches:
        raise ValueError("invalid EPLB topology: " + ", ".join(mismatches))

    if not stats_placement_enabled():
        active_routes = ep_size * tokens * topk
        if active_routes % num_experts != 0:
            raise ValueError(f"EPLB active route count {active_routes} is not divisible by {num_experts} experts")
        if active_routes // num_experts != EPLB_ROUTES_PER_EXPERT:
            raise ValueError(
                f"EPLB requires {EPLB_ROUTES_PER_EXPERT} routes per expert, got {active_routes // num_experts}"
            )


def make_round_robin_tid2eid(
    *,
    vocab: int,
    topk: int = EPLB_TOPK,
    num_experts: int | None = None,
):
    """Build one hash-layer-zero token-to-expert table."""
    import torch

    if num_experts is None:
        num_experts = benchmark_num_experts()
    token_ids = torch.arange(vocab, dtype=torch.int64).reshape(vocab, 1)
    topk_slots = torch.arange(topk, dtype=torch.int64).reshape(1, topk)
    return ((token_ids * topk + topk_slots) % num_experts).to(torch.int32)


def make_rank_offset_input_ids(
    *,
    num_ranks: int = EPLB_EP_SIZE,
    token_rows: int = EPLB_TOKENS,
    active_tokens: int = EPLB_TOKENS,
):
    """Build rank-offset token IDs for a contiguous global route sequence."""
    import torch

    if not 1 <= active_tokens <= token_rows:
        raise ValueError(f"active_tokens must be in [1, {token_rows}], got {active_tokens}")
    rank_starts = torch.arange(num_ranks, dtype=torch.int64).reshape(num_ranks, 1) * active_tokens
    token_offsets = torch.arange(token_rows, dtype=torch.int64).reshape(1, token_rows)
    return rank_starts + token_offsets


def make_eplb_tid2eid_spec(base_spec, *, layer_count: int = 1):
    """Replace one tid2eid TensorSpec with repeated hash-layer-zero tables."""
    from golden import TensorSpec

    if base_spec.name != "tid2eid" or len(base_spec.shape) != 3:
        raise ValueError("base_spec must be a rank-stacked tid2eid TensorSpec")
    num_ranks, vocab, topk = base_spec.shape
    validate_eplb_topology(
        ep_size=num_ranks,
        tp_size=EPLB_TP_SIZE,
        experts_per_rank=benchmark_experts_per_rank(),
        num_experts=benchmark_num_experts(),
        tokens=EPLB_TOKENS,
        topk=topk,
    )
    if layer_count < 1:
        raise ValueError(f"layer_count must be positive, got {layer_count}")

    def init_value():
        table = make_round_robin_tid2eid(vocab=vocab, topk=topk)
        stacked = table.repeat(layer_count, 1)
        return stacked.unsqueeze(0).expand(num_ranks, -1, -1).contiguous()

    return TensorSpec(
        "tid2eid",
        [num_ranks, layer_count * vocab, topk],
        base_spec.dtype,
        init_value=init_value,
        resident=base_spec.resident,
    )


def make_eplb_input_ids_spec(base_spec, *, active_tokens: int = EPLB_TOKENS):
    """Replace an input_ids TensorSpec with rank-offset token IDs."""
    from golden import TensorSpec

    if base_spec.name != "input_ids" or len(base_spec.shape) != 2:
        raise ValueError("base_spec must be a rank-stacked input_ids TensorSpec")
    num_ranks, token_rows = base_spec.shape

    def init_value():
        return make_rank_offset_input_ids(
            num_ranks=num_ranks,
            token_rows=token_rows,
            active_tokens=active_tokens,
        )

    return TensorSpec(
        "input_ids",
        list(base_spec.shape),
        base_spec.dtype,
        init_value=init_value,
        resident=base_spec.resident,
    )


def replace_eplb_routing_specs(
    specs,
    *,
    layer_count: int = 1,
    active_tokens: int = EPLB_TOKENS,
):
    """Apply the fixed EPLB routing inputs and hash-layer-zero scalar identity."""
    import torch
    from golden import ScalarSpec

    replaced = []
    seen = set()
    for spec in specs:
        if spec.name == "tid2eid":
            replaced.append(make_eplb_tid2eid_spec(spec, layer_count=layer_count))
            seen.add(spec.name)
        elif spec.name == "input_ids":
            replaced.append(make_eplb_input_ids_spec(spec, active_tokens=active_tokens))
            seen.add(spec.name)
        elif spec.name == "layer_id":
            replaced.append(ScalarSpec("layer_id", torch.int32, 0))
        else:
            replaced.append(spec)
    missing = {"tid2eid", "input_ids"} - seen
    if missing:
        raise ValueError(f"missing EPLB routing specs: {sorted(missing)}")
    return replaced
