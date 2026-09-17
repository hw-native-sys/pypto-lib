# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side fixtures for DeepSeek-V4 stats-shaped hash-route replay."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from eplb_fixture import (
    CONTIGUOUS_PLACEMENT,
    EPLB_PLACEMENT,
    EPLB_TOPK,
    EXPERT_PLACEMENT_CHOICES,
    STATIC_STATS_PLACEMENT,
)
from eplb_placement import EplbPlacement, balanced_pack_no_redundancy
from stats_route_fixture import apportion_route_counts, make_stats_shaped_route_table


MANIFEST_SCHEMA = "pypto-lib.dsv4-moe-static-placement.v1"
SOURCE_SCHEMA = "pypto-lib.dsv4-moe-stats-jsonl.v1"
DEFAULT_MANIFEST_PATH = Path(__file__).with_name("stats_placement_decode_manifest.json")
EXPECTED_LAYERS = 44
EXPECTED_RANKS = 8
EXPECTED_LOCAL_EXPERTS = 32
EXPECTED_EXPERTS = EXPECTED_RANKS * EXPECTED_LOCAL_EXPERTS
TOKENS_PER_RANK = 8
ROUTES_PER_LAYER = EXPECTED_RANKS * TOKENS_PER_RANK * EPLB_TOPK

GATE_NAMES = frozenset({"gate_w", "gate_bias"})
ROUTED_NAMES = frozenset(
    {
        "routed_w1",
        "routed_w1_scale",
        "routed_w3",
        "routed_w3_scale",
        "routed_w2",
        "routed_w2_scale",
    }
)
PLACED_NAMES = GATE_NAMES | ROUTED_NAMES | {"tid2eid"}
EXPECTED_ALGORITHM = {
    "capacity_per_rank": EXPECTED_LOCAL_EXPERTS,
    "expert_order": "weight_desc_then_logical_id_asc",
    "local_slot_order": "logical_id_asc",
    "name": "capacity_constrained_lpt",
    "rank_tie_break": "load_asc_then_count_asc_then_rank_asc",
    "version": 1,
}


def _require_int_list(value, *, length: int, location: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        actual = len(value) if isinstance(value, list) else type(value).__name__
        raise ValueError(f"{location} must contain {length} integers, got {actual}")
    for index, item in enumerate(value):
        if type(item) is not int or item < 0:
            raise ValueError(f"{location}[{index}] must be a nonnegative integer, got {item!r}")
    return value


def _validate_manifest(manifest: object) -> dict[str, object]:
    if not isinstance(manifest, dict):
        raise ValueError("placement manifest must contain a JSON object")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"placement manifest schema must be {MANIFEST_SCHEMA!r}")

    topology = manifest.get("topology")
    expected_topology = {
        "experts": EXPECTED_EXPERTS,
        "layers": EXPECTED_LAYERS,
        "local_experts": EXPECTED_LOCAL_EXPERTS,
        "ranks": EXPECTED_RANKS,
    }
    if topology != expected_topology:
        raise ValueError(f"placement topology must be {expected_topology}, got {topology!r}")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("placement manifest source must be an object")
    if source.get("schema") != SOURCE_SCHEMA:
        raise ValueError(f"placement manifest source.schema must be {SOURCE_SCHEMA!r}")
    source_records = source.get("records")
    if type(source_records) is not int or source_records < 1:
        raise ValueError("placement manifest source.records must be a positive integer")
    source_sha = source.get("sha256")
    if (
        not isinstance(source_sha, str)
        or len(source_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_sha)
    ):
        raise ValueError("placement manifest source.sha256 must be a lowercase SHA-256 digest")

    if manifest.get("algorithm") != EXPECTED_ALGORITHM:
        raise ValueError(f"placement manifest algorithm must be {EXPECTED_ALGORITHM}")

    filters = manifest.get("filters")
    if not isinstance(filters, dict) or set(filters) != {"phases", "routed_tokens"}:
        raise ValueError("placement manifest filters must contain phases and routed_tokens")
    phases = filters["phases"]
    if phases is not None and (
        not isinstance(phases, list)
        or not phases
        or any(not isinstance(phase, str) or not phase for phase in phases)
        or phases != sorted(set(phases))
    ):
        raise ValueError("placement manifest phase filters must be sorted unique strings or null")
    routed_tokens = filters["routed_tokens"]
    if routed_tokens is not None:
        _require_int_list(
            routed_tokens,
            length=len(routed_tokens),
            location="filters.routed_tokens",
        )
        if not routed_tokens or routed_tokens != sorted(set(routed_tokens)):
            raise ValueError(
                "placement manifest routed_tokens filters must be sorted unique integers or null"
            )

    expert_loads = manifest.get("expert_loads")
    if not isinstance(expert_loads, list) or len(expert_loads) != EXPECTED_LAYERS:
        raise ValueError(f"placement manifest must contain {EXPECTED_LAYERS} expert-load rows")

    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != EXPECTED_LAYERS:
        raise ValueError(f"placement manifest must contain {EXPECTED_LAYERS} layer mappings")

    for layer_id in range(EXPECTED_LAYERS):
        loads = _require_int_list(
            expert_loads[layer_id],
            length=EXPECTED_EXPERTS,
            location=f"expert_loads[{layer_id}]",
        )
        layer = layers[layer_id]
        if not isinstance(layer, dict) or layer.get("layer_id") != layer_id:
            raise ValueError(f"layers[{layer_id}] must describe layer_id {layer_id}")
        expected_kind = "main" if layer_id < EXPECTED_LAYERS - 1 else "mtp"
        if layer.get("kind") != expected_kind:
            raise ValueError(f"layers[{layer_id}].kind must be {expected_kind!r}")
        observations = layer.get("observations")
        if type(observations) is not int or observations < 1:
            raise ValueError(f"layers[{layer_id}].observations must be a positive integer")

        logical_to_physical = _require_int_list(
            layer.get("logical_to_physical"),
            length=EXPECTED_EXPERTS,
            location=f"layers[{layer_id}].logical_to_physical",
        )
        if sorted(logical_to_physical) != list(range(EXPECTED_EXPERTS)):
            raise ValueError(f"layers[{layer_id}].logical_to_physical must be a permutation")

        rank_to_logical = layer.get("rank_to_logical")
        if not isinstance(rank_to_logical, list) or len(rank_to_logical) != EXPECTED_RANKS:
            raise ValueError(f"layers[{layer_id}].rank_to_logical must contain {EXPECTED_RANKS} ranks")
        physical_to_logical = []
        for rank, logical_experts in enumerate(rank_to_logical):
            physical_to_logical.extend(
                _require_int_list(
                    logical_experts,
                    length=EXPECTED_LOCAL_EXPERTS,
                    location=f"layers[{layer_id}].rank_to_logical[{rank}]",
                )
            )
        if sorted(physical_to_logical) != list(range(EXPECTED_EXPERTS)):
            raise ValueError(f"layers[{layer_id}].rank_to_logical must contain every expert once")
        for physical_id, logical_id in enumerate(physical_to_logical):
            if logical_to_physical[logical_id] != physical_id:
                raise ValueError(f"layers[{layer_id}] forward and inverse mappings disagree")

        estimated_rank_loads = _require_int_list(
            layer.get("estimated_rank_loads"),
            length=EXPECTED_RANKS,
            location=f"layers[{layer_id}].estimated_rank_loads",
        )
        expected_loads = [
            sum(loads[logical_id] for logical_id in rank_to_logical[rank]) for rank in range(EXPECTED_RANKS)
        ]
        if estimated_rank_loads != expected_loads:
            raise ValueError(f"layers[{layer_id}].estimated_rank_loads do not match expert_loads")
        if layer.get("routed_tokens") != sum(loads):
            raise ValueError(f"layers[{layer_id}].routed_tokens does not match expert_loads")
    return manifest


@lru_cache(maxsize=None)
def _load_manifest(path: str) -> dict[str, object]:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load placement manifest {path}: {error}") from error
    return _validate_manifest(manifest)


def load_stats_placement_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, object]:
    """Load and validate one static placement manifest."""
    return _load_manifest(str(Path(path).resolve()))


@lru_cache(maxsize=None)
def _pack_eplb_replay_counts(route_counts: tuple[int, ...]) -> EplbPlacement:
    return balanced_pack_no_redundancy(
        route_counts,
        ranks=EXPECTED_RANKS,
        capacity_per_rank=EXPECTED_LOCAL_EXPERTS,
    )


def eplb_replay_placement(manifest: dict[str, object], layer_id: int) -> EplbPlacement:
    """Build an oracle EPLB placement from this fixture's exact route counts."""
    if type(layer_id) is not int or not 0 <= layer_id < EXPECTED_LAYERS:
        raise ValueError(f"layer_id must be in [0, {EXPECTED_LAYERS}), got {layer_id}")
    route_counts = apportion_route_counts(
        manifest["expert_loads"][layer_id],
        total_routes=ROUTES_PER_LAYER,
    )
    return _pack_eplb_replay_counts(route_counts)


def physical_to_logical_for_placement(
    manifest: dict[str, object],
    layer_id: int,
    placement: str,
) -> tuple[int, ...]:
    """Return the rank-major logical expert order for one placement variant."""
    if placement not in EXPERT_PLACEMENT_CHOICES:
        raise ValueError(f"placement must be one of {EXPERT_PLACEMENT_CHOICES}, got {placement!r}")
    if type(layer_id) is not int or not 0 <= layer_id < EXPECTED_LAYERS:
        raise ValueError(f"layer_id must be in [0, {EXPECTED_LAYERS}), got {layer_id}")
    if placement == CONTIGUOUS_PLACEMENT:
        return tuple(range(EXPECTED_EXPERTS))
    if placement == STATIC_STATS_PLACEMENT:
        rank_to_logical = manifest["layers"][layer_id]["rank_to_logical"]
        return tuple(logical_id for logical_experts in rank_to_logical for logical_id in logical_experts)
    return eplb_replay_placement(manifest, layer_id).physical_to_logical


def _place_gate_tensor(tensor, *, physical_to_logical: list[int]):
    import torch

    if list(tensor.shape[:2]) != [EXPECTED_RANKS, EXPECTED_EXPERTS]:
        raise ValueError(
            f"replicated gate tensor must start with [{EXPECTED_RANKS}, {EXPECTED_EXPERTS}], "
            f"got {list(tensor.shape)}"
        )
    index = torch.tensor(physical_to_logical, dtype=torch.int64)
    return tensor.index_select(1, index)


def _write_placed_routed_tensor(
    packed,
    source,
    *,
    physical_to_logical: list[int],
    layer_offset: int,
) -> None:
    """Gather one destination rank at a time to bound host conversion memory."""
    import torch

    if list(source.shape[:2]) != [EXPECTED_RANKS, EXPECTED_LOCAL_EXPERTS]:
        raise ValueError(
            "routed tensor must start with "
            f"[{EXPECTED_RANKS}, {EXPECTED_LOCAL_EXPERTS}], got {list(source.shape)}"
        )
    logical = source.reshape(EXPECTED_EXPERTS, *source.shape[2:])
    for rank in range(EXPECTED_RANKS):
        rank_start = rank * EXPECTED_LOCAL_EXPERTS
        rank_end = rank_start + EXPECTED_LOCAL_EXPERTS
        index = torch.tensor(physical_to_logical[rank_start:rank_end], dtype=torch.int64)
        packed[
            rank,
            layer_offset : layer_offset + EXPECTED_LOCAL_EXPERTS,
            ...,
        ] = logical.index_select(0, index)


def _make_stats_tid2eid_spec(
    base_spec,
    *,
    layer_ids: tuple[int, ...],
    manifest,
    placement: str,
):
    import torch
    from golden import TensorSpec

    if len(base_spec.shape) != 3 or base_spec.shape[0] != EXPECTED_RANKS:
        raise ValueError("tid2eid must be a rank-stacked three-dimensional TensorSpec")
    num_ranks, vocab, topk = base_spec.shape
    global_tokens = num_ranks * TOKENS_PER_RANK
    if placement == EPLB_PLACEMENT and topk != EPLB_TOPK:
        raise ValueError(f"EPLB replay requires topk={EPLB_TOPK}, got {topk}")

    def init_value():
        stacked = torch.empty(
            [num_ranks, len(layer_ids) * vocab, topk],
            dtype=base_spec.dtype,
        )
        for stack_index, layer_id in enumerate(layer_ids):
            logical_routes = make_stats_shaped_route_table(
                manifest["expert_loads"][layer_id],
                num_tokens=global_tokens,
                topk=topk,
            )
            if placement != CONTIGUOUS_PLACEMENT:
                physical_to_logical = physical_to_logical_for_placement(
                    manifest,
                    layer_id,
                    placement,
                )
                logical_to_physical = torch.empty(EXPECTED_EXPERTS, dtype=torch.int64)
                logical_to_physical[torch.tensor(physical_to_logical, dtype=torch.int64)] = torch.arange(
                    EXPECTED_EXPERTS,
                    dtype=torch.int64,
                )
                active_routes = logical_to_physical[logical_routes.to(torch.int64)]
            else:
                active_routes = logical_routes
            active_routes = active_routes.to(base_spec.dtype)
            repeats = (vocab + global_tokens - 1) // global_tokens
            table = active_routes.repeat(repeats, 1)[:vocab]
            start = stack_index * vocab
            stacked[:, start : start + vocab, :] = table.unsqueeze(0)
        return stacked

    return TensorSpec(
        base_spec.name,
        [num_ranks, len(layer_ids) * vocab, topk],
        base_spec.dtype,
        init_value=init_value,
        resident=base_spec.resident,
    )


def make_stats_placement_spec(
    name,
    base_spec,
    *,
    layer_ids,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    placement: str = STATIC_STATS_PLACEMENT,
):
    """Build one stats-workload TensorSpec for the selected physical placement."""
    import torch
    from golden import TensorSpec

    if placement not in EXPERT_PLACEMENT_CHOICES:
        raise ValueError(f"placement must be one of {EXPERT_PLACEMENT_CHOICES}, got {placement!r}")
    if name not in PLACED_NAMES:
        return None
    if placement == CONTIGUOUS_PLACEMENT and name != "tid2eid":
        return None
    layer_ids = tuple(layer_ids)
    if not layer_ids:
        raise ValueError("layer_ids must not be empty")
    if any(type(layer_id) is not int or not 0 <= layer_id < EXPECTED_LAYERS for layer_id in layer_ids):
        raise ValueError(f"layer_ids must be integers in [0, {EXPECTED_LAYERS})")
    manifest = load_stats_placement_manifest(manifest_path)

    if name == "tid2eid":
        return _make_stats_tid2eid_spec(
            base_spec,
            layer_ids=layer_ids,
            manifest=manifest,
            placement=placement,
        )

    expected_axis = EXPECTED_EXPERTS if name in GATE_NAMES else EXPECTED_LOCAL_EXPERTS
    if len(base_spec.shape) < 2 or list(base_spec.shape[:2]) != [EXPECTED_RANKS, expected_axis]:
        raise ValueError(f"{name} must start with [{EXPECTED_RANKS}, {expected_axis}], got {base_spec.shape}")
    packed_shape = [EXPECTED_RANKS, len(layer_ids) * expected_axis, *base_spec.shape[2:]]

    def init_value():
        packed = torch.empty(packed_shape, dtype=base_spec.dtype)
        for stack_index, layer_id in enumerate(layer_ids):
            source = base_spec.create_tensor()
            physical_to_logical = physical_to_logical_for_placement(
                manifest,
                layer_id,
                placement,
            )
            start = stack_index * expected_axis
            if name in GATE_NAMES:
                placed = _place_gate_tensor(source, physical_to_logical=physical_to_logical)
                packed[:, start : start + expected_axis, ...] = placed
            else:
                _write_placed_routed_tensor(
                    packed,
                    source,
                    physical_to_logical=physical_to_logical,
                    layer_offset=start,
                )
        return packed

    return TensorSpec(
        name,
        packed_shape,
        base_spec.dtype,
        init_value=init_value,
        resident=base_spec.resident,
    )


def stats_layer_spec_factory(
    name,
    base_spec,
    layer_count,
    *,
    placement: str = STATIC_STATS_PLACEMENT,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
):
    """Build a main-forward stats-workload override for one layer-stacked spec."""
    return make_stats_placement_spec(
        name,
        base_spec,
        layer_ids=range(layer_count),
        placement=placement,
        manifest_path=manifest_path,
    )


def adapt_single_layer_stats_placement_specs(
    specs,
    *,
    layer_id: int,
    placement: str = STATIC_STATS_PLACEMENT,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
):
    """Replace one layer's MoE specs with stats-workload lazy initializers."""
    adapted = []
    for spec in specs:
        replacement = make_stats_placement_spec(
            spec.name,
            spec,
            layer_ids=[layer_id],
            placement=placement,
            manifest_path=manifest_path,
        )
        adapted.append(spec if replacement is None else replacement)
    return adapted


def adapt_mtp_stats_placement_specs(
    specs,
    *,
    layer_id: int = EXPECTED_LAYERS - 1,
    placement: str = STATIC_STATS_PLACEMENT,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
):
    """Replace MTP MoE specs with stats-workload lazy initializers."""
    return adapt_single_layer_stats_placement_specs(
        specs,
        layer_id=layer_id,
        placement=placement,
        manifest_path=manifest_path,
    )
