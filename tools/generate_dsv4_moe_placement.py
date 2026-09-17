# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate a deterministic static DeepSeek-V4 expert-placement manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


MANIFEST_SCHEMA = "pypto-lib.dsv4-moe-static-placement.v1"
SOURCE_SCHEMA = "pypto-lib.dsv4-moe-stats-jsonl.v1"
EXPECTED_LAYERS = 44
EXPECTED_RANKS = 8
EXPECTED_LOCAL_EXPERTS = 32
EXPECTED_EXPERTS = EXPECTED_RANKS * EXPECTED_LOCAL_EXPERTS


class PlacementError(ValueError):
    """Raised when source statistics cannot produce a valid placement."""


@dataclass(frozen=True)
class Placement:
    """One layer's deterministic mapping and estimated destination loads."""

    logical_to_physical: tuple[int, ...]
    rank_to_logical: tuple[tuple[int, ...], ...]
    estimated_rank_loads: tuple[int, ...]


def _require_int(value: object, location: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PlacementError(f"{location} must be an integer >= {minimum}, got {value!r}")
    return value


def _required(mapping: dict[str, object], key: str, location: str) -> object:
    if key not in mapping:
        raise PlacementError(f"{location} is missing required field {key!r}")
    return mapping[key]


def _validate_record(
    record: object,
    line_number: int,
) -> tuple[int, str, list[dict[str, object]]]:
    location = f"line {line_number}"
    if not isinstance(record, dict):
        raise PlacementError(f"{location} must contain a JSON object")

    _require_int(_required(record, "timestamp_ns", location), f"{location}.timestamp_ns")
    dispatch_id = _require_int(
        _required(record, "dispatch_id", location),
        f"{location}.dispatch_id",
    )
    phase = _required(record, "phase", location)
    if not isinstance(phase, str) or not phase:
        raise PlacementError(f"{location}.phase must be a non-empty string")

    ranks = _require_int(_required(record, "ranks", location), f"{location}.ranks", minimum=1)
    if ranks != EXPECTED_RANKS:
        raise PlacementError(f"{location}.ranks must be {EXPECTED_RANKS}, got {ranks}")
    local_experts = _require_int(
        _required(record, "local_experts", location),
        f"{location}.local_experts",
        minimum=1,
    )
    if local_experts != EXPECTED_LOCAL_EXPERTS:
        raise PlacementError(
            f"{location}.local_experts must be {EXPECTED_LOCAL_EXPERTS}, got {local_experts}"
        )
    if ranks * local_experts != EXPECTED_EXPERTS:
        raise PlacementError(
            f"{location} topology must contain {EXPECTED_EXPERTS} experts, got {ranks} x {local_experts}"
        )

    raw_layers = _required(record, "layers", location)
    if not isinstance(raw_layers, list) or len(raw_layers) != EXPECTED_LAYERS:
        actual = len(raw_layers) if isinstance(raw_layers, list) else type(raw_layers).__name__
        raise PlacementError(f"{location}.layers must contain {EXPECTED_LAYERS} layers, got {actual}")

    layers_by_id: list[dict[str, object] | None] = [None] * EXPECTED_LAYERS
    for position, layer in enumerate(raw_layers):
        layer_location = f"{location}.layers[{position}]"
        if not isinstance(layer, dict):
            raise PlacementError(f"{layer_location} must be a JSON object")
        layer_id = _require_int(_required(layer, "layer_id", layer_location), f"{layer_location}.layer_id")
        if layer_id >= EXPECTED_LAYERS:
            raise PlacementError(
                f"{layer_location}.layer_id must be in [0, {EXPECTED_LAYERS}), got {layer_id}"
            )
        if layers_by_id[layer_id] is not None:
            raise PlacementError(f"{location}.layers contains duplicate layer_id {layer_id}")

        expected_kind = "main" if layer_id < EXPECTED_LAYERS - 1 else "mtp"
        kind = _required(layer, "kind", layer_location)
        if kind != expected_kind:
            raise PlacementError(
                f"{layer_location}.kind must be {expected_kind!r} for layer {layer_id}, got {kind!r}"
            )

        counts = _required(layer, "expert_token_counts", layer_location)
        if not isinstance(counts, list) or len(counts) != EXPECTED_EXPERTS:
            actual = len(counts) if isinstance(counts, list) else type(counts).__name__
            raise PlacementError(
                f"{layer_location}.expert_token_counts must contain {EXPECTED_EXPERTS} counts, got {actual}"
            )
        validated_counts = [
            _require_int(value, f"{layer_location}.expert_token_counts[{expert_id}]")
            for expert_id, value in enumerate(counts)
        ]
        routed_tokens = _require_int(
            _required(layer, "routed_tokens", layer_location),
            f"{layer_location}.routed_tokens",
        )
        count_sum = sum(validated_counts)
        if routed_tokens != count_sum:
            raise PlacementError(
                f"{layer_location}.routed_tokens is {routed_tokens}, but expert counts sum to {count_sum}"
            )
        active_experts = _require_int(
            _required(layer, "active_experts", layer_location),
            f"{layer_location}.active_experts",
        )
        nonzero_experts = sum(value > 0 for value in validated_counts)
        if active_experts != nonzero_experts:
            raise PlacementError(
                f"{layer_location}.active_experts is {active_experts}, "
                f"but {nonzero_experts} expert counts are nonzero"
            )

        layers_by_id[layer_id] = {
            "layer_id": layer_id,
            "kind": kind,
            "routed_tokens": routed_tokens,
            "expert_token_counts": validated_counts,
        }

    if any(layer is None for layer in layers_by_id):
        missing = [layer_id for layer_id, layer in enumerate(layers_by_id) if layer is None]
        raise PlacementError(f"{location}.layers is missing layer IDs {missing}")
    return dispatch_id, phase, [layer for layer in layers_by_id if layer is not None]


def capacity_constrained_lpt(
    weights: Sequence[int],
    *,
    ranks: int = EXPECTED_RANKS,
    capacity: int = EXPECTED_LOCAL_EXPERTS,
) -> Placement:
    """Assign descending weights to the least-loaded non-full rank.

    Experts with equal weights are visited by ascending logical ID. Rank ties
    are resolved by assigned count and then rank ID. Logical experts within a
    rank are sorted before rank-major physical slot IDs are assigned.
    """
    if ranks <= 0 or capacity <= 0:
        raise PlacementError("ranks and capacity must both be positive")
    if len(weights) != ranks * capacity:
        raise PlacementError(
            f"LPT requires ranks x capacity weights ({ranks} x {capacity}), got {len(weights)}"
        )
    validated_weights = [
        _require_int(weight, f"weights[{expert_id}]") for expert_id, weight in enumerate(weights)
    ]

    rank_loads = [0] * ranks
    assigned: list[list[int]] = [[] for _ in range(ranks)]
    expert_order = sorted(
        range(len(validated_weights)), key=lambda expert_id: (-validated_weights[expert_id], expert_id)
    )
    for expert_id in expert_order:
        candidates = (rank for rank in range(ranks) if len(assigned[rank]) < capacity)
        rank = min(
            candidates, key=lambda candidate: (rank_loads[candidate], len(assigned[candidate]), candidate)
        )
        assigned[rank].append(expert_id)
        rank_loads[rank] += validated_weights[expert_id]

    rank_to_logical = tuple(tuple(sorted(experts)) for experts in assigned)
    logical_to_physical = [-1] * len(validated_weights)
    for rank, logical_experts in enumerate(rank_to_logical):
        if len(logical_experts) != capacity:
            raise PlacementError(
                f"internal placement error: rank {rank} has {len(logical_experts)} experts, expected {capacity}"
            )
        for local_slot, logical_expert in enumerate(logical_experts):
            logical_to_physical[logical_expert] = rank * capacity + local_slot

    if sorted(logical_to_physical) != list(range(len(validated_weights))):
        raise PlacementError("internal placement error: logical_to_physical is not a permutation")
    return Placement(
        logical_to_physical=tuple(logical_to_physical),
        rank_to_logical=rank_to_logical,
        estimated_rank_loads=tuple(rank_loads),
    )


def _normalize_phases(phases: Sequence[str] | None) -> tuple[str, ...] | None:
    if phases is None:
        return None
    if not phases:
        raise PlacementError("phase filter must not be empty")
    if any(not isinstance(phase, str) or not phase for phase in phases):
        raise PlacementError("phase filters must be non-empty strings")
    return tuple(sorted(set(phases)))


def _normalize_routed_tokens(values: Sequence[int] | None) -> tuple[int, ...] | None:
    if values is None:
        return None
    if not values:
        raise PlacementError("routed_tokens filter must not be empty")
    normalized = {_require_int(value, "routed_tokens filter") for value in values}
    return tuple(sorted(normalized))


def generate_manifest(
    source: Path,
    *,
    phases: Sequence[str] | None = None,
    routed_tokens: Sequence[int] | None = None,
) -> dict[str, object]:
    """Parse, validate, filter, and aggregate one MoE statistics JSONL file."""
    phase_filter = _normalize_phases(phases)
    routed_filter = _normalize_routed_tokens(routed_tokens)
    phase_set = None if phase_filter is None else set(phase_filter)
    routed_set = None if routed_filter is None else set(routed_filter)

    weights = [[0] * EXPECTED_EXPERTS for _ in range(EXPECTED_LAYERS)]
    observations = [0] * EXPECTED_LAYERS
    routed_totals = [0] * EXPECTED_LAYERS
    digest = hashlib.sha256()
    record_count = 0
    previous_dispatch_id = None

    try:
        with source.open("rb") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                digest.update(raw_line)
                if not raw_line.strip():
                    raise PlacementError(f"line {line_number} is blank")
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PlacementError(f"line {line_number} is not valid JSON: {error}") from error
                dispatch_id, phase, layers = _validate_record(record, line_number)
                if previous_dispatch_id is not None and dispatch_id <= previous_dispatch_id:
                    raise PlacementError(
                        f"line {line_number}.dispatch_id must be greater than "
                        f"the previous value {previous_dispatch_id}, got {dispatch_id}"
                    )
                previous_dispatch_id = dispatch_id
                record_count += 1
                if phase_set is not None and phase not in phase_set:
                    continue
                for layer in layers:
                    layer_id = layer["layer_id"]
                    layer_routed_tokens = layer["routed_tokens"]
                    if routed_set is not None and layer_routed_tokens not in routed_set:
                        continue
                    observations[layer_id] += 1
                    routed_totals[layer_id] += layer_routed_tokens
                    layer_weights = weights[layer_id]
                    for expert_id, count in enumerate(layer["expert_token_counts"]):
                        layer_weights[expert_id] += count
    except OSError as error:
        raise PlacementError(f"cannot read {source}: {error}") from error

    if record_count == 0:
        raise PlacementError("source contains no JSONL records")
    empty_layers = [
        layer_id
        for layer_id in range(EXPECTED_LAYERS)
        if observations[layer_id] == 0 or routed_totals[layer_id] == 0
    ]
    if empty_layers:
        raise PlacementError(
            f"filters selected no positive routed-token observations for layer IDs {empty_layers}"
        )

    manifest_layers = []
    for layer_id, layer_weights in enumerate(weights):
        placement = capacity_constrained_lpt(layer_weights)
        manifest_layers.append(
            {
                "estimated_rank_loads": list(placement.estimated_rank_loads),
                "kind": "main" if layer_id < EXPECTED_LAYERS - 1 else "mtp",
                "layer_id": layer_id,
                "logical_to_physical": list(placement.logical_to_physical),
                "observations": observations[layer_id],
                "rank_to_logical": [list(experts) for experts in placement.rank_to_logical],
                "routed_tokens": routed_totals[layer_id],
            }
        )

    return {
        "algorithm": {
            "capacity_per_rank": EXPECTED_LOCAL_EXPERTS,
            "expert_order": "weight_desc_then_logical_id_asc",
            "local_slot_order": "logical_id_asc",
            "name": "capacity_constrained_lpt",
            "rank_tie_break": "load_asc_then_count_asc_then_rank_asc",
            "version": 1,
        },
        "filters": {
            "phases": None if phase_filter is None else list(phase_filter),
            "routed_tokens": None if routed_filter is None else list(routed_filter),
        },
        "expert_loads": [list(layer_weights) for layer_weights in weights],
        "layers": manifest_layers,
        "schema": MANIFEST_SCHEMA,
        "source": {
            "records": record_count,
            "schema": SOURCE_SCHEMA,
            "sha256": digest.hexdigest(),
        },
        "topology": {
            "experts": EXPECTED_EXPERTS,
            "layers": EXPECTED_LAYERS,
            "local_experts": EXPECTED_LOCAL_EXPERTS,
            "ranks": EXPECTED_RANKS,
        },
    }


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="MoE statistics JSONL input")
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        help="manifest path, or '-' for stdout (default: '-')",
    )
    parser.add_argument(
        "--phase",
        action="append",
        dest="phases",
        help="include this record phase; repeat to include multiple phases",
    )
    parser.add_argument(
        "--routed-tokens",
        action="append",
        type=_nonnegative_int,
        help="include layer observations with this routed_tokens value; repeat as needed",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = None if args.output == "-" else Path(args.output)
    if output is not None and output.resolve() == args.source.resolve():
        print("ERROR: output path must differ from the source path", file=sys.stderr)
        return 2

    try:
        manifest = generate_manifest(
            args.source,
            phases=args.phases,
            routed_tokens=args.routed_tokens,
        )
        text = json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        if output is None:
            sys.stdout.write(text)
        else:
            output.write_text(text, encoding="utf-8")
    except (OSError, PlacementError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
