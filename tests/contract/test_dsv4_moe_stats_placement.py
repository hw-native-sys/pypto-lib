# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Verify the offline DeepSeek-V4 MoE placement generator."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.generate_dsv4_moe_placement import (
    MANIFEST_SCHEMA,
    PlacementError,
    capacity_constrained_lpt,
    generate_manifest,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_GENERATOR = _REPO_ROOT / "tools" / "generate_dsv4_moe_placement.py"


def _record(dispatch_id: int, phase: str, *, count: int = 1) -> dict[str, object]:
    counts = [count] * 256
    return {
        "timestamp_ns": 1_000_000 + dispatch_id,
        "dispatch_id": dispatch_id,
        "phase": phase,
        "ranks": 8,
        "local_experts": 32,
        "layers": [
            {
                "layer_id": layer_id,
                "kind": "main" if layer_id < 43 else "mtp",
                "active_experts": 256 if count else 0,
                "routed_tokens": count * 256,
                "expert_token_counts": counts.copy(),
            }
            for layer_id in range(44)
        ],
    }


def _write_records(path: Path, records: list[dict[str, object]]) -> bytes:
    content = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records).encode()
    path.write_bytes(content)
    return content


def test_manifest_filters_records_and_layer_observations_and_is_invertible(tmp_path: Path) -> None:
    source = tmp_path / "stats.jsonl"
    content = _write_records(
        source,
        [
            _record(0, "prefill", count=1),
            _record(1, "decode_mtp", count=2),
        ],
    )

    manifest = generate_manifest(
        source,
        phases=("decode_mtp",),
        routed_tokens=(512,),
    )

    assert manifest["schema"] == MANIFEST_SCHEMA
    assert manifest["source"] == {
        "records": 2,
        "schema": "pypto-lib.dsv4-moe-stats-jsonl.v1",
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    assert manifest["filters"] == {"phases": ["decode_mtp"], "routed_tokens": [512]}
    assert manifest["topology"] == {"experts": 256, "layers": 44, "local_experts": 32, "ranks": 8}
    assert manifest["expert_loads"] == [[2] * 256 for _ in range(44)]
    assert len(manifest["layers"]) == 44

    for layer_id, layer in enumerate(manifest["layers"]):
        assert layer["layer_id"] == layer_id
        assert layer["kind"] == ("main" if layer_id < 43 else "mtp")
        assert layer["observations"] == 1
        assert layer["routed_tokens"] == 512
        assert layer["estimated_rank_loads"] == [64] * 8
        assert all(len(logical_experts) == 32 for logical_experts in layer["rank_to_logical"])
        flattened = [logical for logical_experts in layer["rank_to_logical"] for logical in logical_experts]
        assert sorted(flattened) == list(range(256))
        assert all(
            layer["logical_to_physical"][logical] == physical for physical, logical in enumerate(flattened)
        )

    first_layer = manifest["layers"][0]
    assert first_layer["rank_to_logical"][0] == list(range(0, 256, 8))
    assert first_layer["logical_to_physical"][0] == 0
    assert first_layer["logical_to_physical"][8] == 1
    assert first_layer["logical_to_physical"][1] == 32


def test_capacity_constrained_lpt_is_deterministic_and_keeps_exact_capacity() -> None:
    weights = [1_000, 900, 800, 700, 600, 500, 400, 300, *([1] * 248)]

    first = capacity_constrained_lpt(weights)
    second = capacity_constrained_lpt(weights)

    assert first == second
    assert all(len(logical_experts) == 32 for logical_experts in first.rank_to_logical)
    assert sorted(first.logical_to_physical) == list(range(256))
    assert sum(first.estimated_rank_loads) == sum(weights)
    assert first.estimated_rank_loads == tuple(
        sum(weights[expert_id] for expert_id in logical_experts) for logical_experts in first.rank_to_logical
    )
    hot_ranks = {first.logical_to_physical[expert_id] // 32 for expert_id in range(8)}
    assert hot_ranks == set(range(8))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("layers", "must contain 44 layers"),
        ("ranks", "ranks must be 8"),
        ("local_experts", "local_experts must be 32"),
        ("counts", "must contain 256 counts"),
        ("routed_tokens", "expert counts sum to"),
        ("active_experts", "expert counts are nonzero"),
    ],
)
def test_source_schema_validation_is_strict(tmp_path: Path, case: str, message: str) -> None:
    record = _record(0, "decode_mtp")
    if case == "layers":
        record["layers"].pop()
    elif case == "ranks":
        record["ranks"] = 7
    elif case == "local_experts":
        record["local_experts"] = 31
    elif case == "counts":
        layer = record["layers"][0]
        layer["expert_token_counts"].pop()
        layer["routed_tokens"] -= 1
        layer["active_experts"] -= 1
    elif case == "routed_tokens":
        record["layers"][0]["routed_tokens"] += 1
    elif case == "active_experts":
        record["layers"][0]["active_experts"] -= 1
    source = tmp_path / f"{case}.jsonl"
    _write_records(source, [record])

    with pytest.raises(PlacementError, match=message):
        generate_manifest(source)


def test_unselected_records_are_still_validated(tmp_path: Path) -> None:
    selected = _record(0, "decode_mtp")
    invalid = _record(1, "prefill")
    invalid["layers"][0]["expert_token_counts"].pop()
    source = tmp_path / "stats.jsonl"
    _write_records(source, [selected, invalid])

    with pytest.raises(PlacementError, match="must contain 256 counts"):
        generate_manifest(source, phases=("decode_mtp",))


def test_dispatch_ids_must_be_strictly_increasing(tmp_path: Path) -> None:
    source = tmp_path / "stats.jsonl"
    _write_records(
        source,
        [
            _record(1, "decode_mtp"),
            _record(1, "decode_mtp"),
        ],
    )

    with pytest.raises(PlacementError, match="must be greater than the previous value"):
        generate_manifest(source)


def test_filters_must_select_positive_observations_for_every_layer(tmp_path: Path) -> None:
    source = tmp_path / "stats.jsonl"
    _write_records(source, [_record(0, "decode_mtp")])

    with pytest.raises(PlacementError, match="no positive routed-token observations"):
        generate_manifest(source, phases=("prefill",))


def test_cli_writes_canonical_compact_manifest(tmp_path: Path) -> None:
    source = tmp_path / "stats.jsonl"
    output = tmp_path / "placement.json"
    _write_records(source, [_record(0, "decode_mtp", count=2)])

    result = subprocess.run(
        [
            sys.executable,
            str(_GENERATOR),
            str(source),
            "--phase",
            "decode_mtp",
            "--routed-tokens",
            "512",
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    text = output.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n " not in text
    assert json.loads(text)["algorithm"]["name"] == "capacity_constrained_lpt"
