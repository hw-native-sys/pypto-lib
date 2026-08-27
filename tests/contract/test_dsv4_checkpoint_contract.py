# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path

import pytest

from models.deepseek_v4_flash_mtp.checkpoint_contract import (
    CACHE_LAYOUT_LOGICAL,
    CACHE_LAYOUT_PHYSICAL,
    CHECKPOINT_FORMAT_HYBRID,
    CHECKPOINT_FORMAT_W8A8,
    ArtifactContractError,
    build_cache_binding_manifest,
    build_checkpoint_manifest,
    load_cache_binding_manifest,
    load_checkpoint_manifest,
    validate_cache_binding_manifest,
    validate_checkpoint_manifest,
    verify_checkpoint_manifest,
)


_SHARD_NAME = "model-00001-of-00001.safetensors"
_DIGESTS = {name: name * 64 for name in "1234"}


def _write_json(path: Path, value: object, *, indent: int | None = None) -> None:
    path.write_text(json.dumps(value, indent=indent), encoding="utf-8")


def _write_raw_safetensors(path: Path, header: bytes, data: bytes) -> None:
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


def _write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> None:
    header: dict[str, object] = {}
    data = bytearray()
    for name, (dtype, shape, payload) in tensors.items():
        start = len(data)
        data.extend(payload)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(data)],
        }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    _write_raw_safetensors(path, encoded, bytes(data))


def _write_checkpoint(checkpoint_dir: Path, checkpoint_format: str) -> dict[str, object]:
    checkpoint_dir.mkdir()
    if checkpoint_format == CHECKPOINT_FORMAT_W8A8:
        quant_method = "compressed-tensors"
        tensors = {
            "model.layers.0.self_attn.q_proj.weight": ("I8", [2], b"\x01\x02"),
            "model.layers.0.self_attn.q_proj.weight_scale": (
                "F32",
                [1],
                struct.pack("<f", 0.5),
            ),
        }
    else:
        quant_method = "fp8"
        tensors = {
            "model.layers.0.mlp.experts.0.down_proj.weight": (
                "F8_E4M3",
                [2],
                b"\x01\x02",
            ),
            "model.layers.0.mlp.experts.0.down_proj.weight_scale": (
                "F8_E8M0",
                [1],
                b"\x7f",
            ),
        }

    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "quantization_config": {"quant_method": quant_method},
    }
    index = {
        "metadata": {"total_size": sum(len(payload) for _, _, payload in tensors.values())},
        "weight_map": {name: _SHARD_NAME for name in tensors},
    }
    _write_json(checkpoint_dir / "config.json", config)
    _write_json(checkpoint_dir / "model.safetensors.index.json", index)
    _write_safetensors(checkpoint_dir / _SHARD_NAME, tensors)
    return {"config": config, "index": index}


@pytest.mark.parametrize("checkpoint_format", [CHECKPOINT_FORMAT_W8A8, CHECKPOINT_FORMAT_HYBRID])
def test_checkpoint_manifest_covers_source_semantics_shards_and_tensors(
    tmp_path: Path,
    checkpoint_format: str,
) -> None:
    source = _write_checkpoint(tmp_path / "candidate", checkpoint_format)

    manifest = build_checkpoint_manifest(tmp_path / "candidate", checkpoint_format=checkpoint_format)

    assert manifest["format"] == checkpoint_format
    assert manifest["config"]["name"] == "config.json"
    assert manifest["index"]["name"] == "model.safetensors.index.json"
    assert manifest["shards"][0]["name"] == _SHARD_NAME
    assert manifest["shards"][0]["size"] > manifest["shards"][0]["data_size"]
    assert {tensor["name"] for tensor in manifest["tensors"]} == set(source["index"]["weight_map"])
    assert str(tmp_path) not in json.dumps(manifest)
    assert validate_checkpoint_manifest(manifest) == manifest


def test_checkpoint_fingerprint_uses_json_semantics_and_verifies_source_bytes(tmp_path: Path) -> None:
    source = _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    checkpoint_dir = tmp_path / "candidate"
    manifest = build_checkpoint_manifest(checkpoint_dir, checkpoint_format=CHECKPOINT_FORMAT_W8A8)

    _write_json(checkpoint_dir / "config.json", source["config"], indent=4)
    reordered_index = {
        "weight_map": source["index"]["weight_map"],
        "metadata": source["index"]["metadata"],
    }
    _write_json(checkpoint_dir / "model.safetensors.index.json", reordered_index, indent=2)
    assert (
        build_checkpoint_manifest(
            checkpoint_dir,
            checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        )
        == manifest
    )

    manifest_path = tmp_path / "checkpoint-manifest.json"
    _write_json(manifest_path, manifest)
    assert load_checkpoint_manifest(manifest_path) == manifest
    assert verify_checkpoint_manifest(checkpoint_dir, manifest) == manifest

    shard_path = checkpoint_dir / _SHARD_NAME
    shard = bytearray(shard_path.read_bytes())
    shard[-1] ^= 1
    shard_path.write_bytes(shard)
    with pytest.raises(ArtifactContractError, match="do not match"):
        verify_checkpoint_manifest(checkpoint_dir, manifest)


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors.index.json"])
def test_checkpoint_rejects_duplicate_json_keys(tmp_path: Path, filename: str) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    duplicate = (
        b'{"model_type":"deepseek_v4","model_type":"deepseek_v4",'
        b'"quantization_config":{"quant_method":"compressed-tensors"}}'
        if filename == "config.json"
        else b'{"weight_map":{"x":"model.safetensors"},"weight_map":{"x":"model.safetensors"}}'
    )
    (tmp_path / "candidate" / filename).write_bytes(duplicate)

    with pytest.raises(ArtifactContractError, match="duplicate JSON key"):
        build_checkpoint_manifest(
            tmp_path / "candidate",
            checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        )


def test_checkpoint_rejects_duplicate_safetensors_header_keys(tmp_path: Path) -> None:
    source = _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    tensor_name = next(iter(source["index"]["weight_map"]))
    info = b'{"dtype":"I8","shape":[2],"data_offsets":[0,2]}'
    header = b'{"' + tensor_name.encode() + b'":' + info + b',"' + tensor_name.encode() + b'":' + info + b"}"
    _write_raw_safetensors(tmp_path / "candidate" / _SHARD_NAME, header, b"\x01\x02")

    with pytest.raises(ArtifactContractError, match="duplicate JSON key"):
        build_checkpoint_manifest(
            tmp_path / "candidate",
            checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        )


@pytest.mark.parametrize(
    "shard_name",
    [
        "/weights/model.safetensors",
        "../model.safetensors",
        "nested/model.safetensors",
        "C:\\weights\\model.safetensors",
    ],
)
def test_checkpoint_rejects_non_basename_shard_references(
    tmp_path: Path,
    shard_name: str,
) -> None:
    source = _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    source["index"]["weight_map"] = {name: shard_name for name in source["index"]["weight_map"]}
    _write_json(tmp_path / "candidate" / "model.safetensors.index.json", source["index"])

    with pytest.raises(ArtifactContractError, match="must be a shard basename"):
        build_checkpoint_manifest(
            tmp_path / "candidate",
            checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        )


def test_checkpoint_rejects_missing_referenced_shard(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    (tmp_path / "candidate" / _SHARD_NAME).unlink()

    with pytest.raises(ArtifactContractError, match="missing checkpoint shard"):
        build_checkpoint_manifest(
            tmp_path / "candidate",
            checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        )


def test_checkpoint_rejects_index_header_inventory_mismatch(tmp_path: Path) -> None:
    source = _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    weight_map = source["index"]["weight_map"]
    weight_map["model.layers.0.unindexed.weight"] = _SHARD_NAME
    _write_json(tmp_path / "candidate" / "model.safetensors.index.json", source["index"])

    with pytest.raises(ArtifactContractError, match="index/header tensor mismatch"):
        build_checkpoint_manifest(
            tmp_path / "candidate",
            checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        )


def test_checkpoint_rejects_out_of_bounds_tensor_data(tmp_path: Path) -> None:
    source = _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    tensor_name = next(iter(source["index"]["weight_map"]))
    header = json.dumps(
        {tensor_name: {"dtype": "I8", "shape": [3], "data_offsets": [0, 3]}},
        separators=(",", ":"),
    ).encode()
    _write_raw_safetensors(tmp_path / "candidate" / _SHARD_NAME, header, b"\x01\x02")

    with pytest.raises(ArtifactContractError, match="out of bounds"):
        build_checkpoint_manifest(
            tmp_path / "candidate",
            checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        )


def test_checkpoint_format_must_match_config_and_tensor_evidence(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)

    with pytest.raises(ArtifactContractError, match="rejects compressed-tensors"):
        build_checkpoint_manifest(
            tmp_path / "candidate",
            checkpoint_format=CHECKPOINT_FORMAT_HYBRID,
        )


def test_checkpoint_manifest_rejects_self_inconsistent_fingerprint(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    manifest = build_checkpoint_manifest(
        tmp_path / "candidate",
        checkpoint_format=CHECKPOINT_FORMAT_W8A8,
    )
    manifest["shards"][0]["sha256"] = _DIGESTS["1"]

    with pytest.raises(ArtifactContractError, match="fingerprint does not match"):
        validate_checkpoint_manifest(manifest)


def _build_logical_binding(checkpoint_manifest: object) -> dict[str, object]:
    return build_cache_binding_manifest(
        source_checkpoint_manifest=checkpoint_manifest,
        converter_fingerprint=_DIGESTS["1"],
        abi_fingerprint=_DIGESTS["2"],
        layout_kind=CACHE_LAYOUT_LOGICAL,
    )


def _validate_logical_binding(binding: object) -> dict[str, object]:
    return validate_cache_binding_manifest(
        binding,
        expected_source_checkpoint_format=CHECKPOINT_FORMAT_W8A8,
        expected_source_checkpoint_fingerprint=binding["source"]["checkpoint_fingerprint"],
        expected_converter_fingerprint=_DIGESTS["1"],
        expected_abi_fingerprint=_DIGESTS["2"],
        expected_layout_kind=CACHE_LAYOUT_LOGICAL,
        expected_placement_fingerprint=None,
    )


def test_logical_cache_binding_is_complete_and_placement_neutral(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    checkpoint = build_checkpoint_manifest(
        tmp_path / "candidate",
        checkpoint_format=CHECKPOINT_FORMAT_W8A8,
    )
    binding = _build_logical_binding(checkpoint)

    assert binding["complete"] is True
    assert binding["source"]["checkpoint_fingerprint"] == checkpoint["fingerprint"]
    assert binding["layout"] == {
        "kind": CACHE_LAYOUT_LOGICAL,
        "placement_fingerprint": None,
    }
    assert _validate_logical_binding(binding) == binding

    path = tmp_path / "cache-binding.json"
    _write_json(path, binding)
    assert (
        load_cache_binding_manifest(
            path,
            expected_source_checkpoint_format=CHECKPOINT_FORMAT_W8A8,
            expected_source_checkpoint_fingerprint=checkpoint["fingerprint"],
            expected_converter_fingerprint=_DIGESTS["1"],
            expected_abi_fingerprint=_DIGESTS["2"],
            expected_layout_kind=CACHE_LAYOUT_LOGICAL,
            expected_placement_fingerprint=None,
        )
        == binding
    )


def test_physical_cache_binding_requires_explicit_placement(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    checkpoint = build_checkpoint_manifest(
        tmp_path / "candidate",
        checkpoint_format=CHECKPOINT_FORMAT_W8A8,
    )

    with pytest.raises(ArtifactContractError, match="placement fingerprint"):
        build_cache_binding_manifest(
            source_checkpoint_manifest=checkpoint,
            converter_fingerprint=_DIGESTS["1"],
            abi_fingerprint=_DIGESTS["2"],
            layout_kind=CACHE_LAYOUT_PHYSICAL,
        )
    with pytest.raises(ArtifactContractError, match="must not carry"):
        build_cache_binding_manifest(
            source_checkpoint_manifest=checkpoint,
            converter_fingerprint=_DIGESTS["1"],
            abi_fingerprint=_DIGESTS["2"],
            layout_kind=CACHE_LAYOUT_LOGICAL,
            placement_fingerprint=_DIGESTS["3"],
        )

    binding = build_cache_binding_manifest(
        source_checkpoint_manifest=checkpoint,
        converter_fingerprint=_DIGESTS["1"],
        abi_fingerprint=_DIGESTS["2"],
        layout_kind=CACHE_LAYOUT_PHYSICAL,
        placement_fingerprint=_DIGESTS["3"],
    )
    assert binding["layout"]["placement_fingerprint"] == _DIGESTS["3"]


@pytest.mark.parametrize(
    ("field", "expected", "message"),
    [
        ("source", _DIGESTS["3"], "source checkpoint fingerprint"),
        ("converter", _DIGESTS["3"], "converter fingerprint"),
        ("abi", _DIGESTS["3"], "ABI fingerprint"),
    ],
)
def test_cache_binding_rejects_explicit_compatibility_mismatch(
    tmp_path: Path,
    field: str,
    expected: str,
    message: str,
) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    checkpoint = build_checkpoint_manifest(
        tmp_path / "candidate",
        checkpoint_format=CHECKPOINT_FORMAT_W8A8,
    )
    binding = _build_logical_binding(checkpoint)
    arguments = {
        "expected_source_checkpoint_format": CHECKPOINT_FORMAT_W8A8,
        "expected_source_checkpoint_fingerprint": checkpoint["fingerprint"],
        "expected_converter_fingerprint": _DIGESTS["1"],
        "expected_abi_fingerprint": _DIGESTS["2"],
        "expected_layout_kind": CACHE_LAYOUT_LOGICAL,
        "expected_placement_fingerprint": None,
    }
    key = {
        "source": "expected_source_checkpoint_fingerprint",
        "converter": "expected_converter_fingerprint",
        "abi": "expected_abi_fingerprint",
    }[field]
    arguments[key] = expected

    with pytest.raises(ArtifactContractError, match=message):
        validate_cache_binding_manifest(binding, **arguments)


def test_cache_binding_rejects_incomplete_or_tampered_manifest(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path / "candidate", CHECKPOINT_FORMAT_W8A8)
    checkpoint = build_checkpoint_manifest(
        tmp_path / "candidate",
        checkpoint_format=CHECKPOINT_FORMAT_W8A8,
    )
    binding = _build_logical_binding(checkpoint)

    incomplete = copy.deepcopy(binding)
    incomplete["complete"] = False
    with pytest.raises(ArtifactContractError, match="complete=true"):
        _validate_logical_binding(incomplete)

    tampered = copy.deepcopy(binding)
    tampered["abi"]["fingerprint"] = _DIGESTS["4"]
    with pytest.raises(ArtifactContractError, match="binding fingerprint does not match"):
        _validate_logical_binding(tampered)
