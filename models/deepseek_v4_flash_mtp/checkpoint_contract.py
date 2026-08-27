# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Fail-closed manifests for DeepSeek-V4 checkpoints and converted weight caches.

The checkpoint manifest identifies one candidate checkpoint by parsed JSON
semantics, the complete referenced safetensors inventory, and the bytes of every
referenced shard. It deliberately contains no host path and does not claim that
the candidate produced any particular routing-statistics or placement artifact.

The cache-binding manifest is a smaller compatibility contract. It binds a
completed cache to its source checkpoint, converter, tensor ABI, and expert
layout. A physically placed cache must name an external placement fingerprint;
a logical-contiguous cache must remain placement-neutral.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath


CHECKPOINT_MANIFEST_SCHEMA = "pypto-lib.dsv4-checkpoint-manifest.v1"
CACHE_BINDING_MANIFEST_SCHEMA = "pypto-lib.dsv4-weight-cache-binding.v1"

CHECKPOINT_FORMAT_HYBRID = "deepseek-v4-flash-hybrid-mxfp4-mxfp8"
CHECKPOINT_FORMAT_W8A8 = "deepseek-v4-flash-w8a8-compressed-tensors"
CHECKPOINT_FORMATS = (CHECKPOINT_FORMAT_HYBRID, CHECKPOINT_FORMAT_W8A8)

CACHE_LAYOUT_LOGICAL = "logical_contiguous"
CACHE_LAYOUT_PHYSICAL = "physical"
CACHE_LAYOUTS = (CACHE_LAYOUT_LOGICAL, CACHE_LAYOUT_PHYSICAL)

CONFIG_FILENAME = "config.json"
INDEX_FILENAME = "model.safetensors.index.json"
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024
HASH_CHUNK_BYTES = 4 * 1024 * 1024

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class ArtifactContractError(ValueError):
    """Raised when a checkpoint or cache manifest violates its contract."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize one JSON value deterministically for semantic hashing."""
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ArtifactContractError(f"value is not canonical JSON: {error}") from error
    return text.encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    """Return the SHA-256 of canonical JSON semantics."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json_bytes(payload: bytes, location: str) -> object:
    if len(payload) > MAX_JSON_BYTES:
        raise ArtifactContractError(f"{location} is too large: {len(payload)} bytes exceeds {MAX_JSON_BYTES}")

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for key, value in pairs:
            if key in parsed:
                raise ArtifactContractError(f"{location} contains duplicate JSON key {key!r}")
            parsed[key] = value
        return parsed

    def reject_nonfinite(token: str) -> None:
        raise ArtifactContractError(f"{location} contains non-finite JSON number {token}")

    try:
        text = payload.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except ArtifactContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactContractError(f"{location} is not strict UTF-8 JSON: {error}") from error


def _load_json_file(path: Path, location: str) -> object:
    try:
        if not path.is_file():
            raise ArtifactContractError(f"missing {location}: {path.name}")
        return _strict_json_bytes(path.read_bytes(), location)
    except ArtifactContractError:
        raise
    except OSError as error:
        raise ArtifactContractError(f"cannot read {location} {path.name}: {error}") from error


def _require_object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ArtifactContractError(f"{location} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, object], keys: set[str], location: str) -> None:
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ArtifactContractError(
            f"{location} fields do not match schema: missing={missing}, extra={extra}"
        )


def _require_sha256(value: object, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactContractError(f"{location} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, location: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactContractError(f"{location} must be a nonnegative integer")
    return value


def _validate_checkpoint_format(checkpoint_format: object) -> str:
    if checkpoint_format not in CHECKPOINT_FORMATS:
        raise ArtifactContractError(
            f"checkpoint format must be one of {CHECKPOINT_FORMATS}, got {checkpoint_format!r}"
        )
    return str(checkpoint_format)


def _validate_dsv4_config(config: dict[str, object], checkpoint_format: str) -> None:
    model_type = str(config.get("model_type", "")).lower()
    raw_architectures = config.get("architectures", [])
    if not isinstance(raw_architectures, list) or any(
        not isinstance(architecture, str) for architecture in raw_architectures
    ):
        raise ArtifactContractError("config.json architectures must be a list of strings")
    architectures = {architecture.lower() for architecture in raw_architectures}
    if model_type != "deepseek_v4" and "deepseekv4forcausallm" not in architectures:
        raise ArtifactContractError("config.json does not identify a DeepSeek-V4 checkpoint")

    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ArtifactContractError("config.json quantization_config must be an object")
    quant_method = quantization.get("quant_method")
    if checkpoint_format == CHECKPOINT_FORMAT_W8A8:
        if quant_method != "compressed-tensors":
            raise ArtifactContractError("W8A8 checkpoint format requires quant_method='compressed-tensors'")
    elif quant_method == "compressed-tensors":
        raise ArtifactContractError("hybrid MXFP4/MXFP8 format rejects compressed-tensors checkpoints")


def _validate_shard_basename(value: object, location: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ArtifactContractError(f"{location} must be a non-empty shard basename")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or value != posix.name
        or "/" in value
        or "\\" in value
        or any(part in {".", ".."} for part in (*posix.parts, *windows.parts))
    ):
        raise ArtifactContractError(f"{location} must be a shard basename, got {value!r}")
    if not value.endswith(".safetensors"):
        raise ArtifactContractError(f"{location} must end with '.safetensors', got {value!r}")
    return value


def _validate_weight_map(index: dict[str, object]) -> dict[str, str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactContractError("checkpoint index weight_map must be a non-empty object")

    validated: dict[str, str] = {}
    for tensor_name, shard_name in weight_map.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ArtifactContractError("checkpoint index tensor names must be non-empty strings")
        validated[tensor_name] = _validate_shard_basename(
            shard_name,
            f"checkpoint index weight_map[{tensor_name!r}]",
        )
    return validated


def _shape_numel(shape: list[int]) -> int:
    numel = 1
    for dimension in shape:
        numel *= dimension
    return numel


def _validate_safetensors_metadata(value: object, shard_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
    ):
        raise ArtifactContractError(f"{shard_name} __metadata__ must map strings to strings")


def _parse_safetensors_header(
    header_bytes: bytes,
    *,
    shard_name: str,
    data_size: int,
) -> list[dict[str, object]]:
    header = _require_object(
        _strict_json_bytes(header_bytes, f"{shard_name} safetensors header"),
        f"{shard_name} safetensors header",
    )
    _validate_safetensors_metadata(header.pop("__metadata__", None), shard_name)
    if not header:
        raise ArtifactContractError(f"{shard_name} safetensors header contains no tensors")

    tensors: list[dict[str, object]] = []
    intervals: list[tuple[int, int, str]] = []
    for tensor_name, raw_info in header.items():
        if not isinstance(tensor_name, str) or not tensor_name:
            raise ArtifactContractError(f"{shard_name} contains an empty tensor name")
        info = _require_object(raw_info, f"{shard_name} tensor {tensor_name!r}")
        _require_exact_keys(info, {"dtype", "shape", "data_offsets"}, f"{shard_name}:{tensor_name}")

        dtype = info["dtype"]
        if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ArtifactContractError(
                f"{shard_name}:{tensor_name} has unsupported safetensors dtype {dtype!r}"
            )
        raw_shape = info["shape"]
        if not isinstance(raw_shape, list):
            raise ArtifactContractError(f"{shard_name}:{tensor_name} shape must be a list")
        shape = [
            _require_nonnegative_int(dimension, f"{shard_name}:{tensor_name} shape[{index}]")
            for index, dimension in enumerate(raw_shape)
        ]
        raw_offsets = info["data_offsets"]
        if not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
            raise ArtifactContractError(f"{shard_name}:{tensor_name} data_offsets must contain two integers")
        start = _require_nonnegative_int(raw_offsets[0], f"{shard_name}:{tensor_name} offset start")
        end = _require_nonnegative_int(raw_offsets[1], f"{shard_name}:{tensor_name} offset end")
        if start > end or end > data_size:
            raise ArtifactContractError(
                f"{shard_name}:{tensor_name} data_offsets [{start}, {end}] are out of bounds "
                f"for {data_size} data bytes"
            )
        expected_bytes = _shape_numel(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ArtifactContractError(
                f"{shard_name}:{tensor_name} byte range is {end - start}, expected {expected_bytes}"
            )
        tensors.append(
            {
                "name": tensor_name,
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, end],
            }
        )
        intervals.append((start, end, tensor_name))

    cursor = 0
    for start, end, tensor_name in sorted(intervals):
        if start != cursor:
            relation = "overlaps prior data" if start < cursor else "leaves an unindexed hole"
            raise ArtifactContractError(
                f"{shard_name}:{tensor_name} starts at {start}, expected {cursor}; {relation}"
            )
        cursor = end
    if cursor != data_size:
        raise ArtifactContractError(
            f"{shard_name} indexes {cursor} data bytes, but the shard contains {data_size}"
        )
    return tensors


def _read_safetensors_shard(path: Path, shard_name: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    try:
        with path.open("rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactContractError(f"checkpoint shard is not a regular file: {shard_name}")
            if before.st_size < 8:
                raise ArtifactContractError(f"{shard_name} is too small for a safetensors header")

            prefix = source.read(8)
            if len(prefix) != 8:
                raise ArtifactContractError(f"cannot read {shard_name} safetensors header length")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 1 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
                raise ArtifactContractError(
                    f"{shard_name} header size {header_size} is outside [2, {MAX_SAFETENSORS_HEADER_BYTES}]"
                )
            data_offset = 8 + header_size
            if data_offset > before.st_size:
                raise ArtifactContractError(
                    f"{shard_name} header extends past the {before.st_size}-byte file"
                )
            header_bytes = source.read(header_size)
            if len(header_bytes) != header_size:
                raise ArtifactContractError(f"cannot read the complete {shard_name} header")
            data_size = before.st_size - data_offset
            tensors = _parse_safetensors_header(
                header_bytes,
                shard_name=shard_name,
                data_size=data_size,
            )

            source.seek(0)
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
            after = os.fstat(source.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ArtifactContractError(f"checkpoint shard changed while hashing: {shard_name}")
    except ArtifactContractError:
        raise
    except FileNotFoundError as error:
        raise ArtifactContractError(f"missing checkpoint shard: {shard_name}") from error
    except OSError as error:
        raise ArtifactContractError(f"cannot read checkpoint shard {shard_name}: {error}") from error

    shard = {
        "name": shard_name,
        "size": before.st_size,
        "header_size": header_size,
        "data_size": data_size,
        "sha256": digest.hexdigest(),
    }
    return shard, tensors


def _validate_inventory_format(tensors: list[dict[str, object]], checkpoint_format: str) -> None:
    dtypes = {tensor["dtype"] for tensor in tensors}
    if checkpoint_format == CHECKPOINT_FORMAT_W8A8 and "I8" not in dtypes:
        raise ArtifactContractError("W8A8 checkpoint contains no I8 weight tensor")
    if checkpoint_format == CHECKPOINT_FORMAT_HYBRID:
        has_mxfp8_weight = any(str(dtype).startswith("F8_E4M3") for dtype in dtypes)
        if not has_mxfp8_weight or "F8_E8M0" not in dtypes:
            raise ArtifactContractError(
                "hybrid checkpoint requires both an E4M3 weight and an F8_E8M0 scale tensor"
            )


def build_checkpoint_manifest(
    checkpoint_dir: Path | str,
    *,
    checkpoint_format: str,
) -> dict[str, object]:
    """Validate and fingerprint one local DeepSeek-V4 checkpoint candidate."""
    checkpoint_format = _validate_checkpoint_format(checkpoint_format)
    root = Path(checkpoint_dir)
    if not root.is_dir():
        raise ArtifactContractError(f"checkpoint directory does not exist: {root}")

    config = _require_object(
        _load_json_file(root / CONFIG_FILENAME, CONFIG_FILENAME),
        CONFIG_FILENAME,
    )
    _validate_dsv4_config(config, checkpoint_format)
    index = _require_object(
        _load_json_file(root / INDEX_FILENAME, INDEX_FILENAME),
        INDEX_FILENAME,
    )
    weight_map = _validate_weight_map(index)

    shards: list[dict[str, object]] = []
    actual_tensor_shards: dict[str, str] = {}
    tensor_info: dict[str, dict[str, object]] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard, shard_tensors = _read_safetensors_shard(root / shard_name, shard_name)
        shards.append(shard)
        for tensor in shard_tensors:
            tensor_name = str(tensor["name"])
            if tensor_name in actual_tensor_shards:
                raise ArtifactContractError(f"tensor {tensor_name!r} appears in multiple referenced shards")
            actual_tensor_shards[tensor_name] = shard_name
            tensor_info[tensor_name] = tensor

    expected_names = set(weight_map)
    actual_names = set(actual_tensor_shards)
    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    wrong_shards = sorted(
        tensor_name
        for tensor_name in expected_names & actual_names
        if weight_map[tensor_name] != actual_tensor_shards[tensor_name]
    )
    if missing or extra or wrong_shards:
        raise ArtifactContractError(
            "checkpoint index/header tensor mismatch: "
            f"missing={missing[:8]}, extra={extra[:8]}, wrong_shard={wrong_shards[:8]}"
        )

    tensors = []
    for tensor_name in sorted(weight_map):
        info = tensor_info[tensor_name]
        tensors.append(
            {
                "name": tensor_name,
                "shard": weight_map[tensor_name],
                "dtype": info["dtype"],
                "shape": info["shape"],
                "data_offsets": info["data_offsets"],
            }
        )
    _validate_inventory_format(tensors, checkpoint_format)

    payload: dict[str, object] = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "format": checkpoint_format,
        "config": {
            "name": CONFIG_FILENAME,
            "semantic_sha256": canonical_json_sha256(config),
        },
        "index": {
            "name": INDEX_FILENAME,
            "semantic_sha256": canonical_json_sha256(index),
        },
        "shards": shards,
        "tensors": tensors,
    }
    return {**payload, "fingerprint": canonical_json_sha256(payload)}


def _validate_checkpoint_manifest_structure(manifest: object) -> dict[str, object]:
    parsed = _require_object(manifest, "checkpoint manifest")
    _require_exact_keys(
        parsed,
        {"schema", "format", "config", "index", "shards", "tensors", "fingerprint"},
        "checkpoint manifest",
    )
    if parsed["schema"] != CHECKPOINT_MANIFEST_SCHEMA:
        raise ArtifactContractError(f"checkpoint manifest schema must be {CHECKPOINT_MANIFEST_SCHEMA!r}")
    checkpoint_format = _validate_checkpoint_format(parsed["format"])

    for field, expected_name in (("config", CONFIG_FILENAME), ("index", INDEX_FILENAME)):
        metadata = _require_object(parsed[field], f"checkpoint manifest {field}")
        _require_exact_keys(metadata, {"name", "semantic_sha256"}, f"checkpoint manifest {field}")
        if metadata["name"] != expected_name:
            raise ArtifactContractError(f"checkpoint manifest {field}.name must be {expected_name!r}")
        _require_sha256(metadata["semantic_sha256"], f"checkpoint manifest {field}.semantic_sha256")

    raw_shards = parsed["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ArtifactContractError("checkpoint manifest shards must be a non-empty list")
    shard_by_name: dict[str, dict[str, object]] = {}
    shard_names = []
    for index, raw_shard in enumerate(raw_shards):
        location = f"checkpoint manifest shards[{index}]"
        shard = _require_object(raw_shard, location)
        _require_exact_keys(
            shard,
            {"name", "size", "header_size", "data_size", "sha256"},
            location,
        )
        name = _validate_shard_basename(shard["name"], f"{location}.name")
        if name in shard_by_name:
            raise ArtifactContractError(f"checkpoint manifest contains duplicate shard {name!r}")
        size = _require_nonnegative_int(shard["size"], f"{location}.size")
        header_size = _require_nonnegative_int(shard["header_size"], f"{location}.header_size")
        data_size = _require_nonnegative_int(shard["data_size"], f"{location}.data_size")
        if not 1 < header_size <= MAX_SAFETENSORS_HEADER_BYTES:
            raise ArtifactContractError(f"{location}.header_size is outside the supported range")
        if size != 8 + header_size + data_size:
            raise ArtifactContractError(f"{location} size does not match header_size + data_size")
        _require_sha256(shard["sha256"], f"{location}.sha256")
        shard_by_name[name] = shard
        shard_names.append(name)
    if shard_names != sorted(shard_names):
        raise ArtifactContractError("checkpoint manifest shards must be sorted by name")

    raw_tensors = parsed["tensors"]
    if not isinstance(raw_tensors, list) or not raw_tensors:
        raise ArtifactContractError("checkpoint manifest tensors must be a non-empty list")
    tensor_names = []
    intervals_by_shard: dict[str, list[tuple[int, int, str]]] = {name: [] for name in shard_by_name}
    validated_tensors: list[dict[str, object]] = []
    seen_tensors = set()
    for index, raw_tensor in enumerate(raw_tensors):
        location = f"checkpoint manifest tensors[{index}]"
        tensor = _require_object(raw_tensor, location)
        _require_exact_keys(
            tensor,
            {"name", "shard", "dtype", "shape", "data_offsets"},
            location,
        )
        name = tensor["name"]
        if not isinstance(name, str) or not name:
            raise ArtifactContractError(f"{location}.name must be a non-empty string")
        if name in seen_tensors:
            raise ArtifactContractError(f"checkpoint manifest contains duplicate tensor {name!r}")
        seen_tensors.add(name)
        tensor_names.append(name)

        shard_name = _validate_shard_basename(tensor["shard"], f"{location}.shard")
        if shard_name not in shard_by_name:
            raise ArtifactContractError(f"{location}.shard references unknown shard {shard_name!r}")
        dtype = tensor["dtype"]
        if not isinstance(dtype, str) or dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise ArtifactContractError(f"{location}.dtype is unsupported: {dtype!r}")
        raw_shape = tensor["shape"]
        if not isinstance(raw_shape, list):
            raise ArtifactContractError(f"{location}.shape must be a list")
        shape = [
            _require_nonnegative_int(dimension, f"{location}.shape[{dimension_index}]")
            for dimension_index, dimension in enumerate(raw_shape)
        ]
        offsets = tensor["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ArtifactContractError(f"{location}.data_offsets must contain two integers")
        start = _require_nonnegative_int(offsets[0], f"{location}.data_offsets[0]")
        end = _require_nonnegative_int(offsets[1], f"{location}.data_offsets[1]")
        data_size = int(shard_by_name[shard_name]["data_size"])
        if start > end or end > data_size:
            raise ArtifactContractError(f"{location}.data_offsets are outside the shard data")
        expected_bytes = _shape_numel(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ArtifactContractError(f"{location}.data_offsets do not match shape and dtype")
        intervals_by_shard[shard_name].append((start, end, name))
        validated_tensors.append(tensor)

    if tensor_names != sorted(tensor_names):
        raise ArtifactContractError("checkpoint manifest tensors must be sorted by name")
    for shard_name, intervals in intervals_by_shard.items():
        cursor = 0
        for start, end, tensor_name in sorted(intervals):
            if start != cursor:
                raise ArtifactContractError(
                    f"checkpoint manifest tensor {tensor_name!r} does not continuously index {shard_name}"
                )
            cursor = end
        if cursor != shard_by_name[shard_name]["data_size"]:
            raise ArtifactContractError(
                f"checkpoint manifest tensors do not cover all data in shard {shard_name}"
            )
    _validate_inventory_format(validated_tensors, checkpoint_format)

    fingerprint = _require_sha256(parsed["fingerprint"], "checkpoint manifest fingerprint")
    payload = dict(parsed)
    payload.pop("fingerprint")
    expected_fingerprint = canonical_json_sha256(payload)
    if fingerprint != expected_fingerprint:
        raise ArtifactContractError("checkpoint manifest fingerprint does not match its canonical contents")
    return parsed


def validate_checkpoint_manifest(manifest: object) -> dict[str, object]:
    """Validate a checkpoint manifest without trusting or opening host paths."""
    return _validate_checkpoint_manifest_structure(manifest)


def load_checkpoint_manifest(path: Path | str) -> dict[str, object]:
    """Load a checkpoint manifest with duplicate-key and schema rejection."""
    manifest_path = Path(path)
    manifest = _load_json_file(manifest_path, "checkpoint manifest")
    return validate_checkpoint_manifest(manifest)


def verify_checkpoint_manifest(
    checkpoint_dir: Path | str,
    manifest: object,
) -> dict[str, object]:
    """Rebuild a checkpoint fingerprint and require an exact manifest match."""
    expected = validate_checkpoint_manifest(manifest)
    actual = build_checkpoint_manifest(
        checkpoint_dir,
        checkpoint_format=str(expected["format"]),
    )
    if actual != expected:
        raise ArtifactContractError("checkpoint contents do not match the supplied checkpoint manifest")
    return expected


def _validate_layout_binding(layout_kind: object, placement_fingerprint: object) -> tuple[str, str | None]:
    if layout_kind not in CACHE_LAYOUTS:
        raise ArtifactContractError(f"cache layout must be one of {CACHE_LAYOUTS}, got {layout_kind!r}")
    layout_kind = str(layout_kind)
    if layout_kind == CACHE_LAYOUT_LOGICAL:
        if placement_fingerprint is not None:
            raise ArtifactContractError("logical_contiguous cache must not carry a placement fingerprint")
        return layout_kind, None
    return layout_kind, _require_sha256(
        placement_fingerprint,
        "physical cache placement fingerprint",
    )


def build_cache_binding_manifest(
    *,
    source_checkpoint_manifest: object,
    converter_fingerprint: str,
    abi_fingerprint: str,
    layout_kind: str,
    placement_fingerprint: str | None = None,
) -> dict[str, object]:
    """Build a completed cache binding without inferring a stats placement."""
    checkpoint = validate_checkpoint_manifest(source_checkpoint_manifest)
    converter_fingerprint = _require_sha256(
        converter_fingerprint,
        "converter fingerprint",
    )
    abi_fingerprint = _require_sha256(abi_fingerprint, "ABI fingerprint")
    layout_kind, placement_fingerprint = _validate_layout_binding(
        layout_kind,
        placement_fingerprint,
    )
    payload: dict[str, object] = {
        "schema": CACHE_BINDING_MANIFEST_SCHEMA,
        "complete": True,
        "source": {
            "checkpoint_format": checkpoint["format"],
            "checkpoint_fingerprint": checkpoint["fingerprint"],
        },
        "converter": {"fingerprint": converter_fingerprint},
        "abi": {"fingerprint": abi_fingerprint},
        "layout": {
            "kind": layout_kind,
            "placement_fingerprint": placement_fingerprint,
        },
    }
    return {**payload, "binding_fingerprint": canonical_json_sha256(payload)}


def _validate_cache_binding_structure(manifest: object) -> dict[str, object]:
    parsed = _require_object(manifest, "cache binding manifest")
    _require_exact_keys(
        parsed,
        {
            "schema",
            "complete",
            "source",
            "converter",
            "abi",
            "layout",
            "binding_fingerprint",
        },
        "cache binding manifest",
    )
    if parsed["schema"] != CACHE_BINDING_MANIFEST_SCHEMA:
        raise ArtifactContractError(
            f"cache binding manifest schema must be {CACHE_BINDING_MANIFEST_SCHEMA!r}"
        )
    if parsed["complete"] is not True:
        raise ArtifactContractError("cache binding manifest must declare complete=true")

    source = _require_object(parsed["source"], "cache binding manifest source")
    _require_exact_keys(
        source,
        {"checkpoint_format", "checkpoint_fingerprint"},
        "cache binding manifest source",
    )
    _validate_checkpoint_format(source["checkpoint_format"])
    _require_sha256(
        source["checkpoint_fingerprint"],
        "cache binding manifest source.checkpoint_fingerprint",
    )

    for field in ("converter", "abi"):
        binding = _require_object(parsed[field], f"cache binding manifest {field}")
        _require_exact_keys(binding, {"fingerprint"}, f"cache binding manifest {field}")
        _require_sha256(binding["fingerprint"], f"cache binding manifest {field}.fingerprint")

    layout = _require_object(parsed["layout"], "cache binding manifest layout")
    _require_exact_keys(
        layout,
        {"kind", "placement_fingerprint"},
        "cache binding manifest layout",
    )
    _validate_layout_binding(layout["kind"], layout["placement_fingerprint"])

    fingerprint = _require_sha256(
        parsed["binding_fingerprint"],
        "cache binding manifest binding_fingerprint",
    )
    payload = dict(parsed)
    payload.pop("binding_fingerprint")
    if fingerprint != canonical_json_sha256(payload):
        raise ArtifactContractError("cache binding fingerprint does not match its canonical contents")
    return parsed


def validate_cache_binding_manifest(
    manifest: object,
    *,
    expected_source_checkpoint_format: str,
    expected_source_checkpoint_fingerprint: str,
    expected_converter_fingerprint: str,
    expected_abi_fingerprint: str,
    expected_layout_kind: str,
    expected_placement_fingerprint: str | None,
) -> dict[str, object]:
    """Validate a cache binding against every caller-selected compatibility key."""
    parsed = _validate_cache_binding_structure(manifest)
    expected_source_checkpoint_format = _validate_checkpoint_format(expected_source_checkpoint_format)
    expected_source_checkpoint_fingerprint = _require_sha256(
        expected_source_checkpoint_fingerprint,
        "expected source checkpoint fingerprint",
    )
    expected_converter_fingerprint = _require_sha256(
        expected_converter_fingerprint,
        "expected converter fingerprint",
    )
    expected_abi_fingerprint = _require_sha256(
        expected_abi_fingerprint,
        "expected ABI fingerprint",
    )
    expected_layout_kind, expected_placement_fingerprint = _validate_layout_binding(
        expected_layout_kind,
        expected_placement_fingerprint,
    )

    actual = {
        "source checkpoint format": parsed["source"]["checkpoint_format"],
        "source checkpoint fingerprint": parsed["source"]["checkpoint_fingerprint"],
        "converter fingerprint": parsed["converter"]["fingerprint"],
        "ABI fingerprint": parsed["abi"]["fingerprint"],
        "layout kind": parsed["layout"]["kind"],
        "placement fingerprint": parsed["layout"]["placement_fingerprint"],
    }
    expected = {
        "source checkpoint format": expected_source_checkpoint_format,
        "source checkpoint fingerprint": expected_source_checkpoint_fingerprint,
        "converter fingerprint": expected_converter_fingerprint,
        "ABI fingerprint": expected_abi_fingerprint,
        "layout kind": expected_layout_kind,
        "placement fingerprint": expected_placement_fingerprint,
    }
    mismatches = [
        f"{field}: expected {expected[field]!r}, got {actual[field]!r}"
        for field in expected
        if expected[field] != actual[field]
    ]
    if mismatches:
        raise ArtifactContractError("cache binding mismatch: " + "; ".join(mismatches))
    return parsed


def load_cache_binding_manifest(
    path: Path | str,
    *,
    expected_source_checkpoint_format: str,
    expected_source_checkpoint_fingerprint: str,
    expected_converter_fingerprint: str,
    expected_abi_fingerprint: str,
    expected_layout_kind: str,
    expected_placement_fingerprint: str | None,
) -> dict[str, object]:
    """Load and validate one cache binding against explicit caller expectations."""
    manifest = _load_json_file(Path(path), "cache binding manifest")
    return validate_cache_binding_manifest(
        manifest,
        expected_source_checkpoint_format=expected_source_checkpoint_format,
        expected_source_checkpoint_fingerprint=expected_source_checkpoint_fingerprint,
        expected_converter_fingerprint=expected_converter_fingerprint,
        expected_abi_fingerprint=expected_abi_fingerprint,
        expected_layout_kind=expected_layout_kind,
        expected_placement_fingerprint=expected_placement_fingerprint,
    )
