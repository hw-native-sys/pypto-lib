# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Canonical, low-volume device-output artifacts for placement A/B checks.

The artifact deliberately stores logical output rows rather than an entire
device-output dictionary. Dense outputs use their leading dimensions as logical
row keys. Paged cache outputs are gathered through explicit slot mappings and
use ``(rank, layer, token)`` as placement-independent row keys.

The directory format is versioned and fail-closed: metadata, capture contracts,
logical keys, row coverage, file names, byte sizes, and SHA-256 digests must all
match exactly before numerical comparison starts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence, TypeAlias

import torch


ARTIFACT_FORMAT = "dsv4-device-output-ab"
ARTIFACT_VERSION = 2
PLACEMENTS = frozenset({"contiguous", "stats", "eplb"})

_MANIFEST_NAME = "manifest.json"
_MANIFEST_DIGEST_NAME = "manifest.sha256"
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PAYLOAD_RE = re.compile(r"tensor-[0-9]{4}-(?:row-keys|values)\.bin\Z")
_INTEGER_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)

_DTYPE_BY_NAME = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
for _optional_dtype_name in (
    "uint16",
    "uint32",
    "uint64",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
):
    if hasattr(torch, _optional_dtype_name):
        _DTYPE_BY_NAME[_optional_dtype_name] = getattr(torch, _optional_dtype_name)
_NAME_BY_DTYPE = {dtype: name for name, dtype in _DTYPE_BY_NAME.items()}


class DeviceOutputArtifactError(ValueError):
    """Raised when an artifact or capture request violates its contract."""


class DeviceOutputMismatchError(AssertionError):
    """Raised when finite device rows violate fixed numerical thresholds."""


def _require_identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise DeviceOutputArtifactError(f"{field} must match {_IDENTIFIER_RE.pattern!r}, got {value!r}")
    return value


def _require_plain_int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DeviceOutputArtifactError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DeviceOutputArtifactError(f"{field} must be a lowercase SHA-256 digest, got {value!r}")
    return value


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    field: str,
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DeviceOutputArtifactError(f"{field} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise DeviceOutputArtifactError(f"{field} keys mismatch: missing={missing}, extra={extra}")
    return value


@dataclass(frozen=True)
class ArtifactMetadata:
    """Identity shared by both placements, plus the producing placement."""

    case: str
    seed: int
    topology: Mapping[str, int]
    placement: str
    placement_manifest_sha256: str
    program_source_sha256: str

    def __post_init__(self) -> None:
        _require_identifier(self.case, field="case")
        _require_plain_int(self.seed, field="seed")
        if not isinstance(self.topology, Mapping) or not self.topology:
            raise DeviceOutputArtifactError("topology must be a non-empty mapping")
        canonical_topology: dict[str, int] = {}
        for key, value in self.topology.items():
            name = _require_identifier(key, field="topology key")
            canonical_topology[name] = _require_plain_int(
                value,
                field=f"topology[{name!r}]",
            )
        if len(canonical_topology) != len(self.topology):
            raise DeviceOutputArtifactError("topology contains duplicate canonical keys")
        if self.placement not in PLACEMENTS:
            raise DeviceOutputArtifactError(
                f"placement must be one of {sorted(PLACEMENTS)}, got {self.placement!r}"
            )
        _require_sha256(
            self.placement_manifest_sha256,
            field="placement_manifest_sha256",
        )
        _require_sha256(
            self.program_source_sha256,
            field="program_source_sha256",
        )
        object.__setattr__(
            self,
            "topology",
            MappingProxyType(dict(sorted(canonical_topology.items()))),
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON representation."""
        return {
            "case": self.case,
            "seed": self.seed,
            "topology": dict(self.topology),
            "placement": self.placement,
            "placement_manifest_sha256": self.placement_manifest_sha256,
            "program_source_sha256": self.program_source_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactMetadata":
        """Parse metadata while rejecting missing and unknown fields."""
        obj = _require_exact_keys(
            value,
            {
                "case",
                "seed",
                "topology",
                "placement",
                "placement_manifest_sha256",
                "program_source_sha256",
            },
            field="metadata",
        )
        topology = obj["topology"]
        if not isinstance(topology, dict):
            raise DeviceOutputArtifactError("metadata.topology must be a JSON object")
        return cls(
            case=obj["case"],  # type: ignore[arg-type]
            seed=obj["seed"],  # type: ignore[arg-type]
            topology=topology,  # type: ignore[arg-type]
            placement=obj["placement"],  # type: ignore[arg-type]
            placement_manifest_sha256=obj["placement_manifest_sha256"],  # type: ignore[arg-type]
            program_source_sha256=obj["program_source_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class RowwiseThresholds:
    """Fixed per-row numerical gates embedded in the artifact contract."""

    min_cosine: float
    max_rel_l2: float
    max_abs: float

    def __post_init__(self) -> None:
        values = {
            "min_cosine": self.min_cosine,
            "max_rel_l2": self.max_rel_l2,
            "max_abs": self.max_abs,
        }
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values.values()):
            raise DeviceOutputArtifactError(f"thresholds must be numeric, got {values}")
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise DeviceOutputArtifactError(f"thresholds must be finite, got {values}")
        if not -1.0 <= float(self.min_cosine) <= 1.0:
            raise DeviceOutputArtifactError(f"min_cosine must be in [-1, 1], got {self.min_cosine}")
        if float(self.max_rel_l2) < 0.0 or float(self.max_abs) < 0.0:
            raise DeviceOutputArtifactError("max_rel_l2 and max_abs must be non-negative")

    def to_dict(self) -> dict[str, float]:
        """Return the canonical JSON representation."""
        return {
            "min_cosine": float(self.min_cosine),
            "max_rel_l2": float(self.max_rel_l2),
            "max_abs": float(self.max_abs),
        }

    @classmethod
    def from_dict(cls, value: object) -> "RowwiseThresholds":
        """Parse thresholds while rejecting missing and unknown fields."""
        obj = _require_exact_keys(
            value,
            {"min_cosine", "max_rel_l2", "max_abs"},
            field="thresholds",
        )
        return cls(
            min_cosine=obj["min_cosine"],  # type: ignore[arg-type]
            max_rel_l2=obj["max_rel_l2"],  # type: ignore[arg-type]
            max_abs=obj["max_abs"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class DenseOutputCapture:
    """Capture a dense output using its leading axes as logical row keys."""

    logical_key: str
    source_name: str
    row_axes: int
    expected_rows: int
    thresholds: RowwiseThresholds

    def __post_init__(self) -> None:
        _require_identifier(self.logical_key, field="logical_key")
        _require_identifier(self.source_name, field="source_name")
        _require_plain_int(self.row_axes, field="row_axes", minimum=1)
        _require_plain_int(self.expected_rows, field="expected_rows")
        if not isinstance(self.thresholds, RowwiseThresholds):
            raise DeviceOutputArtifactError("thresholds must be RowwiseThresholds")


@dataclass(frozen=True)
class MappedPoolCapture:
    """Capture active rows from a rank/layer-stacked paged cache pool."""

    logical_key: str
    source_name: str
    block_size: int
    layer_ids: tuple[int, ...]
    mapping_names: tuple[str | None, ...]
    expected_rows: int
    thresholds: RowwiseThresholds

    def __post_init__(self) -> None:
        _require_identifier(self.logical_key, field="logical_key")
        _require_identifier(self.source_name, field="source_name")
        _require_plain_int(self.block_size, field="block_size", minimum=1)
        _require_plain_int(self.expected_rows, field="expected_rows")
        if not isinstance(self.thresholds, RowwiseThresholds):
            raise DeviceOutputArtifactError("thresholds must be RowwiseThresholds")
        if not isinstance(self.layer_ids, tuple) or not self.layer_ids:
            raise DeviceOutputArtifactError("layer_ids must be a non-empty tuple")
        if not isinstance(self.mapping_names, tuple):
            raise DeviceOutputArtifactError("mapping_names must be a tuple")
        if len(self.layer_ids) != len(self.mapping_names):
            raise DeviceOutputArtifactError("layer_ids and mapping_names must have the same length")
        previous = -1
        for index, layer_id in enumerate(self.layer_ids):
            _require_plain_int(layer_id, field=f"layer_ids[{index}]")
            if layer_id <= previous:
                raise DeviceOutputArtifactError("layer_ids must be unique and strictly increasing")
            previous = layer_id
        for index, mapping_name in enumerate(self.mapping_names):
            if mapping_name is not None:
                _require_identifier(
                    mapping_name,
                    field=f"mapping_names[{index}]",
                )
        if all(name is None for name in self.mapping_names) and self.expected_rows != 0:
            raise DeviceOutputArtifactError("a mapped capture with no mappings must expect zero rows")


CaptureSpec: TypeAlias = DenseOutputCapture | MappedPoolCapture


@dataclass(frozen=True)
class ArtifactSummary:
    """Summary returned after an artifact has been written and verified."""

    path: Path
    placement: str
    row_counts: Mapping[str, int]


@dataclass(frozen=True)
class RowComparison:
    """Metrics and threshold status for one logical row."""

    row_key: tuple[int, ...]
    cosine: float | None
    rel_l2: float | None
    max_abs: float | None
    passed: bool
    detail: str


@dataclass(frozen=True)
class TensorComparison:
    """All rowwise metrics for one logical tensor."""

    logical_key: str
    thresholds: RowwiseThresholds
    rows: tuple[RowComparison, ...]

    @property
    def passed(self) -> bool:
        return all(row.passed for row in self.rows)

    @property
    def min_cosine(self) -> float | None:
        values = [row.cosine for row in self.rows if row.cosine is not None]
        return min(values) if values else None

    @property
    def max_rel_l2(self) -> float | None:
        values = [row.rel_l2 for row in self.rows if row.rel_l2 is not None]
        return max(values) if values else None

    @property
    def max_abs(self) -> float | None:
        values = [row.max_abs for row in self.rows if row.max_abs is not None]
        return max(values) if values else None


@dataclass(frozen=True)
class ComparisonResult:
    """Placement-aware result returned by the A/B comparer."""

    reference_placement: str
    candidate_placement: str
    tensors: tuple[TensorComparison, ...]

    @property
    def passed(self) -> bool:
        return all(tensor.passed for tensor in self.tensors)

    def format_report(self, *, max_failure_rows: int = 16) -> str:
        """Format stable summary lines plus a bounded set of row failures."""
        _require_plain_int(
            max_failure_rows,
            field="max_failure_rows",
        )
        lines = [
            "[DEVICE OUTPUT A/B] "
            f"reference={self.reference_placement} "
            f"candidate={self.candidate_placement} "
            f"status={'PASS' if self.passed else 'FAIL'}"
        ]
        failures: list[str] = []
        for tensor in self.tensors:
            lines.append(
                "[DEVICE OUTPUT A/B] "
                f"key={tensor.logical_key} "
                f"rows={len(tensor.rows)} "
                f"status={'PASS' if tensor.passed else 'FAIL'} "
                f"min_cosine={_format_metric(tensor.min_cosine)} "
                f"max_rel_l2={_format_metric(tensor.max_rel_l2)} "
                f"max_abs={_format_metric(tensor.max_abs)}"
            )
            for row in tensor.rows:
                if not row.passed:
                    failures.append(
                        "[DEVICE OUTPUT A/B] "
                        f"key={tensor.logical_key} row={row.row_key} "
                        f"cosine={_format_metric(row.cosine)} "
                        f"rel_l2={_format_metric(row.rel_l2)} "
                        f"max_abs={_format_metric(row.max_abs)} "
                        f"detail={row.detail}"
                    )
        lines.extend(failures[:max_failure_rows])
        if len(failures) > max_failure_rows:
            lines.append(f"[DEVICE OUTPUT A/B] omitted_failure_rows={len(failures) - max_failure_rows}")
        return "\n".join(lines)

    def require_pass(self) -> None:
        """Raise an assertion-style exception when any row violates the gate."""
        if not self.passed:
            raise DeviceOutputMismatchError(self.format_report())


@dataclass(frozen=True)
class _CapturedTensor:
    logical_key: str
    kind: str
    source_name: str
    capture_contract: Mapping[str, object]
    thresholds: RowwiseThresholds
    row_key_fields: tuple[str, ...]
    row_keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class _LoadedArtifact:
    metadata: ArtifactMetadata
    tensors: Mapping[str, _CapturedTensor]


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the lowercase SHA-256 digest of a regular, non-symlink file."""
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise DeviceOutputArtifactError(f"digest source must be a regular non-symlink file: {target}")
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_metric(value: float | None) -> str:
    return "NA" if value is None else f"{value:.9g}"


def _canonical_json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise DeviceOutputArtifactError(f"artifact metadata is not canonical JSON: {error}") from error
    return (text + "\n").encode("utf-8")


def _json_no_duplicates(data: bytes) -> object:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DeviceOutputArtifactError(f"manifest contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=reject_duplicates)
    except DeviceOutputArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DeviceOutputArtifactError(f"invalid artifact manifest JSON: {error}") from error


def _validate_captures(captures: Sequence[CaptureSpec]) -> tuple[CaptureSpec, ...]:
    if not isinstance(captures, Sequence) or isinstance(captures, (str, bytes)):
        raise DeviceOutputArtifactError("captures must be a sequence")
    validated = tuple(captures)
    if not validated:
        raise DeviceOutputArtifactError("at least one capture is required")
    for capture in validated:
        if not isinstance(capture, (DenseOutputCapture, MappedPoolCapture)):
            raise DeviceOutputArtifactError(f"unsupported capture type: {type(capture).__name__}")
    logical_keys = [capture.logical_key for capture in validated]
    if len(set(logical_keys)) != len(logical_keys):
        raise DeviceOutputArtifactError(f"capture logical keys must be unique, got {logical_keys}")
    return tuple(sorted(validated, key=lambda capture: capture.logical_key))


def _require_tensor(
    tensors: Mapping[str, torch.Tensor],
    name: str,
    *,
    field: str,
) -> torch.Tensor:
    if name not in tensors:
        raise DeviceOutputArtifactError(f"missing {field} tensor {name!r}")
    tensor = tensors[name]
    if not isinstance(tensor, torch.Tensor):
        raise DeviceOutputArtifactError(
            f"{field} {name!r} must be a torch.Tensor, got {type(tensor).__name__}"
        )
    if tensor.layout != torch.strided:
        raise DeviceOutputArtifactError(f"{field} {name!r} must use strided layout, got {tensor.layout}")
    if tensor.dtype not in _NAME_BY_DTYPE:
        raise DeviceOutputArtifactError(f"{field} {name!r} has unsupported dtype {tensor.dtype}")
    return tensor


def _resolve_mapping(
    name: str,
    device_outputs: Mapping[str, torch.Tensor],
    validation_inputs: Mapping[str, torch.Tensor],
    static_tensors: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    if name in validation_inputs:
        return _require_tensor(validation_inputs, name, field="mapping")
    if name in device_outputs:
        return _require_tensor(device_outputs, name, field="mapping")
    return _require_tensor(static_tensors, name, field="mapping")


def _logical_row_keys(row_shape: tuple[int, ...]) -> torch.Tensor:
    rows = math.prod(row_shape)
    keys = torch.empty((rows, len(row_shape)), dtype=torch.int64)
    flat = torch.arange(rows, dtype=torch.int64)
    for axis, dim in enumerate(row_shape):
        stride = math.prod(row_shape[axis + 1 :])
        keys[:, axis] = (flat // stride) % dim
    return keys


def _capture_contract(capture: CaptureSpec) -> dict[str, object]:
    common = {"expected_rows": capture.expected_rows}
    if isinstance(capture, DenseOutputCapture):
        return {**common, "row_axes": capture.row_axes}
    return {
        **common,
        "block_size": capture.block_size,
        "layer_ids": list(capture.layer_ids),
        "mapping_names": list(capture.mapping_names),
    }


def _capture_dense(
    capture: DenseOutputCapture,
    device_outputs: Mapping[str, torch.Tensor],
) -> _CapturedTensor:
    source = _require_tensor(
        device_outputs,
        capture.source_name,
        field="device output",
    )
    if capture.row_axes > source.ndim:
        raise DeviceOutputArtifactError(
            f"{capture.logical_key!r} row_axes={capture.row_axes} exceeds source rank {source.ndim}"
        )
    row_shape = tuple(int(dim) for dim in source.shape[: capture.row_axes])
    row_count = math.prod(row_shape)
    if row_count != capture.expected_rows:
        raise DeviceOutputArtifactError(
            f"{capture.logical_key!r} row coverage {row_count} does not match "
            f"expected {capture.expected_rows}"
        )
    value_shape = (row_count, *source.shape[capture.row_axes :])
    values = source.detach().reshape(value_shape).cpu().contiguous()
    return _CapturedTensor(
        logical_key=capture.logical_key,
        kind="dense",
        source_name=capture.source_name,
        capture_contract=_capture_contract(capture),
        thresholds=capture.thresholds,
        row_key_fields=tuple(f"dim{axis}" for axis in range(capture.row_axes)),
        row_keys=_logical_row_keys(row_shape),
        values=values,
    )


def _capture_mapped_pool(
    capture: MappedPoolCapture,
    device_outputs: Mapping[str, torch.Tensor],
    validation_inputs: Mapping[str, torch.Tensor],
    static_tensors: Mapping[str, torch.Tensor],
) -> _CapturedTensor:
    pool = _require_tensor(
        device_outputs,
        capture.source_name,
        field="device output",
    )
    if pool.ndim < 3:
        raise DeviceOutputArtifactError(
            f"mapped pool {capture.source_name!r} must have shape [rank, stacked_blocks, block_size, ...]"
        )
    rank_count = int(pool.shape[0])
    layer_count = len(capture.layer_ids)
    if rank_count <= 0:
        raise DeviceOutputArtifactError("mapped pool rank axis must be non-empty")
    if pool.shape[1] % layer_count != 0:
        raise DeviceOutputArtifactError(
            f"mapped pool {capture.source_name!r} stacked-block axis "
            f"{pool.shape[1]} is not divisible by {layer_count} layers"
        )
    if int(pool.shape[2]) != capture.block_size:
        raise DeviceOutputArtifactError(
            f"mapped pool {capture.source_name!r} block size {pool.shape[2]} "
            f"does not match capture block size {capture.block_size}"
        )
    blocks_per_layer = int(pool.shape[1]) // layer_count
    rows_per_layer = blocks_per_layer * capture.block_size

    mapping_cache: dict[str, torch.Tensor] = {}
    row_keys: list[tuple[int, int, int]] = []
    source_indices: list[tuple[int, int, int]] = []
    for layer_index, (layer_id, mapping_name) in enumerate(
        zip(capture.layer_ids, capture.mapping_names, strict=True)
    ):
        if mapping_name is None:
            continue
        if mapping_name not in mapping_cache:
            mapping = _resolve_mapping(
                mapping_name,
                device_outputs,
                validation_inputs,
                static_tensors,
            )
            if mapping.dtype not in _INTEGER_DTYPES:
                raise DeviceOutputArtifactError(
                    f"mapping {mapping_name!r} must have an integer dtype, got {mapping.dtype}"
                )
            if mapping.ndim != 2 or int(mapping.shape[0]) != rank_count:
                raise DeviceOutputArtifactError(
                    f"mapping {mapping_name!r} must have shape "
                    f"[{rank_count}, tokens], got {tuple(mapping.shape)}"
                )
            mapping_cache[mapping_name] = mapping.detach().cpu().to(torch.int64)
        mapping_i64 = mapping_cache[mapping_name]
        for rank in range(rank_count):
            rank_mapping = mapping_i64[rank]
            invalid_negative = rank_mapping < -1
            if bool(invalid_negative.any()):
                token = int(invalid_negative.nonzero(as_tuple=False)[0].item())
                raise DeviceOutputArtifactError(
                    f"mapping {mapping_name!r}[{rank}, {token}] is "
                    f"{int(rank_mapping[token])}; only -1 is a negative sentinel"
                )
            active_tokens = (rank_mapping >= 0).nonzero(as_tuple=False).flatten()
            active_rows = rank_mapping[active_tokens]
            if bool((active_rows >= rows_per_layer).any()):
                token_index = int((active_rows >= rows_per_layer).nonzero(as_tuple=False)[0].item())
                token = int(active_tokens[token_index])
                raise DeviceOutputArtifactError(
                    f"mapping {mapping_name!r}[{rank}, {token}] is out of range "
                    f"for {rows_per_layer} rows per layer"
                )
            if active_rows.numel() != torch.unique(active_rows).numel():
                raise DeviceOutputArtifactError(
                    f"mapping {mapping_name!r} has duplicate active rows for rank={rank}, layer={layer_id}"
                )
            for token, physical_row in zip(
                active_tokens.tolist(),
                active_rows.tolist(),
                strict=True,
            ):
                row_keys.append((rank, layer_id, int(token)))
                source_indices.append(
                    (
                        rank,
                        layer_index * blocks_per_layer + int(physical_row) // capture.block_size,
                        int(physical_row) % capture.block_size,
                    )
                )

    order = sorted(range(len(row_keys)), key=row_keys.__getitem__)
    row_keys = [row_keys[index] for index in order]
    source_indices = [source_indices[index] for index in order]
    if len(row_keys) != capture.expected_rows:
        raise DeviceOutputArtifactError(
            f"{capture.logical_key!r} active-row coverage {len(row_keys)} "
            f"does not match expected {capture.expected_rows}"
        )
    if len(set(row_keys)) != len(row_keys):
        raise DeviceOutputArtifactError(f"{capture.logical_key!r} produced duplicate logical row keys")

    if source_indices:
        ranks, blocks, intras = zip(*source_indices, strict=True)
        device = pool.device
        selected = pool[
            torch.tensor(ranks, dtype=torch.int64, device=device),
            torch.tensor(blocks, dtype=torch.int64, device=device),
            torch.tensor(intras, dtype=torch.int64, device=device),
        ]
        values = selected.detach().cpu().contiguous()
    else:
        values = torch.empty(
            (0, *pool.shape[3:]),
            dtype=pool.dtype,
            device="cpu",
        )
    keys = torch.tensor(row_keys, dtype=torch.int64).reshape(-1, 3)
    return _CapturedTensor(
        logical_key=capture.logical_key,
        kind="mapped_pool",
        source_name=capture.source_name,
        capture_contract=_capture_contract(capture),
        thresholds=capture.thresholds,
        row_key_fields=("rank", "layer", "token"),
        row_keys=keys,
        values=values,
    )


def _capture_tensors(
    captures: Sequence[CaptureSpec],
    device_outputs: Mapping[str, torch.Tensor],
    validation_inputs: Mapping[str, torch.Tensor] | None,
    static_tensors: Mapping[str, torch.Tensor] | None,
) -> tuple[_CapturedTensor, ...]:
    if not isinstance(device_outputs, Mapping):
        raise DeviceOutputArtifactError("device_outputs must be a tensor mapping")
    validation = {} if validation_inputs is None else validation_inputs
    static = {} if static_tensors is None else static_tensors
    if not isinstance(validation, Mapping):
        raise DeviceOutputArtifactError("validation_inputs must be a tensor mapping")
    if not isinstance(static, Mapping):
        raise DeviceOutputArtifactError("static_tensors must be a tensor mapping")
    results = []
    for capture in _validate_captures(captures):
        if isinstance(capture, DenseOutputCapture):
            result = _capture_dense(capture, device_outputs)
        else:
            result = _capture_mapped_pool(
                capture,
                device_outputs,
                validation,
                static,
            )
        results.append(result)
    return tuple(results)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    if sys.byteorder != "little":
        raise DeviceOutputArtifactError("device-output artifacts currently require a little-endian host")
    cpu = tensor.detach().cpu().contiguous()
    return cpu.view(torch.uint8).numpy().tobytes(order="C")


def _write_payload(path: Path, tensor: torch.Tensor) -> str:
    data = _tensor_bytes(tensor)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _payload_manifest(path: str, digest: str, tensor: torch.Tensor) -> dict[str, object]:
    return {
        "file": path,
        "sha256": digest,
        "dtype": _NAME_BY_DTYPE[tensor.dtype],
        "shape": list(tensor.shape),
    }


def _ensure_finite_baseline(tensor: _CapturedTensor) -> None:
    finite = torch.isfinite(tensor.values)
    if not bool(finite.all()):
        bad = (~finite).nonzero(as_tuple=False)[0]
        row = int(bad[0]) if bad.numel() else -1
        row_key = tuple(int(value) for value in tensor.row_keys[row].tolist())
        raise DeviceOutputArtifactError(f"baseline {tensor.logical_key!r} row {row_key} contains NaN or Inf")


def write_device_output_artifact(
    path: str | os.PathLike[str],
    *,
    metadata: ArtifactMetadata,
    captures: Sequence[CaptureSpec],
    device_outputs: Mapping[str, torch.Tensor],
    validation_inputs: Mapping[str, torch.Tensor] | None = None,
    static_tensors: Mapping[str, torch.Tensor] | None = None,
) -> ArtifactSummary:
    """Write one immutable reference-placement artifact directory."""
    if not isinstance(metadata, ArtifactMetadata):
        raise DeviceOutputArtifactError("metadata must be ArtifactMetadata")
    target = Path(path)
    if target.exists() or target.is_symlink():
        raise DeviceOutputArtifactError(f"artifact path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    captured = _capture_tensors(
        captures,
        device_outputs,
        validation_inputs,
        static_tensors,
    )
    for tensor in captured:
        _ensure_finite_baseline(tensor)

    temp = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.tmp-",
            dir=target.parent,
        )
    )
    try:
        tensor_entries = []
        for index, tensor in enumerate(captured):
            keys_name = f"tensor-{index:04d}-row-keys.bin"
            values_name = f"tensor-{index:04d}-values.bin"
            keys_digest = _write_payload(temp / keys_name, tensor.row_keys)
            values_digest = _write_payload(temp / values_name, tensor.values)
            tensor_entries.append(
                {
                    "logical_key": tensor.logical_key,
                    "kind": tensor.kind,
                    "source_name": tensor.source_name,
                    "capture": dict(tensor.capture_contract),
                    "thresholds": tensor.thresholds.to_dict(),
                    "row_key_fields": list(tensor.row_key_fields),
                    "row_count": int(tensor.row_keys.shape[0]),
                    "row_keys": _payload_manifest(
                        keys_name,
                        keys_digest,
                        tensor.row_keys,
                    ),
                    "values": _payload_manifest(
                        values_name,
                        values_digest,
                        tensor.values,
                    ),
                }
            )
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "byte_order": "little",
            "metadata": metadata.to_dict(),
            "logical_keys": [tensor.logical_key for tensor in captured],
            "tensors": tensor_entries,
        }
        manifest_bytes = _canonical_json_bytes(manifest)
        (temp / _MANIFEST_NAME).write_bytes(manifest_bytes)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        (temp / _MANIFEST_DIGEST_NAME).write_text(
            manifest_digest + "\n",
            encoding="ascii",
        )
        _load_artifact(temp, captures=captures, expected_metadata=metadata)
        os.rename(temp, target)
    finally:
        if temp.exists():
            shutil.rmtree(temp)

    return ArtifactSummary(
        path=target,
        placement=metadata.placement,
        row_counts=MappingProxyType(
            {tensor.logical_key: int(tensor.row_keys.shape[0]) for tensor in captured}
        ),
    )


def _parse_shape(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise DeviceOutputArtifactError(f"{field} must be a JSON array")
    shape = tuple(_require_plain_int(dim, field=f"{field}[{index}]") for index, dim in enumerate(value))
    return shape


def _load_payload(
    artifact_dir: Path,
    value: object,
    *,
    field: str,
) -> tuple[torch.Tensor, str]:
    obj = _require_exact_keys(
        value,
        {"file", "sha256", "dtype", "shape"},
        field=field,
    )
    file_name = obj["file"]
    if not isinstance(file_name, str) or _PAYLOAD_RE.fullmatch(file_name) is None:
        raise DeviceOutputArtifactError(f"{field}.file is not a canonical payload name: {file_name!r}")
    digest = _require_sha256(obj["sha256"], field=f"{field}.sha256")
    dtype_name = obj["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _DTYPE_BY_NAME:
        raise DeviceOutputArtifactError(f"{field}.dtype is unsupported: {dtype_name!r}")
    shape = _parse_shape(obj["shape"], field=f"{field}.shape")
    payload = artifact_dir / file_name
    if payload.is_symlink() or not payload.is_file():
        raise DeviceOutputArtifactError(f"artifact payload must be a regular non-symlink file: {file_name}")
    data = payload.read_bytes()
    actual_digest = hashlib.sha256(data).hexdigest()
    if actual_digest != digest:
        raise DeviceOutputArtifactError(
            f"artifact payload digest mismatch for {file_name}: expected {digest}, got {actual_digest}"
        )
    dtype = _DTYPE_BY_NAME[dtype_name]
    expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(data) != expected_bytes:
        raise DeviceOutputArtifactError(
            f"artifact payload byte size mismatch for {file_name}: expected {expected_bytes}, got {len(data)}"
        )
    if expected_bytes == 0:
        tensor = torch.empty(shape, dtype=dtype)
    else:
        tensor = torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)
    return tensor, file_name


def _parse_capture_contract(kind: str, value: object) -> dict[str, object]:
    if kind == "dense":
        obj = _require_exact_keys(
            value,
            {"expected_rows", "row_axes"},
            field="tensor.capture",
        )
        return {
            "expected_rows": _require_plain_int(
                obj["expected_rows"],
                field="tensor.capture.expected_rows",
            ),
            "row_axes": _require_plain_int(
                obj["row_axes"],
                field="tensor.capture.row_axes",
                minimum=1,
            ),
        }
    if kind != "mapped_pool":
        raise DeviceOutputArtifactError(f"unsupported artifact tensor kind {kind!r}")
    obj = _require_exact_keys(
        value,
        {"expected_rows", "block_size", "layer_ids", "mapping_names"},
        field="tensor.capture",
    )
    layer_ids_value = obj["layer_ids"]
    mapping_names_value = obj["mapping_names"]
    if not isinstance(layer_ids_value, list) or not isinstance(mapping_names_value, list):
        raise DeviceOutputArtifactError("mapped tensor layer_ids and mapping_names must be JSON arrays")
    layer_ids = [
        _require_plain_int(layer_id, field=f"tensor.capture.layer_ids[{index}]")
        for index, layer_id in enumerate(layer_ids_value)
    ]
    mapping_names = []
    for index, mapping_name in enumerate(mapping_names_value):
        if mapping_name is not None:
            _require_identifier(
                mapping_name,
                field=f"tensor.capture.mapping_names[{index}]",
            )
        mapping_names.append(mapping_name)
    if len(layer_ids) != len(mapping_names):
        raise DeviceOutputArtifactError("mapped tensor layer_ids and mapping_names lengths differ")
    return {
        "expected_rows": _require_plain_int(
            obj["expected_rows"],
            field="tensor.capture.expected_rows",
        ),
        "block_size": _require_plain_int(
            obj["block_size"],
            field="tensor.capture.block_size",
            minimum=1,
        ),
        "layer_ids": layer_ids,
        "mapping_names": mapping_names,
    }


def _rows_are_canonical(keys: torch.Tensor) -> bool:
    rows = [tuple(int(value) for value in row.tolist()) for row in keys]
    return rows == sorted(rows) and len(rows) == len(set(rows))


def _load_artifact(
    path: str | os.PathLike[str],
    *,
    captures: Sequence[CaptureSpec],
    expected_metadata: ArtifactMetadata | None = None,
) -> _LoadedArtifact:
    artifact_dir = Path(path)
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise DeviceOutputArtifactError(f"artifact must be a regular non-symlink directory: {artifact_dir}")
    manifest_path = artifact_dir / _MANIFEST_NAME
    digest_path = artifact_dir / _MANIFEST_DIGEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise DeviceOutputArtifactError("artifact manifest.json is missing or not regular")
    if digest_path.is_symlink() or not digest_path.is_file():
        raise DeviceOutputArtifactError("artifact manifest.sha256 is missing or not regular")
    manifest_bytes = manifest_path.read_bytes()
    if len(manifest_bytes) > 1024 * 1024:
        raise DeviceOutputArtifactError("artifact manifest exceeds 1 MiB")
    digest_text = digest_path.read_text(encoding="ascii")
    expected_digest = digest_text.removesuffix("\n")
    _require_sha256(expected_digest, field="manifest.sha256")
    if digest_text != expected_digest + "\n":
        raise DeviceOutputArtifactError("manifest.sha256 is not canonical")
    actual_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_digest != expected_digest:
        raise DeviceOutputArtifactError(
            f"artifact manifest digest mismatch: expected {expected_digest}, got {actual_digest}"
        )
    manifest = _require_exact_keys(
        _json_no_duplicates(manifest_bytes),
        {"format", "version", "byte_order", "metadata", "logical_keys", "tensors"},
        field="manifest",
    )
    if manifest["format"] != ARTIFACT_FORMAT:
        raise DeviceOutputArtifactError(
            f"artifact format must be {ARTIFACT_FORMAT!r}, got {manifest['format']!r}"
        )
    if manifest["version"] != ARTIFACT_VERSION:
        raise DeviceOutputArtifactError(
            f"artifact version must be {ARTIFACT_VERSION}, got {manifest['version']!r}"
        )
    if manifest["byte_order"] != "little" or sys.byteorder != "little":
        raise DeviceOutputArtifactError("artifact byte order must be little-endian")
    metadata = ArtifactMetadata.from_dict(manifest["metadata"])
    if expected_metadata is not None and metadata != expected_metadata:
        raise DeviceOutputArtifactError("artifact metadata does not match the writer metadata")

    capture_specs = _validate_captures(captures)
    expected_logical_keys = [capture.logical_key for capture in capture_specs]
    logical_keys = manifest["logical_keys"]
    if logical_keys != expected_logical_keys:
        raise DeviceOutputArtifactError(
            f"artifact logical keys mismatch: expected {expected_logical_keys}, got {logical_keys!r}"
        )
    entries = manifest["tensors"]
    if not isinstance(entries, list) or len(entries) != len(capture_specs):
        raise DeviceOutputArtifactError("artifact tensor entry count does not match logical keys")

    loaded: dict[str, _CapturedTensor] = {}
    payload_files: set[str] = set()
    for index, (raw_entry, capture) in enumerate(zip(entries, capture_specs, strict=True)):
        entry = _require_exact_keys(
            raw_entry,
            {
                "logical_key",
                "kind",
                "source_name",
                "capture",
                "thresholds",
                "row_key_fields",
                "row_count",
                "row_keys",
                "values",
            },
            field=f"tensors[{index}]",
        )
        expected_kind = "dense" if isinstance(capture, DenseOutputCapture) else "mapped_pool"
        if entry["logical_key"] != capture.logical_key:
            raise DeviceOutputArtifactError(f"tensor entry {index} logical key mismatch")
        if entry["kind"] != expected_kind:
            raise DeviceOutputArtifactError(f"tensor {capture.logical_key!r} kind mismatch")
        if entry["source_name"] != capture.source_name:
            raise DeviceOutputArtifactError(f"tensor {capture.logical_key!r} source name mismatch")
        parsed_contract = _parse_capture_contract(expected_kind, entry["capture"])
        if parsed_contract != _capture_contract(capture):
            raise DeviceOutputArtifactError(f"tensor {capture.logical_key!r} capture contract mismatch")
        thresholds = RowwiseThresholds.from_dict(entry["thresholds"])
        if thresholds != capture.thresholds:
            raise DeviceOutputArtifactError(f"tensor {capture.logical_key!r} threshold contract mismatch")
        expected_fields = (
            tuple(f"dim{axis}" for axis in range(capture.row_axes))
            if isinstance(capture, DenseOutputCapture)
            else ("rank", "layer", "token")
        )
        if entry["row_key_fields"] != list(expected_fields):
            raise DeviceOutputArtifactError(f"tensor {capture.logical_key!r} row-key fields mismatch")
        row_count = _require_plain_int(
            entry["row_count"],
            field=f"tensor {capture.logical_key!r} row_count",
        )
        if row_count != capture.expected_rows:
            raise DeviceOutputArtifactError(
                f"tensor {capture.logical_key!r} artifact row coverage "
                f"{row_count} does not match expected {capture.expected_rows}"
            )
        row_keys, keys_file = _load_payload(
            artifact_dir,
            entry["row_keys"],
            field=f"tensor {capture.logical_key!r}.row_keys",
        )
        values, values_file = _load_payload(
            artifact_dir,
            entry["values"],
            field=f"tensor {capture.logical_key!r}.values",
        )
        if keys_file in payload_files or values_file in payload_files or keys_file == values_file:
            raise DeviceOutputArtifactError("artifact payload files must be unique")
        payload_files.update((keys_file, values_file))
        if row_keys.dtype != torch.int64 or row_keys.shape != (row_count, len(expected_fields)):
            raise DeviceOutputArtifactError(
                f"tensor {capture.logical_key!r} row-key tensor shape/dtype mismatch"
            )
        if values.ndim < 1 or int(values.shape[0]) != row_count:
            raise DeviceOutputArtifactError(f"tensor {capture.logical_key!r} value tensor row count mismatch")
        if math.prod(values.shape[1:]) == 0:
            raise DeviceOutputArtifactError(f"tensor {capture.logical_key!r} rows must not be empty")
        if not _rows_are_canonical(row_keys):
            raise DeviceOutputArtifactError(
                f"tensor {capture.logical_key!r} row keys are not canonical and unique"
            )
        tensor = _CapturedTensor(
            logical_key=capture.logical_key,
            kind=expected_kind,
            source_name=capture.source_name,
            capture_contract=parsed_contract,
            thresholds=thresholds,
            row_key_fields=expected_fields,
            row_keys=row_keys,
            values=values,
        )
        _ensure_finite_baseline(tensor)
        loaded[tensor.logical_key] = tensor

    expected_files = payload_files | {_MANIFEST_NAME, _MANIFEST_DIGEST_NAME}
    actual_files = set()
    for child in artifact_dir.iterdir():
        if child.is_symlink() or not child.is_file():
            raise DeviceOutputArtifactError(f"artifact contains a non-regular entry: {child.name}")
        actual_files.add(child.name)
    if actual_files != expected_files:
        raise DeviceOutputArtifactError(
            "artifact file set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )
    return _LoadedArtifact(
        metadata=metadata,
        tensors=MappingProxyType(loaded),
    )


def _require_metadata_compatible(
    reference: ArtifactMetadata,
    candidate: ArtifactMetadata,
) -> None:
    fields = (
        "case",
        "seed",
        "topology",
        "placement_manifest_sha256",
        "program_source_sha256",
    )
    mismatches = [
        f"{field}: reference={getattr(reference, field)!r}, candidate={getattr(candidate, field)!r}"
        for field in fields
        if getattr(reference, field) != getattr(candidate, field)
    ]
    if mismatches:
        raise DeviceOutputArtifactError(
            "A/B artifact metadata mismatch (placement is the only field allowed "
            "to differ): " + "; ".join(mismatches)
        )


def _compare_rows(
    actual: _CapturedTensor,
    expected: _CapturedTensor,
) -> TensorComparison:
    if actual.kind != expected.kind:
        raise DeviceOutputArtifactError(f"logical tensor {actual.logical_key!r} kind changed")
    if actual.source_name != expected.source_name:
        raise DeviceOutputArtifactError(f"logical tensor {actual.logical_key!r} source changed")
    if dict(actual.capture_contract) != dict(expected.capture_contract):
        raise DeviceOutputArtifactError(f"logical tensor {actual.logical_key!r} capture contract changed")
    if actual.thresholds != expected.thresholds:
        raise DeviceOutputArtifactError(f"logical tensor {actual.logical_key!r} thresholds changed")
    if actual.row_key_fields != expected.row_key_fields:
        raise DeviceOutputArtifactError(f"logical tensor {actual.logical_key!r} row-key fields changed")
    if not torch.equal(actual.row_keys, expected.row_keys):
        raise DeviceOutputArtifactError(f"logical tensor {actual.logical_key!r} row keys changed")
    if actual.values.shape != expected.values.shape:
        raise DeviceOutputArtifactError(
            f"logical tensor {actual.logical_key!r} shape changed: "
            f"reference={tuple(expected.values.shape)}, "
            f"candidate={tuple(actual.values.shape)}"
        )
    if actual.values.dtype != expected.values.dtype:
        raise DeviceOutputArtifactError(
            f"logical tensor {actual.logical_key!r} dtype changed: "
            f"reference={expected.values.dtype}, candidate={actual.values.dtype}"
        )

    actual_rows = actual.values.reshape(actual.values.shape[0], -1).to(torch.float64)
    expected_rows = expected.values.reshape(expected.values.shape[0], -1).to(torch.float64)
    rows = []
    thresholds = expected.thresholds
    for index in range(actual_rows.shape[0]):
        row_key = tuple(int(value) for value in actual.row_keys[index].tolist())
        actual_row = actual_rows[index]
        expected_row = expected_rows[index]
        if not bool(torch.isfinite(actual_row).all()):
            rows.append(
                RowComparison(
                    row_key=row_key,
                    cosine=None,
                    rel_l2=None,
                    max_abs=None,
                    passed=False,
                    detail="candidate row contains NaN or Inf",
                )
            )
            continue
        if not bool(torch.isfinite(expected_row).all()):
            raise DeviceOutputArtifactError(
                f"reference {actual.logical_key!r} row {row_key} contains NaN or Inf"
            )
        actual_norm = actual_row.norm()
        expected_norm = expected_row.norm()
        denominator = float(actual_norm * expected_norm)
        if denominator > 0.0:
            cosine = float(actual_row @ expected_row) / denominator
            cosine = max(-1.0, min(1.0, cosine))
        elif float(actual_norm) == 0.0 and float(expected_norm) == 0.0:
            cosine = 1.0
        else:
            cosine = 0.0
        difference = actual_row - expected_row
        rel_l2 = float(difference.norm() / expected_norm.clamp_min(1e-12))
        max_abs = float(difference.abs().max())
        if not all(math.isfinite(value) for value in (cosine, rel_l2, max_abs)):
            rows.append(
                RowComparison(
                    row_key=row_key,
                    cosine=None,
                    rel_l2=None,
                    max_abs=None,
                    passed=False,
                    detail="derived row metric is NaN or Inf",
                )
            )
            continue
        violations = []
        if cosine < thresholds.min_cosine:
            violations.append(f"cosine {cosine:.9g} < {thresholds.min_cosine:.9g}")
        if rel_l2 > thresholds.max_rel_l2:
            violations.append(f"rel_l2 {rel_l2:.9g} > {thresholds.max_rel_l2:.9g}")
        if max_abs > thresholds.max_abs:
            violations.append(f"max_abs {max_abs:.9g} > {thresholds.max_abs:.9g}")
        rows.append(
            RowComparison(
                row_key=row_key,
                cosine=cosine,
                rel_l2=rel_l2,
                max_abs=max_abs,
                passed=not violations,
                detail="; ".join(violations),
            )
        )
    return TensorComparison(
        logical_key=actual.logical_key,
        thresholds=thresholds,
        rows=tuple(rows),
    )


def compare_device_output_artifact(
    path: str | os.PathLike[str],
    *,
    metadata: ArtifactMetadata,
    captures: Sequence[CaptureSpec],
    device_outputs: Mapping[str, torch.Tensor],
    validation_inputs: Mapping[str, torch.Tensor] | None = None,
    static_tensors: Mapping[str, torch.Tensor] | None = None,
) -> ComparisonResult:
    """Compare candidate-placement rows with a canonical reference artifact."""
    if not isinstance(metadata, ArtifactMetadata):
        raise DeviceOutputArtifactError("metadata must be ArtifactMetadata")
    capture_specs = _validate_captures(captures)
    artifact = _load_artifact(path, captures=capture_specs)
    _require_metadata_compatible(artifact.metadata, metadata)
    actual = _capture_tensors(
        capture_specs,
        device_outputs,
        validation_inputs,
        static_tensors,
    )
    actual_by_key = {tensor.logical_key: tensor for tensor in actual}
    expected_keys = tuple(artifact.tensors)
    actual_keys = tuple(actual_by_key)
    if actual_keys != expected_keys:
        raise DeviceOutputArtifactError(
            f"candidate logical keys mismatch: reference={expected_keys}, candidate={actual_keys}"
        )
    tensor_results = tuple(_compare_rows(actual_by_key[key], artifact.tensors[key]) for key in expected_keys)
    return ComparisonResult(
        reference_placement=artifact.metadata.placement,
        candidate_placement=metadata.placement,
        tensors=tensor_results,
    )


class DeviceOutputArtifactWriter:
    """Callback that writes a verified reference artifact exactly once."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: ArtifactMetadata,
        captures: Sequence[CaptureSpec],
        static_tensors: Mapping[str, torch.Tensor] | None = None,
        emit_report: bool = True,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.captures = _validate_captures(captures)
        self.static_tensors = static_tensors
        self.emit_report = emit_report

    def __call__(
        self,
        device_outputs: Mapping[str, torch.Tensor],
        validation_inputs: Mapping[str, torch.Tensor] | None = None,
    ) -> ArtifactSummary:
        summary = write_device_output_artifact(
            self.path,
            metadata=self.metadata,
            captures=self.captures,
            device_outputs=device_outputs,
            validation_inputs=validation_inputs,
            static_tensors=self.static_tensors,
        )
        if self.emit_report:
            rows = ",".join(f"{key}:{count}" for key, count in summary.row_counts.items())
            print(
                f"[DEVICE OUTPUT ARTIFACT] path={summary.path} placement={summary.placement} rows={rows}",
                flush=True,
            )
        return summary


class DeviceOutputArtifactComparer:
    """Callback that compares candidate outputs and fails the run by default."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        metadata: ArtifactMetadata,
        captures: Sequence[CaptureSpec],
        static_tensors: Mapping[str, torch.Tensor] | None = None,
        emit_report: bool = True,
        raise_on_mismatch: bool = True,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.captures = _validate_captures(captures)
        self.static_tensors = static_tensors
        self.emit_report = emit_report
        self.raise_on_mismatch = raise_on_mismatch

    def __call__(
        self,
        device_outputs: Mapping[str, torch.Tensor],
        validation_inputs: Mapping[str, torch.Tensor] | None = None,
    ) -> ComparisonResult:
        result = compare_device_output_artifact(
            self.path,
            metadata=self.metadata,
            captures=self.captures,
            device_outputs=device_outputs,
            validation_inputs=validation_inputs,
            static_tensors=self.static_tensors,
        )
        if self.emit_report:
            print(result.format_report(), flush=True)
        if self.raise_on_mismatch:
            result.require_pass()
        return result


__all__ = [
    "ARTIFACT_FORMAT",
    "ARTIFACT_VERSION",
    "ArtifactMetadata",
    "ArtifactSummary",
    "ComparisonResult",
    "DenseOutputCapture",
    "DeviceOutputArtifactComparer",
    "DeviceOutputArtifactError",
    "DeviceOutputArtifactWriter",
    "DeviceOutputMismatchError",
    "MappedPoolCapture",
    "RowComparison",
    "RowwiseThresholds",
    "TensorComparison",
    "compare_device_output_artifact",
    "sha256_file",
    "write_device_output_artifact",
]
