# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lazy reader for the raw ``.bin`` checkpoint layout used by DSV4.1.

The checkpoint workspace stores one tensor per ``.bin`` file and describes it
in a neighbouring ``.bin.json`` file.  ``RawCheckpoint.tensor`` uses
``torch.from_file`` so loading a layer never materializes the full model (or
the multi-gigabyte Engram table) in host memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


_DTYPES = {
    "I8": torch.int8,
    "U8": torch.uint8,
    "I32": torch.int32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
}


def _shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"metadata shape must be a list, got {type(value).__name__}")
    result = tuple(int(dim) for dim in value)
    if any(dim < 0 for dim in result):
        raise ValueError(f"metadata shape contains a negative dimension: {result}")
    return result


class RawCheckpoint:
    """Read complete tensors from a raw checkpoint without copying by default."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        self._metadata: dict[str, dict[str, Any]] = {}

    def _base_name(self, name: str) -> str:
        clean = name.removesuffix(".bin").removesuffix(".bin.json")
        if not clean:
            raise ValueError("checkpoint tensor name is empty")
        return clean

    def metadata(self, name: str) -> dict[str, Any]:
        base = self._base_name(name)
        if base not in self._metadata:
            metadata_path = self.root / f"{base}.bin.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_path}")
            payload = json.loads(metadata_path.read_text())
            if not isinstance(payload, dict):
                raise TypeError(f"checkpoint metadata must be an object: {metadata_path}")
            self._metadata[base] = payload
        return self._metadata[base]

    def tensor(self, name: str, *, clone: bool = False) -> torch.Tensor:
        base = self._base_name(name)
        metadata = self.metadata(base)
        if metadata.get("outcome") not in (None, "complete"):
            raise ValueError(f"checkpoint tensor is not complete: {base} ({metadata.get('outcome')})")
        dtype_name = str(metadata.get("dtype", "")).upper()
        if dtype_name not in _DTYPES:
            raise TypeError(f"unsupported checkpoint dtype for {base}: {dtype_name!r}")
        dtype = _DTYPES[dtype_name]
        shape = _shape(metadata.get("shape"))
        numel = 1
        for dim in shape:
            numel *= dim
        path = self.root / f"{base}.bin"
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint tensor is missing: {path}")
        expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
        actual_bytes = path.stat().st_size
        if actual_bytes < expected_bytes:
            raise ValueError(f"{base} is truncated: {actual_bytes} < {expected_bytes} bytes")
        declared_bytes = metadata.get("bytes")
        if declared_bytes is not None and int(declared_bytes) != expected_bytes:
            raise ValueError(f"{base} metadata bytes do not match shape/dtype: {declared_bytes} != {expected_bytes}")
        value = torch.from_file(str(path), shared=False, size=numel, dtype=dtype).reshape(shape)
        return value.clone() if clone else value

    def require(self, names: list[str]) -> dict[str, torch.Tensor]:
        return {name: self.tensor(name) for name in names}


def flatten_scale(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize a checkpoint ``[N, 1]`` channel scale to ``[N]``."""

    if value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim != 1:
        raise ValueError(f"{name} must be a vector or [N, 1], got {tuple(value.shape)}")
    return value


def load_attention_weights(checkpoint: RawCheckpoint, layer: int) -> dict[str, torch.Tensor]:
    """Load A2/A3 attention tensors for any migrated decoder layer.

    Entries ending in ``_raw`` retain checkpoint output-major layout.  Entries
    ending in ``_canonical`` are transposed to the ``[K, N]`` layout consumed
    by the standalone PyPTO W8A8 helpers.
    """

    if layer < 0:
        raise ValueError("layer must be non-negative")
    prefix = f"layers.{layer}.attn"
    names = [
        f"{prefix}.wq_a.weight",
        f"{prefix}.wq_a.weight_scale",
        f"{prefix}.wq_a.weight_offset",
        f"{prefix}.wq_b.weight",
        f"{prefix}.wq_b.weight_scale",
        f"{prefix}.wq_b.weight_offset",
        f"{prefix}.wkv.weight",
        f"{prefix}.wkv.weight_scale",
        f"{prefix}.wkv.weight_offset",
        f"{prefix}.q_norm.weight",
        f"{prefix}.kv_norm.weight",
        f"{prefix}.wo_a.weight",
        f"{prefix}.wo_b.weight",
        f"{prefix}.attn_sink",
    ]
    optional_names = [
        f"{prefix}.compressor.wkv.weight",
        f"{prefix}.compressor.wgate.weight",
        f"{prefix}.compressor.norm.weight",
        f"{prefix}.indexer.wk.weight",
        f"{prefix}.indexer.k_norm.weight",
        f"{prefix}.indexer.wq_b.weight",
        f"{prefix}.indexer.wq_b.weight_scale",
        f"{prefix}.indexer.wq_b.weight_offset",
        f"{prefix}.indexer.weights_proj.weight",
    ]
    names.extend(
        name for name in optional_names
        if (checkpoint.root / f"{name}.bin.json").is_file()
    )
    raw = checkpoint.require(names)
    result: dict[str, torch.Tensor] = dict(raw)
    for name in ("wq_a", "wq_b", "wkv"):
        result[f"{name}_canonical"] = raw[f"{prefix}.{name}.weight"].transpose(0, 1).contiguous()
        result[f"{name}_scale"] = flatten_scale(raw[f"{prefix}.{name}.weight_scale"], f"{prefix}.{name}.weight_scale")
    result["q_norm"] = raw[f"{prefix}.q_norm.weight"]
    result["kv_norm"] = raw[f"{prefix}.kv_norm.weight"]
    result["wo_a_raw"] = raw[f"{prefix}.wo_a.weight"]
    result["wo_b_raw"] = raw[f"{prefix}.wo_b.weight"]
    result["attn_sink"] = raw[f"{prefix}.attn_sink"]
    # The vLLM A2/A3 W8A8 path is symmetric.  Keep this check next to the
    # conversion so a per-channel asymmetric checkpoint cannot silently be
    # interpreted as a different operator ABI.
    for name in ("wq_a", "wq_b", "wkv"):
        offset = flatten_scale(raw[f"{prefix}.{name}.weight_offset"], f"{prefix}.{name}.weight_offset")
        if not torch.equal(offset, torch.zeros_like(offset)):
            raise ValueError(f"{prefix}.{name} has non-zero weight offsets; A2/A3 W8A8 requires symmetric weights")
        result[f"{name}_offset"] = offset

    # Wo-A is stored output-major as [groups * rank, group_input], while the
    # A2/A3 grouped kernel consumes [groups, group_input, rank].  Wo-B is
    # stored [hidden, groups * rank] and is transposed to [K, N].
    wo_a = raw[f"{prefix}.wo_a.weight"]
    wo_b = raw[f"{prefix}.wo_b.weight"]
    if wo_a.ndim != 2 or wo_b.ndim != 2:
        raise ValueError(f"{prefix} output projections must be matrices")
    if wo_a.shape[0] % 8:
        raise ValueError(f"{prefix}.wo_a first dimension must be divisible by 8")
    groups = 8
    rank = wo_a.shape[0] // groups
    result["wo_a"] = wo_a.reshape(groups, rank, wo_a.shape[1]).transpose(1, 2).contiguous()
    result["wo_b"] = wo_b.transpose(0, 1).contiguous()

    # The source layers carry the ratio-2 compressor/indexer tensors.  The
    # ratio-1 source (L20) uses the same names but only has compressor.wkv.
    compressor_wkv_name = f"{prefix}.compressor.wkv.weight"
    if compressor_wkv_name in raw:
        result["compressor_wkv"] = raw[compressor_wkv_name]
        result["compressor_wkv_canonical"] = raw[compressor_wkv_name].transpose(0, 1).contiguous()
        result["compressor_norm"] = raw[f"{prefix}.compressor.norm.weight"]
    index_prefix = f"{prefix}.indexer"
    if (checkpoint.root / f"{index_prefix}.wk.weight.bin.json").is_file():
        result["index_wk"] = raw[f"{index_prefix}.wk.weight"].transpose(0, 1).contiguous()
        result["index_k_norm"] = raw[f"{index_prefix}.k_norm.weight"]
    if (checkpoint.root / f"{index_prefix}.weights_proj.weight.bin.json").is_file():
        result["index_weights_proj"] = raw[f"{index_prefix}.weights_proj.weight"].transpose(0, 1).contiguous()
    if (checkpoint.root / f"{index_prefix}.wq_b.weight.bin.json").is_file():
        result["index_wq_b"] = raw[f"{index_prefix}.wq_b.weight"].transpose(0, 1).contiguous()
        result["index_wq_b_scale"] = flatten_scale(
            raw[f"{index_prefix}.wq_b.weight_scale"], f"{index_prefix}.wq_b.weight_scale"
        )
        index_offset = flatten_scale(
            raw[f"{index_prefix}.wq_b.weight_offset"], f"{index_prefix}.wq_b.weight_offset"
        )
        if not torch.equal(index_offset, torch.zeros_like(index_offset)):
            raise ValueError(f"{index_prefix}.wq_b has non-zero weight offsets")

    if f"{prefix}.compressor.wgate.weight" in raw:
        result["compressor_wgate"] = raw[f"{prefix}.compressor.wgate.weight"]
        result["compressor_wgate_canonical"] = raw[f"{prefix}.compressor.wgate.weight"].transpose(0, 1).contiguous()
    if "index_wk" in result:
        result["index_wk_raw"] = raw[f"{prefix}.indexer.wk.weight"]
        result["index_wk_canonical"] = result["index_wk"]
        result["index_wq_b_canonical"] = result["index_wq_b"]
        result["index_weights_proj_raw"] = raw[f"{prefix}.indexer.weights_proj.weight"]
    # Keep the original raw tensors for callers that need to inspect the
    # checkpoint layout, while exposing canonical tensors for A2/A3 kernels.
    result["wo_a_raw"] = wo_a
    result["wo_b_raw"] = wo_b
    return result


load_decoder_layer_weights = load_attention_weights


__all__ = ["RawCheckpoint", "flatten_scale", "load_attention_weights", "load_decoder_layer_weights"]
