# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Lazy reader for the Aurora W8A8 Decode checkpoint layout.

The reader owns no model computation.  It resolves a tensor name through the
official safetensors index and loads only that tensor on the host, leaving
PyPTO to own all device computation and quantization boundaries.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class DecodeSafetensorsCheckpoint:
    """Resolve and lazily load tensors from an indexed Aurora export."""

    INDEX_NAME = "quant_model_weights.safetensors.index.json"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {self.root}")
        index_path = self.root / self.INDEX_NAME
        if not index_path.is_file():
            raise FileNotFoundError(f"safetensors index is missing: {index_path}")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"safetensors index has no weight_map: {index_path}")
        self.weight_map: dict[str, str] = {str(name): str(shard) for name, shard in weight_map.items()}
        self.config = self._load_json("config.json")
        self.quantization_description = self._load_json("quant_model_description.json")

    def _load_json(self, name: str) -> dict[str, Any]:
        path = self.root / name
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint metadata is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError(f"checkpoint metadata must be an object: {path}")
        return value

    def has(self, name: str) -> bool:
        return name in self.weight_map or (self.root / name).is_file()

    def shard_for(self, name: str) -> Path:
        try:
            shard = self.weight_map[name]
        except KeyError as exc:
            raise KeyError(f"tensor is absent from the safetensors index: {name}") from exc
        path = self.root / shard
        if not path.is_file():
            raise FileNotFoundError(f"indexed shard is missing: {path}")
        return path

    @contextmanager
    def open_tensor(self, name: str) -> Iterator[Any]:
        """Yield one safetensors handle and the requested tensor name."""

        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Aurora Decode weights require the safetensors package") from exc
        path = self.shard_for(name)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if name not in handle.keys():
                raise KeyError(f"tensor {name!r} is not present in indexed shard {path}")
            yield handle, name

    def tensor(self, name: str) -> Any:
        """Load one host tensor; no tensor is retained by this reader."""

        with self.open_tensor(name) as (handle, tensor_name):
            return handle.get_tensor(tensor_name)

    def require(self, names: list[str] | tuple[str, ...]) -> dict[str, Any]:
        return {name: self.tensor(name) for name in names}

    def layer_tensor_names(self, layer: int) -> tuple[str, ...]:
        if layer < 0:
            raise ValueError("layer must be non-negative")
        prefix = f"layers.{layer}."
        return tuple(sorted(name for name in self.weight_map if name.startswith(prefix)))

    def layer_shards(self, layer: int) -> tuple[Path, ...]:
        return tuple(sorted({self.shard_for(name) for name in self.layer_tensor_names(layer)}))

    def engram_tensor(self, layer: int, kind: str) -> Any:
        if layer not in (1, 14):
            raise ValueError("Aurora Engram tables are defined for layers 1 and 14")
        if kind not in ("weight", "scale"):
            raise ValueError("Engram kind must be 'weight' or 'scale'")
        path = self.root / f"engram_embed_{kind}_l{layer}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(f"Engram tensor file is missing: {path}")
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("Aurora Engram weights require the safetensors package") from exc
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            names = list(handle.keys())
            if len(names) != 1:
                raise ValueError(f"expected one tensor in {path}, found {names}")
            return handle.get_tensor(names[0])


__all__ = ["DecodeSafetensorsCheckpoint"]
