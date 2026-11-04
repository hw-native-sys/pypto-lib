# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inspect a DeepSeek V4.1 Aurora safetensors export without loading weights.

The Decode gate needs provenance and layout evidence before it can submit a
device job.  This module reads metadata and safetensors headers only.  It
never materializes a tensor and deliberately does not hash the 743GB payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_LAYERS = (0, 1, 2, 3, 14, 15, 19, 20, 21, 24, 25)
METADATA_FILES = (
    "config.json",
    "quant_model_weights.safetensors.index.json",
    "quant_model_description.json",
    "aurora_export_manifest.json",
)
SHARD_PATTERN = re.compile(r"quant_model_weights-(\d{5})-of-(\d{5})\.safetensors$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safetensors_header(path: Path) -> tuple[int, str, dict[str, Any]]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors length prefix: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ValueError(f"truncated safetensors header: {path}")
    try:
        payload = json.loads(header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid safetensors header: {path}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"safetensors header must be an object: {path}")
    return header_length, hashlib.sha256(raw_length + header).hexdigest(), payload


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", {})
    keys = (
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "n_routed_experts",
        "n_shared_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "compress_ratios",
        "kv_source_layer_ids",
        "index_source_layer_ids",
        "index_n_heads",
        "index_head_dim",
        "index_topk",
        "candidate_source_layer_id",
        "engram_layer_ids",
        "engram_num_embeddings",
        "engram_n_heads",
        "engram_head_dim",
    )
    return {key: text.get(key) for key in keys}


def inspect_checkpoint(root: str | Path) -> dict[str, Any]:
    """Return a metadata and header manifest for one Aurora export."""

    checkpoint_root = Path(root).expanduser().resolve()
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint_root}")
    missing = [name for name in METADATA_FILES if not (checkpoint_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint metadata is missing: {', '.join(missing)}")

    config = _read_json(checkpoint_root / "config.json")
    index = _read_json(checkpoint_root / "quant_model_weights.safetensors.index.json")
    description = _read_json(checkpoint_root / "quant_model_description.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise TypeError("safetensors index does not contain an object weight_map")

    shard_paths: dict[str, Path] = {}
    for path in sorted(checkpoint_root.glob("quant_model_weights-*-of-*.safetensors")):
        match = SHARD_PATTERN.fullmatch(path.name)
        if match:
            shard_paths[path.name] = path
    expected_shards = {name for name in weight_map.values() if isinstance(name, str) and SHARD_PATTERN.fullmatch(name)}
    declared_totals = {int(SHARD_PATTERN.fullmatch(name).group(2)) for name in expected_shards}
    if len(declared_totals) != 1:
        raise ValueError(f"inconsistent shard totals in index: {sorted(declared_totals)}")
    declared_shard_total = declared_totals.pop()
    if not expected_shards:
        raise ValueError("safetensors index contains no quantized weight shards")
    missing_shards = sorted(expected_shards - shard_paths.keys())

    headers: dict[str, dict[str, Any]] = {}
    for name, path in shard_paths.items():
        headers[name] = {
            "bytes": path.stat().st_size,
            "header_scanned": False,
        }

    target_layer_files: dict[str, list[str]] = {}
    target_tensor_samples: dict[str, dict[str, Any]] = {}
    scanned_headers: dict[str, dict[str, Any]] = {}
    for layer in TARGET_LAYERS:
        prefix = f"layers.{layer}."
        names = sorted(name for name in weight_map if name.startswith(prefix))
        files = {weight_map[name] for name in names}
        target_layer_files[str(layer)] = sorted(files)
        sample_names = [
            name
            for name in (
                f"layers.{layer}.attn.wq_a.weight",
                f"layers.{layer}.attn.wq_a.weight_scale",
                f"layers.{layer}.attn.wq_b.weight",
                f"layers.{layer}.attn.wkv.weight",
                f"layers.{layer}.attn.wo_a.weight",
                f"layers.{layer}.attn.wo_b.weight",
                f"layers.{layer}.attn.indexer.wk.weight",
                f"layers.{layer}.attn.indexer.wq_b.weight",
                f"layers.{layer}.ffn.experts.0.w1.weight",
            )
            if name in weight_map
        ]
        samples: dict[str, Any] = {}
        for name in sample_names:
            shard = weight_map[name]
            if shard not in scanned_headers:
                header_bytes, header_sha256, header = _safetensors_header(checkpoint_root / shard)
                tensors = {key: value for key, value in header.items() if key != "__metadata__"}
                scanned_headers[shard] = header
                headers[shard].update({
                    "header_bytes": header_bytes,
                    "header_sha256": header_sha256,
                    "tensor_count": len(tensors),
                    "header_scanned": True,
                })
            header = scanned_headers[shard]
            tensor = header[name]
            samples[name] = {
                "shard": shard,
                "dtype": tensor.get("dtype"),
                "shape": tensor.get("shape"),
            }
        target_tensor_samples[str(layer)] = samples

    quant_modes = Counter(value for value in description.values() if isinstance(value, str))
    engram_files = sorted(
        name for name in ("engram_embed_weight_l1.safetensors", "engram_embed_scale_l1.safetensors",
                          "engram_embed_weight_l14.safetensors", "engram_embed_scale_l14.safetensors")
        if (checkpoint_root / name).is_file()
    )
    engram_records = []
    for name in engram_files:
        path = checkpoint_root / name
        header_bytes, header_sha256, header = _safetensors_header(path)
        engram_records.append({
            "file": name,
            "bytes": path.stat().st_size,
            "header_bytes": header_bytes,
            "header_sha256": header_sha256,
            "tensors": {
                key: {"dtype": value.get("dtype"), "shape": value.get("shape")}
                for key, value in header.items()
                if key != "__metadata__"
            },
        })
    return {
        "schema_version": 1,
        "status": "metadata_verified" if not missing_shards else "incomplete",
        "root": str(checkpoint_root),
        "payload_hashing": "omitted; safetensors payloads are immutable external assets",
        "metadata": {
            name: {"bytes": (checkpoint_root / name).stat().st_size, "sha256": _sha256(checkpoint_root / name)}
            for name in METADATA_FILES
        },
        "config": _config_summary(config),
        "quantization": {
            "config": config.get("quantization_config"),
            "description_modes": dict(sorted(quant_modes.items())),
        },
        "index": {
            "weight_map_entries": len(weight_map),
            "total_size": index.get("metadata", {}).get("total_size"),
            "weight_shards": len(expected_shards),
            "declared_shard_total": declared_shard_total,
            "present_shards": len(shard_paths),
            "missing_shards": missing_shards,
        },
        "shards": headers,
        "engram_files": engram_records,
        "target_layer_files": target_layer_files,
        "target_tensor_samples": target_tensor_samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = inspect_checkpoint(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "metadata_verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
