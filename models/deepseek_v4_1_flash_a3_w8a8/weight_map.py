# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Map the read-only DeepSeek-V4.1-Flash-w8a8 checkpoint to the A3 W8A8 model arguments."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from config import FLASH as M, LOCAL_HEADS, N_LOCAL_EXPERTS, TP_SIZE, EP_SIZE, LOCAL_O_GROUPS, add_parallel_arguments, DP_SIZE, WORLD_SIZE
from utils import SafeTensorReader, load_embedding_storage, load_quant_description


LAYER_RE = re.compile(r"^layers\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(r"^ffn\.experts\.(\d+)\.(w[123])\.(weight|weight_scale|weight_offset)$")
ENGRAM_TABLE = {"layers.1.engram.embed", "layers.14.engram.embed"}

# FLOAT tags map to BF16/FP32 model arguments; W8A8 tags map to int8 + scale (+ offset).
W8A8_SUFFIXES = ("weight_scale", "weight_offset")


@dataclass
class Mapping:
    checkpoint_name: str
    source_dtype: str
    source_shape: list[int]
    bytes: int
    destination: str
    transform: str
    shard: str
    rank_local_shape: list[int]
    rank_local_bytes: int
    included: bool
    reason: str


def _classify(name: str) -> tuple[bool, str]:
    if name.startswith("vision."):
        return False, "vision tower excluded (text-only phase 1)"
    if name.startswith("aligner."):
        return False, "aligner excluded (text-only phase 1)"
    if name.startswith("image_"):
        return False, "image token tensors excluded (text-only phase 1)"
    if name.startswith("mtp."):
        return False, "MTP/DSpark layers excluded from phase 1"
    layer = LAYER_RE.match(name)
    if layer and int(layer.group(1)) >= M.num_hidden_layers:
        return False, "layer outside the 40-layer text backbone"
    if layer is None and name not in ("embed.weight", "head.weight", "norm.weight"):
        return False, "not a text-backbone tensor"
    if any(name.startswith(table) for table in ENGRAM_TABLE):
        return True, "Engram table (host-offloaded, int8_sym group-32)"
    return True, "text backbone"


def _destination(name: str) -> tuple[str, str, str, list[int]]:
    """Return destination argument, transform, shard kind, and rank-local shape."""
    if name == "embed.weight":
        return "embed_weight", "bf16 keep", "tp_vocab", [M.vocab_size // TP_SIZE, M.hidden_size]
    if name == "head.weight":
        return "lm_head_weight", "bf16 keep", "tp_vocab", [M.vocab_size // TP_SIZE, M.hidden_size]
    if name == "norm.weight":
        return "final_norm", "bf16 keep", "replicated", [M.hidden_size]
    if name.startswith("layers.1.engram.embed") or name.startswith("layers.14.engram.embed"):
        slot = 0 if name.startswith("layers.1.") else 1
        if name.endswith(".scale"):
            return f"engram_table_scale[{slot}]", "int8_sym group-32 fp32 scale", "engram_host", list(_source_shape_cache[name])
        return f"engram_table_weight[{slot}]", "int8 keep", "engram_host", list(_source_shape_cache[name])
    layer = LAYER_RE.match(name)
    layer_id = int(layer.group(1))
    rest = layer.group(2)
    if rest.startswith("engram."):
        return f"layers.{layer_id}.engram.{rest}", "bf16/fp32 keep", "replicated", list(_source_shape_cache[name])
    if rest == "attn_norm.weight":
        return f"layers.{layer_id}.attn_norm", "bf16 keep", "replicated", [M.hidden_size]
    if rest == "ffn_norm.weight":
        return f"layers.{layer_id}.ffn_norm", "bf16 keep", "replicated", [M.hidden_size]
    if rest == "attn.q_norm.weight":
        return f"layers.{layer_id}.q_norm", "bf16 keep", "replicated", [M.q_lora_rank]
    if rest == "attn.kv_norm.weight":
        return f"layers.{layer_id}.kv_norm", "bf16 keep", "replicated", [M.head_dim]
    if rest == "attn.attn_sink":
        return f"layers.{layer_id}.attn_sink", "fp32 keep", "tp_heads", [LOCAL_HEADS]
    if rest == "attn.wq_a.weight":
        return f"layers.{layer_id}.wq_a", "transpose to [K,N] + NZ pack", "replicated", [M.hidden_size, M.q_lora_rank]
    if rest == "attn.wq_a.weight_scale":
        return f"layers.{layer_id}.wq_a_scale", "fp32 keep", "replicated", [M.q_lora_rank]
    if rest == "attn.wq_a.weight_offset":
        return f"layers.{layer_id}.wq_a_offset", "fp32 keep (verified zero)", "replicated", [M.q_lora_rank]
    if rest == "attn.wq_b.weight":
        return f"layers.{layer_id}.wq_b", "transpose to [K,N] + NZ pack", "tp_heads", [M.q_lora_rank, LOCAL_HEADS * M.head_dim]
    if rest == "attn.wq_b.weight_scale":
        return f"layers.{layer_id}.wq_b_scale", "fp32 head-slice", "tp_heads", [LOCAL_HEADS * M.head_dim]
    if rest == "attn.wq_b.weight_offset":
        return f"layers.{layer_id}.wq_b_offset", "fp32 head-slice (verified zero)", "tp_heads", [LOCAL_HEADS * M.head_dim]
    if rest == "attn.wkv.weight":
        return f"layers.{layer_id}.wkv", "transpose to [K,N] + NZ pack", "replicated", [M.hidden_size, M.head_dim]
    if rest == "attn.wkv.weight_scale":
        return f"layers.{layer_id}.wkv_scale", "fp32 keep", "replicated", [M.head_dim]
    if rest == "attn.wkv.weight_offset":
        return f"layers.{layer_id}.wkv_offset", "fp32 keep (verified zero)", "replicated", [M.head_dim]
    if rest == "attn.wo_a.weight":
        return f"layers.{layer_id}.wo_a", "bf16 keep, group-stacked", "tp_groups", [LOCAL_O_GROUPS, M.o_lora_rank, M.num_attention_heads // M.o_groups * M.head_dim]
    if rest == "attn.wo_b.weight":
        return f"layers.{layer_id}.wo_b", "bf16 keep, output-group slice", "tp_groups", [M.hidden_size, LOCAL_O_GROUPS * M.o_lora_rank]
    if rest.startswith("attn.compressor."):
        return f"layers.{layer_id}.{rest}", "bf16/fp32 keep", "replicated", list(_source_shape_cache[name])
    if rest.startswith("attn.indexer."):
        if rest == "attn.indexer.wq_b.weight":
            return f"layers.{layer_id}.indexer_wq_b", "transpose to [K,N] + NZ pack", "replicated", [M.q_lora_rank, M.index_n_heads * M.index_head_dim]
        if rest == "attn.indexer.wq_b.weight_scale":
            return f"layers.{layer_id}.indexer_wq_b_scale", "fp32 replicated index heads", "replicated", [M.index_n_heads * M.index_head_dim]
        if rest == "attn.indexer.wq_b.weight_offset":
            return f"layers.{layer_id}.indexer_wq_b_offset", "fp32 replicated index heads, offset folded", "replicated", [M.index_n_heads * M.index_head_dim]
        return f"layers.{layer_id}.{rest}", "bf16/fp32 keep", "replicated", list(_source_shape_cache[name])
    if rest.startswith("ffn.gate."):
        return f"layers.{layer_id}.{rest}", "fp32/bf16 keep", "replicated", list(_source_shape_cache[name])
    if rest.startswith("hc_"):
        return f"layers.{layer_id}.{rest}", "fp32 keep", "replicated", list(_source_shape_cache[name])
    expert = EXPERT_RE.match(rest)
    if expert:
        global_expert = int(expert.group(1))
        local = global_expert % N_LOCAL_EXPERTS
        rank = global_expert // N_LOCAL_EXPERTS
        if rank >= EP_SIZE:
            return f"layers.{layer_id}.expert[{global_expert}]", "expert outside EP", "ep", [0]
        return f"layers.{layer_id}.experts[{local}]", "int8 [K,N] + fp32 per-channel scale, offset folded", "ep", list(_source_shape_cache[name])
    if rest.startswith("ffn.shared_experts."):
        return f"layers.{layer_id}.{rest}", "int8 [K,N] + fp32 per-channel scale, offset folded", "replicated", list(_source_shape_cache[name])
    return f"layers.{layer_id}.{rest}", "unmapped", "replicated", list(_source_shape_cache[name])


_source_shape_cache: dict[str, tuple[int, ...]] = {}


def build_map(checkpoint: str | Path) -> tuple[list[Mapping], dict]:
    reader = SafeTensorReader(checkpoint)
    tags = load_quant_description(checkpoint)
    storage = load_embedding_storage(checkpoint)
    mappings: list[Mapping] = []
    missing_tags = []
    for name, shard in reader.weight_map.items():
        if name not in tags:
            missing_tags.append(name)
    for name, tag in tags.items():
        if name not in reader.weight_map:
            continue
        shape, _ = reader.shape_and_dtype(name)
        _source_shape_cache[name] = shape
        size = reader.nbytes(name)
        included, reason = _classify(name)
        if included:
            destination, transform, kind, local_shape = _destination(name)
        else:
            destination, transform, kind, local_shape = "", "", "excluded", [0]
        if name.startswith("layers.1.engram.embed") or name.startswith("layers.14.engram.embed"):
            kind = "engram_host"
        rank_bytes = 0
        if included:
            if kind == "replicated":
                rank_bytes = size
            elif kind == "engram_host":
                rank_bytes = 0
            elif kind in ("tp_heads", "tp_groups", "tp_vocab", "tp_index_heads"):
                rank_bytes = size // TP_SIZE
            elif kind == "ep":
                rank_bytes = size // EP_SIZE
        mappings.append(
            Mapping(name, tag, list(shape), size, destination, transform, kind, local_shape, rank_bytes, included, reason)
        )
    summary = {
        "tensors_total": len(mappings),
        "included": sum(1 for item in mappings if item.included),
        "excluded": sum(1 for item in mappings if not item.included),
        "included_bytes": sum(item.bytes for item in mappings if item.included),
        "excluded_bytes": sum(item.bytes for item in mappings if not item.included),
        "rank_local_bytes": sum(item.rank_local_bytes for item in mappings),
        "missing_tag_entries": missing_tags[:20],
        "embedding_storage": storage,
        "topology": {"TP": TP_SIZE, "EP": EP_SIZE, "DP": DP_SIZE, "WORLD": WORLD_SIZE, "local_experts": N_LOCAL_EXPERTS},
        "unmapped": [item.checkpoint_name for item in mappings if item.included and item.transform == "unmapped"],
    }
    return mappings, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    add_parallel_arguments(parser)
    parser.add_argument("--checkpoint", default="/mnt/data/models/DeepSeek-V4.1-Flash-w8a8")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    mappings, summary = build_map(args.checkpoint)
    print(json.dumps(summary, indent=1, default=str))
    if args.output:
        detail = [item for item in mappings if item.included and ".experts[" not in item.destination]
        expert_rows = [item for item in mappings if item.included and ".experts[" in item.destination]
        excluded_families: dict[str, dict] = {}
        for item in mappings:
            if item.included:
                continue
            key = item.reason
            entry = excluded_families.setdefault(key, {"count": 0, "bytes": 0, "examples": []})
            entry["count"] += 1
            entry["bytes"] += item.bytes
            if len(entry["examples"]) < 3:
                entry["examples"].append(item.checkpoint_name)
        payload = {
            "summary": summary,
            "mappings": [asdict(item) for item in detail],
            "expert_summary": {
                "count": len(expert_rows),
                "bytes": sum(item.bytes for item in expert_rows),
                "per_rank_bytes": sum(item.rank_local_bytes for item in expert_rows),
                "destination_pattern": f"layers.N.experts[global_id % {N_LOCAL_EXPERTS}]",
            },
            "excluded": excluded_families,
        }
        Path(args.output).write_text(json.dumps(payload, indent=1, default=str))
        print(f"wrote {args.output}")
    if summary["unmapped"]:
        print("UNMAPPED:", summary["unmapped"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
