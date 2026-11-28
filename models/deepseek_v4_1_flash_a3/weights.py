# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""W8A8 checkpoint reading and TP4/EP16 resident weight sharding."""

import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "models.deepseek_v4_1_flash_a3"

import hashlib
import json
import math
import struct

import torch

from .config import FLASH as M, EP_SIZE, N_LOCAL_EXPERTS, TP_SIZE, WEIGHT_LAYOUT_VERSION
from .config import SHARED_NZ_PANEL, SWA_LAYER_COUNT

ENGRAM_K = (M.engram_max_ngram_size - 1) * M.engram_n_heads * M.engram_head_dim
KV_OUT = (M.hc_mult + 1) * M.hidden_size
_DTYPES = {"F32": torch.float32, "BF16": torch.bfloat16, "I8": torch.int8}


def bf16_nz_shape(matrix_count, rows, columns):
    """Return the physical shape of BF16 Cube weights."""
    return matrix_count, columns // 16, rows // 16, 16, 16


def engram_wkv_nz_shape(matrix_count):
    """Describe the packed BF16 Engram projection buffer for runtime allocation."""
    return matrix_count, KV_OUT // 16, ENGRAM_K // 16, 16, 16


def pack_bf16_nz(weight):
    """Pack the final BF16 matrix dimensions into 16 by 16 Cube fractals."""
    rows, columns = weight.shape[-2:]
    blocks = weight.reshape(-1, rows // 16, 16, columns // 16, 16)
    return blocks.permute(0, 3, 1, 2, 4).contiguous().reshape(weight.shape)


def unpack_bf16_nz(weight):
    """Recover the logical BF16 matrix for a CPU reference."""
    rows, columns = weight.shape[-2:]
    blocks = weight.reshape(-1, columns // 16, rows // 16, 16, 16)
    return blocks.permute(0, 2, 3, 1, 4).contiguous().reshape(weight.shape)


def projection_int8_nz_shape(matrix_count, rows, columns):
    """Return the physical shape of INT8 Cube weights with C0=32."""
    return matrix_count, columns // 32, rows // 16, 16, 32


def pack_projection_int8_nz(weight):
    """Reorder the resident bytes without changing any quantized value."""
    rows, columns = weight.shape
    return weight.reshape(rows // 16, 16, columns // 32, 32).permute(2, 0, 1, 3).contiguous()


def pack_shared_gate_up(gate, up):
    """Interleave output-major gate and up panels in the resident NZ layout."""
    gate_panels = gate.reshape(-1, SHARED_NZ_PANEL, M.hidden_size)
    up_panels = up.reshape_as(gate_panels)
    paired = torch.stack((gate_panels, up_panels), dim=1).reshape(-1, M.hidden_size).t()
    return pack_projection_int8_nz(paired)


class Checkpoint:
    """Read requested original checkpoint tensors and selected embedding rows."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        config = json.loads(self._path("config.json").read_text())
        text = config["text_config"]
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "ascend" or quant.get("model_quant_type") != "W8A8_DYNAMIC":
            raise ValueError("checkpoint must use symmetric W8A8_DYNAMIC quantization")
        required = {
            "hidden_size": M.hidden_size,
            "num_hidden_layers": M.num_hidden_layers,
            "num_attention_heads": M.num_attention_heads,
            "head_dim": M.head_dim,
            "sliding_window": M.sliding_window,
            "kv_source_layer_ids": list(M.kv_source_layer_ids),
            "index_source_layer_ids": list(M.index_source_layer_ids),
            "engram_layer_ids": list(M.engram_layer_ids),
            "engram_num_embeddings": list(M.engram_num_embeddings),
            "engram_max_ngram_size": M.engram_max_ngram_size,
            "engram_n_heads": M.engram_n_heads,
            "engram_head_dim": M.engram_head_dim,
            "engram_compressed_vocab_size": M.engram_compressed_vocab_size,
            "engram_vocab_size": M.engram_vocab_size,
            "engram_pad_token_id": M.engram_pad_token_id,
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"checkpoint {name}={text.get(name)!r}, expected {expected!r}")
        index = json.loads(self._path("quant_model_weights.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"].copy()
        expected_rotation = {"value_projection_rotated": True, "value_basis": "quarot_global",
                             "key_and_gate_basis": "original", "runtime_delta_rotation": False}
        if config.get("engram_rotation_config") != expected_rotation:
            raise ValueError("unsupported Engram key/value basis contract")
        description = json.loads(self._path("quant_model_description.json").read_text())
        rotation = description["optional"]["quarot"]["rotation_map"]["global_rotation"]
        if rotation != "optional/quarot.safetensors":
            raise ValueError("unsupported global rotation payload location")
        self.weight_map["global_rotation"] = rotation
        self._headers = {}
        self._row_tables = {}

    def _path(self, name):
        path = (self.root / name).resolve()
        if self.root not in path.parents:
            raise ValueError(f"checkpoint file leaves its root: {name}")
        return path

    def tensor(self, name: str) -> torch.Tensor:
        """Return an owned CPU tensor without loading the rest of its shard."""
        shard = self.weight_map[name]
        path = self._path(shard)
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"safetensors header is too large: {shard}")
            if shard not in self._headers:
                self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype = _DTYPES[entry["dtype"]]
            shape = entry["shape"]
            start, end = entry["data_offsets"]
            expected_bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            if start < 0 or end - start != expected_bytes or 8 + header_size + end > path.stat().st_size:
                raise ValueError(f"invalid safetensors range for {name}")
            file.seek(8 + header_size + start)
            payload = bytearray(file.read(expected_bytes))
        if len(payload) != expected_bytes:
            raise ValueError(f"truncated safetensors payload for {name}")
        return torch.frombuffer(payload, dtype=dtype).reshape(shape)

    def tensor_rows(self, name: str, rows: torch.Tensor) -> torch.Tensor:
        """Read selected table rows without loading the complete tensor."""
        if rows.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("table row ids must have an integer dtype")
        ids = rows.to(dtype=torch.int64, device="cpu").flatten()
        if name not in self._row_tables:
            shard = self.weight_map[name]
            path = self._path(shard)
            with path.open("rb") as file:
                header_size = struct.unpack("<Q", file.read(8))[0]
                if header_size > 64 * 1024 * 1024:
                    raise ValueError(f"safetensors header is too large: {shard}")
                if shard not in self._headers:
                    self._headers[shard] = json.loads(file.read(header_size))
            entry = self._headers[shard][name]
            dtype, shape = _DTYPES[entry["dtype"]], entry["shape"]
            if len(shape) != 2:
                raise ValueError(f"invalid table rows for {name}")
            row_bytes = shape[1] * torch.empty((), dtype=dtype).element_size()
            start, end = entry["data_offsets"]
            file_bytes = path.stat().st_size
            if start < 0 or end - start != shape[0] * row_bytes or 8 + header_size + end > file_bytes:
                raise ValueError(f"invalid safetensors range for {name}")
            data = torch.from_file(str(path), shared=False, size=file_bytes, dtype=torch.uint8)
            table = data.narrow(0, 8 + header_size + start, end - start).reshape(shape[0], row_bytes)
            self._row_tables[name] = (table, dtype, shape[1])
        table, dtype, width = self._row_tables[name]
        if bool(((ids < 0) | (ids >= table.shape[0])).any()):
            raise ValueError(f"invalid table rows for {name}")
        selected = table.index_select(0, ids)
        return selected.view(dtype).reshape(*rows.shape, width)

    def descriptor(self, name):
        """Read shape and dtype metadata without materializing a tensor."""
        shard = self.weight_map[name]
        if shard not in self._headers:
            with self._path(shard).open("rb") as file:
                size = struct.unpack("<Q", file.read(8))[0]
                if size > 64 * 1024**2:
                    raise ValueError("safetensors header exceeds 64 MiB")
                self._headers[shard] = json.loads(file.read(size))
        return self._headers[shard][name]

    def rotation_blocks(self):
        """Read and validate the checkpoint's orthogonal 32-dimensional rotation blocks."""
        relative = self.weight_map["global_rotation"]
        path = self._path(relative)
        manifest = json.loads(self._path("aurora_export_manifest.json").read_text())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != manifest["files"][relative]["sha256"]:
            raise ValueError("global rotation payload does not match the checkpoint manifest")
        rotation = self.tensor("global_rotation")
        if rotation.dtype != torch.float32 or rotation.shape != (M.hidden_size, M.hidden_size):
            raise ValueError("global rotation must be a hidden-size FP32 square matrix")
        count = M.hidden_size // 32
        view = rotation.reshape(count, 32, count, 32).permute(0, 2, 1, 3)
        indices = torch.arange(count)
        blocks = view[indices, indices].contiguous()
        if int(torch.count_nonzero(rotation)) != int(torch.count_nonzero(blocks)):
            raise ValueError("A3 requires the checkpoint's block-diagonal 32-dimensional global rotation")
        gram = blocks @ blocks.transpose(-1, -2)
        if not bool(torch.isfinite(blocks).all()) or not torch.allclose(gram, torch.eye(32), atol=1e-6, rtol=0):
            raise ValueError("checkpoint global rotation blocks are not orthogonal")
        return blocks

    @property
    def identity(self):
        names = ("config.json", "quant_model_weights.safetensors.index.json", "aurora_export_manifest.json",
                 "quant_model_description.json", self.weight_map["global_rotation"])
        return {name: hashlib.sha256(self._path(name).read_bytes()).hexdigest() for name in names}


def forward_weight_layouts(local_experts=N_LOCAL_EXPERTS):
    """Return the shared prefill/decode ABI for one A3 rank."""
    if local_experts != 24:
        raise ValueError("EP16 requires 24 complete experts per rank")
    layers, d, inter = M.num_hidden_layers, M.hidden_size, M.moe_intermediate_size
    heads, groups = M.num_attention_heads // TP_SIZE, M.o_groups // TP_SIZE
    kv_sources, index_sources = len(M.kv_source_layer_ids), len(M.index_source_layer_ids)
    c2a_sources = sum(M.compress_ratios[source] == 2 for source in M.kv_source_layer_ids)
    result = {
        "engram_wkv": (engram_wkv_nz_shape(len(M.engram_layer_ids)), torch.bfloat16),
        "engram_weight": ((2, M.hc_mult, d), torch.float32),
        "engram_rotation": ((d // 32, 32, 32), torch.float32),
        "hc_attn_fn": ((layers * M.mix_hc, M.hc_dim), torch.float32),
        "hc_attn_scale": ((layers * 3,), torch.float32),
        "hc_attn_base": ((layers * M.mix_hc,), torch.float32),
        "attn_norm_weight": ((layers * d,), torch.bfloat16),
        "q_norm_weight": ((layers * M.q_lora_rank,), torch.bfloat16),
        "kv_norm_weight": ((layers * M.head_dim,), torch.bfloat16),
        "attn_sink": ((layers * heads,), torch.float32),
        "wo_a": (bf16_nz_shape(layers * groups, M.num_attention_heads * M.head_dim // M.o_groups, M.o_lora_rank), torch.bfloat16),
        "wo_b": (bf16_nz_shape(layers, groups * M.o_lora_rank, d), torch.bfloat16),
        "swa_wo_a": ((SWA_LAYER_COUNT * groups, M.num_attention_heads * M.head_dim // M.o_groups, M.o_lora_rank), torch.bfloat16),
        "swa_wo_b": ((SWA_LAYER_COUNT, groups * M.o_lora_rank, d), torch.bfloat16),
        "c2a_compressor_wkv": ((c2a_sources, d, M.head_dim), torch.bfloat16),
        "c2a_compressor_wgate": ((c2a_sources, d, M.head_dim), torch.bfloat16),
        "c1a_compressor_wkv": ((d, M.head_dim), torch.bfloat16),
        "compressor_norm_weight": ((kv_sources * M.head_dim,), torch.bfloat16),
        "index_wk": ((index_sources, M.head_dim, M.index_head_dim), torch.bfloat16),
        "index_norm_weight": ((index_sources * M.index_head_dim,), torch.bfloat16),
        "index_weights_proj": ((index_sources, d, M.index_n_heads), torch.bfloat16),
        "hc_ffn_fn": ((layers * M.mix_hc, M.hc_dim), torch.float32),
        "hc_ffn_scale": ((layers * 3,), torch.float32),
        "hc_ffn_base": ((layers * M.mix_hc,), torch.float32),
        "ffn_norm_weight": ((layers * d,), torch.bfloat16),
        "gate_weight": ((layers * M.n_routed_experts, d), torch.float32),
        "correction_bias": ((layers * M.n_routed_experts,), torch.float32),
        "final_norm_weight": ((d,), torch.bfloat16),
        "lm_head_weight": ((M.vocab_size // TP_SIZE, d), torch.bfloat16),
    }
    projections = {
        "wq_a": (layers, d, M.q_lora_rank),
        "wq_b": (layers, M.q_lora_rank, heads * M.head_dim),
        "wkv": (layers, d, M.head_dim),
        "index_wq_b": (index_sources, M.q_lora_rank, M.index_n_heads * M.index_head_dim),
        "shared_w2": (layers, inter, d),
        "routed_w1": (layers * local_experts, d, inter),
        "routed_w2": (layers * local_experts, inter, d),
        "routed_w3": (layers * local_experts, d, inter),
    }
    for name, shape in projections.items():
        matrices, input_width, output_width = shape
        if name in ("wq_a", "wq_b", "wkv", "shared_w2"):
            bank_shape = projection_int8_nz_shape(matrices, input_width, output_width)
        elif name.startswith("routed_"):
            bank_shape = (matrices, output_width, input_width)
        else:
            bank_shape = shape
        result[name] = (bank_shape, torch.int8)
        result[name + "_scale"] = ((matrices, output_width), torch.float32)

    result["shared_w13"] = (projection_int8_nz_shape(layers, d, 2 * inter), torch.int8)
    result["shared_w1_scale"] = ((layers, inter), torch.float32)
    result["shared_w3_scale"] = ((layers, inter), torch.float32)
    return result


class ResidentWeights:
    """Stream each checkpoint projection into persistent rank-local allocations."""

    layer_projections = (
        ("hc_attn_fn", "hc_attn_fn"), ("hc_attn_scale", "hc_attn_scale"),
        ("hc_attn_base", "hc_attn_base"), ("attn_norm_weight", "attn_norm.weight"),
        ("q_norm_weight", "attn.q_norm.weight"), ("kv_norm_weight", "attn.kv_norm.weight"),
        ("attn_sink", "attn.attn_sink"), ("hc_ffn_fn", "hc_ffn_fn"),
        ("hc_ffn_scale", "hc_ffn_scale"), ("hc_ffn_base", "hc_ffn_base"),
        ("ffn_norm_weight", "ffn_norm.weight"), ("gate_weight", "ffn.gate.weight"),
        ("correction_bias", "ffn.gate.bias"),
    )
    int8_projections = (
        ("wq_a", "attn.wq_a"), ("wq_b", "attn.wq_b"), ("wkv", "attn.wkv"),
        ("shared_w2", "ffn.shared_experts.w2"),
    )

    def __init__(self, checkpoint, *, ep_size=EP_SIZE):
        if (ep_size, TP_SIZE, N_LOCAL_EXPERTS) != (16, 4, 24):
            raise ValueError("resident weights require DP4/TP4/EP16 and 24 local experts")
        self.checkpoint = checkpoint
        self.ranks = ep_size
        self.local_experts = N_LOCAL_EXPERTS
        self.tensor_layouts = forward_weight_layouts()

    @property
    def bytes_per_rank(self):
        return sum(math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                   for shape, dtype in self.tensor_layouts.values())

    @property
    def byte_breakdown(self):
        """Count named resident allocations with the checkpoint's INT8/FP32 scale ABI."""
        result = {name: 0 for name in ("routed_experts", "shared_expert", "attention_tp", "lm_head", "replicated")}
        for name, (shape, dtype) in self.tensor_layouts.items():
            category = "replicated"
            if name.startswith("routed_"):
                category = "routed_experts"
            elif name.startswith("shared_"):
                category = "shared_expert"
            elif name.startswith(("wq_b", "wo_a", "wo_b", "swa_wo_", "attn_sink")):
                category = "attention_tp"
            elif name == "lm_head_weight":
                category = "lm_head"
            result[category] += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        return result

    def projection(self, stem, *, output_slice=None, input_slice=None, output_major=False):
        """Read a symmetric INT8 matrix and preserve the FP32 scale of each output channel."""
        weight = self.checkpoint.tensor(stem + ".weight")
        scale = self.checkpoint.tensor(stem + ".weight_scale")
        offset = self.checkpoint.tensor(stem + ".weight_offset")
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError(f"expected an INT8 matrix: {stem}")
        if scale.dtype != torch.float32 or scale.shape != (weight.shape[0], 1):
            raise ValueError(f"expected per-output-channel FP32 scales: {stem}")
        if offset.shape != scale.shape or bool(torch.count_nonzero(offset)):
            raise ValueError(f"only symmetric zero-offset W8A8 is supported: {stem}")
        if not bool(torch.isfinite(scale).all()) or bool((scale <= 0).any()):
            raise ValueError(f"invalid quantization scales: {stem}")
        if output_slice is not None:
            weight, scale = weight[output_slice], scale[output_slice]
        if input_slice is not None:
            weight = weight[:, input_slice]
        weight = weight.contiguous() if output_major else weight.t().contiguous()
        return weight, scale.reshape(1, -1).contiguous()

    def upload(self, runtime, rank, handles, *, layers=None):
        """Upload each sharded payload and scale once to the selected worker."""
        if rank not in range(self.ranks):
            raise ValueError("rank must be in [0, 16)")
        lane = rank % TP_SIZE
        checkpoint = self.checkpoint

        def copy(target, data, slot=0, slots=1):
            shape, dtype = self.tensor_layouts[target]
            elements = math.prod(shape) // slots
            if data.dtype != dtype:
                raise ValueError(f"checkpoint dtype mismatch for {target}: {data.dtype}, expected {dtype}")
            data = data.contiguous()
            if data.numel() != elements:
                raise ValueError(f"checkpoint shape mismatch for {target}: {data.shape}, expected {elements} elements")
            runtime.copy_to(handles[target].data_ptr, data.data_ptr(), data.numel() * data.element_size(),
                            dst_offset=slot * elements * data.element_size(), worker_id=rank)

        copy("engram_rotation", checkpoint.rotation_blocks())
        total_layers = M.num_hidden_layers
        selected_layers = range(total_layers) if layers is None else sorted(set(layers))
        if any(layer not in range(total_layers) for layer in selected_layers):
            raise ValueError("resident weight layers must be in [0, 40)")
        heads, groups = M.num_attention_heads // TP_SIZE, M.o_groups // TP_SIZE
        for layer in selected_layers:
            for target, suffix in self.int8_projections:
                output_slice = slice(lane * heads * M.head_dim, (lane + 1) * heads * M.head_dim) if target == "wq_b" else None
                weight, scale = self.projection(f"layers.{layer}.{suffix}", output_slice=output_slice)
                if target in ("wq_a", "wq_b", "wkv", "shared_w2"):
                    weight = pack_projection_int8_nz(weight)
                copy(target, weight, layer, total_layers)
                copy(target + "_scale", scale, layer, total_layers)
            shared = f"layers.{layer}.ffn.shared_experts"
            gate, gate_scale = self.projection(shared + ".w1", output_major=True)
            up, up_scale = self.projection(shared + ".w3", output_major=True)
            copy("shared_w13", pack_shared_gate_up(gate, up), layer, total_layers)
            copy("shared_w1_scale", gate_scale, layer, total_layers)
            copy("shared_w3_scale", up_scale, layer, total_layers)
            for target, suffix in self.layer_projections:
                value = checkpoint.tensor(f"layers.{layer}.{suffix}")
                if target == "attn_sink":
                    value = value[lane * heads:(lane + 1) * heads]
                elif target == "gate_weight":
                    value = value.float()
                copy(target, value, layer, total_layers)
            wo_a = checkpoint.tensor(f"layers.{layer}.attn.wo_a.weight")
            wo_a = wo_a.reshape(M.o_groups, M.o_lora_rank, -1)[lane * groups:(lane + 1) * groups]
            wo_a = wo_a.transpose(-1, -2)
            copy("wo_a", pack_bf16_nz(wo_a), layer, total_layers)
            wo_b = checkpoint.tensor(f"layers.{layer}.attn.wo_b.weight")
            width = groups * M.o_lora_rank
            wo_b = wo_b[:, lane * width:(lane + 1) * width].t()
            copy("wo_b", pack_bf16_nz(wo_b), layer, total_layers)
            if layer < SWA_LAYER_COUNT and "swa_wo_a" in handles:
                copy("swa_wo_a", wo_a, layer, SWA_LAYER_COUNT)
                copy("swa_wo_b", wo_b, layer, SWA_LAYER_COUNT)
            for local in range(self.local_experts):
                expert = rank * self.local_experts + local
                for projection in ("w1", "w2", "w3"):
                    weight, scale = self.projection(f"layers.{layer}.ffn.experts.{expert}.{projection}",
                                                    output_major=True)
                    slot = layer * self.local_experts + local
                    copy("routed_" + projection, weight, slot, total_layers * self.local_experts)
                    copy("routed_" + projection + "_scale", scale, slot, total_layers * self.local_experts)
            if (layer + 1) % 10 == 0:
                print(f"A3_RANK_LAYER_WEIGHTS_READY {rank} {layer + 1}", flush=True)
        for slot, layer in enumerate(M.kv_source_layer_ids):
            copy("compressor_norm_weight", checkpoint.tensor(f"layers.{layer}.attn.compressor.norm.weight"), slot, len(M.kv_source_layer_ids))
        c2a = [source for source in M.kv_source_layer_ids if M.compress_ratios[source] == 2]
        for slot, layer in enumerate(c2a):
            for name in ("wkv", "wgate"):
                copy("c2a_compressor_" + name, checkpoint.tensor(f"layers.{layer}.attn.compressor.{name}.weight").t(), slot, len(c2a))
        source = next(source for source in M.kv_source_layer_ids if M.compress_ratios[source] == 1)
        copy("c1a_compressor_wkv", checkpoint.tensor(f"layers.{source}.attn.compressor.wkv.weight").t())
        for slot, layer in enumerate(M.index_source_layer_ids):
            prefix = f"layers.{layer}.attn.indexer."
            weight, scale = self.projection(prefix + "wq_b")
            copy("index_wq_b", weight, slot, len(M.index_source_layer_ids))
            copy("index_wq_b_scale", scale, slot, len(M.index_source_layer_ids))
            for target, suffix in (("index_wk", "wk.weight"), ("index_norm_weight", "k_norm.weight"),
                                   ("index_weights_proj", "weights_proj.weight")):
                name = prefix + suffix
                shape, dtype = self.tensor_layouts[target]
                if name not in checkpoint.weight_map and layer not in M.kv_source_layer_ids:
                    value = torch.zeros(math.prod(shape) // len(M.index_source_layer_ids), dtype=dtype)
                else:
                    value = checkpoint.tensor(name)
                    if target in ("index_wk", "index_weights_proj"):
                        value = value.t().contiguous()
                copy(target, value, slot, len(M.index_source_layer_ids))
        for slot, layer in enumerate(M.engram_layer_ids):
            prefix = f"layers.{layer}.engram."
            engram_wkv = checkpoint.tensor(prefix + "wkv.weight").t()
            copy("engram_wkv", pack_bf16_nz(engram_wkv), slot, 2)
            gate = checkpoint.tensor(prefix + "q_weight").float() * checkpoint.tensor(prefix + "k_weight").float()
            copy("engram_weight", gate, slot, 2)
        copy("final_norm_weight", checkpoint.tensor("norm.weight"))
        vocab = M.vocab_size // TP_SIZE
        rows = torch.arange(lane * vocab, (lane + 1) * vocab)
        copy("lm_head_weight", checkpoint.tensor_rows("head.weight", rows))

    def lookup_rows(self, token_ids):
        """Read the rotated BF16 checkpoint embedding rows."""
        return self.checkpoint.tensor_rows("embed.weight", token_ids).bfloat16()


def audit_checkpoint(weights):
    """Check real projection formats, TP shard boundaries, Engram rows and rotation blocks."""
    checkpoint = weights.checkpoint
    checks = []
    suffixes = ("attn.wq_a", "attn.wq_b", "attn.wkv", "ffn.shared_experts.w1",
                "ffn.experts.0.w1", "ffn.experts.383.w2")
    for layer in (0, 1, 2, 3, 14, 20, 21, 24):
        for suffix in suffixes:
            stem = f"layers.{layer}.{suffix}"
            weight, scale = weights.projection(stem)
            shape = checkpoint.descriptor(stem + ".weight")["shape"]
            if tuple(weight.shape) != tuple(reversed(shape)):
                raise ValueError(f"projection transpose shape mismatch: {stem}")
            if suffix == "attn.wq_b":
                width = weight.shape[1] // TP_SIZE
                shards = [weights.projection(stem, output_slice=slice(lane * width, (lane + 1) * width))
                          for lane in range(TP_SIZE)]
                if not torch.equal(torch.cat([shard[0] for shard in shards], 1), weight):
                    raise ValueError(f"Q-B payload reconstruction mismatch: {stem}")
                if not torch.equal(torch.cat([shard[1] for shard in shards], 1), scale):
                    raise ValueError(f"Q-B scale reconstruction mismatch: {stem}")
            checks.append({"name": stem, "shape": list(weight.shape), "scale_shape": list(scale.shape),
                           "status": "pass"})
        for suffix in ("attn.wo_a.weight", "attn.wo_b.weight"):
            name = f"layers.{layer}.{suffix}"
            weight = checkpoint.tensor(name)
            if weight.dtype != torch.bfloat16:
                raise ValueError(f"output projection must be BF16: {name}")
            if suffix == "attn.wo_a.weight":
                logical = weight.reshape(M.o_groups, M.o_lora_rank, -1)
                groups = M.o_groups // TP_SIZE
                shards = [logical[lane * groups:(lane + 1) * groups].transpose(-1, -2).contiguous()
                          for lane in range(TP_SIZE)]
                restored = torch.cat(shards).transpose(-1, -2).reshape_as(weight)
            else:
                width = weight.shape[1] // TP_SIZE
                shards = [weight[:, lane * width:(lane + 1) * width].t().contiguous() for lane in range(TP_SIZE)]
                restored = torch.cat(shards).t()
            if not torch.equal(restored, weight):
                raise ValueError(f"output projection reconstruction mismatch: {name}")
            checks.append({"name": name, "status": "pass"})
    vocab = M.vocab_size // TP_SIZE
    head_rows = torch.tensor([value for lane in range(TP_SIZE) for value in (lane * vocab, (lane + 1) * vocab - 1)])
    head = checkpoint.tensor_rows("head.weight", head_rows)
    if head.shape != (8, M.hidden_size) or head.dtype != torch.bfloat16 or not bool(torch.isfinite(head).all()):
        raise ValueError("invalid LM head boundary rows")
    for layer in M.engram_layer_ids:
        prefix = f"layers.{layer}.engram.embed."
        shape = checkpoint.descriptor(prefix + "weight")["shape"]
        rows = torch.tensor([0, shape[0] - 1])
        payload, scales = checkpoint.tensor_rows(prefix + "weight", rows), checkpoint.tensor_rows(prefix + "scale", rows)
        if payload.dtype != torch.int8 or scales.dtype != torch.float32 or scales.shape != (2, 8):
            raise ValueError(f"invalid Engram INT8/FP32 table: {prefix}")
        decoded = payload.float() * scales.repeat_interleave(32, dim=-1)
        if not bool(torch.isfinite(decoded).all()):
            raise ValueError(f"nonfinite Engram table: {prefix}")
        checks.append({"name": prefix.removesuffix("."), "boundary_rows": rows.tolist(), "status": "pass"})
    rotation = checkpoint.rotation_blocks()
    checks.append({"name": "global_rotation", "shape": list(rotation.shape), "status": "pass"})
    return {"checkpoint_identity": checkpoint.identity, "weight_layout": WEIGHT_LAYOUT_VERSION,
            "resident_bytes_per_rank": weights.bytes_per_rank, "resident_byte_breakdown": weights.byte_breakdown,
            "resident_gib_per_rank": weights.bytes_per_rank / 1024**3, "projection_and_sharding_checks": checks,
            "lm_head_boundary_rows": head_rows.tolist(), "status": "host_weight_format_and_sharding_pass",
            "limits": "Device model output, peak HBM and performance acceptance remain separate checks."}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    checkpoint = Checkpoint(args.checkpoint)
    weights = ResidentWeights(checkpoint)
    report = {"layout_version": WEIGHT_LAYOUT_VERSION, "checkpoint_identity": checkpoint.identity,
              "resident_bytes_per_rank": weights.bytes_per_rank, "resident_byte_breakdown": weights.byte_breakdown,
              "resident_gib_per_rank": weights.bytes_per_rank / 1024**3,
              "topology": {"dp": 4, "tp": 4, "ep": 16, "local_experts": 24}}
    if args.audit:
        report.update(audit_checkpoint(weights))
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
