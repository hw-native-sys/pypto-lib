# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host weight loader from the frozen checkpoint map to complete-model arguments.

Weights are stored pre-rotated and pre-norm-folded; the loader applies only the
per-channel ``w = (q - offset) * scale`` offset fold, the logical ``[N, K]`` to
``[K, N]`` transpose, and NZ packing.  Routed experts use whole-expert EP partitions; TP slices cover ``wq_b``, ``wo_a``,
``wo_b``, ``embed``, ``head``, and ``attn_sink``. Indexer heads and shared
FFN matrices remain replicated across TP ranks.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
import math
from pathlib import Path

import torch

from config import ENGRAM_GROUP_SIZE, EP_SIZE, FLASH as M, LOCAL_HEADS, LOCAL_O_GROUPS, N_LOCAL_EXPERTS, TP_SIZE, add_parallel_arguments
from utils import SafeTensorReader, pack_nz, unpack_nz


CHECKPOINT = Path("/mnt/data/models/DeepSeek-V4.1-Flash-w8a8")
MAP_PATH = Path(__file__).resolve().parents[2] / "experiments" / "dsv41-flash-w8a8-a3" / "weight_map.json"

# model config
D = M.hidden_size
Q_LORA = M.q_lora_rank
HEAD_DIM = M.head_dim
INTER = M.moe_intermediate_size
LOCAL_VOCAB = M.vocab_size // TP_SIZE
LOCAL_Q_DIM = LOCAL_HEADS * HEAD_DIM
GROUP_IN = M.num_attention_heads // M.o_groups * M.head_dim
# The indexer is replicated over TP (32 heads per rank) so paged scores need
# no cross-rank reduction (INTERFACE.md exception).
INDEX_HEAD_COUNT = M.index_n_heads
LOCAL_INDEX_DIM = INDEX_HEAD_COUNT * M.index_head_dim
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc
HC_PAD_ROWS = 32

# Frozen-map checks.
EXPECTED_INCLUDED = 139664
EXPECTED_EXCLUDED = 3832
EXPECTED_EXPERT_TENSORS = 138240
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.(weight|weight_scale|weight_offset)$")

LAYER_SOURCE_SUFFIXES = (
    "attn_norm.weight",
    "ffn_norm.weight",
    "attn.q_norm.weight",
    "attn.kv_norm.weight",
    "attn.attn_sink",
    "attn.wq_a.weight",
    "attn.wq_a.weight_scale",
    "attn.wq_a.weight_offset",
    "attn.wq_b.weight",
    "attn.wq_b.weight_scale",
    "attn.wq_b.weight_offset",
    "attn.wkv.weight",
    "attn.wkv.weight_scale",
    "attn.wkv.weight_offset",
    "attn.wo_a.weight",
    "attn.wo_b.weight",
    "attn.compressor.wkv.weight",
    "attn.compressor.wgate.weight",
    "attn.compressor.norm.weight",
    "attn.indexer.wq_b.weight",
    "attn.indexer.wq_b.weight_scale",
    "attn.indexer.wq_b.weight_offset",
    "attn.indexer.wk.weight",
    "attn.indexer.weights_proj.weight",
    "attn.indexer.k_norm.weight",
    "ffn.gate.weight",
    "ffn.gate.bias",
    "ffn.gate.bias_vl",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w1.weight_scale",
    "ffn.shared_experts.w1.weight_offset",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w2.weight_scale",
    "ffn.shared_experts.w2.weight_offset",
    "ffn.shared_experts.w3.weight",
    "ffn.shared_experts.w3.weight_scale",
    "ffn.shared_experts.w3.weight_offset",
    "engram.q_weight",
    "engram.k_weight",
    "engram.wkv.weight",
    "engram.embed.weight",
    "engram.embed.scale",
)

OPTIONAL_SOURCES = {
    "comp_wkv": "attn.compressor.wkv.weight",
    "comp_wgate": "attn.compressor.wgate.weight",
    "comp_norm": "attn.compressor.norm.weight",
    "idx_wq_b": "attn.indexer.wq_b.weight",
    "idx_wq_b_scale": "attn.indexer.wq_b.weight_scale",
    "idx_weights_proj": "attn.indexer.weights_proj.weight",
    "idx_wk": "attn.indexer.wk.weight",
    "idx_k_norm": "attn.indexer.k_norm.weight",
    "engram_wkv": "engram.wkv.weight",
    "engram_qk": "engram.q_weight",
}

# Model-argument shapes and dtypes for one layer.
LAYER_ARG_CONTRACT = {
    "attn_norm_w": ((D,), torch.bfloat16),
    "ffn_norm_w": ((D,), torch.bfloat16),
    "q_norm_w": ((Q_LORA,), torch.bfloat16),
    "kv_norm_w": ((HEAD_DIM,), torch.bfloat16),
    "sink": ((LOCAL_HEADS,), torch.float32),
    "wq_a": ((D, Q_LORA), torch.int8),
    "wq_a_scale": ((Q_LORA,), torch.float32),
    "wq_b": ((Q_LORA, LOCAL_Q_DIM), torch.int8),
    "wq_b_scale": ((LOCAL_Q_DIM,), torch.float32),
    "wkv": ((D, HEAD_DIM), torch.int8),
    "wkv_scale": ((HEAD_DIM,), torch.float32),
    "wo_a": ((LOCAL_O_GROUPS, M.o_lora_rank, GROUP_IN), torch.bfloat16),
    "wo_b": ((D, LOCAL_O_GROUPS * M.o_lora_rank), torch.bfloat16),
    "hc_attn_fn": ((HC_PAD_ROWS, HC_DIM), torch.float32),
    "hc_attn_scale": ((3,), torch.float32),
    "hc_attn_base": ((MIX_HC,), torch.float32),
    "hc_ffn_fn": ((HC_PAD_ROWS, HC_DIM), torch.float32),
    "hc_ffn_scale": ((3,), torch.float32),
    "hc_ffn_base": ((MIX_HC,), torch.float32),
    "gate_w": ((M.n_routed_experts, D), torch.float32),
    "gate_b": ((M.n_routed_experts,), torch.float32),
    "shared_w1": ((D, INTER), torch.int8),
    "shared_w1_scale": ((INTER,), torch.float32),
    "shared_w3": ((D, INTER), torch.int8),
    "shared_w3_scale": ((INTER,), torch.float32),
    "shared_w2": ((INTER, D), torch.int8),
    "shared_w2_scale": ((D,), torch.float32),
    "comp_wkv": ((HEAD_DIM, D), torch.bfloat16),
    "comp_wgate": ((HEAD_DIM, D), torch.bfloat16),
    "comp_norm": ((HEAD_DIM,), torch.bfloat16),
    "idx_wq_b": ((Q_LORA, LOCAL_INDEX_DIM), torch.int8),
    "idx_wq_b_scale": ((LOCAL_INDEX_DIM,), torch.float32),
    "idx_weights_proj": ((D, M.index_n_heads), torch.bfloat16),
    "idx_wk": ((M.index_head_dim, HEAD_DIM), torch.bfloat16),
    "idx_k_norm": ((M.index_head_dim,), torch.bfloat16),
    "engram_wkv": (
        ((M.hc_mult + 1) * D, (M.engram_max_ngram_size - 1) * M.engram_n_heads * M.engram_head_dim),
        torch.bfloat16,
    ),
    "engram_qk": ((M.hc_mult, D), torch.float32),
}

EXPERT_ARG_CONTRACT = {
    "w1": ((N_LOCAL_EXPERTS, D, INTER), torch.int8),
    "w1_scale": ((N_LOCAL_EXPERTS, INTER), torch.float32),
    "w3": ((N_LOCAL_EXPERTS, D, INTER), torch.int8),
    "w3_scale": ((N_LOCAL_EXPERTS, INTER), torch.float32),
    "w2": ((N_LOCAL_EXPERTS, INTER, D), torch.int8),
    "w2_scale": ((N_LOCAL_EXPERTS, D), torch.float32),
}

TOP_LEVEL_CONTRACT = {
    "embed_weight": ((LOCAL_VOCAB, D), torch.bfloat16),
    "lm_head_weight": ((LOCAL_VOCAB, D), torch.bfloat16),
    "final_norm": ((D,), torch.bfloat16),
}

# Large-tensor sampling priority per layer (deduplicated against the present arguments).
SAMPLE_PRIORITY = (
    "wq_b",
    "wq_a",
    "idx_wq_b",
    "wo_a",
    "wo_b",
    "wkv",
    "shared_w1",
    "shared_w3",
    "shared_w2",
    "comp_wkv",
    "comp_wgate",
    "gate_w",
    "engram_wkv",
)


@dataclass(frozen=True)
class WeightLeaf:
    """One contiguous transformed payload at a byte offset in a rank argument."""

    argument: str
    rank: int
    byte_offset: int
    tensor: torch.Tensor
    rank_shape: tuple[int, ...]
    logical_prefix: tuple[int, ...] = ()
    source_names: tuple[str, ...] = ()

    def __post_init__(self):
        if type(self.rank) is not int or not 0 <= self.rank < EP_SIZE:
            raise ValueError("weight leaf rank is outside EP workers")
        if not self.rank_shape or any(type(dim) is not int or dim <= 0 for dim in self.rank_shape):
            raise ValueError("weight leaf rank shape must be concrete and positive")
        if self.tensor.device.type != "cpu" or not self.tensor.is_contiguous():
            raise ValueError("weight leaf must be CPU contiguous")
        if not self.tensor.numel():
            raise ValueError("weight leaf must not be empty")
        size = self.tensor.element_size()
        if type(self.byte_offset) is not int or self.byte_offset < 0 or self.byte_offset % size:
            raise ValueError("weight leaf byte offset must be nonnegative and dtype aligned")
        if self.byte_offset + self.tensor.numel() * size > math.prod(self.rank_shape) * size:
            raise ValueError("weight leaf exceeds rank argument")


class WeightLoader:
    """Materialize rank-local model arguments from the frozen checkpoint map."""

    def __init__(self, checkpoint: str | Path = CHECKPOINT, map_path: str | Path = MAP_PATH,
                 rank: int = 0, tp_rank: int | None = None):
        if type(rank) is not int or not 0 <= rank < EP_SIZE:
            raise ValueError(f"rank must be in [0, {EP_SIZE - 1}], got {rank}")
        self.checkpoint = Path(checkpoint)
        self.map_path = Path(map_path)
        self.rank = rank
        self.ep_rank = rank
        self.tp_rank = rank % TP_SIZE if tp_rank is None else tp_rank
        if type(self.tp_rank) is not int or not 0 <= self.tp_rank < TP_SIZE:
            raise ValueError(f"tp_rank must be in [0, {TP_SIZE - 1}], got {self.tp_rank}")
        self.reader = SafeTensorReader(self.checkpoint)
        self.frozen = json.loads(self.map_path.read_text())
        self.mappings = {item["checkpoint_name"]: item for item in self.frozen["mappings"] if item["included"]}
        self.consumed: set[str] = set()
        self.offsets_folded = 0

    # ------------------------------------------------------------------
    # Top-level arguments
    # ------------------------------------------------------------------
    def embed_weight(self) -> torch.Tensor:
        return self._vocab_slice("embed.weight")

    def head_weight(self) -> torch.Tensor:
        return self._vocab_slice("head.weight")

    def final_norm(self) -> torch.Tensor:
        return self._read("norm.weight").to(torch.bfloat16)

    def top_level_arguments(self) -> dict[str, torch.Tensor]:
        return {"embed_weight": self.embed_weight(), "lm_head_weight": self.head_weight(), "final_norm": self.final_norm(), "engram_rotation": self.engram_rotation()}

    def engram_rotation(self) -> torch.Tensor:
        """Load the repeated block32 global rotation used by the original-basis Engram gate."""
        import struct

        cached = getattr(self, "_engram_rotation", None)
        if cached is not None:
            return cached
        contract = json.loads((self.checkpoint / "config.json").read_text())["engram_rotation_config"]
        if contract.get("key_and_gate_basis") != "original" or contract.get("value_basis") != "quarot_global" or not contract.get("value_projection_rotated") or contract.get("runtime_delta_rotation"):
            raise ValueError("unsupported Engram rotation contract")
        path = self.checkpoint / "optional/quarot.safetensors"
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
            entry = header["global_rotation"]
            if entry["shape"] != [D, D]:
                raise ValueError("global rotation shape must match hidden size")
            dtype = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}.get(entry["dtype"])
            if dtype is None:
                raise ValueError("global rotation must have floating-point dtype")
            begin, end = entry["data_offsets"]
            if end - begin != D * D * torch.empty((), dtype=dtype).element_size():
                raise ValueError("global rotation byte extent does not match shape")
            handle.seek(8 + header_size + begin)
            raw = handle.read(end - begin)
        rotation = torch.frombuffer(bytearray(raw), dtype=dtype).reshape(D, D)
        block = rotation[:32, :32].contiguous()
        for start in range(0, D, 32):
            stripe = rotation[start : start + 32]
            if not torch.equal(stripe[:, start : start + 32], block) or torch.count_nonzero(stripe[:, :start]) or torch.count_nonzero(stripe[:, start + 32:]):
                raise ValueError("Engram gate requires repeated block32 global rotation")
        if not torch.isfinite(block).all():
            raise ValueError("Engram rotation contains nonfinite values")
        self._engram_rotation = block.float().contiguous()
        return self._engram_rotation

    def engram_table(self, layer_id: int):
        """Return the host row accessor for one Engram layer table."""
        if layer_id not in M.engram_layer_ids:
            raise ValueError(f"layer {layer_id} has no Engram table")
        from engram import HostEngramTable

        return HostEngramTable(self.checkpoint, f"layers.{layer_id}.engram")

    # ------------------------------------------------------------------
    # Per-layer arguments
    # ------------------------------------------------------------------
    def layer_argument(self, layer_id: int, name: str) -> torch.Tensor:
        """Materialize one named dense/source argument without a layer payload bundle."""
        self._check_layer_id(layer_id)
        ln = lambda suffix: f"layers.{layer_id}.{suffix}"
        quantized = {
            "wq_a": "attn.wq_a.weight", "wq_b": "attn.wq_b.weight", "wkv": "attn.wkv.weight",
            "shared_w1": "ffn.shared_experts.w1.weight", "shared_w2": "ffn.shared_experts.w2.weight",
            "shared_w3": "ffn.shared_experts.w3.weight", "idx_wq_b": "attn.indexer.wq_b.weight",
        }
        base = name.removesuffix("_scale")
        if base in quantized:
            source = ln(quantized[base])
            start, count = 0, None
            if base == "wq_b":
                start, count = self.tp_rank * LOCAL_Q_DIM, LOCAL_Q_DIM
            elif base == "idx_wq_b":
                start, count = 0, LOCAL_INDEX_DIM
            if name.endswith("_scale"):
                scale = self._read(source + "_scale").float().reshape(-1)
                return scale if count is None else scale[start : start + count].contiguous()
            if count is not None:
                return self._w8a8_sliced(source, start, count, transpose=True)[0]
            return self._w8a8(source, transpose=True)[0]
        direct = {
            "attn_norm_w": ("attn_norm.weight", torch.bfloat16),
            "ffn_norm_w": ("ffn_norm.weight", torch.bfloat16),
            "q_norm_w": ("attn.q_norm.weight", torch.bfloat16),
            "kv_norm_w": ("attn.kv_norm.weight", torch.bfloat16),
            "hc_attn_scale": ("hc_attn_scale", torch.float32), "hc_attn_base": ("hc_attn_base", torch.float32),
            "hc_ffn_scale": ("hc_ffn_scale", torch.float32), "hc_ffn_base": ("hc_ffn_base", torch.float32),
            "gate_w": ("ffn.gate.weight", torch.float32), "gate_b": ("ffn.gate.bias", torch.float32),
            "comp_wkv": ("attn.compressor.wkv.weight", torch.bfloat16),
            "comp_wgate": ("attn.compressor.wgate.weight", torch.bfloat16),
            "comp_norm": ("attn.compressor.norm.weight", torch.bfloat16),
            "idx_wk": ("attn.indexer.wk.weight", torch.bfloat16),
            "idx_k_norm": ("attn.indexer.k_norm.weight", torch.bfloat16),
            "engram_wkv": ("engram.wkv.weight", torch.bfloat16),
        }
        if name in direct:
            suffix, dtype = direct[name]
            return self._read(ln(suffix)).to(dtype)
        if name == "sink":
            return self._head_slice(ln("attn.attn_sink")).float()
        if name == "wo_a":
            return self._wo_a(ln("attn.wo_a.weight"))
        if name == "wo_b":
            return self._wo_b(ln("attn.wo_b.weight"))
        if name in ("hc_attn_fn", "hc_ffn_fn"):
            return self._hc_function(ln(name))
        if name == "idx_weights_proj":
            return self._read(ln("attn.indexer.weights_proj.weight")).transpose(0, 1).contiguous()
        if name == "engram_qk":
            return self._read(ln("engram.q_weight")).float() * self._read(ln("engram.k_weight")).float()
        raise KeyError(f"layer {layer_id} has no argument {name}")

    @staticmethod
    def argument_contract(argument, layer_count=None, *, phase="decode"):
        """Return rank-local shape and dtype metadata without reading weight payloads."""
        if phase not in ("decode", "prefill"):
            raise ValueError("invalid streaming phase")
        count = M.num_hidden_layers if layer_count is None else layer_count
        if type(count) is not int or not 0 < count <= M.num_hidden_layers:
            raise ValueError("invalid streaming layer count")
        if argument == "engram_rotation":
            return (32, 32), torch.float32
        if argument in TOP_LEVEL_CONTRACT:
            return TOP_LEVEL_CONTRACT[argument]
        if argument in EXPERT_ARG_CONTRACT:
            shape, dtype = EXPERT_ARG_CONTRACT[argument]
            return (count, *shape), dtype
        name = {"gate_weight": "gate_w", "gate_bias": "gate_b"}.get(argument, argument)
        for prefix, source_count in (("c2a_", 3), ("c1a_", 1), ("re_", 4)):
            if argument.startswith(prefix):
                name, count = argument[len(prefix):], source_count
                break
        if name in ("engram_wkv", "engram_qk"):
            count = len(M.engram_layer_ids)
        shape, dtype = LAYER_ARG_CONTRACT[name]
        if argument.startswith("c1a_") and phase == "prefill":
            return shape, dtype
        return (count, *shape), dtype

    def iter_argument_leaves(self, argument, rank_shape, dtype, *, with_engram=False, zero_chunk_bytes=64 << 20):
        """Yield one transformed leaf at a time in the concrete rank argument layout."""
        shape = tuple(rank_shape)
        if not shape or any(type(dim) is not int or dim <= 0 for dim in shape):
            raise ValueError("rank shape must contain positive concrete dimensions")
        itemsize = torch.empty((), dtype=dtype).element_size()
        if argument.startswith("engram_") and not with_engram:
            expected = (32, 32) if argument == "engram_rotation" else (
                (len(M.engram_layer_ids), *LAYER_ARG_CONTRACT[argument][0]) if argument in ("engram_wkv", "engram_qk") else None)
            expected_dtype = torch.float32 if argument == "engram_rotation" else (
                LAYER_ARG_CONTRACT[argument][1] if argument in ("engram_wkv", "engram_qk") else None)
            if shape != expected or dtype != expected_dtype:
                raise ValueError("disabled Engram resource does not match its ABI")
            if zero_chunk_bytes < itemsize:
                raise ValueError("zero chunk must hold at least one element")
            total = math.prod(shape)
            count = zero_chunk_bytes // itemsize
            for begin in range(0, total, count):
                value = torch.zeros(min(count, total - begin), dtype=dtype)
                yield WeightLeaf(argument, self.rank, begin * itemsize, value, shape)
                del value
            return
        if argument in TOP_LEVEL_CONTRACT or argument == "engram_rotation":
            expected, expected_dtype = self.argument_contract(argument)
            if shape != expected or dtype != expected_dtype:
                raise ValueError(f"{argument}: invalid top-level rank contract")
            if argument == "engram_rotation":
                value = self.engram_rotation()
            else:
                method = {"embed_weight": self.embed_weight, "lm_head_weight": self.head_weight,
                          "final_norm": self.final_norm}[argument]
                value = method()
            if tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"{argument}: leaf shape/dtype does not match rank contract")
            yield WeightLeaf(argument, self.rank, 0, value.contiguous(), shape)
            return
        if argument in EXPERT_ARG_CONTRACT:
            expected, expected_dtype = EXPERT_ARG_CONTRACT[argument]
            if len(shape) != len(expected) + 1 or shape[1:] != expected or dtype != expected_dtype:
                raise ValueError(f"{argument}: invalid expert rank contract")
            if shape[0] > M.num_hidden_layers:
                raise ValueError("expert layer count exceeds model")
            leaf_shape = expected[1:]
            leaf_bytes = math.prod(leaf_shape) * itemsize
            which = int(argument[1])
            for layer in range(shape[0]):
                for local in range(N_LOCAL_EXPERTS):
                    value = self.expert_scale(layer, local, which) if argument.endswith("_scale") else self.expert_tensor(layer, local, which)
                    if tuple(value.shape) != leaf_shape or value.dtype != dtype or not value.is_contiguous():
                        raise ValueError(f"{argument}: transformed expert leaf violates contract")
                    source = f"layers.{layer}.ffn.experts.{self.ep_rank * N_LOCAL_EXPERTS + local}.w{which}.weight"
                    offset = (layer * N_LOCAL_EXPERTS + local) * leaf_bytes
                    sources = (source + "_scale",) if argument.endswith("_scale") else (source, source + "_offset")
                    yield WeightLeaf(argument, self.rank, offset, value, shape, (layer, local), sources)
                    del value
            return
        alias = {"gate_weight": "gate_w", "gate_bias": "gate_b"}
        layers = M.engram_layer_ids if argument in ("engram_wkv", "engram_qk") else None
        name = alias.get(argument, argument)
        for prefix, source_layers in (("c2a_", (2, 8, 14)), ("c1a_", (20,)), ("re_", (24, 28, 32, 36))):
            if argument.startswith(prefix):
                name, layers = argument[len(prefix):], source_layers
                break
        if name not in LAYER_ARG_CONTRACT:
            raise KeyError(f"unknown streaming argument {argument}")
        expected, expected_dtype = LAYER_ARG_CONTRACT[name]
        singleton_c1 = argument.startswith("c1a_") and shape == expected
        if (not singleton_c1 and (len(shape) != len(expected) + 1 or shape[1:] != expected)) or dtype != expected_dtype:
            raise ValueError(f"{argument}: invalid layer rank contract")
        if layers is None:
            if shape[0] > M.num_hidden_layers:
                raise ValueError("layer count exceeds model")
            layers = range(shape[0])
        elif not singleton_c1 and shape[0] != len(layers):
            raise ValueError(f"{argument}: source-slot count mismatch")
        leaf_bytes = math.prod(expected) * itemsize
        for slot, layer in enumerate(layers):
            value = self.layer_argument(layer, name).contiguous()
            if tuple(value.shape) != expected or value.dtype != dtype:
                raise ValueError(f"{argument}: transformed layer leaf violates contract")
            yield WeightLeaf(argument, self.rank, slot * leaf_bytes, value, shape, (layer,))
            del value

    def layer_dense(self, layer_id: int, with_engram: bool = True) -> dict[str, torch.Tensor]:
        """Materialize replicated and TP-sliced dense arguments for one layer."""
        self._check_layer_id(layer_id)
        names = [name for name in LAYER_ARG_CONTRACT if name not in OPTIONAL_SOURCES]
        present = self.optional_arguments(layer_id)
        for trigger, group in (("comp_wkv", ("comp_wkv", "comp_norm")), ("comp_wgate", ("comp_wgate",)),
                               ("idx_wq_b", ("idx_wq_b", "idx_wq_b_scale")),
                               ("idx_weights_proj", ("idx_weights_proj",)), ("idx_wk", ("idx_wk", "idx_k_norm"))):
            if trigger in present:
                names.extend(group)
        if with_engram and "engram_wkv" in present:
            names.extend(("engram_wkv", "engram_qk"))
        return {name: self.layer_argument(layer_id, name) for name in names}

    def layer(self, layer_id: int) -> dict[str, torch.Tensor]:
        """Materialize every argument for one layer, including its routed-expert shard."""
        args = self.layer_dense(layer_id)
        args.update(self.expert_shard(layer_id))
        return args

    def materialize(self, layer_id: int, name: str) -> torch.Tensor:
        """Materialize one named layer argument (dense or routed-expert)."""
        if name in EXPERT_ARG_CONTRACT:
            return self.expert_shard(layer_id)[name]
        return self.layer_dense(layer_id)[name]

    def optional_arguments(self, layer_id: int) -> set[str]:
        """Return the optional argument names present at one layer."""
        self._check_layer_id(layer_id)
        present = set()
        for arg, suffix in OPTIONAL_SOURCES.items():
            if f"layers.{layer_id}.{suffix}" in self.reader.weight_map:
                present.add(arg)
        return present

    def stacked(self, names, layers=None) -> dict[str, torch.Tensor]:
        """Stack named arguments along a leading layer axis for the requested layers."""
        requested = list(names)
        layer_ids = list(range(M.num_hidden_layers)) if layers is None else list(layers)
        result = {name: [] for name in requested}
        for layer_id in layer_ids:
            dense = self.layer_dense(layer_id) if any(name not in EXPERT_ARG_CONTRACT for name in requested) else None
            experts = self.expert_shard(layer_id) if any(name in EXPERT_ARG_CONTRACT for name in requested) else None
            for name in requested:
                values = experts if name in EXPERT_ARG_CONTRACT else dense
                if values is None or name not in values:
                    raise KeyError(f"layer {layer_id} has no argument {name}")
                result[name].append(values[name])
        return {name: torch.stack(values) for name, values in result.items()}

    # ------------------------------------------------------------------
    # Routed experts (EP)
    # ------------------------------------------------------------------
    def expert_tensor(self, layer_id: int, local_expert: int, which: int) -> torch.Tensor:
        """Materialize one local routed expert matrix ``w1``/``w2``/``w3``."""
        self._check_layer_id(layer_id)
        self._check_local_expert(local_expert)
        global_expert = self.ep_rank * N_LOCAL_EXPERTS + local_expert
        source = f"layers.{layer_id}.ffn.experts.{global_expert}.w{which}.weight"
        weight = self._read(source)
        weight = self._fold_offset(weight, source + "_offset")
        return pack_nz(weight.transpose(0, 1).contiguous())

    def expert_scale(self, layer_id: int, local_expert: int, which: int) -> torch.Tensor:
        """Materialize one local routed expert per-channel scale."""
        self._check_layer_id(layer_id)
        self._check_local_expert(local_expert)
        global_expert = self.ep_rank * N_LOCAL_EXPERTS + local_expert
        source = f"layers.{layer_id}.ffn.experts.{global_expert}.w{which}.weight_scale"
        return self._read(source).float().reshape(-1)

    def expert_shard(self, layer_id: int) -> dict[str, torch.Tensor]:
        """Stack the routed experts owned by this EP rank."""
        columns = {"w1": [], "w1_scale": [], "w3": [], "w3_scale": [], "w2": [], "w2_scale": []}
        for local_expert in range(N_LOCAL_EXPERTS):
            for which in (1, 3, 2):
                columns[f"w{which}"].append(self.expert_tensor(layer_id, local_expert, which))
                columns[f"w{which}_scale"].append(self.expert_scale(layer_id, local_expert, which))
        return {name: torch.stack(values) for name, values in columns.items()}

    # ------------------------------------------------------------------
    # Schema coverage
    # ------------------------------------------------------------------
    def included_names(self) -> set[str]:
        """Return every included checkpoint tensor name from the frozen map and index."""
        names = set(self.mappings)
        for name in self.reader.weight_map:
            match = EXPERT_RE.match(name)
            if match and int(match.group(1)) < M.num_hidden_layers:
                names.add(name)
        return names

    def declared_names(self) -> set[str]:
        """Return every checkpoint tensor name the loader schema can consume."""
        names = {"embed.weight", "head.weight", "norm.weight"}
        for layer_id in range(M.num_hidden_layers):
            for suffix in LAYER_SOURCE_SUFFIXES:
                name = f"layers.{layer_id}.{suffix}"
                if name in self.reader.weight_map:
                    names.add(name)
            for expert in range(M.n_routed_experts):
                for which in (1, 2, 3):
                    base = f"layers.{layer_id}.ffn.experts.{expert}.w{which}"
                    for tail in ("weight", "weight_scale", "weight_offset"):
                        name = f"{base}.{tail}"
                        if name in self.reader.weight_map:
                            names.add(name)
        return names

    def known_text_unused(self) -> set[str]:
        """Return included text-backbone tensors that the text-only entry does not take."""
        return {f"layers.{layer_id}.ffn.gate.bias_vl" for layer_id in range(M.num_hidden_layers)}

    # ------------------------------------------------------------------
    # Tensor transforms
    # ------------------------------------------------------------------
    def _read(self, name: str) -> torch.Tensor:
        if name not in self.reader.weight_map:
            raise KeyError(f"missing checkpoint tensor {name}")
        self.consumed.add(name)
        return self.reader.tensor(name)

    def _check_layer_id(self, layer_id: int) -> None:
        if not 0 <= layer_id < M.num_hidden_layers:
            raise ValueError(f"layer_id must be in [0, {M.num_hidden_layers - 1}], got {layer_id}")

    def _check_local_expert(self, local_expert: int) -> None:
        if not 0 <= local_expert < N_LOCAL_EXPERTS:
            raise ValueError(f"local_expert must be in [0, {N_LOCAL_EXPERTS - 1}], got {local_expert}")

    def _vocab_slice(self, name: str) -> torch.Tensor:
        return self._read(name)[self.tp_rank * LOCAL_VOCAB : (self.tp_rank + 1) * LOCAL_VOCAB].contiguous()

    def _head_slice(self, name: str) -> torch.Tensor:
        return self._read(name)[self.tp_rank * LOCAL_HEADS : (self.tp_rank + 1) * LOCAL_HEADS].contiguous()

    def _fold_offset(self, weight: torch.Tensor, offset_name: str) -> torch.Tensor:
        """Apply ``q - offset`` and require the result to remain representable in INT8."""
        offset = self._read(offset_name).float().reshape(-1)
        self.offsets_folded += 1
        if not bool(torch.any(offset != 0)):
            return weight
        adjusted = weight.double() - offset.double().reshape(-1, 1)
        if not torch.equal(adjusted, adjusted.round()) or adjusted.min() < -128 or adjusted.max() > 127:
            raise ValueError(f"{offset_name}: nonzero offset is not representable after the INT8 fold")
        return adjusted.round().to(torch.int8)

    def _w8a8(self, name: str, transpose: bool) -> tuple[torch.Tensor, torch.Tensor]:
        weight = self._fold_offset(self._read(name), name + "_offset")
        scale = self._read(name + "_scale").float().reshape(-1)
        if transpose:
            weight = pack_nz(weight.transpose(0, 1).contiguous())
        return weight, scale

    def _w8a8_sliced(self, name: str, start: int, count: int, transpose: bool) -> tuple[torch.Tensor, torch.Tensor]:
        rows = slice(start, start + count)
        weight = self._read(name)[rows].contiguous()
        offset = self._read(name + "_offset").float().reshape(-1)[rows]
        self.offsets_folded += 1
        if bool(torch.any(offset != 0)):
            adjusted = weight.double() - offset.double().reshape(-1, 1)
            if not torch.equal(adjusted, adjusted.round()) or adjusted.min() < -128 or adjusted.max() > 127:
                raise ValueError(f"{name}_offset: nonzero offset is not representable after the INT8 fold")
            weight = adjusted.round().to(torch.int8)
        scale = self._read(name + "_scale").float().reshape(-1)[rows].contiguous()
        if transpose:
            weight = pack_nz(weight.transpose(0, 1).contiguous())
        return weight, scale

    def _wo_a(self, name: str) -> torch.Tensor:
        rows = LOCAL_O_GROUPS * M.o_lora_rank
        start = self.tp_rank * rows
        local = self._read(name)[start : start + rows]
        return local.reshape(LOCAL_O_GROUPS, M.o_lora_rank, GROUP_IN).contiguous()

    def _wo_b(self, name: str) -> torch.Tensor:
        width = LOCAL_O_GROUPS * M.o_lora_rank
        start = self.tp_rank * width
        return self._read(name)[:, start : start + width].contiguous()

    def _hc_function(self, name: str) -> torch.Tensor:
        function = self._read(name).float()
        if function.shape[0] > HC_PAD_ROWS:
            raise ValueError(f"{name}: {function.shape[0]} rows exceed the {HC_PAD_ROWS}-row kernel tile")
        padding = torch.zeros(HC_PAD_ROWS - function.shape[0], function.shape[1], dtype=function.dtype)
        return torch.cat((function, padding), dim=0)

    # ------------------------------------------------------------------
    # Bounded host validation
    # ------------------------------------------------------------------
    def validate(self, layers: list[int], sample: int) -> list[str]:
        """Check sampled arguments against direct reads and the frozen contract."""
        errors: list[str] = []
        self._validate_frozen_map(errors)
        self._validate_coverage(errors)
        self._validate_top_level(errors)
        for layer_id in layers:
            self._validate_layer(layer_id, sample, errors)
        return errors

    def _validate_frozen_map(self, errors: list[str]) -> None:
        summary = self.frozen["summary"]
        if summary["included"] != EXPECTED_INCLUDED:
            errors.append(f"frozen included count {summary['included']} != {EXPECTED_INCLUDED}")
        if summary["excluded"] != EXPECTED_EXCLUDED:
            errors.append(f"frozen excluded count {summary['excluded']} != {EXPECTED_EXCLUDED}")
        if summary["unmapped"]:
            errors.append(f"frozen map has unmapped included tensors: {summary['unmapped'][:3]}")
        if summary["tensors_total"] != EXPECTED_INCLUDED + EXPECTED_EXCLUDED:
            errors.append(f"frozen tensor total {summary['tensors_total']} is inconsistent")
        expert_names = [name for name in self.included_names() if EXPERT_RE.match(name)]
        if len(expert_names) != EXPECTED_EXPERT_TENSORS:
            errors.append(f"expert tensor count {len(expert_names)} != {EXPECTED_EXPERT_TENSORS}")
        if self.frozen["expert_summary"]["count"] != EXPECTED_EXPERT_TENSORS:
            errors.append("expert_summary count is inconsistent with the index")
        for name, mapping in self.mappings.items():
            shape, _ = self.reader.shape_and_dtype(name)
            if list(shape) != mapping["source_shape"]:
                errors.append(f"{name}: header shape {list(shape)} != frozen {mapping['source_shape']}")

    def _validate_coverage(self, errors: list[str]) -> None:
        included = self.included_names()
        declared = self.declared_names()
        unused = included - declared - self.known_text_unused()
        if unused:
            errors.append(f"unused included tensors not declared by the loader: {sorted(unused)[:3]}")
        if len(included) != EXPECTED_INCLUDED:
            errors.append(f"included name count {len(included)} != {EXPECTED_INCLUDED}")
        print(
            f"[WEIGHT_LOADER] included={len(included)} declared={len(declared)} "
            f"text_unused={len(self.known_text_unused())} unexpected_unused={len(unused)}"
        )

    def _validate_top_level(self, errors: list[str]) -> None:
        arguments = self.top_level_arguments()
        for name, value in arguments.items():
            shape, dtype = TOP_LEVEL_CONTRACT[name]
            if tuple(value.shape) != shape or value.dtype != dtype:
                errors.append(f"{name}: got {tuple(value.shape)} {value.dtype}, want {shape} {dtype}")
        for source, arg in (("embed.weight", "embed_weight"), ("head.weight", "lm_head_weight")):
            start = self.tp_rank * LOCAL_VOCAB
            actual = arguments[arg]
            for offset in (0, LOCAL_VOCAB // 2, LOCAL_VOCAB - 1):
                direct = self._direct_rows(source, [start + offset])[0]
                if not torch.equal(actual[offset], direct):
                    errors.append(f"{arg}: TP row {start + offset} differs from the direct read")
        if not torch.equal(arguments["final_norm"], self.reader.tensor("norm.weight")):
            errors.append("final_norm: direct read mismatch")
        print(
            f"[WEIGHT_LOADER] top-level: embed/head rows={LOCAL_VOCAB} "
            f"tp_slice=[{self.tp_rank * LOCAL_VOCAB}, {(self.tp_rank + 1) * LOCAL_VOCAB})"
        )

    def _direct_rows(self, name: str, rows: list[int]) -> torch.Tensor:
        shape, dtype = self.reader.shape_and_dtype(name)
        if len(shape) != 2:
            raise ValueError(f"{name}: expected a 2-D table, got {shape}")
        element_size = torch.tensor([], dtype=dtype).element_size()
        row_bytes = shape[1] * element_size
        shard, header_size, header = self.reader.header(name)
        begin, _ = header[name]["data_offsets"]
        values = []
        with shard.open("rb") as handle:
            for row in rows:
                handle.seek(8 + header_size + begin + row * row_bytes)
                raw = handle.read(row_bytes)
                values.append(torch.frombuffer(bytearray(raw), dtype=dtype).reshape(shape[1]))
        return torch.stack(values)

    def _validate_layer(self, layer_id: int, sample: int, errors: list[str]) -> None:
        present = self.optional_arguments(layer_id)
        args = self.layer_dense(layer_id)
        expected_optional = set(present)
        for name, value in args.items():
            contract = LAYER_ARG_CONTRACT.get(name)
            if contract is None:
                errors.append(f"layer {layer_id}: no contract for argument {name}")
                continue
            shape, dtype = contract
            if tuple(value.shape) != shape or value.dtype != dtype:
                errors.append(f"layer {layer_id}.{name}: got {tuple(value.shape)} {value.dtype}, want {shape} {dtype}")
        for name in expected_optional:
            if name not in args:
                errors.append(f"layer {layer_id}: present source {OPTIONAL_SOURCES[name]} was not materialized")
        for name in set(OPTIONAL_SOURCES) - expected_optional:
            if name in args:
                errors.append(f"layer {layer_id}: absent source {OPTIONAL_SOURCES[name]} was materialized")
        self._validate_hc_padding(layer_id, args, errors)
        if "engram_qk" in args:
            self._validate_engram_layer(layer_id, args, errors)
        for name in self._sample_names(args, sample):
            self._validate_argument(layer_id, name, args[name], errors)
        self._validate_expert_shard(layer_id, sample, errors)
        print(f"[WEIGHT_LOADER] layer {layer_id}: args={len(args)} optional={sorted(expected_optional)}")

    def _validate_hc_padding(self, layer_id: int, args: dict[str, torch.Tensor], errors: list[str]) -> None:
        import hc

        if HC_PAD_ROWS != hc.MIX_PAD:
            errors.append(f"HC pad rows {HC_PAD_ROWS} != hc.MIX_PAD {hc.MIX_PAD}")
        for arg, source in (("hc_attn_fn", "hc_attn_fn"), ("hc_ffn_fn", "hc_ffn_fn")):
            value = args[arg]
            direct = self.reader.tensor(f"layers.{layer_id}.{source}").float()
            if not torch.equal(value[: direct.shape[0]], direct):
                errors.append(f"layer {layer_id}.{arg}: rows differ from the checkpoint")
            if torch.count_nonzero(value[direct.shape[0] :]) != 0:
                errors.append(f"layer {layer_id}.{arg}: padding tail is not zero")

    def _validate_engram_layer(self, layer_id: int, args: dict[str, torch.Tensor], errors: list[str]) -> None:
        q_weight = self.reader.tensor(f"layers.{layer_id}.engram.q_weight").float()
        k_weight = self.reader.tensor(f"layers.{layer_id}.engram.k_weight").float()
        if not torch.equal(args["engram_qk"], q_weight * k_weight):
            errors.append(f"layer {layer_id}.engram_qk: per-channel q*k fold mismatch")
        if not torch.equal(args["engram_wkv"], self.reader.tensor(f"layers.{layer_id}.engram.wkv.weight")):
            errors.append(f"layer {layer_id}.engram_wkv: direct read mismatch")
        slot = M.engram_layer_ids.index(layer_id)
        table = self.engram_table(layer_id)
        if table.rows != M.engram_num_embeddings[slot] or table.width != M.engram_head_dim:
            errors.append(f"layer {layer_id}: Engram table shape {(table.rows, table.width)} is inconsistent")
        row = table.rows // 2
        actual = table.row(row)
        expected = self._direct_engram_row(table, row)
        self.consumed.update({table.weight_name, table.scale_name})
        if not torch.equal(actual, expected):
            errors.append(f"layer {layer_id}: Engram sampled row {row} dequant mismatch")

    def _direct_engram_row(self, table, row: int) -> torch.Tensor:
        groups = table.width // ENGRAM_GROUP_SIZE
        shard, header_size, header = self.reader.header(table.weight_name)
        begin, _ = header[table.weight_name]["data_offsets"]
        with shard.open("rb") as handle:
            handle.seek(8 + header_size + begin + row * table.width)
            payload = torch.frombuffer(bytearray(handle.read(table.width)), dtype=torch.int8).reshape(1, table.width)
        shard_scale, header_scale, header_values = self.reader.header(table.scale_name)
        begin_scale, _ = header_values[table.scale_name]["data_offsets"]
        with shard_scale.open("rb") as handle:
            handle.seek(8 + header_scale + begin_scale + row * groups * 4)
            scales = torch.frombuffer(bytearray(handle.read(groups * 4)), dtype=torch.float32).reshape(1, groups)
        values = payload.float().reshape(1, groups, ENGRAM_GROUP_SIZE) * scales.float().reshape(1, groups, 1)
        return values.reshape(1, table.width).to(torch.bfloat16)[0]

    def _sample_names(self, args: dict[str, torch.Tensor], sample: int) -> list[str]:
        """Validate every small tensor and ``sample`` large tensors per layer."""
        small = [name for name in sorted(args) if args[name].numel() <= 1_000_000]
        large = [name for name in sorted(args) if args[name].numel() > 1_000_000]
        picked = [name for name in SAMPLE_PRIORITY if name in large][:sample]
        for name in large:
            if len(picked) >= sample:
                break
            if name not in picked:
                picked.append(name)
        return small + picked

    def _validate_argument(self, layer_id: int, name: str, actual: torch.Tensor, errors: list[str]) -> None:
        expected, packed = self._reference_argument(layer_id, name)
        compared = unpack_nz(actual) if packed else actual
        if not torch.equal(compared, expected):
            errors.append(f"layer {layer_id}.{name}: direct-read comparison mismatch")

    def _reference_argument(self, layer_id: int, name: str) -> tuple[torch.Tensor, bool]:
        """Recompute one argument from direct reads; ``packed`` requests NZ unpacking."""
        ln = lambda suffix: f"layers.{layer_id}.{suffix}"
        direct = self.reader.tensor
        if name == "attn_norm_w":
            return direct(ln("attn_norm.weight")).to(torch.bfloat16), False
        if name == "ffn_norm_w":
            return direct(ln("ffn_norm.weight")).to(torch.bfloat16), False
        if name == "q_norm_w":
            return direct(ln("attn.q_norm.weight")).to(torch.bfloat16), False
        if name == "kv_norm_w":
            return direct(ln("attn.kv_norm.weight")).to(torch.bfloat16), False
        if name == "sink":
            start = self.tp_rank * LOCAL_HEADS
            return direct(ln("attn.attn_sink"))[start : start + LOCAL_HEADS].float(), False
        if name == "wq_a":
            return direct(ln("attn.wq_a.weight")).transpose(0, 1).contiguous(), True
        if name == "wq_a_scale":
            return direct(ln("attn.wq_a.weight_scale")).float().reshape(-1), False
        if name == "wq_b":
            start = self.tp_rank * LOCAL_Q_DIM
            rows = direct(ln("attn.wq_b.weight"))[start : start + LOCAL_Q_DIM]
            return rows.transpose(0, 1).contiguous(), True
        if name == "wq_b_scale":
            start = self.tp_rank * LOCAL_Q_DIM
            return direct(ln("attn.wq_b.weight_scale")).float().reshape(-1)[start : start + LOCAL_Q_DIM], False
        if name == "wkv":
            return direct(ln("attn.wkv.weight")).transpose(0, 1).contiguous(), True
        if name == "wkv_scale":
            return direct(ln("attn.wkv.weight_scale")).float().reshape(-1), False
        if name == "wo_a":
            start = self.tp_rank * LOCAL_O_GROUPS * M.o_lora_rank
            rows = direct(ln("attn.wo_a.weight"))[start : start + LOCAL_O_GROUPS * M.o_lora_rank]
            return rows.reshape(LOCAL_O_GROUPS, M.o_lora_rank, GROUP_IN).contiguous(), False
        if name == "wo_b":
            start = self.tp_rank * LOCAL_O_GROUPS * M.o_lora_rank
            width = LOCAL_O_GROUPS * M.o_lora_rank
            return direct(ln("attn.wo_b.weight"))[:, start : start + width].contiguous(), False
        if name in ("hc_attn_fn", "hc_ffn_fn"):
            source = direct(ln(name)).float()
            padding = torch.zeros(HC_PAD_ROWS - source.shape[0], source.shape[1], dtype=source.dtype)
            return torch.cat((source, padding), dim=0), False
        if name in ("hc_attn_scale", "hc_ffn_scale", "hc_attn_base", "hc_ffn_base"):
            return direct(ln(name)).float(), False
        if name == "gate_w":
            return direct(ln("ffn.gate.weight")).float(), False
        if name == "gate_b":
            return direct(ln("ffn.gate.bias")).float(), False
        if name in ("shared_w1", "shared_w2", "shared_w3"):
            return direct(ln(f"ffn.shared_experts.{name[-2:]}.weight")).transpose(0, 1).contiguous(), True
        if name in ("shared_w1_scale", "shared_w2_scale", "shared_w3_scale"):
            return direct(ln(f"ffn.shared_experts.{name[7:9]}.weight_scale")).float().reshape(-1), False
        if name == "comp_wkv":
            return direct(ln("attn.compressor.wkv.weight")).to(torch.bfloat16), False
        if name == "comp_wgate":
            return direct(ln("attn.compressor.wgate.weight")).to(torch.bfloat16), False
        if name == "comp_norm":
            return direct(ln("attn.compressor.norm.weight")).to(torch.bfloat16), False
        if name == "idx_wq_b":
            rows = direct(ln("attn.indexer.wq_b.weight"))[0:LOCAL_INDEX_DIM]
            return rows.transpose(0, 1).contiguous(), True
        if name == "idx_wq_b_scale":
            return direct(ln("attn.indexer.wq_b.weight_scale")).float().reshape(-1)[0:LOCAL_INDEX_DIM], False
        if name == "idx_weights_proj":
            return direct(ln("attn.indexer.weights_proj.weight")).transpose(0, 1).contiguous(), False
        if name == "idx_wk":
            return direct(ln("attn.indexer.wk.weight")).to(torch.bfloat16), False
        if name == "idx_k_norm":
            return direct(ln("attn.indexer.k_norm.weight")).to(torch.bfloat16), False
        if name == "engram_wkv":
            return direct(ln("engram.wkv.weight")).to(torch.bfloat16), False
        if name == "engram_qk":
            return direct(ln("engram.q_weight")).float() * direct(ln("engram.k_weight")).float(), False
        raise KeyError(f"no direct-read reference for {name}")

    def _validate_expert_shard(self, layer_id: int, sample: int, errors: list[str]) -> None:
        for expert in range(M.n_routed_experts):
            owner = expert // N_LOCAL_EXPERTS
            local = expert % N_LOCAL_EXPERTS
            if owner >= EP_SIZE or local >= N_LOCAL_EXPERTS:
                errors.append(f"expert {expert} has no EP owner")
        for name, (shape, _) in EXPERT_ARG_CONTRACT.items():
            if shape[0] != N_LOCAL_EXPERTS:
                errors.append(f"expert contract {name} leading axis {shape[0]} != {N_LOCAL_EXPERTS}")
        local_sample = [0, N_LOCAL_EXPERTS // 2, N_LOCAL_EXPERTS - 1]
        for local in local_sample[: max(1, min(sample, len(local_sample)))]:
            for which in (1, 3, 2):
                actual = self.expert_tensor(layer_id, local, which)
                expected_shape = EXPERT_ARG_CONTRACT[f"w{which}"][0][1:]
                if tuple(actual.shape) != expected_shape:
                    errors.append(f"layer {layer_id} expert {local} w{which}: {tuple(actual.shape)} != {expected_shape}")
                global_expert = self.ep_rank * N_LOCAL_EXPERTS + local
                source = f"layers.{layer_id}.ffn.experts.{global_expert}.w{which}.weight"
                raw = self.reader.tensor(source).double()
                offset = self.reader.tensor(source + "_offset").double().reshape(-1, 1)
                expected = (raw - offset).to(torch.int8).transpose(0, 1).contiguous()
                if not torch.equal(unpack_nz(actual), expected):
                    errors.append(f"layer {layer_id} expert {global_expert} w{which}: direct-read mismatch")
                scale = self.expert_scale(layer_id, local, which)
                expected_scale = self.reader.tensor(source + "_scale").float().reshape(-1)
                if tuple(scale.shape) != EXPERT_ARG_CONTRACT[f"w{which}_scale"][0][1:]:
                    errors.append(f"layer {layer_id} expert {global_expert} w{which}: scale shape mismatch")
                if not torch.equal(scale, expected_scale):
                    errors.append(f"layer {layer_id} expert {global_expert} w{which}: scale mismatch")


def _parse_layers(text: str) -> list[int]:
    layers = [int(item) for item in text.split(",") if item]
    if not layers:
        raise ValueError("--layers must name at least one layer")
    for layer in layers:
        if not 0 <= layer < M.num_hidden_layers:
            raise ValueError(f"layer {layer} is outside [0, {M.num_hidden_layers - 1}]")
    return layers


def main() -> int:
    parser = argparse.ArgumentParser()
    add_parallel_arguments(parser)
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--map", dest="map_path", default=str(MAP_PATH))
    parser.add_argument("--layers", default="0,1,20,39")
    parser.add_argument("--sample", type=int, default=4)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tp-rank", type=int, default=None)
    args = parser.parse_args()

    loader = WeightLoader(args.checkpoint, args.map_path, rank=args.rank, tp_rank=args.tp_rank)
    print(
        f"[WEIGHT_LOADER] checkpoint={args.checkpoint} rank={loader.rank} "
        f"tp_rank={loader.tp_rank} ep_rank={loader.ep_rank}"
    )
    errors = loader.validate(_parse_layers(args.layers), args.sample)
    declared = loader.declared_names()
    undeclared = loader.consumed - declared
    if undeclared:
        errors.append(f"consumed tensors outside the declared schema: {sorted(undeclared)[:3]}")
    print(
        f"[WEIGHT_LOADER] consumed={len(loader.consumed)} offsets_folded={loader.offsets_folded} "
        f"not_consumed_this_run={len(declared - loader.consumed)}"
    )
    if errors:
        for error in errors:
            print(f"[WEIGHT_LOADER] FAIL {error}")
        raise SystemExit(1)
    print("[WEIGHT_LOADER] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
