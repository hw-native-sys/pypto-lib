# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read selected official checkpoint weights in the prefill device ABI.

Only requested tensors are copied from the sharded safetensors checkpoint. The
checkpoint stores linear weights output-major and uses one E8M0 scale per 32x32
block; the A5 attention entries consume input-major MX_B_NN matrices.
"""

import json
import math
import struct
from pathlib import Path

import torch

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.quantization import pack_mx_b_scale


_DTYPES = {
    "F32": torch.float32,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E8M0": torch.float8_e8m0fnu,
    "I8": torch.int8,
}


def _stack(rows):
    if rows[0].dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
        return torch.stack([row.contiguous().view(torch.uint8) for row in rows]).view(rows[0].dtype)
    return torch.stack(rows)


class PrefillCheckpoint:
    """Load text-layer tensors on demand from an official V4.1-Flash checkpoint."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        config = json.loads((self.root / "config.json").read_text())
        text = config["text_config"]
        quant = config.get("quantization_config", {})
        if quant.get("quant_method") != "fp8" or quant.get("weight_block_size") != [32, 32]:
            raise ValueError("checkpoint must use the official block-32 FP8/FP4 format")
        required = {
            "hidden_size": C.D,
            "num_hidden_layers": C.FLASH.num_hidden_layers,
            "num_attention_heads": C.H,
            "head_dim": C.HEAD_DIM,
            "sliding_window": C.FLASH.sliding_window,
            "kv_source_layer_ids": list(C.FLASH.kv_source_layer_ids),
            "index_source_layer_ids": list(C.FLASH.index_source_layer_ids),
        }
        for name, expected in required.items():
            if text.get(name) != expected:
                raise ValueError(f"checkpoint {name}={text.get(name)!r}, expected {expected!r}")
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map = index["weight_map"]
        self._headers = {}

    def tensor(self, name: str) -> torch.Tensor:
        """Return an owned CPU tensor without loading the rest of its shard."""
        shard = self.weight_map[name]
        path = (self.root / shard).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"checkpoint shard leaves its root: {shard}")
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

    def mx_linear(self, stem: str, *, output_range=None, input_range=None):
        """Transpose checkpoint FP8 [out,in] and expand block scales for MX_B_NN."""
        weight = self.tensor(stem + ".weight")
        scale = self.tensor(stem + ".scale")
        if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
            raise ValueError(f"{stem} is not block-32 MXFP8")
        if weight.shape[0] != scale.shape[0] * 32 or weight.shape[1] != scale.shape[1] * 32:
            raise ValueError(f"{stem} has inconsistent FP8 scales")
        if output_range is not None:
            first, last = output_range
            if first % 32 or last % 32 or not 0 <= first < last <= weight.shape[0]:
                raise ValueError("output shard must cover whole FP8 scale blocks")
            weight = weight[first:last]
            scale = scale[first // 32 : last // 32]
        if input_range is not None:
            first, last = input_range
            if first % 32 or last % 32 or not 0 <= first < last <= weight.shape[1]:
                raise ValueError("input shard must cover whole FP8 scale blocks")
            weight = weight[:, first:last]
            scale = scale[:, first // 32 : last // 32]
        logical_scale = scale.T.contiguous().view(torch.uint8).repeat_interleave(32, dim=1)
        return (
            weight.T.contiguous(),
            pack_mx_b_scale(logical_scale).view(torch.float8_e8m0fnu),
        )

    def _attention_weights(self, layer_id: int, world_size: int) -> dict[str, torch.Tensor]:
        """Stack common mHC and local-attention weights for TP/DP host entries."""
        if not 0 <= layer_id < C.FLASH.num_hidden_layers or world_size % C.TP_SIZE:
            raise ValueError("attention layer and TP/DP world size are required")
        stem = f"layers.{layer_id}."
        shared = {
            "hc_attn_fn": self.tensor(stem + "hc_attn_fn"),
            "hc_attn_scale": self.tensor(stem + "hc_attn_scale"),
            "hc_attn_base": self.tensor(stem + "hc_attn_base"),
            "attn_norm_weight": self.tensor(stem + "attn_norm.weight"),
            "q_norm_weight": self.tensor(stem + "attn.q_norm.weight"),
            "kv_norm_weight": self.tensor(stem + "attn.kv_norm.weight"),
        }
        shared["wq_a"], shared["wq_a_scale"] = self.mx_linear(stem + "attn.wq_a")
        shared["wkv"], shared["wkv_scale"] = self.mx_linear(stem + "attn.wkv")
        shards = {name: [value] * world_size for name, value in shared.items()}
        for name in ("wq_b", "wq_b_scale", "wo_a", "wo_b", "wo_b_scale", "attn_sink"):
            shards[name] = []
        sink = self.tensor(stem + "attn.attn_sink")
        # wo_a is a block-diagonal linear. It is dequantized to BF16 at load time,
        # matching the official inference model's grouped einsum.
        wo_a = self.tensor(stem + "attn.wo_a.weight")
        wo_a_scale = self.tensor(stem + "attn.wo_a.scale")
        for rank in range(world_size):
            tp_rank = rank % C.TP_SIZE
            first = tp_rank * C.LOCAL_H * C.HEAD_DIM
            weight, scale = self.mx_linear(
                stem + "attn.wq_b", output_range=(first, first + C.LOCAL_H * C.HEAD_DIM)
            )
            shards["wq_b"].append(weight)
            shards["wq_b_scale"].append(scale)
            shards["attn_sink"].append(sink[tp_rank * C.LOCAL_H : (tp_rank + 1) * C.LOCAL_H].contiguous())
            first = tp_rank * C.LOCAL_O_WIDTH
            weight, scale = self.mx_linear(stem + "attn.wo_b", input_range=(first, first + C.LOCAL_O_WIDTH))
            shards["wo_b"].append(weight)
            shards["wo_b_scale"].append(scale)
            group_first = tp_rank * C.LOCAL_O_GROUPS
            row_first = group_first * C.O_LORA
            row_last = row_first + C.LOCAL_O_WIDTH
            block_scale = wo_a_scale[row_first // 32 : row_last // 32].float()
            block_scale = block_scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
            groups = (wo_a[row_first:row_last].float() * block_scale).to(torch.bfloat16)
            shards["wo_a"].append(groups.reshape(C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN))
        return {name: _stack(rows) for name, rows in shards.items()}

    def swa_attention_weights(self, layer_id: int, world_size: int) -> dict[str, torch.Tensor]:
        """Stack checkpoint layer 0/1 mHC and SWA weights for TP/DP host entries."""
        if layer_id not in (0, 1):
            raise ValueError("SWA weights belong to layers 0 and 1")
        return self._attention_weights(layer_id, world_size)

    def c2a_full_weights(self, layer_id: int, world_size: int) -> dict[str, torch.Tensor]:
        """Stack an encoder C2A source layer's attention and ratio-two compressor."""
        if layer_id not in (2, 8, 14):
            raise ValueError("encoder C2A Full weights belong to layers 2, 8, and 14")
        stem = f"layers.{layer_id}.attn."
        values = self._attention_weights(layer_id, world_size)
        shared = {
            "compressor_wkv": self.tensor(stem + "compressor.wkv.weight").T.float().contiguous(),
            "compressor_wgate": self.tensor(stem + "compressor.wgate.weight").T.float().contiguous(),
            "compressor_norm_weight": self.tensor(stem + "compressor.norm.weight"),
            "index_wk": self.tensor(stem + "indexer.wk.weight").T.contiguous(),
            "index_norm_weight": self.tensor(stem + "indexer.k_norm.weight"),
        }
        values.update({name: _stack([value] * world_size) for name, value in shared.items()})
        index_wq, index_scale = self.mx_linear(stem + "indexer.wq_b")
        index_projection = self.tensor(stem + "indexer.weights_proj.weight").T.contiguous()
        values["index_wq_b"] = _stack([index_wq] * world_size)
        values["index_wq_b_scale"] = _stack([index_scale] * world_size)
        values["index_weights_proj"] = _stack([index_projection] * world_size)
        return values

    def c2a_reuse_weights(self, layer_id: int, world_size: int) -> dict[str, torch.Tensor]:
        """Stack a ratio-two Reuse layer's attention weights; global KV comes from its source."""
        layer = C.FLASH.layer_config(layer_id)
        if layer_id >= 20 or layer.mode != C.AttentionMode.REUSE:
            raise ValueError("encoder C2A Reuse weights require a layer in 3–7, 9–13, or 15–19")
        return self._attention_weights(layer_id, world_size)

    def c1a_full_weights(self, world_size: int) -> dict[str, torch.Tensor]:
        """Stack layer-20 mHC, attention, compressor, and indexer weights."""
        if world_size != C.TP_SIZE:
            raise ValueError("C1A Full host requires one TP group")
        stem = "layers.20.attn."
        values = self._attention_weights(20, world_size)
        shared = {
            "compressor_wkv": self.tensor(stem + "compressor.wkv.weight").T.contiguous(),
            "compressor_norm_weight": self.tensor(stem + "compressor.norm.weight"),
            "index_wk": self.tensor(stem + "indexer.wk.weight").T.contiguous(),
            "index_norm_weight": self.tensor(stem + "indexer.k_norm.weight"),
        }
        values.update({name: _stack([value] * world_size) for name, value in shared.items()})
        # The sparse indexer scores all 32 index heads on each TP rank.
        index_wq, index_scale = self.mx_linear(stem + "indexer.wq_b")
        index_projection = self.tensor(stem + "indexer.weights_proj.weight").T.contiguous()
        values["index_wq_b"] = _stack([index_wq] * world_size)
        values["index_wq_b_scale"] = _stack([index_scale] * world_size)
        values["index_weights_proj"] = _stack([index_projection] * world_size)
        return values

    def decoder_global_weights(self, world_size: int) -> dict[str, torch.Tensor]:
        """Stack layer-20 weights for projection from final encoder rows."""
        stem = "layers.20."
        shared = {
            "attn_norm_weight": self.tensor(stem + "attn_norm.weight"),
            "compressor_wkv": self.tensor(stem + "attn.compressor.wkv.weight").T.contiguous(),
            "compressor_norm_weight": self.tensor(stem + "attn.compressor.norm.weight"),
            "index_wk": self.tensor(stem + "attn.indexer.wk.weight").T.contiguous(),
            "index_norm_weight": self.tensor(stem + "attn.indexer.k_norm.weight"),
        }
        return {name: _stack([value] * world_size) for name, value in shared.items()}


def bind_checkpoint_weights(specs, values: dict[str, torch.Tensor]) -> None:
    """Replace fixture weight inputs after checking the host entry's ABI."""
    found = set()
    for spec in specs:
        if spec.name in values:
            value = values[spec.name]
            if list(value.shape) != spec.shape or value.dtype != spec.dtype:
                raise ValueError(
                    f"checkpoint {spec.name}: {tuple(value.shape)} {value.dtype} does not match "
                    f"{tuple(spec.shape)} {spec.dtype}"
                )
            spec.init_value = value
            found.add(spec.name)
    if found != values.keys():
        raise ValueError(f"checkpoint host entry lacks inputs {sorted(values.keys() - found)}")
