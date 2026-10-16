# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run both fixed Qwen3.8 layer contracts through complete PyPTO layer entries."""

import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from pypto.runtime import RunConfig
from safetensors import safe_open
from safetensors.torch import save_file

from decode_layer import decode_layer
from prefill_layer import prefill_layer

TOKEN_IDS = [760, 4087, 369, 220, 19, 17]
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
EMBEDDING = "model.language_model.embed_tokens.weight"
HIDDEN = 5120
TOKENS_PAD = 16
QKV = 10240
VALUE_HEADS = 48
HEAD_DIM = 128
KV_WIDTH = 1024
MLP_WIDTH = 17408
ROPE_THETA = 10_000_000

WEIGHT_ORDER = (
    "norm_weight",
    "qkv_weight",
    "conv_weight",
    "a_weight",
    "b_weight",
    "z_weight",
    "a_log",
    "dt_bias",
    "delta_norm_weight",
    "out_weight",
    "post_norm_weight",
    "mlp_gate_weight",
    "mlp_up_weight",
    "mlp_down_weight",
    "attn_q_weight",
    "attn_k_weight",
    "attn_v_weight",
    "attn_q_norm_weight",
    "attn_k_norm_weight",
    "attn_o_weight",
    "cos",
    "sin",
)

DUMMY_SHAPES = {
    "qkv_weight": (QKV, HIDDEN),
    "conv_weight": (4, QKV),
    "a_weight": (128, HIDDEN),
    "b_weight": (128, HIDDEN),
    "z_weight": (VALUE_HEADS * HEAD_DIM, HIDDEN),
    "a_log": (1, 128),
    "dt_bias": (1, 128),
    "delta_norm_weight": (1, HEAD_DIM),
    "out_weight": (HIDDEN, VALUE_HEADS * HEAD_DIM),
    "attn_q_weight": (2 * 24 * 256, HIDDEN),
    "attn_k_weight": (KV_WIDTH, HIDDEN),
    "attn_v_weight": (KV_WIDTH, HIDDEN),
    "attn_q_norm_weight": (1, 256),
    "attn_k_norm_weight": (1, 256),
    "attn_o_weight": (HIDDEN, 24 * 256),
}


def load_named_tensors(model_root: Path, names: list[str]) -> dict[str, torch.Tensor]:
    index_path = model_root / "model.safetensors.index.json"
    if hashlib.sha256(index_path.read_bytes()).hexdigest() != INDEX_SHA256:
        raise ValueError("checkpoint index differs from the frozen contract")
    index = json.loads(index_path.read_text())["weight_map"]
    loaded = {}
    for name in names:
        with safe_open(model_root / index[name], framework="pt", device="cpu") as source:
            if name == EMBEDDING:
                loaded[name] = source.get_slice(name)[TOKEN_IDS]
            else:
                loaded[name] = source.get_tensor(name)
    return loaded


def padded_rows(value: torch.Tensor, rows: int = 128) -> torch.Tensor:
    padded = torch.zeros((rows, *value.shape[1:]), dtype=value.dtype)
    padded[: value.shape[0]] = value
    return padded


def rope_angles(positions: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    cos = torch.ones((TOKENS_PAD, 64), dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    for row, position in enumerate(positions):
        angles = [position / (ROPE_THETA ** (2 * i / 64)) for i in range(32)]
        cos[row] = torch.tensor([math.cos(angle) for angle in angles] * 2, dtype=torch.bfloat16)
        sin[row] = torch.tensor([math.sin(angle) for angle in angles] * 2, dtype=torch.bfloat16)
    return cos, sin


def load_layer_weights(model_root: Path, layer: int) -> dict[str, torch.Tensor]:
    prefix = f"model.language_model.layers.{layer}."
    common = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    delta = [
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.conv1d.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.norm.weight",
        "linear_attn.out_proj.weight",
    ]
    attention = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.o_proj.weight",
    ]
    loaded = load_named_tensors(
        model_root, [prefix + name for name in common + (delta if layer == 0 else attention)]
    )

    def get(suffix: str) -> torch.Tensor:
        return loaded[prefix + suffix]

    weights = {
        "norm_weight": get("input_layernorm.weight").reshape(1, HIDDEN),
        "post_norm_weight": get("post_attention_layernorm.weight").reshape(1, HIDDEN),
        "mlp_gate_weight": get("mlp.gate_proj.weight"),
        "mlp_up_weight": get("mlp.up_proj.weight"),
        "mlp_down_weight": get("mlp.down_proj.weight"),
    }
    if layer == 0:
        weights.update(
            {
                "qkv_weight": get("linear_attn.in_proj_qkv.weight"),
                "conv_weight": get("linear_attn.conv1d.weight").squeeze(1).T.contiguous(),
                "a_weight": padded_rows(get("linear_attn.in_proj_a.weight")),
                "b_weight": padded_rows(get("linear_attn.in_proj_b.weight")),
                "z_weight": get("linear_attn.in_proj_z.weight"),
                "a_log": padded_rows(get("linear_attn.A_log").reshape(-1, 1)).T.contiguous(),
                "dt_bias": padded_rows(get("linear_attn.dt_bias").reshape(-1, 1)).T.contiguous(),
                "delta_norm_weight": get("linear_attn.norm.weight").reshape(1, HEAD_DIM),
                "out_weight": get("linear_attn.out_proj.weight"),
            }
        )
    else:
        weights.update(
            {
                "attn_q_weight": get("self_attn.q_proj.weight"),
                "attn_k_weight": get("self_attn.k_proj.weight"),
                "attn_v_weight": get("self_attn.v_proj.weight"),
                "attn_q_norm_weight": get("self_attn.q_norm.weight").reshape(1, 256),
                "attn_k_norm_weight": get("self_attn.k_norm.weight").reshape(1, 256),
                "attn_o_weight": get("self_attn.o_proj.weight"),
            }
        )
    placeholders = {}
    for name, shape in DUMMY_SHAPES.items():
        if name not in weights:
            placeholders.setdefault(shape, torch.zeros(shape, dtype=torch.bfloat16))
            weights[name] = placeholders[shape]
    return weights


def result_buffers() -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((TOKENS_PAD, HIDDEN), dtype=torch.bfloat16),
        torch.empty((TOKENS_PAD, QKV), dtype=torch.bfloat16),
        torch.empty((VALUE_HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32),
        torch.empty((TOKENS_PAD, KV_WIDTH), dtype=torch.bfloat16),
        torch.empty((TOKENS_PAD, KV_WIDTH), dtype=torch.bfloat16),
    )


def serialize_step(
    outputs: dict[str, torch.Tensor],
    layer: int,
    step: str,
    active: int,
    state: tuple[torch.Tensor, ...],
) -> None:
    hidden, conv, recurrent, key, value = state
    prefix = f"layer_{layer}/{step}/"
    outputs[prefix + "hidden"] = hidden[:active].unsqueeze(0).contiguous()
    if layer == 0:
        outputs[prefix + "conv"] = conv[:4].T.contiguous().reshape(1, QKV, 4)
        outputs[prefix + "recurrent"] = recurrent.unsqueeze(0).contiguous()
    else:
        length = 4 if step == "prefill" else 4 + int(step[-1])
        outputs[prefix + "key"] = key[:length].view(length, 4, 256).permute(1, 0, 2).unsqueeze(0).contiguous()
        outputs[prefix + "value"] = (
            value[:length].view(length, 4, 256).permute(1, 0, 2).unsqueeze(0).contiguous()
        )


def run_layer(
    model_root: Path,
    embeddings: torch.Tensor,
    layer: int,
    device: int,
    build_root: Path,
) -> dict[str, torch.Tensor]:
    weights = load_layer_weights(model_root, layer)
    kind = 0 if layer == 0 else 1
    prefill_config = RunConfig(
        platform="a2a3", device_id=device, save_kernels_dir=str(build_root / f"layer_{layer}_prefill")
    )
    decode_config = RunConfig(
        platform="a2a3", device_id=device, save_kernels_dir=str(build_root / f"layer_{layer}_decode")
    )
    prefill_program = prefill_layer.compile(layer_kind=kind, config=prefill_config)
    decode_program = decode_layer.compile(layer_kind=kind, config=decode_config)
    outputs = {}
    x = torch.zeros((TOKENS_PAD, HIDDEN), dtype=torch.bfloat16)
    x[:4] = embeddings[:4]
    weights["cos"], weights["sin"] = rope_angles((0, 1, 2, 3))
    state = result_buffers()
    prefill_program(x, *(weights[name] for name in WEIGHT_ORDER), *state, config=prefill_config)
    serialize_step(outputs, layer, "prefill", 4, state)

    for step, position in ((1, 4), (2, 5)):
        x.zero_()
        x[0] = embeddings[position]
        weights["cos"], weights["sin"] = rope_angles((position,))
        next_state = result_buffers()
        decode_program(
            x,
            *(weights[name] for name in WEIGHT_ORDER),
            *state[1:],
            ctypes.c_int64(position),
            *next_state,
            config=decode_config,
        )
        state = next_state
        serialize_step(outputs, layer, f"decode_{step}", 1, state)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    device = int(os.environ["TASK_DEVICE"])
    model_root = Path(os.environ["MODEL_ROOT"])
    workspace_root = Path(__file__).resolve().parents[4]
    build_root = workspace_root / ".gp" / "c001-build" / "contract_runner"
    build_root.mkdir(parents=True, exist_ok=True)
    embeddings = load_named_tensors(model_root, [EMBEDDING])[EMBEDDING]
    outputs = {}
    for layer in (0, 3):
        outputs.update(run_layer(model_root, embeddings, layer, device, build_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    save_file(outputs, args.output)
    args.metadata.write_text(
        json.dumps(
            {
                "model_revision": REVISION,
                "index_sha256": INDEX_SHA256,
                "token_ids": TOKEN_IDS,
                "prefill_token_count": 4,
                "decode_token_count": 2,
                "execution_backend": "pypto",
                "task_device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()
