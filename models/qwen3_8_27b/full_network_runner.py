# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run complete PyPTO Qwen3.8 text prefill and decode against the real checkpoint."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
from contextlib import ExitStack
from pathlib import Path

import torch
from pypto.runtime import RunConfig
from safetensors import safe_open
from transformers import AutoTokenizer

from decode_fwd import decode_fwd
from full_network_core import (
    ATTN_HEAD_DIM,
    ATTN_LAYERS,
    DELTA_LAYERS,
    HEAD_DIM,
    HIDDEN,
    KV_WIDTH,
    MAX_SEQUENCE,
    MLP_WIDTH,
    NUM_LAYERS,
    QKV,
    Q_WIDTH,
    ROTARY_DIM,
    TOKENS_PAD,
    VALUE_HEADS,
    VALUE_WIDTH,
    VOCAB,
)
from prefill_fwd import prefill_fwd

REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONFIG_SHA256 = "191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"
INDEX_SHA256 = "77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TEXT_WEIGHT_BYTES = 53_791_996_928
PREFIX = "model.language_model."

WEIGHT_ORDER = (
    "embedding", "input_norms", "post_norms", "mlp_gates", "mlp_ups", "mlp_downs",
    "delta_qkvs", "delta_convs", "delta_as", "delta_bs", "delta_zs", "delta_a_logs",
    "delta_dt_biases", "delta_norms", "delta_outs", "attn_qs", "attn_ks", "attn_vs",
    "attn_q_norms", "attn_k_norms", "attn_outs", "final_norm", "lm_head",
)

COMMON_BANKS = {
    "input_norms": ("input_layernorm.weight", (1, HIDDEN)),
    "post_norms": ("post_attention_layernorm.weight", (1, HIDDEN)),
    "mlp_gates": ("mlp.gate_proj.weight", (MLP_WIDTH, HIDDEN)),
    "mlp_ups": ("mlp.up_proj.weight", (MLP_WIDTH, HIDDEN)),
    "mlp_downs": ("mlp.down_proj.weight", (HIDDEN, MLP_WIDTH)),
}
DELTA_BANKS = {
    "delta_qkvs": ("linear_attn.in_proj_qkv.weight", (QKV, HIDDEN)),
    "delta_convs": ("linear_attn.conv1d.weight", (4, QKV)),
    "delta_as": ("linear_attn.in_proj_a.weight", (128, HIDDEN)),
    "delta_bs": ("linear_attn.in_proj_b.weight", (128, HIDDEN)),
    "delta_zs": ("linear_attn.in_proj_z.weight", (VALUE_WIDTH, HIDDEN)),
    "delta_a_logs": ("linear_attn.A_log", (1, 128)),
    "delta_dt_biases": ("linear_attn.dt_bias", (1, 128)),
    "delta_norms": ("linear_attn.norm.weight", (1, HEAD_DIM)),
    "delta_outs": ("linear_attn.out_proj.weight", (HIDDEN, VALUE_WIDTH)),
}
ATTN_BANKS = {
    "attn_qs": ("self_attn.q_proj.weight", (2 * Q_WIDTH, HIDDEN)),
    "attn_ks": ("self_attn.k_proj.weight", (KV_WIDTH, HIDDEN)),
    "attn_vs": ("self_attn.v_proj.weight", (KV_WIDTH, HIDDEN)),
    "attn_q_norms": ("self_attn.q_norm.weight", (1, ATTN_HEAD_DIM)),
    "attn_k_norms": ("self_attn.k_norm.weight", (1, ATTN_HEAD_DIM)),
    "attn_outs": ("self_attn.o_proj.weight", (HIDDEN, Q_WIDTH)),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_checkpoint(root: Path) -> dict[str, str]:
    if _sha256(root / "config.json") != CONFIG_SHA256:
        raise ValueError("checkpoint config differs from frozen contract")
    if _sha256(root / "model.safetensors.index.json") != INDEX_SHA256:
        raise ValueError("checkpoint index differs from frozen contract")
    if _sha256(root / "tokenizer.json") != TOKENIZER_SHA256:
        raise ValueError("checkpoint tokenizer differs from frozen contract")
    metadata = root / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if metadata.read_text().splitlines()[0] != REVISION:
        raise ValueError("checkpoint revision differs from frozen contract")
    config = json.loads((root / "config.json").read_text())["text_config"]
    if (config["num_hidden_layers"], config["hidden_size"], config["intermediate_size"],
            config["vocab_size"], config["layer_types"]) != (
                NUM_LAYERS, HIDDEN, MLP_WIDTH, VOCAB,
                ["full_attention" if layer % 4 == 3 else "linear_attention" for layer in range(NUM_LAYERS)]
            ):
        raise ValueError("text configuration differs from the PyPTO weight layout")
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def _adapt_tensor(value: torch.Tensor, suffix: str, shape: tuple[int, ...]) -> torch.Tensor:
    if value.dtype != torch.bfloat16:
        raise ValueError(f"unexpected checkpoint dtype for {suffix}: {value.dtype}")
    if suffix == "linear_attn.conv1d.weight":
        value = value.squeeze(1).T
    elif value.ndim == 1 and len(shape) == 2 and shape[0] == 1:
        value = value.reshape(1, -1)
    if len(value.shape) != len(shape) or any(actual > wanted for actual, wanted in zip(value.shape, shape)):
        raise ValueError(f"weight shape mismatch for {suffix}: {tuple(value.shape)} vs {shape}")
    if suffix not in {"linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
                      "linear_attn.A_log", "linear_attn.dt_bias"} and tuple(value.shape) != shape:
        raise ValueError(f"unexpected padding for {suffix}: {tuple(value.shape)} vs {shape}")
    return value


def load_text_weight_banks(root: Path, weight_map: dict[str, str]) -> dict[str, torch.Tensor]:
    common_layers = list(range(NUM_LAYERS))
    delta_layers = [layer for layer in common_layers if layer % 4 != 3]
    attention_layers = [layer for layer in common_layers if layer % 4 == 3]
    if len(delta_layers) != DELTA_LAYERS or len(attention_layers) != ATTN_LAYERS:
        raise ValueError("text layer schedule changed")
    expected = {"lm_head.weight", PREFIX + "embed_tokens.weight", PREFIX + "norm.weight"}
    for layers, specs in ((common_layers, COMMON_BANKS), (delta_layers, DELTA_BANKS),
                          (attention_layers, ATTN_BANKS)):
        expected.update(f"{PREFIX}layers.{layer}.{suffix}" for layer in layers
                        for suffix, _ in specs.values())
    actual = {name for name in weight_map if name == "lm_head.weight" or name.startswith(PREFIX)}
    if expected != actual:
        raise ValueError(f"text weight coverage mismatch: missing={sorted(expected - actual)}, "
                         f"extra={sorted(actual - expected)}")

    with ExitStack() as stack:
        shards = {name: stack.enter_context(safe_open(root / name, framework="pt", device="cpu"))
                  for name in sorted({weight_map[key] for key in expected})}

        def source(name: str) -> torch.Tensor:
            result = shards[weight_map[name]].get_tensor(name)
            if result.dtype != torch.bfloat16:
                raise ValueError(f"unexpected checkpoint dtype for {name}: {result.dtype}")
            return result

        weights = {
            "embedding": source(PREFIX + "embed_tokens.weight"),
            "final_norm": source(PREFIX + "norm.weight").reshape(1, HIDDEN),
            "lm_head": source("lm_head.weight"),
        }
        if weights["embedding"].shape != (VOCAB, HIDDEN) or weights["lm_head"].shape != (VOCAB, HIDDEN):
            raise ValueError("embedding or untied LM-head shape changed")

        for layers, specs in ((common_layers, COMMON_BANKS), (delta_layers, DELTA_BANKS),
                              (attention_layers, ATTN_BANKS)):
            for bank_name, (suffix, shape) in specs.items():
                padded = suffix in {"linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight",
                                    "linear_attn.A_log", "linear_attn.dt_bias"}
                bank_shape = (len(layers), *shape)
                bank = (torch.zeros if padded else torch.empty)(bank_shape, dtype=torch.bfloat16)
                for row, layer in enumerate(layers):
                    value = _adapt_tensor(source(f"{PREFIX}layers.{layer}.{suffix}"), suffix, shape)
                    bank[(row, *(slice(0, width) for width in value.shape))].copy_(value)
                weights[bank_name] = bank

    if set(weights) != set(WEIGHT_ORDER):
        raise ValueError("PyPTO weight bank coverage is incomplete")
    bytes_loaded = sum(weight.numel() * weight.element_size() for weight in weights.values())
    if bytes_loaded < TEXT_WEIGHT_BYTES:
        raise ValueError("packed text weights are smaller than the real checkpoint")
    print(f"Loaded {len(expected)} real text tensors into {len(weights)} compact banks "
          f"({bytes_loaded} staged bytes)", flush=True)
    return weights


def rope_table() -> tuple[torch.Tensor, torch.Tensor]:
    angles = [[position / (10_000_000 ** (2 * dim / ROTARY_DIM))
               for dim in range(ROTARY_DIM // 2)] for position in range(MAX_SEQUENCE)]
    cos = torch.tensor([[math.cos(angle) for angle in row] * 2 for row in angles], dtype=torch.bfloat16)
    sin = torch.tensor([[math.sin(angle) for angle in row] * 2 for row in angles], dtype=torch.bfloat16)
    return cos, sin


def fresh_states() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros((DELTA_LAYERS, TOKENS_PAD, QKV), dtype=torch.bfloat16),
        torch.zeros((DELTA_LAYERS, VALUE_HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32),
        torch.zeros((ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH), dtype=torch.bfloat16),
        torch.zeros((ATTN_LAYERS, MAX_SEQUENCE, KV_WIDTH), dtype=torch.bfloat16),
    )


def run_program(program, ids: list[int], position: int, weights: tuple[torch.Tensor, ...],
                cos: torch.Tensor, sin: torch.Tensor, states: tuple[torch.Tensor, ...],
                config: RunConfig) -> torch.Tensor:
    if position >= MAX_SEQUENCE or len(ids) > MAX_SEQUENCE:
        raise ValueError("prompt exceeds the fixed 128-position whole-network ABI")
    token_ids = torch.zeros((MAX_SEQUENCE,), dtype=torch.int64)
    token_ids[:len(ids)] = torch.tensor(ids, dtype=torch.int64)
    logits = torch.empty((1, VOCAB), dtype=torch.float32)
    program(token_ids, ctypes.c_int64(position), *weights, cos, sin, *states, logits, config=config)
    return logits[0].clone()


def score(logits: torch.Tensor, golden_token: int, step: int, context_length: int) -> dict:
    values = logits.float()
    top = torch.topk(values, 2)
    return {
        "step": step,
        "context_length": context_length,
        "golden_token": golden_token,
        "golden_token_logprob": float(torch.log_softmax(values, dim=-1)[golden_token]),
        "top1_id": int(top.indices[0]),
        "top2_id": int(top.indices[1]),
        "top1_top2_margin": float(top.values[0] - top.values[1]),
    }


def run_case(name: str, frozen: dict, tokenizer, prefill_program, decode_program,
             weights: tuple[torch.Tensor, ...], cos: torch.Tensor, sin: torch.Tensor,
             prefill_config: RunConfig, decode_config: RunConfig) -> dict:
    prompt = frozen["prompt_token_ids"]
    continuation = frozen["golden_continuation_ids"]
    length = len(prompt)
    if len(continuation) != 3 or length + 2 >= MAX_SEQUENCE:
        raise ValueError(f"unsupported frozen case: {name}")

    states = fresh_states()
    logits = run_program(prefill_program, prompt, length, weights, cos, sin, states, prefill_config)
    scores = [score(logits, continuation[0], 0, length)]
    for step in (1, 2):
        ids = [*prompt, *continuation[:step]]
        logits = run_program(decode_program, ids, length + step - 1, weights, cos, sin, states, decode_config)
        scores.append(score(logits, continuation[step], step, length + step))
    print(f"{name}: teacher top-1 {[item['top1_id'] for item in scores]}", flush=True)

    free_states = fresh_states()
    logits = run_program(prefill_program, prompt, length, weights, cos, sin, free_states, prefill_config)
    generated = []
    for step in range(3):
        token = int(torch.argmax(logits.float()))
        generated.append(token)
        if step < 2:
            logits = run_program(decode_program, [*prompt, *generated], length + step,
                                 weights, cos, sin, free_states, decode_config)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt_token_ids": prompt,
        "golden_continuation_ids": continuation,
        "prefill_length": length,
        "cache_sequence_length": length + 2,
        "teacher_forced_scores": scores,
        "greedy_tokens": [item["top1_id"] for item in scores],
        "free_greedy_token_ids": generated,
        "free_greedy_text": text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assigned = os.environ["TASK_DEVICE"]
    if not assigned.isdecimal():
        raise ValueError(f"expected one queue-assigned TASK_DEVICE, got {assigned!r}")
    model_root = Path(os.environ["MODEL_ROOT"])
    workspace_root = Path.cwd()
    frozen = json.loads((workspace_root / "reference/qwen3_8_27b/evidence/hf-full-golden.json").read_text())
    weight_map = _check_checkpoint(model_root)
    build_root = Path(os.environ.get("QWEN_BUILD_ROOT", workspace_root / ".gp/c002-full-network"))
    build_root.mkdir(parents=True, exist_ok=True)
    prefill_config = RunConfig(platform="a2a3", device_id=int(assigned),
                               save_kernels_dir=str(build_root / "prefill"))
    decode_config = RunConfig(platform="a2a3", device_id=int(assigned),
                              save_kernels_dir=str(build_root / "decode"))
    print(f"Compiling whole-network programs for TASK_DEVICE={assigned}", flush=True)
    prefill_program = prefill_fwd.compile(config=prefill_config)
    decode_program = decode_fwd.compile(config=decode_config)
    print("Both PyPTO whole-network entries compiled", flush=True)
    banks = load_text_weight_banks(model_root, weight_map)
    weights = tuple(banks[name] for name in WEIGHT_ORDER)
    cos, sin = rope_table()
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    cases = {name: run_case(name, case, tokenizer, prefill_program, decode_program, weights,
                            cos, sin, prefill_config, decode_config)
             for name, case in frozen["cases"].items()}
    report = {
        "model_revision": REVISION,
        "config_sha256": CONFIG_SHA256,
        "index_sha256": INDEX_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "execution_backend": "pypto",
        "entrypoints": ["prefill_fwd", "decode_fwd"],
        "task_device": assigned,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote six-position report to {args.output}", flush=True)


if __name__ == "__main__":
    main()
