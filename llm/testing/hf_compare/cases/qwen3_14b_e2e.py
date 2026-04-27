# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Comparison case: Qwen3-14B end-to-end generation vs Hugging Face.

Run via:

    python -m llm.testing.hf_compare run qwen3_14b.e2e \
        -k hf_model_path=/data/linyifan/models/Qwen3-14B \
        -k prompt_ids=151644,872,198 \
        -k max_new_tokens=4 \
        -k num_layers=2 \
        -k cpu_only=true
"""
from __future__ import annotations

import importlib.util
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..base import ComparisonCase, InputSpec, OutputSelector, TensorSpec, Tolerance, register_case
from ..reference import CallableReference
from ..target import CallableTarget, _resolve_device_id
from ..weight_adapter import PassthroughAdapter


DEFAULT_MODEL_PATH = "/data/linyifan/models/Qwen3-14B"
DEFAULT_PROMPT_IDS = "151644,872,198"
EPS = 1e-6


def _load_qwen3_modules() -> tuple[Any, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    base_dir = repo_root / "examples" / "models" / "qwen3" / "14b"

    def _load(name: str, filename: str) -> Any:
        module_path = base_dir / filename
        spec = importlib.util.spec_from_file_location(name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load Qwen3-14B helper from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return (
        _load("_qwen3_14b_decode_kernel", "qwen3_14b_decode.py"),
        _load("_qwen3_14b_prefill_kernel", "qwen3_14b_prefill.py"),
    )


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "on"}


def _parse_prompt_ids(prompt_ids: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(prompt_ids, (list, tuple)):
        return [int(v) for v in prompt_ids]
    out = [int(part.strip()) for part in str(prompt_ids).split(",") if part.strip()]
    if not out:
        raise ValueError("prompt_ids must contain at least one token id")
    return out


def _state_for_num_layers(
    hf_state: Mapping[str, torch.Tensor],
    *,
    num_layers: int,
) -> dict[str, torch.Tensor]:
    keep: dict[str, torch.Tensor] = {}
    layer_prefixes = tuple(f"model.layers.{idx}." for idx in range(num_layers))
    for key, value in hf_state.items():
        if key.startswith(("model.embed_tokens.", "model.norm.", "lm_head.")):
            keep[key] = value
            continue
        if key.startswith(layer_prefixes):
            keep[key] = value
    return keep


def _extract_layer_weights(
    all_weights: Mapping[str, torch.Tensor],
    *,
    layer_idx: int,
    hidden: int,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    prefix = f"model.layers.{layer_idx}."

    def w(name: str) -> torch.Tensor:
        return all_weights[prefix + name]

    def proj(name: str) -> torch.Tensor:
        return w(name).t().contiguous().to(torch.bfloat16)

    return {
        "input_rms_weight": w("input_layernorm.weight").view(1, hidden).to(torch.float32),
        "wq": proj("self_attn.q_proj.weight"),
        "wk": proj("self_attn.k_proj.weight"),
        "wv": proj("self_attn.v_proj.weight"),
        "q_norm_weight": w("self_attn.q_norm.weight").view(1, head_dim).to(torch.float32),
        "k_norm_weight": w("self_attn.k_norm.weight").view(1, head_dim).to(torch.float32),
        "wo": proj("self_attn.o_proj.weight"),
        "post_rms_weight": w("post_attention_layernorm.weight").view(1, hidden).to(torch.float32),
        "w_gate": proj("mlp.gate_proj.weight"),
        "w_up": proj("mlp.up_proj.weight"),
        "w_down": proj("mlp.down_proj.weight"),
    }


def _build_rope(max_seq: int, head_dim: int, base: float = 1_000_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_seq, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(torch.float32), emb.sin().to(torch.float32)


def _embed_tokens(input_ids: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    return embed_weight[input_ids].to(torch.bfloat16)


def _final_rmsnorm(hidden: torch.Tensor, norm_weight: torch.Tensor) -> torch.Tensor:
    x = hidden.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True) + EPS
    return x * torch.rsqrt(variance) * norm_weight.float()


def _lm_head_forward(hidden: torch.Tensor, lm_weight: torch.Tensor) -> torch.Tensor:
    return torch.matmul(hidden.float(), lm_weight.float().t())


def _greedy_sample(logits: torch.Tensor) -> torch.Tensor:
    return logits.argmax(dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x_lo, x_hi = x[..., :half], x[..., half:]
    cos_lo, cos_hi = cos[..., :half], cos[..., half:]
    sin_lo, sin_hi = sin[..., :half], sin[..., half:]
    return torch.cat([
        x_lo * cos_lo - x_hi * sin_lo,
        x_hi * cos_hi + x_lo * sin_hi,
    ], dim=-1)


def _run_cpu_prefill(
    hidden_states: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    batch: int,
    max_seq: int,
    head_dim: int,
    eps: float = EPS,
) -> torch.Tensor:
    w = layer_weights
    hidden_size = w["wq"].shape[0]
    kv_hidden = w["wk"].shape[1]
    num_heads = hidden_size // head_dim
    num_kv_heads = kv_hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    wq_f = w["wq"].float()
    wk_f = w["wk"].float()
    wv_f = w["wv"].float()
    wo_f = w["wo"].float()
    w_gate_f = w["w_gate"].float()
    w_up_f = w["w_up"].float()
    w_down_f = w["w_down"].float()
    rms_w = w["input_rms_weight"].float()
    post_rms_w = w["post_rms_weight"].float()
    q_norm_w = w["q_norm_weight"].float().view(1, 1, head_dim)
    k_norm_w = w["k_norm_weight"].float().view(1, 1, head_dim)

    out = torch.zeros(batch, max_seq, hidden_size, dtype=torch.bfloat16)

    for b in range(batch):
        seq = int(seq_lens[b].item())
        if seq <= 0:
            continue

        x = hidden_states[b, :seq, :].float()
        variance = x.square().mean(dim=-1, keepdim=True) + eps
        normed = (x / torch.sqrt(variance) * rms_w).to(torch.bfloat16).float()
        q_proj = normed @ wq_f
        k_proj = normed @ wk_f
        v_proj = normed @ wv_f

        q_view = q_proj.view(seq, num_heads, head_dim)
        q_view = q_view * torch.rsqrt(q_view.pow(2).mean(-1, keepdim=True) + eps) * q_norm_w
        k_view = k_proj.view(seq, num_kv_heads, head_dim)
        k_view = k_view * torch.rsqrt(k_view.pow(2).mean(-1, keepdim=True) + eps) * k_norm_w

        cos = rope_cos[:seq, :].unsqueeze(1)
        sin = rope_sin[:seq, :].unsqueeze(1)
        k_rot = _apply_rope(k_view, cos, sin).to(torch.bfloat16)
        v_bf16 = v_proj.view(seq, num_kv_heads, head_dim).to(torch.bfloat16)
        q_rot = _apply_rope(q_view, cos, sin).to(torch.bfloat16)

        cache_off = b * num_kv_heads * max_seq
        for ki in range(num_kv_heads):
            row0 = cache_off + ki * max_seq
            k_cache[row0:row0 + seq, :] = k_rot[:, ki, :]
            v_cache[row0:row0 + seq, :] = v_bf16[:, ki, :]

        attn_result = torch.zeros(seq, hidden_size, dtype=torch.float32)
        for ki in range(num_kv_heads):
            row0 = cache_off + ki * max_seq
            k_cached = k_cache[row0:row0 + seq, :]
            v_cached = v_cache[row0:row0 + seq, :]

            for qi in range(q_per_kv):
                hi = ki * q_per_kv + qi
                q_head = q_rot[:, hi, :]
                scores = (q_head.float() @ k_cached.float().T) * scale
                causal = torch.triu(torch.full((seq, seq), float("-inf")), diagonal=1)
                scores = scores + causal
                attn_w = torch.softmax(scores, dim=-1)
                attn_result[:, hi * head_dim:(hi + 1) * head_dim] = attn_w @ v_cached.float()

        attn_bf16 = attn_result.to(torch.bfloat16).float()
        hs = hidden_states[b, :seq, :].float()
        resid1 = torch.matmul(attn_bf16, wo_f) + hs
        variance = resid1.pow(2).mean(-1, keepdim=True)
        post_normed = (resid1 * torch.rsqrt(variance + eps) * post_rms_w).bfloat16().float()
        gate = torch.matmul(post_normed, w_gate_f)
        up = torch.matmul(post_normed, w_up_f)
        mlp = (gate * torch.sigmoid(gate) * up).bfloat16().float()
        down = torch.matmul(mlp, w_down_f)
        out[b, :seq, :] = (down + resid1).bfloat16()

    return out


def _run_cpu_decode(
    hidden_states: torch.Tensor,
    seq_lens: torch.Tensor,
    layer_weights: dict[str, torch.Tensor],
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    batch: int,
    head_dim: int,
    eps: float = EPS,
) -> torch.Tensor:
    w = layer_weights
    hidden_size = w["wq"].shape[0]
    kv_hidden = w["wk"].shape[1]
    num_heads = hidden_size // head_dim
    num_kv_heads = kv_hidden // head_dim
    q_per_kv = num_heads // num_kv_heads
    max_seq = rope_cos.shape[0]
    scale = 1.0 / math.sqrt(head_dim)

    wq_f = w["wq"].float()
    wk_f = w["wk"].float()
    wv_f = w["wv"].float()
    wo_f = w["wo"].float()
    w_gate_f = w["w_gate"].float()
    w_up_f = w["w_up"].float()
    w_down_f = w["w_down"].float()
    rms_w = w["input_rms_weight"].float()
    post_rms_w = w["post_rms_weight"].float()
    q_norm_w = w["q_norm_weight"].float()
    k_norm_w = w["k_norm_weight"].float()

    out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = int(seq_lens[b].item())
        pos = ctx_len - 1
        x = hidden_states[b:b + 1, :].float()
        variance = x.square().mean(dim=-1, keepdim=True) + eps
        normed = (x / torch.sqrt(variance) * rms_w).to(torch.bfloat16).float()
        q_proj = normed @ wq_f
        k_proj = normed @ wk_f
        v_proj = normed @ wv_f

        q_heads = q_proj.view(num_heads, head_dim)
        q_heads = q_heads * torch.rsqrt(q_heads.pow(2).mean(-1, keepdim=True) + eps) * q_norm_w
        k_heads = k_proj.view(num_kv_heads, head_dim)
        k_heads = k_heads * torch.rsqrt(k_heads.pow(2).mean(-1, keepdim=True) + eps) * k_norm_w

        cos = rope_cos[pos:pos + 1, :]
        sin = rope_sin[pos:pos + 1, :]
        k_rot = _apply_rope(k_heads, cos, sin).to(torch.bfloat16)
        q_rot = _apply_rope(q_heads, cos, sin).to(torch.bfloat16)
        v_bf16 = v_proj.view(num_kv_heads, head_dim).to(torch.bfloat16)

        for ki in range(num_kv_heads):
            cr = b * num_kv_heads * max_seq + ki * max_seq + pos
            k_cache[cr, :] = k_rot[ki]
            v_cache[cr, :] = v_bf16[ki]

        attn_result = torch.zeros(1, hidden_size, dtype=torch.float32)
        for ki in range(num_kv_heads):
            row0 = b * num_kv_heads * max_seq + ki * max_seq
            k_ctx = k_cache[row0:row0 + ctx_len, :]
            v_ctx = v_cache[row0:row0 + ctx_len, :]

            for qi in range(q_per_kv):
                hi = ki * q_per_kv + qi
                q_head = q_rot[hi:hi + 1, :]
                scores = (q_head.float() @ k_ctx.float().T) * scale
                attn_w = torch.softmax(scores, dim=-1)
                attn_result[:, hi * head_dim:(hi + 1) * head_dim] = attn_w @ v_ctx.float()

        attn_bf16 = attn_result.to(torch.bfloat16).float()
        hs = hidden_states[b:b + 1, :].float()
        resid1 = torch.matmul(attn_bf16, wo_f) + hs
        variance = resid1.pow(2).mean(-1, keepdim=True)
        post_normed = (resid1 * torch.rsqrt(variance + eps) * post_rms_w).bfloat16().float()
        gate = torch.matmul(post_normed, w_gate_f)
        up = torch.matmul(post_normed, w_up_f)
        mlp = (gate * torch.sigmoid(gate) * up).bfloat16().float()
        down = torch.matmul(mlp, w_down_f)
        out[b:b + 1, :] = (down + resid1).bfloat16()

    return out


class _NPURunner:
    def __init__(self, dec: Any, prefill: Any, *, batch: int, max_seq: int, platform: str, device_id: int):
        from pypto import ir
        from pypto.backend import BackendType
        from pypto.runtime import execute_compiled

        backend_map = {
            "a2a3": BackendType.Ascend910B,
            "a2a3sim": BackendType.Ascend910B,
            "a5": BackendType.Ascend950,
            "a5sim": BackendType.Ascend950,
        }
        self._execute = execute_compiled
        self._platform = platform
        self._device_id = device_id
        backend = backend_map[platform]
        self._prefill_spec_order = [
            "hidden_states", "seq_lens", "input_rms_weight",
            "wq", "wk", "wv", "q_norm_weight", "k_norm_weight",
            "rope_cos", "rope_sin", "k_cache", "v_cache",
            "wo", "post_rms_weight", "w_gate", "w_up", "w_down", "out",
        ]
        self._decode_spec_order = [
            "hidden_states", "input_rms_weight",
            "wq", "wk", "wv", "q_norm_weight", "k_norm_weight",
            "seq_lens", "rope_cos", "rope_sin", "k_cache", "v_cache",
            "wo", "post_rms_weight", "w_gate", "w_up", "w_down", "out",
        ]

        compiled_prefill = ir.compile(prefill.build_qwen3_14b_prefill_program(batch=batch, max_seq=max_seq), backend_type=backend)
        self._prefill_dir = compiled_prefill.output_dir
        compiled_decode = ir.compile(dec.build_qwen3_decode_program(batch=batch, max_seq=max_seq), backend_type=backend)
        self._decode_dir = compiled_decode.output_dir

    def run_prefill(
        self,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        layer_weights: dict[str, torch.Tensor],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        batch: int,
        max_seq: int,
    ) -> torch.Tensor:
        hidden = layer_weights["wq"].shape[0]
        out = torch.zeros(batch, max_seq, hidden, dtype=torch.bfloat16)
        tensors = {
            "hidden_states": hidden_states,
            "seq_lens": seq_lens,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "out": out,
            **layer_weights,
        }
        ordered = [tensors[name] for name in self._prefill_spec_order]
        self._execute(self._prefill_dir, ordered, platform=self._platform, device_id=self._device_id)
        return tensors["out"]

    def run_decode(
        self,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        layer_weights: dict[str, torch.Tensor],
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        batch: int,
    ) -> torch.Tensor:
        hidden = layer_weights["wq"].shape[0]
        out = torch.zeros(batch, hidden, dtype=torch.bfloat16)
        tensors = {
            "hidden_states": hidden_states,
            "seq_lens": seq_lens,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "k_cache": k_cache,
            "v_cache": v_cache,
            "out": out,
            **layer_weights,
        }
        ordered = [tensors[name] for name in self._decode_spec_order]
        self._execute(self._decode_dir, ordered, platform=self._platform, device_id=self._device_id)
        return tensors["out"]


def _hf_e2e_forward(
    inputs: Mapping[str, torch.Tensor],
    hf_state: Mapping[str, torch.Tensor],
    *,
    model_path: str,
    num_layers: int,
    max_new_tokens: int,
    hf_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    from transformers import AutoConfig
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    cfg = AutoConfig.from_pretrained(model_path)
    cfg._attn_implementation = "eager"
    cfg.num_hidden_layers = num_layers

    model = Qwen3ForCausalLM(cfg).to(hf_dtype).eval()
    sd = {k: v.to(hf_dtype) for k, v in _state_for_num_layers(hf_state, num_layers=num_layers).items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"[qwen3.e2e] unexpected HF keys: {unexpected}")
    missing = [key for key in missing if "rotary_emb.inv_freq" not in key]
    if missing:
        raise RuntimeError(f"[qwen3.e2e] missing HF keys: {missing}")

    input_ids = inputs["input_ids"].to(torch.long)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        logits = out.logits[:, -1, :].float()
        next_token = logits.argmax(dim=-1)
        generated: list[int] = [int(next_token.item())]
        past_key_values = out.past_key_values

        for _ in range(1, max_new_tokens):
            out = model(input_ids=next_token.view(1, 1), use_cache=True, past_key_values=past_key_values)
            logits = out.logits[:, -1, :].float()
            next_token = logits.argmax(dim=-1)
            generated.append(int(next_token.item()))
            past_key_values = out.past_key_values

    return {
        "generated_ids": torch.tensor([generated], dtype=torch.int64),
        "last_logits": logits,
    }


def _run_e2e_target(
    inputs: Mapping[str, torch.Tensor],
    weights: Mapping[str, torch.Tensor],
    *,
    max_new_tokens: int,
    num_layers: int,
    max_seq: int,
    cpu_only: bool,
    platform: str,
    device_id: int | str | None,
) -> dict[str, torch.Tensor]:
    dec, prefill = _load_qwen3_modules()

    input_ids = inputs["input_ids"].to(torch.long)
    batch = int(input_ids.shape[0])
    if batch != 1:
        raise ValueError(f"qwen3_14b.e2e expects batch=1, got {batch}")

    prompt_len = int(input_ids.shape[1])
    if prompt_len + max_new_tokens - 1 > max_seq:
        raise ValueError(
            f"prompt_len + max_new_tokens - 1 = {prompt_len + max_new_tokens - 1} exceeds max_seq={max_seq}"
        )

    hidden = int(prefill.HIDDEN)
    head_dim = int(prefill.HEAD_DIM)
    num_kv_heads = int(prefill.NUM_KV_HEADS)
    cache_rows = batch * num_kv_heads * max_seq

    layer_weights = [
        _extract_layer_weights(dict(weights), layer_idx=idx, hidden=hidden, head_dim=head_dim)
        for idx in range(num_layers)
    ]
    embed_weight = weights["model.embed_tokens.weight"]
    norm_weight = weights["model.norm.weight"].view(1, hidden)
    lm_weight = weights["lm_head.weight"]
    rope_cos, rope_sin = _build_rope(max_seq, head_dim)

    kv_caches = [
        {
            "k_cache": torch.zeros(cache_rows, head_dim, dtype=torch.bfloat16),
            "v_cache": torch.zeros(cache_rows, head_dim, dtype=torch.bfloat16),
        }
        for _ in range(num_layers)
    ]

    resolved_device_id = _resolve_device_id(device_id)
    npu_runner = None if cpu_only else _NPURunner(
        dec,
        prefill,
        batch=batch,
        max_seq=max_seq,
        platform=platform,
        device_id=resolved_device_id,
    )

    hidden_states_full = torch.zeros(batch, max_seq, hidden, dtype=torch.bfloat16)
    hidden_states_full[:, :prompt_len, :] = _embed_tokens(input_ids, embed_weight)
    seq_lens = torch.full((batch,), prompt_len, dtype=torch.int32)

    for layer_idx in range(num_layers):
        if cpu_only:
            hidden_states_full = _run_cpu_prefill(
                hidden_states_full,
                seq_lens,
                layer_weights[layer_idx],
                rope_cos,
                rope_sin,
                kv_caches[layer_idx]["k_cache"],
                kv_caches[layer_idx]["v_cache"],
                batch=batch,
                max_seq=max_seq,
                head_dim=head_dim,
            )
        else:
            hidden_states_full = npu_runner.run_prefill(
                hidden_states_full,
                seq_lens,
                layer_weights[layer_idx],
                rope_cos,
                rope_sin,
                kv_caches[layer_idx]["k_cache"],
                kv_caches[layer_idx]["v_cache"],
                batch=batch,
                max_seq=max_seq,
            )

    last_hidden = hidden_states_full[:, prompt_len - 1, :].float()
    logits = _lm_head_forward(_final_rmsnorm(last_hidden, norm_weight), lm_weight).float()
    next_token = _greedy_sample(logits)
    generated = [int(next_token.item())]
    cur_seq_len = prompt_len + 1

    for _ in range(1, max_new_tokens):
        hidden_states = _embed_tokens(next_token.unsqueeze(0), embed_weight).squeeze(1)
        decode_seq_lens = torch.full((batch,), cur_seq_len, dtype=torch.int32)

        for layer_idx in range(num_layers):
            if cpu_only:
                hidden_states = _run_cpu_decode(
                    hidden_states,
                    decode_seq_lens,
                    layer_weights[layer_idx],
                    rope_cos,
                    rope_sin,
                    kv_caches[layer_idx]["k_cache"],
                    kv_caches[layer_idx]["v_cache"],
                    batch=batch,
                    head_dim=head_dim,
                )
            else:
                hidden_states = npu_runner.run_decode(
                    hidden_states,
                    decode_seq_lens,
                    layer_weights[layer_idx],
                    rope_cos,
                    rope_sin,
                    kv_caches[layer_idx]["k_cache"],
                    kv_caches[layer_idx]["v_cache"],
                    batch=batch,
                )

        logits = _lm_head_forward(_final_rmsnorm(hidden_states.float(), norm_weight), lm_weight).float()
        next_token = _greedy_sample(logits)
        generated.append(int(next_token.item()))
        cur_seq_len += 1

    return {
        "generated_ids": torch.tensor([generated], dtype=torch.int64),
        "last_logits": logits,
    }


@register_case("qwen3_14b.e2e")
def build(
    hf_model_path: str = DEFAULT_MODEL_PATH,
    prompt_ids: str = DEFAULT_PROMPT_IDS,
    cpu_only: Any = True,
    platform: str = "a2a3",
    device_id: int | str | None = None,
    num_layers: int = 2,
    max_new_tokens: int = 4,
    max_seq: int = 256,
    atol: float = 5e-3,
    rtol: float = 5e-3,
    hf_dtype: str = "fp32",
) -> ComparisonCase:
    cpu_only = _coerce_bool(cpu_only)
    device_id = int(device_id) if device_id is not None else None
    num_layers = int(num_layers)
    max_new_tokens = int(max_new_tokens)
    max_seq = int(max_seq)
    atol = float(atol)
    rtol = float(rtol)
    hf_dtype_t = torch.float32 if hf_dtype == "fp32" else torch.bfloat16

    prompt = _parse_prompt_ids(prompt_ids)
    if len(prompt) + max_new_tokens - 1 > max_seq:
        raise ValueError(
            f"prompt length {len(prompt)} with max_new_tokens={max_new_tokens} exceeds max_seq={max_seq}"
        )

    prompt_tensor = torch.tensor(prompt, dtype=torch.int64).view(1, -1)
    input_spec = InputSpec(
        tensors={
            "input_ids": TensorSpec(
                tuple(prompt_tensor.shape),
                torch.int64,
                sampler=lambda s, d, g, t=prompt_tensor: t.clone().to(d),
            ),
        }
    )

    reference = CallableReference(
        name="hf.Qwen3ForCausalLM",
        fn=lambda inp, st: _hf_e2e_forward(
            inp,
            st,
            model_path=hf_model_path,
            num_layers=num_layers,
            max_new_tokens=max_new_tokens,
            hf_dtype=hf_dtype_t,
        ),
    )
    target = CallableTarget(
        name="pytorch.qwen3_14b_e2e" if cpu_only else f"pypto.qwen3_14b_e2e[{platform}]",
        fn=lambda inp, w: _run_e2e_target(
            inp,
            w,
            max_new_tokens=max_new_tokens,
            num_layers=num_layers,
            max_seq=max_seq,
            cpu_only=cpu_only,
            platform=platform,
            device_id=device_id,
        ),
    )

    return ComparisonCase(
        name="qwen3_14b.e2e",
        reference=reference,
        target=target,
        input_spec=input_spec,
        weight_adapter=PassthroughAdapter(),
        selectors=[
            OutputSelector(
                name="generated_ids",
                ref_key="generated_ids",
                tgt_key="generated_ids",
                # int64 fits exactly in float32 for typical token IDs (< 2^24).
                cast_to=torch.float32,
            ),
            OutputSelector(
                name="last_logits",
                ref_key="last_logits",
                tgt_key="last_logits",
                cast_to=torch.float32,
            ),
        ],
        tolerance=Tolerance(atol=atol, rtol=rtol, pass_rate_threshold=1.0),
        hf_weights=hf_model_path,
    )
