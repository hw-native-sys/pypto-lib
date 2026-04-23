# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from .tokenizer import TokenizerAdapter, TransformersTokenizerAdapter
from .types import LayerSpec, LayerWeights, LoadedModel, ModelConfig, RuntimeConfig, RuntimeModel


def _torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    lowered = name.lower()
    if lowered not in mapping:
        raise ValueError(f"Unsupported dtype name: {name}")
    return mapping[lowered]


@dataclass(frozen=True)
class ModelLoadRequest:
    model_id: str
    model_dir: str
    runtime_config: RuntimeConfig | None = None
    model_format: str | None = None
    loader_options: dict[str, object] = field(default_factory=dict)


class ModelFormatLoader(Protocol):
    format_names: tuple[str, ...]

    def supports_format(self, model_format: str) -> bool:
        raise NotImplementedError

    def can_load(self, model_path: Path) -> bool:
        raise NotImplementedError

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        raise NotImplementedError


def _load_safetensors_dir(model_dir: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load weights from a local model directory.") from exc

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        filenames = sorted(set(index_data["weight_map"].values()))
    else:
        filenames = sorted(path.name for path in model_dir.glob("*.safetensors"))
    if not filenames:
        raise FileNotFoundError(f"No .safetensors files found in {model_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for filename in filenames:
        state_dict.update(load_file(str(model_dir / filename)))
    return state_dict


def _require_tensor(state_dict: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in state_dict:
        raise KeyError(f"Missing weight tensor: {name}")
    return state_dict[name]


def _optional_tensor(state_dict: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor | None:
    for name in names:
        if name in state_dict:
            return state_dict[name]
    return None


def _build_model_config(model_id: str, config_data: dict, tokenizer: TokenizerAdapter) -> ModelConfig:
    hidden_size = int(config_data["hidden_size"])
    num_heads = int(config_data["num_attention_heads"])
    num_kv_heads = int(config_data.get("num_key_value_heads", num_heads))
    head_dim = hidden_size // num_heads
    return ModelConfig(
        model_id=model_id,
        architecture=str(config_data.get("architectures", [config_data.get("model_type", "unknown")])[0]),
        vocab_size=int(config_data["vocab_size"]),
        hidden_size=hidden_size,
        intermediate_size=int(config_data["intermediate_size"]),
        num_hidden_layers=int(config_data["num_hidden_layers"]),
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=int(config_data.get("max_position_embeddings", config_data.get("seq_length", 4096))),
        rms_norm_eps=float(config_data.get("rms_norm_eps", config_data.get("layer_norm_epsilon", 1e-6))),
        rope_theta=float(config_data.get("rope_theta", 10000.0)),
        bos_token_id=config_data.get("bos_token_id", tokenizer.bos_token_id),
        eos_token_id=config_data.get("eos_token_id", tokenizer.eos_token_id),
        pad_token_id=config_data.get("pad_token_id", tokenizer.pad_token_id),
        torch_dtype=str(config_data.get("torch_dtype", "float16")),
    )


def _build_layer_specs(config: ModelConfig) -> list[LayerSpec]:
    return [
        LayerSpec(
            layer_idx=layer_idx,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
        )
        for layer_idx in range(config.num_hidden_layers)
    ]


def _cast_weight(weight: torch.Tensor, runtime: RuntimeConfig) -> torch.Tensor:
    dtype = _torch_dtype_from_name(runtime.weight_dtype)
    return weight.to(device=runtime.device, dtype=dtype)


class HuggingFaceDirectoryLoader:
    format_names = ("huggingface", "hf")

    def supports_format(self, model_format: str) -> bool:
        return model_format.lower() in self.format_names

    def can_load(self, model_path: Path) -> bool:
        config_path = model_path / "config.json"
        if not config_path.exists():
            return False
        if (model_path / "model.safetensors.index.json").exists():
            return True
        if any(model_path.glob("*.safetensors")):
            return True
        return False

    def load(self, request: ModelLoadRequest) -> LoadedModel:
        model_path = Path(request.model_dir)
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {model_path}")

        trust_remote_code = bool(request.loader_options.get("trust_remote_code", False))
        tokenizer = TransformersTokenizerAdapter.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote_code,
        )
        config_data = json.loads(config_path.read_text())
        config = _build_model_config(request.model_id, config_data, tokenizer)
        runtime = request.runtime_config or RuntimeConfig(max_seq_len=config.max_position_embeddings)
        layer_specs = _build_layer_specs(config)
        state_dict = _load_safetensors_dir(model_path)

        if config.architecture.lower() not in {"qwen2forcausallm", "qwen3forcausallm", "qwen2model", "qwen3model"}:
            raise ValueError(
                f"Unsupported architecture {config.architecture}. "
                "Current Hugging Face adapter supports Qwen2/Qwen3-style decoder-only models."
            )

        embed_tokens = _cast_weight(_require_tensor(state_dict, "model.embed_tokens.weight"), runtime)
        final_norm = _optional_tensor(state_dict, ["model.norm.weight", "model.final_layernorm.weight"])
        if final_norm is None:
            final_norm = _require_tensor(state_dict, "model.norm.weight")
        final_norm_weight = _cast_weight(final_norm, runtime)
        lm_head = _optional_tensor(state_dict, ["lm_head.weight"])
        if lm_head is None:
            lm_head = embed_tokens
        else:
            lm_head = _cast_weight(lm_head, runtime)

        layers: list[LayerWeights] = []
        default_dtype = _torch_dtype_from_name(runtime.weight_dtype)
        for spec in layer_specs:
            prefix = f"model.layers.{spec.layer_idx}"
            q_norm = _optional_tensor(state_dict, [f"{prefix}.self_attn.q_norm.weight"])
            k_norm = _optional_tensor(state_dict, [f"{prefix}.self_attn.k_norm.weight"])
            if q_norm is None:
                q_norm = torch.ones(spec.head_dim, device=runtime.device, dtype=default_dtype)
            else:
                q_norm = _cast_weight(q_norm, runtime)
            if k_norm is None:
                k_norm = torch.ones(spec.head_dim, device=runtime.device, dtype=default_dtype)
            else:
                k_norm = _cast_weight(k_norm, runtime)
            layers.append(
                LayerWeights(
                    input_rms_weight=_cast_weight(_require_tensor(state_dict, f"{prefix}.input_layernorm.weight"), runtime),
                    wq=_cast_weight(_require_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight"), runtime),
                    wk=_cast_weight(_require_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight"), runtime),
                    wv=_cast_weight(_require_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight"), runtime),
                    q_norm_weight=q_norm,
                    k_norm_weight=k_norm,
                    wo=_cast_weight(_require_tensor(state_dict, f"{prefix}.self_attn.o_proj.weight"), runtime),
                    post_rms_weight=_cast_weight(
                        _require_tensor(state_dict, f"{prefix}.post_attention_layernorm.weight"),
                        runtime,
                    ),
                    w_gate=_cast_weight(_require_tensor(state_dict, f"{prefix}.mlp.gate_proj.weight"), runtime),
                    w_up=_cast_weight(_require_tensor(state_dict, f"{prefix}.mlp.up_proj.weight"), runtime),
                    w_down=_cast_weight(_require_tensor(state_dict, f"{prefix}.mlp.down_proj.weight"), runtime),
                )
            )

        runtime_model = RuntimeModel(
            config=config,
            runtime=runtime,
            embed_tokens=embed_tokens,
            final_norm_weight=final_norm_weight,
            lm_head=lm_head,
            layers=layers,
        )
        return LoadedModel(
            model_id=request.model_id,
            model_dir=str(model_path),
            config=config,
            tokenizer=tokenizer,
            layer_specs=layer_specs,
            runtime_model=runtime_model,
        )


class ModelLoader:
    def __init__(self, format_loaders: list[ModelFormatLoader] | None = None) -> None:
        self._format_loaders = format_loaders or [HuggingFaceDirectoryLoader()]

    def register(self, format_loader: ModelFormatLoader) -> None:
        self._format_loaders.append(format_loader)

    def load(
        self,
        model_id: str,
        model_dir: str,
        runtime_config: RuntimeConfig | None = None,
        model_format: str | None = None,
        **loader_options: object,
    ) -> LoadedModel:
        request = ModelLoadRequest(
            model_id=model_id,
            model_dir=model_dir,
            runtime_config=runtime_config,
            model_format=model_format,
            loader_options=loader_options,
        )
        loader = self._select_loader(request)
        return loader.load(request)

    def _select_loader(self, request: ModelLoadRequest) -> ModelFormatLoader:
        model_path = Path(request.model_dir)
        if request.model_format is not None:
            for loader in self._format_loaders:
                if loader.supports_format(request.model_format):
                    return loader
            supported = sorted({name for loader in self._format_loaders for name in loader.format_names})
            raise ValueError(
                f"Unsupported model_format={request.model_format!r}. "
                f"Registered formats: {', '.join(supported)}"
            )

        for loader in self._format_loaders:
            if loader.can_load(model_path):
                return loader
        raise ValueError(
            f"Could not detect a supported model format in {model_path}. "
            "Pass model_format explicitly or register another format loader."
        )
