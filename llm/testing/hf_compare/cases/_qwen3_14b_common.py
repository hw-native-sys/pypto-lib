# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared constants, weight adapter, and kernel loader for Qwen3-14B hf_compare cases."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

from ..weight_adapter import Cast, Contiguous, DictAdapter, Map, Transpose, View


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "/data/linyifan/models/Qwen3-14B"

HIDDEN = 5120
HEAD_DIM = 128
NUM_HEADS = 40
NUM_KV_HEADS = 8
INTER = 17408


# ---------------------------------------------------------------------------
# Kernel input ordering. Decode and fused decode share an identical signature;
# prefill reorders seq_lens / paged-attention metadata.
# ---------------------------------------------------------------------------
DECODE_SPEC_ORDER: list[str] = [
    "hidden_states",
    "input_rms_weight",
    "wq",
    "wk",
    "wv",
    "q_norm_weight",
    "k_norm_weight",
    "seq_lens",
    "block_table",
    "slot_mapping",
    "rope_cos",
    "rope_sin",
    "k_cache",
    "v_cache",
    "wo",
    "post_rms_weight",
    "w_gate",
    "w_up",
    "w_down",
    "out",
]

PREFILL_SPEC_ORDER: list[str] = [
    "hidden_states",
    "seq_lens",
    "input_rms_weight",
    "wq",
    "wk",
    "wv",
    "q_norm_weight",
    "k_norm_weight",
    "rope_cos",
    "rope_sin",
    "block_table",
    "slot_mapping",
    "k_cache",
    "v_cache",
    "wo",
    "post_rms_weight",
    "w_gate",
    "w_up",
    "w_down",
    "out",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "on"}


def load_kernel_module(rel_path: str, module_name: str) -> ModuleType:
    """Load a kernel ``.py`` file by repo-relative path.

    ``rel_path`` is resolved against the pypto-lib repo root (four levels up
    from this file). ``module_name`` is the sys.modules key the loaded module
    is bound to.
    """
    repo_root = Path(__file__).resolve().parents[4]
    kernel_path = repo_root / rel_path
    spec = importlib.util.spec_from_file_location(module_name, kernel_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load kernel from {kernel_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def single_layer_adapter(layer_idx: int = 0) -> DictAdapter:
    """DictAdapter mapping HF Qwen3-14B layer weights to kernel inputs.

    Identical layout is consumed by the single-layer decode and prefill
    kernels; pick a different ``layer_idx`` to source weights from a non-zero
    HF layer.
    """
    return DictAdapter(
        prefix=f"model.layers.{layer_idx}.",
        mapping={
            "input_rms_weight": Map("input_layernorm.weight",
                                    ops=[View([1, HIDDEN]), Cast(torch.float32)]),
            "wq": Map("self_attn.q_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "wk": Map("self_attn.k_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "wv": Map("self_attn.v_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "q_norm_weight": Map("self_attn.q_norm.weight",
                                 ops=[View([1, HEAD_DIM]), Cast(torch.float32)]),
            "k_norm_weight": Map("self_attn.k_norm.weight",
                                 ops=[View([1, HEAD_DIM]), Cast(torch.float32)]),
            "wo": Map("self_attn.o_proj.weight",
                      ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "post_rms_weight": Map("post_attention_layernorm.weight",
                                   ops=[View([1, HIDDEN]), Cast(torch.float32)]),
            "w_gate": Map("mlp.gate_proj.weight",
                          ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "w_up": Map("mlp.up_proj.weight",
                        ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
            "w_down": Map("mlp.down_proj.weight",
                          ops=[Transpose(), Contiguous(), Cast(torch.bfloat16)]),
        },
    )
