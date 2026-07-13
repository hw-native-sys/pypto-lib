# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Qwen3-14B kernel-ready weight layout preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from constants import QWEN3_14B


TensorExporter = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class PreparedQwen3Weights:
    """Kernel-ready Qwen3 weights exposed to external runtime code."""

    final_norm_weight: torch.Tensor
    padded_lm_head_weight: torch.Tensor
    padded_embed_weight: torch.Tensor
    decode_weights: dict[str, torch.Tensor]


@dataclass
class _KernelLayerWeights:
    """Kernel-ready weights for one transformer layer."""

    input_rms_weight: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    wo: torch.Tensor
    post_rms_weight: torch.Tensor
    w_gate: torch.Tensor
    w_up: torch.Tensor
    w_down: torch.Tensor


def prepare_qwen3_weights(
    runtime_model: Any,
    tensor_exporter: TensorExporter,
    *,
    padded_vocab: int | None = None,
    release_layers: bool = True,
) -> PreparedQwen3Weights:
    """Prepare Qwen3-14B weights in the layout expected by the interface kernels."""
    vocab = int(padded_vocab or QWEN3_14B.vocab)

    lm_head_weight = runtime_model.lm_head
    if lm_head_weight.shape[0] > vocab:
        raise ValueError(
            f"Model vocabulary size {lm_head_weight.shape[0]} exceeds "
            f"the kernel supported vocabulary size {vocab}."
        )
    if lm_head_weight.shape[0] < vocab:
        pad_rows = vocab - lm_head_weight.shape[0]
        # LM-head padding reuses a valid row so padded logits stay finite and deterministic.
        padding = lm_head_weight[:1].expand(pad_rows, -1).clone()
        lm_head_weight = torch.cat([lm_head_weight, padding], dim=0)
    padded_lm_head_weight = tensor_exporter(lm_head_weight.to(torch.bfloat16).contiguous().cpu())

    embed_weight = runtime_model.embed_tokens
    if embed_weight.shape[0] > vocab:
        raise ValueError(
            f"Model embedding vocabulary size {embed_weight.shape[0]} exceeds "
            f"the kernel supported vocabulary size {vocab}."
        )
    if embed_weight.shape[0] < vocab:
        pad_rows = vocab - embed_weight.shape[0]
        # Embedding padding is never a valid sampled token, so zero rows are neutral.
        padding = torch.zeros(
            (pad_rows, embed_weight.shape[1]),
            dtype=embed_weight.dtype,
            device=embed_weight.device,
        )
        embed_weight = torch.cat([embed_weight, padding], dim=0)
    padded_embed_weight = tensor_exporter(embed_weight.to(torch.bfloat16).contiguous().cpu())

    layers = []
    for layer in runtime_model.layers:
        layers.append(_kernel_layer_weights(layer))
        if release_layers:
            _release_layer_weights(layer)

    return PreparedQwen3Weights(
        final_norm_weight=tensor_exporter(runtime_model.final_norm_weight.view(1, -1).float().cpu()),
        padded_lm_head_weight=padded_lm_head_weight,
        padded_embed_weight=padded_embed_weight,
        decode_weights={
            name: tensor_exporter(tensor)
            for name, tensor in _stack_decode_weights(layers).items()
        },
    )


def _stack_decode_weights(layers: list[_KernelLayerWeights]) -> dict[str, torch.Tensor]:
    def cat(attr: str) -> torch.Tensor:
        return torch.cat([getattr(layer, attr) for layer in layers], dim=0)

    return {
        "decode_input_rms_weight": cat("input_rms_weight").contiguous(),
        "decode_wq": cat("wq"),
        "decode_wk": cat("wk"),
        "decode_wv": cat("wv"),
        "decode_q_norm_weight": cat("q_norm_weight").contiguous(),
        "decode_k_norm_weight": cat("k_norm_weight").contiguous(),
        "decode_wo": cat("wo"),
        "decode_post_rms_weight": cat("post_rms_weight").contiguous(),
        "decode_w_gate": cat("w_gate"),
        "decode_w_up": cat("w_up"),
        "decode_w_down": cat("w_down"),
    }


def _kernel_layer_weights(layer: Any) -> _KernelLayerWeights:
    return _KernelLayerWeights(
        input_rms_weight=layer.input_rms_weight.view(1, -1).float().cpu(),
        wq=_kernel_weight(layer.wq),
        wk=_kernel_weight(layer.wk),
        wv=_kernel_weight(layer.wv),
        q_norm_weight=layer.q_norm_weight.view(1, -1).float().cpu(),
        k_norm_weight=layer.k_norm_weight.view(1, -1).float().cpu(),
        wo=_kernel_weight(layer.wo),
        post_rms_weight=layer.post_rms_weight.view(1, -1).float().cpu(),
        w_gate=_kernel_weight(layer.w_gate),
        w_up=_kernel_weight(layer.w_up),
        w_down=_kernel_weight(layer.w_down),
    )


def _kernel_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(0, 1).to(torch.bfloat16).contiguous().cpu()


def _release_layer_weights(layer: Any) -> None:
    empty = torch.empty(0)
    layer.input_rms_weight = empty
    layer.wq = empty
    layer.wk = empty
    layer.wv = empty
    layer.q_norm_weight = empty
    layer.k_norm_weight = empty
    layer.wo = empty
    layer.post_rms_weight = empty
    layer.w_gate = empty
    layer.w_up = empty
    layer.w_down = empty
