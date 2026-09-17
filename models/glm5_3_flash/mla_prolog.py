# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""NoPE MLA prolog for the 11 DSA layers and the MTP layer.

``qk_rope_head_dim`` is 0 and ``mla_use_nope`` is true, so there is **no rope
anywhere in this model** — not here and not in the indexer either. The config
validator rejects a non-zero rope dim outright, ``rope_parameters`` is removed from
the config class, and the model forward passes ``position_embeddings=None``. That
removes the usual DeepSeek MLA rope/nope split and every rope table with it.

    q_resid  = q_a_layernorm(q_a_proj(x))                 [T, 1536]
    query    = q_b_proj(q_resid)                          [T, H, 256]
    kv_pass  = kv_a_layernorm(kv_a_proj_with_mqa(x))      [T, 512]

``q_resid`` is also the indexer's input, so this kernel publishes it separately
rather than fusing the whole prolog. ``q_a_proj``, ``q_b_proj`` and
``kv_a_proj_with_mqa`` carry ``weight_scale_inv`` in the released checkpoint;
whether the deployed msmodelslim W8A8 conversion keeps them INT8 or falls back to
BF16 is the open question tracked with the weight loader — vLLM Ascend builds the
whole MLA block with ``quant_config=None``, while the a2a3 sibling port keeps
``wq_b`` INT8 and ``wq_a`` BF16.

**Prior art.** ``ops/pypto_python/impl/mla_prolog_pypto.py`` and its
``mla_prolog_quant_pypto.py`` sibling in cann-recipes-infer implement the MLA prolog
unquantized and in W8A8 on this hardware generation. Both target the older
``pypto.Tensor`` / ``pypto_impl`` frontend, which this repo's pinned pypto does not
export, so treat them as algorithm and tiling references rather than code.


## ``kv_b_proj`` absorption for the latent decode path

``kv_b_proj`` maps the 512-wide latent to ``H * (qk_nope_head_dim + v_head_dim)``.
The reference expands it per token and attends in full head space, which is right
for prefill but throws away the point of the latent cache in decode. The absorbed
form folds the key half into the query (giving a ``[H, KV_LORA]`` query that attends
directly against the cached latent) and the value half into ``o_proj``. Both are
weight-time transforms, so they live here rather than in a per-token kernel.

``kv_b_proj`` has **no** ``weight_scale_inv`` in the checkpoint — it is BF16
``[32768, 512]`` — so folding it into an INT8 ``o_proj`` is the numerical question
the owner has to settle.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KV_LORA, LOCAL_H, QK_DIM, Q_LORA, T_DYN, V_DIM


def golden_mla_prolog(
    x: torch.Tensor,
    w_q_a: torch.Tensor,
    q_a_norm_weight: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_a: torch.Tensor,
    kv_a_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("MLA prolog golden is assigned with the kernel")


@pl.jit.inline
def mla_prolog(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    w_q_b_scale: pl.Tensor[[LOCAL_H * QK_DIM], pl.FP32],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.BF16],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    kv_latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
):
    raise NotImplementedError("MLA prolog kernel body is assigned independently")


def golden_split_kv_b(w_kv_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ``kv_b_proj`` into its key and value halves.

    Args:
        w_kv_b: ``[H * (QK_DIM + V_DIM), KV_LORA]`` as stored in the checkpoint.

    Returns:
        ``w_k`` of ``[H, QK_DIM, KV_LORA]`` and ``w_v`` of ``[H, V_DIM, KV_LORA]``.
    """
    heads = w_kv_b.shape[0] // (QK_DIM + V_DIM)
    reshaped = w_kv_b.reshape(heads, QK_DIM + V_DIM, KV_LORA)
    return reshaped[:, :QK_DIM], reshaped[:, QK_DIM:]


def golden_absorb_query(query: torch.Tensor, w_k: torch.Tensor) -> torch.Tensor:
    """Fold the key half into the query so it attends against the raw latent."""
    raise NotImplementedError("MLA query absorb golden is assigned with the kernel")


def golden_absorb_output(w_v: torch.Tensor, w_o: torch.Tensor) -> torch.Tensor:
    """Fold the value half into ``o_proj``; returns ``[D, H * KV_LORA]``."""
    raise NotImplementedError("MLA output absorb golden is assigned with the kernel")


__all__ = [
    "golden_absorb_output",
    "golden_absorb_query",
    "golden_mla_prolog",
    "golden_split_kv_b",
    "mla_prolog",
]
