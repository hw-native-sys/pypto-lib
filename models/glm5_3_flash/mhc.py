# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC coefficient generation, stream collapse, residual expansion, and final collapse.

GLM-5.3-Flash carries **two** hyper-connection sites per backbone layer — one around
attention (``hc_attn_fn`` / ``hc_attn_scale`` / ``hc_attn_base``) and one around the
MLP (``hc_ffn_*``) — for 90 sites over 45 layers. The MTP layer has no ``hc_*``
weights and uses a plain residual instead.

``Glm5NextTextHyperConnection`` subclasses ``DeepseekV4HyperConnection`` with no
changes, so the coefficient algebra is the same as ``models/deepseek_v4_1_flash/mhc.py``.
Two things differ and both change the ABI:

* **The stream is BF16, not FP32.** It is seeded by replicating the BF16 embedding
  (``inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1)``) and the decoder layer
  casts the mixes into the stream dtype. DeepSeek-V4 keeps its HC stream in FP32, so
  a kernel cannot be lifted across without changing the tile dtypes. At width 4 this
  is 32 KB of live residual per token instead of 64 KB.
* **The head is an unweighted mean** (``Glm5NextTextHyperHead``), where DeepSeek-V4
  applies the last layer's delayed pre-mix, so ``mhc_head`` takes no mix input. The
  model tail is ``norm(hc_head(streams))``.

The coefficients themselves stay FP32: the projection accumulates in FP32 and the
two sigmoids, the softmax and all 20 Sinkhorn iterations must not be computed in
BF16. Only ``function`` is stored BF16 — checkpoint shapes, read from the
safetensors headers, are ``hc_{attn,ffn}_fn`` BF16 ``[24, 16384]``,
``hc_{attn,ffn}_scale`` FP32 ``[3]`` and ``hc_{attn,ffn}_base`` FP32 ``[24]``.

**Prior art.** ``ops/pypto_python/impl/hc_pre_pypto.py`` in cann-recipes-infer is a
pypto implementation of the hyper-connection pre-collapse on this hardware
generation. It targets the older ``pypto.Tensor`` / ``pypto_impl`` frontend, which
this repo's pinned pypto does not export, so treat it as an algorithm and tiling
reference rather than code.

**Donors, in the right order.** ``models/deepseek_v4_flash_mtp/hc_pre.py`` (434
lines), ``hc_post.py`` (237) and ``hc_head.py`` (288) are implemented, carry no
``# ci:`` tag — so the a2a3 daily sweep runs them — and are the ones to port.
``models/deepseek_v4_flash_dspark/`` has a second implemented set (573 / 239 / 386)
for the larger batch. ``models/deepseek_v4_1_flash/mhc.py`` (559 lines, landed as
#1247) consolidates the four into one file and is the cleanest read, **but it is
a5-only** (``# ci: no-sim`` + ``# ci: a5``, and it splits the entry sentinel so the
a2a3 sweep cannot pick it up), so take its structure, not its tiling.

Against any of them the port is: the FP32 stream becomes BF16, the learned head
becomes an unweighted mean, and the router-side scoring is unrelated.
"""

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, HC_DIM, HC_MULT, MIX_HC, T_DYN
from models.glm5_3_flash.golden import hc_head, hc_mixes, hc_post, hc_pre


def golden_mhc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return hc_mixes(x_hc, function, scale, base)


def golden_mhc_pre(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    return hc_pre(x_hc, pre_mix).to(torch.bfloat16)


def golden_mhc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    return hc_post(sublayer, residual, post_mix, residual_mix).to(torch.bfloat16)


def golden_mhc_head(x_hc: torch.Tensor) -> torch.Tensor:
    return hc_head(x_hc)


@pl.jit.inline
def mhc_mixes(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.BF16],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
):
    raise NotImplementedError("mHC coefficient kernel body is assigned independently")


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("mHC pre kernel body is assigned independently")


@pl.jit.inline
def mhc_post(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
):
    raise NotImplementedError("mHC post kernel body is assigned independently")


@pl.jit.inline
def mhc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.BF16],
):
    raise NotImplementedError("mHC head kernel body is assigned independently")


__all__ = [
    "golden_mhc_head",
    "golden_mhc_mixes",
    "golden_mhc_post",
    "golden_mhc_pre",
    "mhc_head",
    "mhc_mixes",
    "mhc_post",
    "mhc_pre",
]


if __name__ == "__main__":
    from models.glm5_3_flash._golden_smoke import run_mhc_goldens

    run_mhc_goldens(golden_mhc_mixes, golden_mhc_pre, golden_mhc_post, golden_mhc_head)
