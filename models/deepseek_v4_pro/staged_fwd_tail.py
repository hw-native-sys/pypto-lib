# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Per-rank Hyper-Connections head and final RMSNorm for staged bring-up."""

import pypto.language as pl
import pypto.language.distributed as pld

from config import DECODE_TOKENS, PREFILL_TOKENS, PRO_KERNEL as MODEL_CONFIG
from hc_head import hc_head
from moe import N_RANKS
from rmsnorm import rms_norm


# Model configuration.
D = MODEL_CONFIG.hidden_size
HC_MULT = MODEL_CONFIG.hc_mult
HC_DIM = MODEL_CONFIG.hc_dim


@pl.jit
def decode_fwd_tail(
    x_hc: pl.Tensor[[DECODE_TOKENS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[DECODE_TOKENS, D], pl.BF16]],
):
    x_head = pl.create_tensor([DECODE_TOKENS, D], dtype=pl.BF16)
    hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, x_head)
    rms_norm(x_head, final_norm_w, hidden_out)
    return hidden_out


@pl.jit.host
def l3_decode_fwd_tail(
    x_hc: pl.Tensor[[N_RANKS, DECODE_TOKENS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, DECODE_TOKENS, D], pl.BF16]],
):
    for rank in pl.range(pld.world_size()):
        decode_fwd_tail(
            x_hc[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank],
            final_norm_w[rank],
            hidden_out[rank],
            device=rank,
        )


@pl.jit
def prefill_fwd_tail(
    x_hc: pl.Tensor[[PREFILL_TOKENS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[PREFILL_TOKENS, D], pl.BF16]],
):
    x_head = pl.create_tensor([PREFILL_TOKENS, D], dtype=pl.BF16)
    hc_head(x_hc, hc_head_fn, hc_head_scale, hc_head_base, x_head)
    rms_norm(x_head, final_norm_w, hidden_out)
    return hidden_out


@pl.jit.host
def l3_prefill_fwd_tail(
    x_hc: pl.Tensor[[N_RANKS, PREFILL_TOKENS, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, PREFILL_TOKENS, D], pl.BF16]],
):
    for rank in pl.range(pld.world_size()):
        prefill_fwd_tail(
            x_hc[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank],
            final_norm_w[rank],
            hidden_out[rank],
            device=rank,
        )


def build_tensor_specs(token_count):
    """Build the replicated tail weights and one rank-stacked activation tile."""
    import torch
    from golden import TensorSpec

    if token_count not in (DECODE_TOKENS, PREFILL_TOKENS):
        raise ValueError(
            f"tail token count must be decode ({DECODE_TOKENS}) or prefill "
            f"({PREFILL_TOKENS}), got {token_count}"
        )

    base = [5.9166, -3.6223, -2.9324, -3.3124]
    if len(base) != HC_MULT:
        raise ValueError(
            f"hc_head_base fixture has {len(base)} entries; expected HC_MULT={HC_MULT}"
        )
    specs = [
        TensorSpec(
            "x_hc", [N_RANKS, token_count, HC_MULT, D], torch.float32,
            init_value=lambda: torch.zeros(N_RANKS, token_count, HC_MULT, D),
        ),
        TensorSpec(
            "hc_head_fn", [N_RANKS, HC_MULT, HC_DIM], torch.float32,
            init_value=lambda: torch.randn(N_RANKS, HC_MULT, HC_DIM) * 0.0519,
        ),
        TensorSpec(
            "hc_head_scale", [N_RANKS, 1], torch.float32,
            init_value=lambda: torch.full((N_RANKS, 1), 0.076099, dtype=torch.float32),
        ),
        TensorSpec(
            "hc_head_base", [N_RANKS, HC_MULT], torch.float32,
            init_value=lambda: torch.tensor(base, dtype=torch.float32)
            .view(1, HC_MULT)
            .expand(N_RANKS, -1)
            .contiguous(),
        ),
        TensorSpec(
            "final_norm_w", [N_RANKS, D], torch.bfloat16,
            init_value=lambda: (torch.randn(N_RANKS, D) * 0.1 + 1.0).to(torch.bfloat16),
        ),
        TensorSpec("hidden_out", [N_RANKS, token_count, D], torch.bfloat16, is_output=True),
    ]
    for spec in specs[1:5]:
        spec.resident = "stacked"
    return specs
