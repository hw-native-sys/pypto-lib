# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Eco-Tech W8A8 SWA Prefill with BF16 grouped output projections."""

import pypto.language as pl
import torch

from .ecotech_linear import make_ecotech_dynamic_linear, prepare_ecotech_dynamic_weight
from .hc import make_hc_pre_inline
from .ops import make_norm, make_rope
from .rmsnorm import make_prefill_norm
from .swa import make_swa_inline


# model config
GEOMETRY = dict(
    hidden_size=5120, num_attention_heads=64, num_key_value_heads=1,
    head_dim=512, qk_rope_head_dim=64, q_lora_rank=1280, o_lora_rank=1024,
    o_groups=8, sliding_window=128, rms_norm_eps=1e-20, hc_mult=4,
    hc_sinkhorn_iters=20, hc_eps=1e-6,
)


def prepare_product_operands(raw):
    """Adapt CPU source weights to complete KN and grouped Prefill layouts."""
    prepared = {}
    for projection in ("wq_a", "wq_b", "wkv"):
        prefix = f"layers.0.attn.{projection}."
        weight, scale = prepare_ecotech_dynamic_weight(
            raw[prefix + "weight"], raw[prefix + "weight_scale"], raw[prefix + "weight_offset"],
        )
        prepared[projection] = weight
        prepared[projection + "_scale"] = scale
    wo_a, wo_b = raw["layers.0.attn.wo_a.weight"], raw["layers.0.attn.wo_b.weight"]
    if wo_a.dtype != torch.bfloat16 or wo_b.dtype != torch.bfloat16:
        raise ValueError("Output projections require original BF16 weights")
    if tuple(wo_a.shape) != (8192, 4096) or tuple(wo_b.shape) != (5120, 8192):
        raise ValueError("Output projection geometry differs")
    prepared["wo_a"] = wo_a.reshape(8, 1024, 4096).transpose(1, 2).contiguous()
    prepared["wo_a_grouped_prefill"] = wo_a.reshape(8, 1024, 4096).contiguous()
    prepared["wo_b"] = wo_b.T.contiguous()
    names = dict(
        attn_norm="layers.0.attn_norm.weight", q_norm="layers.0.attn.q_norm.weight",
        kv_norm="layers.0.attn.kv_norm.weight", sink="layers.0.attn.attn_sink",
        hc_function="layers.0.hc_attn_fn", hc_scale="layers.0.hc_attn_scale", hc_base="layers.0.hc_attn_base",
    )
    for target, source in names.items():
        prepared[target] = raw[source].contiguous()
    if any(value.device.type != "cpu" for value in prepared.values()):
        raise ValueError("Operand preparation requires CPU tensors")
    return prepared


def make_builders(capacity=16, config=None):
    """Build twelve Prefill stages with explicit predecessor and cache buffers."""
    from . import grouped_output as grouped
    from . import qkv_proj_rope as shared

    config = dict(GEOMETRY) if config is None else config
    if any(config.get(key) != value for key, value in GEOMETRY.items()):
        raise ValueError("Retain complete Eco-Tech SWA geometry")
    if capacity <= 0 or capacity % 16:
        raise ValueError("Capacity must cover complete 16-row tiles")
    dimensions = (
        grouped.TP_SIZE, grouped.LOCAL_H, grouped.LOCAL_O_GROUPS,
        grouped.LOCAL_O_WIDTH, grouped.O_GROUP_IN, grouped.O_LORA,
    )
    if dimensions != (1, 64, 8, 8192, 4096, 1024):
        raise ValueError("This full-width composition requires TP1")
    pre = make_hc_pre_inline(capacity, config)
    norm = pl.jit.inline(make_prefill_norm(capacity, 5120)._func)

    @pl.jit
    def collapse_norm(
        x: pl.Tensor[[capacity, 4, 5120], pl.BF16],
        hc_function: pl.Tensor[[24, 20480], pl.FP32],
        hc_scale: pl.Tensor[[3], pl.FP32],
        hc_base: pl.Tensor[[24], pl.FP32],
        incoming: pl.Tensor[[capacity, 4], pl.FP32],
        norm_weight: pl.Tensor[[5120], pl.BF16],
        next_pre: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        post: pl.InOut[pl.Tensor[[capacity, 4], pl.FP32]],
        comb: pl.InOut[pl.Tensor[[capacity, 4, 4], pl.FP32]],
        collapsed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        normed: pl.InOut[pl.Tensor[[capacity, 5120], pl.BF16]],
        active: pl.Scalar[pl.INT32],
    ):
        collapsed_fp32 = pl.create_tensor([capacity, 5120], dtype=pl.FP32)
        pre(
            x, hc_function, hc_scale, hc_base, incoming, next_pre, post, comb,
            collapsed, collapsed_fp32, active,
        )
        norm(collapsed, norm_weight, normed, active)
        return next_pre, post, comb, collapsed, normed

    def standalone(entry):
        return pl.jit(entry._func)

    return dict(
        collapse_norm=collapse_norm,
        wq_a=make_ecotech_dynamic_linear(capacity, 5120, 1280),
        q_norm=standalone(make_norm(capacity, 1280, 1e-20)),
        wq_b=make_ecotech_dynamic_linear(capacity, 1280, 32768, combine_scales=True),
        q_rope=standalone(make_rope(capacity, 64, 512, 64)),
        wkv=make_ecotech_dynamic_linear(capacity, 5120, 512),
        kv_norm=standalone(make_norm(capacity, 512, 1e-20)),
        kv_rope=standalone(make_rope(capacity, 1, 512, 64)),
        swa=standalone(make_swa_inline(capacity, 2, config)),
        inverse_rope=standalone(shared.make_rope(64, inverse=True, head_dim=512, rope_dim=64)),
        grouped_wo_a=standalone(grouped.prefill_grouped_output),
        wo_b=standalone(shared.make_bf16_projection_staged(8192, 5120)),
    )
