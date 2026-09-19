# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode Block skeleton and mode-selected Attention-stage hardware validation."""

# ci: no-sim
# ci: a5

import argparse
import inspect
import sys
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from pypto.ir import DistributedConfig
from golden import TensorSpec, ratio_allclose, run
from models.deepseek_v4_1_flash import decode_c2a_full as full
from models.deepseek_v4_1_flash import decode_c2a_reuse as reuse
from models.deepseek_v4_1_flash import decode_swa as swa
from models.deepseek_v4_1_flash.config import (
    AUX_WIDTH,
    D,
    DECODE_MAX_TOKENS,
    EP_SIZE,
    HC_MULT,
    MX_GROUP,
    N_LOCAL_EXPERTS,
    RECV_MAX,
    ROUTE_WIDTH,
    TOPK,
    TP_SIZE,
)

from models.deepseek_v4_1_flash import config as C
from models.deepseek_v4_1_flash.attention_common import AttentionGoldenResult
from models.deepseek_v4_1_flash.config import AttentionMode
from models.deepseek_v4_1_flash.decode_c1a_full import golden_decode_c1a_full
from models.deepseek_v4_1_flash.decode_c1a_reindex import golden_decode_c1a_reindex
from models.deepseek_v4_1_flash.decode_c1a_reuse import golden_decode_c1a_reuse
from models.deepseek_v4_1_flash.decode_c2a_full import decode_c2a_full, golden_decode_c2a_full
from models.deepseek_v4_1_flash.decode_c2a_reuse import decode_c2a_reuse, golden_decode_c2a_reuse
from models.deepseek_v4_1_flash.decode_swa import decode_swa, golden_decode_swa, make_norm
from models.deepseek_v4_1_flash.golden import rms_norm
from models.deepseek_v4_1_flash.hc_mixes import golden_mhc_mixes, mhc_mixes
from models.deepseek_v4_1_flash.hc_post import golden_mhc_post, mhc_post
from models.deepseek_v4_1_flash.hc_pre import golden_mhc_pre, mhc_pre
from models.deepseek_v4_1_flash.moe import golden_moe, moe


class DecodeLayerKind(IntEnum):
    """Static decode implementation selected for one backbone layer."""

    SWA = 0
    C2A_FULL = 1
    C2A_REUSE = 2
    C1A_FULL = 3
    C1A_REINDEX = 4
    C1A_REUSE = 5


@dataclass(frozen=True)
class DecodeLayerPlan:
    """Resolved attention implementation and source ownership for one layer."""

    layer_id: int
    kind: DecodeLayerKind
    compression_ratio: int
    kv_source_layer_id: int | None
    index_source_layer_id: int | None
    is_candidate_source: bool


@dataclass(frozen=True)
class DecodeLayerGoldenResult:
    """Block outputs, intermediate boundaries, and attention state updates."""

    output: torch.Tensor
    next_pre_mix: torch.Tensor
    attention_input: torch.Tensor
    attention_output: torch.Tensor
    attention_hidden: torch.Tensor
    ffn_input: torch.Tensor
    ffn_output: torch.Tensor
    attention: AttentionGoldenResult


REPRESENTATIVE_LAYER_IDS = {
    DecodeLayerKind.SWA: 0,
    DecodeLayerKind.C2A_FULL: 2,
    DecodeLayerKind.C2A_REUSE: 3,
    DecodeLayerKind.C1A_FULL: 20,
    DecodeLayerKind.C1A_REINDEX: 24,
    DecodeLayerKind.C1A_REUSE: 21,
}

_ATTENTION_GOLDENS = {
    DecodeLayerKind.SWA: golden_decode_swa,
    DecodeLayerKind.C2A_FULL: golden_decode_c2a_full,
    DecodeLayerKind.C2A_REUSE: golden_decode_c2a_reuse,
    DecodeLayerKind.C1A_FULL: golden_decode_c1a_full,
    DecodeLayerKind.C1A_REINDEX: golden_decode_c1a_reindex,
    DecodeLayerKind.C1A_REUSE: golden_decode_c1a_reuse,
}

# Readiness includes composition ABI agreement and integration acceptance.
_ATTENTION_KERNEL_READY = {
    DecodeLayerKind.SWA: True,
    DecodeLayerKind.C2A_FULL: True,
    DecodeLayerKind.C2A_REUSE: True,
    DecodeLayerKind.C1A_FULL: False,
    DecodeLayerKind.C1A_REINDEX: False,
    DecodeLayerKind.C1A_REUSE: False,
}

_MOE_KERNEL_READY = False

normalize_attention = make_norm(C.D)


def resolve_decode_layer_plan(layer_id: int) -> DecodeLayerPlan:
    """Resolve one layer through the checkpoint-backed model configuration."""
    layer = C.FLASH.layer_config(layer_id)
    if layer.mode == AttentionMode.SWA:
        kind = DecodeLayerKind.SWA
    elif layer.compression_ratio == 2 and layer.mode == AttentionMode.FULL:
        kind = DecodeLayerKind.C2A_FULL
    elif layer.compression_ratio == 2 and layer.mode == AttentionMode.REUSE:
        kind = DecodeLayerKind.C2A_REUSE
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.FULL:
        kind = DecodeLayerKind.C1A_FULL
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.REINDEX:
        kind = DecodeLayerKind.C1A_REINDEX
    elif layer.compression_ratio == 1 and layer.mode == AttentionMode.REUSE:
        kind = DecodeLayerKind.C1A_REUSE
    else:
        raise ValueError(
            f"unsupported decode layer {layer_id}: ratio={layer.compression_ratio}, mode={layer.mode.value}"
        )
    return DecodeLayerPlan(
        layer_id=layer_id,
        kind=kind,
        compression_ratio=layer.compression_ratio,
        kv_source_layer_id=layer.kv_source_layer_id,
        index_source_layer_id=layer.index_source_layer_id,
        is_candidate_source=layer.is_candidate_source,
    )


def decode_layer_kernel_skip_reason(layer_id: int) -> str | None:
    """Return the D1/D2 dependency that prevents device compilation."""
    plan = resolve_decode_layer_plan(layer_id)
    missing = []
    if not _ATTENTION_KERNEL_READY[plan.kind]:
        missing.append(f"{plan.kind.name} attention kernel integration")
        missing.append("C1A cache ABI agreement")
    if not _MOE_KERNEL_READY:
        missing.append("EP8 MoE kernel integration")
    return None if not missing else "decode_layer requires " + " and ".join(missing)


def decode_layer_attention_inputs(layer_id: int) -> tuple[str, ...]:
    """Return the exact golden inputs consumed by the resolved attention mode."""
    golden_fn = _ATTENTION_GOLDENS[resolve_decode_layer_plan(layer_id).kind]
    return tuple(inspect.signature(golden_fn).parameters)


def _select_inputs(function, values: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    selected = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in overrides:
            selected[name] = overrides[name]
        elif name in values:
            selected[name] = values[name]
        elif parameter.default is not inspect.Parameter.empty:
            continue
        else:
            raise KeyError(f"missing {function.__name__} input {name}")
    return selected


def golden_decode_layer(
    layer_id: int,
    x_hc: torch.Tensor,
    incoming_pre_mix: torch.Tensor,
    hc_attn_fn: torch.Tensor,
    hc_attn_scale: torch.Tensor,
    hc_attn_base: torch.Tensor,
    attn_norm_weight: torch.Tensor,
    hc_ffn_fn: torch.Tensor,
    hc_ffn_scale: torch.Tensor,
    hc_ffn_base: torch.Tensor,
    ffn_norm_weight: torch.Tensor,
    attention_inputs: Mapping[str, Any],
    moe_inputs: Mapping[str, Any],
    num_tokens: int | None = None,
) -> DecodeLayerGoldenResult:
    """Evaluate the official mHC-attention-mHC-MoE-mHC Block order."""
    if num_tokens is not None and num_tokens != x_hc.shape[0]:
        raise ValueError("Block golden currently requires all capacity rows to be active")
    plan = resolve_decode_layer_plan(layer_id)
    attn_pre, attn_post, attn_residual = golden_mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)
    attention_input = golden_mhc_pre(x_hc, incoming_pre_mix)
    normalized_attention = rms_norm(attention_input, attn_norm_weight)
    attention_fn = _ATTENTION_GOLDENS[plan.kind]
    attention_kwargs = _select_inputs(attention_fn, attention_inputs, {"x": normalized_attention})
    attention = attention_fn(**attention_kwargs)
    attention_hidden = golden_mhc_post(attention.output, x_hc, attn_post, attn_residual)

    next_pre_mix, ffn_post, ffn_residual = golden_mhc_mixes(
        attention_hidden, hc_ffn_fn, hc_ffn_scale, hc_ffn_base
    )
    ffn_input = golden_mhc_pre(attention_hidden, attn_pre)
    moe_overrides = {"x": ffn_input, "norm_weight": ffn_norm_weight}
    if num_tokens is not None:
        moe_overrides["num_tokens"] = num_tokens
    moe_kwargs = _select_inputs(golden_moe, moe_inputs, moe_overrides)
    ffn_output = golden_moe(**moe_kwargs)
    output = golden_mhc_post(ffn_output, attention_hidden, ffn_post, ffn_residual)
    return DecodeLayerGoldenResult(
        output=output,
        next_pre_mix=next_pre_mix,
        attention_input=attention_input,
        attention_output=attention.output,
        attention_hidden=attention_hidden,
        ffn_input=ffn_input,
        ffn_output=ffn_output,
        attention=attention,
    )


def make_block_rank(layer_id, epochs=1):
    """Build a Block around the same selected Attention routine."""
    reason = decode_layer_kernel_skip_reason(layer_id)
    if reason:
        raise NotImplementedError(reason)
    attention_rank = make_attention_rank(layer_id, epochs)

    @pl.jit
    def decode_layer(
        x_hc: pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32],
        incoming_pre_mix: pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[3], pl.FP32],
        hc_attn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[C.D], pl.BF16],
        wq_a: pl.Tensor[[C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.T_DYN, C.BLOCK_SIZE], pl.INT32],
        window_cache: pl.InOut[pl.Tensor[[C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM], pl.FP8E4M3FN]],
        window_cache_scale: pl.InOut[
            pl.Tensor[[C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0]
        ],
        compressed_cache: pl.InOut[pl.Tensor],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[
                [C.CMP_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP], pl.FP8E4M3FN
            ]
        ],
        request_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor],
        index_cache_scale: pl.InOut[
            pl.Tensor[[C.INDEX_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP], pl.FP8E8M0]
        ],
        index_block_table: pl.Tensor[[C.B_DYN, C.TABLE_DYN], pl.INT32],
        position_ids: pl.Tensor[[C.T_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor,
        compressor_wgate: pl.Tensor[[C.D, C.HEAD_DIM], pl.FP32],
        compressor_state_rows: pl.Tensor[[C.T_DYN], pl.INT64],
        compressor_state: pl.InOut[pl.Tensor[[C.MAX_BATCH_PER_DP, C.STATE_HEADS, C.HEAD_DIM], pl.FP32]],
        compressor_norm_weight: pl.Tensor[[C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[
            [C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ],
        index_weights_proj: pl.Tensor[[C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.InOut[pl.Tensor[[C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        candidate_mask: pl.InOut[pl.Tensor[[C.T_DYN, C.CMP_POSITIONS_DYN], pl.BOOL]],
        hc_ffn_fn: pl.Tensor[[C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[3], pl.FP32],
        hc_ffn_base: pl.Tensor[[C.MIX_HC], pl.FP32],
        ffn_norm_weight: pl.Tensor[[C.D], pl.BF16],
        gate_weight: pl.Tensor[[C.N_EXPERTS, C.D], pl.FP32],
        correction_bias: pl.Tensor[[C.N_EXPERTS], pl.FP32],
        routed_w1: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
        routed_w1_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
        routed_w2: pl.Tensor[[C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER], pl.FP4],
        routed_w2_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER // C.MX_GROUP], pl.FP8E8M0],
        routed_w3: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
        routed_w3_scale: pl.Tensor[[C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0],
        shared_w1: pl.Tensor[[C.D, C.MOE_INTER], pl.FP8E4M3FN],
        shared_w1_scale: pl.Tensor[[C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        shared_w2: pl.Tensor[[C.MOE_INTER, C.D], pl.FP8E4M3FN],
        shared_w2_scale: pl.Tensor[[C.MOE_INTER // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
        shared_w3: pl.Tensor[[C.D, C.MOE_INTER], pl.FP8E4M3FN],
        shared_w3_scale: pl.Tensor[[C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
        token_owners: pl.Tensor[[C.T_DYN], pl.INT32],
        attention_output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
        attention_output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        recv_meta: pld.DistributedTensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS], pl.INT32],
        recv_x: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D], pl.FP8E4M3FN],
        recv_scale: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.D // C.MX_GROUP], pl.UINT8],
        recv_weights: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.AUX_WIDTH], pl.FP32],
        recv_routes: pld.DistributedTensor[[C.N_LOCAL_EXPERTS * C.RECV_MAX, C.ROUTE_WIDTH], pl.INT32],
        arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
        data_arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
        routed_output: pld.DistributedTensor[[C.T_DYN * C.TOPK, C.D], pl.FP32],
        combine_arrived: pld.DistributedTensor[[C.EP_SIZE, 1], pl.INT32],
        x_next: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[C.T_DYN, C.HC_MULT], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
        ep_rank: pl.Scalar[pl.INT32],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x_hc, 0)
        attn_pre = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
        attention_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
        normalized_attention = pl.create_tensor([tokens, D], dtype=pl.BF16)
        attention_output = pl.create_tensor([tokens, D], dtype=pl.BF16)
        attention_hidden = pl.create_tensor([tokens, HC_MULT, D], dtype=pl.FP32)
        attention_rank(
            x_hc,
            incoming_pre_mix,
            hc_attn_fn,
            hc_attn_scale,
            hc_attn_base,
            attn_norm_weight,
            wq_a,
            wq_a_scale,
            q_norm_weight,
            wq_b,
            wq_b_scale,
            wkv,
            wkv_scale,
            kv_norm_weight,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            rope_cos,
            rope_sin,
            window_slots,
            window_indices,
            window_cache,
            window_cache_scale,
            compressed_cache,
            compressed_cache_scale,
            request_ids,
            compressed_lens,
            index_cache,
            index_cache_scale,
            index_block_table,
            position_ids,
            compressed_rope_cos,
            compressed_rope_sin,
            compressor_wkv,
            compressor_wgate,
            compressor_state_rows,
            compressor_state,
            compressor_norm_weight,
            compressed_slots,
            index_wk,
            index_norm_weight,
            index_wq_b,
            index_wq_b_scale,
            index_weights_proj,
            topk_indices,
            attention_input,
            normalized_attention,
            attention_output,
            attention_hidden,
            attn_pre,
            attention_output_window,
            attention_output_arrived,
            ep_rank,
            num_tokens,
            attention_epoch,
        )
        ffn_post = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
        ffn_residual = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
        mhc_mixes(
            attention_hidden,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
            next_pre_mix,
            ffn_post,
            ffn_residual,
        )
        ffn_input = pl.create_tensor([tokens, D], dtype=pl.BF16)
        mhc_pre(attention_hidden, attn_pre, ffn_input)
        ffn_output = pl.create_tensor([tokens, D], dtype=pl.BF16)
        moe(
            ffn_input,
            ffn_norm_weight,
            gate_weight,
            correction_bias,
            routed_w1,
            routed_w1_scale,
            routed_w2,
            routed_w2_scale,
            routed_w3,
            routed_w3_scale,
            shared_w1,
            shared_w1_scale,
            shared_w2,
            shared_w2_scale,
            shared_w3,
            shared_w3_scale,
            token_owners,
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            ffn_output,
            num_tokens,
            ep_rank,
            group_base,
            tp_rank,
            moe_epoch,
        )
        mhc_post(ffn_output, attention_hidden, ffn_post, ffn_residual, x_next)
        return x_next, next_pre_mix

    return decode_layer


def make_block_program(layer_id, world_size, epochs=1):
    """Build the full Block only after all selected dependencies are ready."""
    if world_size != C.EP_SIZE:
        raise ValueError("Block world size must match EP_SIZE")
    decode_layer = make_block_rank(layer_id, epochs)

    @pl.jit.host
    def l3_decode_layer(
        x_hc: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT, C.D], pl.FP32],
        incoming_pre_mix: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT], pl.FP32],
        hc_attn_fn: pl.Tensor[[C.EP_SIZE, C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_attn_scale: pl.Tensor[[C.EP_SIZE, 3], pl.FP32],
        hc_attn_base: pl.Tensor[[C.EP_SIZE, C.MIX_HC], pl.FP32],
        attn_norm_weight: pl.Tensor[[C.EP_SIZE, C.D], pl.BF16],
        wq_a: pl.Tensor[[C.EP_SIZE, C.D, C.Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0],
        q_norm_weight: pl.Tensor[[C.EP_SIZE, C.Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[C.EP_SIZE, C.Q_LORA, C.LOCAL_H * C.HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[C.EP_SIZE, C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0],
        wkv: pl.Tensor[[C.EP_SIZE, C.D, C.HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0],
        kv_norm_weight: pl.Tensor[[C.EP_SIZE, C.HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[C.EP_SIZE, C.LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[C.EP_SIZE, C.LOCAL_O_GROUPS, C.O_LORA, C.O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[C.EP_SIZE, C.LOCAL_O_WIDTH, C.D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[C.EP_SIZE, C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0],
        rope_cos: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT64],
        window_indices: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.BLOCK_SIZE], pl.INT32],
        window_cache: pl.InOut[
            pl.Tensor[[C.EP_SIZE, C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM], pl.FP8E4M3FN]
        ],
        window_cache_scale: pl.InOut[
            pl.Tensor[
                [C.EP_SIZE, C.ORI_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.WINDOW_CACHE_GROUP], pl.FP8E8M0
            ]
        ],
        compressed_cache: pl.InOut[pl.Tensor],
        compressed_cache_scale: pl.InOut[
            pl.Tensor[
                [C.EP_SIZE, C.CMP_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.HEAD_DIM // C.COMPRESSED_CACHE_GROUP],
                pl.FP8E4M3FN,
            ]
        ],
        request_ids: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
        compressed_lens: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
        index_cache: pl.InOut[pl.Tensor],
        index_cache_scale: pl.InOut[
            pl.Tensor[
                [C.EP_SIZE, C.INDEX_BLOCKS_DYN, C.BLOCK_SIZE, 1, C.INDEX_DIM // C.INDEX_CACHE_GROUP],
                pl.FP8E8M0,
            ]
        ],
        index_block_table: pl.Tensor[[C.EP_SIZE, C.B_DYN, C.TABLE_DYN], pl.INT32],
        position_ids: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
        compressed_rope_cos: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressed_rope_sin: pl.Tensor[[C.EP_SIZE, C.T_DYN, C.ROPE_DIM // 2], pl.FP32],
        compressor_wkv: pl.Tensor,
        compressor_wgate: pl.Tensor[[C.EP_SIZE, C.D, C.HEAD_DIM], pl.FP32],
        compressor_state_rows: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT64],
        compressor_state: pl.InOut[
            pl.Tensor[[C.EP_SIZE, C.MAX_BATCH_PER_DP, C.STATE_HEADS, C.HEAD_DIM], pl.FP32]
        ],
        compressor_norm_weight: pl.Tensor[[C.EP_SIZE, C.HEAD_DIM], pl.BF16],
        compressed_slots: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT64],
        index_wk: pl.Tensor[[C.EP_SIZE, C.HEAD_DIM, C.INDEX_DIM], pl.BF16],
        index_norm_weight: pl.Tensor[[C.EP_SIZE, C.INDEX_DIM], pl.BF16],
        index_wq_b: pl.Tensor[[C.EP_SIZE, C.Q_LORA, C.INDEX_H * C.INDEX_DIM], pl.FP8E4M3FN],
        index_wq_b_scale: pl.Tensor[[C.EP_SIZE, C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0],
        index_weights_proj: pl.Tensor[[C.EP_SIZE, C.D, C.INDEX_H], pl.BF16],
        topk_indices: pl.InOut[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.INDEX_TOPK], pl.INT32]],
        candidate_mask: pl.InOut[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.CMP_POSITIONS_DYN], pl.BOOL]],
        hc_ffn_fn: pl.Tensor[[C.EP_SIZE, C.MIX_HC, C.HC_DIM], pl.FP32],
        hc_ffn_scale: pl.Tensor[[C.EP_SIZE, 3], pl.FP32],
        hc_ffn_base: pl.Tensor[[C.EP_SIZE, C.MIX_HC], pl.FP32],
        ffn_norm_weight: pl.Tensor[[C.EP_SIZE, C.D], pl.BF16],
        gate_weight: pl.Tensor[[C.EP_SIZE, C.N_EXPERTS, C.D], pl.FP32],
        correction_bias: pl.Tensor[[C.EP_SIZE, C.N_EXPERTS], pl.FP32],
        routed_w1: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
        routed_w1_scale: pl.Tensor[
            [C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0
        ],
        routed_w2: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER], pl.FP4],
        routed_w2_scale: pl.Tensor[
            [C.EP_SIZE, C.N_LOCAL_EXPERTS, C.D, C.MOE_INTER // C.MX_GROUP], pl.FP8E8M0
        ],
        routed_w3: pl.Tensor[[C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D], pl.FP4],
        routed_w3_scale: pl.Tensor[
            [C.EP_SIZE, C.N_LOCAL_EXPERTS, C.MOE_INTER, C.D // C.MX_GROUP], pl.FP8E8M0
        ],
        shared_w1: pl.Tensor[[C.EP_SIZE, C.D, C.MOE_INTER], pl.FP8E4M3FN],
        shared_w1_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
        shared_w2: pl.Tensor[[C.EP_SIZE, C.MOE_INTER, C.D], pl.FP8E4M3FN],
        shared_w2_scale: pl.Tensor[[C.EP_SIZE, C.MOE_INTER // C.MX_GROUP, C.D], pl.FP8E8M0],
        shared_w3: pl.Tensor[[C.EP_SIZE, C.D, C.MOE_INTER], pl.FP8E4M3FN],
        shared_w3_scale: pl.Tensor[[C.EP_SIZE, C.D // C.MX_GROUP, C.MOE_INTER], pl.FP8E8M0],
        token_owners: pl.Tensor[[C.EP_SIZE, C.T_DYN], pl.INT32],
        x_next: pl.Out[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT, C.D], pl.FP32]],
        next_pre_mix: pl.Out[pl.Tensor[[C.EP_SIZE, C.T_DYN, C.HC_MULT], pl.FP32]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
        moe_epoch: pl.Scalar[pl.INT32],
    ):
        attention_output_buf = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        attention_arrived_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        recv_meta_buf = pld.alloc_window_buffer([EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.FP8E4M3FN)
        recv_scale_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8)
        recv_weights_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes_buf = pld.alloc_window_buffer([N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        data_arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
        routed_output_buf = pld.alloc_window_buffer([DECODE_MAX_TOKENS * TOPK, D], dtype=pl.FP32)
        combine_arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)

        for rank in pl.range(world_size):
            attention_output_window = pld.window(attention_output_buf, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
            attention_output_arrived = pld.window(attention_arrived_buf, [TP_SIZE, 1], dtype=pl.INT32)
            recv_meta = pld.window(recv_meta_buf, [EP_SIZE, N_LOCAL_EXPERTS], dtype=pl.INT32)
            recv_x = pld.window(recv_x_buf, [N_LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.FP8E4M3FN)
            recv_scale = pld.window(
                recv_scale_buf, [N_LOCAL_EXPERTS * RECV_MAX, D // MX_GROUP], dtype=pl.UINT8
            )
            recv_weights = pld.window(
                recv_weights_buf, [N_LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32
            )
            recv_routes = pld.window(
                recv_routes_buf, [N_LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32
            )
            arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
            data_arrived = pld.window(data_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
            routed_output = pld.window(routed_output_buf, [DECODE_MAX_TOKENS * TOPK, D], dtype=pl.FP32)
            combine_arrived = pld.window(combine_arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
            group_base = (rank // TP_SIZE) * TP_SIZE
            tp_rank = rank % TP_SIZE
            decode_layer(
                x_hc[rank],
                incoming_pre_mix[rank],
                hc_attn_fn[rank],
                hc_attn_scale[rank],
                hc_attn_base[rank],
                attn_norm_weight[rank],
                wq_a[rank],
                wq_a_scale[rank],
                q_norm_weight[rank],
                wq_b[rank],
                wq_b_scale[rank],
                wkv[rank],
                wkv_scale[rank],
                kv_norm_weight[rank],
                attn_sink[rank],
                wo_a[rank],
                wo_b[rank],
                wo_b_scale[rank],
                rope_cos[rank],
                rope_sin[rank],
                window_slots[rank],
                window_indices[rank],
                window_cache[rank],
                window_cache_scale[rank],
                compressed_cache[rank],
                compressed_cache_scale[rank],
                request_ids[rank],
                compressed_lens[rank],
                index_cache[rank],
                index_cache_scale[rank],
                index_block_table[rank],
                position_ids[rank],
                compressed_rope_cos[rank],
                compressed_rope_sin[rank],
                compressor_wkv[rank],
                compressor_wgate[rank],
                compressor_state_rows[rank],
                compressor_state[rank],
                compressor_norm_weight[rank],
                compressed_slots[rank],
                index_wk[rank],
                index_norm_weight[rank],
                index_wq_b[rank],
                index_wq_b_scale[rank],
                index_weights_proj[rank],
                topk_indices[rank],
                candidate_mask[rank],
                hc_ffn_fn[rank],
                hc_ffn_scale[rank],
                hc_ffn_base[rank],
                ffn_norm_weight[rank],
                gate_weight[rank],
                correction_bias[rank],
                routed_w1[rank],
                routed_w1_scale[rank],
                routed_w2[rank],
                routed_w2_scale[rank],
                routed_w3[rank],
                routed_w3_scale[rank],
                shared_w1[rank],
                shared_w1_scale[rank],
                shared_w2[rank],
                shared_w2_scale[rank],
                shared_w3[rank],
                shared_w3_scale[rank],
                token_owners[rank],
                attention_output_window,
                attention_output_arrived,
                recv_meta,
                recv_x,
                recv_scale,
                recv_weights,
                recv_routes,
                arrived,
                data_arrived,
                routed_output,
                combine_arrived,
                x_next[rank],
                next_pre_mix[rank],
                num_tokens,
                rank,
                group_base,
                tp_rank,
                attention_epoch,
                moe_epoch,
                device=rank,
            )

    return l3_decode_layer


def attention_half_skip_reason(layer_id):
    """Check only half-layer dependencies; MoE does not gate this entry."""
    kind = resolve_decode_layer_plan(layer_id).kind
    if not _ATTENTION_KERNEL_READY[kind]:
        return f"{kind.name} attention kernel integration and C1A cache ABI agreement are pending"
    return None


# The union adapters keep selection outside the JIT dependency graph.
@pl.jit.inline(auto_scope=False)
def _swa(
    x: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    attn_sink: pl.Tensor,
    wo_a: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor,
    window_cache_scale: pl.Tensor,
    compressed_cache: pl.Tensor,
    compressed_cache_scale: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    index_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    compressed_rope_cos: pl.Tensor,
    compressed_rope_sin: pl.Tensor,
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor,
    compressor_state_rows: pl.Tensor,
    compressor_state: pl.Tensor,
    compressor_norm_weight: pl.Tensor,
    compressed_slots: pl.Tensor,
    index_wk: pl.Tensor,
    index_norm_weight: pl.Tensor,
    index_wq_b: pl.Tensor,
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor,
    topk_indices: pl.Tensor,
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    decode_swa(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    return output


@pl.jit.inline(auto_scope=False)
def _c2a_full(
    x: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    attn_sink: pl.Tensor,
    wo_a: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor,
    window_cache_scale: pl.Tensor,
    compressed_cache: pl.Tensor,
    compressed_cache_scale: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    index_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    compressed_rope_cos: pl.Tensor,
    compressed_rope_sin: pl.Tensor,
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor,
    compressor_state_rows: pl.Tensor,
    compressor_state: pl.Tensor,
    compressor_norm_weight: pl.Tensor,
    compressed_slots: pl.Tensor,
    index_wk: pl.Tensor,
    index_norm_weight: pl.Tensor,
    index_wq_b: pl.Tensor,
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor,
    topk_indices: pl.Tensor,
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    decode_c2a_full(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        compressed_cache,
        compressed_cache_scale,
        request_ids,
        compressed_lens,
        index_cache,
        index_cache_scale,
        index_block_table,
        position_ids,
        compressed_rope_cos,
        compressed_rope_sin,
        compressor_wkv,
        compressor_wgate,
        compressor_state_rows,
        compressor_state,
        compressor_norm_weight,
        compressed_slots,
        index_wk,
        index_norm_weight,
        index_wq_b,
        index_wq_b_scale,
        index_weights_proj,
        topk_indices,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    return output


@pl.jit.inline(auto_scope=False)
def _c2a_reuse(
    x: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor,
    attn_sink: pl.Tensor,
    wo_a: pl.Tensor,
    wo_b: pl.Tensor,
    wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    window_slots: pl.Tensor,
    window_indices: pl.Tensor,
    window_cache: pl.Tensor,
    window_cache_scale: pl.Tensor,
    compressed_cache: pl.Tensor,
    compressed_cache_scale: pl.Tensor,
    request_ids: pl.Tensor,
    compressed_lens: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    index_block_table: pl.Tensor,
    position_ids: pl.Tensor,
    compressed_rope_cos: pl.Tensor,
    compressed_rope_sin: pl.Tensor,
    compressor_wkv: pl.Tensor,
    compressor_wgate: pl.Tensor,
    compressor_state_rows: pl.Tensor,
    compressor_state: pl.Tensor,
    compressor_norm_weight: pl.Tensor,
    compressed_slots: pl.Tensor,
    index_wk: pl.Tensor,
    index_norm_weight: pl.Tensor,
    index_wq_b: pl.Tensor,
    index_wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor,
    topk_indices: pl.Tensor,
    output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
    output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    output: pl.Tensor,
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    attention_epoch: pl.Scalar[pl.INT32],
):
    decode_c2a_reuse(
        x,
        wq_a,
        wq_a_scale,
        q_norm_weight,
        wq_b,
        wq_b_scale,
        wkv,
        wkv_scale,
        kv_norm_weight,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
        rope_cos,
        rope_sin,
        window_slots,
        window_indices,
        window_cache,
        window_cache_scale,
        compressed_cache,
        compressed_cache_scale,
        topk_indices,
        output_window,
        output_arrived,
        output,
        group_base,
        tp_rank,
        num_tokens,
        attention_epoch,
    )
    return output


def make_attention_rank(layer_id, epochs):
    """Select the leaf before JIT discovery; share orchestration across stages."""
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be in [1, 1000]")
    reason = attention_half_skip_reason(layer_id)
    if reason:
        raise NotImplementedError(reason)
    kind = resolve_decode_layer_plan(layer_id).kind
    attention = {
        DecodeLayerKind.SWA: _swa,
        DecodeLayerKind.C2A_FULL: _c2a_full,
        DecodeLayerKind.C2A_REUSE: _c2a_reuse,
    }[kind]

    @pl.jit
    def attention_rank(
        x_hc: pl.Tensor,
        incoming_pre_mix: pl.Tensor,
        hc_attn_fn: pl.Tensor,
        hc_attn_scale: pl.Tensor,
        hc_attn_base: pl.Tensor,
        attn_norm_weight: pl.Tensor,
        wq_a: pl.Tensor,
        wq_a_scale: pl.Tensor[[C.D // C.MX_GROUP, C.Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor,
        wq_b: pl.Tensor,
        wq_b_scale: pl.Tensor[[C.Q_LORA // C.MX_GROUP, C.LOCAL_H * C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor,
        wkv_scale: pl.Tensor[[C.D // C.MX_GROUP, C.HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor,
        attn_sink: pl.Tensor,
        wo_a: pl.Tensor,
        wo_b: pl.Tensor,
        wo_b_scale: pl.Tensor[[C.LOCAL_O_WIDTH // C.MX_GROUP, C.D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor,
        rope_sin: pl.Tensor,
        window_slots: pl.Tensor,
        window_indices: pl.Tensor,
        window_cache: pl.InOut[pl.Tensor],
        window_cache_scale: pl.InOut[pl.Tensor],
        compressed_cache: pl.InOut[pl.Tensor],
        compressed_cache_scale: pl.InOut[pl.Tensor],
        request_ids: pl.Tensor,
        compressed_lens: pl.Tensor,
        index_cache: pl.InOut[pl.Tensor],
        index_cache_scale: pl.InOut[pl.Tensor],
        index_block_table: pl.Tensor,
        position_ids: pl.Tensor,
        compressed_rope_cos: pl.Tensor,
        compressed_rope_sin: pl.Tensor,
        compressor_wkv: pl.Tensor,
        compressor_wgate: pl.Tensor,
        compressor_state_rows: pl.Tensor,
        compressor_state: pl.InOut[pl.Tensor],
        compressor_norm_weight: pl.Tensor,
        compressed_slots: pl.Tensor,
        index_wk: pl.Tensor,
        index_norm_weight: pl.Tensor,
        index_wq_b: pl.Tensor,
        index_wq_b_scale: pl.Tensor[
            [C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
        ],
        index_weights_proj: pl.Tensor,
        topk_indices: pl.InOut[pl.Tensor],
        attention_input: pl.Out[pl.Tensor],
        normalized_attention: pl.InOut[pl.Tensor],
        attention_output: pl.InOut[pl.Tensor],
        attention_hidden: pl.Out[pl.Tensor],
        attention_pre_mix: pl.Out[pl.Tensor],
        output_window: pld.DistributedTensor[[C.DECODE_MAX_TOKENS, C.D], pl.FP32],
        output_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
        rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x_hc, 0)
        post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
        residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
        mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
        mhc_pre(x_hc, incoming_pre_mix, attention_input)
        normalize_attention(attention_input, attn_norm_weight, normalized_attention, num_tokens)
        for step in pl.range(epochs):
            attention(
                normalized_attention,
                wq_a,
                wq_a_scale,
                q_norm_weight,
                wq_b,
                wq_b_scale,
                wkv,
                wkv_scale,
                kv_norm_weight,
                attn_sink,
                wo_a,
                wo_b,
                wo_b_scale,
                rope_cos,
                rope_sin,
                window_slots,
                window_indices,
                window_cache,
                window_cache_scale,
                compressed_cache,
                compressed_cache_scale,
                request_ids,
                compressed_lens,
                index_cache,
                index_cache_scale,
                index_block_table,
                position_ids,
                compressed_rope_cos,
                compressed_rope_sin,
                compressor_wkv,
                compressor_wgate,
                compressor_state_rows,
                compressor_state,
                compressor_norm_weight,
                compressed_slots,
                index_wk,
                index_norm_weight,
                index_wq_b,
                index_wq_b_scale,
                index_weights_proj,
                topk_indices,
                output_window,
                output_arrived,
                attention_output,
                rank // TP_SIZE * TP_SIZE,
                rank % TP_SIZE,
                num_tokens,
                attention_epoch + step,
            )
        mhc_post(attention_output, x_hc, post_mix, residual_mix, attention_hidden)
        return attention_hidden, attention_pre_mix

    return attention_rank


def make_attention_program(layer_id, world_size, epochs, specs):
    """Build one mode-specific half-layer with persistent communication epochs."""
    attention_rank = make_attention_rank(layer_id, epochs)
    if world_size != C.TP_SIZE:
        raise ValueError("Attention world size must match TP_SIZE")
    if tuple(spec.name for spec in specs) != ATTENTION_SPEC_NAMES:
        raise ValueError("Attention specs must match the host parameter names and order")
    dtypes = {
        torch.bfloat16: pl.BF16,
        torch.float32: pl.FP32,
        torch.float8_e4m3fn: pl.FP8E4M3FN,
        torch.float8_e8m0fnu: pl.FP8E8M0,
        torch.int32: pl.INT32,
        torch.int64: pl.INT64,
        torch.uint8: pl.UINT8,
    }
    shapes = [spec.shape for spec in specs if isinstance(spec, TensorSpec)]
    tensor_dtypes = [dtypes[spec.dtype] for spec in specs if isinstance(spec, TensorSpec)]

    @pl.jit.host
    def attention_group(
        x_hc: pl.Tensor[shapes[0], tensor_dtypes[0]],
        incoming_pre_mix: pl.Tensor[shapes[1], tensor_dtypes[1]],
        hc_attn_fn: pl.Tensor[shapes[2], tensor_dtypes[2]],
        hc_attn_scale: pl.Tensor[shapes[3], tensor_dtypes[3]],
        hc_attn_base: pl.Tensor[shapes[4], tensor_dtypes[4]],
        attn_norm_weight: pl.Tensor[shapes[5], tensor_dtypes[5]],
        wq_a: pl.Tensor[shapes[6], tensor_dtypes[6]],
        wq_a_scale: pl.Tensor[shapes[7], tensor_dtypes[7]],
        q_norm_weight: pl.Tensor[shapes[8], tensor_dtypes[8]],
        wq_b: pl.Tensor[shapes[9], tensor_dtypes[9]],
        wq_b_scale: pl.Tensor[shapes[10], tensor_dtypes[10]],
        wkv: pl.Tensor[shapes[11], tensor_dtypes[11]],
        wkv_scale: pl.Tensor[shapes[12], tensor_dtypes[12]],
        kv_norm_weight: pl.Tensor[shapes[13], tensor_dtypes[13]],
        attn_sink: pl.Tensor[shapes[14], tensor_dtypes[14]],
        wo_a: pl.Tensor[shapes[15], tensor_dtypes[15]],
        wo_b: pl.Tensor[shapes[16], tensor_dtypes[16]],
        wo_b_scale: pl.Tensor[shapes[17], tensor_dtypes[17]],
        rope_cos: pl.Tensor[shapes[18], tensor_dtypes[18]],
        rope_sin: pl.Tensor[shapes[19], tensor_dtypes[19]],
        window_slots: pl.Tensor[shapes[20], tensor_dtypes[20]],
        window_indices: pl.Tensor[shapes[21], tensor_dtypes[21]],
        window_cache: pl.InOut[pl.Tensor[shapes[22], tensor_dtypes[22]]],
        window_cache_scale: pl.InOut[pl.Tensor[shapes[23], tensor_dtypes[23]]],
        compressed_cache: pl.InOut[pl.Tensor[shapes[24], tensor_dtypes[24]]],
        compressed_cache_scale: pl.InOut[pl.Tensor[shapes[25], tensor_dtypes[25]]],
        request_ids: pl.Tensor[shapes[26], tensor_dtypes[26]],
        compressed_lens: pl.Tensor[shapes[27], tensor_dtypes[27]],
        index_cache: pl.InOut[pl.Tensor[shapes[28], tensor_dtypes[28]]],
        index_cache_scale: pl.InOut[pl.Tensor[shapes[29], tensor_dtypes[29]]],
        index_block_table: pl.Tensor[shapes[30], tensor_dtypes[30]],
        position_ids: pl.Tensor[shapes[31], tensor_dtypes[31]],
        compressed_rope_cos: pl.Tensor[shapes[32], tensor_dtypes[32]],
        compressed_rope_sin: pl.Tensor[shapes[33], tensor_dtypes[33]],
        compressor_wkv: pl.Tensor[shapes[34], tensor_dtypes[34]],
        compressor_wgate: pl.Tensor[shapes[35], tensor_dtypes[35]],
        compressor_state_rows: pl.Tensor[shapes[36], tensor_dtypes[36]],
        compressor_state: pl.InOut[pl.Tensor[shapes[37], tensor_dtypes[37]]],
        compressor_norm_weight: pl.Tensor[shapes[38], tensor_dtypes[38]],
        compressed_slots: pl.Tensor[shapes[39], tensor_dtypes[39]],
        index_wk: pl.Tensor[shapes[40], tensor_dtypes[40]],
        index_norm_weight: pl.Tensor[shapes[41], tensor_dtypes[41]],
        index_wq_b: pl.Tensor[shapes[42], tensor_dtypes[42]],
        index_wq_b_scale: pl.Tensor[shapes[43], tensor_dtypes[43]],
        index_weights_proj: pl.Tensor[shapes[44], tensor_dtypes[44]],
        topk_indices: pl.InOut[pl.Tensor[shapes[45], tensor_dtypes[45]]],
        attention_input: pl.Out[pl.Tensor[shapes[46], tensor_dtypes[46]]],
        normalized_attention: pl.InOut[pl.Tensor[shapes[47], tensor_dtypes[47]]],
        attention_output: pl.InOut[pl.Tensor[shapes[48], tensor_dtypes[48]]],
        attention_hidden: pl.Out[pl.Tensor[shapes[49], tensor_dtypes[49]]],
        attention_pre_mix: pl.Out[pl.Tensor[shapes[50], tensor_dtypes[50]]],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        data_buffer = pld.alloc_window_buffer([DECODE_MAX_TOKENS, D], dtype=pl.FP32)
        arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
        for rank in pl.range(world_size):
            data = pld.window(data_buffer, [DECODE_MAX_TOKENS, D], dtype=pl.FP32)
            arrived = pld.window(arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
            attention_rank(
                x_hc[rank],
                incoming_pre_mix[rank],
                hc_attn_fn[rank],
                hc_attn_scale[rank],
                hc_attn_base[rank],
                attn_norm_weight[rank],
                wq_a[rank],
                wq_a_scale[rank],
                q_norm_weight[rank],
                wq_b[rank],
                wq_b_scale[rank],
                wkv[rank],
                wkv_scale[rank],
                kv_norm_weight[rank],
                attn_sink[rank],
                wo_a[rank],
                wo_b[rank],
                wo_b_scale[rank],
                rope_cos[rank],
                rope_sin[rank],
                window_slots[rank],
                window_indices[rank],
                window_cache[rank],
                window_cache_scale[rank],
                compressed_cache[rank],
                compressed_cache_scale[rank],
                request_ids[rank],
                compressed_lens[rank],
                index_cache[rank],
                index_cache_scale[rank],
                index_block_table[rank],
                position_ids[rank],
                compressed_rope_cos[rank],
                compressed_rope_sin[rank],
                compressor_wkv[rank],
                compressor_wgate[rank],
                compressor_state_rows[rank],
                compressor_state[rank],
                compressor_norm_weight[rank],
                compressed_slots[rank],
                index_wk[rank],
                index_norm_weight[rank],
                index_wq_b[rank],
                index_wq_b_scale[rank],
                index_weights_proj[rank],
                topk_indices[rank],
                attention_input[rank],
                normalized_attention[rank],
                attention_output[rank],
                attention_hidden[rank],
                attention_pre_mix[rank],
                data,
                arrived,
                rank,
                num_tokens,
                attention_epoch,
                device=rank,
            )

    return attention_group


def make_decode_layer_program(layer_id, world_size, epochs, *, stage="attention", specs=None):
    """Select the stage in Python before building its JIT dependency graph."""
    if stage == "attention":
        if specs is None:
            raise ValueError("Attention host entry requires tensor specs")
        return make_attention_program(layer_id, world_size, epochs, specs)
    if stage == "block":
        return make_block_program(layer_id, world_size, epochs)
    raise ValueError(f"unknown decode stage: {stage}")


UNION_NAMES = (
    "wq_a",
    "wq_a_scale",
    "q_norm_weight",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "wkv_scale",
    "kv_norm_weight",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "rope_cos",
    "rope_sin",
    "window_slots",
    "window_indices",
    "window_cache",
    "window_cache_scale",
    "compressed_cache",
    "compressed_cache_scale",
    "request_ids",
    "compressed_lens",
    "index_cache",
    "index_cache_scale",
    "index_block_table",
    "position_ids",
    "compressed_rope_cos",
    "compressed_rope_sin",
    "compressor_wkv",
    "compressor_wgate",
    "compressor_state_rows",
    "compressor_state",
    "compressor_norm_weight",
    "compressed_slots",
    "index_wk",
    "index_norm_weight",
    "index_wq_b",
    "index_wq_b_scale",
    "index_weights_proj",
    "topk_indices",
)

ATTENTION_SPEC_NAMES = (
    "x_hc",
    "incoming_pre_mix",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_weight",
    *UNION_NAMES,
    "attention_input",
    "normalized_attention",
    "attention_output",
    "attention_hidden",
    "attention_pre_mix",
    "num_tokens",
    "attention_epoch",
)


def build_specs(args, kind, initial_cache):
    """Reuse leaf fixtures and add replicated mHC inputs and visible boundaries."""
    if kind == DecodeLayerKind.SWA:
        leaf_specs = swa.build_specs(args, "decode")
    elif kind == DecodeLayerKind.C2A_FULL:
        leaf_specs = full.build_specs(args, "decode")
    else:
        leaf_specs = reuse.build_specs(args, "decode", initial_cache)
        leaf_specs = [
            replace(spec, name="topk_indices") if spec.name == "compressed_indices" else spec
            for spec in leaf_specs
        ]
    specs = [spec for spec in leaf_specs if spec.name not in ("x", "output")]
    present = {spec.name for spec in specs}
    for name in UNION_NAMES:
        if name not in present:
            if name == "index_wq_b_scale":
                shape = [args.tp, C.Q_LORA // C.MX_GROUP, C.INDEX_H * C.INDEX_DIM]
                specs.append(TensorSpec(name, shape, torch.float8_e8m0fnu, resident="stacked"))
            else:
                specs.append(TensorSpec(name, [args.tp, 1], torch.float32, resident="stacked"))
    generator = torch.Generator().manual_seed(args.seed + 1000)
    shapes = {
        "x_hc": [args.tokens, C.HC_MULT, C.D],
        "incoming_pre_mix": [args.tokens, C.HC_MULT],
        "hc_attn_fn": [C.MIX_HC, C.HC_DIM],
        "hc_attn_scale": [3],
        "hc_attn_base": [C.MIX_HC],
        "attn_norm_weight": [C.D],
    }
    for name, shape in shapes.items():
        value = torch.randn(shape, generator=generator)
        if name == "hc_attn_fn":
            value /= C.HC_DIM**0.5
        elif name == "attn_norm_weight":
            value = torch.ones(shape, dtype=torch.bfloat16)
        elif name == "incoming_pre_mix":
            value = torch.softmax(value, dim=-1)
        specs.append(
            TensorSpec(
                name,
                [args.tp, *shape],
                value.dtype,
                init_value=value.unsqueeze(0).repeat(args.tp, *([1] * len(shape))),
                resident="stacked",
            )
        )
    for name, shape, dtype in (
        ("attention_input", [args.tokens, C.D], torch.bfloat16),
        ("normalized_attention", [args.tokens, C.D], torch.bfloat16),
        ("attention_output", [args.tokens, C.D], torch.bfloat16),
        ("attention_hidden", [args.tokens, C.HC_MULT, C.D], torch.float32),
        ("attention_pre_mix", [args.tokens, C.HC_MULT], torch.float32),
    ):
        sentinel = 13.0 if name in ("normalized_attention", "attention_output") else 0.0
        specs.append(TensorSpec(name, [args.tp, *shape], dtype, init_value=sentinel, resident="stacked"))
    by_name = {spec.name: spec for spec in specs}
    return [by_name[name] for name in ATTENTION_SPEC_NAMES]


def make_golden(kind, epochs):
    """Check mHC boundaries around the leaf file's TP-reduced reference."""

    def golden_half(tensors):
        world = tensors["x_hc"].shape[0]
        post = []
        residual = []
        active = int(tensors["num_tokens"])
        for rank in range(world):
            pre, post_mix, residual_mix = golden_mhc_mixes(
                tensors["x_hc"][rank],
                tensors["hc_attn_fn"][rank],
                tensors["hc_attn_scale"][rank],
                tensors["hc_attn_base"][rank],
            )
            tensors["attention_pre_mix"][rank].copy_(pre)
            collapsed = golden_mhc_pre(tensors["x_hc"][rank], tensors["incoming_pre_mix"][rank])
            tensors["attention_input"][rank].copy_(collapsed)
            tensors["normalized_attention"][rank, :active].copy_(
                rms_norm(collapsed[:active], tensors["attn_norm_weight"][rank])
            )
            post.append(post_mix)
            residual.append(residual_mix)
        leaf = dict(tensors, x=tensors["normalized_attention"], output=tensors["attention_output"])
        if kind == DecodeLayerKind.SWA:
            for _ in range(epochs):
                swa.golden_swa(leaf)
        elif kind == DecodeLayerKind.C2A_FULL:
            full.make_golden(epochs)(leaf)
        else:
            leaf["compressed_indices"] = tensors["topk_indices"]
            for _ in range(epochs):
                reuse.golden_c2a_reuse(leaf)
        for rank in range(world):
            tensors["attention_hidden"][rank].copy_(
                golden_mhc_post(
                    tensors["attention_output"][rank], tensors["x_hc"][rank], post[rank], residual[rank]
                )
            )

    return golden_half


def compare_unchanged(name):
    """Compare non-owner storage byte-for-byte, including CPU FP8 scales."""

    def compare(actual, expected, **kwargs):
        return torch.equal(
            actual.view(torch.uint8), expected.view(torch.uint8)
        ), f"{name}: non-owner state must stay exact"

    return compare


def compare_normalized(actual, expected, *, inputs, **kwargs):
    active = int(inputs["num_tokens"])
    passed, detail = ratio_allclose(atol=1e-4, rtol=1 / 128)(
        actual[:, :active], expected[:, :active], inputs=inputs, **kwargs
    )
    if not torch.equal(actual[:, active:].view(torch.uint8), expected[:, active:].view(torch.uint8)):
        return False, "inactive normalized suffix must stay byte-exact"
    return passed, detail


def compare_attention_hidden(actual, expected, *, inputs, **kwargs):
    """Keep inactive mHC results out of the active precision denominator."""
    active = int(inputs["num_tokens"])
    for rank in range(actual.shape[0]):
        passed, detail = full.compare_output(actual[rank, :active], expected[rank, :active], **kwargs)
        if not passed:
            return False, f"rank {rank}: {detail}"
        if active < actual.shape[1]:
            passed, detail = full.compare_output(actual[rank, active:], expected[rank, active:], **kwargs)
            if not passed:
                return False, f"rank {rank} inactive mHC suffix: {detail}"
    return True, "active mHC precision and independent inactive suffix checks passed"


def main():
    parser = argparse.ArgumentParser(description="DeepSeek V4.1 decode layer composition")
    parser.add_argument("--stage", choices=("attention", "block"), default="attention")
    parser.add_argument("--cpu-golden", action="store_true")
    parser.add_argument("-p", "--platform", default="a5")
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("--tp", type=int, choices=(1, 4), default=4)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--active-tokens", type=int)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--case", choices=("mixed", "shuffle", "prefix"), default="mixed")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    if args.cpu_golden:
        if args.stage != "block":
            parser.error("--cpu-golden requires --stage block")
        from models.deepseek_v4_1_flash._golden_smoke import run_decode_layer_goldens

        run_decode_layer_goldens(golden_decode_layer, REPRESENTATIVE_LAYER_IDS.values())
        return
    if args.stage == "block":
        reason = decode_layer_kernel_skip_reason(args.layer_id)
        parser.error(reason or "Block hardware fixture is pending integration")
    if not 1 <= args.tokens <= C.DECODE_MAX_TOKENS or not 1 <= args.epochs <= 1000:
        parser.error("tokens or epochs out of range")
    args.active_tokens = args.tokens if args.active_tokens is None else args.active_tokens
    if not 1 <= args.active_tokens <= args.tokens or not 1 <= args.requests <= args.active_tokens:
        parser.error("require 1 <= requests <= active tokens <= tokens")
    args.dp, args.bench = 1, False
    if args.tp != C.TP_SIZE:
        parser.error("--tp must match the import-time tensor parallel configuration")
    devices = (
        list(range(args.tp))
        if args.device is None or args.compile_only
        else [int(device) for device in args.device.split(",")]
    )
    if len(devices) != args.tp or len(set(devices)) != args.tp or min(devices) < 0:
        parser.error("--device must name exactly TP distinct nonnegative device IDs")
    kind = resolve_decode_layer_plan(args.layer_id).kind
    reason = attention_half_skip_reason(args.layer_id)
    if reason:
        parser.error(reason)
    if args.active_tokens != args.tokens and kind != DecodeLayerKind.C2A_REUSE:
        parser.error("inactive suffix validation currently requires C2A Reuse")
    initial_cache = {}
    comparisons = {
        "attention_input": ratio_allclose(atol=1e-4, rtol=1 / 128),
        "normalized_attention": compare_normalized,
        "attention_pre_mix": ratio_allclose(atol=1e-4, rtol=1e-4),
        "attention_hidden": compare_attention_hidden,
    }
    if kind == DecodeLayerKind.SWA:
        comparisons.update(
            attention_output=swa.compare_reduced,
            window_cache=swa.compare_distributed_cache,
            window_cache_scale=swa.compare_scales,
        )
    elif kind == DecodeLayerKind.C2A_FULL:
        comparisons.update(
            attention_output=full.compare_replicated(full.compare_output),
            compressor_state=full.compare_per_rank(full.compare_state),
            topk_indices=full.compare_per_rank(full.compare_topk),
        )
        for name in ("window_cache", "compressed_cache", "index_cache"):
            comparisons[name] = full.compare_per_rank(full.compare_cache(name), full.CACHE_SLOTS[name])
            comparisons[name + "_scale"] = swa.compare_scales
    else:
        comparisons["attention_output"] = full.compare_replicated(reuse.compare_active_output)
        for name in reuse.REUSE_MUTABLE_NAMES:
            comparisons[name] = reuse.compare_owned_cache(name, initial_cache)
        for name in ("compressed_cache", "compressed_cache_scale", "topk_indices"):
            comparisons[name] = compare_unchanged(name)
    specs = build_specs(args, kind, initial_cache)
    result = run(
        fn=make_decode_layer_program(args.layer_id, args.tp, args.epochs, stage=args.stage, specs=specs),
        specs=specs,
        golden_fn=make_golden(kind, args.epochs),
        compare_fn=comparisons,
        config={
            "platform": args.platform,
            "distributed_config": DistributedConfig(device_ids=devices, num_sub_workers=0),
        },
        compile_only=args.compile_only,
        save_data=args.save_data,
    )
    print(f"[HALF] layer={args.layer_id} kind={kind.name} work_dir={result.work_dir}")
    if not result.passed:
        raise SystemExit(result.error or 1)


__all__ = [
    "DecodeLayerGoldenResult",
    "DecodeLayerKind",
    "DecodeLayerPlan",
    "REPRESENTATIVE_LAYER_IDS",
    "attention_half_skip_reason",
    "decode_layer_attention_inputs",
    "decode_layer_kernel_skip_reason",
    "golden_decode_layer",
    "make_decode_layer_program",
    "resolve_decode_layer_plan",
]


if __name__ == "__main__":
    main()
