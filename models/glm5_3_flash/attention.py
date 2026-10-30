# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""GLM-5.3-Flash sparse MLA attention, projections and cache updates."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from dataclasses import dataclass

import pypto.language as pl
import torch

from models.glm5_3_flash.config import (
    BLOCK_SIZE, FLASH, KV_LORA, LOCAL_H, Q_LORA, QK_DIM, TOPK_INDEX_WIDTH, TP_SIZE, V_DIM, D,
    Glm53FlashConfig,
)
from models.glm5_3_flash.elementwise import accurate_divide, accurate_exp, accurate_row_sum_parts
from models.glm5_3_flash.linear import (
    bf16_linear, golden_bf16_linear, w8a8_projection_bf16, w8a8_projection_fp32,
)
from models.glm5_3_flash.quantization import quantize_per_token_int8, w8a8_dynamic_linear
from models.glm5_3_flash.rmsnorm import rms_norm, rmsnorm
from models.glm5_3_flash.utils import (
    _FLOAT32_MIN, Checkpoint, _check, _check_sparse_attention_impl, _randn, _report, _selected_rows,
    _sparse_attention_fixture, component_hidden, golden_absorb_output, golden_split_kv_b, paged_slots,
)

BLOCK_TABLE_DYN = pl.dynamic("GLM53_BLOCK_TABLE_DYN")
B_DYN = pl.dynamic("GLM53_B_DYN")
TABLE_DYN = pl.dynamic("GLM53_TABLE_DYN")
T_DYN = pl.dynamic("GLM53_T_DYN")

OUTPUT_MM_T_TILE = 16  # cube M tile; rows are a 16-row boxed tile
O_N_TILE = 256  # o_proj output-column tile over D

PROJECTION_MM_T_TILE = 16  # cube M tile; rows are a 16-row boxed tile
QA_N_TILE = 128  # q_a_proj output-column tile
KVA_N_TILE = 128  # kv_a_proj output-column tile
QB_N_TILE = 128  # q_b_proj output-column tile
ABSORB_N_TILE = 128  # absorb_query output-column tile

PREFILL_SOFTMAX_SCALE = QK_DIM**-0.5
PREFILL_NEG_INF = -1.0e20  # finite floor; a true -inf would make an all-masked row NaN
PREFILL_ATTN_K_TILE = 128  # selected rows per sparse block
PREFILL_H_TILE = min(LOCAL_H, 16)
PREFILL_H_PAD = ((PREFILL_H_TILE + 15) // 16) * 16
PREFILL_MM_T_TILE = 16  # cube M tile of the value expansion
EXPAND_N_TILE = 128  # value-expansion output-column tile over V_DIM
EXPAND_K_TILE = 256  # value-expansion KV_LORA reduction tile

DECODE_SOFTMAX_SCALE = QK_DIM**-0.5
DECODE_NEG_INF = -1.0e20  # finite floor; a true -inf would make an all-masked row NaN
DECODE_ATTN_K_TILE = 128  # selected rows per sparse block
NUM_QK_CORES = 24  # qk_pv dispatch lanes = a2a3 AIC count
DECODE_H_TILE = min(LOCAL_H, 16)
DECODE_H_PAD = ((DECODE_H_TILE + 15) // 16) * 16
DECODE_ROWS_PER_REQUEST = 4


@pl.jit.inline
def mla_epilog_prefill(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """Project head-space attention rows through the transposed output weight."""
    t_dim = pl.tensor.dim(attn_out, 0)
    t_mm = ((t_dim + OUTPUT_MM_T_TILE - 1) // OUTPUT_MM_T_TILE) * OUTPUT_MM_T_TILE
    attn_flat = pl.reshape(attn_out, [t_dim, LOCAL_H * V_DIM])
    for o_idx in pl.spmd(D // O_N_TILE, name_hint="mla_epilog_prefill"):
        n0 = o_idx * O_N_TILE
        for tc in pl.range(t_mm // OUTPUT_MM_T_TILE):
            t0 = tc * OUTPUT_MM_T_TILE
            valid_rows = pl.min(OUTPUT_MM_T_TILE, t_dim - t0)
            acc = bf16_linear(
                attn_flat, w_o, t0, n0, valid_rows, pl.cast(O_N_TILE, pl.INDEX), OUTPUT_MM_T_TILE, O_N_TILE,
                LOCAL_H * V_DIM, True,
            )
            output[t0:t0 + OUTPUT_MM_T_TILE, n0:n0 + O_N_TILE] = pl.set_validshape(acc, valid_rows, O_N_TILE)
    return output


@pl.jit.inline
def mla_epilog_w8a8(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.INT8],
    w_o_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """Publish the native W8A8 output projection's row-parallel FP32 partial."""
    rows = pl.tensor.dim(attn_out, 0)
    attn_flat = pl.reshape(attn_out, [rows, LOCAL_H * V_DIM])
    w8a8_projection_fp32(attn_flat, w_o, w_o_scale, output, LOCAL_H * V_DIM, D)
    return output


@pl.jit.inline
def mla_epilog_decode(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    w_o_absorbed: pl.Tensor[[D, LOCAL_H * KV_LORA], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """Project latent-space attention rows through the value-absorbed output weight."""
    t_dim = pl.tensor.dim(attn_out, 0)
    t_mm = ((t_dim + OUTPUT_MM_T_TILE - 1) // OUTPUT_MM_T_TILE) * OUTPUT_MM_T_TILE
    attn_flat = pl.reshape(attn_out, [t_dim, LOCAL_H * KV_LORA])
    for o_idx in pl.spmd(D // O_N_TILE, name_hint="mla_epilog_decode"):
        n0 = o_idx * O_N_TILE
        for tc in pl.range(t_mm // OUTPUT_MM_T_TILE):
            t0 = tc * OUTPUT_MM_T_TILE
            valid_rows = pl.min(OUTPUT_MM_T_TILE, t_dim - t0)
            acc = bf16_linear(
                attn_flat, w_o_absorbed, t0, n0, valid_rows, pl.cast(O_N_TILE, pl.INDEX), OUTPUT_MM_T_TILE, O_N_TILE,
                LOCAL_H * KV_LORA, True,
            )
            output[t0:t0 + OUTPUT_MM_T_TILE, n0:n0 + O_N_TILE] = pl.set_validshape(acc, valid_rows, O_N_TILE)
    return output


@pl.jit
def mla_epilog_w8a8_test(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.INT8],
    w_o_scale: pl.Tensor[[D], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mla_epilog_w8a8(attn_out, w_o, w_o_scale, output)
    return output


@pl.jit
def mla_epilog_prefill_test(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    """Run one head-space epilog for golden.run validation."""
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mla_epilog_prefill(attn_out, w_o, output)
    return output


@pl.jit
def mla_epilog_decode_test(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    w_o_absorbed: pl.Tensor[[D, LOCAL_H * KV_LORA], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    """Run one latent-space epilog for golden.run validation."""
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mla_epilog_decode(attn_out, w_o_absorbed, output)
    return output


@pl.jit.inline
def mla_prolog(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.BF16],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    kv_latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
):
    """Project packed tokens into BF16 normalized queries and latent KV rows."""
    t_dim = pl.tensor.dim(x, 0)
    t_mm = ((t_dim + PROJECTION_MM_T_TILE - 1) // PROJECTION_MM_T_TILE) * PROJECTION_MM_T_TILE
    q_a = pl.create_tensor([t_dim, Q_LORA], dtype=pl.BF16)
    for qa_idx in pl.spmd(Q_LORA // QA_N_TILE, name_hint="mla_q_a_proj"):
        n0 = qa_idx * QA_N_TILE
        for tc in pl.range(t_mm // PROJECTION_MM_T_TILE):
            t0 = tc * PROJECTION_MM_T_TILE
            rows = pl.min(PROJECTION_MM_T_TILE, t_dim - t0)
            acc = bf16_linear(x, w_q_a, t0, n0, rows, pl.cast(QA_N_TILE, pl.INDEX), PROJECTION_MM_T_TILE, QA_N_TILE, D, True)
            q_a_tile = pl.cast(acc, pl.BF16, mode="rint")
            q_a[t0:t0 + PROJECTION_MM_T_TILE, n0:n0 + QA_N_TILE] = pl.set_validshape(q_a_tile, rows, QA_N_TILE)
    rmsnorm(q_a, q_a_norm_weight, q_resid)

    query_flat = pl.reshape(query, [t_dim, LOCAL_H * QK_DIM])
    for qb_idx in pl.spmd((LOCAL_H * QK_DIM) // QB_N_TILE, name_hint="mla_q_b_proj"):
        n0 = qb_idx * QB_N_TILE
        for tc in pl.range(t_mm // PROJECTION_MM_T_TILE):
            t0 = tc * PROJECTION_MM_T_TILE
            rows = pl.min(PROJECTION_MM_T_TILE, t_dim - t0)
            acc = bf16_linear(
                q_resid, w_q_b, t0, n0, rows, pl.cast(QB_N_TILE, pl.INDEX), PROJECTION_MM_T_TILE, QB_N_TILE, Q_LORA,
                True,
            )
            value = pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, QB_N_TILE)
            query_flat[t0:t0 + PROJECTION_MM_T_TILE, n0:n0 + QB_N_TILE] = value

    kv_a = pl.create_tensor([t_dim, KV_LORA], dtype=pl.BF16)
    for kv_idx in pl.spmd(KV_LORA // KVA_N_TILE, name_hint="mla_kv_a_proj"):
        n0 = kv_idx * KVA_N_TILE
        for tc in pl.range(t_mm // PROJECTION_MM_T_TILE):
            t0 = tc * PROJECTION_MM_T_TILE
            rows = pl.min(PROJECTION_MM_T_TILE, t_dim - t0)
            acc = bf16_linear(x, w_kv_a, t0, n0, rows, pl.cast(KVA_N_TILE, pl.INDEX), PROJECTION_MM_T_TILE, KVA_N_TILE, D, True)
            kv_a_tile = pl.cast(acc, pl.BF16, mode="rint")
            kv_a[t0:t0 + PROJECTION_MM_T_TILE, n0:n0 + KVA_N_TILE] = pl.set_validshape(kv_a_tile, rows, KVA_N_TILE)
    rmsnorm(kv_a, kv_a_norm_weight, kv_latent)
    return q_resid, query, kv_latent


@pl.jit.inline
def mla_prolog_w8a8(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.INT8],
    w_q_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    w_q_b_scale: pl.Tensor[[LOCAL_H * QK_DIM], pl.FP32],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.INT8],
    w_kv_a_scale: pl.Tensor[[KV_LORA], pl.FP32],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    kv_latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
):
    """Run all three MLA projections with native INT8 weights and dynamic activations."""
    rows = pl.tensor.dim(x, 0)
    q_a = pl.create_tensor([rows, Q_LORA], dtype=pl.BF16)
    w8a8_projection_bf16(x, w_q_a, w_q_a_scale, q_a, D, Q_LORA)
    q_resid_rows = pl.reshape(q_resid, [rows, Q_LORA])
    rmsnorm(q_a, q_a_norm_weight, q_resid_rows)
    query_flat = pl.reshape(query, [rows, LOCAL_H * QK_DIM])
    w8a8_projection_bf16(q_resid, w_q_b, w_q_b_scale, query_flat, Q_LORA, LOCAL_H * QK_DIM)
    kv_a = pl.create_tensor([rows, KV_LORA], dtype=pl.BF16)
    w8a8_projection_bf16(x, w_kv_a, w_kv_a_scale, kv_a, D, KV_LORA)
    kv_latent_rows = pl.reshape(kv_latent, [rows, KV_LORA])
    rmsnorm(kv_a, kv_a_norm_weight, kv_latent_rows)
    return q_resid, query, kv_latent


@pl.jit
def mla_prolog_w8a8_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.INT8],
    w_q_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.INT8],
    w_q_b_scale: pl.Tensor[[LOCAL_H * QK_DIM], pl.FP32],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.INT8],
    w_kv_a_scale: pl.Tensor[[KV_LORA], pl.FP32],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.BF16]],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16]],
    kv_latent: pl.Out[pl.Tensor[[T_DYN, KV_LORA], pl.BF16]],
):
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    query.bind_dynamic(0, T_DYN)
    kv_latent.bind_dynamic(0, T_DYN)
    mla_prolog_w8a8(
        x, w_q_a, w_q_a_scale,
        q_a_norm_weight,
        w_q_b, w_q_b_scale,
        w_kv_a, w_kv_a_scale,
        kv_a_norm_weight, q_resid, query, kv_latent,
    )
    return q_resid, query, kv_latent


@pl.jit
def mla_prolog_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    w_q_a: pl.Tensor[[Q_LORA, D], pl.BF16],
    q_a_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    w_q_b: pl.Tensor[[LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    w_kv_a: pl.Tensor[[KV_LORA, D], pl.BF16],
    kv_a_norm_weight: pl.Tensor[[KV_LORA], pl.BF16],
    q_resid: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.BF16]],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16]],
    kv_latent: pl.Out[pl.Tensor[[T_DYN, KV_LORA], pl.BF16]],
):
    """Run one prolog for golden.run validation."""
    x.bind_dynamic(0, T_DYN)
    q_resid.bind_dynamic(0, T_DYN)
    query.bind_dynamic(0, T_DYN)
    kv_latent.bind_dynamic(0, T_DYN)
    mla_prolog(x, w_q_a, q_a_norm_weight, w_q_b, w_kv_a, kv_a_norm_weight, q_resid, query, kv_latent)
    return q_resid, query, kv_latent


@pl.jit.inline
def absorb_query(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    absorbed: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    query_extent: pl.constexpr = 0,
):
    """Multiply each query head by the BF16 key projection into latent space."""
    if query_extent > 0:
        t_dim = query_extent
    else:
        t_dim = pl.tensor.dim(query, 0)
    t_mm = ((t_dim + PROJECTION_MM_T_TILE - 1) // PROJECTION_MM_T_TILE) * PROJECTION_MM_T_TILE
    absorbed_flat = pl.reshape(absorbed, [t_dim, LOCAL_H * KV_LORA])
    for head in pl.range(LOCAL_H):
        query_head = pl.create_tensor([t_dim, QK_DIM], dtype=pl.BF16)
        weight_head = pl.create_tensor([QK_DIM, KV_LORA], dtype=pl.BF16)
        for block in pl.spmd(t_mm // PROJECTION_MM_T_TILE, name_hint="mla_query_head"):
            t0 = block * PROJECTION_MM_T_TILE
            rows = pl.min(PROJECTION_MM_T_TILE, t_dim - t0)
            part = pl.slice(query, [PROJECTION_MM_T_TILE, 1, QK_DIM], [t0, head, 0], valid_shape=[rows, 1, QK_DIM], drop_dims=[1])
            query_head[t0 : t0 + PROJECTION_MM_T_TILE, :] = part
        for block in pl.spmd((QK_DIM // 128) * (KV_LORA // 128), name_hint="mla_key_head"):
            k0 = (block // (KV_LORA // 128)) * 128
            n0 = (block % (KV_LORA // 128)) * 128
            weight_part = pl.slice(w_k, [1, 128, 128], [head, k0, n0], drop_dims=[0])
            weight_head[k0 : k0 + 128, n0 : n0 + 128] = weight_part
        for block in pl.spmd(KV_LORA // ABSORB_N_TILE, name_hint="mla_absorb_query"):
            n0 = block * ABSORB_N_TILE
            for tc in pl.range(t_mm // PROJECTION_MM_T_TILE):
                t0 = tc * PROJECTION_MM_T_TILE
                rows = pl.min(PROJECTION_MM_T_TILE, t_dim - t0)
                acc = bf16_linear(
                    query_head, weight_head, t0, n0, rows, pl.cast(ABSORB_N_TILE, pl.INDEX), PROJECTION_MM_T_TILE,
                    ABSORB_N_TILE, QK_DIM, False,
                )
                value = pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, ABSORB_N_TILE)
                absorbed_col = head * KV_LORA + n0
                absorbed_flat[t0:t0 + PROJECTION_MM_T_TILE, absorbed_col:absorbed_col + ABSORB_N_TILE] = value
    return absorbed


@pl.jit
def mla_absorb_query_test(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    absorbed: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16]],
):
    """Run one query absorption for golden.run validation."""
    query.bind_dynamic(0, T_DYN)
    absorbed.bind_dynamic(0, T_DYN)
    absorb_query(query, w_k, absorbed, 0)
    return absorbed


@pl.jit.inline
def prefill_sparse_attn(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    w_v: pl.Tensor[[LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    query_extent: pl.constexpr = 0,
):
    """Sparse attention over the selected latent rows, prefill path."""
    if query_extent > 0:
        t_dim = query_extent
    else:
        t_dim = pl.tensor.dim(query, 0)
    absorbed = pl.create_tensor([t_dim, LOCAL_H, KV_LORA], dtype=pl.BF16)
    absorb_query(query, w_k, absorbed, query_extent)

    # Head-major: the value expansion needs its M rows contiguous per head.
    context = pl.create_tensor([LOCAL_H * t_dim, KV_LORA], dtype=pl.BF16)
    # A zero-token call would leave both launches with zero blocks, which the
    # compiler must lower as a guarded empty launch (LegalizeSpmdLaunches);
    # proving the positive count at the source keeps the launch unguarded and
    # skips the whole block when there is nothing to attend.
    absorbed_rows = pl.reshape(absorbed, [t_dim * LOCAL_H, KV_LORA])
    if t_dim > 0:
        with pl.spmd(t_dim, name_hint="prefill_sparse_flash") as flash_tid:
            token = pl.tile.get_block_idx()
            request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
            for _ in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                running_m = pl.reshape(pl.full([1, PREFILL_H_PAD // 2], dtype=pl.FP32, value=PREFILL_NEG_INF), [PREFILL_H_PAD // 2, 1])
                running_l = pl.reshape(pl.full([1, PREFILL_H_PAD // 2], dtype=pl.FP32, value=0.0), [PREFILL_H_PAD // 2, 1])
                running_o = pl.full([PREFILL_H_PAD // 2, KV_LORA], dtype=pl.FP32, value=0.0)
            for sb, (m_iter, l_iter, o_iter) in pl.range(
                ((TOPK_INDEX_WIDTH + PREFILL_ATTN_K_TILE - 1) // PREFILL_ATTN_K_TILE),
                init_values=(running_m, running_l, running_o),
            ):
                w0 = sb * PREFILL_ATTN_K_TILE
                kv_tile = pl.create_l1([PREFILL_ATTN_K_TILE, KV_LORA], pl.BF16)
                # Lane 0 is -1 only when the whole block is padding.
                if pl.read(topk_indices, [token, w0]) >= 0:
                    for r in pl.range(PREFILL_ATTN_K_TILE):
                        if w0 + r < TOPK_INDEX_WIDTH:
                            position = pl.read(topk_indices, [token, w0 + r])
                            if position >= 0:
                                logical = pl.cast(position, pl.INDEX)
                                page = pl.cast(pl.read(block_table, [request, logical // BLOCK_SIZE]), pl.INDEX)
                                source = page * BLOCK_SIZE + logical % BLOCK_SIZE
                                kv_tile = pl.gather_row(kv_tile, latent_cache, [r, 0], [source, 0], [1, KV_LORA])
                            else:
                                kv_tile = pl.gather_row(kv_tile, latent_cache, [r, 0], [0, 0], [1, KV_LORA])
                        else:
                            kv_tile = pl.gather_row(kv_tile, latent_cache, [r, 0], [0, 0], [1, KV_LORA])
                else:
                    kv_tile = pl.gather_row(kv_tile, latent_cache, [0, 0], [0, 0], [PREFILL_ATTN_K_TILE, KV_LORA])

                # Release the query pipe slot before the PV weights use the same FIFO.
                for q_aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    q0 = q_aiv_id * (PREFILL_H_PAD // 2)
                    q_valid = pl.max(0, pl.min(PREFILL_H_PAD // 2, PREFILL_H_TILE - q0))
                    q_start = token * LOCAL_H + q0
                    q_part = pl.slice(absorbed_rows, [PREFILL_H_PAD // 2, KV_LORA], [q_start, 0], valid_shape=[q_valid, KV_LORA])
                    q_padded = pl.fillpad(q_part, pad_value=pl.PadValue.zero)
                    q_tile = pl.aic_gather(q_padded)
                raw = pl.matmul(q_tile, kv_tile, b_trans=True, out_dtype=pl.FP32)
                for _ in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    raw_part = pl.aiv_shard(raw)
                    bias_row = pl.full([1, PREFILL_ATTN_K_TILE], dtype=pl.FP32, value=PREFILL_NEG_INF)
                    for r in pl.range(PREFILL_ATTN_K_TILE):
                        if w0 + r < TOPK_INDEX_WIDTH:
                            if pl.read(topk_indices, [token, w0 + r]) >= 0:
                                pl.write(bias_row, [0, r], 0.0)
                    scores = pl.col_expand_add(pl.mul(raw_part, PREFILL_SOFTMAX_SCALE), bias_row)
                    block_m = pl.row_max(scores)
                    weights = pl.exp(pl.row_expand_sub(scores, block_m))
                    block_l = pl.row_sum(weights)
                    weights_bf16 = pl.cast(weights, target_type=pl.BF16, mode="rint")
                    weights_mat = pl.aic_gather(weights_bf16)
                block_o = pl.matmul(weights_mat, kv_tile, out_dtype=pl.FP32)
                for _ in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    block_o_part = pl.aiv_shard(block_o)
                    next_m = pl.maximum(m_iter, block_m)
                    alpha = pl.exp(pl.sub(m_iter, next_m))
                    beta = pl.exp(pl.sub(block_m, next_m))
                    next_l = pl.add(pl.mul(alpha, l_iter), pl.mul(beta, block_l))
                    next_o = pl.add(pl.row_expand_mul(o_iter, alpha), pl.row_expand_mul(block_o_part, beta))
                running_m, running_l, running_o = pl.yield_(next_m, next_l, next_o)

            for output_aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                # Each lane writes only its real head rows.
                if pl.read(topk_indices, [token, 0]) >= 0:
                    normalized = pl.cast(pl.row_expand_div(running_o, running_l), pl.BF16, mode="rint")
                    for local_head in pl.unroll(PREFILL_H_PAD // 2):
                        head = output_aiv_id * (PREFILL_H_PAD // 2) + local_head
                        if head < PREFILL_H_TILE:
                            ctx_row = head * t_dim + token
                            context[ctx_row : ctx_row + 1, :] = normalized[local_head : local_head + 1, :]
                else:
                    empty_row = pl.full([1, KV_LORA], dtype=pl.BF16, value=0.0)
                    for local_head in pl.unroll(PREFILL_H_PAD // 2):
                        head = output_aiv_id * (PREFILL_H_PAD // 2) + local_head
                        if head < PREFILL_H_TILE:
                            ctx_row = head * t_dim + token
                            context[ctx_row : ctx_row + 1, :] = empty_row

        output_flat = pl.reshape(output, [t_dim, LOCAL_H * V_DIM])
        w_v_flat = pl.reshape(w_v, [LOCAL_H * V_DIM, KV_LORA])
        t_mm = ((t_dim + PREFILL_MM_T_TILE - 1) // PREFILL_MM_T_TILE) * PREFILL_MM_T_TILE
        with pl.spmd(LOCAL_H * (V_DIM // EXPAND_N_TILE), name_hint="prefill_value_expand", deps=[flash_tid]):
            expand_idx = pl.tile.get_block_idx()
            head = expand_idx // (V_DIM // EXPAND_N_TILE)
            n0 = (expand_idx % (V_DIM // EXPAND_N_TILE)) * EXPAND_N_TILE
            for tc in pl.range(t_mm // PREFILL_MM_T_TILE):
                t0 = tc * PREFILL_MM_T_TILE
                valid_rows = pl.min(PREFILL_MM_T_TILE, t_dim - t0)
                acc = pl.create_tensor([PREFILL_MM_T_TILE, EXPAND_N_TILE], dtype=pl.FP32)
                for kb in pl.pipeline(KV_LORA // EXPAND_K_TILE, stage=2):
                    k0 = kb * EXPAND_K_TILE
                    ctx_tile = pl.slice(context, [PREFILL_MM_T_TILE, EXPAND_K_TILE], [head * t_dim + t0, k0], valid_shape=[valid_rows, EXPAND_K_TILE])
                    w_tile = w_v_flat[head * V_DIM + n0:head * V_DIM + n0 + EXPAND_N_TILE, k0:k0 + EXPAND_K_TILE]
                    acc = pl.matmul_acc(acc, ctx_tile, w_tile, b_trans=True, init_cond=(kb == 0))
                expanded = pl.cast(acc, target_type=pl.BF16, mode="rint")
                output_flat_set_validshape = pl.set_validshape(expanded, valid_rows, EXPAND_N_TILE)
                output_col = head * V_DIM + n0
                output_flat[t0:t0 + PREFILL_MM_T_TILE, output_col:output_col + EXPAND_N_TILE] = output_flat_set_validshape
    return output


if PREFILL_H_TILE != LOCAL_H:
    prefill_sparse_attn = None


@pl.jit
def prefill_sparse_attn_test(
    query: pl.Tensor[[T_DYN, LOCAL_H, QK_DIM], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    w_k: pl.Tensor[[LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    w_v: pl.Tensor[[LOCAL_H, V_DIM, KV_LORA], pl.BF16],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16]],
):
    """Run one sparse prefill step for golden.run validation."""
    query.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    block_table.bind_dynamic(1, BLOCK_TABLE_DYN)
    prefill_sparse_attn(query, latent_cache, block_table, request_ids, w_k, w_v, topk_indices, output, 0)
    return output


@pl.jit.inline
def mla_cache_write(
    latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
):
    """Scatter one step's latent rows into the paged pool, one block per row.

    Returns the region's TaskId so a later gather can order against the write.
    """
    tokens = pl.tensor.dim(latent, 0)
    with pl.spmd(tokens, name_hint="mla_cache_write") as write_tid:
        token = pl.tile.get_block_idx()
        slot_i32 = pl.read(slots, [token])
        if slot_i32 >= 0:
            slot = pl.cast(slot_i32, pl.INDEX)
            pl.store(pl.load(latent, [token, 0], [1, KV_LORA]), [slot, 0], cache)
    return write_tid


@pl.jit
def mla_cache_write_test(
    latent: pl.Tensor[[T_DYN, KV_LORA], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT32],
    cache: pl.InOut[pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16]],
):
    """Run one scatter for golden.run validation."""
    latent.bind_dynamic(0, T_DYN)
    slots.bind_dynamic(0, T_DYN)
    mla_cache_write(latent, slots, cache)
    return cache


@pl.jit.inline
def decode_sparse_attn(
    absorbed_query: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
):
    """Attend over front-packed logical positions in the latent cache.

    Compensated FP32 QK scores feed BF16-probability PV products. Block
    statistics merge in FP32 before corrected division. Padding is a suffix
    of -1 indices; an entirely padded row produces zero output.

    H_PAD preserves the 16-row Cube minimum and 32-byte reduction alignment.
    NEG_INF must remain finite for rows whose every block is skipped."""
    t_dim = pl.tensor.dim(absorbed_query, 0)

    # Allocate the full final tile, including the padding after TOPK_INDEX_WIDTH.
    raw_scores = pl.create_tensor(
        [
            t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * DECODE_H_PAD,
            ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE) * DECODE_ATTN_K_TILE,
        ],
        dtype=pl.FP32,
    )
    q_flat = pl.reshape(absorbed_query, [t_dim * LOCAL_H, KV_LORA])
    with pl.spmd(t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE), name_hint="mla_decode_scores") as scores_tid:
        item = pl.tile.get_block_idx()
        token = item // ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE))
        head = item // ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE) % ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE)
        sb = item % ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE)
        w0 = sb * DECODE_ATTN_K_TILE
        if pl.read(topk_indices, [token, w0]) >= 0:
            request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
            score_block = pl.full([DECODE_H_PAD, DECODE_ATTN_K_TILE], dtype=pl.FP32, value=0.0)
            for selected in pl.range(DECODE_ATTN_K_TILE):
                if w0 + selected < TOPK_INDEX_WIDTH:
                    position = pl.read(topk_indices, [token, w0 + selected])
                    if position >= 0:
                        logical = pl.cast(position, pl.INDEX)
                        page = pl.cast(pl.read(block_table, [request, logical // BLOCK_SIZE]), pl.INDEX)
                        source = page * BLOCK_SIZE + logical % BLOCK_SIZE
                        total = pl.full([1, DECODE_H_PAD], dtype=pl.FP32, value=0.0)
                        correction = pl.full([1, DECODE_H_PAD], dtype=pl.FP32, value=0.0)
                        for k0 in pl.pipeline(0, KV_LORA, 128, stage=1):
                            query_part = q_flat[token * LOCAL_H + head * DECODE_H_TILE:token * LOCAL_H + head * DECODE_H_TILE + DECODE_H_TILE, k0:k0 + 128]
                            query_value = pl.create_tensor([DECODE_H_PAD, 128], dtype=pl.FP32)
                            query_value[:, :] = pl.full([DECODE_H_PAD, 128], dtype=pl.FP32, value=0.0)
                            query_value[0:DECODE_H_TILE, :] = pl.cast(query_part, pl.FP32)
                            key_value = pl.cast(latent_cache[source:source + 1, k0:k0 + 128], pl.FP32)
                            products = pl.create_tensor([DECODE_H_PAD, 128], dtype=pl.FP32)
                            products[:, :] = pl.col_expand_mul(query_value, key_value)
                            high, low = accurate_row_sum_parts(products)
                            part_sum = pl.reshape(high, [1, DECODE_H_PAD])
                            tail = pl.reshape(low, [1, DECODE_H_PAD])
                            updated = pl.add(total, part_sum)
                            recovered = pl.sub(updated, total)
                            error = pl.add(pl.sub(total, pl.sub(updated, recovered)), pl.sub(part_sum, recovered))
                            correction = pl.add(correction, pl.add(error, tail))
                            total = updated
                        dot = pl.add(total, correction)
                        for h in pl.range(DECODE_H_PAD):
                            pl.write(score_block, [h, selected], pl.read(dot, [0, h]))
            score_row = (token * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) + head) * DECODE_H_PAD
            raw_scores[score_row:score_row + DECODE_H_PAD, w0:w0 + DECODE_ATTN_K_TILE] = score_block

    blk_oi = pl.create_tensor([t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE) * DECODE_H_PAD, KV_LORA], dtype=pl.FP32)
    blk_li = pl.create_tensor([t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE) * DECODE_H_PAD, 1], dtype=pl.FP32)
    blk_mi = pl.create_tensor([t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE) * DECODE_H_PAD, 1], dtype=pl.FP32)
    with pl.spmd(NUM_QK_CORES, name_hint="mla_decode_qk_pv", deps=[scores_tid]) as qk_tid:
        lane = pl.tile.get_block_idx()
        items = t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE)
        for it in pl.range((items - lane + NUM_QK_CORES - 1) // NUM_QK_CORES):
            item = lane + it * NUM_QK_CORES
            token = item // ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE))
            head = item // ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE) % ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE)
            sb = item % ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE)
            w0 = sb * DECODE_ATTN_K_TILE
            request = pl.cast(pl.read(request_ids, [token]), pl.INDEX)
            blk_row = (
                (token * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) + head)
                * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE)
                + sb
            ) * DECODE_H_PAD
            # The first selection is -1 only when the whole block is padding.
            if pl.read(topk_indices, [token, w0]) < 0:
                for empty_aiv_id in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                    if empty_aiv_id == 0:
                        empty_m = pl.reshape(pl.full([1, DECODE_H_PAD], dtype=pl.FP32, value=DECODE_NEG_INF), [DECODE_H_PAD, 1])
                        empty_l = pl.reshape(pl.full([1, DECODE_H_PAD], dtype=pl.FP32, value=0.0), [DECODE_H_PAD, 1])
                        empty_o = pl.full([DECODE_H_PAD, KV_LORA], dtype=pl.FP32, value=0.0)
                        blk_mi[blk_row:blk_row + DECODE_H_PAD, :] = empty_m
                        blk_li[blk_row:blk_row + DECODE_H_PAD, :] = empty_l
                        blk_oi[blk_row:blk_row + DECODE_H_PAD, :] = empty_o
            else:
                kv_tile = pl.create_l1([DECODE_ATTN_K_TILE, KV_LORA], pl.BF16)
                for r in pl.range(DECODE_ATTN_K_TILE):
                    if w0 + r < TOPK_INDEX_WIDTH:
                        position = pl.read(topk_indices, [token, w0 + r])
                        if position >= 0:
                            logical = pl.cast(position, pl.INDEX)
                            page = pl.cast(pl.read(block_table, [request, logical // BLOCK_SIZE]), pl.INDEX)
                            source = page * BLOCK_SIZE + logical % BLOCK_SIZE
                            kv_tile = pl.gather_row(kv_tile, latent_cache, [r, 0], [source, 0], [1, KV_LORA])
                        else:
                            kv_tile = pl.gather_row(kv_tile, latent_cache, [r, 0], [0, 0], [1, KV_LORA])
                    else:
                        kv_tile = pl.gather_row(kv_tile, latent_cache, [r, 0], [0, 0], [1, KV_LORA])

                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    raw_part = raw_scores[(token * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) + head) * DECODE_H_PAD + aiv_id * (DECODE_H_PAD // 2):(token * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) + head) * DECODE_H_PAD + aiv_id * (DECODE_H_PAD // 2) + DECODE_H_PAD // 2, w0:w0 + DECODE_ATTN_K_TILE]
                    bias_row = pl.full([1, DECODE_ATTN_K_TILE], dtype=pl.FP32, value=DECODE_NEG_INF)
                    for r in pl.range(DECODE_ATTN_K_TILE):
                        if w0 + r < TOPK_INDEX_WIDTH:
                            if pl.read(topk_indices, [token, w0 + r]) >= 0:
                                pl.write(bias_row, [0, r], 0.0)
                    scores = pl.col_expand_add(pl.mul(raw_part, DECODE_SOFTMAX_SCALE), bias_row)
                    mi = pl.row_max(scores)
                    shifted = pl.create_tensor([DECODE_H_PAD // 2, DECODE_ATTN_K_TILE], dtype=pl.FP32)
                    shifted[:, :] = pl.row_expand_sub(scores, mi)
                    weights = accurate_exp(shifted)
                    weights = pl.col_expand_mul(weights, pl.cmp(bias_row, 0.0, cmp_type=0))
                    li = pl.row_sum(weights)
                    weights_bf16 = pl.cast(weights, target_type=pl.BF16, mode="rint")
                    weights_mat = pl.aic_gather(weights_bf16)
                    row0 = blk_row + aiv_id * (DECODE_H_PAD // 2)
                    blk_mi[row0:row0 + DECODE_H_PAD // 2, :] = mi
                    blk_li[row0:row0 + DECODE_H_PAD // 2, :] = li
                oi = pl.matmul(weights_mat, kv_tile, out_dtype=pl.FP32)
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    oi_part = pl.aiv_shard(oi)
                    row0 = blk_row + aiv_id * (DECODE_H_PAD // 2)
                    blk_oi[row0:row0 + DECODE_H_PAD // 2, :] = oi_part

    merged_denominators = pl.create_tensor([t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * DECODE_H_PAD, 1], dtype=pl.FP32)
    merged_numerators = pl.create_tensor([t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * DECODE_H_PAD, KV_LORA], dtype=pl.FP32)
    with pl.spmd(t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE), name_hint="mla_decode_merge", deps=[qk_tid]) as merge_tid:
        task = pl.tile.get_block_idx()
        token = task // ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE)
        head = task % ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE)
        base = (
            (token * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) + head)
            * ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE)
        ) * DECODE_H_PAD
        merged_mi = blk_mi[base : base + DECODE_H_PAD, 0:1]
        merged_li = blk_li[base : base + DECODE_H_PAD, 0:1]
        merged_oi = blk_oi[base : base + DECODE_H_PAD, 0:KV_LORA]
        for sb in pl.pipeline(1, ((TOPK_INDEX_WIDTH + DECODE_ATTN_K_TILE - 1) // DECODE_ATTN_K_TILE), stage=2):
            row = base + sb * DECODE_H_PAD
            cur_mi = blk_mi[row : row + DECODE_H_PAD, 0:1]
            cur_li = blk_li[row : row + DECODE_H_PAD, 0:1]
            cur_oi = blk_oi[row : row + DECODE_H_PAD, 0:KV_LORA]
            new_mi = pl.maximum(merged_mi, cur_mi)
            alpha = pl.exp(pl.sub(merged_mi, new_mi))
            beta = pl.exp(pl.sub(cur_mi, new_mi))
            merged_li = pl.add(pl.mul(alpha, merged_li), pl.mul(beta, cur_li))
            merged_oi = pl.add(pl.row_expand_mul(merged_oi, alpha), pl.row_expand_mul(cur_oi, beta))
            merged_mi = new_mi
        merged_numerators[task * DECODE_H_PAD:(task + 1) * DECODE_H_PAD, :] = merged_oi
        merged_denominators[task * DECODE_H_PAD:(task + 1) * DECODE_H_PAD, :] = merged_li

    # A separate column-tiled normalization bounds the compensated division's
    # vector working set while retaining the FP32 merged numerator.
    out_flat = pl.reshape(output, [t_dim * LOCAL_H, KV_LORA])
    with pl.spmd(t_dim * ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE) * (KV_LORA // 128), name_hint="mla_decode_normalize", deps=[merge_tid]):
        part = pl.tile.get_block_idx()
        task = part // (KV_LORA // 128)
        col = part % (KV_LORA // 128) * 128
        token = task // ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE)
        head = task % ((LOCAL_H + DECODE_H_TILE - 1) // DECODE_H_TILE)
        # Lane 0 of the row is -1 only when the row selects nothing at all; every block
        # is then neutral and ``merged_li`` is zero, which a division would turn to NaN.
        if pl.read(topk_indices, [token, 0]) >= 0:
            numerator = pl.create_tensor([DECODE_H_PAD, 128], dtype=pl.FP32)
            numerator[:, :] = merged_numerators[task * DECODE_H_PAD:task * DECODE_H_PAD + DECODE_H_PAD, col:col + 128]
            denominator = pl.create_tensor([DECODE_H_PAD, 128], dtype=pl.FP32)
            denominator_value_full = pl.full([DECODE_H_PAD, 128], dtype=pl.FP32, value=1.0)
            denominator_value_slice = merged_denominators[task * DECODE_H_PAD:task * DECODE_H_PAD + DECODE_H_PAD, 0:1]
            denominator[:, :] = pl.row_expand_mul(denominator_value_full, denominator_value_slice)
            normalized = accurate_divide(numerator, denominator)
            out_bf16 = pl.cast(normalized, target_type=pl.BF16, mode="rint")
            output_row = token * LOCAL_H + head * DECODE_H_TILE
            out_flat[output_row:output_row + DECODE_H_TILE, col:col + 128] = out_bf16[0:DECODE_H_TILE, :]
        else:
            value_full = pl.full([DECODE_H_TILE, 128], dtype=pl.BF16, value=0.0)
            output_row = token * LOCAL_H + head * DECODE_H_TILE
            out_flat[output_row:output_row + DECODE_H_TILE, col:col + 128] = value_full
    return output


@pl.jit
def decode_sparse_attn_test(
    absorbed_query: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    latent_cache: pl.Tensor[[TABLE_DYN * BLOCK_SIZE, KV_LORA], pl.BF16],
    block_table: pl.Tensor[[B_DYN, BLOCK_TABLE_DYN], pl.INT32],
    request_ids: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, TOPK_INDEX_WIDTH], pl.INT32],
    output: pl.Out[pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16]],
):
    """Run one sparse decode step for golden.run validation."""
    absorbed_query.bind_dynamic(0, T_DYN)
    request_ids.bind_dynamic(0, T_DYN)
    topk_indices.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    block_table.bind_dynamic(1, BLOCK_TABLE_DYN)
    decode_sparse_attn(absorbed_query, latent_cache, block_table, request_ids, topk_indices, output)
    return output


def golden_mla_epilog(
    attn_out: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """Project flattened attention rows with a prepared BF16 output weight."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return golden_bf16_linear(flattened, w_o)


def output_run_native_case(options):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5358)
        hidden = component_hidden(options, LOCAL_H * V_DIM, generator)
        values = {"attn_out": hidden.reshape(options.tokens, LOCAL_H, V_DIM)}
        if options.checkpoint is not None:
            attention = Checkpoint(options.checkpoint).attention(options.layer, tp=TP_SIZE, mla_mode="native")
            values.update(w_o=attention["w_o"], w_o_scale=attention["w_o_scale"].reshape(-1))
        else:
            values["w_o"] = torch.randint(-128, 128, (D, LOCAL_H * V_DIM), generator=generator, dtype=torch.int8)
            values["w_o_scale"] = torch.rand(D, generator=generator) * 0.001 + 0.0001
        return values

    specs = [
        TensorSpec("attn_out", (options.tokens, LOCAL_H, V_DIM), torch.bfloat16, init_value=lambda : inputs()["attn_out"]),
        TensorSpec("w_o", (D, LOCAL_H * V_DIM), torch.int8, init_value=lambda: inputs()["w_o"]),
        TensorSpec("w_o_scale", (D,), torch.float32, init_value=lambda: inputs()["w_o_scale"]),
        TensorSpec("output", (options.tokens, D), torch.float32),
    ]

    def golden(values):
        quantized, scales = quantize_per_token_int8(values["attn_out"].flatten(1))
        products = quantized.int() @ values["w_o"].int().t()
        values["output"].copy_(products.float() * scales * values["w_o_scale"])

    result = run(
        fn=mla_epilog_w8a8_test,
        specs=specs,
        golden_fn=golden,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={"output": ratio_allclose(atol=1e-05, rtol=1e-05, max_error_ratio=0)},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def _epilog_specs(tokens: int, width: int, weight_name: str):
    """Build one deterministic epilog case at a given head-space width."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(73)

    def init_attn():
        return torch.randn(tokens, LOCAL_H, width, generator=generator).bfloat16()

    def init_weight():
        return (torch.randn(D, LOCAL_H * width, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("attn_out", [tokens, LOCAL_H, width], torch.bfloat16, init_value=init_attn),
        TensorSpec(weight_name, [D, LOCAL_H * width], torch.bfloat16, init_value=init_weight),
        TensorSpec("output", [tokens, D], torch.float32),
    ]


def build_mla_epilog_prefill_specs(tokens: int = 20):
    """Head-space epilog: the reduction is ``LOCAL_H * V_DIM``."""
    return _epilog_specs(tokens, V_DIM, "w_o")


def build_mla_epilog_decode_specs(tokens: int = 20):
    """Latent-space epilog: the reduction is the wider ``LOCAL_H * KV_LORA``."""
    return _epilog_specs(tokens, KV_LORA, "w_o_absorbed")


def golden_mla_epilog_prefill_case(tensors):
    """Fill the expected partial sum for :func:`build_mla_epilog_prefill_specs`."""
    tensors["output"][:] = golden_mla_epilog(tensors["attn_out"], tensors["w_o"])


def golden_mla_epilog_decode_case(tensors):
    """Fill the expected partial sum for :func:`build_mla_epilog_decode_specs`."""
    tensors["output"][:] = golden_mla_epilog(tensors["attn_out"], tensors["w_o_absorbed"])


def run_mla_epilog_goldens() -> None:
    """Check both output projections keep their row-parallel FP32 partial-sum contract."""
    torch.manual_seed(41)
    tokens, heads, model_dim = 4, 2, 32
    v_dim, kv_lora = 16, 24

    attn_out = torch.randn(tokens, heads, v_dim, dtype=torch.bfloat16)
    w_o = torch.randn(model_dim, heads * v_dim, dtype=torch.bfloat16) * 0.05
    prefill = golden_mla_epilog(attn_out, w_o)
    _check(prefill.shape == (tokens, model_dim), f"prefill epilog shape {tuple(prefill.shape)}")
    _check(prefill.dtype is torch.float32, f"prefill epilog dtype {prefill.dtype}")
    expected = torch.nn.functional.linear(attn_out.float().flatten(-2), w_o.float())
    _check(torch.allclose(prefill, expected, atol=1e-5), "prefill epilog is not a plain projection")

    latent_out = torch.randn(tokens, heads, kv_lora, dtype=torch.bfloat16)
    w_o_absorbed = torch.randn(model_dim, heads * kv_lora, dtype=torch.bfloat16) * 0.05
    decode = golden_mla_epilog(latent_out, w_o_absorbed)
    _check(decode.shape == (tokens, model_dim), f"decode epilog shape {tuple(decode.shape)}")
    _check(decode.dtype is torch.float32, f"decode epilog dtype {decode.dtype}")

    _report("mla_epilog")


def golden_mla_prolog(
    x: torch.Tensor,
    w_q_a: torch.Tensor,
    q_a_norm_weight: torch.Tensor,
    w_q_b: torch.Tensor,
    w_kv_a: torch.Tensor,
    kv_a_norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one packed token batch into ``q_resid``, ``query`` and ``kv_latent``.

    Every projection on this path has been prepared as BF16 by the loader, so nothing
    is quantized here. ``qk_rope_head_dim`` is 0, so there is no rope half to split.

    ``query`` is projected from the **rounded** ``q_resid`` rather than from an FP32
    intermediate: ``q_resid`` is also published to the indexer, and both consumers must
    see the same rows. The down-projections round to BF16 before RMSNorm,
    and normalization preserves the BF16 boundary before the gamma product.

    Args:
        x: ``[T, D]`` packed hidden states.
        w_q_a: ``[Q_LORA, D]`` query down-projection.
        q_a_norm_weight: ``[Q_LORA]`` gamma of ``q_a_layernorm``.
        w_q_b: ``[H * QK_DIM, Q_LORA]`` query up-projection, rank-local in ``H``.
        w_kv_a: ``[KV_LORA, D]`` latent down-projection (``num_kv_heads`` is 1, so this
            is not sharded).
        kv_a_norm_weight: ``[KV_LORA]`` gamma of ``kv_a_layernorm``.

    Returns:
        ``q_resid`` ``[T, Q_LORA]``, ``query`` ``[T, H, QK_DIM]`` and ``kv_latent``
        ``[T, KV_LORA]``, all in ``x``'s dtype.
    """
    dtype = x.dtype
    q_a = golden_bf16_linear(x, w_q_a).to(dtype)
    kv_a = golden_bf16_linear(x, w_kv_a).to(dtype)
    q_resid = rms_norm(q_a, q_a_norm_weight)
    kv_latent = rms_norm(kv_a, kv_a_norm_weight)
    heads = w_q_b.shape[0] // QK_DIM
    query = golden_bf16_linear(q_resid, w_q_b).to(dtype).unflatten(-1, (heads, QK_DIM))
    return q_resid, query, kv_latent


def golden_mla_prolog_w8a8(values):
    q_a = w8a8_dynamic_linear(values["x"], values["w_q_a"], values["w_q_a_scale"])
    q_resid = rms_norm(q_a, values["q_a_norm_weight"])
    query = w8a8_dynamic_linear(q_resid, values["w_q_b"], values["w_q_b_scale"])
    kv_a = w8a8_dynamic_linear(values["x"], values["w_kv_a"], values["w_kv_a_scale"])
    values["q_resid"].copy_(q_resid)
    values["query"].copy_(query.reshape(-1, LOCAL_H, QK_DIM))
    values["kv_latent"].copy_(rms_norm(kv_a, values["kv_a_norm_weight"]))


def projection_run_native_case(options):
    from functools import lru_cache

    from golden import TensorSpec, ratio_allclose, run

    layouts = {
        "x": ((options.tokens, D), torch.bfloat16),
        "w_q_a": ((Q_LORA, D), torch.int8),
        "w_q_a_scale": ((Q_LORA,), torch.float32),
        "q_a_norm_weight": ((Q_LORA,), torch.bfloat16),
        "w_q_b": ((LOCAL_H * QK_DIM, Q_LORA), torch.int8),
        "w_q_b_scale": ((LOCAL_H * QK_DIM,), torch.float32),
        "w_kv_a": ((KV_LORA, D), torch.int8),
        "w_kv_a_scale": ((KV_LORA,), torch.float32),
        "kv_a_norm_weight": ((KV_LORA,), torch.bfloat16),
    }

    @lru_cache(maxsize=1)
    def inputs():
        generator = torch.Generator().manual_seed(5356)
        values = {"x": component_hidden(options, D, generator)}
        if options.checkpoint is not None:
            values.update(Checkpoint(options.checkpoint).attention(options.layer, tp=TP_SIZE, mla_mode="native"))
        else:
            for name, (shape, dtype) in layouts.items():
                if name == "x":
                    continue
                if dtype == torch.int8:
                    values[name] = torch.randint(-128, 128, shape, generator=generator, dtype=dtype)
                elif dtype == torch.float32:
                    values[name] = torch.rand(shape, generator=generator) * 0.001 + 0.0001
                else:
                    values[name] = torch.ones(shape, dtype=dtype)
        return values

    specs = [
        TensorSpec(name, shape, dtype, init_value=lambda key=name: inputs()[key])
        for name, (shape, dtype) in layouts.items()
    ]
    for name, shape in (
        ("q_resid", (options.tokens, Q_LORA)),
        ("query", (options.tokens, LOCAL_H, QK_DIM)),
        ("kv_latent", (options.tokens, KV_LORA)),
    ):
        specs.append(TensorSpec(name, shape, torch.bfloat16))
    result = run(
        fn=mla_prolog_w8a8_test,
        specs=specs,
        golden_fn=golden_mla_prolog_w8a8,
        config=dict(backend_type=options.backend_type, execution_mode=options.execution_mode, device_id=options.device),
        compare_fn={name: ratio_allclose(atol=0.01, rtol=0.01, max_error_ratio=0) for name in ("q_resid", "query", "kv_latent")},
        compile_only=options.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def build_mla_prolog_tensor_specs(tokens: int = 20):
    """Build one deterministic prolog at the real per-rank shapes.

    The default token count is not a multiple of either tile, so both the cube M
    tail and the rms-norm token tail carry real rows.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(59)

    def normal(*shape, scale=1.0):
        def init():
            return (torch.randn(*shape, generator=generator) * scale).bfloat16()

        return init

    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=normal(tokens, D)),
        TensorSpec("w_q_a", [Q_LORA, D], torch.bfloat16, init_value=normal(Q_LORA, D, scale=0.02)),
        TensorSpec("q_a_norm_weight", [Q_LORA], torch.bfloat16, init_value=normal(Q_LORA)),
        TensorSpec("w_q_b", [LOCAL_H * QK_DIM, Q_LORA], torch.bfloat16, init_value=normal(LOCAL_H * QK_DIM, Q_LORA, scale=0.03)),
        TensorSpec("w_kv_a", [KV_LORA, D], torch.bfloat16, init_value=normal(KV_LORA, D, scale=0.02)),
        TensorSpec("kv_a_norm_weight", [KV_LORA], torch.bfloat16, init_value=normal(KV_LORA)),
        TensorSpec("q_resid", [tokens, Q_LORA], torch.bfloat16),
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16),
        TensorSpec("kv_latent", [tokens, KV_LORA], torch.bfloat16),
    ]


def golden_mla_prolog_case(tensors):
    """Fill the expected outputs for :func:`build_mla_prolog_tensor_specs`."""
    (q_resid, query, kv_latent) = golden_mla_prolog(
        tensors["x"], tensors["w_q_a"], tensors["q_a_norm_weight"], tensors["w_q_b"], tensors["w_kv_a"],
        tensors["kv_a_norm_weight"],
    )
    tensors["q_resid"][:] = q_resid
    tensors["query"][:] = query
    tensors["kv_latent"][:] = kv_latent


def build_mla_absorb_tensor_specs(tokens: int = 20):
    """Build one deterministic absorption at the real per-rank shapes."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(61)

    def init_query():
        return torch.randn(tokens, LOCAL_H, QK_DIM, generator=generator).bfloat16()

    def init_w_k():
        return (torch.randn(LOCAL_H, QK_DIM, KV_LORA, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16, init_value=init_query),
        TensorSpec("w_k", [LOCAL_H, QK_DIM, KV_LORA], torch.bfloat16, init_value=init_w_k),
        TensorSpec("absorbed", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


def golden_mla_absorb_case(tensors):
    """Fill the expected absorbed query for :func:`build_mla_absorb_tensor_specs`."""
    tensors["absorbed"][:] = golden_absorb_query(tensors["query"], tensors["w_k"])


def golden_absorb_query(query: torch.Tensor, w_k: torch.Tensor) -> torch.Tensor:
    """Fold the key half into the query so it attends against the raw latent.

    The exact identity ``q . K^T = (q @ w_k) . latent^T`` uses
    ``K = latent @ w_k^T``. This reference rounds an independent wide product to FP32,
    then rounds the absorbed
    query to the input dtype. BF16 inputs therefore introduce an intermediate BF16
    rounding compared with an expanded FP32 key computation.

    Args:
        query: ``[T, H, QK_DIM]`` from :func:`golden_mla_prolog`.
        w_k: ``[H, QK_DIM, KV_LORA]`` key half from :func:`golden_split_kv_b`.

    Returns:
        ``[T, H, KV_LORA]`` query in latent space, in ``query``'s dtype.
    """
    absorbed = torch.einsum("thd,hdk->thk", query.double(), w_k.double()).float()
    return absorbed.to(query.dtype)


def run_mla_prolog_goldens(golden_prolog, golden_split, golden_absorb_query, golden_absorb_output) -> None:
    """Exercise the MLA prolog and both weight-time absorptions on reduced shapes.

    ``qk_head_dim``, ``v_head_dim`` and ``kv_lora_rank`` stay at their real values
    because :func:`golden_split_kv_b` reads them from the config; the hidden size, the
    query rank, the head count and the token count are reduced.
    """
    torch.manual_seed(23)
    tokens, hidden, q_lora, heads = 4, 64, 48, 2
    qk, v_dim, kv_lora = FLASH.qk_head_dim, FLASH.v_head_dim, FLASH.kv_lora_rank

    x = torch.randn(tokens, hidden, dtype=torch.bfloat16)
    w_q_a = torch.randn(q_lora, hidden, dtype=torch.bfloat16) * 0.05
    q_a_norm = torch.randn(q_lora, dtype=torch.bfloat16)
    w_q_b = torch.randn(heads * qk, q_lora, dtype=torch.bfloat16) * 0.05
    w_kv_a = torch.randn(kv_lora, hidden, dtype=torch.bfloat16) * 0.05
    kv_a_norm = torch.randn(kv_lora, dtype=torch.bfloat16)

    q_resid, query, kv_latent = golden_prolog(x, w_q_a, q_a_norm, w_q_b, w_kv_a, kv_a_norm)
    _check(q_resid.shape == (tokens, q_lora), f"q_resid shape {tuple(q_resid.shape)}")
    _check(query.shape == (tokens, heads, qk), f"query shape {tuple(query.shape)}")
    _check(kv_latent.shape == (tokens, kv_lora), f"kv_latent shape {tuple(kv_latent.shape)}")
    for name, value in (("q_resid", q_resid), ("query", query), ("kv_latent", kv_latent)):
        _check(value.dtype is torch.bfloat16, f"{name} dtype {value.dtype}")
    # The query must be the projection of the published q_resid.
    replayed = (q_resid.double() @ w_q_b.double().t()).float()
    replayed = replayed.unflatten(-1, (heads, qk)).to(torch.bfloat16)
    _check(torch.equal(query, replayed), "query must be projected from the published q_resid")

    w_kv_b = torch.randn(heads * (qk + v_dim), kv_lora, dtype=torch.bfloat16) * 0.02
    w_k, w_v = golden_split(w_kv_b)
    _check(w_k.shape == (heads, qk, kv_lora), f"w_k shape {tuple(w_k.shape)}")
    _check(w_v.shape == (heads, v_dim, kv_lora), f"w_v shape {tuple(w_v.shape)}")

    # Absorbing the key half must leave every score unchanged.
    latent = torch.randn(5, kv_lora, dtype=torch.bfloat16)
    expanded_key = torch.einsum("nk,hdk->nhd", latent.float(), w_k.float())
    scores_expanded = torch.einsum("thd,nhd->thn", query.float(), expanded_key)
    absorbed_query = golden_absorb_query(query, w_k)
    _check(absorbed_query.shape == (tokens, heads, kv_lora), f"absorbed query shape {tuple(absorbed_query.shape)}")
    scores_absorbed = torch.einsum("thk,nk->thn", absorbed_query.float(), latent.float())
    deviation = (scores_absorbed - scores_expanded).abs().max() / scores_expanded.abs().max()
    _check(float(deviation) < 2e-2, f"query absorption changed the scores: {float(deviation)}")

    # Absorbing the value half must leave the projected output unchanged.
    model_dim = 32
    context = torch.randn(tokens, heads, kv_lora, dtype=torch.bfloat16)
    w_o = torch.randn(model_dim, heads * v_dim, dtype=torch.bfloat16) * 0.05
    expanded_out = torch.einsum("thk,hvk->thv", context.float(), w_v.float())
    reference = torch.nn.functional.linear(expanded_out.flatten(-2), w_o.float())
    w_o_absorbed = golden_absorb_output(w_v, w_o)
    _check(w_o_absorbed.shape == (model_dim, heads * kv_lora), f"absorbed o_proj shape {tuple(w_o_absorbed.shape)}")
    projected = torch.nn.functional.linear(context.float().flatten(-2), w_o_absorbed.float())
    deviation = (projected - reference).abs().max() / reference.abs().max()
    _check(float(deviation) < 2e-2, f"output absorption changed the projection: {float(deviation)}")

    _report("mla_prolog")


def golden_prefill_sparse_attn(
    query: torch.Tensor,
    latent_cache: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Attend over the indexer's selected rows in full head space.

    ``topk_indices`` holds **logical per-request positions**, so they are resolved
    through ``block_table`` with :func:`models.glm5_3_flash.metadata.paged_slots` — the
    same mapping the kernel performs on device. A ``-1`` entry marks a padded slot and
    is the *only* masking this kernel may apply: the indexer has already filtered for
    causality and request padding, so re-applying a causal mask here would drop rows the
    reference keeps. A row whose slots are all ``-1`` contributes nothing and yields a
    zero output row rather than a NaN.

    This independent mathematical reference expands keys and values from the gathered
    latent rows in FP32. The implementation absorbs ``w_k`` into the query and rounds
    its intermediates to BF16. This reference materialises ``[T, W, H, *]`` and is
    intended for bounded numerical comparisons.

    Args:
        query: ``[T, H, QK_DIM]`` from :func:`models.glm5_3_flash.mla_prolog.golden_mla_prolog`.
        latent_cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` paged 512-latent pool.
        w_k: ``[H, QK_DIM, KV_LORA]`` key half of ``kv_b_proj``.
        w_v: ``[H, V_DIM, KV_LORA]`` value half of ``kv_b_proj``.
        topk_indices: ``[T, W]`` int32 logical positions, ``-1`` padded.
        block_table: ``[B, BLOCK_TABLE]`` int32 logical-to-physical page map.
        request_ids: ``[T]`` int32 owning request of each query row.

    Returns:
        ``[T, H, V_DIM]`` attention output in ``query``'s dtype.
    """
    valid = topk_indices >= 0
    positions = topk_indices.clamp(min=0).to(torch.int64)
    rows = paged_slots(positions, request_ids.to(torch.int64).unsqueeze(-1), block_table)
    latent = latent_cache[rows].float()
    key = torch.einsum("twk,hdk->twhd", latent, w_k.float())
    value = torch.einsum("twk,hvk->twhv", latent, w_v.float())
    scale = query.shape[-1] ** -0.5
    scores = torch.einsum("thd,twhd->thw", query.float(), key) * scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
    return torch.einsum("thw,twhv->thv", weights, value).to(query.dtype)


def _golden_prefill_sparse_attn_impl(
    query: torch.Tensor,
    latent_cache: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Emulate BF16 absorption, exp weights and contexts with FP32 online softmax.

    The expanded FP32 reference remains :func:`golden_prefill_sparse_attn`.
    """
    tokens, heads, _ = query.shape
    latent_width = latent_cache.shape[-1]
    output = query.new_zeros((tokens, heads, w_v.shape[1]))
    absorbed = golden_absorb_query(query, w_k).float()
    value_weight = w_v.float()
    scale = query.shape[-1] ** -0.5
    selection_width = topk_indices.shape[1]
    for token in range(tokens):
        if selection_width == 0 or int(topk_indices[token, 0]) < 0:
            continue
        running_m = torch.full((heads, 1), PREFILL_NEG_INF, dtype=torch.float32, device=query.device)
        running_l = torch.zeros_like(running_m)
        running_o = torch.zeros((heads, latent_width), dtype=torch.float32, device=query.device)
        for start in range(0, selection_width, PREFILL_ATTN_K_TILE):
            selected = topk_indices[token, start : start + PREFILL_ATTN_K_TILE]
            positions = selected.new_full((PREFILL_ATTN_K_TILE,), -1)
            positions[: selected.numel()] = selected
            valid = positions >= 0
            slots = positions.new_zeros((PREFILL_ATTN_K_TILE,), dtype=torch.int64)
            slots[valid] = paged_slots(positions[valid], request_ids[token], block_table)
            latent = latent_cache[slots].float()
            scores = torch.matmul(absorbed[token], latent.T) * scale
            scores = scores + (~valid).to(torch.float32).unsqueeze(0) * PREFILL_NEG_INF
            block_m = scores.amax(dim=-1, keepdim=True)
            weights = torch.exp(scores - block_m)
            block_l = weights.sum(dim=-1, keepdim=True)
            block_o = torch.matmul(weights.to(torch.bfloat16).float(), latent)
            next_m = torch.maximum(running_m, block_m)
            alpha = torch.exp(running_m - next_m)
            beta = torch.exp(block_m - next_m)
            running_l = alpha * running_l + beta * block_l
            running_o = alpha * running_o + beta * block_o
            running_m = next_m
        context = (running_o / running_l).to(torch.bfloat16).float()
        expanded = torch.einsum("hk,hvk->hv", context, value_weight)
        output[token] = expanded.to(query.dtype)
    return output


def build_prefill_sparse_attn_specs(requests: int = 2, rows: int = 6, pages_per_request: int = 4):
    """Build one deterministic prefill selection with ragged per-token counts.

    The counts ramp past ``ATTN_K_TILE`` so rows span one to four sparse blocks, which
    is what exercises the ``pl.yield_`` merge rather than just the single-block path.
    Row ``rows`` selects nothing, which is the padded-batch row the kernel must answer
    with zeros rather than a division by zero; the one-block path it would otherwise
    have covered is still covered by row 0.

    Every row is front packed, which is the indexer's ABI: the ``-1`` lanes are one
    suffix per row. A fixture that interleaved them would be invalid input, not a
    harder case, so none is built here.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(89)
    tokens = requests * rows
    cache_rows = requests * pages_per_request * BLOCK_SIZE
    visible = pages_per_request * BLOCK_SIZE

    def init_query():
        return torch.randn(tokens, LOCAL_H, QK_DIM, generator=generator).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator).bfloat16()

    def init_w_k():
        return (torch.randn(LOCAL_H, QK_DIM, KV_LORA, generator=generator) * 0.02).bfloat16()

    def init_w_v():
        return (torch.randn(LOCAL_H, V_DIM, KV_LORA, generator=generator) * 0.02).bfloat16()

    def init_block_table():
        pages = torch.randperm(requests * pages_per_request, generator=generator)
        return pages.reshape(requests, pages_per_request).to(torch.int32)

    def init_request_ids():
        return (torch.arange(tokens, dtype=torch.int32) // rows).to(torch.int32)

    def init_indices():
        indices = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
        for token in range(tokens):
            if token == rows:
                continue
            count = min(37 + 113 * (token % rows), visible - 1)
            chosen = torch.randperm(visible - 1, generator=generator)[: count - 1]
            selected = torch.cat([chosen, torch.tensor([visible - 1])]).to(torch.int32)
            indices[token, : selected.numel()] = selected
        return indices

    return [
        TensorSpec("query", [tokens, LOCAL_H, QK_DIM], torch.bfloat16, init_value=init_query),
        TensorSpec("latent_cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
        TensorSpec("block_table", [requests, pages_per_request], torch.int32, init_value=init_block_table),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=init_request_ids),
        TensorSpec("w_k", [LOCAL_H, QK_DIM, KV_LORA], torch.bfloat16, init_value=init_w_k),
        TensorSpec("w_v", [LOCAL_H, V_DIM, KV_LORA], torch.bfloat16, init_value=init_w_v),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32, init_value=init_indices),
        TensorSpec("output", [tokens, LOCAL_H, V_DIM], torch.bfloat16),
    ]


def golden_prefill_sparse_attn_case(tensors):
    """Fill the mixed-precision expected output for the prefill kernel."""
    tensors["output"][:] = _golden_prefill_sparse_attn_impl(
        tensors["query"], tensors["latent_cache"], tensors["w_k"], tensors["w_v"], tensors["topk_indices"],
        tensors["block_table"], tensors["request_ids"],
    )


def run_prefill_sparse_attn_golden(golden_prefill, golden_prefill_impl) -> None:
    """Check independent attention mathematics and the mixed-precision implementation reference."""
    cache, block_table, request_ids, indices, query, w_k, w_v = _sparse_attention_fixture(31)
    tokens, heads, _ = query.shape
    v_dim = w_v.shape[1]
    scale = query.shape[-1] ** -0.5

    out = golden_prefill(query, cache, w_k, w_v, indices, block_table, request_ids)
    _check(out.shape == (tokens, heads, v_dim), f"prefill output shape {tuple(out.shape)}")
    _check(out.dtype is torch.bfloat16, f"prefill output dtype {out.dtype}")

    for token in range(tokens):
        rows = _selected_rows(indices[token], int(request_ids[token]), block_table)
        for head in range(heads):
            keys = torch.stack([w_k[head].float() @ cache[row].float() for row in rows])
            values = torch.stack([w_v[head].float() @ cache[row].float() for row in rows])
            weights = torch.softmax(keys @ query[token, head].float() * scale, dim=0)
            expected = (weights.unsqueeze(-1) * values).sum(dim=0)
            deviation = (out[token, head].float() - expected).abs().max()
            _check(
                float(deviation) < 0.05 * max(float(expected.abs().max()), 0.001),
                f"prefill row {token} head {head} deviates by {float(deviation)}",
            )

    # Padded slots must not reach the softmax.
    trimmed = indices[:, :4].contiguous()
    padded = torch.full((indices.shape[0], 2), -1, dtype=torch.int32)
    same = golden_prefill(query, cache, w_k, w_v, torch.cat([trimmed, padded], dim=1), block_table, request_ids)
    reference = golden_prefill(query, cache, w_k, w_v, trimmed, block_table, request_ids)
    _check(torch.equal(same, reference), "padded -1 slots changed the prefill result")

    _check_sparse_attention_impl(golden_prefill_impl, prefill=True)
    _report("prefill_sparse_attn")


@dataclass
class MlaTensors:
    """NoPE-MLA weights of one DSA layer (FP32). ``kv_b`` stays unsplit."""

    w_q_a: torch.Tensor  # [q_lora, D]
    q_a_norm: torch.Tensor  # [q_lora]
    w_q_b: torch.Tensor  # [H * qk_dim, q_lora]
    w_kv_a: torch.Tensor  # [kv_lora, D] (NoPE: no rope columns)
    kv_a_norm: torch.Tensor  # [kv_lora]
    kv_b: torch.Tensor  # [H * (qk_dim + v_dim), kv_lora]
    w_o: torch.Tensor  # [D, H * v_dim]


def topk_attention_mask(topk_indices: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Additive ``[B, 1, S, kv_len]`` FP32 mask from the indexer output (eager path).

    Port of ``build_attention_mask_from_topk``: scatter the selected positions
    into a per-query count, then map ``0 -> finfo.min``.
    """
    batch_size, q_len, _ = topk_indices.shape
    topk_valid = topk_indices.ge(0) & topk_indices.lt(kv_len)
    safe_indices = topk_indices.clamp(0, kv_len - 1).to(torch.int64)
    selected_counts = torch.zeros(batch_size, q_len, kv_len, dtype=torch.int32)
    selected_counts.scatter_add_(-1, safe_indices, topk_valid.to(torch.int32))
    mask = selected_counts.ne(0)[:, None]
    return torch.where(mask, 0.0, _FLOAT32_MIN)


def synthetic_mla(model: Glm53FlashConfig, generator: torch.Generator) -> MlaTensors:
    heads, qk, v = model.num_attention_heads, model.qk_head_dim, model.v_head_dim
    return MlaTensors(
        w_q_a=_randn((model.q_lora_rank, model.hidden_size), generator),
        q_a_norm=1.0 + torch.randn(model.q_lora_rank, generator=generator) * 0.05,
        w_q_b=_randn((heads * qk, model.q_lora_rank), generator),
        w_kv_a=_randn((model.kv_lora_rank, model.hidden_size), generator),
        kv_a_norm=1.0 + torch.randn(model.kv_lora_rank, generator=generator) * 0.05,
        kv_b=_randn((heads * (qk + v), model.kv_lora_rank), generator),
        w_o=_randn((model.hidden_size, heads * v), generator),
    )


def golden_mla_cache_write(
    cache: torch.Tensor,
    latent: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Scatter one step's latent rows into the paged pool, skipping ``-1`` slots.

    Args:
        cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` pool holding the prior state.
        latent: ``[T, KV_LORA]`` rows from :func:`models.glm5_3_flash.mla_prolog.golden_mla_prolog`.
        slots: ``[T]`` physical rows from ``ForwardMetadata.mla_slots``, ``-1`` for a row
            that owns no cache position.

    Returns:
        A new pool with the selected rows replaced; the input is left untouched.
    """
    updated = cache.clone()
    written = slots >= 0
    updated[slots.to(torch.long)[written]] = latent.to(cache.dtype)[written]
    return updated


def build_mla_cache_tensor_specs(tokens: int = 24, pages: int = 4):
    """Build one deterministic scatter: distinct slots, a padded row, a spare page.

    The slots are distinct and deliberately unordered across pages, and row 3 carries
    ``-1``. ``cache`` is ``InOut`` so its untouched rows are uploaded and can be
    asserted, rather than read back as allocator residue.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(53)
    cache_rows = pages * BLOCK_SIZE

    def init_latent():
        return torch.randn(tokens, KV_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator, dtype=torch.float32).bfloat16()

    def init_slots():
        # Distinct rows, drawn out of order.
        chosen = torch.randperm(cache_rows, generator=generator)[:tokens]
        slots = chosen.to(torch.int32)
        slots[3] = -1
        return slots

    return [
        TensorSpec("latent", [tokens, KV_LORA], torch.bfloat16, init_value=init_latent),
        TensorSpec("slots", [tokens], torch.int32, init_value=init_slots),
        TensorSpec("cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
    ]


def golden_mla_cache_case(tensors):
    """Fill the expected pool state for :func:`build_mla_cache_tensor_specs`."""
    tensors["cache"][:] = golden_mla_cache_write(tensors["cache"], tensors["latent"], tensors["slots"])


def run_mla_cache_golden(golden_cache_write) -> None:
    """Check that the latent cache write scatters by slot and touches nothing else."""
    torch.manual_seed(29)
    rows, kv_lora = 16, 8
    cache = torch.randn(rows, kv_lora, dtype=torch.bfloat16)
    original = cache.clone()
    latent = torch.randn(4, kv_lora, dtype=torch.bfloat16)
    # Row 2 owns no cache position.
    slots = torch.tensor([5, 0, -1, 11], dtype=torch.int32)

    updated = golden_cache_write(cache, latent, slots)
    _check(updated.shape == cache.shape, f"cache shape {tuple(updated.shape)}")
    _check(updated.dtype is cache.dtype, f"cache dtype {updated.dtype}")
    _check(torch.equal(cache, original), "the reference must not write the cache in place")
    written = [slot for slot in slots.tolist() if slot >= 0]
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        _check(torch.equal(updated[slot], latent[row]), f"slot {slot} did not receive its row")
    untouched = [index for index in range(rows) if index not in written]
    _check(torch.equal(updated[untouched], original[untouched]), "the write disturbed rows outside its slot mapping")
    # A -1 slot must not wrap into the tail of the pool.
    _check(torch.equal(updated[-1], original[-1]), "a -1 slot wrapped into the last row")

    _report("mla_cache")


def _golden_decode_sparse_attn_math(
    absorbed_query: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Attend in latent space over the indexer's selected rows.

    Selection, resolution through ``block_table`` and ``-1`` masking follow the prefill
    path exactly. What differs is that the query already carries the key half of
    ``kv_b_proj`` (see
    :func:`models.glm5_3_flash.mla_prolog.golden_absorb_query`), so the cached 512-wide
    rows serve as both keys and values and nothing is expanded per token. The value half
    comes back out through the absorbed ``o_proj``.

    The softmax scale still belongs to the **un-absorbed** head dim, ``QK_DIM ** -0.5``:
    absorption is an exact rewrite of the score, not a change of its space, so the scale
    must not be derived from the query's latent width.

    Each request contributes ``1 + MTP_SPEC_TOKENS`` rows and every row carries its own
    index list, so the gather is ragged even within one request and ``request_ids`` is
    per row rather than per request.

    Args:
        absorbed_query: ``[T, H, KV_LORA]`` query folded into latent space.
        latent_cache: ``[TABLE * BLOCK_SIZE, KV_LORA]`` paged 512-latent pool.
        topk_indices: ``[T, W]`` int32 logical positions, ``-1`` padded.
        block_table: ``[B, BLOCK_TABLE]`` int32 logical-to-physical page map.
        request_ids: ``[T]`` int32 owning request of each decode row.

    Returns:
        ``[T, H, KV_LORA]`` latent-space context in ``absorbed_query``'s dtype.
    """
    valid = topk_indices >= 0
    positions = topk_indices.clamp(min=0).to(torch.int64)
    rows = paged_slots(positions, request_ids.to(torch.int64).unsqueeze(-1), block_table)
    latent = latent_cache[rows].float()
    scores = torch.einsum("thk,twk->thw", absorbed_query.float(), latent) * (QK_DIM**-0.5)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1))
    return torch.einsum("thw,twk->thk", weights, latent).to(absorbed_query.dtype)


def golden_decode_sparse_attn(
    absorbed_query: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
) -> torch.Tensor:
    """Use FP32 block statistics and BF16 exponential operands for PV.

    The BF16 cast precedes each block's PV product; the denominator keeps the
    FP32 exponentials. The FP32 mathematical reference is a separate diagnostic.
    """
    output = torch.zeros_like(absorbed_query)
    query_fp32 = absorbed_query.float()
    heads, width = absorbed_query.shape[1:]
    for token in range(absorbed_query.shape[0]):
        maximum = query_fp32.new_full((heads, 1), DECODE_NEG_INF)
        denominator = torch.zeros_like(maximum)
        numerator = query_fp32.new_zeros((heads, width))
        for start in range(0, topk_indices.shape[1], DECODE_ATTN_K_TILE):
            positions = topk_indices.new_full((DECODE_ATTN_K_TILE,), -1)
            count = min(DECODE_ATTN_K_TILE, topk_indices.shape[1] - start)
            positions[:count] = topk_indices[token, start : start + count]
            if positions[0] < 0:
                continue
            valid = positions >= 0
            latent = latent_cache[0].float().expand(DECODE_ATTN_K_TILE, width).clone()
            rows = paged_slots(positions[valid].long(), request_ids[token].long(), block_table)
            latent[valid] = latent_cache[rows].float()
            scores = (query_fp32[token].double() @ latent.double().T).float() * DECODE_SOFTMAX_SCALE
            scores = scores + (~valid).float().unsqueeze(0) * DECODE_NEG_INF
            block_maximum = scores.amax(dim=-1, keepdim=True)
            exponentials = torch.exp((scores - block_maximum).double()).float()
            block_sum = exponentials.double().sum(dim=-1, keepdim=True).float()
            block_output = (exponentials.bfloat16().double() @ latent.double()).float()
            next_maximum = torch.maximum(maximum, block_maximum)
            alpha = torch.exp((maximum - next_maximum).double()).float()
            beta = torch.exp((block_maximum - next_maximum).double()).float()
            denominator = alpha * denominator + beta * block_sum
            numerator = alpha * numerator + beta * block_output
            maximum = next_maximum
        if topk_indices.shape[1] and topk_indices[token, 0] >= 0:
            output[token] = (numerator.double() / denominator.double()).float().to(absorbed_query.dtype)
    return output


def build_decode_sparse_attn_specs(requests: int = 2, pages_per_request: int = 4):
    """Build one deterministic selection: ragged valid counts, padded tails.

    Live rows select different numbers of positions and keep their own tail, as
    ``index_kpool_always_select_tail`` guarantees. The final row selects nothing and
    must produce exact zeros. Counts ramp past ``ATTN_K_TILE`` so rows span several
    sparse blocks and the merge sees more than one live partial.

    Every row is front packed, which is the indexer's ABI: the ``-1`` lanes are one
    suffix per row. A fixture that interleaved them would be invalid input, not a
    harder case, so none is built here.
    """
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(83)
    tokens = requests * DECODE_ROWS_PER_REQUEST
    cache_rows = requests * pages_per_request * BLOCK_SIZE
    visible = pages_per_request * BLOCK_SIZE

    def init_query():
        return torch.randn(tokens, LOCAL_H, KV_LORA, generator=generator).bfloat16()

    def init_cache():
        return torch.randn(cache_rows, KV_LORA, generator=generator).bfloat16()

    def init_block_table():
        pages = torch.randperm(requests * pages_per_request, generator=generator)
        return pages.reshape(requests, pages_per_request).to(torch.int32)

    def init_request_ids():
        return torch.arange(tokens, dtype=torch.int32) // DECODE_ROWS_PER_REQUEST

    def init_indices():
        indices = torch.full((tokens, TOPK_INDEX_WIDTH), -1, dtype=torch.int32)
        for token in range(tokens):
            if token == tokens - 1:
                continue
            count = min(37 + 61 * token, visible - 1)
            chosen = torch.randperm(visible - 1, generator=generator)[: count - 1]
            selected = torch.cat([chosen, torch.tensor([visible - 1])]).to(torch.int32)
            indices[token, : selected.numel()] = selected
        return indices

    return [
        TensorSpec("absorbed_query", [tokens, LOCAL_H, KV_LORA], torch.bfloat16, init_value=init_query),
        TensorSpec("latent_cache", [cache_rows, KV_LORA], torch.bfloat16, init_value=init_cache),
        TensorSpec("block_table", [requests, pages_per_request], torch.int32, init_value=init_block_table),
        TensorSpec("request_ids", [tokens], torch.int32, init_value=init_request_ids),
        TensorSpec("topk_indices", [tokens, TOPK_INDEX_WIDTH], torch.int32, init_value=init_indices),
        TensorSpec("output", [tokens, LOCAL_H, KV_LORA], torch.bfloat16),
    ]


def golden_decode_sparse_attn_case(tensors):
    """Fill the CPU reference for :func:`build_decode_sparse_attn_specs`."""
    tensors["output"][:] = golden_decode_sparse_attn(
        tensors["absorbed_query"], tensors["latent_cache"], tensors["topk_indices"], tensors["block_table"],
        tensors["request_ids"],
    )


def run_decode_sparse_attn_golden(golden_decode, golden_decode_impl) -> None:
    """Check independent latent attention and the mixed-precision implementation reference."""
    cache, block_table, request_ids, indices, query, w_k, _ = _sparse_attention_fixture(37)
    tokens, heads, _ = query.shape
    kv_lora = cache.shape[-1]
    # The scale belongs to the un-absorbed head dim.
    absorbed = torch.einsum("thd,hdk->thk", query.float(), w_k.float()).to(torch.bfloat16)
    scale = query.shape[-1] ** -0.5

    out = golden_decode(absorbed, cache, indices, block_table, request_ids)
    _check(out.shape == (tokens, heads, kv_lora), f"decode output shape {tuple(out.shape)}")
    _check(out.dtype is torch.bfloat16, f"decode output dtype {out.dtype}")

    for token in range(tokens):
        rows = _selected_rows(indices[token], int(request_ids[token]), block_table)
        latent = torch.stack([cache[row].float() for row in rows])
        for head in range(heads):
            weights = torch.softmax(latent @ absorbed[token, head].float() * scale, dim=0)
            expected = (weights.unsqueeze(-1) * latent).sum(dim=0)
            deviation = (out[token, head].float() - expected).abs().max()
            _check(
                float(deviation) < 0.05 * max(float(expected.abs().max()), 0.001),
                f"decode row {token} head {head} deviates by {float(deviation)}",
            )

    trimmed = indices[:, :4].contiguous()
    padded = torch.full((indices.shape[0], 2), -1, dtype=torch.int32)
    same = golden_decode(absorbed, cache, torch.cat([trimmed, padded], dim=1), block_table, request_ids)
    reference = golden_decode(absorbed, cache, trimmed, block_table, request_ids)
    _check(torch.equal(same, reference), "padded -1 slots changed the decode result")

    _check_sparse_attention_impl(golden_decode_impl, prefill=False)
    _report("decode_sparse_attn")


def run_output_case(argv=None):
    """Prove the goldens on CPU, then validate both epilogs on device.

    The output is FP32 out of BF16 operands, so only the accumulation order separates it
    from the reference and the budget is correspondingly tight.
    """
    import argparse

    from golden import ratio_allclose, run

    run_mla_epilog_goldens()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--mode", choices=["all", "bf16", "native"], default="all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden-input", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args(argv)
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.checkpoint is not None or args.hidden_input is not None:
        if args.mode != "native":
            parser.error("--checkpoint and --hidden-input require --mode native")
    if args.mode in ("all", "native"):
        output_run_native_case(args)
        if args.mode == "native":
            return

    compare = ratio_allclose(atol=1e-4, rtol=1e-3)
    cases = (
        (
            "prefill",
            mla_epilog_prefill_test,
            build_mla_epilog_prefill_specs(args.tokens),
            golden_mla_epilog_prefill_case,
        ),
        (
            "decode",
            mla_epilog_decode_test,
            build_mla_epilog_decode_specs(args.tokens),
            golden_mla_epilog_decode_case,
        ),
    )
    for name, fn, specs, golden_fn in cases:
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
            rtol=0.001,
            atol=0.0001,
            compare_fn={"output": compare},
            compile_only=args.compile_only,
        )
        print(f"{name}: {result}")
        if not result.passed:
            raise SystemExit(result.error or 1)


def run_projection_case(argv=None):
    """Prove the goldens on CPU, then validate the prolog and the absorption on device.

    The BF16 outputs are budgeted at one rounding: kernel and reference share the input
    rounding, so what is left is the FP32 accumulation order plus the output rounding.
    """
    import argparse

    from golden import ratio_allclose, run

    run_mla_prolog_goldens(golden_mla_prolog, golden_split_kv_b, golden_absorb_query, golden_absorb_output)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--mode", choices=["all", "bf16", "native"], default="all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hidden-input", type=Path)
    parser.add_argument("--layer", type=int, default=3)
    args = parser.parse_args(argv)
    if args.checkpoint is not None or args.hidden_input is not None:
        if args.mode != "native":
            parser.error("--checkpoint and --hidden-input require --mode native")
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if args.mode in ("all", "native"):
        projection_run_native_case(args)
        if args.mode == "native":
            return

    compare = ratio_allclose(atol=1e-4, rtol=1.0 / 128)
    result = run(
        fn=mla_prolog_test,
        specs=build_mla_prolog_tensor_specs(args.tokens),
        golden_fn=golden_mla_prolog_case,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
        rtol=1.0 / 128,
        atol=0.0001,
        compare_fn={"q_resid": compare, "query": compare, "kv_latent": compare},
        compile_only=args.compile_only,
    )
    print(f"prolog: {result}")
    if not result.passed:
        raise SystemExit(result.error or 1)

    absorb_result = run(
        fn=mla_absorb_query_test,
        specs=build_mla_absorb_tensor_specs(args.tokens),
        golden_fn=golden_mla_absorb_case,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
        rtol=1.0 / 128,
        atol=0.0001,
        compare_fn={"absorbed": compare},
        compile_only=args.compile_only,
    )
    print(f"absorb_query: {absorb_result}")
    if not absorb_result.passed:
        raise SystemExit(absorb_result.error or 1)


def run_prefill_case(argv=None):
    """Validate sparse prefill against its mixed-precision implementation golden."""
    import argparse

    from golden import ratio_allclose, run

    if prefill_sparse_attn is None:
        raise ValueError(f"sparse MLA attention needs TP >= 4 (LOCAL_H={LOCAL_H} > 16 heads per tile)")

    run_prefill_sparse_attn_golden(golden_prefill_sparse_attn, _golden_prefill_sparse_attn_impl)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)

    result = run(
        fn=prefill_sparse_attn_test,
        specs=build_prefill_sparse_attn_specs(args.requests, args.rows, args.pages),
        golden_fn=golden_prefill_sparse_attn_case,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
        rtol=1.0 / 32,
        atol=0.002,
        compare_fn={"output": ratio_allclose(atol=0.002, rtol=1.0 / 32)},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def run_cache_case(argv=None):
    """Prove the golden on CPU, then validate the scatter on device.

    The device case runs at zero tolerance: a scatter either moves the bytes or is
    wrong, so there is no precision budget to spend.
    """
    import argparse

    from golden import run

    run_mla_cache_golden(golden_mla_cache_write)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=24)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)

    result = run(
        fn=mla_cache_write_test,
        specs=build_mla_cache_tensor_specs(args.tokens, args.pages),
        golden_fn=golden_mla_cache_case,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
        rtol=0.0,
        atol=0.0,
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def run_decode_case(argv=None):
    """Prove the golden on CPU, then validate the sparse decode on device.

    The BF16 output carries the flash block split and the merge rescale on top of the
    input rounding, so its budget is looser than a plain projection's.
    """
    import argparse

    from golden import ratio_allclose, run

    run_decode_sparse_attn_golden(_golden_decode_sparse_attn_math, golden_decode_sparse_attn)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)

    result = run(
        fn=decode_sparse_attn_test,
        specs=build_decode_sparse_attn_specs(args.requests, args.pages),
        golden_fn=golden_decode_sparse_attn_case,
        config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
        rtol=1.0 / 64,
        atol=0.001,
        compare_fn={"output": ratio_allclose(atol=0.001, rtol=1.0 / 64)},
        compile_only=args.compile_only,
    )
    print(result)
    if not result.passed:
        raise SystemExit(result.error or 1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    components = ["all", "projection", "cache", "output", "decode", "prefill"]
    parser.add_argument("--component", choices=components, default="all")
    options, remaining = parser.parse_known_args(argv)
    if options.component == "all" and any(arg in ("-h", "--help") for arg in remaining):
        help_parser = argparse.ArgumentParser(description=__doc__, parents=[parser])
        help_parser.epilog = "Use --component NAME --help for stage options."
        help_parser.parse_args(argv)
    cases = {
        "projection": run_projection_case,
        "cache": run_cache_case,
        "output": run_output_case,
        "decode": run_decode_case,
        "prefill": run_prefill_case,
    }
    selected = cases if options.component == "all" else {options.component: cases[options.component]}
    for run_case in selected.values():
        run_case(remaining)


if __name__ == "__main__":
    main()
