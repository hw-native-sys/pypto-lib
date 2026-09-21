# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Q/KV projection, normalization, and RoPE stages."""

import pypto.language as pl

from models.deepseek_v4_1_flash.attention_ops import (
    MX_M_TILE,
    N_TILE,
    make_norm,
    make_mx_projection,
    make_rope,
)
from models.deepseek_v4_1_flash.config import D, HEAD_DIM, LOCAL_H, Q_LORA, ROPE_DIM, T_DYN


_PREFILL_WORKERS = 64
_PREFILL_PROJECTION_K_TILE = 32


@pl.jit.inline
def _prefill_project_qa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Preserve the scale-corrected group-32 Prefill SWA Q projection."""
    scale_storage = pl.tensor.view(scale, [Q_LORA // 16, D // 2], layout=pl.ND)
    for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
        t0 = mt * MX_M_TILE
        for block in pl.spmd(Q_LORA // N_TILE, name_hint="prefill_attention_q_a"):
            n0 = block * N_TILE
            rows = pl.min(MX_M_TILE, num_tokens - t0)
            acc = pl.tile.full([MX_M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(D // (2 * _PREFILL_PROJECTION_K_TILE)):
                raw = pl.load(scale_storage, [n0 // 16, kb * 32], [N_TILE // 16, 32])
                raw_u8 = pl.reinterpret_view(raw, pl.UINT8)
                codes = pl.ands(pl.cast(pl.reinterpret_view(raw_u8, pl.INT8), pl.INT32), 255)
                scale_pair = pl.reinterpret_view(pl.maximum(pl.shls(codes, 23), 4194304), pl.FP32)
                for half in pl.unroll(2):
                    k0 = (kb * 2 + half) * _PREFILL_PROJECTION_K_TILE
                    if half == 0:
                        gathered_scale = pl.tile.gather_mask(
                            scale_pair, mask_pattern=pl.tile.MaskPattern.P0101
                        )
                    else:
                        gathered_scale = pl.tile.gather_mask(
                            scale_pair, mask_pattern=pl.tile.MaskPattern.P1010
                        )
                    sb = pl.reshape(gathered_scale, [1, N_TILE])
                    source = pl.load(
                        x,
                        [t0, k0],
                        [MX_M_TILE, _PREFILL_PROJECTION_K_TILE],
                        valid_shape=[rows, _PREFILL_PROJECTION_K_TILE],
                    )
                    source = pl.set_validshape(
                        pl.fillpad(source, pad_value=pl.PadValue.zero),
                        MX_M_TILE,
                        _PREFILL_PROJECTION_K_TILE,
                    )
                    value = pl.cast(source, pl.FP32)
                    reduce_tmp = pl.create_tile([MX_M_TILE, _PREFILL_PROJECTION_K_TILE], dtype=pl.FP32)
                    maximum = pl.maximum(pl.row_max(pl.abs(value), tmp_tile=reduce_tmp), 1e-4)
                    bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                    exponent = pl.shrs(pl.add(bits, 8388607), 23)
                    sa = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                    payload = pl.cast(pl.row_expand_div(value, sa), pl.FP8E4M3FN, mode="rint")
                    a = pl.cast(payload, pl.BF16)
                    b = pl.cast(
                        pl.load(weight, [k0, n0], [_PREFILL_PROJECTION_K_TILE, N_TILE]), pl.BF16
                    )
                    part = pl.col_expand_mul(pl.row_expand_mul(pl.matmul(a, b), sa), sb)
                    acc = pl.add(acc, part)
            result = pl.set_validshape(pl.cast(acc, pl.BF16, mode="rint"), rows, N_TILE)
            output = pl.store(result, [t0, n0], output)
    return output


def _make_q_proj_qr(project, normalize):
    @pl.jit.inline(auto_scope=False)
    def q_proj_qr(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x, 0)
        projected = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        project(x, wq_a, wq_a_scale, projected, num_tokens)
        normalize(projected, q_norm_weight, query_latent, num_tokens)
        return query_latent

    return q_proj_qr


def _make_q_proj_rope(project, rotate):
    @pl.jit.inline(auto_scope=False)
    def q_proj_rope(
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(query_latent, 0)
        expanded = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        project(query_latent, wq_b, wq_b_scale, expanded, num_tokens)
        rotate(expanded, rope_cos, rope_sin, query, num_tokens)
        return query

    return q_proj_rope


def _make_kv_proj_rope(project, normalize, rotate):
    @pl.jit.inline(auto_scope=False)
    def kv_proj_rope(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(x, 0)
        projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        project(x, wkv, wkv_scale, projected, num_tokens)
        normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        normalize(projected, kv_norm_weight, normalized, num_tokens)
        rotate(normalized, rope_cos, rope_sin, window_kv, num_tokens)
        return window_kv

    return kv_proj_rope


@pl.jit.inline(auto_scope=False)
def prefill_kv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    projected: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run Prefill KV preprocessing with caller-owned dependency scratch."""
    tokens = pl.tensor.dim(x, 0)
    _project_kv(x, wkv, wkv_scale, projected, num_tokens)
    normalized = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    _prefill_normalize_kv(projected, kv_norm_weight, normalized, num_tokens)
    _prefill_rotate_kv(normalized, rope_cos, rope_sin, window_kv, num_tokens)
    return window_kv


@pl.jit.inline(auto_scope=False)
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
    window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    q_proj_qr(x, wq_a, wq_a_scale, q_norm_weight, query_latent, num_tokens)
    q_proj_rope(query_latent, wq_b, wq_b_scale, rope_cos, rope_sin, query, num_tokens)
    kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, window_kv, num_tokens)
    return query_latent, query, window_kv


def make_qkv_proj_rope_with_deps(
    project_a,
    normalize_a,
    project_b,
    rotate_query,
    project_kv,
    normalize_kv,
    rotate_kv,
):
    """Compose dependency-aware C1A decode primitives behind the shared stage boundary."""

    @pl.jit.inline(auto_scope=False)
    def qkv_proj_rope_with_deps(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        query_latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(x, 0)
        projected_q = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
        qa_tid = project_a(x, wq_a, wq_a_scale, projected_q, num_tokens, ready)
        qr_tid = normalize_a(projected_q, q_norm_weight, query_latent, num_tokens, qa_tid)
        expanded_q = pl.create_tensor([tokens, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        qb_tid = project_b(query_latent, wq_b, wq_b_scale, expanded_q, num_tokens, qr_tid)
        q_tid = rotate_query(expanded_q, rope_cos, rope_sin, query, num_tokens, qb_tid)
        projected_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kv_tid = project_kv(x, wkv, wkv_scale, projected_kv, num_tokens, ready)
        normalized_kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        kvn_tid = normalize_kv(projected_kv, kv_norm_weight, normalized_kv, num_tokens, kv_tid)
        kvr_tid = rotate_kv(normalized_kv, rope_cos, rope_sin, window_kv, num_tokens, kvn_tid)
        return qr_tid, q_tid, kvr_tid

    return qkv_proj_rope_with_deps


_project_qa = make_mx_projection(D, Q_LORA, name_hint="attention_q_a")
_project_qb = make_mx_projection(Q_LORA, LOCAL_H * HEAD_DIM, name_hint="attention_q_b")
_project_kv = make_mx_projection(D, HEAD_DIM, name_hint="attention_kv")
_normalize_q = make_norm(Q_LORA, name_hint="attention_q_norm")
_normalize_kv = make_norm(HEAD_DIM, name_hint="attention_kv_norm")
_rotate_q = make_rope(LOCAL_H, name_hint="attention_q_rope")
_rotate_kv = make_rope(1, name_hint="attention_kv_rope")

_prefill_normalize_q = make_norm(
    Q_LORA, max_workers=_PREFILL_WORKERS, name_hint="prefill_attention_q_norm"
)
_prefill_normalize_kv = make_norm(
    HEAD_DIM, max_workers=_PREFILL_WORKERS, name_hint="prefill_attention_kv_norm"
)
_prefill_rotate_q = make_rope(
    LOCAL_H, max_workers=_PREFILL_WORKERS, name_hint="prefill_attention_q_rope"
)
_prefill_rotate_kv = make_rope(
    1, max_workers=_PREFILL_WORKERS, name_hint="prefill_attention_kv_rope"
)

q_proj_qr = _make_q_proj_qr(_project_qa, _normalize_q)
q_proj_rope = _make_q_proj_rope(_project_qb, _rotate_q)
kv_proj_rope = _make_kv_proj_rope(_project_kv, _normalize_kv, _rotate_kv)
prefill_q_proj_qr = _make_q_proj_qr(_prefill_project_qa, _prefill_normalize_q)
prefill_q_proj_rope = _make_q_proj_rope(_project_qb, _prefill_rotate_q)


__all__ = [
    "kv_proj_rope",
    "make_qkv_proj_rope_with_deps",
    "prefill_kv_proj_rope",
    "prefill_q_proj_qr",
    "prefill_q_proj_rope",
    "q_proj_qr",
    "q_proj_rope",
    "qkv_proj_rope",
]
