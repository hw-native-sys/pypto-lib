# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Q/KV projection, normalization, RoPE, and window cache publication."""


from pathlib import Path

import pypto.language as pl
from pypto.runtime import pto_isa_include_dir
import torch

from config import FLASH as M, TP_SIZE
from gemm import (
    MX_M_TILE, MX_K_TILE,
    golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum,
)
from rmsnorm import golden_rms_norm


# Dynamic shape variables.
QA_INPUT_ROWS = pl.dynamic("QA_INPUT_ROWS")
QA_SCALE_COLS = pl.dynamic("QA_SCALE_COLS")
QA_OUTPUT_ROWS = pl.dynamic("QA_OUTPUT_ROWS")
KV_INPUT_ROWS = pl.dynamic("KV_INPUT_ROWS")
KV_SCALE_COLS = pl.dynamic("KV_SCALE_COLS")
KV_OUTPUT_ROWS = pl.dynamic("KV_OUTPUT_ROWS")
QB_INPUT_ROWS = pl.dynamic("QB_INPUT_ROWS")
QB_SCALE_COLS = pl.dynamic("QB_SCALE_COLS")
QB_OUTPUT_ROWS = pl.dynamic("QB_OUTPUT_ROWS")

# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank

# tiling
Q_A_N_TILE = 64
Q_B_N_TILE = 256
KV_N_TILE = 64
PROJ_M_TILE = 80
WINDOW_PUBLISH_WORKER_TILE = 12


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "proj_q_a.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _proj_q_a(
    payload: pl.Tensor[[QA_INPUT_ROWS, D], pl.FP8E4M3FN],
    weight: pl.Tensor,
    activation_scale: pl.Tensor[[1, QA_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[QA_OUTPUT_ROWS, Q_LORA], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project ND activations and NZ weights with packed MX scales."""
    return output


@pl.jit.extern(core_type="mixed", aic_source=Path(__file__).parent / "kernels" / "proj_q_b.cpp",
               aiv_source=Path(__file__).parent / "kernels" / "proj_q_b.cpp", include_dirs=(pto_isa_include_dir(),))
def _proj_q_b(
    payload: pl.Tensor[[QB_INPUT_ROWS, Q_LORA], pl.FP8E4M3FN],
    weight: pl.Tensor,
    activation_scale: pl.Tensor[[1, QB_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[QB_OUTPUT_ROWS, LOCAL_H * HEAD_DIM], pl.BF16]],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    output_width: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project Q and rotate the BF16-rounded tail on the paired Vector core."""
    return output


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "proj_kv.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _proj_kv(
    payload: pl.Tensor[[KV_INPUT_ROWS, D], pl.FP8E4M3FN],
    weight: pl.Tensor,
    activation_scale: pl.Tensor[[1, KV_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[KV_OUTPUT_ROWS, HEAD_DIM], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
) -> pl.Tensor:
    """Project ND activations and NZ weights with packed MX scales."""
    return output


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "qkv_quant.cpp",
               include_dirs=(pto_isa_include_dir(),))
def qkv_quant(
    x: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    """Quantize BF16 projection inputs with M16-packed E8M0 scales."""
    ...


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "q_norm_quant.cpp",
               include_dirs=(pto_isa_include_dir(),))
def q_norm_quant(
    projected: pl.Tensor,
    weight: pl.Tensor,
    latent: pl.Out[pl.Tensor],
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
    rows: pl.Scalar[pl.INT32],
):
    """Normalize latent rows and write M16-packed MXFP8 outputs."""
    ...


@pl.jit.inline(auto_scope=False)
def quantize_projection_input(
    x: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Quantize shared Q/KV input into M80-padded payload and M16-packed scales."""
    capacity = pl.tensor.dim(x, 0)
    padded_rows = ((capacity + PROJ_M_TILE - 1) // PROJ_M_TILE) * PROJ_M_TILE
    payload = pl.create_tensor([padded_rows, D], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, padded_rows * (D // 32)], dtype=pl.FP8E8M0)
    quant_blocks = (padded_rows // MX_M_TILE) * (D // MX_K_TILE)
    workers = pl.min(pl.min(quant_blocks, 32), 2 * pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="qkv_quant"):
        qkv_quant(x, payload, scales, num_tokens, padded_rows)
    return payload, scales, padded_rows


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.Tensor,
    query: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
    qa_worker_limit: pl.constexpr = 32,
    qb_worker_limit: pl.constexpr = 32,
):
    """Project NZ QA/QB weights, rotate Q, and retain normalized rows for Index-Q."""
    payload_rows = padded_rows
    qb_payload = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales = pl.create_tensor([1, payload_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    projected = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.BF16)
    qa_tasks = ((num_tokens + PROJ_M_TILE - 1) // PROJ_M_TILE) * (Q_LORA // Q_A_N_TILE)
    qa_workers = pl.max(pl.min(qa_tasks, qa_worker_limit), 1)
    with pl.spmd(qa_workers, name_hint="proj_q_a"):
        projected = _proj_q_a(payload, wq_a, scales, wq_a_scale, projected, num_tokens)
    norm_workers = pl.max(pl.min(payload_rows // MX_M_TILE, 32), 1)
    latent_rows = pl.tensor.dim(query_latent, 0)
    norm_latent = pl.tensor.slice(query_latent, [latent_rows, Q_LORA], [0, 0])
    norm_payload = pl.tensor.slice(qb_payload, [payload_rows, Q_LORA], [0, 0])
    norm_scales = pl.tensor.slice(qb_scales, [1, payload_rows * (Q_LORA // 32)], [0, 0])
    with pl.spmd(norm_workers, name_hint="q_norm_quant"):
        q_norm_quant(projected, q_norm_weight, norm_latent, norm_payload, norm_scales, num_tokens)
    qb_tasks = ((num_tokens + PROJ_M_TILE - 1) // PROJ_M_TILE) * (LOCAL_H * HEAD_DIM // Q_B_N_TILE)
    qb_workers = pl.max(pl.min(qb_tasks, qb_worker_limit), 1)
    with pl.spmd(qb_workers, name_hint="proj_q_b"):
        query = _proj_q_b(
            qb_payload, wq_b, qb_scales, wq_b_scale,
            query, rope_cos, rope_sin, num_tokens, LOCAL_H * HEAD_DIM, 0,
        )
    return query_latent, query, qb_payload, qb_scales


@pl.jit.extern(core_type="aiv", source=Path(__file__).parent / "kernels" / "norm_rope_publish.cpp",
               include_dirs=(pto_isa_include_dir(),))
def norm_rope_publish(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
) -> pl.Tensor:
    """Normalize and rotate BF16 rows, then quantize and publish group-32 FP8 cache pages."""
    ...


@pl.jit.inline(auto_scope=False)
def kv_proj(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor,
    kv_projection: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr = 32,
):
    """Project NZ KV weights to BF16 rows in M80 groups."""
    tasks = ((num_tokens + PROJ_M_TILE - 1) // PROJ_M_TILE) * (HEAD_DIM // KV_N_TILE)
    workers = pl.max(pl.min(tasks, worker_limit), 1)
    with pl.spmd(workers, name_hint="proj_kv", sync_start=True):
        kv_projection = _proj_kv(payload, wkv, scales, wkv_scale, kv_projection, num_tokens, 0)
    return kv_projection


@pl.jit.inline(auto_scope=False)
def window_publish(
    kv_projection: pl.Tensor,
    kv_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    cache_scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Fuse RMSNorm, RoPE, group-32 FP8 quantization, and window cache publication."""
    workers = pl.max(1, pl.min(num_tokens, WINDOW_PUBLISH_WORKER_TILE))
    blocks = pl.tensor.dim(cache_scale, 0)
    local_scales = pl.create_tensor([blocks, 128, 1, HEAD_DIM // 32], dtype=pl.BF16)
    local_rows = pl.reshape(local_scales, [blocks * 128, HEAD_DIM // 32])
    with pl.spmd(workers, name_hint="norm_rope_pub_win", deps=[cache_ready]) as published:
        cache = norm_rope_publish(kv_projection, kv_norm_weight, rope_cos, rope_sin, slots, cache, local_scales, num_tokens)
    output_rows = pl.reshape(cache_scale, [blocks * 128, HEAD_DIM // 32])
    with pl.spmd(1, name_hint="publish_window_scales", deps=[published]) as completed:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens):
            slot = pl.read(slots, [token])
            if slot >= 0 and slot < blocks * 128:
                scale = pl.load(local_rows, [slot, 0], [1, HEAD_DIM // 32])
                pl.store(scale, [slot, 0], output_rows)
    return completed


def golden_rope_interleave(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply adjacent-pair rotary embedding to the final dimension."""
    value = x.float().unflatten(-1, (-1, 2))
    real, imag = value.unbind(dim=-1)
    if inverse:
        sin = -sin
    while cos.ndim < real.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.stack((real * cos - imag * sin, imag * cos + real * sin), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def golden_rope_tail(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing 2*cos.shape[-1] lanes as adjacent (real, imag) pairs."""
    rope_dim = cos.shape[-1] * 2
    rotated = golden_rope_interleave(x[..., -rope_dim:], cos, sin, inverse)
    return torch.cat((x[..., :-rope_dim], rotated), dim=-1)


def golden_qkv_proj_rope(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequential_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    from quant import unpack_fp8_nz

    if wq_a.dtype == torch.float8_e4m3fn:
        wq_a = unpack_fp8_nz(wq_a)
    if wq_b.dtype == torch.float8_e4m3fn:
        wq_b = unpack_fp8_nz(wq_b)
    if wkv.dtype == torch.float8_e4m3fn:
        wkv = unpack_fp8_nz(wkv)

    def project(value, weight, scale):
        if sequential_query and scale is not None:
            return golden_gemm_bf16_via_mxfp8_to_bf16_sequential(value, weight, scale)
        return golden_mxfp8_linear_grouped_sum(value, weight, scale)

    if wq_a_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(x, wq_a, wq_a_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(x, wq_a, wq_a_scale)
    qr = golden_rms_norm(projected, q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = project(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = golden_rms_norm(project(x, wkv, wkv_scale), kv_norm_weight)
    q = golden_rope_tail(q, cos, sin)
    kv = golden_rope_tail(kv, cos, sin)
    return q, kv, qr
