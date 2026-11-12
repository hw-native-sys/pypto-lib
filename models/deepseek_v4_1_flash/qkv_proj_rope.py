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

from config import FLASH as M
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
LOCAL_H = M.num_attention_heads
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
NATIVE_VALIDATION_WORKERS = 32


@pl.jit.extern(core_type="aic", source=Path(__file__).parent / "kernels" / "proj_q_a.cpp",
               include_dirs=(pto_isa_include_dir(),))
def _proj_q_a(
    payload: pl.Tensor[[QA_INPUT_ROWS, D], pl.FP8E4M3FN],
    weight: pl.Tensor,
    activation_scale: pl.Tensor[[1, QA_SCALE_COLS], pl.FP8E8M0],
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[QA_OUTPUT_ROWS, Q_LORA], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
    row_tile: pl.Scalar[pl.INDEX],
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
    row_tile: pl.Scalar[pl.INDEX],
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
    row_tile: pl.Scalar[pl.INDEX],
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
    row_tile: pl.constexpr = 80,
):
    """Quantize shared Q/KV inputs with padded rows and M16-packed scales."""
    capacity = pl.tensor.dim(x, 0)
    projection_rows = ((capacity + row_tile - 1) // row_tile) * row_tile
    # Index-Q reads M80-padded input rows.
    index_rows = ((capacity + PROJ_M_TILE - 1) // PROJ_M_TILE) * PROJ_M_TILE
    padded_rows = pl.max(projection_rows, index_rows)
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
    qa_row_tile: pl.constexpr = 80,
    qb_row_tile: pl.constexpr = 80,
):
    """Project NZ QA/QB weights, rotate Q, and retain normalized rows for Index-Q."""
    # Native QA tiles: M80/N64 or M256/N128; QB tiles: M80/N256 or M128/N256.
    if qa_row_tile == 256:
        qa_n_tile = 128
    else:
        qa_n_tile = Q_A_N_TILE
    payload_rows = padded_rows
    qb_payload = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales = pl.create_tensor([1, payload_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    projected = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.BF16)
    qa_tasks = ((num_tokens + qa_row_tile - 1) // qa_row_tile) * (Q_LORA // qa_n_tile)
    qa_workers = pl.max(pl.min(qa_tasks, qa_worker_limit), 1)
    with pl.spmd(qa_workers, name_hint="proj_q_a"):
        projected = _proj_q_a(payload, wq_a, scales, wq_a_scale, projected, num_tokens, qa_row_tile)
    norm_workers = pl.max(pl.min(payload_rows // MX_M_TILE, 32), 1)
    latent_rows = pl.tensor.dim(query_latent, 0)
    norm_latent = pl.tensor.slice(query_latent, [latent_rows, Q_LORA], [0, 0])
    norm_payload = pl.tensor.slice(qb_payload, [payload_rows, Q_LORA], [0, 0])
    norm_scales = pl.tensor.slice(qb_scales, [1, payload_rows * (Q_LORA // 32)], [0, 0])
    with pl.spmd(norm_workers, name_hint="q_norm_quant"):
        q_norm_quant(projected, q_norm_weight, norm_latent, norm_payload, norm_scales, num_tokens)
    qb_tasks = ((num_tokens + qb_row_tile - 1) // qb_row_tile) * (LOCAL_H * HEAD_DIM // Q_B_N_TILE)
    qb_workers = pl.max(pl.min(qb_tasks, qb_worker_limit), 1)
    with pl.spmd(qb_workers, name_hint="proj_q_b"):
        query = _proj_q_b(
            qb_payload, wq_b, qb_scales, wq_b_scale,
            query, rope_cos, rope_sin, num_tokens, LOCAL_H * HEAD_DIM, 0, qb_row_tile,
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
    row_tile: pl.constexpr = 80,
):
    """Project NZ KV weights to BF16 with bounded active rows."""
    # Native KV tiles: M80/N64 or M256/N256.
    if row_tile == 256:
        n_tile = 256
    else:
        n_tile = KV_N_TILE
    tasks = ((num_tokens + row_tile - 1) // row_tile) * (HEAD_DIM // n_tile)
    workers = pl.max(pl.min(tasks, worker_limit), 1)
    with pl.spmd(workers, name_hint="proj_kv", sync_start=True):
        kv_projection = _proj_kv(payload, wkv, scales, wkv_scale, kv_projection, num_tokens, 0, row_tile)
    return kv_projection


@pl.jit.inline(auto_scope=False)
def window_publish(
    kv_projection: pl.Tensor,
    kv_norm_weight: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Fuse RMSNorm, RoPE, group-32 FP8 quantization, and window cache publication."""
    workers = pl.max(1, pl.min(num_tokens, WINDOW_PUBLISH_WORKER_TILE))
    with pl.spmd(workers, name_hint="norm_rope_pub_win", deps=[cache_ready]) as published:
        cache = norm_rope_publish(kv_projection, kv_norm_weight, rope_cos, rope_sin, slots, cache, num_tokens)
    return published



@pl.jit.inline(auto_scope=False)
def proj_q_a_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
):
    scale_width = weight.shape[0] // 32
    for mt in pl.parallel((rows + 79) // 80):
        scale_begin = mt * 80 * scale_width
        scale_end = scale_begin + 80 * scale_width
        sa = activation_scale[0, scale_begin:scale_end]
        sa = pl.reshape(sa, [80, scale_width])
        sa = pl.tensor.view(sa, [80, scale_width], layout=pl.MX_A_ZZ)
        for worker in pl.spmd(pl.min(worker_limit, 20)):
            for base in pl.range(0, 20, worker_limit):
                if base + worker < 20:
                    row = mt * 80
                    col = (base + worker) * 64
                    full_a = pl.load(payload, [row, 0], [80, 5120], target_memory=pl.Mem.Mat)
                    a0 = pl.tile.slice(full_a, [80, 256], [0, 0])
                    sa0 = pl.load(sa, [0, 0], [80, 8])
                    b0 = pl.load(weight, [0, col], [256, 64], target_memory=pl.Mem.Mat)
                    sb0 = pl.load(weight_scale, [0, col], [8, 64])
                    acc = pl.matmul_mx(a0, sa0, b0, sb0)
                    for k in pl.pipeline(256, 5120, 256, stage=2):
                        av = pl.tile.slice(full_a, [80, 256], [0, k])
                        sav = pl.load(sa, [0, k // 32], [80, 8])
                        bv = pl.load(weight, [k, col], [256, 64], target_memory=pl.Mem.Mat)
                        sbv = pl.load(weight_scale, [k // 32, col], [8, 64])
                        acc = pl.matmul_mx_acc(acc, av, sav, bv, sbv)
                    valid = pl.set_validshape(acc, pl.min(80, rows - row), 64)
                    output = pl.store(valid, [row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def proj_q_b_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    width: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
):
    projected = pl.create_tensor([rows, width], dtype=pl.BF16)
    for mt in pl.parallel((rows + 79) // 80):
        scale_begin = mt * 80 * 40
        scale_end = scale_begin + 80 * 40
        sa = activation_scale[0, scale_begin:scale_end]
        sa = pl.reshape(sa, [80, 40])
        sa = pl.tensor.view(sa, [80, 40], layout=pl.MX_A_ZZ)
        for worker in pl.spmd(pl.min(worker_limit, width // 256)):
            row = mt * 80
            full_a = pl.load(payload, [row, 0], [80, 1280], target_memory=pl.Mem.Mat)
            sa_cache_0 = pl.load(sa, [0, 0], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_1 = pl.load(sa, [0, 4], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_2 = pl.load(sa, [0, 8], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_3 = pl.load(sa, [0, 12], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_4 = pl.load(sa, [0, 16], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_5 = pl.load(sa, [0, 20], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_6 = pl.load(sa, [0, 24], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_7 = pl.load(sa, [0, 28], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_8 = pl.load(sa, [0, 32], [80, 4], target_memory=pl.Mem.Mat)
            sa_cache_9 = pl.load(sa, [0, 36], [80, 4], target_memory=pl.Mem.Mat)
            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 7)
            pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 7)
            sb_cache_0 = pl.load(weight_scale, [0, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_1 = pl.load(weight_scale, [4, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_2 = pl.load(weight_scale, [8, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_3 = pl.load(weight_scale, [12, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_4 = pl.load(weight_scale, [16, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_5 = pl.load(weight_scale, [20, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_6 = pl.load(weight_scale, [24, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_7 = pl.load(weight_scale, [28, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_8 = pl.load(weight_scale, [32, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            sb_cache_9 = pl.load(weight_scale, [36, worker * 256], [4, 256], target_memory=pl.Mem.Mat)
            pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 6)
            for preload in pl.range(4):
                if preload * 256 < 1280:
                    b_pre: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[preload], pl.Mem.Mat] = pl.load(weight, [preload * 256, worker * 256], [256, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, preload)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
            pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
            pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, 4)
            for base in pl.range(0, width // 256, worker_limit):
                if base + worker < width // 256:
                    col = (base + worker) * 256
                    wave = base // worker_limit
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 6)
                    acc = pl.tile.create([80, 256], pl.FP32, target_memory=pl.Mem.Acc)
                    slot = (wave + 0) % 4
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    bv: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.tile.create([256, 256], pl.FP8E4M3FN, target_memory=pl.Mem.Mat)
                    k = 0 * 256 + 0 * 128
                    sa_part = sa_cache_0
                    sb_part = sb_cache_0
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[0], pl.Mem.Right] = pl.tile.extract(bv, 0 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[0 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[0 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_0 = pl.load(weight_scale, [0, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, 4)
                    acc = pl.matmul_mx(a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    k = 0 * 256 + 1 * 128
                    sa_part = sa_cache_1
                    sb_part = sb_cache_1
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[1], pl.Mem.Right] = pl.tile.extract(bv, 1 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[1 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[1 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_1 = pl.load(weight_scale, [4, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    next_col = col
                    if next_col < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        b_next: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.load(weight, [1024, next_col], [256, 256], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    slot = (wave + 1) % 4
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    bv: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.tile.create([256, 256], pl.FP8E4M3FN, target_memory=pl.Mem.Mat)
                    k = 1 * 256 + 0 * 128
                    sa_part = sa_cache_2
                    sb_part = sb_cache_2
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[0], pl.Mem.Right] = pl.tile.extract(bv, 0 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[0 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[0 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_2 = pl.load(weight_scale, [8, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    k = 1 * 256 + 1 * 128
                    sa_part = sa_cache_3
                    sb_part = sb_cache_3
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[1], pl.Mem.Right] = pl.tile.extract(bv, 1 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[1 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[1 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_3 = pl.load(weight_scale, [12, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    next_col = col + worker_limit * 256
                    if next_col < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        b_next: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.load(weight, [0, next_col], [256, 256], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    slot = (wave + 2) % 4
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    bv: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.tile.create([256, 256], pl.FP8E4M3FN, target_memory=pl.Mem.Mat)
                    k = 2 * 256 + 0 * 128
                    sa_part = sa_cache_4
                    sb_part = sb_cache_4
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[0], pl.Mem.Right] = pl.tile.extract(bv, 0 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[0 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[0 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_4 = pl.load(weight_scale, [16, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    k = 2 * 256 + 1 * 128
                    sa_part = sa_cache_5
                    sb_part = sb_cache_5
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[1], pl.Mem.Right] = pl.tile.extract(bv, 1 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[1 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[1 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_5 = pl.load(weight_scale, [20, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    next_col = col + worker_limit * 256
                    if next_col < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        b_next: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.load(weight, [256, next_col], [256, 256], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    slot = (wave + 3) % 4
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    bv: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.tile.create([256, 256], pl.FP8E4M3FN, target_memory=pl.Mem.Mat)
                    k = 3 * 256 + 0 * 128
                    sa_part = sa_cache_6
                    sb_part = sb_cache_6
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[0], pl.Mem.Right] = pl.tile.extract(bv, 0 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[0 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[0 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_6 = pl.load(weight_scale, [24, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    k = 3 * 256 + 1 * 128
                    sa_part = sa_cache_7
                    sb_part = sb_cache_7
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[1], pl.Mem.Right] = pl.tile.extract(bv, 1 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[1 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[1 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_7 = pl.load(weight_scale, [28, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    next_col = col + worker_limit * 256
                    if next_col < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        b_next: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.load(weight, [512, next_col], [256, 256], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    slot = (wave + 4) % 4
                    pl.system.wait_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    bv: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.tile.create([256, 256], pl.FP8E4M3FN, target_memory=pl.Mem.Mat)
                    k = 4 * 256 + 0 * 128
                    sa_part = sa_cache_8
                    sb_part = sb_cache_8
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[0], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[0], pl.Mem.Right] = pl.tile.extract(bv, 0 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[0 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[0 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_8 = pl.load(weight_scale, [32, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 0)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
                    k = 4 * 256 + 1 * 128
                    sa_part = sa_cache_9
                    sb_part = sb_cache_9
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    a_l0: pl.Tile[[80, 128], pl.FP8E4M3FN, pl.MemRef('a_l0', slots=2)[1], pl.Mem.Left] = pl.tile.extract(full_a, 0, k, [80, 128], target_memory=pl.Mem.Left)
                    b_l0: pl.Tile[[128, 256], pl.FP8E4M3FN, pl.MemRef('b_l0', slots=2)[1], pl.Mem.Right] = pl.tile.extract(bv, 1 * 128, 0, [128, 256], target_memory=pl.Mem.Right)
                    sa_l0: pl.Tile[[80, 4], pl.FP8E8M0, pl.MemRef('sa_l0', slots=4)[1 * 2], pl.Mem.LeftScale] = pl.tile.move(sa_part, target_memory=pl.Mem.LeftScale)
                    sb_l0: pl.Tile[[4, 256], pl.FP8E8M0, pl.MemRef('sb_l0', slots=4)[1 * 2], pl.Mem.RightScale] = pl.tile.move(sb_part, target_memory=pl.Mem.RightScale)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, 6)
                        sb_cache_9 = pl.load(weight_scale, [36, col + worker_limit * 256], [4, 256], target_memory=pl.Mem.Mat)
                    pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.M, 1)
                    acc = pl.matmul_mx_acc(acc, a_l0, sa_l0, b_l0, sb_l0)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
                    next_col = col + worker_limit * 256
                    if next_col < width:
                        pl.system.set_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        pl.system.wait_flag(pl.PipeType.MTE1, pl.PipeType.MTE2, slot)
                        b_next: pl.Tile[[256, 256], pl.FP8E4M3FN, pl.MemRef('b_l1', slots=4)[slot], pl.Mem.Mat] = pl.load(weight, [768, next_col], [256, 256], target_memory=pl.Mem.Mat)
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, slot)
                    if col + worker_limit * 256 < width:
                        pl.system.set_flag(pl.PipeType.MTE2, pl.PipeType.MTE1, 6)
                    pl.system.set_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
                    pl.system.wait_flag(pl.PipeType.M, pl.PipeType.FIX, 0)
                    valid = pl.set_validshape(acc, pl.min(80, rows - row), 256)
                    projected = pl.store(valid, [row, col], projected)
                    pl.system.set_flag(pl.PipeType.FIX, pl.PipeType.M, 4)
            pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 0)
            pl.system.wait_flag(pl.PipeType.M, pl.PipeType.MTE1, 1)
            pl.system.wait_flag(pl.PipeType.FIX, pl.PipeType.M, 4)
            pl.system.barrier(pipe=pl.PipeType.ALL)
    for task in pl.spmd((rows + 79) // 80 * (width // 512)):
        row = task // (width // 512) * 80
        col = task % (width // 512) * 512
        valid_rows = pl.min(80, rows - row)
        source = pl.load(projected, [row, col], [80, 512], valid_shape=[valid_rows, 512], target_memory=pl.Mem.Vec)
        cv = pl.load(cosine, [row_start + row, 0], [80, 32], valid_shape=[valid_rows, 32], target_memory=pl.Mem.Vec)
        sv = pl.load(sine, [row_start + row, 0], [80, 32], valid_shape=[valid_rows, 32], target_memory=pl.Mem.Vec)
        with pl.reg_scope():
            full = pl.reg.full_mask(lanes=64)
            lane = pl.reg.index_gen(0, lanes=64, dtype=pl.UINT32)
            one = pl.reg.broadcast(1, lanes=64, dtype=pl.UINT32)
            pair = pl.reg.shift_rights(lane, 1, full)
            swap = pl.reg.bitwise_xor(lane, one)
            odd_bit = pl.reg.bitwise_and(lane, one)
            is_even = pl.reg.compares(odd_bit, 0, full, predicate='eq')
            for r in pl.hardware_loop(valid_rows):
                trig_offset = pl.cast(r * 32, pl.UINT32)
                trig_index = pl.reg.adds(pair, trig_offset, full)
                c = pl.reg.gather(cv, trig_index, full)
                s = pl.reg.gather(sv, trig_index, full)
                raw = pl.reg.load(source, offset=r * 512 + 448, lanes=64)
                x = pl.reg.convert(raw, dtype=pl.FP32)
                swapped = pl.reg.gather_lanes(x, swap)
                a = pl.reg.mul(x, c)
                b = pl.reg.mul(swapped, s)
                real = pl.reg.sub(a, b)
                imag = pl.reg.add(b, a)
                rotated = pl.reg.select(is_even, real, imag)
                bf = pl.reg.convert(rotated, dtype=pl.BF16, rounding='R', saturate=False)
                _rotated = pl.reg.store(source, offset=r * 512 + 448, value=bf)
        valid_source = pl.set_validshape(source, valid_rows, 512)
        output = pl.store(valid_source, [row_start + row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def proj_kv_pypto(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Tensor,
    rows: pl.Scalar[pl.INDEX],
    row_start: pl.Scalar[pl.INDEX],
    worker_limit: pl.constexpr = 32,
):
    scale_width = weight.shape[0] // 32
    for mt in pl.parallel((rows + 79) // 80):
        scale_begin = mt * 80 * scale_width
        scale_end = scale_begin + 80 * scale_width
        sa = activation_scale[0, scale_begin:scale_end]
        sa = pl.reshape(sa, [80, scale_width])
        sa = pl.tensor.view(sa, [80, scale_width], layout=pl.MX_A_ZZ)
        for worker in pl.spmd(pl.min(worker_limit, 8), sync_start=True):
            for base in pl.range(0, 8, worker_limit):
                if base + worker < 8:
                    row = mt * 80
                    col = (base + worker) * 64
                    full_a = pl.load(payload, [row, 0], [80, 5120], target_memory=pl.Mem.Mat)
                    a0 = pl.tile.slice(full_a, [80, 256], [0, 0])
                    sa0 = pl.load(sa, [0, 0], [80, 8])
                    b0 = pl.load(weight, [0, col], [256, 64], target_memory=pl.Mem.Mat)
                    sb0 = pl.load(weight_scale, [0, col], [8, 64])
                    acc = pl.matmul_mx(a0, sa0, b0, sb0)
                    for k in pl.pipeline(256, 5120, 256, stage=2):
                        av = pl.tile.slice(full_a, [80, 256], [0, k])
                        sav = pl.load(sa, [0, k // 32], [80, 8])
                        bv = pl.load(weight, [k, col], [256, 64], target_memory=pl.Mem.Mat)
                        sbv = pl.load(weight_scale, [k // 32, col], [8, 64])
                        acc = pl.matmul_mx_acc(acc, av, sav, bv, sbv)
                    valid = pl.set_validshape(acc, pl.min(80, rows - row), 64)
                    output = pl.store(valid, [row_start + row, col], output)
    return output


@pl.jit.inline(auto_scope=False)
def q_proj_rope_pypto(
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
    projected = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.BF16)
    projected = proj_q_a_pypto(payload, wq_a, scales, wq_a_scale, projected, num_tokens, qa_worker_limit)
    qb_payload = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales = pl.create_tensor([1, payload_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    norm_workers = pl.max(pl.min(payload_rows // MX_M_TILE, 32), 1)
    latent_rows = pl.tensor.dim(query_latent, 0)
    norm_latent = query_latent[:latent_rows, :Q_LORA]
    norm_payload = qb_payload[:payload_rows, :Q_LORA]
    norm_scales = qb_scales[:, :payload_rows * (Q_LORA // 32)]
    with pl.spmd(norm_workers, name_hint="q_norm_quant"):
        q_norm_quant(projected, q_norm_weight, norm_latent, norm_payload, norm_scales, num_tokens)
    query = proj_q_b_pypto(
        qb_payload, wq_b, qb_scales, wq_b_scale,
        query, rope_cos, rope_sin, num_tokens, LOCAL_H * HEAD_DIM, 0, qb_worker_limit,
    )
    return (query_latent, query, qb_payload, qb_scales)


@pl.jit.inline(auto_scope=False)
def kv_proj_pypto(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wkv: pl.Tensor,
    wkv_scale: pl.Tensor,
    kv_projection: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    worker_limit: pl.constexpr = 32,
):
    """Project NZ KV weights to BF16 rows in M80 groups."""
    kv_projection = proj_kv_pypto(payload, wkv, scales, wkv_scale, kv_projection, num_tokens, 0, worker_limit)
    return kv_projection


@pl.jit
def qkv_native(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN, pl.NZ],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN, pl.NZ],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, 512], pl.FP8E4M3FN, pl.NZ],
    wkv_scale: pl.Tensor[[D // 32, 512], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.InOut[pl.Tensor],
    query: pl.InOut[pl.Tensor],
    kv_projection: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    query_latent, query, q_payload, q_scales = q_proj_rope_pypto(
        payload, scales, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        rope_cos, rope_sin, query_latent, query, num_tokens, padded_rows,
        qa_worker_limit=NATIVE_VALIDATION_WORKERS, qb_worker_limit=NATIVE_VALIDATION_WORKERS,
    )
    kv_projection = kv_proj_pypto(
        payload, scales, wkv, wkv_scale, kv_projection, num_tokens, worker_limit=NATIVE_VALIDATION_WORKERS,
    )
    return query_latent, query, kv_projection


@pl.jit
def qkv_cpp(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, 512], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, 512], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.InOut[pl.Tensor],
    query: pl.InOut[pl.Tensor],
    kv_projection: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    padded_rows: pl.Scalar[pl.INDEX],
):
    query_latent, query, q_payload, q_scales = q_proj_rope(
        payload, scales, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        rope_cos, rope_sin, query_latent, query, num_tokens, padded_rows,
        qa_worker_limit=NATIVE_VALIDATION_WORKERS, qb_worker_limit=NATIVE_VALIDATION_WORKERS,
        qa_row_tile=80, qb_row_tile=80,
    )
    kv_projection = kv_proj(
        payload, scales, wkv, wkv_scale, kv_projection, num_tokens, worker_limit=NATIVE_VALIDATION_WORKERS, row_tile=80,
    )
    return query_latent, query, kv_projection



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


def build_qkv_validation_specs(active, capacity):
    """Build model-sized NZ weights, packed MX scales, and canary output buffers."""
    from golden import ScalarSpec, TensorSpec
    from quant import generate_mxfp8_weight_kn, pack_fp8_nz, pack_mx_a_scale
    from quant import _quantize_mxfp8_activation, encode_e8m0

    padded = (capacity + 79) // 80 * 80
    values = {}

    def inputs():
        if values:
            return values
        generator = torch.Generator().manual_seed(17)
        source = torch.randn(padded, D, generator=generator).bfloat16()
        source[active:] = 0
        payload, scale = _quantize_mxfp8_activation(source, scale_alg=0)
        values["payload"] = payload.flatten(-2)
        codes = pack_mx_a_scale(encode_e8m0(scale)).reshape(1, -1)
        values["scales"] = codes.view(torch.float8_e8m0fnu)
        for name, width, reduction, seed in (("wq_a", Q_LORA, D, 19),
                                             ("wq_b", LOCAL_H * HEAD_DIM, Q_LORA, 23),
                                             ("wkv", HEAD_DIM, D, 29)):
            weight, scales = generate_mxfp8_weight_kn(width, reduction, reduction**-0.5, seed=seed)
            values[name] = pack_fp8_nz(weight)
            values[name + "_scale"] = scales
        values["q_norm_weight"] = torch.ones(Q_LORA, dtype=torch.bfloat16)
        frequencies = 10000.0 ** (-torch.arange(0, M.qk_rope_head_dim, 2).float() / M.qk_rope_head_dim)
        angles = (8192 + torch.arange(capacity)).float()[:, None] * frequencies
        values.update(rope_cos=angles.cos(), rope_sin=angles.sin())
        return values

    fp8, scale_dtype = torch.float8_e4m3fn, torch.float8_e8m0fnu
    shapes = (
        ("payload", [padded, D], fp8), ("scales", [1, padded * (D // 32)], scale_dtype),
        ("wq_a", [D, Q_LORA], fp8), ("wq_a_scale", [D // 32, Q_LORA], scale_dtype),
        ("q_norm_weight", [Q_LORA], torch.bfloat16),
        ("wq_b", [Q_LORA, LOCAL_H * HEAD_DIM], fp8),
        ("wq_b_scale", [Q_LORA // 32, LOCAL_H * HEAD_DIM], scale_dtype),
        ("wkv", [D, HEAD_DIM], fp8), ("wkv_scale", [D // 32, HEAD_DIM], scale_dtype),
        ("rope_cos", [capacity, M.qk_rope_head_dim // 2], torch.float32),
        ("rope_sin", [capacity, M.qk_rope_head_dim // 2], torch.float32),
    )
    specs = [TensorSpec(name, shape, dtype, init_value=lambda name=name: inputs()[name])
             for name, shape, dtype in shapes]
    for name, width in (("query_latent", Q_LORA), ("query", LOCAL_H * HEAD_DIM), ("kv_projection", HEAD_DIM)):
        specs.append(TensorSpec(name, [capacity, width], torch.bfloat16, init_value=13.0))
    return specs + [ScalarSpec("num_tokens", torch.int32, active), ScalarSpec("padded_rows", torch.int64, padded)]


def golden_qkv_validation(values):
    """Compute Q normalization, Q RoPE, and raw KV projection with K32 accumulation."""
    from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential as project
    from quant import decode_e8m0, unpack_mx_a_scale, unpack_fp8_nz

    active = int(values["num_tokens"])
    if active == 0:
        return
    padded = values["payload"].shape[0]
    codes = values["scales"].view(torch.uint8).reshape(padded, D // 32)
    scales = decode_e8m0(unpack_mx_a_scale(codes))[:active].repeat_interleave(32, dim=-1)
    source = (values["payload"][:active].float() * scales).bfloat16()
    projected = project(source, unpack_fp8_nz(values["wq_a"]), values["wq_a_scale"])
    latent = golden_rms_norm(projected, values["q_norm_weight"])
    query = project(latent, unpack_fp8_nz(values["wq_b"]), values["wq_b_scale"])
    query = query.reshape(active, LOCAL_H, HEAD_DIM)
    query = golden_rope_tail(query, values["rope_cos"][:active], values["rope_sin"][:active])
    values["query_latent"][:active].copy_(latent)
    values["query"][:active].copy_(query.flatten(1).bfloat16())
    values["kv_projection"][:active].copy_(project(source, unpack_fp8_nz(values["wkv"]), values["wkv_scale"]))


def compare_qkv_output(actual, expected, *, inputs, **kwargs):
    """Check the active prefix and require exact preservation of inactive output rows."""
    from golden import ratio_allclose

    active = int(inputs["num_tokens"])
    compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128, valid_rows=active)
    passed, detail = compare(actual, expected, inputs=inputs, **kwargs)
    tail_ok = torch.equal(actual[active:], expected[active:])
    return passed and tail_ok, f"{detail}; inactive canary preserved={tail_ok}"
