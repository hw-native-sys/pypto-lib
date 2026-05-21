# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ruff: noqa: F401,F403,F405,F821
"""DeepSeek-V4 prefill indexer scaffold.

Kernel body is intentionally empty; golden follows the torch reference for this stage.
"""

import pypto.language as pl

from decode_indexer import *  # noqa: F401,F403
from decode_indexer import build_tensor_specs as _build_tensor_specs
from decode_indexer import _int8_quant_per_row


@pl.jit
def prefill_indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    cos: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    sin: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.FP32],
    even_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    odd_select: pl.Tensor[[ROPE_HEAD_DIM, ROPE_HEAD_DIM // 2], pl.BF16],
    hadamard: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_kv_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_score_state: pl.Tensor[[B, INNER_STATE_LEN, INNER_OUT_DIM], pl.FP32],
    inner_wkv: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_wgate: pl.Tensor[[D, INNER_OUT_DIM], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.FP32],
    idx_kv_cache: pl.Out[pl.Tensor[[B, IDX_KV_LEN, IDX_HEAD_DIM], pl.BF16]],
    score: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.FP32]],
    topk_idxs: pl.Out[pl.Tensor[[B, S, SCORE_LEN], pl.INT32]],
    start_pos: pl.Scalar[pl.INT32],
    offset: pl.Scalar[pl.INT32],
    inner_rotate: pl.Scalar[pl.BOOL],
):
    # TODO: kernel implementation
    return idx_kv_cache, score, topk_idxs


def golden_prefill_indexer(tensors):
    import torch

    from decode_indexer_compressor import golden_compressor

    x = tensors["x"].float()
    qr = tensors["qr"]
    qr_scale = tensors["qr_scale"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float()
    weights_proj = tensors["weights_proj"].float()
    cos = tensors["cos"]
    sin = tensors["sin"]
    hadamard = tensors["hadamard"].float()

    start_pos = int(tensors["start_pos"])
    offset = int(tensors["offset"])
    bsz, seqlen, _ = x.shape
    ratio, rd = COMPRESS_RATIO, ROPE_HEAD_DIM
    end_pos = start_pos + seqlen

    inner_tensors = {
        "x": tensors["x"],
        "kv": tensors["inner_kv"],
        "kv_state": tensors["inner_kv_state"],
        "score_state": tensors["inner_score_state"],
        "wkv": tensors["inner_wkv"],
        "wgate": tensors["inner_wgate"],
        "ape": tensors["inner_ape"],
        "norm_w": tensors["inner_norm_w"],
        "cos": tensors["cos"],
        "sin": tensors["sin"],
        "even_select": tensors["even_select"],
        "odd_select": tensors["odd_select"],
        "hadamard": tensors["hadamard"],
        "kv_cache": tensors["idx_kv_cache"],
        "start_pos": tensors["start_pos"],
        "rotate": tensors["inner_rotate"],
    }
    golden_compressor(inner_tensors)

    cache_len = end_pos // ratio
    score_full = torch.full((bsz, seqlen, SCORE_LEN), FP32_NEG_INF, dtype=torch.float32)
    topk_idxs = torch.full((bsz, seqlen, SCORE_LEN), -1, dtype=torch.int32)

    if cache_len > 0:
        q_i32 = qr.to(torch.int32) @ wq_b.to(torch.int32)
        q = (q_i32.float() * qr_scale * wq_b_scale.view(1, -1)).view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)

        x_pair = q[..., -rd:].unflatten(-1, (-1, 2))
        x0, x1 = x_pair[..., 0], x_pair[..., 1]
        cos_v, sin_v = cos.view(-1), sin.view(-1)
        y0 = (x0 * cos_v - x1 * sin_v).to(torch.bfloat16)
        y1 = (x0 * sin_v + x1 * cos_v).to(torch.bfloat16)
        q = torch.cat([q[..., :-rd], torch.stack([y0, y1], dim=-1).flatten(-2)], dim=-1)
        q = q @ hadamard

        weights = (x @ weights_proj) * WEIGHTS_SCALE
        idx_kv_cache = tensors["idx_kv_cache"].float()
        kv_view = idx_kv_cache[:bsz, :cache_len]
        q_i8, q_scale = _int8_quant_per_row(q.reshape(B * S * IDX_N_HEADS, IDX_HEAD_DIM))
        kv_i8, kv_scale = _int8_quant_per_row(kv_view.reshape(B * cache_len, IDX_HEAD_DIM))
        q_i8 = q_i8.view(B, S, IDX_N_HEADS, IDX_HEAD_DIM)
        q_scale = q_scale.view(B, S, IDX_N_HEADS, 1)
        kv_i8 = kv_i8.view(B, cache_len, IDX_HEAD_DIM)
        kv_scale = kv_scale.view(B, cache_len, 1)
        score_i32 = torch.einsum("bshd,btd->bsht", q_i8.to(torch.int32), kv_i8.to(torch.int32))
        score = score_i32.float() * q_scale * kv_scale.view(B, 1, 1, cache_len)
        score = (torch.relu(score) * weights.unsqueeze(-1)).sum(dim=2)
        score_full[..., :cache_len] = score.to(torch.float32)
        k = min(IDX_TOPK, cache_len)
        _, idx = score.topk(k, dim=-1)
        topk_idxs[..., :k] = idx.to(torch.int32) + offset

    tensors["score"][:] = score_full
    tensors["topk_idxs"][:] = topk_idxs.view(B, S, SCORE_LEN)


def build_tensor_specs(*args, **kwargs):
    return _build_tensor_specs(*args, **kwargs)
