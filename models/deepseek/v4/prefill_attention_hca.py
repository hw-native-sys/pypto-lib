# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 prefill HCA attention bring-up.

Correctness-first standalone for the ratio-128 HCA prefill path.  The first
implementation targets the current debug shape from config, B=1 and S=128,
and intentionally reuses prefill-side kernels instead of importing decode HCA.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ
from prefill_hc_post import golden_prefill_hc_post, prefill_hc_post
from prefill_hc_pre import golden_prefill_hc_pre, prefill_hc_pre
from prefill_compressor_ratio128 import (
    golden_prefill_compressor_ratio128,
    prefill_compressor_ratio128,
)
from prefill_qkv_proj_rope import golden_prefill_qkv_proj_rope, prefill_qkv_proj_rope_core
from prefill_sparse_attn import (
    CMP_BLOCK_NUM as SPARSE_CMP_BLOCK_NUM,
    CMP_MAX_BLOCKS as SPARSE_CMP_MAX_BLOCKS,
    ORI_BLOCK_NUM as SPARSE_ORI_BLOCK_NUM,
    ORI_MAX_BLOCKS as SPARSE_ORI_MAX_BLOCKS,
    ROPE_CHUNK as SPARSE_ROPE_CHUNK,
    ROPE_INTERLEAVE_CHUNK as SPARSE_ROPE_INTERLEAVE_CHUNK,
    TOPK as SPARSE_TOPK,
    _quant_w_per_channel,
    golden_prefill_sparse_attn,
    prefill_sparse_attn,
)


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
ROPE_HALF = ROPE_DIM // 2
NOPE_HEAD_DIM = M.nope_head_dim
NOPE_DIM = NOPE_HEAD_DIM
Q_LORA = M.q_lora_rank
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128
MAIN_OUT_DIM = HEAD_DIM
MAIN_STATE_LEN = COMPRESS_RATIO
IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
START_POS = 0

CMP_K_CHUNK = 512
CMP_OUT_CHUNK = 32
CMP_HEAD_CHUNK = 64
CMP_T_CHUNK = T
CMP_K_BLOCKS = D // CMP_K_CHUNK
CMP_OUT_BLOCKS = MAIN_OUT_DIM // CMP_OUT_CHUNK
CMP_HEAD_BLOCKS = HEAD_DIM // CMP_HEAD_CHUNK
HCA_TOPK_TOKEN_TILE = 16
HCA_KV_STORE_TILE = 16

assert S == COMPRESS_RATIO, "first prefill HCA bring-up targets one ratio-128 prompt chunk"
assert WIN == BLOCK_SIZE, "prefill HCA currently assumes one window page per batch"
assert SPARSE_ORI_BLOCK_NUM == B * SPARSE_ORI_MAX_BLOCKS
assert SPARSE_CMP_BLOCK_NUM == B * SPARSE_CMP_MAX_BLOCKS
assert PREFILL_COMPRESSED_LEN == 1


@pl.jit.inline
def _prefill_hca_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    rope_cos_t: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    cmp_cos: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_sin: pl.Tensor[[PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], pl.FP32],
    start_pos: pl.Scalar[pl.INT32],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cmp_rope_rows"):
        cmp_offset = COMPRESS_RATIO - (start_pos % COMPRESS_RATIO)
        cmp_pos = pl.cast(start_pos + cmp_offset - COMPRESS_RATIO, pl.INDEX)
        cmp_cos[:, :] = pl.cast(freqs_cos[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2], target_type=pl.FP32)
        cmp_sin[:, :] = pl.cast(freqs_sin[cmp_pos : cmp_pos + 1, 0 : ROPE_HEAD_DIM // 2], target_type=pl.FP32)
    for rope_b in pl.parallel(0, B, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_rope_rows"):
            pos = pl.cast(start_pos, pl.INDEX)
            cos_rows = freqs_cos[pos : pos + S, 0:ROPE_DIM]
            sin_rows = freqs_sin[pos : pos + S, 0:ROPE_DIM]
            rope_row = rope_b * S
            rope_cos_t[rope_row : rope_row + S, 0:ROPE_DIM] = cos_rows
            rope_sin_t[rope_row : rope_row + S, 0:ROPE_DIM] = sin_rows
    return rope_cos_t, rope_sin_t, cmp_cos, cmp_sin


@pl.jit.inline
def _prefill_hca_write_prompt_kv(
    kv: pl.Tensor[[T, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[SPARSE_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
):
    ori_kv_flat = pl.reshape(ori_kv, [SPARSE_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * SPARSE_ORI_MAX_BLOCKS])
    for t0 in pl.parallel(0, T, HCA_KV_STORE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_write_prompt_kv"):
            kv_store_b = t0 // S
            kv_store_s = t0 - kv_store_b * S
            kv_store_block_pos = kv_store_b * SPARSE_ORI_MAX_BLOCKS + kv_store_s // BLOCK_SIZE
            kv_store_blk = pl.cast(pl.read(ori_block_table_flat, [kv_store_block_pos]), pl.INDEX)
            kv_store_row = kv_store_blk * BLOCK_SIZE + kv_store_s
            ori_kv_flat[kv_store_row : kv_store_row + HCA_KV_STORE_TILE, 0:HEAD_DIM] = kv[
                t0 : t0 + HCA_KV_STORE_TILE,
                0:HEAD_DIM,
            ]
    return pl.reshape(ori_kv_flat, [SPARSE_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])


@pl.jit
def prefill_attention_hca(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_even_idx: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.INT32],
    cmp_odd_idx: pl.Tensor[[1, ROPE_HEAD_DIM // 2], pl.INT32],
    cmp_kv_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[B, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    kv_cache: pl.Tensor[[SPARSE_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_block_table: pl.Tensor[[B, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_mixed, post, comb = prefill_hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q, kv, qr, qr_scale = prefill_qkv_proj_rope_core(
        x_mixed,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        start_pos,
    )

    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    cmp_cos = pl.create_tensor([PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([PREFILL_COMPRESSED_LEN, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    rope_cos_t, rope_sin_t, cmp_cos, cmp_sin = _prefill_hca_rope_rows(
        freqs_cos,
        freqs_sin,
        rope_cos_t,
        rope_sin_t,
        cmp_cos,
        cmp_sin,
        start_pos,
    )

    kv_cache = _prefill_hca_write_prompt_kv(kv, kv_cache, ori_block_table)
    cmp_dense_kv = pl.create_tensor([B, PREFILL_COMPRESSED_LEN, HEAD_DIM], dtype=pl.FP32)
    cmp_dense_cache = pl.create_tensor([B, IDX_KV_LEN, HEAD_DIM], dtype=pl.BF16)
    cmp_dense_kv, cmp_kv_state, cmp_score_state, cmp_dense_cache = prefill_compressor_ratio128(
        x_mixed,
        cmp_dense_kv,
        cmp_kv_state,
        cmp_score_state,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_cos,
        cmp_sin,
        cmp_even_idx,
        cmp_odd_idx,
        cmp_dense_cache,
        start_pos,
    )
    cmp_kv_flat = pl.reshape(cmp_kv, [SPARSE_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_dense_cache_flat = pl.reshape(cmp_dense_cache, [B * IDX_KV_LEN, HEAD_DIM])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [B * SPARSE_CMP_MAX_BLOCKS])
    for bridge_b in pl.parallel(0, B, 1):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cmp_cache_bridge"):
            cmp_cache_slot = pl.cast(start_pos // COMPRESS_RATIO, pl.INDEX)
            cmp_block_pos = bridge_b * SPARSE_CMP_MAX_BLOCKS + cmp_cache_slot // BLOCK_SIZE
            cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [cmp_block_pos]), pl.INDEX)
            cmp_row = cmp_blk * BLOCK_SIZE + cmp_cache_slot % BLOCK_SIZE
            cmp_dense_row = bridge_b * IDX_KV_LEN + cmp_cache_slot
            cmp_kv_flat[cmp_row : cmp_row + 1, 0:HEAD_DIM] = cmp_dense_cache_flat[
                cmp_dense_row : cmp_dense_row + 1,
                0:HEAD_DIM,
            ]
    cmp_kv = pl.reshape(cmp_kv_flat, [SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        seqused_kv,
        rope_cos_t,
        rope_sin_t,
        even_select_local,
        odd_select_local,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    # Seed static metadata for the imported hc_post inline function.  Passing a
    # reshape expression directly makes the JIT lose the callee parameter shape.
    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = prefill_hc_post(
        attn_out_3d,
        x_hc,
        post,
        comb,
        x_out,
    )
    return x_out


def _quant_w_per_output_channel(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_prefill_attention_hca(tensors):
    import torch

    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_prefill_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post,
        "comb": comb,
    })

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_prefill_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
        "start_pos": tensors["start_pos"],
    })

    ori_kv = tensors["kv_cache"]
    ori_kv.zero_()
    kv_batched = kv.view(B, S, HEAD_DIM)
    for b in range(B):
        for s in range(S):
            blk_id = int(tensors["ori_block_table"][b, s // BLOCK_SIZE].item())
            if blk_id >= 0:
                ori_kv[blk_id, s % BLOCK_SIZE, 0, :] = kv_batched[b, s]

    cmp_kv = tensors["cmp_kv"]
    cmp_kv.zero_()
    dense_kv = torch.zeros(B, PREFILL_COMPRESSED_LEN, HEAD_DIM, dtype=torch.float32)
    dense_cache = torch.zeros(B, IDX_KV_LEN, HEAD_DIM, dtype=torch.bfloat16)
    start_pos = int(tensors["start_pos"])
    cmp_offset = COMPRESS_RATIO - (start_pos % COMPRESS_RATIO)
    cmp_pos = start_pos + cmp_offset - COMPRESS_RATIO
    cmp_tensors = {
        "x": x_mixed,
        "kv": dense_kv,
        "kv_state": tensors["cmp_kv_state"],
        "score_state": tensors["cmp_score_state"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "cos": tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0 : ROPE_HALF].float(),
        "sin": tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0 : ROPE_HALF].float(),
        "kv_cache": dense_cache,
        "start_pos": tensors["start_pos"],
    }
    golden_prefill_compressor_ratio128(cmp_tensors)
    tensors["cmp_kv_state"][:] = cmp_tensors["kv_state"]
    tensors["cmp_score_state"][:] = cmp_tensors["score_state"]
    cmp_cache_slot = start_pos // COMPRESS_RATIO
    for b in range(B):
        blk_id = int(tensors["cmp_block_table"][b, cmp_cache_slot // BLOCK_SIZE].item())
        if blk_id >= 0:
            cmp_kv[blk_id, cmp_cache_slot % BLOCK_SIZE, 0, :] = dense_cache[b, cmp_cache_slot]

    positions = torch.arange(start_pos, start_pos + S, device=tensors["freqs_cos"].device)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_DIM)
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).unsqueeze(0).expand(B, S, ROPE_DIM)
    rope_cos_t = rope_cos_t.reshape(T, ROPE_DIM).contiguous()
    rope_sin_t = rope_sin_t.reshape(T, ROPE_DIM).contiguous()

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": ori_kv,
        "ori_block_table": tensors["ori_block_table"],
        "cmp_kv": cmp_kv,
        "cmp_block_table": tensors["cmp_block_table"],
        "cmp_sparse_indices": tensors["cmp_sparse_indices"],
        "attn_sink": tensors["attn_sink"],
        "seqused_kv": tensors["seqused_kv"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "even_select_local": tensors["even_select_local"],
        "odd_select_local": tensors["odd_select_local"],
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_prefill_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post,
        "comb": comb,
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(start_pos: int = START_POS):
    import torch
    from golden import ScalarSpec, TensorSpec

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale

    def init_x_hc():
        return seeded_uniform((B, S, HC_MULT, D), 1, 0.1)
    def init_hc_attn_fn():
        return seeded_uniform((MIX_HC, HC_DIM), 2, HC_DIM ** -0.5)
    def init_hc_attn_scale():
        return torch.ones(3) * 0.5
    def init_hc_attn_base():
        return torch.zeros(MIX_HC)
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return seeded_uniform((D, Q_LORA), 3, D ** -0.5)
    def init_wq_b():
        return seeded_uniform((Q_LORA, H * HEAD_DIM), 4, Q_LORA ** -0.5)
    def init_wkv():
        return seeded_uniform((D, HEAD_DIM), 5, D ** -0.5)
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)
    def init_even_select_t():
        matrix = torch.zeros((ROPE_HALF, ROPE_DIM))
        for i in range(ROPE_HALF):
            matrix[i, 2 * i] = 1
        return matrix
    def init_odd_select_t():
        matrix = torch.zeros((ROPE_HALF, ROPE_DIM))
        for i in range(ROPE_HALF):
            matrix[i, 2 * i + 1] = 1
        return matrix
    def init_even_select_local():
        matrix = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            matrix[2 * i, i] = 1
        return matrix
    def init_odd_select_local():
        matrix = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            matrix[2 * i + 1, i] = 1
        return matrix
    def init_cmp_wkv():
        return seeded_uniform((D, MAIN_OUT_DIM), 6, D ** -0.5)
    def init_cmp_wgate():
        return seeded_uniform((D, MAIN_OUT_DIM), 7, D ** -0.5)
    def init_cmp_ape():
        return seeded_uniform((COMPRESS_RATIO, MAIN_OUT_DIM), 8, 0.1)
    def init_cmp_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cmp_even_idx():
        return torch.arange(0, ROPE_DIM, 2, dtype=torch.int32).unsqueeze(0)
    def init_cmp_odd_idx():
        return torch.arange(1, ROPE_DIM, 2, dtype=torch.int32).unsqueeze(0)
    def init_cmp_state():
        return torch.zeros(B, MAIN_STATE_LEN, MAIN_OUT_DIM)
    def init_kv_cache():
        return torch.zeros(SPARSE_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_ori_block_table():
        table = torch.full((B, SPARSE_ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            table[b, 0] = b
        return table
    def init_cmp_kv():
        return torch.zeros(SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
    def init_cmp_block_table():
        table = torch.full((B, SPARSE_CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            table[b, 0] = b
        return table
    def init_cmp_sparse_indices():
        topk_idxs = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
        compressed_raw_idx = S + start_pos // COMPRESS_RATIO
        for b in range(B):
            row_base = b * S
            for s in range(S):
                topk_idxs[row_base + s, :s + 1] = torch.arange(s + 1, dtype=torch.int32)
                if s == S - 1:
                    topk_idxs[row_base + s, S] = compressed_raw_idx
        return topk_idxs
    def init_attn_sink():
        return torch.zeros(H)
    def init_seqused_kv():
        return torch.full((B,), S, dtype=torch.int32)
    def init_wo_a():
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 9, O_GROUP_IN ** -0.5)
    def init_wo_b():
        return seeded_uniform((D, O_GROUPS * O_LORA), 10, (O_GROUPS * O_LORA) ** -0.5)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.float32, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("even_select_t", [ROPE_HALF, ROPE_DIM], torch.bfloat16, init_value=init_even_select_t),
        TensorSpec("odd_select_t", [ROPE_HALF, ROPE_DIM], torch.bfloat16, init_value=init_odd_select_t),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.float32, init_value=init_cmp_norm_w),
        TensorSpec("cmp_even_idx", [1, ROPE_HALF], torch.int32, init_value=init_cmp_even_idx),
        TensorSpec("cmp_odd_idx", [1, ROPE_HALF], torch.int32, init_value=init_cmp_odd_idx),
        TensorSpec("cmp_kv_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_state),
        TensorSpec("cmp_score_state", [B, MAIN_STATE_LEN, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_state),
        TensorSpec("kv_cache", [SPARSE_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("ori_block_table", [B, SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, SPARSE_TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("start_pos", torch.int32, start_pos),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=prefill_attention_hca,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_prefill_attention_hca,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_allclose(atol=3e-3, rtol=2.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
