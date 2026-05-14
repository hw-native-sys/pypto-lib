# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 SWA (Sliding Window Attention) decode orchestration — `compress_ratio == 0` path.
Active in layers 0/1/7 of the model (3 of the 8 layers in demo). No KV compression, so neither
compressor nor indexer is invoked; topk for sparse_attn is window_topk_idxs only and the KV cache
holds only the sliding window (no compressed portion). YaRN frequency scaling is also disabled
in this path (model.py:478-479 selects base rope_theta when compress_ratio==0).
Companion files: attention_csa_draft.py (ratio=4)
                 attention_hca_draft.py (ratio=128)."""


import pypto.language as pl

from config import DEMO as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, INT8_SCALE_MAX, INT8_AMAX_EPS
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from sparse_attn import sparse_attn


# model config
B = DECODE_BATCH
S = DECODE_SEQ
T = B * S
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
NOPE_HEAD_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
MAX_SEQ_LEN = M.max_position_embeddings
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
O_GROUP_IN = H * HEAD_DIM // O_GROUPS

# kernel-local (SWA: ratio-0, no compressor/indexer)
ORI_MAX_BLOCKS = 1                  # WIN==BLOCK_SIZE → 1 ori block per batch
MAX_BLOCKS = ORI_MAX_BLOCKS         # SWA: only ori, no cmp portion
BLOCK_NUM = B * MAX_BLOCKS
TOPK = WIN                          # SWA: sparse_attn topk = window only
SPARSE_IDX_TOPK = M.index_topk      # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = 64          # sparse_attn cmp pool size (unused by SWA but part of its contract)
SPARSE_CMP_BLOCK_NUM = B * SPARSE_CMP_MAX_BLOCKS
START_POS = 127      # ScalarSpec default; full-window decode fixture; SWA has no compression constraint

# tiling
Q_PROJ_OUT_CHUNK = 128
Q_PROJ_HEAD_BLOCKS = (H * HEAD_DIM) // Q_PROJ_OUT_CHUNK
SPARSE_ROPE_CHUNK = 16
SPARSE_ROPE_INTERLEAVE_CHUNK = 2 * SPARSE_ROPE_CHUNK


@pl.jit.inline
def attention_swa(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    start_pos: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([B, S, D], dtype=pl.BF16)
    post_t = pl.create_tensor([B, S, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([B, S, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_mixed = hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post_t,
        comb_t,
    )

    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_rope_step"):
        pos = pl.cast(start_pos, pl.INDEX)
        cos_row = pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos, 0]), target_type=pl.FP32)
        sin_row = pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos, 0]), target_type=pl.FP32)
        rope_cos_fp32 = pl.col_expand(
            pl.full([T, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
            cos_row,
        )
        rope_sin_fp32 = pl.col_expand(
            pl.full([T, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0),
            sin_row,
        )
        rope_cos_t = pl.cast(rope_cos_fp32, target_type=pl.BF16)
        rope_sin_t = pl.cast(rope_sin_fp32, target_type=pl.BF16)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q = qkv_proj_rope(
        x_mixed,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        even_select_t,
        odd_select_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    kv_cache_flat = pl.reshape(kv_cache, [BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    block_table_flat = pl.reshape(block_table, [B * MAX_BLOCKS])
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="swa_scatter_kv"):
        ori_slot = start_pos % WIN
        for b in pl.parallel(0, B, 1, chunk=16):
            blk_id = pl.cast(pl.read(block_table_flat, [b]), pl.INDEX)
            dst_row = blk_id * BLOCK_SIZE + ori_slot
            kv_cache_flat = pl.assemble(
                kv_cache_flat,
                kv[b:b + 1, 0:HEAD_DIM],
                [dst_row, 0],
            )
    kv_cache = pl.reshape(kv_cache_flat, [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])

    sparse_topk = pl.create_tensor([T, SPARSE_TOPK], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="swa_topk"):
        idx_row = pl.arange(0, [1, WIN], dtype=pl.INT32)
        pad_row = pl.full([1, SPARSE_IDX_TOPK], dtype=pl.INT32, value=-1)
        sparse_topk_row = pl.concat(idx_row, pad_row)
        sparse_topk = pl.col_expand(
            pl.full([T, SPARSE_TOPK], dtype=pl.INT32, value=-1),
            sparse_topk_row,
        )

    cmp_kv_dummy = pl.create_tensor([SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_block_table_dummy = pl.create_tensor([B, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer, name_hint="swa_cmp_dummy"):
        cmp_block_table_dummy = pl.full([B, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32, value=-1)

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn(
        q,
        kv_cache,
        block_table,
        cmp_kv_dummy,
        cmp_block_table_dummy,
        sparse_topk,
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

    attn_out_3d = pl.create_tensor([B, S, D], dtype=pl.BF16)
    attn_out_3d = pl.reshape(attn_out, [B, S, D])
    x_out = hc_post(
        attn_out_3d,
        x_hc,
        post_t,
        comb_t,
        x_out,
    )
    return x_out


@pl.jit
def attention_swa_test(
    x_hc: pl.Tensor[[B, S, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    even_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    odd_select_t: pl.Tensor[[ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], pl.BF16],
    even_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    odd_select_local: pl.Tensor[[SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, MAX_BLOCKS], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    seqused_kv: pl.Tensor[[B], pl.INT32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[B, S, HC_MULT, D], pl.BF16]],
    start_pos: pl.Scalar[pl.INT32],
):
    x_out = attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, even_select_t, odd_select_t,
        even_select_local, odd_select_local,
        kv_cache, block_table,
        attn_sink, seqused_kv,
        wo_a, wo_b, wo_b_scale,
        x_out,
        start_pos,
    )
    return x_out


def golden_attention_swa(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from sparse_attn import golden_sparse_attn
    from hc_post import golden_hc_post

    # ---- Block.hc_pre (model.py:691) ----
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post_t = torch.zeros(B, S, HC_MULT)
    comb_t = torch.zeros(B, S, HC_MULT, HC_MULT)
    golden_hc_pre({
        "x": tensors["x_hc"],
        "hc_fn": tensors["hc_attn_fn"],
        "hc_scale": tensors["hc_attn_scale"],
        "hc_base": tensors["hc_attn_base"],
        "x_mixed": x_mixed,
        "post": post_t,
        "comb": comb_t,
    })

    # ===== Attention.forward (model.py:484-543), ratio==0 branch =====
    start_pos = int(tensors["start_pos"])
    bsz, seqlen, _ = x_mixed.shape
    win = WIN
    rd = ROPE_HEAD_DIM

    if start_pos == 0:
        return  # prefill — decode-only orchestration skips

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    step_cos = freqs_cos[start_pos:start_pos + 1]                            # [1, rd]
    step_sin = freqs_sin[start_pos:start_pos + 1]
    rope_cos_T = step_cos.expand(T, rd).contiguous()
    rope_sin_T = step_sin.expand(T, rd).contiguous()

    # q + win kv (model.py:495-504)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_qkv_proj_rope({
        "x": x_mixed,
        "norm_w": tensors["attn_norm_w"],
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_T,
        "rope_sin": rope_sin_T,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,                                                              # qr unused on SWA path
        "qr_scale": qr_scale,
    })

    # window topk only (model.py:507; ratio==0 skips lines 508-514)
    topk_idxs = torch.full((T, TOPK), -1, dtype=torch.int32)
    topk_idxs[:, :win] = torch.arange(win, dtype=torch.int32)

    # ori_kv scatter (model.py:530)
    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    ori_slot = start_pos % win
    for b in range(B):
        blk_id = int(block_table[b, ori_slot // BLOCK_SIZE].item())
        intra = ori_slot % BLOCK_SIZE
        kv_cache[blk_id, intra, 0] = kv[b]

    # sparse_attn (model.py:533); window-only uses the full sparse_attn topk contract with an empty cmp tail.
    sparse_topk = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
    sparse_topk[:, :WIN] = topk_idxs
    seqused_kv = tensors["seqused_kv"]
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    cmp_kv_dummy = torch.zeros(SPARSE_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    cmp_block_table_dummy = torch.full((B, SPARSE_CMP_MAX_BLOCKS), -1, dtype=torch.int32)
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": block_table[:, :ORI_MAX_BLOCKS],
        "cmp_kv": cmp_kv_dummy,
        "cmp_block_table": cmp_block_table_dummy,
        "cmp_sparse_indices": sparse_topk,
        "attn_sink": tensors["attn_sink"],
        "seqused_kv": seqused_kv.view(B),
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "even_select_local": tensors["even_select_local"],
        "odd_select_local": tensors["odd_select_local"],
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # ===== Block.hc_post (model.py:694) =====
    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out.view(B, S, D),
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def build_tensor_specs():
    import torch  # type: ignore[import]
    from golden import ScalarSpec, TensorSpec

    def round_half_away_from_zero(x):
        return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = round_half_away_from_zero(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():
        return torch.randn(B, S, HC_MULT, D) * 0.05
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    def init_hc_attn_scale():
        return torch.ones(3) * 0.5
    def init_hc_attn_base():
        return torch.zeros(MIX_HC)
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return torch.randn(D, Q_LORA) / D ** 0.5
    def init_wq_b():
        return torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5
    def init_wkv():
        return torch.randn(D, HEAD_DIM) / D ** 0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_even_select_t():
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i] = 1
        return m
    def init_odd_select_t():
        m = torch.zeros((ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM))
        for i in range(ROPE_HEAD_DIM // 2):
            m[i, 2 * i + 1] = 1
        return m
    def init_even_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i, i] = 1
        return m
    def init_odd_select_local():
        m = torch.zeros((SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK))
        for i in range(SPARSE_ROPE_CHUNK):
            m[2 * i + 1, i] = 1
        return m

    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_kv_cache():
        return init_normalized_cache((BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM))

    def init_block_table():
        tbl = torch.full((B, MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(MAX_BLOCKS):
                tbl[b, j] = b * MAX_BLOCKS + j
        return tbl

    def init_attn_sink():
        return torch.zeros(H)
    def init_seqused_kv():
        return torch.full((B,), min(WIN, START_POS + 1), dtype=torch.int32)
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wq_b_scale = wq_b_scale.view(Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [B, S, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.float32, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [Q_PROJ_HEAD_BLOCKS, Q_PROJ_OUT_CHUNK], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("even_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_even_select_t),
        TensorSpec("odd_select_t", [ROPE_HEAD_DIM // 2, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_odd_select_t),
        TensorSpec("even_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_even_select_local),
        TensorSpec("odd_select_local", [SPARSE_ROPE_INTERLEAVE_CHUNK, SPARSE_ROPE_CHUNK], torch.bfloat16, init_value=init_odd_select_local),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("block_table", [B, MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [B, S, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("start_pos", torch.int32, START_POS),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, ratio_allclose, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=attention_swa_test,
        specs=build_tensor_specs(),
        golden_fn=golden_attention_swa,
        config=RunConfig(
            # qkv_proj_rope and sparse_attn both use W8A8/BF16 stages; the
            # random KV-cache fixture exercises a less diluted attention output
            # than the previous all-zero cache.
            # x_out uses ratio_allclose with the standard W8A8 attention
            # tolerance (atol=1e-4, rtol=1/128, 0.5% outlier allowance);
            # the RunConfig defaults below only apply to any other outputs
            # that lack a custom comparator.
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            },
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                enable_l2_swimlane=args.enable_l2_swimlane,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
