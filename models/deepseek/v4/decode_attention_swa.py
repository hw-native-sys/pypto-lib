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

import config
from config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, INT8_SCALE_MAX, INT8_AMAX_EPS
from hc_pre import hc_pre
from hc_post import hc_post
from qkv_proj_rope import qkv_proj_rope
from rmsnorm import attn_norm
# NOTE: `sparse_attn` is imported lower down, AFTER config.SPARSE_TOPK_EFF is set,
# so decode_sparse_attn bakes SWA's pruned sparse-K width at its import (issue #507).


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
TOPK = WIN                          # SWA: sparse_attn topk = window only
SPARSE_IDX_TOPK = M.index_topk      # sparse_attn module's IDX_TOPK (static shape contract)
SPARSE_TOPK = WIN + SPARSE_IDX_TOPK
SPARSE_CMP_MAX_BLOCKS = 64          # sparse_attn cmp pool size (unused by SWA but part of its contract)

# tiling
SPARSE_ROPE_TILE = 16
SPARSE_ROPE_INTERLEAVE_TILE = 2 * SPARSE_ROPE_TILE

# Specialize sparse_attn to SWA's window-only sparse-K width (issue #507): WIN
# (2 blocks after the min-2 floor) instead of the full WIN+IDX_TOPK (5 blocks),
# pruning the padding compressed blocks. Publish it to config BEFORE importing
# sparse_attn (same convention as moe_ep flipping EP_ROUTING_GLOBAL), so the kernel
# and its torch golden bake this width at import.
config.SPARSE_TOPK_EFF = WIN
from decode_sparse_attn import sparse_attn  # noqa: E402  (must follow the config set above)


@pl.jit.inline
def attention_swa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    # Compressed KV pool: SWA has no compressor, so this is an all-zero host
    # placeholder the variant never gathers from (its compressed topk slots are
    # all -1). It must still be a host-provided input (like HCA's real cmp_kv) --
    # an internal create_tensor would be a 512 MB task-heap scratch that exhausts
    # the orchestrator's heap and deadlocks.
    cmp_kv: pl.Tensor[[B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T, HC_MULT, D], pl.BF16],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
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
        for b in pl.parallel(B):
            for s_idx in pl.range(S):
                t = b * S + s_idx
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                cos_row = pl.cast(pl.slice(freqs_cos, [1, ROPE_HEAD_DIM], [pos_b, 0]), target_type=pl.FP32)
                sin_row = pl.cast(pl.slice(freqs_sin, [1, ROPE_HEAD_DIM], [pos_b, 0]), target_type=pl.FP32)
                rope_cos_t = pl.assemble(rope_cos_t, pl.cast(cos_row, target_type=pl.BF16, mode="rint"), [t, 0])
                rope_sin_t = pl.assemble(rope_sin_t, pl.cast(sin_row, target_type=pl.BF16, mode="rint"), [t, 0])

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed_t = attn_norm(x_mixed, attn_norm_w, x_normed_t)
    q = qkv_proj_rope(
        x_normed_t,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        rope_cos_t,
        rope_sin_t,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
    )

    sparse_topk = pl.create_tensor([T, WIN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_overlay_topk"):
        for topk_b in pl.range(B):
            for topk_s in pl.range(S):
                topk_t = topk_b * S + topk_s
                topk_abs_pos = pl.read(position_ids, [topk_t])
                if topk_abs_pos >= WIN - 1:
                    topk_win_start = (topk_abs_pos % WIN) + 1
                    for topk_k in pl.range(WIN):
                        topk_val = (topk_win_start + topk_k) % WIN
                        topk_out = topk_val
                        for topk_os in pl.range(S):
                            if topk_os <= topk_s:
                                topk_overlay_t = topk_b * S + topk_os
                                topk_overlay_pos = pl.read(position_ids, [topk_overlay_t])
                                if topk_val == topk_overlay_pos % WIN:
                                    topk_out = WIN + topk_os
                        pl.write(sparse_topk, [topk_t, topk_k], pl.cast(topk_out, pl.INT32))
                else:
                    for topk_k in pl.range(WIN):
                        if topk_k <= topk_abs_pos:
                            topk_out = topk_k
                            for topk_os in pl.range(S):
                                if topk_os <= topk_s:
                                    topk_overlay_t = topk_b * S + topk_os
                                    topk_overlay_pos = pl.read(position_ids, [topk_overlay_t])
                                    if topk_k == topk_overlay_pos % WIN:
                                        topk_out = WIN + topk_os
                            pl.write(sparse_topk, [topk_t, topk_k], pl.cast(topk_out, pl.INT32))
                        else:
                            pl.write(sparse_topk, [topk_t, topk_k], pl.cast(-1, pl.INT32))

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    sparse_attn(
        q,
        kv_cache,
        block_table,
        kv,
        cmp_kv,
        cmp_block_table,
        sparse_topk,
        attn_sink,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )

    # Commit new tokens to the cache AFTER sparse_attn reads the pre-update
    # history (the current token reaches attention via the `kv` overlay).
    kv_cache_flat = pl.reshape(kv_cache, [B * ORI_MAX_BLOCKS * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_cache_writeback"):
        # No-op self-copy: marks kv_cache add_inout so the runtime orders this
        # write after the gather's read (WAR); see pypto-lib#481.
        kc_touch = kv_cache_flat[0 : 1, 0 : HEAD_DIM]
        kv_cache_flat[0 : 1, 0 : HEAD_DIM] = kc_touch
        for write_t in pl.range(T):
            write_row = pl.cast(pl.read(ori_slot_mapping, [write_t]), pl.INDEX)
            if write_row >= 0:
                kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    x_out = hc_post(
        attn_out,
        x_hc,
        post_t,
        comb_t,
        x_out,
    )
    return x_out


@pl.jit
def attention_swa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.FP32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # KV cache (sliding-window only: [0, WIN) ori; no cmp portion)
    kv_cache: pl.Tensor[[B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    # all-zero compressed-KV placeholder (SWA has no compressor); see attention_swa
    cmp_kv: pl.Tensor[[B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
):
    x_out = attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, block_table, ori_slot_mapping, position_ids,
        cmp_kv, cmp_block_table,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        x_out,
    )
    return x_out


def golden_attention_swa(tensors):
    """End-to-end orchestration for the ratio=0 (SWA) layers.
    Mirrors Block.hc_pre + Attention.forward (decode branch, ratio==0 path: no compressor,
    no indexer, no cmp_kv) + Block.hc_post."""
    import torch

    from hc_pre import golden_hc_pre
    from qkv_proj_rope import golden_qkv_proj_rope
    from rmsnorm import golden_attn_norm
    from decode_sparse_attn import golden_sparse_attn
    from hc_post import golden_hc_post

    # ---- Block.hc_pre (model.py:691) ----
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post_t = torch.zeros(T, HC_MULT)
    comb_t = torch.zeros(T, HC_MULT * HC_MULT)
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
    position_ids = tensors["position_ids"].to(torch.int64)
    bsz, seqlen = B, S
    win = WIN
    rd = ROPE_HEAD_DIM

    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    rope_cos_T = torch.empty(T, rd, dtype=freqs_cos.dtype)
    rope_sin_T = torch.empty(T, rd, dtype=freqs_sin.dtype)
    for t in range(T):
        pos = int(position_ids[t].item())
        rope_cos_T[t] = freqs_cos[pos]
        rope_sin_T[t] = freqs_sin[pos]

    # q + win kv (model.py:495-504)
    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    x_normed = golden_attn_norm(x_mixed, tensors["attn_norm_w"])
    golden_qkv_proj_rope({
        "x": x_normed,
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

    kv_cache = tensors["kv_cache"]
    block_table = tensors["block_table"]
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    # Compressed cache is a host-provided all-zero placeholder for SWA (never
    # gathered: every compressed topk slot stays -1).
    cmp_kv_dummy = tensors["cmp_kv"]
    cmp_block_table_dummy = tensors["cmp_block_table"]

    sparse_topk_all = torch.full((T, WIN), -1, dtype=torch.int32)
    for t in range(T):
        b = t // S
        s = t % S
        abs_pos = int(position_ids[t].item())
        overlay_slots = {
            int(position_ids[b * S + os].item()) % win: os
            for os in range(s + 1)
        }
        if abs_pos >= win - 1:
            win_start = (abs_pos % win) + 1
            vals = ((torch.arange(win, dtype=torch.int32) + win_start) % win).tolist()
        else:
            vals = list(range(abs_pos + 1))
        for k, raw in enumerate(vals):
            if raw in overlay_slots:
                sparse_topk_all[t, k] = WIN + overlay_slots[raw]
            else:
                sparse_topk_all[t, k] = raw
    golden_sparse_attn({
        "q": q,
        "ori_kv": kv_cache,
        "ori_block_table": block_table[:, :ORI_MAX_BLOCKS],
        "mtp_kv_overlay": kv,
        "cmp_kv": cmp_kv_dummy,
        "cmp_block_table": cmp_block_table_dummy,
        "cmp_sparse_indices": sparse_topk_all,
        "attn_sink": tensors["attn_sink"],
        "freqs_cos": rope_cos_T,
        "freqs_sin": rope_sin_T,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    # In-place sliding-window KV-cache update (validated as an inout tensor).
    ori_slot_mapping = tensors["ori_slot_mapping"].to(torch.int64)
    for t in range(T):
        write_row = int(ori_slot_mapping[t].item())
        if write_row >= 0:
            write_blk = write_row // BLOCK_SIZE
            write_intra = write_row % BLOCK_SIZE
            kv_cache[write_blk, write_intra, 0] = kv[t]

    # ===== Block.hc_post (model.py:694) =====
    y = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out,
        "residual": tensors["x_hc"],
        "post": post_t,
        "comb": comb_t,
        "y": y,
    })

    tensors["x_out"][:] = y


def build_tensor_specs(start_pos=None):
    import torch  # type: ignore[import]
    from golden import TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, 0, dtype=torch.bfloat16)

    def quant_w_per_output_channel(w):
        amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.view(1, H * HEAD_DIM)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def quant_w_per_row(w):
        amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = w.float() * scale_quant.unsqueeze(-1)
        w_i32 = torch.round(scaled).to(torch.int32)
        w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
        w_i8 = w_i32.to(torch.float16).to(torch.int8)
        return w_i8, (1.0 / scale_quant).float()

    def init_x_hc():
        return torch.randn(T, HC_MULT, D) * 0.05
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
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_normalized_cache(shape):
        cache = torch.randn(*shape)
        denom = cache.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(EPS)
        return (cache / denom).to(torch.bfloat16)

    def init_kv_cache():
        return init_normalized_cache((B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM))

    def init_block_table():
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_attn_sink():
        return torch.zeros(H)
    def init_start_pos():
        if start_pos is not None:
            return torch.full((B,), start_pos, dtype=torch.int32)
        # Values span the sliding-window regimes and wraparound write slots.
        pattern = torch.tensor([9, 31, 62, WIN - 1], dtype=torch.int32)
        return pattern.repeat((B + pattern.numel() - 1) // pattern.numel())[:B].clone()
    def init_position_ids():
        starts = init_start_pos().to(torch.int64)
        positions = torch.empty((T,), dtype=torch.int32)
        for t in range(T):
            b = t // S
            s = t - b * S
            positions[t] = starts[b] + s
        return positions
    def init_ori_slot_mapping():
        starts = init_start_pos().to(torch.int64)
        block_table = init_block_table().to(torch.int64)
        mapping = torch.full((T,), -1, dtype=torch.int64)
        for t in range(T):
            b = t // S
            s = t - b * S
            slot = int((starts[b].item() + s) % WIN)
            blk = int(block_table[b, slot // BLOCK_SIZE].item())
            mapping[t] = blk * BLOCK_SIZE + slot % BLOCK_SIZE
        return mapping
    def init_wo_a():
        return torch.randn(O_GROUPS, O_LORA, O_GROUP_IN) / O_GROUP_IN ** 0.5
    def init_wo_b():
        return torch.randn(D, O_GROUPS * O_LORA) / (O_GROUPS * O_LORA) ** 0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_row(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
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
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [B * ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache, is_output=True),
        TensorSpec("block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_kv", [B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=lambda: torch.zeros(B * SPARSE_CMP_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)),
        TensorSpec("cmp_block_table", [B, SPARSE_CMP_MAX_BLOCKS], torch.int32,
                   init_value=lambda: (torch.arange(B, dtype=torch.int32).unsqueeze(1) * SPARSE_CMP_MAX_BLOCKS
                                       + torch.arange(SPARSE_CMP_MAX_BLOCKS, dtype=torch.int32).unsqueeze(0))),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None,
                        help="If set, use this single start_pos for all batches; "
                             "otherwise use the default per-batch coverage pattern.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=attention_swa_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_attention_swa,
        runtime_dir=args.runtime_dir,
        golden_data=args.golden_data,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            # Precision reference: CANN model-level criterion (quantized rel < 1e-2) —
            # cann-recipes-infer/.agents/agents/model-infer-reviewer.md
            "x_out": ratio_reldiff(diff_thd=0.01, pct_thd=0.005, max_diff_hd=10),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
