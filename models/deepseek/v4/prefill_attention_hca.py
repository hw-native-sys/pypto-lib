# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill HCA attention bring-up.

Correctness-first standalone for the ratio-128 HCA prefill path. The public
contract is single-request token-major prefill: the layer owns the
per-request loop and feeds this op one contiguous run of <=T tokens. HCA
consumes lowered metadata such as position_ids, dense slot mappings, and sparse
indices.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ
from hc_post import golden_hc_post, hc_post
from hc_pre import golden_hc_pre, hc_pre
from prefill_compressor_ratio128 import (
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    golden_prefill_compressor_ratio128,
    prefill_compressor_ratio128,
)
from qkv_proj_rope import golden_qkv_proj_rope, materialize_rope_rows, qkv_proj_rope
from rmsnorm import golden_rms_norm, rms_norm
from prefill_sparse_attn import (
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
IDX_TOPK = M.index_topk
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# prefill_sparse_attn cache/topk contract (mirrors prefill_sparse_attn).
SPARSE_TOPK = WIN + IDX_TOPK
SPARSE_ORI_MAX_BLOCKS = (S + BLOCK_SIZE - 1) // BLOCK_SIZE
SPARSE_ORI_BLOCK_NUM = B * SPARSE_ORI_MAX_BLOCKS
PREFILL_MAX_COMPRESSED = max(1, min(IDX_TOPK, WIN + WIN // 2))
SPARSE_CMP_MAX_BLOCKS = max(1, (PREFILL_MAX_COMPRESSED + BLOCK_SIZE - 1) // BLOCK_SIZE)
SPARSE_CMP_BLOCK_NUM = B * SPARSE_CMP_MAX_BLOCKS

COMPRESS_RATIO = 128
MAIN_OUT_DIM = HEAD_DIM
MAIN_STATE_LEN = COMPRESS_RATIO
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
START_POS = 0
HCA_ORI_BLOCK_NUM = SPARSE_ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = SPARSE_CMP_MAX_BLOCKS

assert S == COMPRESS_RATIO, "first prefill HCA bring-up targets one ratio-128 prompt chunk"
assert WIN == BLOCK_SIZE, "prefill HCA currently assumes one window page per batch"
assert SPARSE_ORI_BLOCK_NUM == B * SPARSE_ORI_MAX_BLOCKS
assert SPARSE_CMP_BLOCK_NUM == B * SPARSE_CMP_MAX_BLOCKS
assert PREFILL_COMPRESSED_LEN == 1


@pl.jit.inline
def prefill_attention_hca(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_norm(x_mixed, attn_norm_w, x_normed)

    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos,
        freqs_sin,
        position_ids,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
    )

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale,
    )

    prefill_compressor_ratio128(
        x_normed, cmp_kv_state, cmp_score_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        freqs_cos, freqs_sin, cmp_kv,
        position_ids, num_tokens, cmp_slot_mapping, state_slot_mapping,
    )

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    prefill_sparse_attn(
        q, kv_cache, ori_block_table, kv,
        cmp_kv, cmp_block_table,
        cmp_sparse_indices, cmp_sparse_lens,
        attn_sink, num_tokens,
        rope_cos_t, rope_sin_t,
        wo_a, wo_b, wo_b_scale, attn_out,
    )
    # Commit new tokens to the cache AFTER sparse_attn reads the pre-update
    # history (the current tokens reach attention via the `kv` overlay).
    kv_cache_flat = pl.reshape(kv_cache, [HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cache_writeback"):
        # No-op self-copy: marks kv_cache add_inout so the runtime orders this
        # write after the gather's read (WAR); see pypto-lib#481.
        kc_touch = kv_cache_flat[0:1, 0:HEAD_DIM]
        kv_cache_flat[0:1, 0:HEAD_DIM] = kc_touch
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]

    hc_post(attn_out, x_hc, post, comb, x_out)
    return x_out


@pl.jit
def prefill_attention_hca_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    cmp_kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    ori_block_table: pl.Tensor[[SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[T], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    prefill_attention_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_kv_state, cmp_score_state, compress_state_block_table,
        kv_cache, ori_slot_mapping, ori_block_table,
        cmp_kv, cmp_block_table,
        cmp_sparse_indices, cmp_sparse_lens,
        position_ids, cmp_slot_mapping, state_slot_mapping,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out, num_tokens,
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

    num_tokens = int(tensors["num_tokens"])
    x_hc_rect = tensors["x_hc"].view(B, S, HC_MULT, D)
    x_hc_flat = x_hc_rect.view(T, HC_MULT, D)
    x_mixed = torch.zeros(T, D, dtype=torch.bfloat16)
    post = torch.zeros(T, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(T, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre({
        "x": x_hc_flat,
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
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    rope_cos_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({
        "x": x_normed.view(T, D),
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos_t,
        "rope_sin": rope_sin_t,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
    })

    ori_kv = tensors["kv_cache"]
    ori_kv_flat = ori_kv.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)

    cmp_kv = tensors["cmp_kv"]
    golden_prefill_compressor_ratio128({
        "x": x_normed.view(T, D),
        "kv_state": tensors["cmp_kv_state"],
        "score_state": tensors["cmp_score_state"],
        "compress_state_block_table": tensors["compress_state_block_table"],
        "wkv": tensors["cmp_wkv"],
        "wgate": tensors["cmp_wgate"],
        "ape": tensors["cmp_ape"],
        "norm_w": tensors["cmp_norm_w"],
        "freqs_cos": tensors["freqs_cos"],
        "freqs_sin": tensors["freqs_sin"],
        "cmp_kv": cmp_kv,
        "position_ids": tensors["position_ids"],
        "num_tokens": tensors["num_tokens"],
        "cmp_slot_mapping": tensors["cmp_slot_mapping"],
        "state_slot_mapping": tensors["state_slot_mapping"],
    })

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": ori_kv,
        "ori_block_table": tensors["ori_block_table"],
        "kv_overlay": kv,
        "cmp_kv": cmp_kv,
        "cmp_block_table": tensors["cmp_block_table"],
        "cmp_sparse_indices": tensors["cmp_sparse_indices"],
        "cmp_sparse_lens": tensors["cmp_sparse_lens"],
        "attn_sink": tensors["attn_sink"],
        "num_tokens": tensors["num_tokens"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            ori_kv_flat[dst_row, :] = kv[t]

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out.view(T, D),
        "residual": x_hc_flat,
        "post": post,
        "comb": comb,
        "y": y.view(T, HC_MULT, D),
    })
    tensors["x_out"][:] = y.view(T, HC_MULT, D)


def build_tensor_specs(
    start_pos: int = START_POS,
    num_tokens: int = T,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    # Single-request geometry: q_len = num_tokens (active prefix), context_len =
    # start_pos (absolute position base, a multiple of S=WIN under chunked prefill).
    context_len = start_pos
    q_len = num_tokens
    if num_tokens <= 0 or num_tokens > T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    if context_len < 0:
        raise ValueError(f"context length must be non-negative, got {context_len}")
    max_position = context_len + q_len - 1
    if max_position >= MAX_SEQ_LEN:
        raise ValueError(f"position id {max_position} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")

    def token_meta():
        # Single-request absolute positions: pos[t] = context_len + local_idx
        # Padding rows keep their arange default; they are inactive.
        local_pos = torch.zeros(T, dtype=torch.int32)
        pos = torch.arange(T, dtype=torch.int32)
        for local_s in range(q_len):
            local_pos[local_s] = local_s
            pos[local_s] = context_len + local_s
        return local_pos, pos

    def validate_overlay_topk(topk_idxs, pos, sparse_lens=None):
        current = {int(pos[t].item()): t for t in range(num_tokens)}

        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            seen_window_abs = set()
            seen_cmp = set()

            sparse_len = SPARSE_TOPK if sparse_lens is None else int(sparse_lens[t].item())
            for raw_i in topk_idxs[t, :sparse_len].tolist():
                raw = int(raw_i)
                if raw < 0:
                    continue
                if raw < WIN:
                    candidates = [
                        key_abs
                        for key_abs in range(key_start_abs, abs_pos + 1)
                        if key_abs % WIN == raw
                    ]
                    if len(candidates) != 1:
                        raise ValueError(f"ambiguous ring raw={raw} for HCA token {t}")
                    key_abs = candidates[0]
                    if key_abs in current:
                        raise ValueError(f"current suffix abs_pos={key_abs} must use HCA overlay for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate window abs_pos={key_abs} for HCA token {t}")
                    seen_window_abs.add(key_abs)
                elif raw < WIN + T:
                    overlay_t = raw - WIN
                    if overlay_t >= num_tokens:
                        raise ValueError(f"HCA overlay raw={raw} points past active tokens for token {t}")
                    key_abs = int(pos[overlay_t].item())
                    if key_abs > abs_pos:
                        raise ValueError(f"HCA overlay raw={raw} is future key abs_pos={key_abs} for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate overlay abs_pos={key_abs} for HCA token {t}")
                    seen_window_abs.add(key_abs)
                else:
                    cmp_slot = raw - (WIN + T)
                    visible_cmp = (abs_pos + 1) // COMPRESS_RATIO
                    if cmp_slot < 0 or cmp_slot >= visible_cmp:
                        raise ValueError(
                            f"HCA compressed raw={raw} slot={cmp_slot} is not visible for token {t}"
                        )
                    if cmp_slot in seen_cmp:
                        raise ValueError(f"duplicate compressed slot={cmp_slot} for HCA token {t}")
                    seen_cmp.add(cmp_slot)

    def cmp_write_records():
        records = []
        for local_s in range(q_len):
            abs_len = context_len + local_s + 1
            if abs_len >= COMPRESS_RATIO and abs_len % COMPRESS_RATIO == 0:
                token_id = local_s
                cmp_slot = abs_len // COMPRESS_RATIO - 1
                records.append((token_id, cmp_slot))
        return records


    def init_x_hc():
        x = torch.empty(T, HC_MULT, D).uniform_(-1, 1)
        x[num_tokens:] = 0
        return x
    # Real layer-9 (HCA, ratio-128) hc_attn scale/base (fn synthetic at real magnitude). A
    # synthetic scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out
    # and the hc residual to near-zero in x_out where W8A8 noise blows up the relative tail.
    # Mirrors decode_attention_hca.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0495
    def init_hc_attn_scale():
        return torch.tensor([0.079046, 0.04213, 0.121901])
    def init_hc_attn_base():
        return torch.tensor([
            -3.3004, 2.5553, -2.2787, -3.4925,
            -3.8197, -3.4161, -2.7144, -2.9181,
            2.362, -2.4746, -2.1352, -3.2216,
            -4.474, 2.2488, -2.1053, -3.1675,
            -2.8362, -1.9042, 2.0432, -3.062,
            -2.7902, -3.0908, -3.002, 3.1161,
        ])
    def init_attn_norm_w():
        return torch.ones(D)
    def init_wq_a():
        return (torch.rand(D, Q_LORA) - 0.5) * D ** -0.5
    def init_wq_b():
        return (torch.rand(Q_LORA, H * HEAD_DIM) - 0.5) * Q_LORA ** -0.5
    def init_wkv():
        return (torch.rand(D, HEAD_DIM) - 0.5) * D ** -0.5
    def init_gamma_cq():
        return torch.ones(Q_LORA)
    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)
    def init_freqs_cos():
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    # Quant-faithful HCA (ratio-128) main compressor fixtures (mean l7/l9 of extract_weights_flash):
    # zero-mean Gaussian BF16 weights at the measured std; RMSNorm gamma near the measured mean.
    # Mirrors decode_attention_hca / decode_compressor_ratio128.
    def init_cmp_wkv():
        return torch.randn(D, MAIN_OUT_DIM) * 0.0246
    def init_cmp_wgate():
        return torch.randn(D, MAIN_OUT_DIM) * 0.0316
    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.0340
    def init_cmp_norm_w():
        return 0.1001 + torch.randn(HEAD_DIM,) * 0.0549
    def init_compress_state_block_table():
        table = torch.full((HCA_STATE_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(HCA_STATE_MAX_BLOCKS):
            table[block] = (block * 17 + 3) % HCA_STATE_MAX_BLOCKS
        return table
    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(table[block].item()) * HCA_STATE_BLOCK_SIZE + intra
    def init_cmp_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for abs_pos in range(max(0, context_len - COMPRESS_RATIO), context_len):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(MAIN_OUT_DIM,) - 0.5) * 0.05
        return state
    def init_cmp_score_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for abs_pos in range(max(0, context_len - COMPRESS_RATIO), context_len):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (torch.rand(MAIN_OUT_DIM,) - 0.5) * 0.05
        return state
    def cache_row_from_table(table, slot):
        block = slot // BLOCK_SIZE
        intra = slot % BLOCK_SIZE
        phys_block = int(table[block].item())
        if phys_block < 0:
            return -1
        return phys_block * BLOCK_SIZE + intra
    def init_kv_cache():
        cache = torch.zeros(HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_ori_block_table()
        if context_len > 0:
            prefix_start = max(0, context_len - WIN)
            prefix = ((torch.rand(context_len, HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)
            for pos_i in range(prefix_start, context_len):
                row = cache_row_from_table(table, pos_i % WIN)
                if row >= 0:
                    cache_flat[row] = prefix[pos_i]
        return cache
    def init_ori_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        local_pos, _ = token_meta()
        table = init_ori_block_table()
        for t in range(num_tokens):
            logical_pos = context_len + int(local_pos[t].item())
            mapping[t] = cache_row_from_table(table, logical_pos % WIN)
        return mapping
    def init_ori_block_table():
        # Single-request paged table: one window page mapped to physical block 0.
        table = torch.full((SPARSE_ORI_MAX_BLOCKS,), -1, dtype=torch.int32)
        table[0] = 0
        return table
    def init_cmp_kv():
        cache = torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_cmp_block_table()
        completed = context_len // COMPRESS_RATIO
        if completed > 0:
            prefix_cmp = ((torch.rand(completed, HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)
            for cmp_slot in range(completed):
                row = cache_row_from_table(table, cmp_slot)
                if row >= 0:
                    cache_flat[row] = prefix_cmp[cmp_slot]
        return cache
    def init_cmp_block_table():
        # Single-request paged table: one compressed page mapped to physical block 0.
        table = torch.full((SPARSE_CMP_MAX_BLOCKS,), -1, dtype=torch.int32)
        table[0] = 0
        return table
    def init_cmp_sparse_indices():
        topk_idxs = torch.full((T, SPARSE_TOPK), -1, dtype=torch.int32)
        local_pos, pos = token_meta()
        current = {int(pos[t].item()): t for t in range(num_tokens)}
        for t in range(num_tokens):
            position = context_len + int(local_pos[t].item())
            window_start = max(0, position - WIN + 1)
            cursor = 0
            for visible_pos in range(window_start, position + 1):
                overlay_t = current.get(visible_pos)
                if overlay_t is not None and overlay_t <= t:
                    topk_idxs[t, cursor] = WIN + overlay_t
                else:
                    topk_idxs[t, cursor] = visible_pos % WIN
                cursor += 1
            visible_cmp = (position + 1) // COMPRESS_RATIO
            for cmp_slot in range(visible_cmp):
                if cursor >= SPARSE_TOPK:
                    break
                topk_idxs[t, cursor] = WIN + T + cmp_slot
                cursor += 1
        sparse_lens = torch.zeros(T, dtype=torch.int32)
        for t in range(num_tokens):
            valid = (topk_idxs[t] >= 0).nonzero()
            if valid.numel():
                sparse_lens[t] = int(valid[-1].item()) + 1
        validate_overlay_topk(topk_idxs, pos, sparse_lens)
        return topk_idxs
    def init_cmp_sparse_lens():
        topk_idxs = init_cmp_sparse_indices()
        lens = torch.zeros(T, dtype=torch.int32)
        for t in range(num_tokens):
            valid = (topk_idxs[t] >= 0).nonzero()
            if valid.numel():
                lens[t] = int(valid[-1].item()) + 1
        return lens
    def init_position_ids():
        return token_meta()[1]
    def init_cmp_slot_mapping():
        out = torch.full((T,), -1, dtype=torch.int64)
        table = init_cmp_block_table()
        records = cmp_write_records()
        for token_id, cmp_slot in records:
            out[token_id] = cache_row_from_table(table, cmp_slot)
        return out
    def init_state_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        _, pos = token_meta()
        for t in range(num_tokens):
            mapping[t] = state_row(int(pos[t].item()))
        return mapping
    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5
    def init_wo_b():
        return (torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
        TensorSpec("hc_attn_fn", [MIX_HC, HC_DIM], torch.float32, init_value=init_hc_attn_fn),
        TensorSpec("hc_attn_scale", [3], torch.float32, init_value=init_hc_attn_scale),
        TensorSpec("hc_attn_base", [MIX_HC], torch.float32, init_value=init_hc_attn_base),
        TensorSpec("attn_norm_w", [D], torch.bfloat16, init_value=init_attn_norm_w),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=init_gamma_cq),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=init_gamma_ckv),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        # Compressor caches are written in-place but not validated here (decode
        # parity); the dedicated prefill_compressor_ratio128 test covers them.
        TensorSpec(
            "cmp_kv_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_state,
        ),
        TensorSpec(
            "cmp_score_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_score_state,
        ),
        TensorSpec("compress_state_block_table", [HCA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec(
            "kv_cache",
            [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
            is_output=True,
        ),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("ori_block_table", [SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec(
            "cmp_kv",
            [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("cmp_block_table", [SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, SPARSE_TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("cmp_sparse_lens", [T], torch.int32, init_value=init_cmp_sparse_lens),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [T], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [T], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


def valid_ratio_reldiff(
    num_tokens: int,
    diff_thd: float,
    pct_thd: float,
    max_diff_hd: float,
):
    """Relative-diff comparator restricted to the valid (active) token rows.

    Mirrors decode_attention_hca's ``ratio_reldiff`` bar and prefill_layer's
    ``valid_ratio_reldiff`` pattern: the packed buffer carries up to
    ``T`` rows but only the leading ``num_tokens`` are active, so the trailing
    padding rows (whose device scratch is undefined) are sliced off before the
    relative-diff check.
    """
    from golden import ratio_reldiff

    base_cmp = ratio_reldiff(diff_thd=diff_thd, pct_thd=pct_thd, max_diff_hd=max_diff_hd)

    def cmp(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        return base_cmp(
            actual[:num_tokens],
            expected[:num_tokens],
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )

    cmp.__name__ = f"valid_ratio_reldiff(num_tokens={num_tokens})"
    return cmp


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 packed prefill HCA correctness test.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="context_len (multiple of S=WIN); fixture-only, lowered into token metadata.")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Active token count (q_len), capped by T; passed to the kernel as num_tokens.")
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    compare_tokens = args.num_tokens

    result = run_jit(
        fn=prefill_attention_hca_test,
        specs=build_tensor_specs(
            args.start_pos,
            args.num_tokens,
        ),
        golden_fn=golden_prefill_attention_hca,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        rtol=1e-2,
        atol=1e-2,
        compile_only=args.compile_only,
        compare_fn={
            "x_out": valid_ratio_reldiff(compare_tokens, diff_thd=5e-3, pct_thd=0.005, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
