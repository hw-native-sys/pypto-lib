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
contract is token-major packed prefill with static capacity and runtime active
sizes. HCA consumes lowered metadata such as token_to_request, position_ids,
dense slot mappings, and sparse indices.
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
from rmsnorm import golden_attn_norm, attn_norm
from prefill_sparse_attn import (
    CMP_BLOCK_NUM as SPARSE_CMP_BLOCK_NUM,
    CMP_MAX_BLOCKS as SPARSE_CMP_MAX_BLOCKS,
    ORI_BLOCK_NUM as SPARSE_ORI_BLOCK_NUM,
    ORI_MAX_BLOCKS as SPARSE_ORI_MAX_BLOCKS,
    TOPK as SPARSE_TOPK,
    _quant_w_per_channel,
    golden_prefill_sparse_attn,
    prefill_sparse_attn,
)


B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
MAX_REQS = 2
MAX_TOKENS = T
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
PREFILL_COMPRESSED_LEN = S // COMPRESS_RATIO
START_POS = 0
HCA_ORI_BLOCK_NUM = MAX_REQS * SPARSE_ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = MAX_REQS * SPARSE_CMP_MAX_BLOCKS
HCA_CASES = (
    "custom",
    "basic1",
    "basic128",
    "basic96",
    "basic17",
    "suffix64_16",
    "suffix96_17",
    "suffix96_32",
    "suffix100_50",
    "suffix100_128",
    "suffix128_17",
    "suffix255_1",
    "suffix1000_50",
    "hetero_smoke",
    "hetero_boundary",
    "hetero_mixed_cmp_pos",
    "hetero_mixed_overlay_cmp",
    "hetero_boundary_overlay_cmp",
    "hetero_long_suffix_overlay_cmp",
    "hetero_full_capacity_overlay_cmp",
    "hetero_single_long_mix_overlay_cmp",
    "cmp_sparse_lens_boundary",
    "issue511_state_slot_boundary",
)

HCA_KV_STORE_TILE = 16
HCA_WRITEBACK_DEP_COLS = 16

assert S == COMPRESS_RATIO, "first prefill HCA bring-up targets one ratio-128 prompt chunk"
assert WIN == BLOCK_SIZE, "prefill HCA currently assumes one window page per batch"
assert SPARSE_ORI_BLOCK_NUM == B * SPARSE_ORI_MAX_BLOCKS
assert SPARSE_CMP_BLOCK_NUM == B * SPARSE_CMP_MAX_BLOCKS
assert PREFILL_COMPRESSED_LEN == 1
assert HCA_WRITEBACK_DEP_COLS == 16, "16 BF16 values form a 32-byte dependency sentinel"
assert HCA_WRITEBACK_DEP_COLS < HEAD_DIM, "writeback dependency sentinel must be narrower than a KV row"


@pl.jit.inline
def _prefill_hca_cache_writeback_overlay(
    kv: pl.Tensor[[MAX_TOKENS, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    attn_out: pl.Tensor[[MAX_TOKENS, D], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    ori_kv_flat = pl.reshape(ori_kv, [HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for t0 in pl.parallel(0, MAX_TOKENS, HCA_KV_STORE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cache_writeback_overlay"):
            for dt in pl.range(HCA_KV_STORE_TILE):
                t = t0 + dt
                if t < num_tokens:
                    dst_row_raw = pl.read(ori_slot_mapping, [t])
                    if dst_row_raw >= 0:
                        dst_row = pl.cast(dst_row_raw, pl.INDEX)
                        dep = pl.cast(
                            attn_out[t : t + 1, 0:HCA_WRITEBACK_DEP_COLS],
                            target_type=pl.FP32,
                        )
                        dep = pl.mul(dep, 0.0)
                        kv_head = pl.cast(kv[t : t + 1, 0:HCA_WRITEBACK_DEP_COLS], target_type=pl.FP32)
                        kv_head_dep = pl.cast(pl.add(kv_head, dep), target_type=pl.BF16)
                        ori_kv_flat[dst_row : dst_row + 1, 0:HCA_WRITEBACK_DEP_COLS] = kv_head_dep
                        ori_kv_flat[dst_row : dst_row + 1, HCA_WRITEBACK_DEP_COLS:HEAD_DIM] = kv[
                            t : t + 1,
                            HCA_WRITEBACK_DEP_COLS:HEAD_DIM,
                        ]
    return pl.reshape(ori_kv_flat, [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])


@pl.jit
def prefill_attention_hca(
    x_hc: pl.Tensor[[MAX_TOKENS, HC_MULT, D], pl.BF16],
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
    cmp_wkv: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_wgate: pl.Tensor[[D, MAIN_OUT_DIM], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.FP32],
    cmp_kv_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[MAX_REQS, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[MAX_TOKENS, SPARSE_TOPK], pl.INT32],
    cmp_sparse_lens: pl.Tensor[[MAX_TOKENS], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    state_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[MAX_TOKENS, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    x_mixed = hc_pre(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed = attn_norm(x_mixed, attn_norm_w, x_normed)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
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
    q = qkv_proj_rope(
        x_normed,
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

    cmp_kv, cmp_kv_state, cmp_score_state = prefill_compressor_ratio128(
        x_normed,
        cmp_kv_state,
        cmp_score_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        freqs_cos,
        freqs_sin,
        cmp_kv,
        token_to_request,
        position_ids,
        num_tokens,
        cmp_slot_mapping,
        state_slot_mapping,
    )

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        kv,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        cmp_sparse_lens,
        attn_sink,
        token_to_request,
        num_tokens,
        rope_cos_t,
        rope_sin_t,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    kv_cache = _prefill_hca_cache_writeback_overlay(kv, kv_cache, ori_slot_mapping, attn_out, num_tokens)

    x_out = hc_post(
        attn_out,
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
    x_normed = golden_attn_norm(x_mixed, tensors["attn_norm_w"])
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
        "token_to_request": tensors["token_to_request"],
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
        "token_to_request": tensors["token_to_request"],
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
    tensors["x_out"][:] = y.view(MAX_TOKENS, HC_MULT, D)


def _resolve_hca_case(
    start_pos: int = START_POS,
    num_tokens: int = MAX_TOKENS,
    hca_case: str = "custom",
    hetero_smoke: bool = False,
    hetero_boundary: bool = False,
    hetero_mixed_cmp_pos: bool = False,
):
    alias_count = int(hetero_smoke) + int(hetero_boundary) + int(hetero_mixed_cmp_pos)
    if alias_count > 1:
        raise ValueError("--hetero-* flags are mutually exclusive")
    if hca_case != "custom" and alias_count:
        raise ValueError("--hca-case cannot be combined with --hetero-* aliases")
    if hetero_smoke:
        hca_case = "hetero_smoke"
    elif hetero_boundary:
        hca_case = "hetero_boundary"
    elif hetero_mixed_cmp_pos:
        hca_case = "hetero_mixed_cmp_pos"

    if hca_case == "custom":
        q_lens_values = [num_tokens, 0]
        context_lens_values = [start_pos, 0]
    elif hca_case == "basic1":
        q_lens_values = [1, 0]
        context_lens_values = [0, 0]
    elif hca_case == "basic128":
        q_lens_values = [128, 0]
        context_lens_values = [0, 0]
    elif hca_case == "basic96":
        q_lens_values = [96, 0]
        context_lens_values = [0, 0]
    elif hca_case == "basic17":
        q_lens_values = [17, 0]
        context_lens_values = [0, 0]
    elif hca_case == "suffix64_16":
        q_lens_values = [16, 0]
        context_lens_values = [64, 0]
    elif hca_case == "suffix96_17":
        q_lens_values = [17, 0]
        context_lens_values = [96, 0]
    elif hca_case == "suffix96_32":
        q_lens_values = [32, 0]
        context_lens_values = [96, 0]
    elif hca_case == "suffix100_50":
        q_lens_values = [50, 0]
        context_lens_values = [100, 0]
    elif hca_case == "suffix100_128":
        q_lens_values = [128, 0]
        context_lens_values = [100, 0]
    elif hca_case == "suffix128_17":
        q_lens_values = [17, 0]
        context_lens_values = [128, 0]
    elif hca_case == "suffix255_1":
        q_lens_values = [1, 0]
        context_lens_values = [255, 0]
    elif hca_case == "suffix1000_50":
        q_lens_values = [50, 0]
        context_lens_values = [1000, 0]
    elif hca_case == "hetero_smoke":
        q_lens_values = [32, 64]
        context_lens_values = [64, 0]
    elif hca_case == "hetero_boundary":
        q_lens_values = [32, 32]
        context_lens_values = [96, 96]
    elif hca_case == "hetero_mixed_cmp_pos":
        q_lens_values = [32, 32]
        context_lens_values = [96, 224]
    elif hca_case == "hetero_mixed_overlay_cmp":
        q_lens_values = [50, 16]
        context_lens_values = [100, 64]
    elif hca_case == "hetero_boundary_overlay_cmp":
        q_lens_values = [50, 40]
        context_lens_values = [100, 230]
    elif hca_case == "hetero_long_suffix_overlay_cmp":
        q_lens_values = [30, 20]
        context_lens_values = [200, 500]
    elif hca_case == "hetero_full_capacity_overlay_cmp":
        q_lens_values = [96, 32]
        context_lens_values = [1000, 352]
    elif hca_case == "hetero_single_long_mix_overlay_cmp":
        q_lens_values = [1, 127]
        context_lens_values = [255, 385]
    elif hca_case == "cmp_sparse_lens_boundary":
        q_lens_values = [50, 0]
        context_lens_values = [100, 0]
    elif hca_case == "issue511_state_slot_boundary":
        q_lens_values = [2, 2]
        context_lens_values = [126, 254]
    else:
        raise ValueError(f"unknown --hca-case {hca_case!r}; expected one of {HCA_CASES}")

    active_tokens = sum(q_lens_values)
    if active_tokens <= 0 or active_tokens > MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [1, {MAX_TOKENS}], got {active_tokens}")
    if min(context_lens_values) < 0:
        raise ValueError(f"context lengths must be non-negative, got {context_lens_values}")
    max_position = max((ctx + q_len - 1 for ctx, q_len in zip(context_lens_values, q_lens_values) if q_len > 0), default=0)
    if max_position >= MAX_SEQ_LEN:
        raise ValueError(f"position id {max_position} exceeds MAX_SEQ_LEN={MAX_SEQ_LEN}")
    return hca_case, q_lens_values, context_lens_values, active_tokens


def build_tensor_specs(
    start_pos: int = START_POS,
    num_tokens: int = MAX_TOKENS,
    hca_case: str = "custom",
    hetero_smoke: bool = False,
    hetero_boundary: bool = False,
    hetero_mixed_cmp_pos: bool = False,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from rope_tables import build_deepseek_v4_rope_tables

    shared_freqs_cos, shared_freqs_sin = build_deepseek_v4_rope_tables(M, COMPRESS_RATIO, dtype=torch.bfloat16)

    _, q_lens_values, context_lens_values, num_tokens = _resolve_hca_case(
        start_pos,
        num_tokens,
        hca_case,
        hetero_smoke,
        hetero_boundary,
        hetero_mixed_cmp_pos,
    )

    def token_meta():
        token_to_req = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        local_pos = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        pos = torch.arange(MAX_TOKENS, dtype=torch.int32)
        cursor = 0
        for req, q_len in enumerate(q_lens_values):
            ctx = context_lens_values[req]
            for local_s in range(q_len):
                t = cursor + local_s
                token_to_req[t] = req
                local_pos[t] = local_s
                pos[t] = ctx + local_s
            cursor += q_len
        return token_to_req, local_pos, pos

    def validate_overlay_topk(topk_idxs, token_to_req, pos, sparse_lens=None):
        current_by_req = [dict() for _ in range(MAX_REQS)]
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            current_by_req[req][int(pos[t].item())] = t

        for t in range(num_tokens):
            req = int(token_to_req[t].item())
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
                    if key_abs in current_by_req[req]:
                        raise ValueError(f"current suffix abs_pos={key_abs} must use HCA overlay for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate window abs_pos={key_abs} for HCA token {t}")
                    seen_window_abs.add(key_abs)
                elif raw < WIN + MAX_TOKENS:
                    overlay_t = raw - WIN
                    if overlay_t >= num_tokens:
                        raise ValueError(f"HCA overlay raw={raw} points past active tokens for token {t}")
                    overlay_req = int(token_to_req[overlay_t].item())
                    if overlay_req != req:
                        raise ValueError(f"HCA overlay raw={raw} crosses request {overlay_req}->{req}")
                    key_abs = int(pos[overlay_t].item())
                    if key_abs > abs_pos:
                        raise ValueError(f"HCA overlay raw={raw} is future key abs_pos={key_abs} for token {t}")
                    if key_abs in seen_window_abs:
                        raise ValueError(f"duplicate overlay abs_pos={key_abs} for HCA token {t}")
                    seen_window_abs.add(key_abs)
                else:
                    cmp_slot = raw - (WIN + MAX_TOKENS)
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
        cursor = 0
        for req, q_len in enumerate(q_lens_values):
            ctx = context_lens_values[req]
            for local_s in range(q_len):
                abs_len = ctx + local_s + 1
                if abs_len >= COMPRESS_RATIO and abs_len % COMPRESS_RATIO == 0:
                    token_id = cursor + local_s
                    cmp_slot = abs_len // COMPRESS_RATIO - 1
                    records.append((req, token_id, cmp_slot))
            cursor += q_len
        return records

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale

    def init_x_hc():
        x = seeded_uniform((MAX_TOKENS, HC_MULT, D), 1, 0.1)
        x[num_tokens:] = 0
        return x
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
        return shared_freqs_cos.clone()
    def init_freqs_sin():
        return shared_freqs_sin.clone()
    def init_cmp_wkv():
        return seeded_uniform((D, MAIN_OUT_DIM), 6, D ** -0.5)
    def init_cmp_wgate():
        return seeded_uniform((D, MAIN_OUT_DIM), 7, D ** -0.5)
    def init_cmp_ape():
        return seeded_uniform((COMPRESS_RATIO, MAIN_OUT_DIM), 8, 0.1)
    def init_cmp_norm_w():
        return torch.ones(HEAD_DIM)
    def init_compress_state_block_table():
        table = torch.full((MAX_REQS, HCA_STATE_MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            for block in range(HCA_STATE_MAX_BLOCKS):
                table[req, block] = req * HCA_STATE_MAX_BLOCKS + ((block * 17 + 3) % HCA_STATE_MAX_BLOCKS)
        return table
    def state_row(req, abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        table = init_compress_state_block_table()
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(table[req, block].item()) * HCA_STATE_BLOCK_SIZE + intra
    def init_cmp_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            for abs_pos in range(max(0, ctx - COMPRESS_RATIO), ctx):
                row = state_row(req, abs_pos)
                if row >= 0:
                    flat[row] = seeded_uniform((MAIN_OUT_DIM,), 12000 + req * 65536 + abs_pos, 0.05)
        return state
    def init_cmp_score_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM)
        flat = state.view(-1, MAIN_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            for abs_pos in range(max(0, ctx - COMPRESS_RATIO), ctx):
                row = state_row(req, abs_pos)
                if row >= 0:
                    flat[row] = seeded_uniform((MAIN_OUT_DIM,), 13000 + req * 65536 + abs_pos, 0.05)
        return state
    def init_kv_cache():
        cache = torch.zeros(HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        for req, ctx in enumerate(context_lens_values):
            if ctx > 0:
                prefix_start = max(0, ctx - WIN)
                prefix = seeded_uniform((ctx, HEAD_DIM), 11 + req, 0.1).to(torch.bfloat16)
                for pos_i in range(prefix_start, ctx):
                    cache_flat[req * BLOCK_SIZE + pos_i % WIN] = prefix[pos_i]
        return cache
    def init_ori_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        token_to_req, local_pos, _ = token_meta()
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            logical_pos = context_lens_values[req] + int(local_pos[t].item())
            mapping[t] = req * BLOCK_SIZE + logical_pos % WIN
        return mapping
    def init_ori_block_table():
        table = torch.full((MAX_REQS, SPARSE_ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(MAX_REQS):
            table[b, 0] = b
        return table
    def init_cmp_kv():
        cache = torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        for req, ctx in enumerate(context_lens_values):
            completed = ctx // COMPRESS_RATIO
            if completed > 0:
                prefix_cmp = seeded_uniform((completed, HEAD_DIM), 14 + req, 0.1).to(torch.bfloat16)
                for cmp_slot in range(completed):
                    cache_flat[req * BLOCK_SIZE + cmp_slot] = prefix_cmp[cmp_slot]
        return cache
    def init_cmp_block_table():
        table = torch.full((MAX_REQS, SPARSE_CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(MAX_REQS):
            table[b, 0] = b
        return table
    def init_cmp_sparse_indices():
        topk_idxs = torch.full((MAX_TOKENS, SPARSE_TOPK), -1, dtype=torch.int32)
        token_to_req, local_pos, pos = token_meta()
        current_by_req = [dict() for _ in range(MAX_REQS)]
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            current_by_req[req][int(pos[t].item())] = t
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            position = context_lens_values[req] + int(local_pos[t].item())
            window_start = max(0, position - WIN + 1)
            cursor = 0
            for visible_pos in range(window_start, position + 1):
                overlay_t = current_by_req[req].get(visible_pos)
                if overlay_t is not None and overlay_t <= t:
                    topk_idxs[t, cursor] = WIN + overlay_t
                else:
                    topk_idxs[t, cursor] = visible_pos % WIN
                cursor += 1
            visible_cmp = (position + 1) // COMPRESS_RATIO
            for cmp_slot in range(visible_cmp):
                if cursor >= SPARSE_TOPK:
                    break
                topk_idxs[t, cursor] = WIN + MAX_TOKENS + cmp_slot
                cursor += 1
        sparse_lens = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        for t in range(num_tokens):
            valid = (topk_idxs[t] >= 0).nonzero()
            if valid.numel():
                sparse_lens[t] = int(valid[-1].item()) + 1
        validate_overlay_topk(topk_idxs, token_to_req, pos, sparse_lens)
        return topk_idxs
    def init_cmp_sparse_lens():
        topk_idxs = init_cmp_sparse_indices()
        lens = torch.zeros(MAX_TOKENS, dtype=torch.int32)
        for t in range(num_tokens):
            valid = (topk_idxs[t] >= 0).nonzero()
            if valid.numel():
                lens[t] = int(valid[-1].item()) + 1
                if hca_case == "cmp_sparse_lens_boundary":
                    lens[t] = max(1, int(lens[t].item()) - 8)
        return lens
    def init_token_to_request():
        return token_meta()[0]
    def init_position_ids():
        return token_meta()[2]
    def init_cmp_slot_mapping():
        out = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        for req, token_id, cmp_slot in cmp_write_records():
            out[token_id] = req * BLOCK_SIZE + cmp_slot
        return out
    def init_state_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int64)
        token_to_req, _, pos = token_meta()
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            mapping[t] = state_row(req, int(pos[t].item()))
        return mapping
    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 9, O_GROUP_IN ** -0.5)
    def init_wo_b():
        return seeded_uniform((D, O_GROUPS * O_LORA), 10, (O_GROUPS * O_LORA) ** -0.5)

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [MAX_TOKENS, HC_MULT, D], torch.bfloat16, init_value=init_x_hc),
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
        TensorSpec("cmp_wkv", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [D, MAIN_OUT_DIM], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.float32, init_value=init_cmp_norm_w),
        TensorSpec(
            "cmp_kv_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_state,
            is_output=True,
        ),
        TensorSpec(
            "cmp_score_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_score_state,
            is_output=True,
        ),
        TensorSpec("compress_state_block_table", [MAX_REQS, HCA_STATE_MAX_BLOCKS], torch.int32, init_value=init_compress_state_block_table),
        TensorSpec(
            "kv_cache",
            [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
            is_output=True,
        ),
        TensorSpec("ori_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("ori_block_table", [MAX_REQS, SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec(
            "cmp_kv",
            [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
            is_output=True,
        ),
        TensorSpec("cmp_block_table", [MAX_REQS, SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [MAX_TOKENS, SPARSE_TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("cmp_sparse_lens", [MAX_TOKENS], torch.int32, init_value=init_cmp_sparse_lens),
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [MAX_TOKENS], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [MAX_TOKENS, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


def active_x_out_compare(num_tokens: int):
    from golden import ratio_allclose

    base_cmp = ratio_allclose(atol=3e-3, rtol=2.0 / 128)

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

    cmp.__name__ = f"active_x_out_compare(num_tokens={num_tokens})"
    return cmp


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser(
        description=(
            "Standalone DeepSeek V4 packed prefill HCA correctness test. "
            "CLI shape/scenario options only build fixture/golden tensors and lowered metadata; "
            "the JIT kernel itself consumes num_tokens/token_to_request/position_ids/slot mappings/topk/write records."
        )
    )
    parser.add_argument(
        "-p", "--platform",
        type=str,
        default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
        help="PyPTO compile/runtime backend for this standalone validation. Default: %(default)s.",
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        default=0,
        help="NPU device id passed to runtime_cfg.device_id. Under task-submit, '{}' is usually substituted here.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        default=False,
        help="Compile/codegen only. This is also the implicit behavior on *sim platforms used by CI.",
    )
    parser.add_argument(
        "--start-pos",
        type=int,
        default=START_POS,
        help=(
            "Fixture-only context length for request 0 when --hca-case=custom. "
            "It is lowered into position_ids, ori_slot_mapping, cmp_sparse_indices, and compressor state; "
            "it is not a JIT kernel argument."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=MAX_TOKENS,
        help=(
            "Fixture active token count for --hca-case=custom, capped by MAX_TOKENS. "
            "The value is passed to the kernel as num_tokens and also controls x_out active-token comparison."
        ),
    )
    parser.add_argument(
        "--hca-case",
        type=str,
        default="custom",
        choices=HCA_CASES,
        help=(
            "Named fixture scenario. custom uses --start-pos/--num-tokens; other cases cover short prefill, "
            "suffix prefill, MAX_REQS=2 hetero requests, and mixed compressed write positions."
        ),
    )
    parser.add_argument(
        "--hetero-smoke",
        action="store_true",
        default=False,
        help="Alias for --hca-case hetero_smoke; validates two requests with different context/q lengths.",
    )
    parser.add_argument(
        "--hetero-boundary",
        action="store_true",
        default=False,
        help="Alias for --hca-case hetero_boundary; both requests close the ratio128 boundary at pos127.",
    )
    parser.add_argument(
        "--hetero-mixed-cmp-pos",
        action="store_true",
        default=False,
        help="Alias for --hca-case hetero_mixed_cmp_pos; req0 writes pos127 and req1 writes pos255.",
    )
    parser.add_argument(
        "--enable-l2-swimlane",
        action="store_true",
        default=False,
        help="Enable L2 swimlane profiling/report generation in runtime_cfg for this validation run.",
    )
    args = parser.parse_args()
    try:
        _, _, _, compare_tokens = _resolve_hca_case(
            args.start_pos,
            args.num_tokens,
            args.hca_case,
            args.hetero_smoke,
            args.hetero_boundary,
            args.hetero_mixed_cmp_pos,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    result = run_jit(
        fn=prefill_attention_hca,
        specs=build_tensor_specs(
            args.start_pos,
            args.num_tokens,
            args.hca_case,
            args.hetero_smoke,
            args.hetero_boundary,
            args.hetero_mixed_cmp_pos,
        ),
        golden_fn=golden_prefill_attention_hca,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compile_only=args.compile_only or args.platform.endswith("sim"),
        compare_fn={
            "x_out": active_x_out_compare(compare_tokens),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
