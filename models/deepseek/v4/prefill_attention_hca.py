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
slot mappings, sparse indices, and compressed write records.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ
from hc_post import golden_hc_post, hc_post
from prefill_hc_pre import golden_prefill_hc_pre, prefill_hc_pre_packed
from prefill_compressor_ratio128 import prefill_compressor_ratio128_packed
from prefill_qkv_proj_rope import prefill_packed_qkv_proj_rope_core
from prefill_rmsnorm import golden_prefill_attn_norm, prefill_packed_attn_norm
from prefill_sparse_attn import (
    CMP_BLOCK_NUM as SPARSE_CMP_BLOCK_NUM,
    CMP_MAX_BLOCKS as SPARSE_CMP_MAX_BLOCKS,
    ORI_BLOCK_NUM as SPARSE_ORI_BLOCK_NUM,
    ORI_MAX_BLOCKS as SPARSE_ORI_MAX_BLOCKS,
    PREFILL_ATTN_TILE as SPARSE_PREFILL_ATTN_TILE,
    SOFTMAX_SCALE,
    TOPK as SPARSE_TOPK,
    _quant_w_per_channel,
    prefill_hca_packed_sparse_attn,
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
MAX_CMP_WRITES = MAX_REQS * max(1, MAX_TOKENS // COMPRESS_RATIO)
HCA_ORI_BLOCK_NUM = MAX_REQS * SPARSE_ORI_MAX_BLOCKS
HCA_CMP_BLOCK_NUM = MAX_REQS * SPARSE_CMP_MAX_BLOCKS
HCA_CASES = (
    "custom",
    "basic128",
    "basic96",
    "basic17",
    "suffix96_17",
    "suffix96_32",
    "suffix128_17",
    "hetero_smoke",
    "hetero_boundary",
    "hetero_mixed_cmp_pos",
)

HCA_KV_STORE_TILE = 16

assert S == COMPRESS_RATIO, "first prefill HCA bring-up targets one ratio-128 prompt chunk"
assert WIN == BLOCK_SIZE, "prefill HCA currently assumes one window page per batch"
assert SPARSE_ORI_BLOCK_NUM == B * SPARSE_ORI_MAX_BLOCKS
assert SPARSE_CMP_BLOCK_NUM == B * SPARSE_CMP_MAX_BLOCKS
assert PREFILL_COMPRESSED_LEN == 1


@pl.jit.inline
def _prefill_hca_write_prompt_kv(
    kv: pl.Tensor[[MAX_TOKENS, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    ori_kv_flat = pl.reshape(ori_kv, [HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for t0 in pl.parallel(0, MAX_TOKENS, HCA_KV_STORE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_write_prompt_kv"):
            for dt in pl.range(HCA_KV_STORE_TILE):
                t = t0 + dt
                if t < num_tokens:
                    kv_store_row = pl.cast(pl.read(ori_slot_mapping, [t]), pl.INDEX)
                    ori_kv_flat[kv_store_row : kv_store_row + 1, 0:HEAD_DIM] = kv[t : t + 1, 0:HEAD_DIM]
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
    cmp_kv_state: pl.Tensor[[MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    cmp_score_state: pl.Tensor[[MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM], pl.FP32],
    kv_cache: pl.Tensor[[HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT32],
    ori_block_table: pl.Tensor[[MAX_REQS, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Out[pl.Tensor[[HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[MAX_REQS, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[MAX_TOKENS, SPARSE_TOPK], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_cmp_writes: pl.Scalar[pl.INT32],
    cmp_write_token_ids: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[MAX_CMP_WRITES], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[MAX_TOKENS, HC_MULT, D], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT, HC_MULT], dtype=pl.FP32)
    x_mixed, post, comb = prefill_hc_pre_packed(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed = prefill_packed_attn_norm(x_mixed, attn_norm_w, x_normed)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    q, kv, qr, qr_scale = prefill_packed_qkv_proj_rope_core(
        x_normed,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        freqs_cos,
        freqs_sin,
        gamma_cq,
        gamma_ckv,
        q,
        kv,
        qr,
        qr_scale,
        rope_cos_t,
        rope_sin_t,
        position_ids,
        num_tokens,
    )

    kv_cache = _prefill_hca_write_prompt_kv(kv, kv_cache, ori_slot_mapping, num_tokens)
    cmp_kv, cmp_kv_state, cmp_score_state = prefill_compressor_ratio128_packed(
        x_normed,
        cmp_kv_state,
        cmp_score_state,
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
        num_cmp_writes,
        cmp_write_token_ids,
        cmp_slot_mapping,
    )

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_hca_packed_sparse_attn(
        q,
        kv_cache,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
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

    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    comb_t = pl.reshape(comb, [T, HC_MULT * HC_MULT])
    x_out = hc_post(
        attn_out,
        x_hc,
        post,
        comb_t,
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


def _int8_quant_per_row(x):
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = torch.round(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _golden_hca_packed_sparse_attn(tensors, q, ori_kv, cmp_kv, rope_cos_t, rope_sin_t, attn_out):
    import torch

    num_tokens = int(tensors["num_tokens"])
    q_f32 = q.float()
    ori_kv_f32 = ori_kv.float()
    cmp_kv_f32 = cmp_kv.float()
    ori_block_table = tensors["ori_block_table"]
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    token_to_request = tensors["token_to_request"]
    attn_sink = tensors["attn_sink"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)
    for t in range(num_tokens):
        req = int(token_to_request[t].item())
        gathered = []
        for raw_i in cmp_sparse_indices[t, :SPARSE_TOPK].tolist():
            raw = int(raw_i)
            if raw < 0:
                continue
            if raw < S:
                blk_id = int(ori_block_table[req, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv_f32[blk_id, intra, 0])
            else:
                cmp_slot = raw - S
                if cmp_slot >= HCA_CMP_BLOCK_NUM * BLOCK_SIZE:
                    continue
                blk_id = int(cmp_block_table[req, cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv_f32[blk_id, intra, 0])
        if not gathered:
            continue
        kv_b = torch.stack(gathered, dim=0)

        mi = None
        li = None
        oi = None
        for tile_start in range(0, kv_b.shape[0], SPARSE_PREFILL_ATTN_TILE):
            kv_tile = kv_b[tile_start : tile_start + SPARSE_PREFILL_ATTN_TILE]
            scores = (q_f32[t] @ kv_tile.T) * SOFTMAX_SCALE
            cur_mi = scores.max(dim=-1, keepdim=True).values
            exp_scores_bf16 = torch.exp(scores - cur_mi).to(torch.bfloat16)
            cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)
            cur_oi = exp_scores_bf16.float() @ kv_tile.to(torch.bfloat16).float()
            if mi is None:
                mi = cur_mi
                li = cur_li
                oi = cur_oi
            else:
                mi_new = torch.maximum(mi, cur_mi)
                alpha = torch.exp(mi - mi_new)
                beta = torch.exp(cur_mi - mi_new)
                li = alpha * li + beta * cur_li
                oi = oi * alpha + cur_oi * beta
                mi = mi_new

        if mi is not None:
            denom = li + torch.exp(attn_sink.unsqueeze(-1) - mi)
            o[t] = oi / denom

    rope_pair = o[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pair[..., 0]
    rope_odd = rope_pair[..., 1]
    cos_half = rope_cos_t.float()[:, :ROPE_HALF].unsqueeze(1)
    sin_half = rope_sin_t.float()[:, :ROPE_HALF].unsqueeze(1)
    inv_even = (rope_even * cos_half + rope_odd * sin_half).to(torch.bfloat16).float()
    inv_odd = (rope_odd * cos_half - rope_even * sin_half).to(torch.bfloat16).float()
    o_rope = torch.stack([inv_even, inv_odd], dim=-1).flatten(-2)
    o = torch.cat([o[..., :NOPE_DIM], o_rope], dim=-1).to(torch.bfloat16)

    o_model = o.float().view(T, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("tgd,grd->tgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(1).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)
    attn_out[:] = out.to(torch.bfloat16)


def _golden_hca_packed_qkv_proj_rope(tensors, x_normed, q, kv, qr, qr_scale):
    import torch

    x = x_normed.float()
    wq_a = tensors["wq_a"].float()
    wq_b = tensors["wq_b"]
    wq_b_scale = tensors["wq_b_scale"].float().view(-1)
    wkv = tensors["wkv"].float()
    freqs_cos = tensors["freqs_cos"]
    freqs_sin = tensors["freqs_sin"]
    gamma_cq = tensors["gamma_cq"].float()
    gamma_ckv = tensors["gamma_ckv"].float()
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_flat = freqs_cos.index_select(0, positions).contiguous()
    rope_sin_flat = freqs_sin.index_select(0, positions).contiguous()

    def int8_quant_per_row(v):
        rows = v.reshape(-1, v.shape[-1]).float()
        amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
        scale_quant = INT8_SCALE_MAX / amax
        scaled = rows * scale_quant
        out_i32 = torch.round(scaled).to(torch.int32)
        out_half = out_i32.to(torch.float16)
        out_i8 = out_half.to(torch.int8)
        return out_i8.reshape_as(v), (1.0 / scale_quant).reshape(*v.shape[:-1], 1)

    def rms_norm(v, gamma, eps=EPS):
        inv = torch.rsqrt(v.square().mean(-1, keepdim=True) + eps)
        return v * inv * gamma

    def matmul_bf16_input_fp32(a, b):
        return torch.matmul(a.to(torch.bfloat16).float(), b.to(torch.bfloat16).float()).float()

    def apply_rope(x_rope, cos, sin):
        x_pair = x_rope.unflatten(-1, (-1, 2))
        x_even, x_odd = x_pair[..., 0], x_pair[..., 1]
        cos_v = cos[..., :ROPE_HALF]
        sin_v = sin[..., :ROPE_HALF]
        while cos_v.ndim < x_even.ndim:
            cos_v = cos_v.unsqueeze(-2)
            sin_v = sin_v.unsqueeze(-2)
        y_even = (x_even * cos_v - x_odd * sin_v).to(torch.bfloat16)
        y_odd = (x_even * sin_v + x_odd * cos_v).to(torch.bfloat16)
        return torch.stack([y_even, y_odd], dim=-1).flatten(-2)

    token_x = x.reshape(T, D)
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)
    qr_i8, qr_scale_out = int8_quant_per_row(qr_out.to(torch.bfloat16).float())
    q_i32 = torch.matmul(qr_i8.to(torch.int32), wq_b.to(torch.int32))
    q_full = (q_i32.float() * qr_scale_out * wq_b_scale.view(1, -1)).view(T, H, HEAD_DIM)
    q_full = q_full * torch.rsqrt(q_full.square().mean(-1, keepdim=True) + EPS)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = apply_rope(q_full[..., NOPE_DIM:], rope_cos_flat, rope_sin_flat)
    q_out = torch.cat([q_nope, q_rope], dim=-1)

    kv_full = rms_norm(matmul_bf16_input_fp32(token_x, wkv), gamma_ckv)
    kv_nope = kv_full[..., :NOPE_DIM]
    kv_rope_in = kv_full[..., NOPE_DIM:].unsqueeze(1)
    kv_rope = apply_rope(kv_rope_in, rope_cos_flat, rope_sin_flat).squeeze(1)
    kv_out = torch.cat([kv_nope, kv_rope], dim=-1)

    q[:] = q_out.to(torch.bfloat16)
    kv[:] = kv_out.to(torch.bfloat16)
    qr[:] = qr_i8
    qr_scale[:] = qr_scale_out


def golden_prefill_attention_hca(tensors):
    import torch

    num_tokens = int(tensors["num_tokens"])
    x_hc_rect = tensors["x_hc"].view(B, S, HC_MULT, D)
    x_mixed = torch.zeros(B, S, D, dtype=torch.bfloat16)
    post = torch.zeros(B, S, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(B, S, HC_MULT, HC_MULT, dtype=torch.float32)
    golden_prefill_hc_pre({
        "x": x_hc_rect,
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
    x_normed = golden_prefill_attn_norm(x_mixed, tensors["attn_norm_w"])
    _golden_hca_packed_qkv_proj_rope(tensors, x_normed, q, kv, qr, qr_scale)

    ori_kv = tensors["kv_cache"]
    ori_kv_flat = ori_kv.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            ori_kv_flat[dst_row, :] = kv[t]

    cmp_kv = tensors["cmp_kv"]
    num_cmp_writes = int(tensors["num_cmp_writes"])
    kv_proj = x_normed.float() @ tensors["cmp_wkv"].float()
    score_proj = x_normed.float() @ tensors["cmp_wgate"].float()
    kv_state = tensors["cmp_kv_state"]
    score_state = tensors["cmp_score_state"]
    for t in range(num_tokens):
        req = int(tensors["token_to_request"][t].item())
        state_slot = int(tensors["position_ids"][t].item()) % COMPRESS_RATIO
        kv_state[req, state_slot, :] = kv_proj.view(MAX_TOKENS, MAIN_OUT_DIM)[t]
        score_state[req, state_slot, :] = (
            score_proj.view(MAX_TOKENS, MAIN_OUT_DIM)[t] + tensors["cmp_ape"][state_slot]
        )
    cmp_kv_flat = cmp_kv.view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for write_i in range(num_cmp_writes):
        token_id = int(tensors["cmp_write_token_ids"][write_i].item())
        req = int(tensors["token_to_request"][token_id].item())
        dst_row = int(tensors["cmp_slot_mapping"][write_i].item())
        pooled = (kv_state[req] * score_state[req].softmax(dim=0)).sum(dim=0, keepdim=True)
        pooled = pooled.to(torch.bfloat16).float()
        inv = torch.rsqrt(pooled.square().mean(dim=-1, keepdim=True) + EPS)
        normed = (pooled * inv * tensors["cmp_norm_w"].float().view(1, HEAD_DIM)).to(torch.bfloat16)
        rope_pair = normed[..., NOPE_DIM:].unflatten(-1, (-1, 2))
        rope_even = rope_pair[..., 0]
        rope_odd = rope_pair[..., 1]
        cmp_pos = int(tensors["position_ids"][token_id].item()) + 1 - COMPRESS_RATIO
        cos = tensors["freqs_cos"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        sin = tensors["freqs_sin"][cmp_pos : cmp_pos + 1, 0:ROPE_HALF].float()
        rot_even = (rope_even.float() * cos - rope_odd.float() * sin).to(torch.bfloat16)
        rot_odd = (rope_even.float() * sin + rope_odd.float() * cos).to(torch.bfloat16)
        rope_full = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        normed[:, NOPE_DIM:] = rope_full
        cmp_kv_flat[dst_row : dst_row + 1, :] = normed

    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    _golden_hca_packed_sparse_attn(tensors, q, ori_kv, cmp_kv, rope_cos_t, rope_sin_t, attn_out)

    y = torch.zeros(B, S, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out.view(T, D),
        "residual": x_hc_rect.view(T, HC_MULT, D),
        "post": post.view(T, HC_MULT),
        "comb": comb.view(T, HC_MULT * HC_MULT),
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
    elif hca_case == "basic128":
        q_lens_values = [128, 0]
        context_lens_values = [0, 0]
    elif hca_case == "basic96":
        q_lens_values = [96, 0]
        context_lens_values = [0, 0]
    elif hca_case == "basic17":
        q_lens_values = [17, 0]
        context_lens_values = [0, 0]
    elif hca_case == "suffix96_17":
        q_lens_values = [17, 0]
        context_lens_values = [96, 0]
    elif hca_case == "suffix96_32":
        q_lens_values = [32, 0]
        context_lens_values = [96, 0]
    elif hca_case == "suffix128_17":
        q_lens_values = [17, 0]
        context_lens_values = [128, 0]
    elif hca_case == "hetero_smoke":
        q_lens_values = [32, 64]
        context_lens_values = [64, 0]
    elif hca_case == "hetero_boundary":
        q_lens_values = [32, 32]
        context_lens_values = [96, 96]
    elif hca_case == "hetero_mixed_cmp_pos":
        q_lens_values = [32, 32]
        context_lens_values = [96, 224]
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
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_DIM).reshape(MAX_SEQ_LEN, ROPE_DIM) * 1e-3)
    def init_cmp_wkv():
        return seeded_uniform((D, MAIN_OUT_DIM), 6, D ** -0.5)
    def init_cmp_wgate():
        return seeded_uniform((D, MAIN_OUT_DIM), 7, D ** -0.5)
    def init_cmp_ape():
        return seeded_uniform((COMPRESS_RATIO, MAIN_OUT_DIM), 8, 0.1)
    def init_cmp_norm_w():
        return torch.ones(HEAD_DIM)
    def init_cmp_state():
        state = torch.zeros(MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            partial_len = ctx % COMPRESS_RATIO
            if partial_len > 0:
                state[req, :partial_len, :] = seeded_uniform((partial_len, MAIN_OUT_DIM), 12 + req, 0.1)
        return state
    def init_cmp_score_state():
        state = torch.zeros(MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM)
        for req, ctx in enumerate(context_lens_values):
            partial_len = ctx % COMPRESS_RATIO
            if partial_len > 0:
                state[req, :partial_len, :] = seeded_uniform((partial_len, MAIN_OUT_DIM), 13 + req, 0.1)
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
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int32)
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
        token_to_req, local_pos, _ = token_meta()
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            position = context_lens_values[req] + int(local_pos[t].item())
            window_start = max(0, position - WIN + 1)
            cursor = 0
            for visible_pos in range(window_start, position + 1):
                topk_idxs[t, cursor] = visible_pos % WIN
                cursor += 1
            visible_cmp = (position + 1) // COMPRESS_RATIO
            for cmp_slot in range(visible_cmp):
                if cursor >= SPARSE_TOPK:
                    break
                topk_idxs[t, cursor] = S + cmp_slot
                cursor += 1
        return topk_idxs
    def init_token_to_request():
        return token_meta()[0]
    def init_position_ids():
        return token_meta()[2]
    def init_cmp_write_token_ids():
        out = torch.full((MAX_CMP_WRITES,), -1, dtype=torch.int32)
        for i, (_, token_id, _) in enumerate(cmp_write_records()):
            out[i] = token_id
        return out
    def init_cmp_slot_mapping():
        out = torch.full((MAX_CMP_WRITES,), -1, dtype=torch.int32)
        for i, (req, _, cmp_slot) in enumerate(cmp_write_records()):
            out[i] = req * BLOCK_SIZE + cmp_slot
        return out
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
            [MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_state,
            is_output=True,
        ),
        TensorSpec(
            "cmp_score_state",
            [MAX_REQS, MAIN_STATE_LEN, MAIN_OUT_DIM],
            torch.float32,
            init_value=init_cmp_score_state,
            is_output=True,
        ),
        TensorSpec(
            "kv_cache",
            [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
            is_output=True,
        ),
        TensorSpec("ori_slot_mapping", [MAX_TOKENS], torch.int32, init_value=init_ori_slot_mapping),
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
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        ScalarSpec("num_cmp_writes", torch.int32, len(cmp_write_records())),
        TensorSpec("cmp_write_token_ids", [MAX_CMP_WRITES], torch.int32, init_value=init_cmp_write_token_ids),
        TensorSpec("cmp_slot_mapping", [MAX_CMP_WRITES], torch.int32, init_value=init_cmp_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [MAX_TOKENS, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


def packed_x_out_compare(num_tokens: int):
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

    cmp.__name__ = f"packed_x_out_compare(num_tokens={num_tokens})"
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
        compare_fn={
            "x_out": packed_x_out_compare(compare_tokens),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
