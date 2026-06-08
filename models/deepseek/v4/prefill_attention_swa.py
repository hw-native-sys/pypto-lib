# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 packed prefill SWA attention.

The public contract is token-major packed prefill with static capacity and
runtime active sizes. SWA consumes lowered metadata such as token_to_request,
position_ids, slot mappings, and window-ring sparse indices.
"""

import pypto.language as pl

from config import BLOCK_SIZE, FLASH as M, INT8_AMAX_EPS, INT8_SCALE_MAX, PREFILL_BATCH, PREFILL_SEQ
from hc_post import golden_hc_post, hc_post
from prefill_hc_pre import golden_prefill_hc_pre, prefill_hc_pre_packed
from prefill_qkv_proj_rope import prefill_packed_qkv_proj_rope_core
from prefill_rmsnorm import prefill_packed_attn_norm
from prefill_sparse_attn import (
    CMP_MAX_BLOCKS as SPARSE_CMP_MAX_BLOCKS,
    ORI_BLOCK_NUM as SPARSE_ORI_BLOCK_NUM,
    ORI_MAX_BLOCKS as SPARSE_ORI_MAX_BLOCKS,
    PREFILL_ATTN_TILE as SPARSE_PREFILL_ATTN_TILE,
    TOPK as SPARSE_TOPK,
    _int8_quant_per_row,
    _quant_w_per_channel,
    prefill_hca_packed_sparse_attn,
)


# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
MAX_REQS = 2
MAX_TOKENS = T
EPS = M.rms_norm_eps
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
NOPE_DIM = M.nope_head_dim
NOPE_HEAD_DIM = NOPE_DIM
Q_LORA = M.q_lora_rank
ROPE_HALF = ROPE_DIM // 2
HALF_ROPE = ROPE_HALF
MAX_SEQ_LEN = M.max_position_embeddings
WIN = M.sliding_window
SOFTMAX_SCALE = M.softmax_scale
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
HC_DIM = M.hc_dim
HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = M.hc_sinkhorn_iters
HC_EPS = M.hc_eps
O_LORA = M.o_lora_rank
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

# SWA cache/topk contract. The ratio-0 path has only the sliding-window cache.
ORI_MAX_BLOCKS = 1
MAX_BLOCKS = ORI_MAX_BLOCKS
BLOCK_NUM = MAX_REQS * MAX_BLOCKS
START_POS = 0
SWA_CMP_BLOCK_NUM = MAX_REQS * SPARSE_CMP_MAX_BLOCKS

# HC tiling, mirrored from hc_pre/hc_post but using prefill B/S/T.
MIX_PAD = 32
NEG_INF = -1e20
T_TILE = 16
RMS_T_TILE = 16
LINEAR_T_TILE = 16
COMB_T_TILE = 16
RMS_K_CHUNK = 128
LINEAR_K_CHUNK = 512
D_CHUNK = 512
RMS_K_BLOCKS = HC_DIM // RMS_K_CHUNK
LINEAR_K_BLOCKS = HC_DIM // LINEAR_K_CHUNK
D_BLOCKS = D // D_CHUNK
RMS_PIPE_STAGE = 1 if T >= 64 else 4

KV_CACHE_WRITE_TILE = 16

assert WIN == BLOCK_SIZE, "SWA prefill currently assumes one window page per batch"
assert S <= WIN, "SWA prefill tile must not exceed the sliding-window ring size"
assert T % KV_CACHE_WRITE_TILE == 0, "KV cache write tile must divide packed token capacity"
assert SPARSE_ORI_BLOCK_NUM == B * SPARSE_ORI_MAX_BLOCKS
assert SPARSE_ORI_MAX_BLOCKS == ORI_MAX_BLOCKS


@pl.jit.inline
def prefill_swa_write_kv_cache_packed(
    kv: pl.Tensor[[MAX_TOKENS, HEAD_DIM], pl.BF16],
    kv_cache: pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    kv_cache_flat = pl.reshape(kv_cache, [BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    for t0 in pl.parallel(0, MAX_TOKENS, KV_CACHE_WRITE_TILE):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_kv_cache_store"):
            for dt in pl.range(KV_CACHE_WRITE_TILE):
                t = t0 + dt
                if t < num_tokens:
                    dst_row = pl.cast(pl.read(ori_slot_mapping, [t]), pl.INDEX)
                    kv_cache_flat[dst_row : dst_row + 1, 0:HEAD_DIM] = kv[t : t + 1, 0:HEAD_DIM]
    return pl.reshape(kv_cache_flat, [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM])


@pl.jit
def prefill_attention_swa(
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
    kv_cache: pl.Out[pl.Tensor[[BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[MAX_REQS, MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[MAX_TOKENS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[MAX_TOKENS, SPARSE_TOPK], pl.INT32],
    token_to_request: pl.Tensor[[MAX_TOKENS], pl.INT32],
    position_ids: pl.Tensor[[MAX_TOKENS], pl.INT32],
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
    # Full prefill path mirrors the official block: hc_pre -> qkv/rope -> SWA
    # attention/o_proj -> KV writeback -> hc_post.
    x_mixed, post, comb = prefill_hc_pre_packed(
        x_hc,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        x_mixed,
        post,
        comb,
    )

    # Reuse the shared prefill QKV/RoPE projection to stay aligned with decode.
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    x_normed = prefill_packed_attn_norm(x_mixed, attn_norm_w, x_normed)
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

    kv_cache = prefill_swa_write_kv_cache_packed(kv, kv_cache, ori_slot_mapping, num_tokens)
    cmp_kv = pl.create_tensor([SWA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], dtype=pl.BF16)
    cmp_block_table = pl.create_tensor([MAX_REQS, SPARSE_CMP_MAX_BLOCKS], dtype=pl.INT32)

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    attn_out = prefill_hca_packed_sparse_attn(
        q,
        kv_cache,
        block_table,
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
    return kv_cache, x_out


def _quant_w_per_output_channel(w):
    import torch

    amax = w.float().abs().amax(dim=0).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(1, -1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def _quant_w_per_row(w):
    import torch

    amax = w.float().abs().amax(dim=1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.view(-1, 1)
    w_i32 = torch.round(scaled).to(torch.int32)
    w_i32 = torch.clamp(w_i32, -int(INT8_SCALE_MAX), int(INT8_SCALE_MAX))
    w_i8 = w_i32.to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def _golden_swa_packed_qkv_proj_rope(tensors, x_mixed, q, kv, qr, qr_scale):
    import torch

    x = x_mixed.float()
    norm_w = tensors["attn_norm_w"].float()
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

    token_x = rms_norm(x.reshape(T, D), norm_w)
    qr_out = rms_norm(matmul_bf16_input_fp32(token_x, wq_a), gamma_cq)
    qr_i8, qr_scale_out = _int8_quant_per_row(qr_out.to(torch.bfloat16).float())
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


def _golden_swa_packed_sparse_attn(tensors, q, ori_kv, cmp_kv, rope_cos_t, rope_sin_t, attn_out):
    import torch

    num_tokens = int(tensors["num_tokens"])
    q_f32 = q.float()
    ori_kv_f32 = ori_kv.float()
    cmp_kv_f32 = cmp_kv.float()
    ori_block_table = tensors["block_table"]
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
                if cmp_slot >= SWA_CMP_BLOCK_NUM * BLOCK_SIZE:
                    continue
                gathered.append(cmp_kv_f32[0, cmp_slot % BLOCK_SIZE, 0])
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


def golden_prefill_attention_swa(tensors):
    """Torch reference for token-major packed SWA prefill."""
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
    _golden_swa_packed_qkv_proj_rope(tensors, x_mixed, q, kv, qr, qr_scale)

    kv_cache = tensors["kv_cache"]
    kv_cache_flat = kv_cache.view(BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]

    cmp_kv = torch.zeros(SWA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    _golden_swa_packed_sparse_attn(tensors, q, kv_cache, cmp_kv, rope_cos_t, rope_sin_t, attn_out)

    y = torch.zeros(T, HC_MULT, D, dtype=torch.bfloat16)
    golden_hc_post({
        "x": attn_out.view(T, D),
        "residual": x_hc_rect.view(T, HC_MULT, D),
        "post": post.view(T, HC_MULT),
        "comb": comb.view(T, HC_MULT, HC_MULT),
        "y": y,
    })
    tensors["x_out"][:] = y


def build_tensor_specs(
    start_pos: int = START_POS,
    num_tokens: int = MAX_TOKENS,
    hetero_smoke: bool = False,
    hetero_boundary: bool = False,
):
    import torch
    from golden import ScalarSpec, TensorSpec

    if hetero_smoke and hetero_boundary:
        raise ValueError("--hetero-smoke and --hetero-boundary are mutually exclusive")
    if hetero_boundary:
        q_lens_values = [32, 32]
        context_lens_values = [96, 96]
        num_tokens = sum(q_lens_values)
    elif hetero_smoke:
        q_lens_values = [32, 64]
        context_lens_values = [64, 0]
        num_tokens = sum(q_lens_values)
    else:
        q_lens_values = [num_tokens, 0]
        context_lens_values = [start_pos, 0]

    if num_tokens <= 0 or num_tokens > MAX_TOKENS:
        raise ValueError(f"num_tokens must be in [1, {MAX_TOKENS}], got {num_tokens}")
    max_position = max(ctx + q_len for ctx, q_len in zip(context_lens_values, q_lens_values))
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")
    if max_position > MAX_SEQ_LEN:
        raise ValueError(f"position_ids exceed MAX_SEQ_LEN={MAX_SEQ_LEN}: got {max_position}")

    def seeded_uniform(shape, seed, scale=1.0):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return (torch.rand(*shape, generator=generator) - 0.5) * scale

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
        return torch.cos(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_freqs_sin():
        return torch.sin(torch.arange(MAX_SEQ_LEN * ROPE_HEAD_DIM).reshape(MAX_SEQ_LEN, ROPE_HEAD_DIM) * 1e-3)
    def init_block_table():
        tbl = torch.full((MAX_REQS, MAX_BLOCKS), -1, dtype=torch.int32)
        for req in range(MAX_REQS):
            tbl[req, 0] = req
        return tbl
    def init_kv_cache():
        cache = torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        for req, ctx in enumerate(context_lens_values):
            start = max(0, ctx - WIN)
            for abs_pos in range(start, ctx):
                row = req * BLOCK_SIZE + abs_pos % WIN
                value = seeded_uniform((HEAD_DIM,), 11 + req * 4096 + abs_pos, 0.1)
                cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_ori_slot_mapping():
        mapping = torch.full((MAX_TOKENS,), -1, dtype=torch.int32)
        token_to_req, _, pos = token_meta()
        for t in range(num_tokens):
            req = int(token_to_req[t].item())
            mapping[t] = req * BLOCK_SIZE + int(pos[t].item()) % WIN
        return mapping
    def init_cmp_sparse_indices():
        topk_idxs = torch.full((MAX_TOKENS, SPARSE_TOPK), -1, dtype=torch.int32)
        _, _, pos = token_meta()
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for key_i in range(window_valid):
                topk_idxs[t, key_i] = (key_start_abs + key_i) % WIN
        return topk_idxs
    def init_token_to_request():
        return token_meta()[0]
    def init_position_ids():
        return token_meta()[2]
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
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16,
                   init_value=init_kv_cache, is_output=True),
        TensorSpec("block_table", [MAX_REQS, MAX_BLOCKS], torch.int32, init_value=init_block_table),
        TensorSpec("ori_slot_mapping", [MAX_TOKENS], torch.int32, init_value=init_ori_slot_mapping),
        TensorSpec("cmp_sparse_indices", [MAX_TOKENS, SPARSE_TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("token_to_request", [MAX_TOKENS], torch.int32, init_value=init_token_to_request),
        TensorSpec("position_ids", [MAX_TOKENS], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [MAX_TOKENS, HC_MULT, D], torch.bfloat16, is_output=True),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


def packed_x_out_compare(num_tokens: int):
    from golden import ratio_allclose

    base_cmp = ratio_allclose(atol=6e-3, rtol=2.0 / 128)

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
    from golden import ratio_allclose, run_jit

    parser = argparse.ArgumentParser(
        description=(
            "Standalone DeepSeek V4 packed prefill SWA correctness test. "
            "SWA is pure sliding-window attention; CLI scenario options generate fixture/golden tensors "
            "and lowered token metadata, not extra JIT kernel parameters."
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
            "Fixture-only context length for request 0. It is lowered into position_ids, "
            "ori_slot_mapping, and window-ring cmp_sparse_indices; it is not a JIT argument."
        ),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=MAX_TOKENS,
        help=(
            "Fixture active token count, capped by MAX_TOKENS. The value is passed to the kernel as "
            "num_tokens and controls x_out active-token comparison."
        ),
    )
    parser.add_argument(
        "--hetero-smoke",
        action="store_true",
        default=False,
        help="Fixture alias for a MAX_REQS=2 smoke case with different context/q lengths.",
    )
    parser.add_argument(
        "--hetero-boundary",
        action="store_true",
        default=False,
        help="Fixture alias for a MAX_REQS=2 boundary/wrap case with independent request window caches.",
    )
    parser.add_argument(
        "--enable-l2-swimlane",
        action="store_true",
        default=False,
        help="Enable L2 swimlane profiling/report generation in runtime_cfg for this validation run.",
    )
    parser.add_argument(
        "--block-dim",
        type=int,
        default=None,
        help="Optional compile/runtime block_dim override forwarded through runtime_cfg; leave unset for default.",
    )
    parser.add_argument(
        "--aicpu-thread-num",
        type=int,
        default=None,
        help="Optional AICPU scheduler thread count override forwarded through runtime_cfg; leave unset for default.",
    )
    args = parser.parse_args()
    if args.hetero_smoke and args.hetero_boundary:
        raise SystemExit("--hetero-smoke and --hetero-boundary are mutually exclusive")
    compare_tokens = 64 if args.hetero_boundary else (96 if args.hetero_smoke else args.num_tokens)

    result = run_jit(
        fn=prefill_attention_swa,
        specs=build_tensor_specs(args.start_pos, args.num_tokens, args.hetero_smoke, args.hetero_boundary),
        golden_fn=golden_prefill_attention_swa,
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
            block_dim=args.block_dim,
            aicpu_thread_num=args.aicpu_thread_num,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": packed_x_out_compare(compare_tokens),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
