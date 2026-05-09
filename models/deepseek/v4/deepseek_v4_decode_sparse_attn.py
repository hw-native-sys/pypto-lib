# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 sparse attention with grouped output projection (decode).

Corresponds to model.py Attention.forward decode branch lines 533-542:
    o = sparse_attn(q, kv_cache, attn_sink, topk_idxs, softmax_scale)
    apply_rotary_emb(o[..., -rope_dim:], freqs_cis, inverse=True)
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    x = self.wo_b(o.flatten(2))

Inputs:
- q                 : query tensor from MLA prolog (RoPE already applied)
- ori_kv / cmp_kv   : paged sliding-window and paged compressed KV pools
- cmp_sparse_indices: per-token absolute indices (window + compressed concat)
                      computed by orchestrator (window topk + indexer/HCA topk)
- attn_sink         : per-head sink term added inside softmax
- freqs_cos/sin     : split-half inverse-RoPE tables for the sparse-attn output
- wo_a              : grouped first-stage output-projection weights from model.py:537-541
- wo_b / wo_b_scale : grouped second-stage output-projection W8 per-channel weights

Standalone contract:
- `cmp_sparse_indices[t, :]` may contain `-1` pads.
- entries in `[0, WIN)` address the logical sliding-window ring slots.
- entries in `[WIN, WIN + cmp_valid)` address compressed cache slots, where
    `cmp_valid = max(seqused_kv[b] - min(WIN, seqused_kv[b]), 0)`.
- the grouped projection layout matches:
    `o.view(bsz, seqlen, self.n_local_groups, -1)` with
    `self.n_local_groups == O_GROUPS` and `-1 == O_GROUP_IN`.

The standalone harness exposes `--compress-ratio {0,4,128}` for testing.
"""


import pypto.language as pl


B = 16
T = B
H = 64
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
WIN = 128
MAX_SEQ_LEN = 4096
IDX_TOPK = 512
TOPK = WIN + IDX_TOPK
SUPPORTED_COMPRESS_RATIOS = (0, 4, 128)
DEFAULT_COMPRESS_RATIO = 128

BLOCK_SIZE = 128
ORI_MAX_BLOCKS = 1
ORI_BLOCK_NUM = B * ORI_MAX_BLOCKS
CMP_MAX_BLOCKS = 64
CMP_BLOCK_NUM = B * CMP_MAX_BLOCKS

SOFTMAX_SCALE = HEAD_DIM ** -0.5
HALF_ROPE = ROPE_DIM // 2
MATMUL_ROW_PAD = 16

D = 4096
O_LORA = 1024
O_GROUPS = 8
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

A_K_CHUNK = 128
A_N_CHUNK = 128
B_K_CHUNK = 128
B_N_CHUNK = 256

INT8_SCALE_MAX = 127.0
INT8_AMAX_EPS = 1e-4
QUANT_CHUNK = 256


def get_standalone_cmp_valid(compress_ratio: int) -> int:
    """Map demo compress-ratio modes to the valid compressed-cache tail length."""
    if compress_ratio == 0:
        return 0
    if compress_ratio == 4:
        return IDX_TOPK
    if compress_ratio == 128:
        return MAX_SEQ_LEN // compress_ratio
    raise ValueError(f"Unsupported compress_ratio={compress_ratio}; expected one of {SUPPORTED_COMPRESS_RATIOS}")


@pl.jit.inline
def deepseek_v4_decode_sparse_attn_with_proj(
    q:                  pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
    ori_kv:             pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    ori_block_table:    pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
    cmp_kv:             pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    cmp_block_table:    pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
    attn_sink:          pl.Tensor[[H],                                            pl.FP32],
    seqused_kv:         pl.Tensor[[B],                                            pl.INT32],
    freqs_cos:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    freqs_sin:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    wo_a:               pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],                 pl.BF16],
    wo_b:               pl.Tensor[[D, O_GROUPS * O_LORA],                         pl.INT8],
    wo_b_scale:         pl.Tensor[[D],                                            pl.FP32],
    attn_out:           pl.Tensor[[T, D],                                         pl.BF16],
):
    """Run sparse decode attention, inverse RoPE, and grouped output projection."""
    A_K_BLOCKS = O_GROUP_IN // A_K_CHUNK
    A_N_BLOCKS = O_LORA // A_N_CHUNK
    B_K_BLOCKS = (O_GROUPS * O_LORA) // B_K_CHUNK
    B_N_BLOCKS = D // B_N_CHUNK

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    ori_kv_flat = pl.reshape(ori_kv, [ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    ori_block_table_flat = pl.reshape(ori_block_table, [B * ORI_MAX_BLOCKS])
    cmp_kv_flat = pl.reshape(cmp_kv, [CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM])
    cmp_block_table_flat = pl.reshape(cmp_block_table, [B * CMP_MAX_BLOCKS])
    cmp_sparse_indices_flat = pl.reshape(cmp_sparse_indices, [T * TOPK])
    attn_stage = pl.create_tensor([T * H, HEAD_DIM], dtype=pl.BF16)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)

    for b in pl.range(B):
        seq_used = pl.read(seqused_kv, [b])
        window_valid = pl.min(WIN, seq_used)
        cmp_valid = seq_used - window_valid
        cmp_topk_valid = pl.min(IDX_TOPK, cmp_valid)
        sparse_k = window_valid + cmp_topk_valid
        ori_block_base = b * ORI_MAX_BLOCKS
        cmp_block_base = b * CMP_MAX_BLOCKS
        kv_topk_batch = pl.create_tensor([TOPK, HEAD_DIM], dtype=pl.BF16)
        raw_idx_pos = 0

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_gather_kv_topk"):
            raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
            for kk in pl.range(window_valid):
                raw_idx_pos = b * TOPK + kk
                raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
                ori_slot = raw_idx // BLOCK_SIZE
                ori_block_pos = ori_block_base + ori_slot
                ori_blk = pl.cast(pl.read(ori_block_table_flat, [ori_block_pos]), pl.INDEX)
                ori_intra = raw_idx % BLOCK_SIZE
                ori_row = ori_blk * BLOCK_SIZE + ori_intra
                kv_topk_batch = pl.assemble(
                    kv_topk_batch,
                    ori_kv_flat[ori_row : ori_row + 1, 0 : HEAD_DIM],
                    [kk, 0],
                )

            for kk in pl.range(cmp_topk_valid):
                raw_idx_pos = b * TOPK + window_valid + kk
                raw_idx = pl.read(cmp_sparse_indices_flat, [raw_idx_pos])
                cmp_slot = raw_idx - WIN
                cmp_block_slot = cmp_slot // BLOCK_SIZE
                cmp_block_pos = cmp_block_base + cmp_block_slot
                cmp_blk = pl.cast(pl.read(cmp_block_table_flat, [cmp_block_pos]), pl.INDEX)
                cmp_intra = cmp_slot % BLOCK_SIZE
                cmp_row = cmp_blk * BLOCK_SIZE + cmp_intra
                kv_topk_batch = pl.assemble(
                    kv_topk_batch,
                    cmp_kv_flat[cmp_row : cmp_row + 1, 0 : HEAD_DIM],
                    [window_valid + kk, 0],
                )

        for h in pl.parallel(0, H, 1):
            head_row = b * H + h
            group_idx = h // HEADS_PER_GROUP
            head_in_group = h % HEADS_PER_GROUP
            packed_row = group_idx * T + b
            packed_col = head_in_group * HEAD_DIM
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_init"):
                q_batch = pl.col_expand(
                    pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                    pl.cast(q_flat[head_row : head_row + 1, 0 : HEAD_DIM], target_type=pl.FP32),
                )

                kv_batch = pl.col_expand(
                    pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                    pl.cast(kv_topk_batch[0 : 1, 0 : HEAD_DIM], target_type=pl.FP32),
                )
                oi = kv_batch
                score0 = pl.row_sum(pl.mul(q_batch, kv_batch))
                mi = pl.mul(score0, SOFTMAX_SCALE)
                li = pl.exp(pl.sub(mi, mi))

            for kk in pl.range(1, sparse_k):
                with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_accum"):
                    kv_batch = pl.col_expand(
                        pl.full([MATMUL_ROW_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0),
                        pl.cast(kv_topk_batch[kk : kk + 1, 0 : HEAD_DIM], target_type=pl.FP32),
                    )
                    cur_score = pl.row_sum(pl.mul(q_batch, kv_batch))
                    cur_mi = pl.mul(cur_score, SOFTMAX_SCALE)
                    mi_new = pl.maximum(mi, cur_mi)
                    alpha = pl.exp(pl.sub(mi, mi_new))
                    beta = pl.exp(pl.sub(cur_mi, mi_new))
                    li = pl.add(pl.mul(alpha, li), beta)
                    oi = pl.add(
                        pl.row_expand_mul(oi, alpha),
                        pl.row_expand_mul(kv_batch, beta),
                    )
                    mi = mi_new

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_sparse_attn_norm"):
                sink_tile = pl.add(pl.sub(mi, mi), pl.read(attn_sink, [h]))
                denom = pl.add(li, pl.exp(pl.sub(sink_tile, mi)))
                oi_out = pl.row_expand_div(oi, denom)
                attn_stage_row = pl.cast(
                    oi_out[0 : 1, 0 : HEAD_DIM],
                    target_type=pl.BF16,
                )
                attn_stage = pl.assemble(
                    attn_stage,
                    attn_stage_row,
                    [head_row, 0],
                )

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_inverse_rope"):
                o_nope = attn_stage[head_row : head_row + 1, 0 : NOPE_DIM]
                cos_lo = pl.cast(freqs_cos[b : b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                cos_hi = pl.cast(freqs_cos[b : b + 1, HALF_ROPE : ROPE_DIM], target_type=pl.FP32)
                sin_lo = pl.cast(freqs_sin[b : b + 1, 0 : HALF_ROPE], target_type=pl.FP32)
                sin_hi = pl.cast(freqs_sin[b : b + 1, HALF_ROPE : ROPE_DIM], target_type=pl.FP32)
                x_lo = pl.cast(attn_stage[head_row : head_row + 1, NOPE_DIM : NOPE_DIM + HALF_ROPE], target_type=pl.FP32)
                x_hi = pl.cast(attn_stage[head_row : head_row + 1, NOPE_DIM + HALF_ROPE : HEAD_DIM], target_type=pl.FP32)
                y_lo = pl.add(
                    pl.col_expand_mul(x_lo, cos_lo),
                    pl.col_expand_mul(x_hi, sin_lo),
                )
                y_hi = pl.sub(
                    pl.col_expand_mul(x_hi, cos_hi),
                    pl.col_expand_mul(x_lo, sin_hi),
                )
                o_packed = pl.assemble(o_packed, o_nope, [packed_row, packed_col])
                o_packed = pl.assemble(
                    o_packed,
                    pl.cast(y_lo, target_type=pl.BF16),
                    [packed_row, packed_col + NOPE_DIM],
                )
                o_packed = pl.assemble(
                    o_packed,
                    pl.cast(y_hi, target_type=pl.BF16),
                    [packed_row, packed_col + NOPE_DIM + HALF_ROPE],
                )

    o_r = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.BF16)
    o_r_i8 = pl.create_tensor([T, O_GROUPS * O_LORA], dtype=pl.INT8)

    for g in pl.parallel(0, O_GROUPS, 1):
        row_base_o = g * T
        out_col_g = g * O_LORA

        for nb in pl.parallel(0, A_N_BLOCKS, 1):
            n0 = nb * A_N_CHUNK

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_accum"):
                xa0_chunk = o_packed[row_base_o:row_base_o + T, 0:A_K_CHUNK]
                wa0_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, 0:A_K_CHUNK]
                acc_a = pl.matmul(xa0_chunk, wa0_chunk, b_trans=True, out_dtype=pl.FP32)
                for kb in pl.range(1, A_K_BLOCKS):
                    k0 = kb * A_K_CHUNK
                    xa_k_chunk = o_packed[row_base_o:row_base_o + T, k0:k0 + A_K_CHUNK]
                    wa_k_chunk = wo_a[g:g + 1, n0:n0 + A_N_CHUNK, k0:k0 + A_K_CHUNK]
                    acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True)

            with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_a_store"):
                acc_a_2d = pl.reshape(acc_a, [T, A_N_CHUNK])
                o_r[:, out_col_g + n0:out_col_g + n0 + A_N_CHUNK] = pl.cast(
                    acc_a_2d,
                    target_type=pl.BF16,
                )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_quant"):
        or_amax = pl.full([1, T], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for k0 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
            or_a_f32 = pl.cast(o_r[:, k0:k0 + QUANT_CHUNK], target_type=pl.FP32)
            or_a_abs = pl.maximum(or_a_f32, pl.neg(or_a_f32))
            or_a_max = pl.reshape(pl.row_max(or_a_abs), [1, T])
            or_amax = pl.maximum(or_amax, or_a_max)
        or_sq_row = pl.div(pl.full([1, T], dtype=pl.FP32, value=INT8_SCALE_MAX), or_amax)
        o_r_scale_dq = pl.reshape(pl.recip(or_sq_row), [T, 1])
        or_sq_col = pl.reshape(or_sq_row, [T, 1])
        for k1 in pl.range(0, O_GROUPS * O_LORA, QUANT_CHUNK):
            or_q_f32 = pl.cast(o_r[:, k1:k1 + QUANT_CHUNK], target_type=pl.FP32)
            or_q_scaled = pl.row_expand_mul(or_q_f32, or_sq_col)
            or_q_i32 = pl.cast(or_q_scaled, target_type=pl.INT32, mode="round")
            or_q_half = pl.cast(or_q_i32, target_type=pl.FP16, mode="round")
            o_r_i8[:, k1:k1 + QUANT_CHUNK] = pl.cast(or_q_half, target_type=pl.INT8, mode="trunc")

    for nb in pl.parallel(0, B_N_BLOCKS, 1):
        n0 = nb * B_N_CHUNK

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_accum"):
            xb0_chunk = o_r_i8[:, 0:B_K_CHUNK]
            wb0_chunk = wo_b[n0:n0 + B_N_CHUNK, 0:B_K_CHUNK]
            acc_b = pl.matmul(xb0_chunk, wb0_chunk, b_trans=True, out_dtype=pl.INT32)
            for kb in pl.range(1, B_K_BLOCKS):
                k0 = kb * B_K_CHUNK
                xb_k_chunk = o_r_i8[:, k0:k0 + B_K_CHUNK]
                wb_k_chunk = wo_b[n0:n0 + B_N_CHUNK, k0:k0 + B_K_CHUNK]
                acc_b = pl.matmul_acc(acc_b, xb_k_chunk, wb_k_chunk, b_trans=True)

        with pl.at(level=pl.Level.CORE_GROUP, name_hint="cfa_proj_stage_b_store"):
            wb_scale_chunk = pl.reshape(wo_b_scale[n0:n0 + B_N_CHUNK], [1, B_N_CHUNK])
            attn_chunk = pl.cast(acc_b, target_type=pl.FP32, mode="none")
            attn_chunk = pl.col_expand_mul(pl.row_expand_mul(attn_chunk, o_r_scale_dq), wb_scale_chunk)
            attn_out[:, n0:n0 + B_N_CHUNK] = pl.cast(attn_chunk, target_type=pl.BF16)

    return attn_out


@pl.jit
def deepseek_v4_decode_sparse_attn_with_proj_test(
    q:                  pl.Tensor[[T, H, HEAD_DIM],                               pl.BF16],
    ori_kv:             pl.Tensor[[ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    ori_block_table:    pl.Tensor[[B, ORI_MAX_BLOCKS],                            pl.INT32],
    cmp_kv:             pl.Tensor[[CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],       pl.BF16],
    cmp_block_table:    pl.Tensor[[B, CMP_MAX_BLOCKS],                            pl.INT32],
    cmp_sparse_indices: pl.Tensor[[T, TOPK],                                      pl.INT32],
    attn_sink:          pl.Tensor[[H],                                            pl.FP32],
    seqused_kv:         pl.Tensor[[B],                                            pl.INT32],
    freqs_cos:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    freqs_sin:          pl.Tensor[[T, ROPE_DIM],                                  pl.BF16],
    wo_a:               pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN],                 pl.BF16],
    wo_b:               pl.Tensor[[D, O_GROUPS * O_LORA],                         pl.INT8],
    wo_b_scale:         pl.Tensor[[D],                                            pl.FP32],
    attn_out:           pl.Out[pl.Tensor[[T, D],                                  pl.BF16]],
):
    attn_out = deepseek_v4_decode_sparse_attn_with_proj(
        q,
        ori_kv,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        cmp_sparse_indices,
        attn_sink,
        seqused_kv,
        freqs_cos,
        freqs_sin,
        wo_a,
        wo_b,
        wo_b_scale,
        attn_out,
    )
    return attn_out


def _round_half_away_from_zero(x):
    import torch

    return torch.sign(x) * torch.floor(torch.abs(x) + 0.5)


def _int8_quant_per_row(x):
    """Per-row INT8 symmetric quant matching the runtime W8A8C16 activation path."""
    import torch

    rows = x.float().reshape(-1, x.shape[-1])
    amax = rows.abs().amax(dim=-1, keepdim=True).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = rows * scale_quant
    out_i8 = _round_half_away_from_zero(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    scale_dequant = 1.0 / scale_quant
    return out_i8.reshape_as(x), scale_dequant.reshape(*x.shape[:-1], 1)


def _quant_w_per_channel(w):
    """Per-output-channel INT8 quant on the last axis."""
    import torch

    amax = w.float().abs().amax(dim=-1).clamp_min(INT8_AMAX_EPS)
    scale_quant = INT8_SCALE_MAX / amax
    scaled = w.float() * scale_quant.unsqueeze(-1)
    w_i8 = _round_half_away_from_zero(scaled).to(torch.int32).to(torch.float16).to(torch.int8)
    return w_i8, (1.0 / scale_quant).float()


def golden_deepseek_v4_decode_sparse_attn_with_proj(tensors):
    """Torch reference: sparse_attn decode path followed by grouped o_proj."""
    import torch

    q = tensors["q"].float()
    ori_kv = tensors["ori_kv"].float()
    ori_block_table = tensors["ori_block_table"]
    cmp_kv = tensors["cmp_kv"].float()
    cmp_block_table = tensors["cmp_block_table"]
    cmp_sparse_indices = tensors["cmp_sparse_indices"]
    attn_sink = tensors["attn_sink"].float()
    seqused_kv = tensors["seqused_kv"]
    cos = tensors["freqs_cos"].float()
    sin = tensors["freqs_sin"].float()
    wo_a = tensors["wo_a"].float()
    wo_b_i8 = tensors["wo_b"]
    wo_b_scale = tensors["wo_b_scale"].float()

    o = torch.zeros(T, H, HEAD_DIM)

    for b in range(B):
        seq_used = int(seqused_kv[b].item())
        window_valid = min(WIN, seq_used)
        cmp_valid = max(seq_used - window_valid, 0)
        gathered = []

        for raw in cmp_sparse_indices[b].tolist():
            if raw < 0:
                continue
            if raw < WIN:
                if raw >= window_valid:
                    continue
                blk_id = int(ori_block_table[b, raw // BLOCK_SIZE].item())
                intra = raw % BLOCK_SIZE
                gathered.append(ori_kv[blk_id, intra, 0])
            else:
                cmp_slot = raw - WIN
                if cmp_slot >= cmp_valid:
                    continue
                blk_id = int(cmp_block_table[b, cmp_slot // BLOCK_SIZE].item())
                intra = cmp_slot % BLOCK_SIZE
                gathered.append(cmp_kv[blk_id, intra, 0])

        if not gathered:
            continue

        kv_b = torch.stack(gathered, dim=0)
        q_b = q[b]
        scores = (q_b @ kv_b.T) * SOFTMAX_SCALE
        score_max = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - score_max)
        oi_num = exp_scores @ kv_b
        li = exp_scores.sum(dim=-1, keepdim=True)
        denom = li + torch.exp(attn_sink.unsqueeze(-1) - score_max)
        o[b] = oi_num / denom

    rope_lo = o[..., NOPE_DIM : NOPE_DIM + HALF_ROPE]
    rope_hi = o[..., NOPE_DIM + HALF_ROPE :]
    cos_lo = cos[:, :HALF_ROPE].unsqueeze(1)
    cos_hi = cos[:, HALF_ROPE:].unsqueeze(1)
    sin_lo = sin[:, :HALF_ROPE].unsqueeze(1)
    sin_hi = sin[:, HALF_ROPE:].unsqueeze(1)
    inv_lo = rope_lo * cos_lo + rope_hi * sin_lo
    inv_hi = rope_hi * cos_hi - rope_lo * sin_hi
    o = torch.cat([o[..., :NOPE_DIM], inv_lo, inv_hi], dim=-1).to(torch.bfloat16)

    seq_per_batch = T // B
    o_model = o.float().view(B, seq_per_batch, O_GROUPS, O_GROUP_IN)
    o_r = torch.einsum("bsgd,grd->bsgr", o_model, wo_a)
    o_r = o_r.to(torch.bfloat16).float()
    o_r_q = o_r.flatten(2).view(T, O_GROUPS * O_LORA)
    o_r_i8, o_r_scale = _int8_quant_per_row(o_r_q)
    acc = o_r_i8.to(torch.int32) @ wo_b_i8.to(torch.int32).T
    out = acc.float() * o_r_scale * wo_b_scale.unsqueeze(0)

    tensors["attn_out"][:] = out.to(torch.bfloat16)


def build_tensor_specs(compress_ratio: int = DEFAULT_COMPRESS_RATIO):
    """Build deterministic demo tensors for the merged standalone harness."""
    import torch
    from golden import TensorSpec

    cmp_valid = get_standalone_cmp_valid(compress_ratio)
    sparse_k = WIN + cmp_valid

    def seeded_uniform(shape, seed):
        """Create a deterministic centered uniform tensor for repeatable tests."""
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.rand(*shape, generator=generator) - 0.5

    def init_q():
        """Initialize the query tensor used by the decode attention stage."""
        return seeded_uniform((T, H, HEAD_DIM), 1)

    def init_ori_kv():
        """Initialize the sliding-window KV cache pages."""
        return seeded_uniform((ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 2)

    def init_cmp_kv():
        """Initialize the compressed-cache KV pages."""
        return seeded_uniform((CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM), 3)

    def init_attn_sink():
        """Initialize the per-head sink logits to zero."""
        return torch.zeros(H)

    def init_ori_block_table():
        """Build the demo block table for the sliding-window cache pages."""
        tbl = torch.full((B, ORI_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(ORI_MAX_BLOCKS):
                tbl[b, j] = b * ORI_MAX_BLOCKS + j
        return tbl

    def init_cmp_block_table():
        """Build the demo block table for the compressed-cache pages."""
        tbl = torch.full((B, CMP_MAX_BLOCKS), -1, dtype=torch.int32)
        for b in range(B):
            for j in range(CMP_MAX_BLOCKS):
                tbl[b, j] = b * CMP_MAX_BLOCKS + j
        return tbl

    def init_cmp_sparse_indices():
        """Build the sparse index list with a full window prefix and padded compressed tail."""
        win_part = torch.arange(WIN, dtype=torch.int32).unsqueeze(0).expand(T, -1)
        cmp_part = torch.full((T, IDX_TOPK), -1, dtype=torch.int32)
        cmp_part[:, :cmp_valid] = (torch.arange(cmp_valid, dtype=torch.int32) + WIN).unsqueeze(0).expand(T, -1)
        return torch.cat([win_part, cmp_part], dim=-1).contiguous()

    def init_seqused_kv():
        """Expose the demo sequence-used length that matches the chosen ratio mode."""
        return torch.tensor([sparse_k] * B, dtype=torch.int32)

    def init_cos():
        """Build the split-half cosine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        cos_half = torch.cos(angles)
        return torch.cat([cos_half, cos_half], dim=-1)

    def init_sin():
        """Build the split-half sine table used by the inverse-RoPE reference."""
        angles = torch.arange(T * HALF_ROPE).reshape(T, HALF_ROPE) * 1e-3
        sin_half = torch.sin(angles)
        return torch.cat([sin_half, sin_half], dim=-1)

    def init_wo_a():
        """Initialize the grouped first-stage output-projection weights."""
        return seeded_uniform((O_GROUPS, O_LORA, O_GROUP_IN), 4) / (O_GROUP_IN ** 0.5)

    wo_b_bf16 = (seeded_uniform((D, O_GROUPS * O_LORA), 5) / ((O_GROUPS * O_LORA) ** 0.5)).to(torch.bfloat16)
    wo_b_i8, wo_b_scale = _quant_w_per_channel(wo_b_bf16)

    def init_wo_b():
        """Initialize the second-stage output-projection weights in per-channel INT8 form."""
        return wo_b_i8

    def init_wo_b_scale():
        """Initialize the dequant scales paired with the INT8 second-stage weights."""
        return wo_b_scale

    return [
        TensorSpec("q", [T, H, HEAD_DIM], torch.bfloat16, init_value=init_q),
        TensorSpec("ori_kv", [ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_ori_kv),
        TensorSpec("ori_block_table", [B, ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec("cmp_kv", [CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_cmp_kv),
        TensorSpec("cmp_block_table", [B, CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("cmp_sparse_indices", [T, TOPK], torch.int32, init_value=init_cmp_sparse_indices),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("seqused_kv", [B], torch.int32, init_value=init_seqused_kv),
        TensorSpec("freqs_cos", [T, ROPE_DIM], torch.bfloat16, init_value=init_cos),
        TensorSpec("freqs_sin", [T, ROPE_DIM], torch.bfloat16, init_value=init_sin),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=init_wo_b),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=init_wo_b_scale),
        TensorSpec("attn_out", [T, D], torch.bfloat16, is_output=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compress-ratio", type=int, default=DEFAULT_COMPRESS_RATIO,
                        choices=list(SUPPORTED_COMPRESS_RATIOS))
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run_jit(
        fn=deepseek_v4_decode_sparse_attn_with_proj_test,
        specs=build_tensor_specs(args.compress_ratio),
        golden_fn=golden_deepseek_v4_decode_sparse_attn_with_proj,
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile=dict(dump_passes=True),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
