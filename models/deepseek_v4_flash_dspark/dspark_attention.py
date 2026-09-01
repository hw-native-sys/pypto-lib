# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 DSpark drafter attention over the paged sliding-window cache.

One anchor-first draft query block per request. Query rows remain on their DSA-CP
token owner, while the KV input contains the complete rank-major CP-group stream.
Every draft row sees the trailing context window plus the whole draft block, so
the visible slot list is per request and carries no causal mask inside the block.
Context KV is already resident (see dspark_context_kv); this kernel commits the
block's own group KV before reading window and block through one index list.
"""

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    DECODE_BATCH,
    DSPARK_SPEC_TOKENS,
    FLASH as M,
    KV_ORI_BLOCK_NUM,
    TP,
)
from decode_o_proj import LOCAL_T_PAD
from qkv_proj_rope import (
    kv_proj_rope,
    materialize_rope_rows_dynamic,
    q_proj_rope,
    rope_prepare,
)


# Dynamic shape variables.
ORI_BLOCK_NUM_DYN = pl.dynamic("DSPARK_ATTENTION_ORI_BLOCK_NUM_DYN")
KV_T_DYN = pl.dynamic("DSPARK_ATTENTION_KV_T_DYN")

# model config
B = DECODE_BATCH // TP
S = DSPARK_SPEC_TOKENS                   # anchor-first draft query rows per request
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
WIN = M.sliding_window
O_GROUPS = M.o_groups
HEADS_PER_GROUP = H // O_GROUPS
O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM
MAX_SEQ_LEN = M.max_position_embeddings
ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
SOFTMAX_SCALE = M.softmax_scale
VISIBLE_ROWS = WIN + S                   # trailing window + whole draft block

# tiling
ATTN_K_TILE = 64
SPARSE_BLOCKS = (VISIBLE_ROWS + ATTN_K_TILE - 1) // ATTN_K_TILE
INDEX_WIDTH = SPARSE_BLOCKS * ATTN_K_TILE
BIAS_B_TILE = 8                          # 2 request blocks
H_TILE = 16                              # 4 head blocks
NEG_INF = -1.0e20


@pl.jit.inline
def dspark_attention(
    x: pl.Tensor[[T, D], pl.BF16],
    kv_x: pl.Tensor[[KV_T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    kv_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    o_packed_heads: pl.Tensor[
        [O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16
    ],
):
    kv_tokens = pl.tensor.dim(kv_position_ids, 0)
    rope_cos_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_DIM], dtype=pl.BF16)
    for rope_t in pl.spmd(T, name_hint="dspark_q_rope_rows"):
        rope_position = pl.cast(pl.read(position_ids, [rope_t]), pl.INDEX)
        rope_cos_t[rope_t : rope_t + 1, :] = freqs_cos[
            rope_position : rope_position + 1, :
        ]
        rope_sin_t[rope_t : rope_t + 1, :] = freqs_sin[
            rope_position : rope_position + 1, :
        ]

    rope_cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([T, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(rope_cos_t, rope_sin_t, rope_cos_il, rope_sin_signed, rope_swap_idx)

    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    q_proj_rope(
        x, wq_a, wq_b, wq_b_scale, gamma_cq,
        rope_cos_il, rope_sin_signed, rope_swap_idx,
        q, qr, qr_scale,
    )

    kv_rope_cos_t = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.BF16)
    kv_rope_sin_t = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.BF16)
    materialize_rope_rows_dynamic(
        freqs_cos,
        freqs_sin,
        kv_position_ids,
        kv_rope_cos_t,
        kv_rope_sin_t,
    )
    kv_rope_cos_il = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.FP32)
    kv_rope_sin_signed = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.FP32)
    kv_rope_swap_idx = pl.create_tensor([kv_tokens, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(
        kv_rope_cos_t,
        kv_rope_sin_t,
        kv_rope_cos_il,
        kv_rope_sin_signed,
        kv_rope_swap_idx,
    )

    # DSA-CP keeps Q on its token owner while every rank updates the complete
    # group KV stream before running attention for its local queries.
    late_dep = pl.system.task_dummy(deps=[])
    kv = pl.create_tensor([kv_tokens, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(
        kv_x,
        wkv,
        gamma_ckv,
        kv_rope_cos_il,
        kv_rope_sin_signed,
        kv_rope_swap_idx,
        kv,
        late_dep,
    )

    # Commit the block's own KV and build the visible-length mask in one task; the
    # gather below reads those rows back through swa_indices.
    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    sparse_bias = pl.create_tensor([B, INDEX_WIDTH], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="dspark_kv_commit_valid_bias"):
        for write_t in pl.range(kv_tokens):
            write_row_i64 = pl.read(kv_slot_mapping, [write_t])
            if write_row_i64 >= 0:
                write_row = pl.cast(write_row_i64, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, 0:HEAD_DIM] = kv[write_t : write_t + 1, 0:HEAD_DIM]
        v_col = pl.cast(pl.arange(0, [1, INDEX_WIDTH], dtype=pl.INT32), target_type=pl.FP32)
        for v_blk in pl.range(B // BIAS_B_TILE):
            v_b0 = v_blk * BIAS_B_TILE
            v_col_m = pl.col_expand(pl.full([BIAS_B_TILE, INDEX_WIDTH], dtype=pl.FP32, value=0.0), v_col)
            v_lens = pl.cast(pl.reshape(swa_lens[v_b0 : v_b0 + BIAS_B_TILE], [BIAS_B_TILE, 1]), target_type=pl.FP32)
            v_valid = pl.minimum(pl.maximum(pl.neg(pl.row_expand_sub(v_col_m, v_lens)), 0.0), 1.0)
            sparse_bias[v_b0 : v_b0 + BIAS_B_TILE, 0:INDEX_WIDTH] = pl.mul(pl.sub(v_valid, 1.0), -NEG_INF)

    # One index row per request: build_dspark_swa_indices repeats it across the
    # block's query rows, so the gather runs B times, not T.
    visible_kv = pl.create_tensor([B * INDEX_WIDTH, HEAD_DIM], dtype=pl.BF16)
    for g_task in pl.spmd(B * SPARSE_BLOCKS, name_hint="dspark_gather_kv"):
        g_b = g_task // SPARSE_BLOCKS
        g_c0 = (g_task % SPARSE_BLOCKS) * ATTN_K_TILE
        g_base = g_b * INDEX_WIDTH
        for g_c in pl.range(g_c0, g_c0 + ATTN_K_TILE):
            g_slot_i32 = pl.read(swa_indices, [g_b, g_c])
            g_dst = g_base + g_c
            if g_slot_i32 >= 0:
                g_slot = pl.cast(g_slot_i32, pl.INDEX)
                visible_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = kv_cache_flat[g_slot : g_slot + 1, 0:HEAD_DIM]
            else:
                visible_kv[g_dst : g_dst + 1, 0:HEAD_DIM] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)

    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    sparse_mi = pl.create_tensor([T * SPARSE_BLOCKS * H, 1], dtype=pl.FP32)
    sparse_li = pl.create_tensor([T * SPARSE_BLOCKS * H, 1], dtype=pl.FP32)
    sparse_oi = pl.create_tensor([T * SPARSE_BLOCKS * H, HEAD_DIM], dtype=pl.FP32)
    for qk_idx in pl.spmd(T * SPARSE_BLOCKS, name_hint="dspark_qk_pv"):
        qk_token_idx = qk_idx // SPARSE_BLOCKS
        qk_sparse_block = qk_idx % SPARSE_BLOCKS
        qk_batch_idx = qk_token_idx // S
        qk_q_row = qk_token_idx * H
        qk_s0 = qk_sparse_block * ATTN_K_TILE
        kv_row = qk_batch_idx * INDEX_WIDTH + qk_s0
        q_tile = q_flat[qk_q_row : qk_q_row + H, :]
        kv_tile = visible_kv[kv_row : kv_row + ATTN_K_TILE, :]
        bias_row = sparse_bias[qk_batch_idx : qk_batch_idx + 1, qk_s0 : qk_s0 + ATTN_K_TILE]
        score_raw = pl.matmul(q_tile, kv_tile, b_trans=True, out_dtype=pl.FP32)
        score_scaled = pl.mul(score_raw, SOFTMAX_SCALE)
        # Masked lanes carry NEG_INF over zeroed kv rows and exp to ~0; an all-masked
        # block dies in the merge scale below.
        scores = pl.col_expand_add(score_scaled, bias_row)
        score_max = pl.row_max(scores)
        score_exp = pl.exp(pl.row_expand_sub(scores, score_max))
        score_sum = pl.row_sum(score_exp)
        score_prob = pl.cast(score_exp, target_type=pl.BF16, mode="rint")
        attn_raw = pl.matmul(score_prob, kv_tile, out_dtype=pl.FP32)
        qk_partial_row = (qk_token_idx * SPARSE_BLOCKS + qk_sparse_block) * H
        sparse_mi[qk_partial_row : qk_partial_row + H, 0:1] = score_max
        sparse_li[qk_partial_row : qk_partial_row + H, 0:1] = score_sum
        sparse_oi[qk_partial_row : qk_partial_row + H, :] = attn_raw

    o_packed_flat = pl.reshape(
        o_packed_heads,
        [O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM],
    )
    for merge_idx in pl.spmd(T * (H // H_TILE), name_hint="dspark_merge_norm"):
        merge_token_idx = merge_idx // (H // H_TILE)
        merge_head0 = (merge_idx % (H // H_TILE)) * H_TILE
        merge_partial_row0 = merge_token_idx * SPARSE_BLOCKS * H + merge_head0
        running_max = sparse_mi[merge_partial_row0 : merge_partial_row0 + H_TILE, 0:1]
        running_sum = sparse_li[merge_partial_row0 : merge_partial_row0 + H_TILE, 0:1]
        running_out = sparse_oi[merge_partial_row0 : merge_partial_row0 + H_TILE, :]
        for merge_sparse_block in pl.range(1, SPARSE_BLOCKS):
            merge_partial_row = merge_partial_row0 + merge_sparse_block * H
            block_max = sparse_mi[merge_partial_row : merge_partial_row + H_TILE, 0:1]
            block_sum = sparse_li[merge_partial_row : merge_partial_row + H_TILE, 0:1]
            block_out = sparse_oi[merge_partial_row : merge_partial_row + H_TILE, :]
            merged_max = pl.maximum(running_max, block_max)
            running_scale = pl.exp(pl.sub(running_max, merged_max))
            block_scale = pl.exp(pl.sub(block_max, merged_max))
            running_out_scaled = pl.row_expand_mul(running_out, running_scale)
            block_out_scaled = pl.row_expand_mul(block_out, block_scale)
            running_out = pl.add(running_out_scaled, block_out_scaled)
            running_sum_scaled = pl.mul(running_sum, running_scale)
            block_sum_scaled = pl.mul(block_sum, block_scale)
            running_sum = pl.add(running_sum_scaled, block_sum_scaled)
            running_max = merged_max
        sink_col = pl.reshape(attn_sink[merge_head0 : merge_head0 + H_TILE], [H_TILE, 1])
        sink_exp = pl.exp(pl.sub(sink_col, running_max))
        denominator = pl.add(running_sum, sink_exp)
        attn_normed = pl.row_expand_div(running_out, denominator)

        attn_normed_bf16 = pl.cast(attn_normed, target_type=pl.BF16, mode="rint")
        attn_nope = attn_normed_bf16[:, 0:NOPE_DIM]
        attn_rope = attn_normed[:, NOPE_DIM:HEAD_DIM]
        rope_even = pl.gather(attn_rope, mask_pattern=pl.tile.MaskPattern.P0101)
        rope_odd = pl.gather(attn_rope, mask_pattern=pl.tile.MaskPattern.P1010)
        cos_half = rope_cos_t[merge_token_idx : merge_token_idx + 1, 0:ROPE_HALF]
        sin_half = rope_sin_t[merge_token_idx : merge_token_idx + 1, 0:ROPE_HALF]
        cos_fp32 = pl.cast(cos_half, target_type=pl.FP32, mode="none")
        sin_fp32 = pl.cast(sin_half, target_type=pl.FP32, mode="none")
        inverse_even_cos = pl.col_expand_mul(rope_even, cos_fp32)
        inverse_odd_sin = pl.col_expand_mul(rope_odd, sin_fp32)
        inverse_even = pl.add(inverse_even_cos, inverse_odd_sin)
        inverse_even_sin = pl.col_expand_mul(rope_even, sin_fp32)
        inverse_odd_cos = pl.col_expand_mul(rope_odd, cos_fp32)
        inverse_odd = pl.sub(inverse_odd_cos, inverse_even_sin)
        inverse_rope = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=0.0)
        inverse_rope = pl.tensor.scatter(inverse_even, mask_pattern=pl.tile.MaskPattern.P0101, dst=inverse_rope)
        inverse_rope = pl.tensor.scatter(inverse_odd, mask_pattern=pl.tile.MaskPattern.P1010, dst=inverse_rope)
        inverse_rope_bf16 = pl.cast(inverse_rope, target_type=pl.BF16, mode="rint")
        for merge_group in pl.unroll(H_TILE // HEADS_PER_GROUP):
            group = merge_head0 // HEADS_PER_GROUP + merge_group
            group_head = merge_group * HEADS_PER_GROUP
            packed_row = (group * LOCAL_T_PAD + merge_token_idx) * HEADS_PER_GROUP
            o_packed_flat[
                packed_row : packed_row + HEADS_PER_GROUP, 0:NOPE_DIM
            ] = attn_nope[
                group_head : group_head + HEADS_PER_GROUP, 0:NOPE_DIM
            ]
            o_packed_flat[
                packed_row : packed_row + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM
            ] = inverse_rope_bf16[
                group_head : group_head + HEADS_PER_GROUP, 0:ROPE_DIM
            ]

    return kv_cache, o_packed_heads


@pl.jit
def dspark_attention_test(
    x: pl.Tensor[[T, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    slot_mapping: pl.Tensor[[T], pl.INT64],
    swa_indices: pl.Tensor[[B, INDEX_WIDTH], pl.INT32],
    swa_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    o_packed_heads: pl.Out[
        pl.Tensor[[O_GROUPS, LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM], pl.BF16]
    ],
):
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    o_packed_flat = pl.reshape(
        o_packed_heads,
        [O_GROUPS * LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM],
    )
    kv_cache, o_packed_flat = dspark_attention(
        x,
        x,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, position_ids, position_ids,
        kv_cache, slot_mapping, swa_indices, swa_lens,
        attn_sink, o_packed_flat,
    )
    return kv_cache, o_packed_heads


def golden_dspark_attention(tensors):
    import torch
    from qkv_proj_rope import golden_qkv_proj_rope

    positions = tensors["position_ids"].long()
    rope_cos = tensors["freqs_cos"].index_select(0, positions)
    rope_sin = tensors["freqs_sin"].index_select(0, positions)

    q = torch.zeros(T, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(T, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(T, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(T, 1, dtype=torch.float32)
    golden_qkv_proj_rope({
        "x": tensors["x"],
        "wq_a": tensors["wq_a"],
        "wq_b": tensors["wq_b"],
        "wq_b_scale": tensors["wq_b_scale"],
        "wkv": tensors["wkv"],
        "rope_cos": rope_cos,
        "rope_sin": rope_sin,
        "gamma_cq": tensors["gamma_cq"],
        "gamma_ckv": tensors["gamma_ckv"],
        "q": q,
        "kv": kv,
        "qr": qr,
        "qr_scale": qr_scale,
    })

    kv_cache_flat = tensors["kv_cache"].view(-1, HEAD_DIM)
    slots = tensors["slot_mapping"]
    for token_idx in range(T):
        cache_row = int(slots[token_idx].item())
        if cache_row >= 0:
            kv_cache_flat[cache_row] = kv[token_idx]

    sink = tensors["attn_sink"].float().view(H, 1)
    cols = torch.arange(INDEX_WIDTH)
    attn_heads = torch.empty(T, H, HEAD_DIM, dtype=torch.float32)
    for batch_idx in range(B):
        index_row = tensors["swa_indices"][batch_idx]
        visible_kv = torch.zeros(INDEX_WIDTH, HEAD_DIM, dtype=torch.float32)
        for col in range(INDEX_WIDTH):
            slot = int(index_row[col].item())
            if slot >= 0:
                visible_kv[col] = kv_cache_flat[slot].float()
        bias = torch.where(cols < int(tensors["swa_lens"][batch_idx].item()), 0.0, NEG_INF)
        for seq_idx in range(S):
            token_idx = batch_idx * S + seq_idx
            scores = torch.matmul(q[token_idx].float(), visible_kv.t()) * SOFTMAX_SCALE + bias
            score_max = scores.amax(dim=-1, keepdim=True)
            score_exp = torch.exp(scores - score_max)
            denominator = score_exp.sum(dim=-1, keepdim=True) + torch.exp(sink - score_max)
            attn_heads[token_idx] = torch.matmul(score_exp, visible_kv) / denominator

    rope_pairs = attn_heads[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    rope_even = rope_pairs[..., 0]
    rope_odd = rope_pairs[..., 1]
    cos_half = rope_cos[:, :ROPE_HALF].float().unsqueeze(-2)
    sin_half = rope_sin[:, :ROPE_HALF].float().unsqueeze(-2)
    inverse_even = rope_even * cos_half + rope_odd * sin_half
    inverse_odd = rope_odd * cos_half - rope_even * sin_half
    attn_heads[..., NOPE_DIM:] = torch.stack([inverse_even, inverse_odd], dim=-1).flatten(-2)

    packed = tensors["o_packed_heads"].view(
        O_GROUPS, LOCAL_T_PAD, HEADS_PER_GROUP, HEAD_DIM
    )
    grouped = attn_heads.to(torch.bfloat16).view(
        T, O_GROUPS, HEADS_PER_GROUP, HEAD_DIM
    )
    packed[:, :T] = grouped.permute(1, 0, 2, 3)


def build_tensor_specs(start_pos=None):
    import torch
    from golden import TensorSpec
    from utils import (
        block_table,
        build_rope_tables,
        paged_slot_mapping,
        position_ids_from_starts,
        quant_w_per_channel,
        resolve_start_positions,
        swa_decode_start_set,
    )

    freqs_cos, freqs_sin = build_rope_tables(M, 0, dtype=torch.bfloat16)

    def init_start_pos():
        return resolve_start_positions(
            start_pos,
            batch=B,
            seq=S,
            max_seq_len=MAX_SEQ_LEN,
            default_fn=lambda: swa_decode_start_set(batch=B, window=WIN),
        )

    def init_block_table():
        return block_table(batch=B, table_blocks=ORI_MAX_BLOCKS, physical_blocks=KV_ORI_BLOCK_NUM)

    def init_position_ids():
        return position_ids_from_starts(init_start_pos(), seq=S).reshape(-1).contiguous()

    def init_slot_mapping():
        return paged_slot_mapping(
            position_ids_from_starts(init_start_pos(), seq=S), init_block_table(), block_size=BLOCK_SIZE
        ).reshape(-1).contiguous()

    def init_swa_metadata():
        # build_dspark_swa_indices: window start clamped at 0, visible = window + block.
        starts = init_start_pos().to(torch.int64)
        table = init_block_table()
        seq_lens = starts + S
        window_starts = (starts - WIN).clamp(min=0)
        visible_lens = seq_lens - window_starts
        cols = torch.arange(INDEX_WIDTH, dtype=torch.int64)
        pos = window_starts.unsqueeze(1) + cols.unsqueeze(0)
        slots = paged_slot_mapping(pos, table, block_size=BLOCK_SIZE)
        slots = torch.where(cols.unsqueeze(0) < visible_lens.unsqueeze(1), slots, -1)
        return slots.to(torch.int32).contiguous(), visible_lens.to(torch.int32).contiguous()

    def init_swa_indices():
        return init_swa_metadata()[0]

    def init_swa_lens():
        return init_swa_metadata()[1]

    def init_kv_cache():
        cache = torch.randn(KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        rms = cache.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(M.rms_norm_eps)
        return (cache / rms).to(torch.bfloat16)

    def init_x():
        return torch.randn(T, D, dtype=torch.bfloat16) * 0.05

    def init_wq_a():
        return (torch.randn(D, Q_LORA) / D ** 0.5).to(torch.bfloat16)

    def init_wkv():
        return (torch.randn(D, HEAD_DIM) / D ** 0.5).to(torch.bfloat16)

    wq_b_bf16 = (torch.randn(Q_LORA, H * HEAD_DIM) / Q_LORA ** 0.5).to(torch.bfloat16)
    wq_b_i8, wq_b_scale = quant_w_per_channel(wq_b_bf16.t().contiguous())
    wq_b_i8 = wq_b_i8.t().contiguous()

    return [
        TensorSpec("x", [T, D], torch.bfloat16, init_value=init_x),
        TensorSpec("wq_a", [D, Q_LORA], torch.bfloat16, init_value=init_wq_a),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.int8, init_value=lambda: wq_b_i8),
        TensorSpec("wq_b_scale", [H * HEAD_DIM], torch.float32, init_value=lambda: wq_b_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=init_wkv),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16, init_value=lambda: torch.ones(Q_LORA)),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.ones(HEAD_DIM)),
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=lambda: freqs_cos.clone()),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_DIM], torch.bfloat16, init_value=lambda: freqs_sin.clone()),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec(
            "kv_cache",
            [KV_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
        ),
        TensorSpec("slot_mapping", [T], torch.int64, init_value=init_slot_mapping),
        TensorSpec("swa_indices", [B, INDEX_WIDTH], torch.int32, init_value=init_swa_indices),
        TensorSpec("swa_lens", [B], torch.int32, init_value=init_swa_lens),
        TensorSpec("attn_sink", [H], torch.float32, init_value=lambda: torch.zeros(H)),
        TensorSpec(
            "o_packed_heads",
            [O_GROUPS, LOCAL_T_PAD * HEADS_PER_GROUP, HEAD_DIM],
            torch.bfloat16,
        ),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek-V4 DSpark drafter attention validation.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--start-pos", type=int, default=None)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=(0, 1, 2, 4))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        fn=dspark_attention_test,
        specs=build_tensor_specs(args.start_pos),
        golden_fn=golden_dspark_attention,
        compile_cfg=dict(dump_passes=args.dump_passes),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "o_packed_heads": ratio_allclose(
                atol=1e-2,
                rtol=1e-2,
                max_error_ratio=0.05,
                valid_rows=T * HEADS_PER_GROUP,
                valid_axis=1,
            ),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
