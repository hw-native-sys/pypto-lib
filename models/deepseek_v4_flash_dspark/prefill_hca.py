# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
"""DeepSeek-V4 packed prefill HCA (ratio-128) attention over one contiguous run of <=T tokens."""

import functools

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    FLASH as M,
    HCA_STATE_PHYSICAL_BLOCKS,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    KV_ORI_BLOCK_NUM,
    PREFILL_SEQ,
)

from hc_post import golden_hc_post, hc_post
from hc_pre import golden_hc_pre, hc_pre
from prefill_compressor_ratio128 import (
    HCA_STATE_BLOCK_NUM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    golden_prefill_compressor_ratio128,
    prefill_compressor_ratio128,
)
from prefill_metadata import QUERY_START_LOC_DYN, REQUESTS_DYN
from qkv_proj_rope import golden_qkv_proj_rope, qkv_proj_rope
from rmsnorm import golden_rms_norm, rms_norm
from prefill_sparse_attn import (
    golden_prefill_sparse_attn,
    hca_streaming_attn_physical,
)

import pypto.language.distributed as pld

from prefill_cp_token_allgather import (
    PREFILL_GROUP_CAP,
    TP_SIZE,
    prefill_cp_token_allgather_step,
)
from prefill_o_proj import (
    O_PROJ_LOCAL_COLS,
    O_PROJ_LOCAL_GROUPS,
    O_PROJ_SCRATCH_COLS,
    O_PROJ_SCRATCH_D,
    O_PROJ_SCRATCH_GROUPS,
    O_PROJ_SCRATCH_INPUT,
    O_PROJ_SCRATCH_RANK,
    O_PROJ_WO_A_WINDOW_COLS,
    O_PROJ_WO_A_WINDOW_ROWS,
    O_PROJ_WO_B_WINDOW_COLS,
    O_PROJ_WO_B_WINDOW_ROWS,
    gather_o_proj_full_weights,
)
from qkv_proj_rope import kv_proj_rope, q_proj_rope, rope_prepare


# Dynamic shape variables.
T_DYN = pl.dynamic("PREFILL_HCA_T_DYN")
CP_Q_T_DYN = pl.dynamic("PREFILL_HCA_CP_Q_T_DYN")
CP_KV_T_DYN = pl.dynamic("PREFILL_HCA_CP_KV_T_DYN")
ORI_BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")
CMP_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_CMP_BLOCK_NUM_DYN")
STATE_BLOCK_NUM_DYN = pl.dynamic("PREFILL_HCA_STATE_BLOCK_NUM_DYN")

# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
NOPE_HEAD_DIM = M.nope_head_dim
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

COMPRESS_RATIO = 128
MAIN_OUT_DIM = HEAD_DIM
MAIN_COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM
START_POS = 0

# paged KV cache
SPARSE_ORI_MAX_BLOCKS = (MAX_SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE
SPARSE_ORI_BLOCK_NUM = KV_ORI_BLOCK_NUM
SPARSE_CMP_MAX_BLOCKS = (MAX_SEQ_LEN // COMPRESS_RATIO + BLOCK_SIZE - 1) // BLOCK_SIZE
SPARSE_CMP_BLOCK_NUM = SPARSE_CMP_MAX_BLOCKS
HCA_ORI_BLOCK_NUM = SPARSE_ORI_BLOCK_NUM
HCA_CMP_BLOCK_NUM = SPARSE_CMP_BLOCK_NUM



@pl.jit.inline
def prefill_attention_hca(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[T_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_mixed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    post = pl.create_tensor([t_dim, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([t_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post, comb)

    x_normed = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    # Defers kv_proj_matmul one hop behind rms_norm so qr_proj_matmul dispatches first.
    late_dep = pl.system.task_dummy(deps=[rms_tid])

    q = pl.create_tensor([t_dim, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([t_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    qkv_proj_rope(
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
        late_dep,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [ori_cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cache_write") as ori_cache_write_tid:
        for write_t in pl.range(t_dim):
            write_row_raw = pl.read(ori_slot_mapping, [write_t])
            if write_row_raw >= 0:
                write_row = pl.cast(write_row_raw, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, :] = kv[write_t : write_t + 1, :]

    prefill_compressor_ratio128(
        x_normed,
        query_start_loc,
        compress_state,
        compress_state_block_table,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        cmp_freqs_cos,
        cmp_freqs_sin,
        cmp_kv,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
    )

    swa_indices = pl.create_tensor([t_dim, WIN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_swa_indices") as swa_indices_tid:
        for idx_t in pl.range(t_dim):
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            request_id = pl.read(local_request_ids, [idx_t])
            if request_id >= 0:
                abs_pos = pl.read(position_ids, [idx_t])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(ori_block_table, [request_id, pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(swa_row, [0, win_col], row)
            swa_indices[idx_t : idx_t + 1, 0:WIN] = swa_row

    # Streaming-attention input publication fence.
    cmp_cache_rows = pl.tensor.dim(cmp_kv, 0) * BLOCK_SIZE
    state_rows = pl.tensor.dim(compress_state, 0) * HCA_STATE_BLOCK_SIZE
    cmp_cache_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    compress_state_flat = pl.reshape(compress_state, [state_rows, MAIN_COMPRESS_STATE_DIM])
    q_ready_flat = pl.reshape(q, [t_dim * H, HEAD_DIM])
    cache_ready_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cache_ready",
        deps=[ori_cache_write_tid, swa_indices_tid],
    ) as cache_ready_dep:
        ready_bit = pl.cast(1, pl.INT32)
        for ready_t in pl.range(t_dim):
            q_ready_tile = pl.load(q_ready_flat, [ready_t * H, 0], [1, 16])
            q_ready_bits = pl.reinterpret_view(q_ready_tile, pl.INT16)
            q_ready_sample = pl.tile.read(q_ready_bits, [0, 0])
            ready_bit = ready_bit + pl.cast(q_ready_sample, pl.INT32)
            state_ready_row_raw = pl.read(state_slot_mapping, [ready_t])
            if state_ready_row_raw >= 0:
                state_ready_row = pl.cast(state_ready_row_raw, pl.INDEX)
                state_ready_sample = pl.read(compress_state_flat, [state_ready_row, 0])
                state_ready_bit = pl.cast(state_ready_sample == state_ready_sample, pl.INT32)
                ready_bit = ready_bit * state_ready_bit
            cmp_ready_row_raw = pl.read(cmp_slot_mapping, [ready_t])
            if cmp_ready_row_raw >= 0:
                cmp_ready_row = pl.cast(cmp_ready_row_raw, pl.INDEX)
                cmp_ready_tile = pl.load(cmp_cache_flat, [cmp_ready_row, 0], [1, 16])
                cmp_ready_bits = pl.reinterpret_view(cmp_ready_tile, pl.INT16)
                cmp_ready_sample = pl.tile.read(cmp_ready_bits, [0, 0])
                cmp_ready_value = pl.cast(cmp_ready_sample, pl.INT32)
                ready_bit = ready_bit + cmp_ready_value
        pl.write(cache_ready_fence, [0], ready_bit)

    o_proj_weight_dep = pl.system.task_dummy(deps=[])
    attn_out = pl.create_tensor([t_dim, D], dtype=pl.BF16)
    with pl.spmd(t_dim, name_hint="prefill_hca_pad_output_init") as pad_output_tid:
        pad_t = pl.tile.get_block_idx()
        if pl.read(local_request_ids, [pad_t]) < 0:
            attn_out[pad_t : pad_t + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    request_dep = pl.system.task_dummy(deps=[cache_ready_dep, pad_output_tid])
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    for request in pl.range(request_count):
        request_start = pl.cast(pl.read(query_start_loc, [request]), pl.INDEX)
        request_end = pl.cast(pl.read(query_start_loc, [request + 1]), pl.INDEX)
        request_rows = request_end - request_start
        if request_rows > 0:
            request_dep = hca_streaming_attn_physical(
                q,
                kv_cache, swa_indices,
                cmp_kv, cmp_block_table[request],
                position_ids, attn_sink,
                freqs_cos, freqs_sin,
                wo_a, wo_b, wo_b_scale,
                attn_out,
                request_dep,
                o_proj_weight_dep,
                request_start,
                request_rows,
            )

    hc_post(attn_out, x_hc, post, comb, x_out)
    return x_out


@pl.jit
def prefill_attention_hca_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[T_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    x_hc.bind_dynamic(0, T_DYN)
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    local_request_ids.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    ori_block_table.bind_dynamic(0, REQUESTS_DYN)
    cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    prefill_attention_hca(
        x_hc,
        query_start_loc,
        local_request_ids,
        hc_attn_fn,
        hc_attn_scale,
        hc_attn_base,
        attn_norm_w,
        wq_a,
        wq_b,
        wq_b_scale,
        wkv,
        gamma_cq,
        gamma_ckv,
        freqs_cos,
        freqs_sin,
        cmp_freqs_cos,
        cmp_freqs_sin,
        cmp_wkv,
        cmp_wgate,
        cmp_ape,
        cmp_norm_w,
        compress_state,
        compress_state_block_table,
        kv_cache,
        ori_slot_mapping,
        ori_block_table,
        cmp_kv,
        cmp_block_table,
        position_ids,
        cmp_slot_mapping,
        state_slot_mapping,
        attn_sink,
        wo_a,
        wo_b,
        wo_b_scale,
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

    from utils import cache_row_from_table

    token_count = tensors["x_hc"].shape[0]
    x_hc_flat = tensors["x_hc"].view(token_count, HC_MULT, D)
    x_mixed = torch.zeros(token_count, D, dtype=torch.bfloat16)
    post = torch.zeros(token_count, HC_MULT, dtype=torch.float32)
    comb = torch.zeros(token_count, HC_MULT * HC_MULT, dtype=torch.float32)
    golden_hc_pre(
        {
            "x": x_hc_flat,
            "hc_fn": tensors["hc_attn_fn"],
            "hc_scale": tensors["hc_attn_scale"],
            "hc_base": tensors["hc_attn_base"],
            "x_mixed": x_mixed,
            "post": post,
            "comb": comb,
        }
    )

    q = torch.zeros(token_count, H, HEAD_DIM, dtype=torch.bfloat16)
    kv = torch.zeros(token_count, HEAD_DIM, dtype=torch.bfloat16)
    qr = torch.zeros(token_count, Q_LORA, dtype=torch.int8)
    qr_scale = torch.zeros(token_count, 1, dtype=torch.float32)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    rope_cos_t = tensors["freqs_cos"]
    rope_sin_t = tensors["freqs_sin"]
    golden_qkv_proj_rope(
        {
            "x": x_normed.view(token_count, D),
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
        }
    )

    ori_kv = tensors["kv_cache"]
    ori_kv_flat = ori_kv.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for t in range(token_count):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            ori_kv_flat[dst_row, :] = kv[t]

    cmp_kv = tensors["cmp_kv"]
    query_start_loc = tensors["query_start_loc"]
    for request in range(query_start_loc.numel() - 1):
        request_start = int(query_start_loc[request].item())
        request_end = int(query_start_loc[request + 1].item())
        if request_end <= request_start:
            continue
        request_rows = slice(request_start, request_end)
        golden_prefill_compressor_ratio128(
            {
                "x": x_normed[request_rows].view(request_end - request_start, D),
                "compress_state": tensors["compress_state"],
                "compress_state_block_table": tensors["compress_state_block_table"][request : request + 1],
                "wkv": tensors["cmp_wkv"],
                "wgate": tensors["cmp_wgate"],
                "ape": tensors["cmp_ape"],
                "norm_w": tensors["cmp_norm_w"],
                "cmp_freqs_cos": tensors["cmp_freqs_cos"][request_rows],
                "cmp_freqs_sin": tensors["cmp_freqs_sin"][request_rows],
                "cmp_kv": cmp_kv,
                "position_ids": tensors["position_ids"][request_rows],
                "cmp_slot_mapping": tensors["cmp_slot_mapping"][request_rows],
                "state_slot_mapping": tensors["state_slot_mapping"][request_rows],
            }
        )

    def build_sparse_metadata():
        swa_idx = torch.full((token_count, WIN), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        request_ids = tensors["local_request_ids"]
        active = request_ids >= 0
        max_position = int(pos[active].max().item()) if active.any() else -1
        max_visible_cmp = min((max_position + 1) // COMPRESS_RATIO, SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE)
        cmp_idx = torch.full((token_count, max(1, max_visible_cmp)), -1, dtype=torch.int32)
        cmp_cap = SPARSE_CMP_MAX_BLOCKS * BLOCK_SIZE
        for t in range(token_count):
            request_id = int(request_ids[t].item())
            if request_id < 0:
                continue
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(tensors["ori_block_table"][request_id], key_abs)
                if row >= 0:
                    swa_idx[t, k] = row
            visible_cmp = min((abs_pos + 1) // COMPRESS_RATIO, max_visible_cmp, cmp_cap)
            if visible_cmp > 0:
                cmp_idx[t, :visible_cmp] = torch.arange(visible_cmp, dtype=torch.int32)
        return swa_idx, cmp_idx

    swa_indices, cmp_indices = build_sparse_metadata()
    attn_out = torch.zeros(token_count, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn(
        {
            "q": q,
            "ori_kv": ori_kv,
            "swa_indices": swa_indices,
            "cmp_kv": cmp_kv,
            "cmp_block_table": tensors["cmp_block_table"],
            "local_request_ids": tensors["local_request_ids"],
            "cmp_indices": cmp_indices,
            "attn_sink": tensors["attn_sink"],
            "freqs_cos": rope_cos_t,
            "freqs_sin": rope_sin_t,
            "wo_a": tensors["wo_a"],
            "wo_b": tensors["wo_b"],
            "wo_b_scale": tensors["wo_b_scale"],
            "attn_out": attn_out,
        }
    )

    y = torch.zeros(token_count, HC_MULT, D, dtype=torch.float32)
    golden_hc_post(
        {
            "x": attn_out.view(token_count, D),
            "residual": x_hc_flat,
            "post": post,
            "comb": comb,
            "y": y,
        }
    )
    tensors["x_out"][:] = y


@functools.lru_cache(maxsize=None)
def _state_block_table(max_blocks, physical_blocks):
    """Constant scrambled state block table [max_blocks]."""
    import torch

    blocks = torch.arange(max_blocks, dtype=torch.int32)
    return (blocks * 17 + 3) % physical_blocks


def build_tensor_specs(start_pos: int = START_POS, token_count: int = PREFILL_SEQ):
    import torch
    from golden import TensorSpec
    from utils import cache_row_from_table, quant_w_per_channel, token_local_rope

    # Single-request geometry: the physical token dimension is q_len.
    context_len = start_pos
    q_len = token_count
    if token_count <= 0 or token_count > PREFILL_GROUP_CAP:
        raise ValueError(f"token_count must be in [1, {PREFILL_GROUP_CAP}], got {token_count}")
    if context_len < 0:
        raise ValueError(f"context length must be non-negative, got {context_len}")
    max_position = context_len + q_len - 1
    if max_position >= MAX_SEQ_LEN:
        raise ValueError(f"start_pos + token_count must be <= {MAX_SEQ_LEN}, got {context_len + q_len}")

    def token_meta():
        local_pos = torch.arange(q_len, dtype=torch.int32)
        pos = torch.arange(context_len, context_len + q_len, dtype=torch.int32)
        return local_pos, pos

    _, rope_positions = token_meta()
    shared_freqs_cos, shared_freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, rope_positions,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )
    cmp_positions = torch.where(
        (rope_positions + 1) % COMPRESS_RATIO == 0,
        rope_positions - (COMPRESS_RATIO - 1),
        torch.zeros_like(rope_positions),
    )
    shared_cmp_freqs_cos, shared_cmp_freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, cmp_positions,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )

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
        return torch.empty(token_count, HC_MULT, D).uniform_(-1, 1)

    def init_query_start_loc():
        return torch.tensor([0, token_count], dtype=torch.int32)

    def init_local_request_ids():
        return torch.zeros(token_count, dtype=torch.int32)

    # Real layer-9 (HCA, ratio-128) hc_attn scale/base, fn synthetic at real magnitude. A synthetic
    # scale=0.5/base=0 cancels attn_out and the hc residual to near-zero in x_out, where W8A8 noise
    # blows up the relative tail.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.0495

    def init_hc_attn_scale():
        return torch.tensor([0.079046, 0.04213, 0.121901])

    def init_hc_attn_base():
        return torch.tensor(
            [
                -3.3004,
                2.5553,
                -2.2787,
                -3.4925,
                -3.8197,
                -3.4161,
                -2.7144,
                -2.9181,
                2.362,
                -2.4746,
                -2.1352,
                -3.2216,
                -4.474,
                2.2488,
                -2.1053,
                -3.1675,
                -2.8362,
                -1.9042,
                2.0432,
                -3.062,
                -2.7902,
                -3.0908,
                -3.002,
                3.1161,
            ]
        )

    def init_attn_norm_w():
        return torch.ones(D)

    def init_wq_a():
        return (torch.rand(D, Q_LORA) - 0.5) * D**-0.5

    def init_wq_b():
        return (torch.rand(Q_LORA, H * HEAD_DIM) - 0.5) * Q_LORA**-0.5

    def init_wkv():
        return (torch.rand(D, HEAD_DIM) - 0.5) * D**-0.5

    def init_gamma_cq():
        return torch.ones(Q_LORA)

    def init_gamma_ckv():
        return torch.ones(HEAD_DIM)

    def init_freqs_cos():
        return shared_freqs_cos.clone()

    def init_freqs_sin():
        return shared_freqs_sin.clone()

    def init_cmp_freqs_cos():
        return shared_cmp_freqs_cos.clone()

    def init_cmp_freqs_sin():
        return shared_cmp_freqs_sin.clone()

    # Quant-faithful HCA (ratio-128) main compressor fixtures (mean l7/l9 of extract_weights_flash):
    # zero-mean Gaussian BF16 weights at the measured std; RMSNorm gamma near the measured mean.
    def init_cmp_wkv():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0240

    def init_cmp_wgate():
        return torch.randn(MAIN_OUT_DIM, D) * 0.0309

    def init_cmp_ape():
        return torch.randn(COMPRESS_RATIO, MAIN_OUT_DIM) * 0.0332

    def init_cmp_norm_w():
        return (
            0.0982
            + torch.randn(
                HEAD_DIM,
            )
            * 0.0539
        )

    state_table = _state_block_table(HCA_STATE_MAX_BLOCKS, HCA_STATE_PHYSICAL_BLOCKS)

    def init_compress_state_block_table():
        return state_table.clone().unsqueeze(0)

    def state_row(abs_pos):
        if abs_pos < 0 or abs_pos >= MAX_SEQ_LEN:
            return -1
        block = abs_pos // HCA_STATE_BLOCK_SIZE
        intra = abs_pos % HCA_STATE_BLOCK_SIZE
        return int(state_table[block].item()) * HCA_STATE_BLOCK_SIZE + intra

    def init_compress_state():
        state = torch.zeros(HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM)
        flat = state.view(-1, MAIN_COMPRESS_STATE_DIM)
        for abs_pos in range(max(0, context_len - COMPRESS_RATIO), context_len):
            row = state_row(abs_pos)
            if row >= 0:
                flat[row] = (
                    torch.rand(
                        MAIN_COMPRESS_STATE_DIM,
                    )
                    - 0.5
                ) * 0.05
        return state

    def init_kv_cache():
        cache = torch.zeros(HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_ori_block_table()[0]
        if context_len > 0:
            prefix_start = max(0, context_len - WIN)
            prefix = ((torch.rand(context_len - prefix_start, HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)
            for pos_i in range(prefix_start, context_len):
                row = cache_row_from_table(table, pos_i)
                if row >= 0:
                    cache_flat[row] = prefix[pos_i - prefix_start]
        return cache

    def init_ori_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        local_pos, _ = token_meta()
        table = init_ori_block_table()[0]
        for t in range(token_count):
            logical_pos = context_len + int(local_pos[t].item())
            mapping[t] = cache_row_from_table(table, logical_pos)
        return mapping

    def init_ori_block_table():
        table = torch.full((SPARSE_ORI_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(SPARSE_ORI_MAX_BLOCKS):
            table[block] = block % HCA_ORI_BLOCK_NUM
        return table.unsqueeze(0)

    def init_cmp_kv():
        cache = torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(HCA_CMP_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_cmp_block_table()[0]
        completed = context_len // COMPRESS_RATIO
        if completed > 0:
            prefix_cmp = ((torch.rand(completed, HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)
            for cmp_slot in range(completed):
                row = cache_row_from_table(table, cmp_slot)
                if row >= 0:
                    cache_flat[row] = prefix_cmp[cmp_slot]
        return cache

    def init_cmp_block_table():
        table = torch.full((SPARSE_CMP_MAX_BLOCKS,), -1, dtype=torch.int32)
        for block in range(min(SPARSE_CMP_MAX_BLOCKS, HCA_CMP_BLOCK_NUM)):
            table[block] = block
        return table.unsqueeze(0)

    def init_position_ids():
        return token_meta()[1]

    def init_cmp_slot_mapping():
        out = torch.full((token_count,), -1, dtype=torch.int64)
        table = init_cmp_block_table()[0]
        records = cmp_write_records()
        for token_id, cmp_slot in records:
            out[token_id] = cache_row_from_table(table, cmp_slot)
        return out

    def init_state_slot_mapping():
        mapping = torch.full((token_count,), -1, dtype=torch.int64)
        _, pos = token_meta()
        for t in range(token_count):
            mapping[t] = state_row(int(pos[t].item()))
        return mapping

    def init_attn_sink():
        return torch.zeros(H)

    def init_wo_a():
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN**-0.5

    def init_wo_b():
        return (torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [token_count, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("query_start_loc", [2], torch.int32, init_value=init_query_start_loc),
        TensorSpec("local_request_ids", [token_count], torch.int32, init_value=init_local_request_ids),
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
        TensorSpec("freqs_cos", [token_count, ROPE_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [token_count, ROPE_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("cmp_freqs_cos", [token_count, ROPE_DIM], torch.bfloat16, init_value=init_cmp_freqs_cos),
        TensorSpec("cmp_freqs_sin", [token_count, ROPE_DIM], torch.bfloat16, init_value=init_cmp_freqs_sin),
        TensorSpec("cmp_wkv", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wkv),
        TensorSpec("cmp_wgate", [MAIN_OUT_DIM, D], torch.bfloat16, init_value=init_cmp_wgate),
        TensorSpec("cmp_ape", [COMPRESS_RATIO, MAIN_OUT_DIM], torch.float32, init_value=init_cmp_ape),
        TensorSpec("cmp_norm_w", [HEAD_DIM], torch.bfloat16, init_value=init_cmp_norm_w),
        TensorSpec(
            "compress_state",
            [HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM],
            torch.float32,
            init_value=init_compress_state,
        ),
        TensorSpec(
            "compress_state_block_table",
            [1, HCA_STATE_MAX_BLOCKS],
            torch.int32,
            init_value=init_compress_state_block_table,
        ),
        TensorSpec(
            "kv_cache",
            [HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_kv_cache,
        ),
        TensorSpec("ori_slot_mapping", [token_count], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("ori_block_table", [1, SPARSE_ORI_MAX_BLOCKS], torch.int32, init_value=init_ori_block_table),
        TensorSpec(
            "cmp_kv",
            [HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            torch.bfloat16,
            init_value=init_cmp_kv,
        ),
        TensorSpec("cmp_block_table", [1, SPARSE_CMP_MAX_BLOCKS], torch.int32, init_value=init_cmp_block_table),
        TensorSpec("position_ids", [token_count], torch.int32, init_value=init_position_ids),
        TensorSpec("cmp_slot_mapping", [token_count], torch.int64, init_value=init_cmp_slot_mapping),
        TensorSpec("state_slot_mapping", [token_count], torch.int64, init_value=init_state_slot_mapping),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [token_count, HC_MULT, D], torch.float32),
    ]


@pl.jit.inline
def prefill_attention_hca_cp_core(
    x_normed_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    x_normed_full: pl.Tensor[[CP_KV_T_DYN, D], pl.BF16],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    freqs_cos_local: pl.Tensor[[CP_Q_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_local: pl.Tensor[[CP_Q_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_cos_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin_full: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out_local: pl.Tensor[[CP_Q_T_DYN, D], pl.BF16],
    late_dep: pl.Scalar[pl.TASK_ID],
    o_proj_weight_dep: pl.Scalar[pl.TASK_ID],
):
    """HCA attention body: local Q/o_proj and replicated full KV/compressor."""
    q_dim = pl.tensor.dim(x_normed_local, 0)
    kv_dim = pl.tensor.dim(x_normed_full, 0)

    q_cos_il = pl.create_tensor([q_dim, ROPE_DIM], dtype=pl.FP32)
    q_sin_signed = pl.create_tensor([q_dim, ROPE_DIM], dtype=pl.FP32)
    q_swap_idx = pl.create_tensor([q_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos_local, freqs_sin_local, q_cos_il, q_sin_signed, q_swap_idx)

    q = pl.create_tensor([q_dim, H, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([q_dim, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([q_dim, 1], dtype=pl.FP32)
    q_proj_rope(x_normed_local, wq_a, wq_b, wq_b_scale, gamma_cq, q_cos_il, q_sin_signed, q_swap_idx, q, qr, qr_scale)

    kv_cos_il = pl.create_tensor([kv_dim, ROPE_DIM], dtype=pl.FP32)
    kv_sin_signed = pl.create_tensor([kv_dim, ROPE_DIM], dtype=pl.FP32)
    kv_swap_idx = pl.create_tensor([kv_dim, ROPE_DIM], dtype=pl.INT32)
    rope_prepare(freqs_cos_full, freqs_sin_full, kv_cos_il, kv_sin_signed, kv_swap_idx)

    kv_full = pl.create_tensor([kv_dim, HEAD_DIM], dtype=pl.BF16)
    kv_proj_rope(x_normed_full, wkv, gamma_ckv, kv_cos_il, kv_sin_signed, kv_swap_idx, kv_full, late_dep)

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    ori_cache_rows = ori_block_num * BLOCK_SIZE
    kv_cache_flat = pl.reshape(kv_cache, [ori_cache_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cp_cache_write") as ori_cache_write_tid:
        for write_t in pl.range(kv_dim):
            write_row_raw = pl.read(ori_slot_mapping_full, [write_t])
            if write_row_raw >= 0:
                write_row = pl.cast(write_row_raw, pl.INDEX)
                kv_cache_flat[write_row : write_row + 1, :] = kv_full[write_t : write_t + 1, :]

    prefill_compressor_ratio128(
        x_normed_full,
        query_start_loc,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_freqs_cos_full, cmp_freqs_sin_full,
        cmp_kv,
        position_ids_full, cmp_slot_mapping_full, state_slot_mapping_full,
    )

    swa_indices = pl.create_tensor([q_dim, WIN], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cp_swa_indices") as swa_indices_tid:
        for idx_t in pl.range(q_dim):
            swa_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            request_id = pl.read(local_request_ids, [idx_t])
            if request_id >= 0:
                abs_pos = pl.read(position_ids_local, [idx_t])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(ori_block_table, [request_id, pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(swa_row, [0, win_col], row)
            swa_indices[idx_t : idx_t + 1, 0:WIN] = swa_row

    # Streaming-attention input publication fence.
    cmp_cache_rows = pl.tensor.dim(cmp_kv, 0) * BLOCK_SIZE
    state_rows = pl.tensor.dim(compress_state, 0) * HCA_STATE_BLOCK_SIZE
    cmp_cache_flat = pl.reshape(cmp_kv, [cmp_cache_rows, HEAD_DIM])
    compress_state_flat = pl.reshape(compress_state, [state_rows, MAIN_COMPRESS_STATE_DIM])
    q_ready_flat = pl.reshape(q, [q_dim * H, HEAD_DIM])
    cache_ready_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="prefill_hca_cp_cache_ready",
        deps=[ori_cache_write_tid, swa_indices_tid],
    ) as cache_ready_dep:
        ready_bit = pl.cast(1, pl.INT32)
        for ready_t in pl.range(kv_dim):
            if ready_t < q_dim:
                q_ready_tile = pl.load(q_ready_flat, [ready_t * H, 0], [1, 16])
                q_ready_bits = pl.reinterpret_view(q_ready_tile, pl.INT16)
                q_ready_sample = pl.tile.read(q_ready_bits, [0, 0])
                ready_bit = ready_bit + pl.cast(q_ready_sample, pl.INT32)
            state_ready_row_raw = pl.read(state_slot_mapping_full, [ready_t])
            if state_ready_row_raw >= 0:
                state_ready_row = pl.cast(state_ready_row_raw, pl.INDEX)
                state_ready_sample = pl.read(compress_state_flat, [state_ready_row, 0])
                state_ready_bit = pl.cast(state_ready_sample == state_ready_sample, pl.INT32)
                ready_bit = ready_bit * state_ready_bit
            cmp_ready_row_raw = pl.read(cmp_slot_mapping_full, [ready_t])
            if cmp_ready_row_raw >= 0:
                cmp_ready_row = pl.cast(cmp_ready_row_raw, pl.INDEX)
                cmp_ready_tile = pl.load(cmp_cache_flat, [cmp_ready_row, 0], [1, 16])
                cmp_ready_bits = pl.reinterpret_view(cmp_ready_tile, pl.INT16)
                cmp_ready_sample = pl.tile.read(cmp_ready_bits, [0, 0])
                cmp_ready_value = pl.cast(cmp_ready_sample, pl.INT32)
                ready_bit = ready_bit + cmp_ready_value
        pl.write(cache_ready_fence, [0], ready_bit)

    # Per-request HCA streaming over rank-local packed query intervals.
    with pl.spmd(q_dim, name_hint="prefill_hca_cp_pad_output_init") as pad_output_tid:
        pad_t = pl.tile.get_block_idx()
        if pl.read(local_request_ids, [pad_t]) < 0:
            attn_out_local[pad_t : pad_t + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
    request_dep = pl.system.task_dummy(deps=[cache_ready_dep, pad_output_tid])
    request_count = pl.tensor.dim(query_start_loc, 0) - 1
    for request in pl.range(request_count):
        local_start = pl.cast(0, pl.INDEX)
        request_rows = pl.cast(0, pl.INDEX)
        for local_t in pl.range(q_dim):
            request_id = pl.read(local_request_ids, [local_t])
            if request_id == request:
                if request_rows == 0:
                    local_start = local_t
                request_rows = request_rows + 1
        if request_rows > 0:
            request_dep = hca_streaming_attn_physical(
                q,
                kv_cache, swa_indices,
                cmp_kv, cmp_block_table[request],
                position_ids_local, attn_sink,
                freqs_cos_local, freqs_sin_local,
                wo_a, wo_b, wo_b_scale,
                attn_out_local,
                request_dep,
                o_proj_weight_dep,
                local_start,
                request_rows,
            )
    return attn_out_local


@pl.jit.inline
def prefill_attention_hca_cp(
    x_hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a_local: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_local: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    wo_a_full: pl.Tensor[[O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], pl.BF16],
    wo_b_full: pl.Tensor[[O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], pl.INT8],
    x_out_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_order_fence: pl.Tensor[[1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    weight_epoch: pl.Scalar[pl.INT32],
):
    """DSA-CP HCA with replicated HC state at both layer boundaries."""
    o_proj_weight_dep = gather_o_proj_full_weights(
        wo_a_local, wo_b_local,
        wo_a_full, wo_b_full,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, weight_epoch,
    )
    q_dim = pl.tensor.dim(position_ids_local, 0)
    kv_dim = pl.tensor.dim(x_hc_full, 0)

    x_mixed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    post_full = pl.create_tensor([kv_dim, HC_MULT], dtype=pl.FP32)
    comb_full = pl.create_tensor([kv_dim, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc_full, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed_full, post_full, comb_full)

    x_normed_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed_full, attn_norm_w, x_normed_full)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    local_base = tp_rank * q_dim
    x_normed_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    freqs_cos_local = pl.create_tensor([q_dim, ROPE_DIM], dtype=pl.BF16)
    freqs_sin_local = pl.create_tensor([q_dim, ROPE_DIM], dtype=pl.BF16)
    for local_row in pl.spmd(q_dim, name_hint="prefill_hca_cp_query_slice"):
        query_row = pl.load(x_normed_full, [local_base + local_row, 0], [1, D], target_memory=pl.MemorySpace.Vec)
        pl.store(query_row, [local_row, 0], x_normed_local)
        query_cos = pl.load(
            freqs_cos, [local_base + local_row, 0], [1, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        query_sin = pl.load(
            freqs_sin, [local_base + local_row, 0], [1, ROPE_DIM],
            target_memory=pl.MemorySpace.Vec,
        )
        pl.store(query_cos, [local_row, 0], freqs_cos_local)
        pl.store(query_sin, [local_row, 0], freqs_sin_local)

    attn_out_local = pl.create_tensor([q_dim, D], dtype=pl.BF16)
    attn_out_local = prefill_attention_hca_cp_core(
        x_normed_local, x_normed_full,
        query_start_loc, local_request_ids,
        wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv,
        freqs_cos_local, freqs_sin_local,
        freqs_cos, freqs_sin,
        cmp_freqs_cos, cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, ori_slot_mapping_full, ori_block_table,
        cmp_kv, cmp_block_table,
        position_ids_local, position_ids_full,
        cmp_slot_mapping_full, state_slot_mapping_full,
        attn_sink,
        wo_a_full, wo_b_full, wo_b_scale,
        attn_out_local,
        late_dep, o_proj_weight_dep,
    )

    attn_out_full = pl.create_tensor([kv_dim, D], dtype=pl.BF16)
    attn_out_full, gather_signal = prefill_cp_token_allgather_step(
        attn_out_local, attn_out_full,
        gather_window, gather_signal,
        group_base, tp_rank,
    )

    hc_post(attn_out_full, x_hc_full, post_full, comb_full, x_out_full)
    return x_out_full, gather_signal


@pl.jit
def prefill_attention_hca_cp_test(
    x_hc_full: pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
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
    freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[CP_KV_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out_full: pl.Out[pl.Tensor[[CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[PREFILL_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_wo_a_window: pld.DistributedTensor[[O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], pl.BF16],
    o_proj_wo_b_window: pld.DistributedTensor[[O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], pl.INT8],
    o_proj_weight_ready: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_proj_weight_consumed: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
):
    """Run one DSA-CP rank's share of an HCA block."""
    x_hc_full.bind_dynamic(0, CP_KV_T_DYN)
    query_start_loc.bind_dynamic(0, QUERY_START_LOC_DYN)
    local_request_ids.bind_dynamic(0, CP_Q_T_DYN)
    compress_state.bind_dynamic(0, STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, REQUESTS_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CMP_BLOCK_NUM_DYN)
    ori_block_table.bind_dynamic(0, REQUESTS_DYN)
    cmp_block_table.bind_dynamic(0, REQUESTS_DYN)
    ori_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    freqs_cos.bind_dynamic(0, CP_KV_T_DYN)
    freqs_sin.bind_dynamic(0, CP_KV_T_DYN)
    cmp_freqs_cos.bind_dynamic(0, CP_KV_T_DYN)
    cmp_freqs_sin.bind_dynamic(0, CP_KV_T_DYN)
    position_ids_local.bind_dynamic(0, CP_Q_T_DYN)
    position_ids_full.bind_dynamic(0, CP_KV_T_DYN)
    cmp_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    state_slot_mapping_full.bind_dynamic(0, CP_KV_T_DYN)
    x_out_full.bind_dynamic(0, CP_KV_T_DYN)

    wo_a_full = pl.create_tensor([O_PROJ_SCRATCH_GROUPS, O_PROJ_SCRATCH_RANK, O_PROJ_SCRATCH_INPUT], dtype=pl.BF16)
    wo_b_full = pl.create_tensor([O_PROJ_SCRATCH_D, O_PROJ_SCRATCH_COLS], dtype=pl.INT8)
    o_proj_order_fence = pl.create_tensor([1], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_hca_o_proj_order_init"):
        pl.write(o_proj_order_fence, [0], pl.cast(0, pl.INT32))
    x_out_full, gather_signal = prefill_attention_hca_cp(
        x_hc_full,
        query_start_loc, local_request_ids,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_freqs_cos, cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, ori_slot_mapping_full, ori_block_table,
        cmp_kv, cmp_block_table,
        position_ids_local, position_ids_full,
        cmp_slot_mapping_full, state_slot_mapping_full,
        attn_sink, wo_a, wo_b, wo_b_scale,
        wo_a_full, wo_b_full,
        x_out_full,
        gather_window, gather_signal,
        o_proj_wo_a_window, o_proj_wo_b_window,
        o_proj_weight_ready, o_proj_weight_consumed,
        o_proj_order_fence,
        group_base, tp_rank, pl.const(1, pl.INT32),
    )
    return x_out_full


@pl.jit.host
def l3_prefill_attention_hca_cp(
    x_hc_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, HC_MULT, D], pl.FP32],
    query_start_loc: pl.Tensor[[TP_SIZE, QUERY_START_LOC_DYN], pl.INT32],
    local_request_ids: pl.Tensor[[TP_SIZE, CP_Q_T_DYN], pl.INT32],
    hc_attn_fn: pl.Tensor[[TP_SIZE, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[TP_SIZE, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[TP_SIZE, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[TP_SIZE, D], pl.BF16],
    wq_a: pl.Tensor[[TP_SIZE, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[TP_SIZE, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[TP_SIZE, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[TP_SIZE, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[TP_SIZE, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[TP_SIZE, CP_KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[TP_SIZE, MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[TP_SIZE, COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[TP_SIZE, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[[TP_SIZE, STATE_BLOCK_NUM_DYN, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM], pl.FP32]
    ],
    compress_state_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, HCA_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[TP_SIZE, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    ori_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    ori_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, SPARSE_ORI_MAX_BLOCKS], pl.INT32],
    cmp_kv: pl.InOut[pl.Tensor[[TP_SIZE, CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[TP_SIZE, REQUESTS_DYN, SPARSE_CMP_MAX_BLOCKS], pl.INT32],
    position_ids_local: pl.Tensor[[TP_SIZE, CP_Q_T_DYN], pl.INT32],
    position_ids_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT32],
    cmp_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    state_slot_mapping_full: pl.Tensor[[TP_SIZE, CP_KV_T_DYN], pl.INT64],
    attn_sink: pl.Tensor[[TP_SIZE, H], pl.FP32],
    wo_a: pl.Tensor[[TP_SIZE, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[TP_SIZE, D, O_PROJ_LOCAL_COLS], pl.INT8],
    wo_b_scale: pl.Tensor[[TP_SIZE, D], pl.FP32],
    x_out_full: pl.Out[pl.Tensor[[TP_SIZE, CP_KV_T_DYN, HC_MULT, D], pl.FP32]],
):
    """Launch one DSA-CP HCA block per rank."""
    x_hc_full.bind_dynamic(1, CP_KV_T_DYN)
    query_start_loc.bind_dynamic(1, QUERY_START_LOC_DYN)
    local_request_ids.bind_dynamic(1, CP_Q_T_DYN)
    compress_state.bind_dynamic(1, STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, REQUESTS_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, CMP_BLOCK_NUM_DYN)
    ori_block_table.bind_dynamic(1, REQUESTS_DYN)
    cmp_block_table.bind_dynamic(1, REQUESTS_DYN)
    ori_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    freqs_cos.bind_dynamic(1, CP_KV_T_DYN)
    freqs_sin.bind_dynamic(1, CP_KV_T_DYN)
    cmp_freqs_cos.bind_dynamic(1, CP_KV_T_DYN)
    cmp_freqs_sin.bind_dynamic(1, CP_KV_T_DYN)
    position_ids_local.bind_dynamic(1, CP_Q_T_DYN)
    position_ids_full.bind_dynamic(1, CP_KV_T_DYN)
    cmp_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    state_slot_mapping_full.bind_dynamic(1, CP_KV_T_DYN)
    x_out_full.bind_dynamic(1, CP_KV_T_DYN)

    gather_window_buf = pld.alloc_window_buffer([PREFILL_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_wo_a_window_buf = pld.alloc_window_buffer([O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16)
    o_proj_wo_b_window_buf = pld.alloc_window_buffer([O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8)
    o_proj_weight_ready_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_proj_weight_consumed_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [PREFILL_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_wo_a_window = pld.window(
            o_proj_wo_a_window_buf, [O_PROJ_WO_A_WINDOW_ROWS, O_PROJ_WO_A_WINDOW_COLS], dtype=pl.BF16
        )
        o_proj_wo_b_window = pld.window(
            o_proj_wo_b_window_buf, [O_PROJ_WO_B_WINDOW_ROWS, O_PROJ_WO_B_WINDOW_COLS], dtype=pl.INT8
        )
        o_proj_weight_ready = pld.window(o_proj_weight_ready_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_proj_weight_consumed = pld.window(o_proj_weight_consumed_buf, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_attention_hca_cp_test(
            x_hc_full[rank],
            query_start_loc[rank], local_request_ids[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            cmp_freqs_cos[rank], cmp_freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            compress_state[rank], compress_state_block_table[rank],
            kv_cache[rank], ori_slot_mapping_full[rank], ori_block_table[rank],
            cmp_kv[rank], cmp_block_table[rank],
            position_ids_local[rank], position_ids_full[rank],
            cmp_slot_mapping_full[rank], state_slot_mapping_full[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            x_out_full[rank],
            gather_window, gather_signal,
            o_proj_wo_a_window, o_proj_wo_b_window,
            o_proj_weight_ready, o_proj_weight_consumed,
            0, rank,
            device=rank,
        )



def build_cp_tensor_specs(
    start_pos: int = START_POS,
    token_count: int = PREFILL_SEQ,
    tp_size: int = TP_SIZE,
):
    """Replicate layer-boundary state and full inputs while sharding query positions."""
    import torch

    from golden import TensorSpec
    from prefill_cp_token_allgather import cp_stack, materialize_spec

    if tp_size != TP_SIZE:
        raise ValueError(f"tp_size={tp_size} must match import-time TP_SIZE={TP_SIZE}")
    if token_count <= 0 or token_count > PREFILL_GROUP_CAP:
        raise ValueError(f"token_count must be in [1, {PREFILL_GROUP_CAP}], got {token_count}")
    if start_pos < 0 or start_pos + token_count > MAX_SEQ_LEN:
        raise ValueError(f"start_pos + token_count must be <= {MAX_SEQ_LEN}, got {start_pos + token_count}")
    if token_count % tp_size != 0:
        raise ValueError(f"token_count={token_count} must be a multiple of tp_size={tp_size}")
    local_t = token_count // tp_size

    # Token-major KV-side inputs: replicated whole, never sliced.
    full_names = ("ori_slot_mapping", "cmp_slot_mapping", "state_slot_mapping")

    specs = []
    for spec in build_tensor_specs(start_pos, token_count):
        value = materialize_spec(spec)
        if spec.name == "query_start_loc":
            specs.append(TensorSpec(
                "query_start_loc", [tp_size, 2], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "local_request_ids":
            specs.append(TensorSpec(
                "local_request_ids", [tp_size, local_t], spec.dtype,
                init_value=value.reshape(tp_size, local_t).contiguous(),
            ))
        elif spec.name == "x_hc":
            specs.append(TensorSpec(
                "x_hc_full", [tp_size, token_count, HC_MULT, D], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name in full_names:
            specs.append(TensorSpec(
                f"{spec.name}_full", [tp_size, token_count], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "position_ids":
            specs.append(TensorSpec(
                "position_ids_local", [tp_size, local_t], spec.dtype,
                init_value=value.reshape(tp_size, local_t).contiguous(),
            ))
            specs.append(TensorSpec(
                "position_ids_full", [tp_size, token_count], spec.dtype,
                init_value=cp_stack(value, tp_size),
            ))
        elif spec.name == "wo_a":
            shards = [value[rank * O_PROJ_LOCAL_GROUPS : (rank + 1) * O_PROJ_LOCAL_GROUPS] for rank in range(tp_size)]
            specs.append(TensorSpec(
                "wo_a", [tp_size, O_PROJ_LOCAL_GROUPS, O_LORA, O_GROUP_IN], spec.dtype,
                init_value=torch.stack(shards).contiguous(),
            ))
        elif spec.name == "wo_b":
            shards = [value[:, rank * O_PROJ_LOCAL_COLS : (rank + 1) * O_PROJ_LOCAL_COLS] for rank in range(tp_size)]
            specs.append(TensorSpec(
                "wo_b", [tp_size, D, O_PROJ_LOCAL_COLS], spec.dtype,
                init_value=torch.stack(shards).contiguous(),
            ))
        elif spec.name == "x_out":
            specs.append(TensorSpec("x_out_full", [tp_size, token_count, HC_MULT, D], spec.dtype))
        else:
            specs.append(TensorSpec(
                spec.name, [tp_size, *spec.shape], spec.dtype,
                init_value=cp_stack(value, tp_size), 
            ))
    return specs


def build_ragged2_cp_tensor_specs(tp_size: int = TP_SIZE):
    """Build the two-request rank-crossing HCA fixture from the B1 CP specs."""
    import torch

    from golden import TensorSpec
    from prefill_cp_token_allgather import cp_stack
    from utils import (
        block_table as make_block_table,
        cache_row_from_table,
        compressed_slot_mapping,
        ori_slot_mapping as make_ori_slot_mapping,
        state_slot_mapping as make_state_slot_mapping,
        token_local_rope,
    )

    if tp_size != 2:
        raise ValueError(f"ragged2 requires tp_size=2, got {tp_size}")

    token_count = 8
    request_starts = (126, 30)
    request_positions = (
        torch.tensor([126, 127, 128], dtype=torch.int32),
        torch.tensor([30, 31, 32, 33], dtype=torch.int32),
    )
    position_ids = torch.cat((*request_positions, torch.zeros(1, dtype=torch.int32)))
    query_start_loc = torch.tensor([0, 3, 7], dtype=torch.int32)
    request_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, -1], dtype=torch.int32)

    ori_block_table = make_block_table(batch=2, table_blocks=SPARSE_ORI_MAX_BLOCKS, physical_blocks=HCA_ORI_BLOCK_NUM)
    cmp_block_table = make_block_table(batch=2, table_blocks=SPARSE_CMP_MAX_BLOCKS, physical_blocks=HCA_CMP_BLOCK_NUM)
    compress_state_block_table = make_block_table(
        batch=2, table_blocks=HCA_STATE_MAX_BLOCKS,
        physical_blocks=HCA_STATE_BLOCK_NUM,
    )

    ori_mappings = []
    cmp_mappings = []
    state_mappings = []
    state_size = HCA_STATE_BLOCK_SIZE
    for request, positions in enumerate(request_positions):
        positions_2d = positions.unsqueeze(0)
        request_ori_table = ori_block_table[request : request + 1]
        request_cmp_table = cmp_block_table[request : request + 1]
        request_state_table = compress_state_block_table[request : request + 1]
        ori_mapping = make_ori_slot_mapping(positions_2d, request_ori_table)
        cmp_mapping = compressed_slot_mapping(positions_2d, request_cmp_table, compress_ratio=COMPRESS_RATIO)
        state_mapping = make_state_slot_mapping(positions_2d, request_state_table, state_block_size=state_size)
        ori_mappings.append(ori_mapping.reshape(-1))
        cmp_mappings.append(cmp_mapping.reshape(-1))
        state_mappings.append(state_mapping.reshape(-1))
    pad_mapping = torch.full((1,), -1, dtype=torch.int64)
    ori_slot_mapping = torch.cat((*ori_mappings, pad_mapping))
    cmp_slot_mapping = torch.cat((*cmp_mappings, pad_mapping))
    state_slot_mapping = torch.cat((*state_mappings, pad_mapping))

    kv_cache = torch.zeros(HCA_ORI_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    kv_cache_flat = kv_cache.view(HCA_ORI_BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
    for request, start_pos in enumerate(request_starts):
        for position in range(max(0, start_pos - WIN), start_pos):
            row = cache_row_from_table(ori_block_table[request], position)
            kv_cache_flat[row] = ((torch.rand(HEAD_DIM) - 0.5) * 0.1).to(torch.bfloat16)

    cmp_kv = torch.zeros(HCA_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16)
    compress_state_shape = (HCA_STATE_BLOCK_NUM, HCA_STATE_BLOCK_SIZE, MAIN_COMPRESS_STATE_DIM)
    compress_state = torch.zeros(compress_state_shape, dtype=torch.float32)
    compress_state_flat = compress_state.view(-1, MAIN_COMPRESS_STATE_DIM)
    for request, start_pos in enumerate(request_starts):
        request_state_table = compress_state_block_table[request]
        for position in range(max(0, start_pos - COMPRESS_RATIO), start_pos):
            row = cache_row_from_table(request_state_table, position, block_size=state_size)
            compress_state_flat[row] = (torch.rand(MAIN_COMPRESS_STATE_DIM) - 0.5) * 0.05

    freqs_cos, freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, position_ids,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )
    cmp_positions = torch.where(
        (position_ids + 1) % COMPRESS_RATIO == 0,
        position_ids - (COMPRESS_RATIO - 1),
        torch.zeros_like(position_ids),
    )
    cmp_freqs_cos, cmp_freqs_sin = token_local_rope(
        M, COMPRESS_RATIO, cmp_positions,
        max_seq_len=MAX_SEQ_LEN, dtype=torch.bfloat16,
    )

    replacements = {
        "query_start_loc": cp_stack(query_start_loc, tp_size),
        "local_request_ids": request_ids.reshape(tp_size, token_count // tp_size).contiguous(),
        "freqs_cos": cp_stack(freqs_cos, tp_size),
        "freqs_sin": cp_stack(freqs_sin, tp_size),
        "cmp_freqs_cos": cp_stack(cmp_freqs_cos, tp_size),
        "cmp_freqs_sin": cp_stack(cmp_freqs_sin, tp_size),
        "compress_state": cp_stack(compress_state, tp_size),
        "compress_state_block_table": cp_stack(compress_state_block_table, tp_size),
        "kv_cache": cp_stack(kv_cache, tp_size),
        "ori_slot_mapping_full": cp_stack(ori_slot_mapping, tp_size),
        "ori_block_table": cp_stack(ori_block_table, tp_size),
        "cmp_kv": cp_stack(cmp_kv, tp_size),
        "cmp_block_table": cp_stack(cmp_block_table, tp_size),
        "position_ids_local": position_ids.reshape(tp_size, token_count // tp_size).contiguous(),
        "position_ids_full": cp_stack(position_ids, tp_size),
        "cmp_slot_mapping_full": cp_stack(cmp_slot_mapping, tp_size),
        "state_slot_mapping_full": cp_stack(state_slot_mapping, tp_size),
    }

    specs = []
    for spec in build_cp_tensor_specs(start_pos=0, token_count=token_count, tp_size=tp_size):
        value = replacements.get(spec.name)
        if value is None:
            specs.append(spec)
            continue
        replacement_spec = TensorSpec(
            spec.name, list(value.shape), spec.dtype, init_value=value,
            is_output=spec.is_output, resident=spec.resident,
        )
        specs.append(replacement_spec)
    return specs


def golden_prefill_attention_hca_cp(tensors):
    """Run the full-stream reference and replicate layer outputs and caches per rank."""
    import torch

    tp_size = tensors["x_hc_full"].shape[0]
    token_count = tensors["x_hc_full"].shape[1]

    full = {name: tensors[name][0] for name in (
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
        "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
        "compress_state_block_table", "ori_block_table", "cmp_block_table",
        "attn_sink", "wo_b_scale",
    )}
    full["wo_a"] = torch.cat([tensors["wo_a"][rank] for rank in range(tp_size)], dim=0)
    full["wo_b"] = torch.cat([tensors["wo_b"][rank] for rank in range(tp_size)], dim=1)
    full["x_hc"] = tensors["x_hc_full"][0]
    full["compress_state"] = tensors["compress_state"][0].clone()
    full["kv_cache"] = tensors["kv_cache"][0].clone()
    full["cmp_kv"] = tensors["cmp_kv"][0].clone()
    full["ori_slot_mapping"] = tensors["ori_slot_mapping_full"][0]
    full["cmp_slot_mapping"] = tensors["cmp_slot_mapping_full"][0]
    full["state_slot_mapping"] = tensors["state_slot_mapping_full"][0]
    full["position_ids"] = tensors["position_ids_full"][0]
    full["query_start_loc"] = tensors["query_start_loc"][0]
    full["local_request_ids"] = torch.cat(
        [tensors["local_request_ids"][rank] for rank in range(tp_size)]
    )
    full["x_out"] = torch.zeros(token_count, HC_MULT, D, dtype=torch.float32)

    golden_prefill_attention_hca(full)

    tensors["x_out_full"][:] = full["x_out"].unsqueeze(0).expand(tp_size, *full["x_out"].shape)
    for name in ("kv_cache", "cmp_kv", "compress_state"):
        tensors[name][:] = full[name].unsqueeze(0).expand(tp_size, *full[name].shape)


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser(
        description="Standalone DeepSeek V4 packed prefill HCA correctness test."
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"]
    )
    parser.add_argument(
        "-d", "--device", type=str, default=",".join(str(i) for i in range(TP_SIZE)),
        help="comma-separated rank group; one ID at --tp 1",
    )
    parser.add_argument(
        "--tp", type=int, default=TP_SIZE, help="rank-group size; must match the import-time --tp"
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS)
    parser.add_argument(
        "--token-count", "--num-tokens", dest="token_count", type=int, default=None,
        help=f"B1 physical query-token extent across the group; defaults to {PREFILL_SEQ}. ragged2 is fixed at 8.",
    )
    parser.add_argument(
        "--case", choices=["b1", "ragged2"], default="b1",
        help="Fixture case; ragged2 is the fixed two-request TP2 boundary case.",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    if args.tp != TP_SIZE:
        raise SystemExit(f"--tp={args.tp} does not match import-time TP_SIZE={TP_SIZE}")
    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) != TP_SIZE:
        parser.error(f"need exactly {TP_SIZE} devices, got {device_ids}")
    if args.case == "ragged2" and TP_SIZE != 2:
        parser.error("--case ragged2 requires --tp 2")
    if args.case == "ragged2" and args.start_pos != 0:
        parser.error("--case ragged2 has fixed request starts and requires --start-pos 0")
    if args.token_count is None:
        args.token_count = 8 if args.case == "ragged2" else PREFILL_SEQ
    if args.case == "ragged2" and args.token_count != 8:
        parser.error("--case ragged2 has a fixed physical extent and requires --token-count 8")
    if args.case == "b1" and args.token_count % TP_SIZE != 0:
        parser.error(f"--token-count must be a multiple of --tp={TP_SIZE}, got {args.token_count}")

    if TP_SIZE == 1:
        result = run(
            fn=prefill_attention_hca_test,
            specs=build_tensor_specs(args.start_pos, args.token_count),
            golden_fn=golden_prefill_attention_hca,
            compile_cfg=dict(dump_passes=args.dump_passes),
            runtime_cfg=dict(
                platform=args.platform,
                device_id=device_ids[0],
                enable_chip_swimlane=args.enable_chip_swimlane,
                enable_dep_gen=args.enable_dep_gen,
            ),
            compile_only=args.compile_only,
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out": ratio_reldiff(diff_thd=5e-3, pct_thd=0.005, max_diff_hd=1),
                "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
            },
        )
    else:
        from pypto.ir.distributed_compiled_program import DistributedConfig

        specs = (
            build_ragged2_cp_tensor_specs(TP_SIZE)
            if args.case == "ragged2"
            else build_cp_tensor_specs(args.start_pos, args.token_count, TP_SIZE)
        )
        result = run(
            fn=l3_prefill_attention_hca_cp,
            specs=specs,
            golden_fn=golden_prefill_attention_hca_cp,
            compile_cfg=dict(
                dump_passes=args.dump_passes,
                distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
            ),
            runtime_cfg=dict(platform=args.platform),
            compile_only=args.compile_only,
            rtol=1e-2,
            atol=1e-2,
            compare_fn={
                "x_out_full": ratio_reldiff(diff_thd=5e-3, pct_thd=0.005, max_diff_hd=1),
                "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "cmp_kv": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
                "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
            },
        )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
