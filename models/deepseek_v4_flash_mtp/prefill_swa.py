# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek V4 CP SWA prefill and the MTP cached-tail adapter."""
# ci: devices=2

import sys

# Standalone CI passes its borrowed device list without a --cp argument.
# Resolve fixture topology before importing modules that freeze CP shapes.
if __name__ == "__main__":
    import argparse

    fixture_parser = argparse.ArgumentParser(add_help=False)
    fixture_parser.add_argument("-d", "--device", default="0")
    fixture_parser.add_argument("--cp", type=int, default=None)
    fixture_args, _ = fixture_parser.parse_known_args()
    _run_cp_fixture = fixture_args.cp is not None or "," in fixture_args.device
    if fixture_args.cp is None and _run_cp_fixture:
        sys.argv += ["--cp", str(len(fixture_args.device.split(",")))]

import torch
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from prefill_cp_zigzag import (
    CP_CHOICES,
    CP_SIZE,
    CP_TAIL_WINDOW_ROWS,
    MAX_SEGMENT_TILES,
    ROW_TILE,
    cp_final_window_sources,
    cp_owner_part,
    cp_owner_rank,
    cp_reverse_index,
    cp_segment_layout,
)
from prefill_cp_exchange import _prefill_cp_hidden_tail_exchange_wave
from golden import TensorSpec
from qkv_proj_rope import build_tensor_specs as build_qkv_tensor_specs, rope_prepare
from prefill_sparse_attn import (
    BIAS_TOKEN_TILE,
    PREFILL_ATTN_BLOCKS,
    build_tensor_specs as build_sparse_attn_tensor_specs,
)
from utils import build_rope_tables

import pypto.language as pl

from config import (
    BLOCK_SIZE,
    FLASH as M,
    FP32_NEG_INF,
    INT8_AMAX_EPS,
    INT8_SCALE_MAX,
    PREFILL_BATCH,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_ORI_BLOCK_NUM,
    PREFILL_ORI_MAX_BLOCKS,
    PREFILL_SEQ,
)
from hc_post import golden_hc_post_prefill
from hc_pre import golden_hc_pre
from qkv_proj_rope import golden_qkv_proj_rope, materialize_rope_rows, prefill_attention_prolog, kv_proj_rope
from rmsnorm import golden_rms_norm
from prefill_sparse_attn import (
    PREFILL_ATTN_TILE,
    PREFILL_SPARSE_PAD,
    SPARSE_BIAS_COLS,
    VALID_BLOCK_MASK_COLS,
    golden_prefill_sparse_attn,
    prefill_staged_attention,
)


# Dynamic shape variables.
BLOCK_NUM_DYN = pl.dynamic("PREFILL_ORI_BLOCK_NUM_DYN")

# model config
B = PREFILL_BATCH
S = PREFILL_SEQ
T = B * S
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HEAD_DIM = ROPE_DIM
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

# paged KV cache. The ratio-0 path has only the sliding-window cache: one
# request, one window page, so block count / table length / per-request window
# block count all collapse to 1.
BLOCK_NUM = PREFILL_ORI_BLOCK_NUM
CMP_BLOCK_NUM = PREFILL_CMP_BLOCK_NUM
SPARSE_CMP_MAX_BLOCKS = PREFILL_CMP_MAX_BLOCKS
START_POS = 0

assert WIN == BLOCK_SIZE, "SWA prefill currently assumes one window page per batch"
assert S == WIN, "SWA overlay raw-index contract maps current suffix rows as WIN+t"


@pl.jit.inline
def prefill_mtp_attention_swa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    post = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rope_cos_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_t = pl.create_tensor([T, ROPE_HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(freqs_cos, freqs_sin, position_ids, num_tokens, rope_cos_t, rope_sin_t)

    # Reuse the shared prefill QKV/RoPE projection to stay aligned with decode.
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    cos_il = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    sin_signed = pl.create_tensor([T, ROPE_DIM], dtype=pl.FP32)
    swap_idx = pl.create_tensor([T, ROPE_DIM], dtype=pl.INT32)
    rms_tid = prefill_attention_prolog(
        x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, gamma_cq, rope_cos_t, rope_sin_t,
        x_normed, post, comb, cos_il, sin_signed, swap_idx, q, qr, qr_scale,
    )
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    kv_proj_rope(x_normed, wkv, gamma_ckv, cos_il, sin_signed, swap_idx, kv, late_dep)

    block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [block_num * BLOCK_SIZE, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_cache_write"):
        for write_t in pl.range(T):
            if write_t < num_tokens:
                write_row_raw = pl.read(ori_slot_mapping, [write_t])
                if write_row_raw >= 0:
                    write_row = pl.cast(write_row_raw, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, :] = kv[write_t : write_t + 1, :]

    swa_indices = pl.create_tensor([T, WIN], dtype=pl.INT32)
    valid_block_mask = pl.create_tensor([T, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefill_swa_window_indices"):
        for idx_t in pl.range(T):
            idx_row = pl.full([1, WIN], dtype=pl.INT32, value=-1)
            mask_row = pl.full([1, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
            if idx_t < num_tokens:
                abs_pos = pl.read(position_ids, [idx_t])
                window_valid = pl.min(pl.cast(WIN, pl.INT32), abs_pos + 1)
                key_start_abs = abs_pos + 1 - window_valid
                for win_col in pl.range(WIN):
                    win_col_i32 = pl.cast(win_col, pl.INT32)
                    if win_col_i32 < window_valid:
                        key_abs = key_start_abs + win_col_i32
                        blk_slot = key_abs // BLOCK_SIZE
                        blk = pl.read(block_table, [pl.cast(blk_slot, pl.INDEX)])
                        if blk >= 0:
                            row = pl.cast(blk * BLOCK_SIZE + (key_abs - blk_slot * BLOCK_SIZE), pl.INT32)
                            pl.write(idx_row, [0, win_col], row)
                            if win_col < SPARSE_BIAS_COLS:
                                pl.write(mask_row, [0, win_col // PREFILL_ATTN_TILE], pl.cast(1, pl.INT32))
            swa_indices = pl.assemble(swa_indices, idx_row, [idx_t, 0])
            valid_block_mask = pl.assemble(valid_block_mask, mask_row, [idx_t, 0])

    # Cache layout is local to this caller; attention math is shared with CP.
    sparse_kv = pl.create_tensor([T * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([T, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    with pl.spmd(T, name_hint="prefill_swa_stage_sources") as stage_tid:
        stage_t = pl.tile.get_block_idx()
        stage = pl.full([WIN, HEAD_DIM], dtype=pl.BF16, value=0.0)
        bias = pl.full([1, PREFILL_SPARSE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
        if stage_t < num_tokens:
            for stage_col in pl.range(WIN):
                source_row = pl.read(swa_indices, [stage_t, stage_col])
                if source_row >= 0:
                    source = pl.cast(source_row, pl.INDEX)
                    stage[stage_col:stage_col + 1, :] = kv_cache_flat[source:source + 1, :]
                    pl.write(bias, [0, stage_col], pl.cast(0.0, pl.FP32))
        stage_base = stage_t * PREFILL_SPARSE_PAD
        sparse_kv[stage_base:stage_base + WIN, :] = stage
        sparse_bias[stage_t:stage_t + 1, :] = bias
    prefill_staged_attention(
        q, sparse_kv, sparse_bias, valid_block_mask, attn_sink,
        rope_cos_t, rope_sin_t, wo_a, wo_b, wo_b_scale,
        x_hc, post, comb, x_out, num_tokens, stage_tid,
    )

    return kv_cache, x_out


@pl.jit
def prefill_mtp_attention_swa_test(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    block_table: pl.Tensor[[BLOCK_NUM], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
):
    prefill_mtp_attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_cache, block_table, ori_slot_mapping,
        position_ids,
        attn_sink, wo_a, wo_b, wo_b_scale,
        x_out, num_tokens,
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


def golden_prefill_attention_swa(tensors):
    """Torch reference for token-major packed SWA prefill."""
    import torch

    from utils import cache_row_from_table

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
    rope_cos_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    rope_sin_t = torch.zeros(T, ROPE_DIM, dtype=torch.bfloat16)
    x_normed = golden_rms_norm(x_mixed, tensors["attn_norm_w"])
    positions = tensors["position_ids"].to(torch.long)
    rope_cos_t = tensors["freqs_cos"].index_select(0, positions).contiguous()
    rope_sin_t = tensors["freqs_sin"].index_select(0, positions).contiguous()
    golden_qkv_proj_rope({
        "x": x_normed,
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

    kv_cache_in = tensors["kv_cache"].clone()
    kv_cache_flat = kv_cache_in.view(kv_cache_in.shape[0] * BLOCK_SIZE, HEAD_DIM)
    for t in range(num_tokens):
        dst_row = int(tensors["ori_slot_mapping"][t].item())
        if dst_row >= 0:
            kv_cache_flat[dst_row, :] = kv[t]

    def build_swa_metadata():
        idx = torch.full((T, WIN), -1, dtype=torch.int32)
        pos = tensors["position_ids"]
        table = tensors["block_table"]
        for t in range(num_tokens):
            abs_pos = int(pos[t].item())
            window_valid = min(WIN, abs_pos + 1)
            key_start_abs = abs_pos + 1 - window_valid
            for k, key_abs in enumerate(range(key_start_abs, abs_pos + 1)):
                row = cache_row_from_table(table, key_abs)
                if row >= 0:
                    idx[t, k] = row
        return idx

    attn_out = torch.zeros(T, D, dtype=torch.bfloat16)
    golden_prefill_sparse_attn({
        "q": q,
        "ori_kv": kv_cache_in,
        "swa_indices": build_swa_metadata(),
        "cmp_kv": torch.zeros(CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM, dtype=torch.bfloat16),
        "cmp_block_table": torch.zeros(SPARSE_CMP_MAX_BLOCKS, dtype=torch.int32),
        "cmp_indices": torch.full((T, IDX_TOPK), -1, dtype=torch.int32),
        "attn_sink": tensors["attn_sink"],
        "num_tokens": tensors["num_tokens"],
        "freqs_cos": rope_cos_t,
        "freqs_sin": rope_sin_t,
        "wo_a": tensors["wo_a"],
        "wo_b": tensors["wo_b"],
        "wo_b_scale": tensors["wo_b_scale"],
        "attn_out": attn_out,
    })

    tensors["kv_cache"][:] = kv_cache_in

    y = torch.zeros(T, HC_MULT, D, dtype=torch.float32)
    golden_hc_post_prefill({
        "x": attn_out.view(T, D),
        "residual": x_hc_flat,
        "post": post,
        "comb": comb,
        "y": y,
        "num_tokens": tensors["num_tokens"],
    })
    tensors["x_out"][:] = y


def build_tensor_specs(
    start_pos: int = START_POS,
    num_tokens: int = T,
):
    import torch
    from golden import ScalarSpec, TensorSpec
    from utils import build_rope_tables, cache_row_from_table, quant_w_per_channel

    shared_freqs_cos, shared_freqs_sin = build_rope_tables(M, 0, dtype=torch.bfloat16)

    # Single-request geometry: q_len = num_tokens (active prefix), context_len =
    # start_pos (absolute position base, a multiple of S=WIN under chunked prefill).
    context_len = start_pos
    q_len = num_tokens

    if num_tokens <= 0 or num_tokens > T:
        raise ValueError(f"num_tokens must be in [1, {T}], got {num_tokens}")
    max_position = context_len + q_len
    if context_len < 0:
        raise ValueError(f"context_len must be non-negative, got {context_len}")
    if max_position > MAX_SEQ_LEN:
        raise ValueError(f"position_ids exceed MAX_SEQ_LEN={MAX_SEQ_LEN}: got {max_position}")


    def token_pos():
        # Single-request absolute positions: pos[t] = context_len + local_idx
        # Padding rows keep their arange default; they are inactive.
        pos = torch.arange(T, dtype=torch.int32)
        for local_s in range(q_len):
            pos[local_s] = context_len + local_s
        return pos

    def init_x_hc():
        x = torch.empty(T, HC_MULT, D).uniform_(-1, 1)
        x[num_tokens:] = 0
        return x
    # Real layer-0 (SWA) hc_attn scale/base (fn synthetic at real magnitude). A synthetic
    # scale=0.5/base=0 leaves hc_pre post~=1 + near-uniform comb, cancelling attn_out and the
    # hc residual to near-zero in x_out where quant noise blows up the relative tail. Mirrors
    # decode_swa.
    def init_hc_attn_fn():
        return torch.randn(MIX_HC, HC_DIM) * 0.039
    def init_hc_attn_scale():
        return torch.tensor([2.076026, 0.018729, 0.245936])
    def init_hc_attn_base():
        return torch.tensor([
            3.9083, -2.0399, -2.2033, -2.017,
            -2.4443, -10.3158, -8.9943, -6.3581,
            9.8577, -9.5177, -24.8724, -22.8929,
            -21.545, 0.7791, -3.386, 1.1948,
            -20.9605, -0.7702, 1.4218, -4.8994,
            1.5177, -29.7663, -30.1413, -1.2413,
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
    def init_block_table():
        tbl = torch.full((BLOCK_NUM,), -1, dtype=torch.int32)
        for block in range(BLOCK_NUM):
            tbl[block] = block
        return tbl
    def init_kv_cache():
        cache = torch.zeros(BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM)
        cache_flat = cache.view(BLOCK_NUM * BLOCK_SIZE, HEAD_DIM)
        table = init_block_table()
        start = max(0, context_len - WIN)
        for abs_pos in range(start, context_len):
            row = cache_row_from_table(table, abs_pos)
            value = (torch.rand(HEAD_DIM) - 0.5) * 0.1
            if row >= 0:
                cache_flat[row] = value.to(torch.bfloat16)
        return cache
    def init_ori_slot_mapping():
        mapping = torch.full((T,), -1, dtype=torch.int64)
        pos = token_pos()
        table = init_block_table()
        for t in range(num_tokens):
            mapping[t] = cache_row_from_table(table, int(pos[t].item()))
        return mapping
    def init_position_ids():
        return token_pos()
    def init_attn_sink():
        return torch.zeros(H)
    def init_wo_a():
        return (torch.rand(O_GROUPS, O_LORA, O_GROUP_IN) - 0.5) * O_GROUP_IN ** -0.5
    def init_wo_b():
        return (torch.rand(D, O_GROUPS * O_LORA) - 0.5) * (O_GROUPS * O_LORA) ** -0.5

    wq_b_bf16 = init_wq_b().to(torch.bfloat16)
    wq_b_i8, wq_b_scale = _quant_w_per_output_channel(wq_b_bf16)
    wo_b_bf16 = init_wo_b().to(torch.bfloat16)
    wo_b_i8, wo_b_scale = quant_w_per_channel(wo_b_bf16)

    return [
        TensorSpec("x_hc", [T, HC_MULT, D], torch.float32, init_value=init_x_hc),
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
        TensorSpec("freqs_cos", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_cos),
        TensorSpec("freqs_sin", [MAX_SEQ_LEN, ROPE_HEAD_DIM], torch.bfloat16, init_value=init_freqs_sin),
        TensorSpec("kv_cache", [BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], torch.bfloat16, init_value=init_kv_cache),
        TensorSpec("block_table", [BLOCK_NUM], torch.int32, init_value=init_block_table),
        TensorSpec("ori_slot_mapping", [T], torch.int64, init_value=init_ori_slot_mapping),
        TensorSpec("position_ids", [T], torch.int32, init_value=init_position_ids),
        TensorSpec("attn_sink", [H], torch.float32, init_value=init_attn_sink),
        TensorSpec("wo_a", [O_GROUPS, O_LORA, O_GROUP_IN], torch.bfloat16, init_value=init_wo_a),
        TensorSpec("wo_b", [D, O_GROUPS * O_LORA], torch.int8, init_value=lambda: wo_b_i8),
        TensorSpec("wo_b_scale", [D], torch.float32, init_value=lambda: wo_b_scale),
        TensorSpec("x_out", [T, HC_MULT, D], torch.float32),
        ScalarSpec("num_tokens", torch.int32, num_tokens),
    ]


# model config


# CP layout
NUM_SEGMENTS = 2 * CP_SIZE
TAIL_ROWS = WIN
BLOCK_ROWS = BLOCK_SIZE
LOCAL_PARTS = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
ROWS_PER_AUGMENTED_PART = (MAX_SEGMENT_TILES + 1) * TAIL_ROWS
LOCAL_AUGMENTED_ROWS = LOCAL_PARTS * ROWS_PER_AUGMENTED_PART
SEGMENT_ROWS = MAX_SEGMENT_TILES * TAIL_ROWS

# Segment staging and projection scratch exceed the runtime's 256 MiB
# default output heap. Match the production MTP prefill allocation;
# this is a standalone L3 harness setting and does not change the model ABI.
PREFILL_CP_SWA_RING_HEAP = (1024 * 1024 * 1024,) * 4


ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_ROWS
RAW_BLOCKS_DYN = pl.dynamic("CP_SWA_RAW_BLOCKS_DYN")

# Sparse overlay rows.
OVERLAY_BASE = ORI_CACHE_ROWS
PRED_OVERLAY_ROWS = TAIL_ROWS
CUR_OVERLAY_ROWS = TAIL_ROWS
OVERLAY_ROWS = PRED_OVERLAY_ROWS + CUR_OVERLAY_ROWS
OVERLAY_SOURCES = 2

# Fixture logical-to-physical block mapping.
NUM_RING_BLOCKS = ORI_CACHE_ROWS // BLOCK_ROWS
IDENTITY_BLOCK_TABLE = torch.arange(NUM_RING_BLOCKS, dtype=torch.int32)


def ring_phys_row(abs_pos: int) -> int:
    ring_row = abs_pos % ORI_CACHE_ROWS
    block = ring_row // BLOCK_ROWS
    intra = ring_row % BLOCK_ROWS
    return int(IDENTITY_BLOCK_TABLE[block].item()) * BLOCK_ROWS + intra


def owner_segments(cp_size: int):
    """Return each rank's two logical segments."""
    table = [[-1, -1] for _ in range(cp_size)]
    for seg in range(2 * cp_size):
        rank = cp_owner_rank(seg, cp_size)
        part = cp_owner_part(seg, cp_size)
        table[rank][part] = seg
    return table


def pred_segment(segment: int) -> int:
    return segment - 1 if segment > 0 else -1


def active_tile(segment_len: int, tile: int) -> int:
    return max(0, min(TAIL_ROWS, segment_len - tile * TAIL_ROWS))


def tail_start(seg_start: int, seg_len: int) -> int:
    return seg_start + max(0, seg_len - TAIL_ROWS)


def build_metadata(cp_size: int = CP_SIZE, *, num_tokens: int | None = None, prefix: int = 0):
    """Build CP metadata with real lengths and fixed backing shapes.

    ``prefix`` is this request's already-resident token count from earlier
    chunks. Window keys below it are addressed as raw-cache rows rather than
    overlay rows, which is what makes the leaf read history out of the paged KV
    cache instead of the in-chunk overlay.
    """
    if num_tokens is None:
        num_tokens = 2 * cp_size * MAX_SEGMENT_TILES * TAIL_ROWS
    segment_span, starts, lengths = cp_segment_layout(num_tokens, cp_size, prefix=prefix)
    nseg = 2 * cp_size
    parts = owner_segments(cp_size)

    seg_starts_t = torch.tensor(starts, dtype=torch.int32)
    seg_lens_t = torch.tensor(lengths, dtype=torch.int32)
    segment_tail_positions = torch.full((nseg, TAIL_ROWS), -1, dtype=torch.int32)
    for segment in range(nseg):
        valid = min(TAIL_ROWS, lengths[segment])
        if valid > 0:
            tail_pos0 = tail_start(starts[segment], lengths[segment])
            segment_tail_positions[segment, :valid] = torch.arange(tail_pos0, tail_pos0 + valid, dtype=torch.int32)
    owner_segs_t = torch.tensor(parts, dtype=torch.int32)
    reverse_index_t = cp_reverse_index(cp_size).to(torch.int32)

    seg_active_lengths = torch.full((cp_size, LOCAL_PARTS), -1, dtype=torch.int32)
    predecessor_segments = torch.full((cp_size, LOCAL_PARTS), -1, dtype=torch.int32)
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            seg = parts[rank][part]
            seg_active_lengths[rank, part] = lengths[seg]
            predecessor_segments[rank, part] = pred_segment(seg)

    # Inactive query metadata.
    q_pos = torch.zeros((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS), dtype=torch.int32)
    q_req = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS), -1, dtype=torch.int32)
    ov_pos = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS), -1, dtype=torch.int32)
    ov_req = torch.full_like(ov_pos, -1)
    ov_len = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, 2), -1, dtype=torch.int32)
    swa = torch.full((cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN), -1, dtype=torch.int32)

    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            segment = parts[rank][part]
            seg_len = lengths[segment]
            for tile in range(MAX_SEGMENT_TILES):
                active = active_tile(seg_len, tile)
                tile_start = starts[segment] + tile * TAIL_ROWS

                if active > 0:
                    q_pos[rank, part, tile, :active] = torch.arange(tile_start, tile_start + active, dtype=torch.int32)
                    q_req[rank, part, tile, :active] = 0

                if tile == 0:
                    pred_seg = pred_segment(segment)
                    if pred_seg >= 0:
                        pred_len = min(TAIL_ROWS, lengths[pred_seg])
                        pred_start = tail_start(starts[pred_seg], lengths[pred_seg])
                    else:
                        pred_len = 0
                        pred_start = 0
                else:
                    pred_len = active_tile(seg_len, tile - 1)
                    pred_start = starts[segment] + (tile - 1) * TAIL_ROWS
                if pred_len > 0:
                    ov_pos[rank, part, tile, :pred_len] = torch.arange(
                        pred_start, pred_start + pred_len, dtype=torch.int32)
                    ov_req[rank, part, tile, :pred_len] = 0
                if active > 0:
                    ov_pos[rank, part, tile, PRED_OVERLAY_ROWS:PRED_OVERLAY_ROWS + active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32
                    )
                    ov_req[rank, part, tile, PRED_OVERLAY_ROWS:PRED_OVERLAY_ROWS + active] = 0
                ov_len[rank, part, tile, 0] = pred_len
                ov_len[rank, part, tile, 1] = active

                if active:
                    query_abs = tile_start + torch.arange(active, dtype=torch.int32)[:, None]
                    key_abs = query_abs - WIN + 1 + torch.arange(WIN, dtype=torch.int32)[None, :]
                    current = (key_abs >= tile_start) & (key_abs < tile_start + active)
                    previous = (key_abs >= pred_start) & (key_abs < pred_start + pred_len)
                    # Keys below the chunk base live in this request's paged KV
                    # cache, written by an earlier chunk. Address them as raw
                    # cache rows; the leaf takes its history branch on those.
                    history = (key_abs >= 0) & (key_abs < prefix)
                    ring = key_abs.clamp(min=0) % ORI_CACHE_ROWS
                    history_rows = (
                        IDENTITY_BLOCK_TABLE[ring // BLOCK_ROWS].to(torch.int32) * BLOCK_ROWS
                        + ring % BLOCK_ROWS
                    )
                    swa[rank, part, tile, :active] = torch.where(
                        current,
                        OVERLAY_BASE + PRED_OVERLAY_ROWS + key_abs - tile_start,
                        torch.where(
                            previous,
                            OVERLAY_BASE + key_abs - pred_start,
                            torch.where(history, history_rows, -1),
                        ),
                    )

    final_seg_src, final_row_src = cp_final_window_sources(lengths)
    final_seg_src = final_seg_src.to(torch.int32)
    final_row_src = final_row_src.to(torch.int32)
    total = sum(lengths)
    final_slot = torch.full((TAIL_ROWS,), -1, dtype=torch.int32)
    for row in range(TAIL_ROWS):
        abs_pos = prefix + total - TAIL_ROWS + row
        if abs_pos < prefix:
            continue
        final_slot[row] = ring_phys_row(abs_pos)

    tensors = {
        "segment_lens": seg_lens_t,
        "segment_starts": seg_starts_t,
        "segment_tail_positions": segment_tail_positions,
        "owner_segments": owner_segs_t,
        "reverse_index": reverse_index_t,
        "segment_active_lengths": seg_active_lengths,
        "predecessor_segments": predecessor_segments,
        "query_position_ids": q_pos,
        "query_token_to_request": q_req,
        "overlay_position_ids": ov_pos,
        "overlay_token_to_request": ov_req,
        "overlay_active_lengths": ov_len,
        "swa_indices": swa,
        "final_win_seg_src": final_seg_src,
        "final_win_row_src": final_row_src,
        "final_slot_mapping": final_slot,
    }
    ctx = {
        "cp_size": cp_size,
        "prefix": prefix,
        "segment_span": segment_span,
        "lengths": lengths,
        "starts": starts,
        "owner_segments": parts,
        "block_table": IDENTITY_BLOCK_TABLE,
    }
    return tensors, ctx


@pl.jit.inline
def _cp_swa_history_exchange(
    kv_cache: pl.Tensor[[RAW_BLOCKS_DYN, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16],
    history_slots: pl.Tensor[[TAIL_ROWS], pl.INT32],
    window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    history_raw: pl.Out[pl.Tensor[[TAIL_ROWS, HEAD_DIM], pl.BF16]],
    cache_owner: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    epoch: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Consume an owner raw-history phase before reusing the hidden-tail window."""
    raw_rows = pl.tensor.dim(kv_cache, 0) * BLOCK_ROWS
    raw_flat = pl.reshape(kv_cache, [raw_rows, HEAD_DIM])
    payload = pl.create_tensor([TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_swa_owner_history") as history_tid:
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(signal=consumed, offsets=[peer, 0], expected=epoch, cmp=pld.WaitCmp.Ge)
        if my_rank == cache_owner:
            for row in pl.range(TAIL_ROWS):
                payload[row:row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
                source = pl.read(history_slots, [row])
                if source >= 0 and source < raw_rows:
                    payload[row:row + 1, :] = raw_flat[source:source + 1, :]
            for peer in pl.range(CP_SIZE):
                pld.tensor.put(
                    dst=window, peer=peer, src=payload,
                    dst_offsets=[0, 0], src_offsets=[0, 0], shape=[TAIL_ROWS, HEAD_DIM],
                    chunk_rows=8, chunk_cols=HEAD_DIM, pipeline=True,
                )
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(target=ready, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(signal=ready, offsets=[peer, 0], expected=epoch + 1, cmp=pld.WaitCmp.Ge)
        for row in pl.range(0, TAIL_ROWS, 8):
            history_raw[row:row + 8, :] = window[row:row + 8, 0:HEAD_DIM]
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(target=consumed, peer=peer, offsets=[my_rank, 0], value=1, op=pld.NotifyOp.AtomicAdd)
    return history_tid


@pl.jit.inline
def _cp_swa_stage_sources(
    history_raw: pl.Tensor[[TAIL_ROWS, HEAD_DIM], pl.BF16],
    local_kv: pl.Tensor[[LOCAL_ROWS, HEAD_DIM], pl.BF16],
    predecessor_kv: pl.Tensor[[LOCAL_PARTS * TAIL_ROWS, HEAD_DIM], pl.BF16],
    query_positions: pl.Tensor[[LOCAL_ROWS], pl.INT32],
    query_requests: pl.Tensor[[LOCAL_ROWS], pl.INT32],
    overlay_positions: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_requests: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_ROWS, WIN], pl.INT32],
    sparse_kv: pl.Tensor[[SEGMENT_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[SEGMENT_ROWS, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[SEGMENT_ROWS, VALID_BLOCK_MASK_COLS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_SOURCES], pl.INT32],
    segment_offset: pl.Scalar[pl.INDEX],
    prior_dep: pl.Scalar[pl.TASK_ID],
):
    """Stage one logical segment from persistent/predecessor/current sources."""
    prefix = pl.read(segment_starts_t, [0])
    with pl.spmd((SEGMENT_ROWS // 2) * PREFILL_ATTN_BLOCKS, name_hint="gather_kv", deps=[prior_dep]) as gather_tid:
        block = pl.tile.get_block_idx()
        schedule = block // PREFILL_ATTN_BLOCKS
        sb = block - schedule * PREFILL_ATTN_BLOCKS
        token_block = (SEGMENT_ROWS // 2) - 1 - schedule
        t0 = token_block * 2
        k0 = sb * PREFILL_ATTN_TILE
        for dt in pl.range(2):
            stage_row = t0 + dt
            row = segment_offset + stage_row
            if row < LOCAL_ROWS:
                stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                out_base = stage_row * PREFILL_SPARSE_PAD + k0
                for ki in pl.range(PREFILL_ATTN_TILE):
                    col = k0 + ki
                    if col < WIN:
                        raw = pl.read(swa_indices, [row, col])
                        if raw >= 0:
                            q_abs = pl.read(query_positions, [row])
                            q_req = pl.read(query_requests, [row])
                            key_abs = q_abs - WIN + 1 + col
                            # Source kind follows the absolute key; physical page ids do
                            # not share the index space of the current-chunk overlays.
                            if key_abs < prefix:
                                history_row = key_abs - prefix + TAIL_ROWS
                                if history_row >= 0 and key_abs <= q_abs and q_req >= 0:
                                    stage[ki:ki + 1, :] = history_raw[history_row:history_row + 1, :]
                            elif raw < OVERLAY_BASE + OVERLAY_ROWS:
                                tile = row // TAIL_ROWS
                                ov_row = raw - OVERLAY_BASE
                                if ov_row >= PRED_OVERLAY_ROWS:
                                    source_kind = 1
                                    src_row = ov_row - PRED_OVERLAY_ROWS
                                else:
                                    source_kind = 0
                                    src_row = ov_row
                                ov_idx = src_row
                                if source_kind == 1:
                                    ov_idx = PRED_OVERLAY_ROWS + src_row
                                ov_active = pl.read(overlay_active_lengths, [tile, source_kind])
                                ov_abs = pl.read(overlay_positions, [tile, ov_idx])
                                ov_req = pl.read(overlay_requests, [tile, ov_idx])
                                if src_row >= 0 and src_row < ov_active:
                                    if ov_abs == key_abs and ov_abs <= q_abs and ov_req == q_req and ov_req >= 0:
                                        if source_kind == 1:
                                            src = tile * TAIL_ROWS + src_row
                                            stage[ki:ki + 1, :] = local_kv[src:src + 1, :]
                                        elif tile % MAX_SEGMENT_TILES == 0:
                                            part = tile // MAX_SEGMENT_TILES
                                            pred = pl.read(predecessor_segments, [part])
                                            if pred >= 0:
                                                src = part * TAIL_ROWS + src_row
                                                stage[ki:ki + 1, :] = predecessor_kv[src:src + 1, :]
                                        else:
                                            src = (tile - 1) * TAIL_ROWS + src_row
                                            stage[ki:ki + 1, :] = local_kv[src:src + 1, :]
                sparse_kv[out_base:out_base + PREFILL_ATTN_TILE, :] = stage
    with pl.spmd(SEGMENT_ROWS // BIAS_TOKEN_TILE, name_hint="build_bias", deps=[prior_dep]) as bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        source_t0 = segment_offset + bias_t0
        # PyPTO 0.60 removed orchestration-side ``create_tensor(init_value=...)``.
        # Seed the whole staged row in the writer kernel, then overwrite the
        # physical SWA columns below.  The remaining sparse blocks stay masked.
        sparse_bias[
            bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:PREFILL_SPARSE_PAD
        ] = pl.full([BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
        valid_block_mask[
            bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:VALID_BLOCK_MASK_COLS
        ] = pl.full([BIAS_TOKEN_TILE, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, value=0)
        bias_idx = pl.cast(swa_indices[source_t0:source_t0 + BIAS_TOKEN_TILE, 0:WIN], target_type=pl.FP32)
        flags = pl.minimum(pl.maximum(pl.add(bias_idx, 1.0), 0.0), 1.0)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN] = pl.mul(pl.sub(flags, 1.0), -FP32_NEG_INF)
    return gather_tid, bias_tid


@pl.jit.inline
def prefill_attention_swa(
    x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[RAW_BLOCKS_DYN, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]],
    history_slots: pl.Tensor[[TAIL_ROWS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]],
    completion_token: pl.Out[pl.Tensor[[NUM_LOCAL_TILES, 1, 8], pl.FP32]],
    cache_owner_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    tail_epoch: pl.Scalar[pl.INT32],
):
    """CP-SWA attention math (inline). Shared by the standalone rank child
    and the layer composition child. Inlining avoids child-in-child nesting
    (@pl.jit cannot call another @pl.jit).

    ``tail_epoch`` is the cross-layer tail-exchange communication epoch
    (number of preceding tail-window phases; zero for a standalone call)."""
    history_raw = pl.create_tensor([TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_swa_history_seed") as history_seed_tid:
        for row in pl.range(0, TAIL_ROWS, 8):
            history_raw[row:row + 8, :] = pl.full([8, HEAD_DIM], dtype=pl.BF16, value=0.0)
    history_ready_tid = history_seed_tid
    history_phases = pl.cast(0, pl.INT32)
    if pl.read(segment_starts_t, [0]) > 0:
        history_ready_tid = _cp_swa_history_exchange(
            kv_cache, history_slots, hidden_tail_window, ready, consumed, history_raw,
            cache_owner_rank, my_rank, tail_epoch,
        )
        history_phases = pl.cast(1, pl.INT32)
    current_tail_epoch = pl.cast(tail_epoch + history_phases, pl.INT32)
    q = pl.create_tensor([LOCAL_ROWS, H, HEAD_DIM], dtype=pl.BF16)
    post = pl.create_tensor([LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    rope_cos_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_sin_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    rope_cos_il = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    logical_hidden = pl.create_tensor([CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    sparse_kv = pl.create_tensor([SEGMENT_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([SEGMENT_ROWS, PREFILL_SPARSE_PAD], dtype=pl.FP32)
    x_flat = pl.reshape(x_hc, [NUM_LOCAL_TILES * TAIL_ROWS, HC_MULT, D])
    qr = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, 1], dtype=pl.FP32)
    local_kv = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    normed = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, D], dtype=pl.BF16)
    q_pos_flat = pl.reshape(query_positions, [NUM_LOCAL_TILES * TAIL_ROWS])
    q_req_flat = pl.reshape(query_requests, [NUM_LOCAL_TILES * TAIL_ROWS])
    ov_pos_flat = pl.reshape(overlay_positions, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    ov_req_flat = pl.reshape(overlay_requests, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    ov_active_flat = pl.reshape(overlay_active_lengths, [NUM_LOCAL_TILES, OVERLAY_SOURCES])
    swa_flat = pl.reshape(swa_indices, [NUM_LOCAL_TILES * TAIL_ROWS, WIN])
    # Recipes CP prefill is one rank-local 1024-token semantic projection. The
    # two logical 512-token segments stay visible in the CP metadata below;
    # they are not exposed as eight fixed 128-token projection calls.
    materialize_rope_rows(
        freqs_cos, freqs_sin,
        q_pos_flat,
        pl.const(LOCAL_ROWS, pl.INT32),
        rope_cos_flat,
        rope_sin_flat,
    )
    prefill_attention_prolog(
        x_flat, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, gamma_cq, rope_cos_flat, rope_sin_flat,
        normed, post, comb, rope_cos_il, rope_sin_signed, rope_swap_idx,
        q, qr, qr_scale,
    )

    local_hidden_tail = pl.create_tensor([LOCAL_PARTS * TAIL_ROWS, D], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_tail_assemble"):
        for part in pl.range(LOCAL_PARTS):
            zero = pl.const(0, pl.INT32)
            for tile, (segment_total,) in pl.range(MAX_SEGMENT_TILES, init_values=(zero,)):
                active = pl.read(overlay_active_lengths, [part, tile, 1])
                final_total = pl.yield_(segment_total + active)
            tail_start = pl.max(final_total - TAIL_ROWS, 0)
            for row in pl.range(TAIL_ROWS):
                tail_offset = tail_start + row
                local_hidden_tail[
                    part * TAIL_ROWS + row:part * TAIL_ROWS + row + 1
                ] = pl.full([1, D], dtype=pl.BF16, value=0.0)
                if tail_offset < final_total:
                    src = part * MAX_SEGMENT_TILES * TAIL_ROWS + tail_offset
                    local_hidden_tail[part * TAIL_ROWS + row: part * TAIL_ROWS + row + 1] = normed[src:src + 1]

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_swa_tail_exchange", deps=[history_ready_tid]) as tail_exchange_tid:
        _prefill_cp_hidden_tail_exchange_wave(
            local_hidden_tail,
            reverse_index,
            owner_rank_table,
            hidden_tail_window,
            ready,
            consumed,
            logical_hidden,
            my_rank,
            pl.cast(0, pl.INT32),
            current_tail_epoch,
        )

    # Recipes lowers [predecessor128, current512] for each owned segment,
    # then projects KV locally after the normalized hidden-tail exchange.
    augmented_hidden = pl.create_tensor([LOCAL_AUGMENTED_ROWS, D], dtype=pl.BF16)
    augmented_positions = pl.create_tensor([LOCAL_AUGMENTED_ROWS], dtype=pl.INT32)
    with pl.spmd(LOCAL_PARTS, name_hint="cp_swa_augmented_hidden_lowering", deps=[tail_exchange_tid]) as augmented_lowering_tid:
        part = pl.tile.get_block_idx()
        augmented_row0 = part * ROWS_PER_AUGMENTED_PART
        predecessor = pl.read(predecessor_segments, [part])
        predecessor_valid = pl.read(overlay_active_lengths, [part, 0, 0])
        for row in pl.range(TAIL_ROWS):
            destination = augmented_row0 + row
            augmented_hidden[destination:destination + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
            pl.write(augmented_positions, [destination], pl.cast(0, pl.INT32))
            if predecessor >= 0 and row < predecessor_valid:
                source = predecessor * TAIL_ROWS + row
                augmented_hidden[destination:destination + 1, :] = (logical_hidden[source:source + 1, :])
                position = pl.read(segment_tail_positions, [predecessor, row])
                if position >= 0:
                    pl.write(augmented_positions, [destination], position)
        for tile in pl.range(MAX_SEGMENT_TILES):
            active = pl.read(overlay_active_lengths, [part, tile, 1])
            local_row0 = (part * MAX_SEGMENT_TILES + tile) * TAIL_ROWS
            augmented_tile0 = augmented_row0 + (tile + 1) * TAIL_ROWS
            for row in pl.range(TAIL_ROWS):
                destination = augmented_tile0 + row
                augmented_hidden[destination:destination + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
                pl.write(augmented_positions, [destination], pl.cast(0, pl.INT32))
                if row < active:
                    source = local_row0 + row
                    augmented_hidden[destination:destination + 1, :] = (normed[source:source + 1, :])
                    pl.write(augmented_positions, [destination], pl.read(q_pos_flat, [source]))

    augmented_rope_cos = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    augmented_rope_sin = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    augmented_rope_cos_il = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    augmented_rope_sin_signed = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    augmented_rope_swap_idx = pl.create_tensor([LOCAL_AUGMENTED_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    augmented_kv = pl.create_tensor([LOCAL_AUGMENTED_ROWS, HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos, freqs_sin,
        augmented_positions,
        pl.const(LOCAL_AUGMENTED_ROWS, pl.INT32),
        augmented_rope_cos,
        augmented_rope_sin,
    )
    rope_prepare(
        augmented_rope_cos,
        augmented_rope_sin,
        augmented_rope_cos_il,
        augmented_rope_sin_signed,
        augmented_rope_swap_idx,
    )
    kv_proj_rope(
        augmented_hidden,
        wkv,
        gamma_ckv,
        augmented_rope_cos_il,
        augmented_rope_sin_signed,
        augmented_rope_swap_idx,
        augmented_kv,
        augmented_lowering_tid,
    )

    predecessor_kv = pl.create_tensor([LOCAL_PARTS * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_swa_augmented_kv_scatter"):
        for part in pl.range(LOCAL_PARTS):
            augmented_row0 = part * ROWS_PER_AUGMENTED_PART
            local_row0 = part * MAX_SEGMENT_TILES * TAIL_ROWS
            predecessor_row0 = part * TAIL_ROWS
            for row0 in pl.range(0, TAIL_ROWS, ROW_TILE):
                predecessor_kv[predecessor_row0 + row0: predecessor_row0 + row0 + ROW_TILE, :] = augmented_kv[
                    augmented_row0 + row0:
                    augmented_row0 + row0 + ROW_TILE,
                    :,
                ]
            for row0 in pl.range(0, MAX_SEGMENT_TILES * TAIL_ROWS, ROW_TILE):
                local_kv[local_row0 + row0:local_row0 + row0 + ROW_TILE, :] = augmented_kv[
                    augmented_row0 + TAIL_ROWS + row0:
                    augmented_row0 + TAIL_ROWS + row0 + ROW_TILE,
                    :,
                ]

    # The final decode window is also reconstructed from gathered hidden and
    # projected locally; it may span more than one logical segment.
    final_hidden = pl.create_tensor([TAIL_ROWS, D], dtype=pl.BF16)
    final_positions = pl.create_tensor([TAIL_ROWS], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_swa_final_hidden_lowering", deps=[tail_exchange_tid]) as final_hidden_tid:
        for row in pl.range(TAIL_ROWS):
            final_hidden[row:row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
            pl.write(final_positions, [row], pl.cast(0, pl.INT32))
            segment = pl.read(final_win_seg_src, [row])
            source_row = pl.read(final_win_row_src, [row])
            if segment >= 0 and source_row >= 0:
                source = segment * TAIL_ROWS + source_row
                final_hidden[row:row + 1, :] = logical_hidden[source:source + 1, :]
                position = pl.read(segment_tail_positions, [segment, source_row])
                if position >= 0:
                    pl.write(final_positions, [row], position)

    final_rope_cos = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    final_rope_sin = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16)
    final_rope_cos_il = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    final_rope_sin_signed = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.FP32)
    final_rope_swap_idx = pl.create_tensor([TAIL_ROWS, ROPE_HEAD_DIM], dtype=pl.INT32)
    final_kv = pl.create_tensor([TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    materialize_rope_rows(
        freqs_cos, freqs_sin,
        final_positions,
        pl.const(TAIL_ROWS, pl.INT32),
        final_rope_cos,
        final_rope_sin,
    )
    rope_prepare(final_rope_cos, final_rope_sin, final_rope_cos_il, final_rope_sin_signed, final_rope_swap_idx)
    kv_proj_rope(
        final_hidden,
        wkv,
        gamma_ckv,
        final_rope_cos_il,
        final_rope_sin_signed,
        final_rope_swap_idx,
        final_kv,
        final_hidden_tid,
    )

    raw_blocks = pl.tensor.dim(kv_cache, 0)
    raw_rows = raw_blocks * BLOCK_ROWS
    valid_mask = pl.create_tensor([SEGMENT_ROWS, VALID_BLOCK_MASK_COLS], dtype=pl.INT32)
    gather_tid, bias_tid = _cp_swa_stage_sources(
        history_raw, local_kv, predecessor_kv, q_pos_flat, q_req_flat,
        ov_pos_flat, ov_req_flat, predecessor_segments, segment_starts_t, swa_flat,
        sparse_kv, sparse_bias, valid_mask,
        ov_active_flat, pl.const(0, pl.INDEX), tail_exchange_tid,
    )
    stage_ready_tid = pl.system.task_dummy(deps=[gather_tid, bias_tid])
    # Each CP rank owns two logical segments. Tensor extents follow their
    # runtime lengths; storage offsets retain the rank-local capacity layout.
    # Completion dependencies serialize scoped scratch reuse across segments.
    part0_active = (
        pl.read(overlay_active_lengths, [0, 0, 1])
        + pl.read(overlay_active_lengths, [0, 1, 1])
        + pl.read(overlay_active_lengths, [0, 2, 1])
        + pl.read(overlay_active_lengths, [0, 3, 1])
    )
    part1_active = (
        pl.read(overlay_active_lengths, [1, 0, 1])
        + pl.read(overlay_active_lengths, [1, 1, 1])
        + pl.read(overlay_active_lengths, [1, 2, 1])
        + pl.read(overlay_active_lengths, [1, 3, 1])
    )

    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    q_part0 = pl.slice(q, [SEGMENT_ROWS, H, HEAD_DIM], [0, 0, 0])
    kv_part0 = pl.slice(sparse_kv, [SEGMENT_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM], [0, 0])
    bias_part0 = pl.slice(sparse_bias, [SEGMENT_ROWS, PREFILL_SPARSE_PAD], [0, 0])
    mask_part0 = pl.slice(valid_mask, [SEGMENT_ROWS, VALID_BLOCK_MASK_COLS], [0, 0])
    cos_part0 = pl.slice(rope_cos_flat, [SEGMENT_ROWS, ROPE_DIM], [0, 0])
    sin_part0 = pl.slice(rope_sin_flat, [SEGMENT_ROWS, ROPE_DIM], [0, 0])
    residual_part0 = pl.slice(x_flat, [SEGMENT_ROWS, HC_MULT, D], [0, 0, 0])
    post_part0 = pl.slice(post, [SEGMENT_ROWS, HC_MULT], [0, 0])
    comb_part0 = pl.slice(comb, [SEGMENT_ROWS, HC_MULT * HC_MULT], [0, 0])
    out_part0 = pl.slice(x_out_flat, [SEGMENT_ROWS, HC_MULT, D], [0, 0, 0])
    part0_attn_tid = prefill_staged_attention(
        q_part0, kv_part0, bias_part0, mask_part0, attn_sink,
        cos_part0, sin_part0, wo_a, wo_b, wo_b_scale,
        residual_part0, post_part0, comb_part0, out_part0,
        part0_active, stage_ready_tid,
    )
    gather1_tid, bias1_tid = _cp_swa_stage_sources(
        history_raw, local_kv, predecessor_kv, q_pos_flat, q_req_flat,
        ov_pos_flat, ov_req_flat, predecessor_segments, segment_starts_t, swa_flat,
        sparse_kv, sparse_bias, valid_mask,
        ov_active_flat, pl.const(SEGMENT_ROWS, pl.INDEX), part0_attn_tid,
    )
    stage1_ready_tid = pl.system.task_dummy(deps=[gather1_tid, bias1_tid])
    # Both segments have consumed the old cache before its final window is committed.
    cache_commit_flat = pl.reshape(kv_cache, [raw_rows, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_swa_cache_commit", deps=[stage1_ready_tid]) as raw_commit_tid:
        for row in pl.range(TAIL_ROWS):
            seg = final_win_seg_src[row]
            src_row = final_win_row_src[row]
            dst = final_slot_mapping[row]
            if my_rank == cache_owner_rank and seg >= 0 and src_row >= 0 and dst >= 0 and dst < raw_rows:
                cache_commit_flat[dst:dst + 1] = final_kv[row:row + 1]

    q_part1 = pl.slice(q, [SEGMENT_ROWS, H, HEAD_DIM], [SEGMENT_ROWS, 0, 0])
    kv_part1 = pl.slice(sparse_kv, [SEGMENT_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM], [0, 0])
    bias_part1 = pl.slice(sparse_bias, [SEGMENT_ROWS, PREFILL_SPARSE_PAD], [0, 0])
    mask_part1 = pl.slice(valid_mask, [SEGMENT_ROWS, VALID_BLOCK_MASK_COLS], [0, 0])
    cos_part1 = pl.slice(rope_cos_flat, [SEGMENT_ROWS, ROPE_DIM], [SEGMENT_ROWS, 0])
    sin_part1 = pl.slice(rope_sin_flat, [SEGMENT_ROWS, ROPE_DIM], [SEGMENT_ROWS, 0])
    residual_part1 = pl.slice(x_flat, [SEGMENT_ROWS, HC_MULT, D], [SEGMENT_ROWS, 0, 0])
    post_part1 = pl.slice(post, [SEGMENT_ROWS, HC_MULT], [SEGMENT_ROWS, 0])
    comb_part1 = pl.slice(comb, [SEGMENT_ROWS, HC_MULT * HC_MULT], [SEGMENT_ROWS, 0])
    out_part1 = pl.slice(x_out_flat, [SEGMENT_ROWS, HC_MULT, D], [SEGMENT_ROWS, 0, 0])
    attention_done_tid = prefill_staged_attention(
        q_part1, kv_part1, bias_part1, mask_part1, attn_sink,
        cos_part1, sin_part1, wo_a, wo_b, wo_b_scale,
        residual_part1, post_part1, comb_part1, out_part1,
        part1_active, stage1_ready_tid,
    )

    resource_done_tid = pl.system.task_dummy(deps=[tail_exchange_tid, raw_commit_tid, attention_done_tid])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_swa_rank_complete", deps=[resource_done_tid]):
        for tile in pl.range(NUM_LOCAL_TILES):
            completion_token[tile : tile + 1, 0:1, 0:8] = pl.slice(x_out_flat, [1, 1, 8], [tile * TAIL_ROWS, 0, 0])
    x_out = pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D])
    return x_out


@pl.jit
def prefill_cp_swa_rank(
    x_hc: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[RAW_BLOCKS_DYN, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]],
    history_slots: pl.Tensor[[TAIL_ROWS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    query_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    overlay_positions: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_requests: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hidden_tail_window: pld.DistributedTensor[[CP_TAIL_WINDOW_ROWS, D], pl.BF16],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    x_out: pl.Out[pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]],
    cache_owner_rank_t: pl.Tensor[[1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    tail_epoch: pl.Scalar[pl.INT32],
):
    """Standalone CP-SWA rank child. Delegates to the inline core so the
    standalone test preserves the original @pl.jit entry point."""
    completion_token = pl.create_tensor([NUM_LOCAL_TILES, 1, 8], dtype=pl.FP32)
    return prefill_attention_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, kv_cache, history_slots,
        attn_sink, wo_a, wo_b, wo_b_scale,
        segment_starts_t, segment_tail_positions, predecessor_segments,
        query_positions, query_requests,
        overlay_positions, overlay_requests,
        overlay_active_lengths, swa_indices,
        reverse_index, owner_rank_table,
        final_win_seg_src, final_win_row_src, final_slot_mapping,
        hidden_tail_window, ready, consumed,
        x_out, completion_token, pl.read(cache_owner_rank_t, [0]), my_rank, tail_epoch,
    )


@pl.jit.host
def prefill_cp_swa_test(
    x_hc: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32],
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
    kv_cache: pl.InOut[pl.Tensor[[CP_SIZE, RAW_BLOCKS_DYN, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]],
    history_slots: pl.Tensor[[CP_SIZE, TAIL_ROWS], pl.INT32],
    cache_owner_rank_t: pl.Tensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    predecessor_segments: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    query_position_ids: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    query_token_to_request: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32],
    overlay_position_ids: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_token_to_request: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32],
    swa_indices: pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    x_out: pl.Out[pl.Tensor[[CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32]],
):
    """Launch one CP-SWA child per rank."""
    window_buf = pld.alloc_window_buffer([CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        window = pld.window(window_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16)
        ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        consumed = pld.window(consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        prefill_cp_swa_rank(
            x_hc[rank],
            hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin, kv_cache[rank], history_slots[rank],
            attn_sink, wo_a, wo_b, wo_b_scale,
            segment_starts_t, segment_tail_positions,
            predecessor_segments[rank],
            query_position_ids[rank], query_token_to_request[rank],
            overlay_position_ids[rank], overlay_token_to_request[rank],
            overlay_active_lengths[rank], swa_indices[rank],
            reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            window, ready, consumed,
            x_out[rank], cache_owner_rank_t[rank], rank, pl.cast(0, pl.INT32),
            device=rank,
        )


def build_cp_tensor_specs(cp_size: int = CP_SIZE, *, num_tokens: int | None = None, prefix: int = 0):
    meta, ctx = build_metadata(cp_size, num_tokens=num_tokens, prefix=prefix)
    qkv_specs = {spec.name: spec for spec in build_qkv_tensor_specs(1, TAIL_ROWS)}
    sparse_specs = { spec.name: spec for spec in build_sparse_attn_tensor_specs(0, TAIL_ROWS) }
    qkv_names = ("wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv")
    tail_names = ("attn_sink", "wo_a", "wo_b", "wo_b_scale")
    base = {name: qkv_specs[name].create_tensor() for name in qkv_names}
    base.update({name: sparse_specs[name].create_tensor() for name in tail_names})
    base["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    base["hc_attn_scale"] = torch.randn(3)
    base["hc_attn_base"] = torch.randn(MIX_HC)
    base["attn_norm_w"] = torch.ones(D, dtype=torch.bfloat16)
    base["freqs_cos"], base["freqs_sin"] = build_rope_tables(M, 0, dtype=torch.bfloat16)
    max_pos = max(ctx["starts"][s] + ctx["lengths"][s] for s in range(2 * cp_size))
    all_x = torch.empty(max_pos + TAIL_ROWS, HC_MULT, D).uniform_(-1, 1)
    x = torch.zeros(cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D)
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            for tile in range(MAX_SEGMENT_TILES):
                active = int(meta["overlay_active_lengths"][rank, part, tile, 1])
                pos = meta["query_position_ids"][rank, part, tile, :active]
                if active:
                    x[rank, part, tile, :active] = all_x[pos.long()]
    cache = torch.zeros(cp_size, ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM, dtype=torch.bfloat16)
    ori_kv = sparse_specs["ori_kv"].create_tensor()
    cache[:, :ori_kv.shape[0]] = ori_kv
    # Seed the window an earlier chunk would have left behind, so a wrong
    # history row reads a different value instead of an incidental zero.
    if prefix:
        cache_rows = cache.reshape(cp_size, -1, HEAD_DIM)
        history = torch.empty(TAIL_ROWS, HEAD_DIM, dtype=torch.float32).uniform_(-1.0, 1.0)
        for offset, position in enumerate(range(max(0, prefix - TAIL_ROWS), prefix)):
            cache_rows[:, ring_phys_row(position)] = history[offset].to(torch.bfloat16)
        cache = cache_rows.reshape(cache.shape)
    owner_rank = torch.tensor([cp_owner_rank(s, cp_size) for s in range(2 * cp_size)], dtype=torch.int32)
    specs = [TensorSpec("x_hc", list(x.shape), torch.float32, init_value=x)]
    for name in (
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin",
    ):
        specs.append(TensorSpec(name, list(base[name].shape), base[name].dtype, init_value=base[name]))
    specs.append(TensorSpec("kv_cache", list(cache.shape), torch.bfloat16, init_value=cache))
    history_slots = torch.full((cp_size, TAIL_ROWS), -1, dtype=torch.int32)
    for row in range(TAIL_ROWS):
        position = prefix - TAIL_ROWS + row
        if position >= 0:
            history_slots[:, row] = ring_phys_row(position)
    specs.append(TensorSpec("history_slots", list(history_slots.shape), torch.int32, init_value=history_slots))
    specs.append(TensorSpec("cache_owner_rank_t", [cp_size, 1], torch.int32,
                            init_value=torch.zeros(cp_size, 1, dtype=torch.int32)))
    for name in tail_names:
        specs.append(TensorSpec(name, list(base[name].shape), base[name].dtype, init_value=base[name]))
    segment_starts = meta["segment_starts"]
    specs.append(TensorSpec("segment_starts_t", list(segment_starts.shape), torch.int32, init_value=segment_starts))
    segment_tail_positions = meta["segment_tail_positions"]
    specs.append(
        TensorSpec(
            "segment_tail_positions",
            list(segment_tail_positions.shape),
            torch.int32,
            init_value=segment_tail_positions,
        )
    )
    for name in (
        "predecessor_segments", "query_position_ids", "query_token_to_request",
        "overlay_position_ids", "overlay_token_to_request", "overlay_active_lengths", "swa_indices",
    ):
        value = meta[name]
        specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    # Spec order must match the kernel signature: run binds its dummy compile
    # args positionally, so owner_rank_table sits between reverse_index and the
    # final_win_* triple exactly as prefill_cp_swa_test declares them.
    specs.append(TensorSpec("reverse_index", list(meta["reverse_index"].shape), meta["reverse_index"].dtype, init_value=meta["reverse_index"]))
    specs.append(TensorSpec("owner_rank_table", list(owner_rank.shape), owner_rank.dtype, init_value=owner_rank))
    for name in ("final_win_seg_src", "final_win_row_src", "final_slot_mapping"):
        specs.append(TensorSpec(name, list(meta[name].shape), meta[name].dtype, init_value=meta[name]))
    specs.append(TensorSpec("x_out", list(x.shape), torch.float32))
    return specs, ctx


def golden_prefill_cp_swa(tensors):
    """Compose CP-SWA golden outputs in logical-segment order."""
    import torch

    cp = tensors["x_hc"].shape[0]
    metadata_names = (
        "predecessor_segments", "query_position_ids", "query_token_to_request",
        "overlay_position_ids", "overlay_token_to_request", "overlay_active_lengths", "swa_indices",
    )
    meta = {name: tensors[name] for name in metadata_names}
    ctx_case = getattr(golden_prefill_cp_swa, "_ctx", None)
    if ctx_case is None:
        raise RuntimeError("CP-SWA golden context was not installed by the fixture")
    parts = ctx_case["owner_segments"]
    initial_cache = tensors["kv_cache"].clone()
    local_kvs = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    local_q = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16)
    local_post = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT)
    local_comb = torch.zeros(cp, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT * HC_MULT)
    logical = torch.zeros(2 * cp, TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
    for rank in range(cp):
        for part in range(LOCAL_PARTS):
            seg = parts[rank][part]
            for tile in range(MAX_SEGMENT_TILES):
                active = int(meta["overlay_active_lengths"][rank, part, tile, 1])
                x_tile = tensors["x_hc"][rank, part, tile]
                xm = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                post = torch.zeros(TAIL_ROWS, HC_MULT)
                comb = torch.zeros(TAIL_ROWS, HC_MULT * HC_MULT)
                golden_hc_pre(
                    {
                        "x": x_tile,
                        "hc_fn": tensors["hc_attn_fn"], "hc_scale": tensors["hc_attn_scale"],
                        "hc_base": tensors["hc_attn_base"],
                        "x_mixed": xm, "post": post, "comb": comb,
                    }
                )
                positions = meta["query_position_ids"][rank, part, tile].long()
                norm = golden_rms_norm(xm, tensors["attn_norm_w"])
                q = torch.zeros(TAIL_ROWS, H, HEAD_DIM, dtype=torch.bfloat16)
                kv = torch.zeros(TAIL_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                qr = torch.zeros(TAIL_ROWS, Q_LORA, dtype=torch.int8)
                qrs = torch.zeros(TAIL_ROWS, 1)
                golden_qkv_proj_rope(
                    {
                        "x": norm,
                        "wq_a": tensors["wq_a"], "wq_b": tensors["wq_b"],
                        "wq_b_scale": tensors["wq_b_scale"], "wkv": tensors["wkv"],
                        "rope_cos": tensors["freqs_cos"].index_select(0, positions),
                        "rope_sin": tensors["freqs_sin"].index_select(0, positions),
                        "gamma_cq": tensors["gamma_cq"], "gamma_ckv": tensors["gamma_ckv"],
                        "q": q, "kv": kv, "qr": qr, "qr_scale": qrs,
                    }
                )
                local_q[rank, part, tile] = q
                local_kvs[rank, part, tile] = kv
                local_post[rank, part, tile] = post
                local_comb[rank, part, tile] = comb
    # Final projected tail for each logical segment.
    for rank in range(cp):
        for part in range(LOCAL_PARTS):
            seg = parts[rank][part]
            active_lengths = [
                int(meta["overlay_active_lengths"][rank, part, tile, 1])
                for tile in range(MAX_SEGMENT_TILES)
            ]
            rows = [local_kvs[rank, part, tile, :active] for tile, active in enumerate(active_lengths) if active > 0]
            if rows:
                segment_rows = torch.cat(rows, dim=0)
                valid = min(TAIL_ROWS, segment_rows.shape[0])
                logical[seg, :valid] = segment_rows[-valid:]
    out = torch.zeros_like(tensors["x_out"])
    for rank in range(cp):
        cache_owner = int(tensors["cache_owner_rank_t"][rank, 0])
        cache_flat = initial_cache[cache_owner].reshape(-1, HEAD_DIM)
        for part in range(LOCAL_PARTS):
            for tile in range(MAX_SEGMENT_TILES):
                active = int(meta["overlay_active_lengths"][rank, part, tile, 1])
                fake = torch.zeros(ORI_CACHE_ROWS + OVERLAY_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                fake[:cache_flat.shape[0]] = cache_flat
                seg = parts[rank][part]
                pred = int(meta["predecessor_segments"][rank, part])
                if tile == 0 and pred >= 0:
                    fake[OVERLAY_BASE:OVERLAY_BASE + TAIL_ROWS] = logical[pred]
                elif tile > 0:
                    fake[OVERLAY_BASE:OVERLAY_BASE + TAIL_ROWS] = local_kvs[rank, part, tile - 1]
                fake[OVERLAY_BASE + TAIL_ROWS:OVERLAY_BASE + OVERLAY_ROWS] = local_kvs[rank, part, tile]
                fake_cache = fake.view((ORI_CACHE_ROWS + OVERLAY_ROWS) // BLOCK_ROWS, BLOCK_ROWS, 1, HEAD_DIM)
                attn = torch.zeros(TAIL_ROWS, D, dtype=torch.bfloat16)
                positions = meta["query_position_ids"][rank, part, tile].long()
                golden_prefill_sparse_attn(
                    {
                        "q": local_q[rank, part, tile], "ori_kv": fake_cache,
                        "swa_indices": meta["swa_indices"][rank, part, tile],
                        "cmp_kv": torch.zeros(1, BLOCK_ROWS, 1, HEAD_DIM, dtype=torch.bfloat16),
                        "cmp_block_table": torch.zeros(1, dtype=torch.int32),
                        "cmp_indices": torch.full((TAIL_ROWS, 1), -1, dtype=torch.int32),
                        "attn_sink": tensors["attn_sink"], "num_tokens": active,
                        "freqs_cos": tensors["freqs_cos"].index_select(0, positions),
                        "freqs_sin": tensors["freqs_sin"].index_select(0, positions),
                        "wo_a": tensors["wo_a"], "wo_b": tensors["wo_b"], "wo_b_scale": tensors["wo_b_scale"],
                        "attn_out": attn,
                    }
                )
                y = torch.zeros(TAIL_ROWS, HC_MULT, D)
                golden_hc_post_prefill(
                    {
                        "x": attn, "residual": tensors["x_hc"][rank, part, tile],
                        "post": local_post[rank, part, tile], "comb": local_comb[rank, part, tile],
                        "y": y, "num_tokens": active,
                    }
                )
                out[rank, part, tile] = y
    tensors["x_out"][:] = out
    final_cache = initial_cache.clone().reshape(cp, -1, HEAD_DIM)
    for row in range(TAIL_ROWS):
        seg = int(tensors["final_win_seg_src"][row])
        src_row = int(tensors["final_win_row_src"][row])
        dst = int(tensors["final_slot_mapping"][row])
        if seg >= 0 and src_row >= 0 and dst >= 0:
            final_cache[:, dst] = logical[seg, src_row]
    final_cache = final_cache.reshape_as(tensors["kv_cache"])
    for rank in range(cp):
        if rank == int(tensors["cache_owner_rank_t"][rank, 0]):
            tensors["kv_cache"][rank].copy_(final_cache[rank])


if __name__ == "__main__" and not _run_cp_fixture:
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 packed prefill SWA correctness test.")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--start-pos", type=int, default=START_POS,
                        help="context_len (multiple of S=WIN); fixture-only, lowered into token metadata.")
    parser.add_argument("--num-tokens", type=int, default=T,
                        help="Active token count (q_len), capped by T; passed to the kernel as num_tokens.")
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()
    compare_tokens = args.num_tokens

    result = run(
        fn=prefill_mtp_attention_swa_test,
        specs=build_tensor_specs(args.start_pos, args.num_tokens),
        golden_fn=golden_prefill_attention_swa,
        config=dict(
            dump_passes=args.dump_passes,
            platform=args.platform,
            device_id=args.device,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        compile_only=args.compile_only,
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.005, max_diff_hd=1,
                                   valid_rows=compare_tokens, zero_tail=True),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__" and _run_cp_fixture:
    import argparse

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 context-parallel SWA test.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", default=",".join(str(i) for i in range(CP_SIZE)))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--save-data",
        action="store_true",
        default=False,
        help="persist inputs and golden outputs for replay",
    )
    parser.add_argument(
        "--golden-data",
        type=str,
        default=None,
        help="directory containing cached in/ and out/ tensors",
    )
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=list(CP_CHOICES))
    parser.add_argument("--num-tokens", type=int, default=None, help="actual request length; defaults to full capacity")
    parser.add_argument("--prefix", type=int, default=0,
                        help="tokens already resident in this request's caches from earlier chunks")
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0, choices=range(5))
    args = parser.parse_args()

    from golden import ratio_allclose, ratio_reldiff, run

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(f"CP{args.cp} requires {args.cp} devices, got {device_ids}")
    specs, ctx = build_cp_tensor_specs(args.cp, num_tokens=args.num_tokens, prefix=args.prefix)
    golden_prefill_cp_swa._ctx = ctx
    result = run(
        fn=prefill_cp_swa_test,
        specs=specs,
        golden_fn=golden_prefill_cp_swa,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        config=dict(
            distributed_config=DistributedConfig(device_ids=device_ids[:args.cp], num_sub_workers=0),
            dump_passes=args.dump_passes,
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            ring_heap=PREFILL_CP_SWA_RING_HEAP,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "x_out": ratio_reldiff(diff_thd=3e-3, pct_thd=0.005, max_diff_hd=1),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
