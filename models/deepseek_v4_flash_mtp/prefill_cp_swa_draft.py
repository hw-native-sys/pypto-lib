# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek V4 context-parallel SWA prefill."""

import sys

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from config import (
    BLOCK_SIZE,
    FLASH as M,
    FP32_NEG_INF,
    PREFILL_ORI_MAX_BLOCKS,
)
from prefill_cp_zigzag import (
    CP_CHOICES,
    CP_DEFAULT,
    CP_TAIL_WINDOW_ROWS,
    HEAD_DIM,
    _prefill_cp_zigzag_kv_tail_exchange_wave,
    cp_final_window_sources,
    cp_owner_part,
    cp_owner_rank,
    cp_reverse_index,
)

from golden import TensorSpec
from hc_post import golden_hc_post_prefill, hc_post_prefill
from hc_pre import golden_hc_pre, hc_pre
from qkv_proj_rope import (
    build_tensor_specs as build_qkv_tensor_specs,
    golden_qkv_proj_rope,
    materialize_rope_rows,
    qkv_proj_rope,
)
from rmsnorm import golden_rms_norm, rms_norm
from prefill_sparse_attn import (
    BIAS_TOKEN_TILE,
    PREFILL_ATTN_BLOCKS,
    PREFILL_ATTN_TILE,
    PREFILL_SPARSE_PAD,
    SPARSE_BIAS_COLS,
    VALID_BLOCK_MASK_COLS,
    sparse_attn_math,
    build_tensor_specs as build_sparse_attn_tensor_specs,
    golden_prefill_sparse_attn,
)
from utils import build_rope_tables

# model config
D = M.hidden_size
H = M.num_attention_heads
HC_DIM = M.hc_dim
HC_MULT = M.hc_mult
MIX_HC = M.mix_hc
O_GROUPS = M.o_groups
O_GROUP_IN = H * M.head_dim // O_GROUPS
O_LORA = M.o_lora_rank
Q_LORA = M.q_lora_rank
ROPE_HEAD_DIM = M.qk_rope_head_dim
ROPE_DIM = ROPE_HEAD_DIM
MAX_SEQ_LEN = M.max_position_embeddings


def _parse_static_int(name: str, default: int) -> int:
    flag = f"--{name}"
    for i, token in enumerate(sys.argv):
        if token == flag and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if token.startswith(f"{flag}="):
            return int(token.split("=", 1)[1])
    return default


# CP layout
CP_SIZE = _parse_static_int("cp", CP_DEFAULT)
NUM_SEGMENTS = 2 * CP_SIZE
WIN = M.sliding_window
TAIL_ROWS = WIN
BLOCK_ROWS = BLOCK_SIZE
MAX_SEGMENT_TILES = 2
LOCAL_PARTS = 2
NUM_LOCAL_TILES = LOCAL_PARTS * MAX_SEGMENT_TILES
LOCAL_ROWS = NUM_LOCAL_TILES * TAIL_ROWS
LOCAL_SPARSE_ROWS = LOCAL_ROWS * PREFILL_SPARSE_PAD

ORI_MAX_BLOCKS = PREFILL_ORI_MAX_BLOCKS
ORI_CACHE_ROWS = ORI_MAX_BLOCKS * BLOCK_ROWS

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
    if tile == 0:
        return min(segment_len, TAIL_ROWS)
    return max(0, min(segment_len, 2 * TAIL_ROWS) - TAIL_ROWS)


def tail_start(seg_start: int, seg_len: int) -> int:
    return seg_start + max(0, seg_len - TAIL_ROWS)


def segment_starts(prefix: int, segment_span: int, nseg: int):
    return [prefix + s * segment_span for s in range(nseg)]


def lower_key_build(key_abs, segment, tile, starts, lengths, prefix):
    """Lower an absolute key position into the persistent or overlay row."""
    seg_start = starts[segment]
    seg_len = lengths[segment]
    cur_start = seg_start + tile * TAIL_ROWS
    cur_len = active_tile(seg_len, tile)

    if key_abs < prefix:
        return ring_phys_row(key_abs)
    if cur_start <= key_abs < cur_start + cur_len:
        return OVERLAY_BASE + CUR_OVERLAY_ROWS + (key_abs - cur_start)
    if key_abs < cur_start:
        if tile == 0:
            pred_seg = pred_segment(segment)
            if pred_seg < 0:
                return -1
            pred_len = min(TAIL_ROWS, lengths[pred_seg])
            pred_tail = tail_start(starts[pred_seg], lengths[pred_seg])
            if pred_tail <= key_abs < pred_tail + pred_len:
                return OVERLAY_BASE + (key_abs - pred_tail)
            return -1
        prev_start = cur_start - TAIL_ROWS
        prev_len = active_tile(seg_len, 0)
        if prev_start <= key_abs < prev_start + prev_len:
            return OVERLAY_BASE + (key_abs - prev_start)
        return -1
    return -1


def build_metadata(cp_size: int = CP_SIZE):
    """Build the canonical zero-history CP metadata."""
    prefix = 0
    segment_span = TAIL_ROWS
    lengths = [TAIL_ROWS] * (2 * cp_size)
    nseg = 2 * cp_size
    starts = segment_starts(prefix, segment_span, nseg)
    parts = owner_segments(cp_size)

    seg_starts_t = torch.tensor(starts, dtype=torch.int32)
    seg_lens_t = torch.tensor(lengths, dtype=torch.int32)
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
                    q_pos[rank, part, tile, :active] = torch.arange(
                        tile_start, tile_start + active, dtype=torch.int32)
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
                    pred_len = active_tile(seg_len, 0)
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

                for query_row in range(active):
                    query_abs = tile_start + query_row
                    for col in range(WIN):
                        key_abs = query_abs - WIN + 1 + col
                        if key_abs < 0 or key_abs > query_abs:
                            continue
                        raw = lower_key_build(key_abs, segment, tile, starts, lengths, prefix)
                        swa[rank, part, tile, query_row, col] = raw

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
def _cp_swa_stage_sources(
    cache_flat: pl.Tensor[[ORI_CACHE_ROWS, HEAD_DIM], pl.BF16],
    local_kv: pl.Tensor[[LOCAL_ROWS, HEAD_DIM], pl.BF16],
    logical_tails: pl.Tensor[[CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16],
    query_positions: pl.Tensor[[LOCAL_ROWS], pl.INT32],
    query_requests: pl.Tensor[[LOCAL_ROWS], pl.INT32],
    overlay_positions: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32],
    overlay_requests: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_ROWS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    swa_indices: pl.Tensor[[LOCAL_ROWS, WIN], pl.INT32],
    sparse_kv: pl.Tensor[[LOCAL_SPARSE_ROWS, HEAD_DIM], pl.BF16],
    sparse_bias: pl.Tensor[[NUM_LOCAL_TILES * TAIL_ROWS, PREFILL_SPARSE_PAD], pl.FP32],
    valid_block_mask: pl.Tensor[[NUM_LOCAL_TILES * TAIL_ROWS, VALID_BLOCK_MASK_COLS], pl.INT32],
    overlay_active_lengths: pl.Tensor[[NUM_LOCAL_TILES, OVERLAY_SOURCES], pl.INT32],
):
    """Stage the accepted persistent/predecessor/current source ABI."""
    prefix = pl.read(segment_starts_t, [0])
    with pl.spmd((LOCAL_ROWS // 2) * PREFILL_ATTN_BLOCKS, name_hint="gather_kv") as gather_tid:
        block = pl.tile.get_block_idx()
        schedule = block // PREFILL_ATTN_BLOCKS
        sb = block - schedule * PREFILL_ATTN_BLOCKS
        token_block = (LOCAL_ROWS // 2) - 1 - schedule
        t0 = token_block * 2
        k0 = sb * PREFILL_ATTN_TILE
        for dt in pl.range(2):
            row = t0 + dt
            if row < LOCAL_ROWS:
                stage = pl.full([PREFILL_ATTN_TILE, HEAD_DIM], dtype=pl.BF16, value=0.0)
                out_base = row * PREFILL_SPARSE_PAD + k0
                for ki in pl.range(PREFILL_ATTN_TILE):
                    col = k0 + ki
                    if col < WIN:
                        raw = pl.read(swa_indices, [row, col])
                        if raw >= 0:
                            q_abs = pl.read(query_positions, [row])
                            q_req = pl.read(query_requests, [row])
                            key_abs = q_abs - WIN + 1 + col
                            if raw < ORI_CACHE_ROWS:
                                if key_abs < prefix and key_abs <= q_abs and q_req >= 0:
                                    src = pl.cast(raw, pl.INDEX)
                                    stage[ki:ki + 1, :] = cache_flat[src:src + 1, :]
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
                                                src = pred * TAIL_ROWS + src_row
                                                stage[ki:ki + 1, :] = logical_tails[src:src + 1, :]
                                        else:
                                            src = (tile - 1) * TAIL_ROWS + src_row
                                            stage[ki:ki + 1, :] = local_kv[src:src + 1, :]
                sparse_kv[out_base:out_base + PREFILL_ATTN_TILE, :] = stage
    with pl.spmd(LOCAL_ROWS // BIAS_TOKEN_TILE, name_hint="build_bias") as bias_tid:
        bias_blk = pl.tile.get_block_idx()
        bias_t0 = bias_blk * BIAS_TOKEN_TILE
        bias_idx = pl.cast(
            swa_indices[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN],
            target_type=pl.FP32,
        )
        flags = pl.minimum(pl.maximum(pl.add(bias_idx, 1.0), 0.0), 1.0)
        sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, 0:WIN] = pl.mul(pl.sub(flags, 1.0), -FP32_NEG_INF)
        if SPARSE_BIAS_COLS < PREFILL_SPARSE_PAD:
            sparse_bias[bias_t0:bias_t0 + BIAS_TOKEN_TILE, SPARSE_BIAS_COLS:PREFILL_SPARSE_PAD] = pl.full(
                [BIAS_TOKEN_TILE, PREFILL_SPARSE_PAD - SPARSE_BIAS_COLS], dtype=pl.FP32, value=FP32_NEG_INF)
    return gather_tid, bias_tid


@pl.jit.inline
def prefill_cp_swa_core(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32
    ],
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
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    completion_token: pl.Out[
        pl.Tensor[[NUM_LOCAL_TILES, 1, 8], pl.FP32]
    ],
    my_rank: pl.Scalar[pl.INT32],
    tail_epoch: pl.Scalar[pl.INT32],
):
    """CP-SWA attention math (inline). Shared by the standalone rank child
    and the layer composition child. Inlining avoids child-in-child nesting
    (@pl.jit cannot call another @pl.jit).

    ``tail_epoch`` is the cross-layer tail-exchange communication epoch
    (layer-ordinal for SWA; zero for standalone / single-layer)."""
    q = pl.create_tensor([LOCAL_ROWS, H, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    post = pl.create_tensor([LOCAL_ROWS, HC_MULT], dtype=pl.FP32)
    comb = pl.create_tensor([LOCAL_ROWS, HC_MULT * HC_MULT], dtype=pl.FP32)
    rope_cos_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    rope_sin_flat = pl.create_tensor([LOCAL_ROWS, ROPE_HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    logical_tails = pl.create_tensor([CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_kv = pl.create_tensor([LOCAL_SPARSE_ROWS, HEAD_DIM], dtype=pl.BF16)
    sparse_bias = pl.create_tensor([LOCAL_ROWS, PREFILL_SPARSE_PAD], dtype=pl.FP32, init_value=FP32_NEG_INF)
    x_flat = pl.reshape(x_hc, [NUM_LOCAL_TILES * TAIL_ROWS, HC_MULT, D])
    qr = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, 1], dtype=pl.FP32)
    local_kv = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16)
    x_mixed = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, D], dtype=pl.BF16)
    normed = pl.create_tensor([NUM_LOCAL_TILES * TAIL_ROWS, D], dtype=pl.BF16)
    q_pos_flat = pl.reshape(query_positions, [NUM_LOCAL_TILES * TAIL_ROWS])
    q_req_flat = pl.reshape(query_requests, [NUM_LOCAL_TILES * TAIL_ROWS])
    ov_pos_flat = pl.reshape(overlay_positions, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    ov_req_flat = pl.reshape(overlay_requests, [NUM_LOCAL_TILES, OVERLAY_ROWS])
    ov_active_flat = pl.reshape(overlay_active_lengths, [NUM_LOCAL_TILES, OVERLAY_SOURCES])
    swa_flat = pl.reshape(swa_indices, [NUM_LOCAL_TILES * TAIL_ROWS, WIN])
    for tile in pl.range(NUM_LOCAL_TILES):
        t0 = tile * TAIL_ROWS
        x_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [t0, 0, 0])
        mixed_tile = pl.slice(x_mixed, [TAIL_ROWS, D], [t0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [t0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [t0, 0])
        position_tile = pl.slice(q_pos_flat, [TAIL_ROWS], [t0])
        rope_cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [t0, 0])
        rope_sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_HEAD_DIM], [t0, 0])
        normed_tile = pl.slice(normed, [TAIL_ROWS, D], [t0, 0])
        qr_tile = pl.slice(qr, [TAIL_ROWS, Q_LORA], [t0, 0])
        qr_scale_tile = pl.slice(qr_scale, [TAIL_ROWS, 1], [t0, 0])
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [t0, 0, 0])
        kv_tile = pl.slice(local_kv, [TAIL_ROWS, HEAD_DIM], [t0, 0])
        hc_pre(x_tile, hc_attn_fn, hc_attn_scale, hc_attn_base, mixed_tile, post_tile, comb_tile)
        rms_tid = rms_norm(mixed_tile, attn_norm_w, normed_tile)
        # Fixed sparse-attention tile.
        late_dep = pl.system.task_dummy(deps=[rms_tid])
        active = pl.read(overlay_active_lengths, [tile // MAX_SEGMENT_TILES, tile % MAX_SEGMENT_TILES, 1])
        materialize_rope_rows(
            freqs_cos, freqs_sin, position_tile, active,
            rope_cos_tile, rope_sin_tile,
        )
        qkv_proj_rope(
            normed_tile,
            wq_a, wq_b, wq_b_scale, wkv,
            rope_cos_tile, rope_sin_tile,
            gamma_cq, gamma_ckv,
            q_tile, kv_tile, qr_tile, qr_scale_tile, late_dep,
        )

    local_tail = pl.create_tensor([2 * TAIL_ROWS, HEAD_DIM], dtype=pl.BF16, init_value=0.0)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="cp_tail_assemble"):
        for part in pl.range(LOCAL_PARTS):
            tile0_len = pl.read(overlay_active_lengths, [part, 0, 1])
            tile1_len = pl.read(overlay_active_lengths, [part, 1, 1])
            segment_total = tile0_len + tile1_len
            tail_start = pl.max(segment_total - TAIL_ROWS, 0)
            for row in pl.range(TAIL_ROWS):
                tail_offset = tail_start + row
                if tail_offset < segment_total:
                    if tail_offset < TAIL_ROWS:
                        src = part * MAX_SEGMENT_TILES * TAIL_ROWS + tail_offset
                    else:
                        src = part * MAX_SEGMENT_TILES * TAIL_ROWS + TAIL_ROWS + tail_offset - TAIL_ROWS
                    local_tail[part * TAIL_ROWS + row:part * TAIL_ROWS + row + 1] = local_kv[src:src + 1]

    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_swa_tail_exchange",
    ) as tail_exchange_tid:
        _prefill_cp_zigzag_kv_tail_exchange_wave(
            local_tail, reverse_index, owner_rank_table,
            kv_tail_window, ready, consumed, logical_tails,
            my_rank, tail_epoch,
        )
    cache_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    valid_mask = pl.create_tensor([LOCAL_ROWS, VALID_BLOCK_MASK_COLS], dtype=pl.INT32, init_value=0)
    _cp_swa_stage_sources(
        cache_flat, local_kv, logical_tails, q_pos_flat, q_req_flat,
        ov_pos_flat, ov_req_flat, predecessor_segments, segment_starts_t, swa_flat,
        sparse_kv, sparse_bias, valid_mask,
        ov_active_flat,
    )
    cache_commit_flat = pl.reshape(kv_cache, [ORI_CACHE_ROWS, HEAD_DIM])
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_swa_cache_commit",
    ) as raw_commit_tid:
        for row in pl.range(TAIL_ROWS):
            seg = final_win_seg_src[row]
            src_row = final_win_row_src[row]
            dst = final_slot_mapping[row]
            if seg >= 0 and src_row >= 0 and dst >= 0:
                src = seg * TAIL_ROWS + src_row
                cache_commit_flat[dst:dst + 1] = logical_tails[src:src + 1]

    x_flat = pl.reshape(x_hc, [LOCAL_ROWS, HC_MULT, D])
    x_out_flat = pl.reshape(x_out, [LOCAL_ROWS, HC_MULT, D])
    for tile in pl.range(NUM_LOCAL_TILES):
        t0 = tile * TAIL_ROWS
        sparse0 = tile * TAIL_ROWS * PREFILL_SPARSE_PAD
        part = tile // MAX_SEGMENT_TILES
        part_tile = tile % MAX_SEGMENT_TILES
        q_tile = pl.slice(q, [TAIL_ROWS, H, HEAD_DIM], [t0, 0, 0])
        sparse_kv_tile = pl.slice(
            sparse_kv,
            [TAIL_ROWS * PREFILL_SPARSE_PAD, HEAD_DIM],
            [sparse0, 0],
        )
        bias_tile = pl.slice(
            sparse_bias, [TAIL_ROWS, PREFILL_SPARSE_PAD], [t0, 0]
        )
        mask_tile = pl.slice(
            valid_mask, [TAIL_ROWS, VALID_BLOCK_MASK_COLS], [t0, 0]
        )
        cos_tile = pl.slice(rope_cos_flat, [TAIL_ROWS, ROPE_DIM], [t0, 0])
        sin_tile = pl.slice(rope_sin_flat, [TAIL_ROWS, ROPE_DIM], [t0, 0])
        post_tile = pl.slice(post, [TAIL_ROWS, HC_MULT], [t0, 0])
        comb_tile = pl.slice(comb, [TAIL_ROWS, HC_MULT * HC_MULT], [t0, 0])
        x_tile = pl.slice(x_flat, [TAIL_ROWS, HC_MULT, D], [t0, 0, 0])
        active = pl.read(overlay_active_lengths, [part, part_tile, 1])
        attn_out_tile = pl.create_tensor([TAIL_ROWS, D], dtype=pl.BF16)
        y_tile = pl.slice(
            x_out_flat, [TAIL_ROWS, HC_MULT, D], [t0, 0, 0]
        )
        sparse_attn_math(
            q=q_tile, sparse_kv=sparse_kv_tile, sparse_bias=bias_tile,
            valid_block_mask=mask_tile, attn_sink=attn_sink,
            freqs_cos=cos_tile, freqs_sin=sin_tile, wo_a=wo_a,
            wo_b=wo_b, wo_b_scale=wo_b_scale,
            attn_out=attn_out_tile, num_tokens=active,
        )
        hc_post_prefill(
            attn_out_tile, x_tile, post_tile, comb_tile, y_tile, active,
        )

    resource_done_tid = pl.system.task_dummy(
        deps=[tail_exchange_tid, raw_commit_tid]
    )
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="cp_swa_rank_complete",
        deps=[resource_done_tid],
    ):
        for tile in pl.range(NUM_LOCAL_TILES):
            completion_token[tile : tile + 1, 0:1, 0:8] = pl.slice(
                x_out_flat, [1, 1, 8], [tile * TAIL_ROWS, 0, 0]
            )
    x_out = pl.reshape(x_out_flat, [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D])
    return x_out


@pl.jit
def prefill_cp_swa_rank(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32
    ],
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
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    x_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
            pl.FP32,
        ]
    ],
    my_rank: pl.Scalar[pl.INT32],
    tail_epoch: pl.Scalar[pl.INT32],
):
    """Standalone CP-SWA rank child. Delegates to the inline core so the
    standalone test preserves the original @pl.jit entry point."""
    completion_token = pl.create_tensor(
        [NUM_LOCAL_TILES, 1, 8], dtype=pl.FP32, init_value=0.0
    )
    return prefill_cp_swa_core(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
        wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, kv_cache,
        attn_sink, wo_a, wo_b, wo_b_scale,
        segment_starts_t, predecessor_segments,
        query_positions, query_requests,
        overlay_positions, overlay_requests,
        overlay_active_lengths, swa_indices,
        reverse_index, owner_rank_table,
        final_win_seg_src, final_win_row_src, final_slot_mapping,
        kv_tail_window, ready, consumed,
        x_out, completion_token, my_rank, tail_epoch,
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
    kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16
        ]
    ],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
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
    window_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        window = pld.window(
            window_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        consumed = pld.window(
            consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        prefill_cp_swa_rank(
            x_hc[rank],
            hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin, kv_cache[rank],
            attn_sink, wo_a, wo_b, wo_b_scale,
            segment_starts_t, predecessor_segments[rank],
            query_position_ids[rank], query_token_to_request[rank],
            overlay_position_ids[rank], overlay_token_to_request[rank],
            overlay_active_lengths[rank], swa_indices[rank],
            reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            window, ready, consumed,
            x_out[rank], rank, pl.cast(0, pl.INT32),
            device=rank,
        )


def build_tensor_specs(cp_size: int = CP_SIZE):
    meta, ctx = build_metadata(cp_size)
    torch.manual_seed(4100 + cp_size * 31)
    qkv_specs = {spec.name: spec for spec in build_qkv_tensor_specs(1, TAIL_ROWS)}
    sparse_specs = {
        spec.name: spec
        for spec in build_sparse_attn_tensor_specs(0, TAIL_ROWS)
    }
    qkv_names = (
        "wq_a",
        "wq_b",
        "wq_b_scale",
        "wkv",
        "gamma_cq",
        "gamma_ckv",
    )
    tail_names = ("attn_sink", "wo_a", "wo_b", "wo_b_scale")
    base = {name: qkv_specs[name].create_tensor() for name in qkv_names}
    base.update({name: sparse_specs[name].create_tensor() for name in tail_names})
    base["hc_attn_fn"] = torch.randn(MIX_HC, HC_DIM) / HC_DIM ** 0.5
    base["hc_attn_scale"] = torch.randn(3)
    base["hc_attn_base"] = torch.randn(MIX_HC)
    base["attn_norm_w"] = torch.ones(D, dtype=torch.bfloat16)
    base["freqs_cos"], base["freqs_sin"] = build_rope_tables(
        M, 0, dtype=torch.bfloat16
    )
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
    owner_rank = torch.tensor([cp_owner_rank(s, cp_size) for s in range(2 * cp_size)], dtype=torch.int32)
    specs = [TensorSpec("x_hc", list(x.shape), torch.float32, init_value=x)]
    for name in (
        "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
        "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
        "freqs_cos", "freqs_sin",
    ):
        specs.append(TensorSpec(name, list(base[name].shape), base[name].dtype, init_value=base[name]))
    specs.append(TensorSpec("kv_cache", list(cache.shape), torch.bfloat16, init_value=cache, is_output=True))
    for name in tail_names:
        specs.append(TensorSpec(name, list(base[name].shape), base[name].dtype, init_value=base[name]))
    segment_starts = meta["segment_starts"]
    specs.append(TensorSpec("segment_starts_t", list(segment_starts.shape), torch.int32, init_value=segment_starts))
    for name in (
        "predecessor_segments", "query_position_ids", "query_token_to_request",
        "overlay_position_ids", "overlay_token_to_request", "overlay_active_lengths", "swa_indices",
    ):
        value = meta[name]
        specs.append(TensorSpec(name, list(value.shape), value.dtype, init_value=value))
    # Spec order must match the kernel signature: run_jit binds its dummy compile
    # args positionally, so owner_rank_table sits between reverse_index and the
    # final_win_* triple exactly as prefill_cp_swa_test declares them.
    specs.append(TensorSpec("reverse_index", list(meta["reverse_index"].shape), meta["reverse_index"].dtype, init_value=meta["reverse_index"]))
    specs.append(TensorSpec("owner_rank_table", list(owner_rank.shape), owner_rank.dtype, init_value=owner_rank))
    for name in ("final_win_seg_src", "final_win_row_src", "final_slot_mapping"):
        specs.append(TensorSpec(name, list(meta[name].shape), meta[name].dtype, init_value=meta[name]))
    specs.append(TensorSpec("x_out", list(x.shape), torch.float32, is_output=True))
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
    lengths = ctx_case["lengths"]
    starts = ctx_case["starts"]
    parts = ctx_case["owner_segments"]
    prefix = ctx_case["prefix"]
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
            tile0_len = int(meta["overlay_active_lengths"][rank, part, 0, 1])
            tile1_len = int(meta["overlay_active_lengths"][rank, part, 1, 1])
            total = tile0_len + tile1_len
            tail_start = max(0, total - TAIL_ROWS)
            for row in range(TAIL_ROWS):
                tail_offset = tail_start + row
                if tail_offset >= total:
                    continue
                if tail_offset < TAIL_ROWS:
                    logical[seg, row] = local_kvs[rank, part, 0, tail_offset]
                else:
                    logical[seg, row] = local_kvs[rank, part, 1, tail_offset - TAIL_ROWS]
    out = torch.zeros_like(tensors["x_out"])
    for rank in range(cp):
        cache_flat = initial_cache[rank].reshape(-1, HEAD_DIM)
        for part in range(LOCAL_PARTS):
            for tile in range(MAX_SEGMENT_TILES):
                active = int(meta["overlay_active_lengths"][rank, part, tile, 1])
                fake = torch.zeros(ORI_CACHE_ROWS + OVERLAY_ROWS, HEAD_DIM, dtype=torch.bfloat16)
                fake[:ORI_CACHE_ROWS] = cache_flat
                seg = parts[rank][part]
                pred = int(meta["predecessor_segments"][rank, part])
                if tile == 0 and pred >= 0:
                    fake[OVERLAY_BASE:OVERLAY_BASE + TAIL_ROWS] = logical[pred]
                elif tile == 1:
                    fake[OVERLAY_BASE:OVERLAY_BASE + TAIL_ROWS] = local_kvs[rank, part, 0]
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
    tensors["kv_cache"][:] = final_cache.reshape_as(tensors["kv_cache"])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone DeepSeek V4 context-parallel SWA test.")
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", default=",".join(str(i) for i in range(CP_SIZE)))
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--cp", type=int, default=CP_SIZE, choices=list(CP_CHOICES))
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    args = parser.parse_args()

    from golden import ratio_allclose, ratio_reldiff, run_jit

    device_ids = [int(device) for device in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(f"CP{args.cp} requires {args.cp} devices, got {device_ids}")
    specs, ctx = build_tensor_specs(args.cp)
    golden_prefill_cp_swa._ctx = ctx
    result = run_jit(
        fn=prefill_cp_swa_test,
        specs=specs,
        golden_fn=golden_prefill_cp_swa,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:args.cp], num_sub_workers=0),
            dump_passes=args.dump_passes,
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
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
