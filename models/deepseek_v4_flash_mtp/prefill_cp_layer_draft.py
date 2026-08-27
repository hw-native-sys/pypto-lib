# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 context-parallel prefill single layer.

The rank-local graph composes accepted CP attention with four fixed-size
baseline MoE waves. Attention, layer-stage, and MoE communication windows
remain separate so each protocol has one clear owner.
"""

import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

# The prefill path routes PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe (which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS

# Import moe first. It applies the EP override before dependent modules bake
# config-derived MoE shapes.
from moe import (
    AUX_PAD,
    D,
    HC_DIM,
    HC_MULT,
    IDX_PAD,
    MIX_HC,
    MOE_INTER,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    RECV_MAX,
    T,
    TOPK,
    VOCAB,
    build_tensor_specs as build_moe_tensor_specs,
    clear_moe_signals,
    golden_moe,
    moe,
)
from prefill_cp_swa_draft import (
    CP_CHOICES,
    CP_SIZE,
    CP_TAIL_WINDOW_ROWS,
    H,
    HEAD_DIM,
    LOCAL_PARTS,
    MAX_SEQ_LEN,
    MAX_SEGMENT_TILES,
    NUM_SEGMENTS,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    ORI_MAX_BLOCKS,
    OVERLAY_ROWS,
    OVERLAY_SOURCES,
    Q_LORA,
    ROPE_HEAD_DIM,
    TAIL_ROWS,
    WIN,
    build_tensor_specs as build_swa_tensor_specs,
    golden_prefill_cp_swa,
    prefill_cp_swa_core,
)
from prefill_cp_hca_draft import (
    BLOCK_SIZE,
    CMP_META_DIM,
    CMP_WINDOW_ROWS,
    COMPRESS_RATIO,
    COMPRESS_STATE_DIM,
    HCA_STATE_MAX_BLOCKS,
    HCA_STATE_PHYSICAL_BLOCKS,
    HCA_STATE_BLOCK_SIZE,
    IDX_TOPK,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    STATE_META_DIM,
    STATE_WINDOW_ROWS,
    build_tensor_specs as build_hca_tensor_specs,
    golden_prefill_cp_hca,
    prefill_cp_hca_core,
)
from prefill_cp_csa_draft import (
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    IDX_CACHE_MAX_BLOCKS,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM,
    INNER_STATE_BLOCK_SIZE,
    INNER_STATE_DIM,
    INNER_STATE_MAX_BLOCKS,
    LOCAL_LEAVES,
    MAIN_OUT_DIM,
    MAIN_STATE_BLOCK_SIZE,
    MAIN_STATE_DIM,
    MAIN_STATE_MAX_BLOCKS,
    MAX_COMPRESS_LEAVES,
    META_DIM,
    PREFILL_IDX_BLOCK_NUM,
    RECORDS_PER_WINDOW,
    SCALE_TILE_COLS,
    STATE_RECORDS_PER_WINDOW,
    build_tensor_specs as build_csa_tensor_specs,
    golden_prefill_cp_csa,
    prefill_cp_csa_core,
)
from golden import ScalarSpec, TensorSpec, ratio_allclose, ratio_reldiff, run_jit

# ---------------------------------------------------------------------------
# Static CP/EP contract
# ---------------------------------------------------------------------------
assert CP_SIZE in CP_CHOICES, f"--cp must be one of {CP_CHOICES} (got {CP_SIZE})"
assert CP_SIZE == N_RANKS, (
    f"CP layer requires CP == EP == pld.world_size() (got CP={CP_SIZE}, EP={N_RANKS})"
)
assert T == 128, f"CP layer requires T == 128 (got {T})"
assert LOCAL_PARTS == 2
assert MAX_SEGMENT_TILES == 2
NUM_MOE_WAVES = LOCAL_PARTS * MAX_SEGMENT_TILES
assert NUM_MOE_WAVES == 4
assert RECV_MAX == N_RANKS * T, (
    f"CP layer requires RECV_MAX == N_RANKS * T (got {RECV_MAX} vs {N_RANKS * T})"
)
META_WINDOW_ROWS = N_RANKS
PAYLOAD_WINDOW_ROWS = N_LOCAL * RECV_MAX
ROUTED_WINDOW_ROWS = N_ROUTES

# Supported representative layer IDs for each attention kind.
SWA_LAYER_ID = 0
HCA_LAYER_ID = 3
CSA_LAYER_ID = 2


def _assert_layer_id(layer_id: int) -> None:
    if layer_id not in (SWA_LAYER_ID, CSA_LAYER_ID, HCA_LAYER_ID):
        raise RuntimeError(
            f"current CP layer accepts only layer_id={SWA_LAYER_ID} (SWA), "
            f"layer_id={CSA_LAYER_ID} (CSA), or layer_id={HCA_LAYER_ID} (HCA)."
        )


# ---------------------------------------------------------------------------
# Local stage sequencing
# ---------------------------------------------------------------------------
# Attention and MoE close their own distributed protocols. The layer only needs
# a rank-local tensor chain from attention completion through the four MoE
# waves; adding another distributed barrier here can block MoE's own rendezvous.
COPY_TOKEN_TILE = 4      # FP32 copy tile: 4 tokens x 1 HC lane x D = 64 KiB
assert T % COPY_TOKEN_TILE == 0


@pl.jit.inline
def _attention_stage_barrier_from_x_attn(
    x_attn_flat: pl.Tensor[[NUM_MOE_WAVES * T, HC_MULT, D], pl.FP32],
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    stage_tokens: pl.Out[
        pl.Tensor[[NUM_MOE_WAVES + 1, 1, 8], pl.FP32]
    ],
):
    """Publish a rank-local token after the first SWA/HCA output is ready."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_stage_token"):
        completion = pl.slice(x_attn_flat, [1, 1, 8], [0, 0, 0])
        stage_tokens[0:1, 0:1, 0:8] = completion
    return stage_tokens


@pl.jit.inline
def _attention_stage_barrier_from_completion(
    completion_token: pl.Tensor[[NUM_MOE_WAVES, 1, 8], pl.FP32],
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    stage_tokens: pl.Out[
        pl.Tensor[[NUM_MOE_WAVES + 1, 1, 8], pl.FP32]
    ],
):
    """Publish a rank-local token after the attention leaf fan-in closes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_stage_token"):
        completion = pl.slice(completion_token, [1, 1, 8], [0, 0, 0])
        stage_tokens[0:1, 0:1, 0:8] = completion
    return stage_tokens


@pl.jit.inline
def _record_wave_completion(
    wave_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    wave: pl.Scalar[pl.INT32],
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    stage_tokens: pl.Out[
        pl.Tensor[[NUM_MOE_WAVES + 1, 1, 8], pl.FP32]
    ],
):
    """Admit the next local wave after this rank completes the current MoE."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="wave_complete"):
        stage_tokens[wave + 1 : wave + 2, 0:1, 0:8] = pl.slice(
            wave_out, [1, 1, 8], [0, 0, 0]
        )
    return stage_tokens


@pl.jit.inline
def _publish_x_next_after_stage(
    x_next_work: pl.Tensor[[4 * T, HC_MULT, D], pl.FP32],
    active_flat: pl.Tensor[[NUM_MOE_WAVES, OVERLAY_SOURCES], pl.INT32],
    x_next: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
    ],
) -> pl.Tensor[
    [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
]:
    """Publish active rows and zero inactive rows through one HBM copy task.

    The task reads the rank-local MoE outputs, so AUTO dependency tracking
    naturally fans in all four wave producers. It does not require a stage-5
    edge and may overlap the final barrier and signal cleanup."""
    x_next_flat = pl.reshape(
        x_next, [NUM_MOE_WAVES * T, HC_MULT, D]
    )
    wave_blocks = (T // COPY_TOKEN_TILE) * HC_MULT
    with pl.spmd(
        NUM_MOE_WAVES * wave_blocks,
        name_hint="publish_x_next",
    ):
        block = pl.tile.get_block_idx()
        wave = block // wave_blocks
        wave_block = block % wave_blocks
        token_block = wave_block // HC_MULT
        hc_lane = wave_block % HC_MULT
        token0 = token_block * COPY_TOKEN_TILE
        active = pl.read(active_flat, [wave, 1])
        for dt in pl.range(COPY_TOKEN_TILE):
            token = token0 + dt
            row = wave * T + token
            if token < active:
                x_next_flat[
                    row : row + 1,
                    hc_lane : hc_lane + 1,
                    0:D,
                ] = pl.slice(
                    x_next_work,
                    [1, 1, D],
                    [row, hc_lane, 0],
                )
            else:
                x_next_flat[
                    row : row + 1,
                    hc_lane : hc_lane + 1,
                    0:D,
                ] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)
    return pl.reshape(
        x_next_flat,
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
    )


@pl.jit.inline
def _prefill_layer_cp_moe_tail(
    # Attention output (child-local, written by the accepted CP attention core).
    x_attn: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
    ],
    # Wave-active metadata (read for effective_tokens and publication).
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    input_ids: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64
    ],
    # MoE weights/tables (rank-local slices, same shapes as moe() takes).
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # MoE distributed windows (reused across the four waves; ordered by wave
    # through the stage counter).
    recv_meta: pld.DistributedTensor[[META_WINDOW_ROWS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[ROUTED_WINDOW_ROWS, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    # Layer stage synchronization window.
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    stage_tokens: pl.Tensor[[NUM_MOE_WAVES + 1, 1, 8], pl.FP32],
    # Layer output.
    x_next: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    # Scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]:
    """Shared MoE tail: attention boundary -> four MoE waves -> publish
    x_next -> clear MoE signals after a mode-specific attention barrier."""
    x_attn_flat = pl.reshape(x_attn, [NUM_MOE_WAVES * T, HC_MULT, D])

    x_next_work = pl.create_tensor(
        [NUM_MOE_WAVES * T, HC_MULT, D], dtype=pl.FP32, init_value=0.0
    )
    active_flat = pl.reshape(
        overlay_active_lengths, [NUM_MOE_WAVES, OVERLAY_SOURCES]
    )
    input_ids_flat = pl.reshape(input_ids, [NUM_MOE_WAVES * T])

    one = pl.cast(1, pl.INT32)
    x_moe_ready = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32, init_value=0.0
    )
    for wave in pl.range(NUM_MOE_WAVES):
        row0 = wave * T
        effective_tokens = pl.max(pl.read(active_flat, [wave, 1]), one)
        x_src = pl.slice(x_attn_flat, [T, HC_MULT, D], [row0, 0, 0])
        ids = pl.slice(input_ids_flat, [T], [row0])
        with pl.spmd(
            (T // COPY_TOKEN_TILE) * HC_MULT,
            name_hint="moe_input_copy",
        ):
            copy_idx = pl.tile.get_block_idx()
            token_block = copy_idx // HC_MULT
            hc_lane = copy_idx % HC_MULT
            token0 = token_block * COPY_TOKEN_TILE
            _tok = pl.read(stage_tokens, [wave, 0, 0])
            x_moe_ready[
                token0 : token0 + COPY_TOKEN_TILE,
                hc_lane : hc_lane + 1,
                0:D,
            ] = pl.slice(
                x_src,
                [COPY_TOKEN_TILE, 1, D],
                [token0, hc_lane, 0],
            )
        wave_out = pl.create_tensor(
            [T, HC_MULT, D], dtype=pl.FP32, init_value=0.0
        )
        moe(
            x_moe_ready, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale, wave_out,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, effective_tokens, my_rank, pl.cast(wave + 1, pl.INT32),
        )
        x_next_work = pl.assemble(x_next_work, wave_out, [row0, 0, 0])
        stage_tokens = _record_wave_completion(
            wave_out, wave, stage_done, my_rank, stage_tokens
        )

    # Materialize an aligned rank-3 anchor for final signal cleanup.
    anchor_tile = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32, init_value=0.0)
    final_row = (NUM_MOE_WAVES - 1) * T
    with pl.spmd(
        1,
        name_hint="final_completion_anchor",
    ):
        _anchor_idx = pl.tile.get_block_idx()
        completion = pl.add(
            pl.slice(x_next_work, [1, 1, 8], [0, 0, 0]),
            pl.slice(stage_tokens, [1, 1, 8], [NUM_MOE_WAVES, 0, 0]),
        )
        for wave in pl.range(1, NUM_MOE_WAVES):
            completion = pl.add(
                completion,
                pl.slice(x_next_work, [1, 1, 8], [wave * T, 0, 0]),
            )
        anchor_tile[0:1, 0:1, 0:8] = completion
    # Publication naturally depends on all rank-local wave outputs.
    x_next = _publish_x_next_after_stage(x_next_work, active_flat, x_next)
    clear_moe_signals(anchor_tile, arrived, data_arrived, combine_arrived)
    return x_next


# ---------------------------------------------------------------------------
# Rank-local child
# ---------------------------------------------------------------------------
@pl.jit(auto_scope=False)
def prefill_layer_cp_swa(
    # CP hidden + SWA attention inputs (same set as prefill_cp_swa_test passes
    # into prefill_cp_swa_core).
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
        pl.Tensor[[ORI_MAX_BLOCKS, TAIL_ROWS, 1, HEAD_DIM], pl.BF16]
    ],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_position_ids: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_token_to_request: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_position_ids: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_token_to_request: pl.Tensor[
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
    # MoE inputs (same shapes as moe() takes, minus the windows passed below).
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # MoE distributed windows (reused across the four waves; ordered by wave
    # through the stage counter).
    recv_meta: pld.DistributedTensor[[META_WINDOW_ROWS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[ROUTED_WINDOW_ROWS, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    # Layer stage synchronization window.
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Layer output.
    x_next: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    # Scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]:
    """One CP-SWA rank: attention -> all-rank stage 1 -> four MoE waves ->
    stage 5 -> publish x_next -> clear MoE signals."""
    # Child-local attention output (NOT a TensorSpec / host output).
    x_attn = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    completion_token = pl.create_tensor(
        [NUM_MOE_WAVES, 1, 8], dtype=pl.FP32, init_value=0.0
    )
    with pl.scope():
        prefill_cp_swa_core(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin, kv_cache,
            attn_sink, wo_a, wo_b, wo_b_scale,
            segment_starts_t, predecessor_segments,
            query_position_ids, query_token_to_request,
            overlay_position_ids, overlay_token_to_request,
            overlay_active_lengths, swa_indices,
            reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            kv_tail_window, ready, consumed,
            x_attn, completion_token, my_rank, pl.cast(0, pl.INT32),
        )

    with pl.scope():
        stage_tokens = pl.create_tensor(
            [NUM_MOE_WAVES + 1, 1, 8], dtype=pl.FP32, init_value=0.0
        )
        stage_tokens = _attention_stage_barrier_from_completion(
            completion_token, stage_done, my_rank, stage_tokens
        )
        x_next = _prefill_layer_cp_moe_tail(
            x_attn, overlay_active_lengths, input_ids,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done, stage_tokens, x_next, layer_id, my_rank,
        )
    return x_next


# ---------------------------------------------------------------------------
# Rank-local child: HCA
# ---------------------------------------------------------------------------
@pl.jit(auto_scope=False)
def prefill_layer_cp_hca(
    # CP hidden + HCA attention inputs (same set as prefill_cp_hca_rank passes
    # into prefill_cp_hca_core).
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
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                HCA_STATE_PHYSICAL_BLOCKS,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
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
    cmp_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    cmp_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    cmp_meta_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32
    ],
    state_window: pld.DistributedTensor[
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], pl.FP32
    ],
    state_meta_window: pld.DistributedTensor[
        [CP_SIZE, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    # MoE inputs (same shapes as moe() takes, minus the windows passed below).
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # MoE distributed windows (reused across the four waves; ordered by wave
    # through the stage counter).
    recv_meta: pld.DistributedTensor[[META_WINDOW_ROWS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[ROUTED_WINDOW_ROWS, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    # Layer stage synchronization window.
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Layer output.
    x_next: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    # Scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]:
    """One CP-HCA rank: attention -> all-rank stage 1 -> four MoE waves ->
    stage 5 -> publish x_next -> clear MoE signals."""
    # Child-local attention output (NOT a TensorSpec / host output).
    x_attn = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    with pl.scope():
        prefill_cp_hca_core(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
            compress_state, compress_state_block_table,
            kv_cache, cmp_kv, cmp_block_table,
            segment_starts_t, segment_active_lengths,
            owner_segments_t, predecessor_segments,
            query_positions, query_requests,
            overlay_positions, overlay_requests,
            overlay_active_lengths, swa_indices, cmp_indices,
            segment_tail_positions,
            snapshot_positions, snapshot_valid, final_segment_t,
            reverse_index, owner_rank_table, owner_part_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            hidden_tail_window, kv_tail_window,
            tail_ready, tail_consumed,
            cmp_window, cmp_meta_window,
            state_window, state_meta_window,
            compact_ready, compact_consumed,
            attn_sink, wo_a, wo_b, wo_b_scale,
            x_attn, my_rank,
            pl.cast(0, pl.INT32), pl.cast(0, pl.INT32),
        )

    with pl.scope():
        x_attn_flat = pl.reshape(x_attn, [NUM_MOE_WAVES * T, HC_MULT, D])
        stage_tokens = pl.create_tensor(
            [NUM_MOE_WAVES + 1, 1, 8], dtype=pl.FP32, init_value=0.0
        )
        stage_tokens = _attention_stage_barrier_from_x_attn(
            x_attn_flat, stage_done, my_rank, stage_tokens
        )
        x_next = _prefill_layer_cp_moe_tail(
            x_attn, overlay_active_lengths, input_ids,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done, stage_tokens, x_next, layer_id, my_rank,
        )
    return x_next



@pl.jit(auto_scope=False)
def prefill_layer_cp_csa(
    # CP hidden + CSA attention inputs (same set as prefill_cp_csa_rank passes
    # into prefill_cp_csa_core; x_out becomes the child-local x_attn).
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
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
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    main_state_workspace0: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace0: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_workspace1: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace1: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    compress_state: pl.InOut[
        pl.Tensor[
            [CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[[MAIN_STATE_MAX_BLOCKS], pl.INT32],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[
        [INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[[ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[[PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]
    ],
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    query_requests: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
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
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32
    ],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[T], pl.INT32],
    final_win_row_src: pl.Tensor[[T], pl.INT32],
    final_slot_mapping: pl.Tensor[[T], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[
        [LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32
    ],
    effective_x_workspace: pl.Tensor[[LOCAL_LEAVES * T, D], pl.BF16],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    main_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, HEAD_DIM], pl.BF16
    ],
    idx_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, IDX_HEAD_DIM], pl.INT8
    ],
    scale_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, SCALE_TILE_COLS], pl.FP32
    ],
    record_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, META_DIM], pl.INT32
    ],
    main_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], pl.FP32
    ],
    main_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    inner_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], pl.FP32
    ],
    inner_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    # MoE inputs (same shapes as moe() takes, minus the windows passed below).
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64],
    routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[D], pl.FP32],
    # MoE distributed windows (reused across the four waves; ordered by wave
    # through the stage counter).
    recv_meta: pld.DistributedTensor[[META_WINDOW_ROWS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[PAYLOAD_WINDOW_ROWS, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[ROUTED_WINDOW_ROWS, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[META_WINDOW_ROWS, 1], pl.INT32],
    # Layer stage synchronization window.
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Layer output.
    x_next: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    # Scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]:
    """One CP-CSA rank: attention -> all-rank stage 1 -> four MoE waves ->
    stage 5 -> publish x_next -> clear MoE signals."""
    # Child-local attention output (NOT a TensorSpec / host output).
    x_attn = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    # The leaf publishes one row per x_out tile after its communication and
    # persistent cache/state updates have retired.
    completion_token = pl.create_tensor(
        [NUM_MOE_WAVES, 1, 8], dtype=pl.FP32, init_value=0.0
    )
    # The accepted inline attention core writes x_attn, mutates
    # kv_cache / cmp_kv / compress_state / inner_compress_state /
    # idx_kv_cache / idx_kv_scale, AND publishes completion_token via its
    # terminal cp_csa_rank_complete task.
    with pl.scope():
        prefill_cp_csa_core(
            x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale,
            wkv, gamma_cq, gamma_ckv, freqs_cos, freqs_sin, cmp_wkv, cmp_wgate, cmp_ape,
            cmp_norm_w, hadamard_idx, idx_wq_b, idx_wq_b_scale, idx_weights_proj, inner_wkv, inner_wgate, inner_ape,
            inner_norm_w, main_state_workspace0, inner_state_workspace0, main_state_workspace1, inner_state_workspace1, compress_state, compress_state_block_table, inner_compress_state,
            inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, segment_starts_t,
            segment_lengths_t, segment_active_lengths, owner_segments_t, predecessor_segments, query_positions, query_requests, overlay_positions, overlay_requests,
            overlay_active_lengths, swa_indices, final_segment_t, reverse_index, owner_rank_table, final_win_seg_src, final_win_row_src, final_slot_mapping,
            leaf_positions_input, leaf_main_slots_input, leaf_idx_slots_input, leaf_main_state_slots_input, leaf_inner_state_slots_input, leaf_num_tokens_input, effective_x_workspace, hidden_tail_window,
            kv_tail_window, tail_ready, tail_consumed, main_window, idx_window, scale_window, record_window, main_state_window,
            main_state_meta_window, inner_state_window, inner_state_meta_window, compact_ready, compact_consumed, attn_sink, wo_a, wo_b,
            wo_b_scale, x_attn, completion_token, my_rank,
            pl.cast(0, pl.INT32), pl.cast(0, pl.INT32),
        )

    # Shared MoE tail: attention boundary -> four waves -> publish x_next.
    with pl.scope():
        stage_tokens = pl.create_tensor(
            [NUM_MOE_WAVES + 1, 1, 8], dtype=pl.FP32, init_value=0.0
        )
        stage_tokens = _attention_stage_barrier_from_completion(
            completion_token, stage_done, my_rank, stage_tokens
        )
        x_next = _prefill_layer_cp_moe_tail(
            x_attn, overlay_active_lengths, input_ids,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done, stage_tokens, x_next, layer_id, my_rank,
        )
    return x_next


# ---------------------------------------------------------------------------
# Host launcher
# ---------------------------------------------------------------------------
@pl.jit.host
def l3_prefill_layer_cp_swa(
    x_hc: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
        pl.FP32,
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
        pl.Tensor[
            [CP_SIZE, ORI_MAX_BLOCKS, TAIL_ROWS, 1, HEAD_DIM], pl.BF16
        ]
    ],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    predecessor_segments: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    query_position_ids: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_token_to_request: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_position_ids: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_token_to_request: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64
    ],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[
        pl.Tensor[
            [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    layer_id: pl.Scalar[pl.INT32],
):
    """Launch one CP-SWA layer child per rank owning three window domains:
    SWA tail exchange, baseline MoE, and layer stage synchronization."""
    # Domain 1: SWA tail exchange (same as prefill_cp_swa_test).
    kv_tail_window_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    # Domain 2: baseline MoE window bank (same as l3_prefill_layer).
    recv_meta_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([PAYLOAD_WINDOW_ROWS, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer(
        [PAYLOAD_WINDOW_ROWS, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [PAYLOAD_WINDOW_ROWS, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([ROUTED_WINDOW_ROWS, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)

    # Domain 3: layer stage synchronization (monotonic counter, 1..5).
    stage_done_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        kv_tail_window = pld.window(
            kv_tail_window_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        ready = pld.window(ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        consumed = pld.window(consumed_buf, [CP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [META_WINDOW_ROWS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [PAYLOAD_WINDOW_ROWS, D], dtype=pl.INT8)
        recv_aux = pld.window(
            recv_aux_buf, [PAYLOAD_WINDOW_ROWS, AUX_PAD], dtype=pl.FP32
        )
        recv_route = pld.window(
            recv_route_buf, [PAYLOAD_WINDOW_ROWS, IDX_PAD], dtype=pl.INT32
        )
        arrived = pld.window(arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32
        )
        routed_y_buf = pld.window(routed_y_buf_buf, [ROUTED_WINDOW_ROWS, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32
        )
        stage_done = pld.window(stage_done_buf, [CP_SIZE, 1], dtype=pl.INT32)
        # SWA attention weights are shared across ranks (passed directly, not
        # rank-materialized); MoE weights are rank-sliced (resident="stacked").
        prefill_layer_cp_swa(
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
            kv_tail_window, ready, consumed,
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done,
            x_next[rank],
            layer_id, rank,
            device=rank,
        )


@pl.jit.host
def l3_prefill_layer_cp_hca(
    x_hc: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
        pl.FP32,
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
    cmp_wkv: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HEAD_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[
        pl.Tensor[
            [
                CP_SIZE,
                HCA_STATE_PHYSICAL_BLOCKS,
                HCA_STATE_BLOCK_SIZE,
                COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [CP_SIZE, HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [
                CP_SIZE,
                PREFILL_CMP_BLOCK_NUM,
                BLOCK_SIZE,
                1,
                HEAD_DIM,
            ],
            pl.BF16,
        ]
    ],
    cmp_block_table: pl.Tensor[
        [CP_SIZE, PREFILL_CMP_MAX_BLOCKS], pl.INT32
    ],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS], pl.INT32
    ],
    owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS], pl.INT32
    ],
    query_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    query_requests: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, WIN], pl.INT32
    ],
    cmp_indices: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    segment_tail_positions: pl.Tensor[
        [NUM_SEGMENTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, TAIL_ROWS], pl.INT32
    ],
    snapshot_valid: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_win_row_src: pl.Tensor[[TAIL_ROWS], pl.INT32],
    final_slot_mapping: pl.Tensor[[TAIL_ROWS], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64
    ],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[
        pl.Tensor[
            [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    layer_id: pl.Scalar[pl.INT32],
):
    """Launch one CP-HCA layer child per rank owning four window domains:
    HCA tail exchange, HCA compact/state exchange, baseline MoE, and layer
    stage synchronization."""
    # Domain 1: HCA raw-tail exchange (same as prefill_cp_hca_test).
    hidden_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
    )
    kv_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    tail_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )

    # Domain 2: HCA compact/state exchange (same as prefill_cp_hca_test).
    cmp_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    cmp_meta_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32
    )
    state_window_buf = pld.alloc_window_buffer(
        [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM], dtype=pl.FP32
    )
    state_meta_window_buf = pld.alloc_window_buffer(
        [CP_SIZE, STATE_META_DIM], dtype=pl.INT32
    )
    compact_ready_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )
    compact_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )

    # Domain 3: baseline MoE window bank (same as l3_prefill_layer_cp_swa).
    recv_meta_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([PAYLOAD_WINDOW_ROWS, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer(
        [PAYLOAD_WINDOW_ROWS, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [PAYLOAD_WINDOW_ROWS, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([ROUTED_WINDOW_ROWS, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)

    # Domain 4: layer stage synchronization (monotonic counter, 1..5).
    stage_done_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        hidden_tail_window = pld.window(
            hidden_tail_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
        )
        kv_tail_window = pld.window(
            kv_tail_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        tail_ready = pld.window(
            tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        tail_consumed = pld.window(
            tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        cmp_window = pld.window(
            cmp_window_buf, [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        cmp_meta_window = pld.window(
            cmp_meta_window_buf, [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32
        )
        state_window = pld.window(
            state_window_buf, [STATE_WINDOW_ROWS, COMPRESS_STATE_DIM],
            dtype=pl.FP32,
        )
        state_meta_window = pld.window(
            state_meta_window_buf, [CP_SIZE, STATE_META_DIM], dtype=pl.INT32
        )
        compact_ready = pld.window(
            compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        compact_consumed = pld.window(
            compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        recv_meta = pld.window(recv_meta_buf, [META_WINDOW_ROWS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [PAYLOAD_WINDOW_ROWS, D], dtype=pl.INT8)
        recv_aux = pld.window(
            recv_aux_buf, [PAYLOAD_WINDOW_ROWS, AUX_PAD], dtype=pl.FP32
        )
        recv_route = pld.window(
            recv_route_buf, [PAYLOAD_WINDOW_ROWS, IDX_PAD], dtype=pl.INT32
        )
        arrived = pld.window(arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32
        )
        routed_y_buf = pld.window(routed_y_buf_buf, [ROUTED_WINDOW_ROWS, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32
        )
        stage_done = pld.window(stage_done_buf, [CP_SIZE, 1], dtype=pl.INT32)
        # HCA attention weights are shared across ranks (passed directly, not
        # rank-materialized); persistent state and MoE weights are rank-sliced.
        prefill_layer_cp_hca(
            x_hc[rank],
            hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
            compress_state[rank], compress_state_block_table[rank],
            kv_cache[rank], cmp_kv[rank], cmp_block_table[rank],
            segment_starts_t, segment_active_lengths[rank],
            owner_segments_t[rank], predecessor_segments[rank],
            query_positions[rank], query_requests[rank],
            overlay_positions[rank], overlay_requests[rank],
            overlay_active_lengths[rank], swa_indices[rank], cmp_indices[rank],
            segment_tail_positions,
            snapshot_positions[rank], snapshot_valid[rank], final_segment_t,
            reverse_index, owner_rank_table, owner_part_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            hidden_tail_window, kv_tail_window,
            tail_ready, tail_consumed,
            cmp_window, cmp_meta_window,
            state_window, state_meta_window,
            compact_ready, compact_consumed,
            attn_sink, wo_a, wo_b, wo_b_scale,
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done,
            x_next[rank],
            layer_id, rank,
            device=rank,
        )



@pl.jit.host
def l3_prefill_layer_cp_csa(
    # CP hidden + CSA attention inputs (CP_SIZE-prefixed host ABI).
    x_hc: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
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
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    compress_state: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    compress_state_block_table: pl.Tensor[
        [CP_SIZE, MAIN_STATE_MAX_BLOCKS], pl.INT32
    ],
    inner_compress_state: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    inner_compress_state_block_table: pl.Tensor[
        [CP_SIZE, INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, ORI_MAX_BLOCKS, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_kv: pl.InOut[
        pl.Tensor[
            [CP_SIZE, PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    cmp_block_table: pl.Tensor[
        [CP_SIZE, PREFILL_CMP_MAX_BLOCKS], pl.INT32
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[
            [CP_SIZE, PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32
        ]
    ],
    idx_block_table: pl.Tensor[[CP_SIZE, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    segment_starts_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    segment_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    predecessor_segments: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    query_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    query_requests: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT32
    ],
    overlay_positions: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_requests: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_ROWS], pl.INT32
    ],
    overlay_active_lengths: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    swa_indices: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, WIN], pl.INT32
    ],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    reverse_index: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    owner_rank_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    final_win_seg_src: pl.Tensor[[T], pl.INT32],
    final_win_row_src: pl.Tensor[[T], pl.INT32],
    final_slot_mapping: pl.Tensor[[T], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_COMPRESS_LEAVES], pl.INT32
    ],
    effective_x_workspace: pl.Tensor[
        [CP_SIZE, LOCAL_LEAVES * T, D], pl.BF16
    ],
    main_state_workspace0: pl.Tensor[
        [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace0: pl.Tensor[
        [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_workspace1: pl.Tensor[
        [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace1: pl.Tensor[
        [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM],
        pl.FP32,
    ],
    # MoE inputs (rank-sliced at launch; shared where HCA host shares).
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64
    ],
    routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    x_next: pl.Out[
        pl.Tensor[
            [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    layer_id: pl.Scalar[pl.INT32],
):
    """Launch one CP-CSA layer child per rank owning three window domains:
    CSA compact exchange, baseline MoE, and layer stage synchronization."""
    # Domain 1: CSA compact/state exchange (same as prefill_cp_csa_test).
    hidden_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
    )
    kv_tail_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    tail_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    main_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, HEAD_DIM], dtype=pl.BF16
    )
    idx_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8
    )
    scale_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP32
    )
    record_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32
    )
    main_state_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM], dtype=pl.FP32
    )
    main_state_meta_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
    )
    inner_state_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM], dtype=pl.FP32
    )
    inner_state_meta_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
    )
    compact_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    compact_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    # Domain 2: baseline MoE window bank (same as l3_prefill_layer_cp_swa).
    recv_meta_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([PAYLOAD_WINDOW_ROWS, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer(
        [PAYLOAD_WINDOW_ROWS, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [PAYLOAD_WINDOW_ROWS, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([ROUTED_WINDOW_ROWS, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([META_WINDOW_ROWS, 1], dtype=pl.INT32)

    # Domain 3: layer stage synchronization (monotonic counter, 1..5).
    stage_done_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        hidden_tail_window = pld.window(
            hidden_tail_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
        )
        kv_tail_window = pld.window(
            kv_tail_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        tail_ready = pld.window(tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        tail_consumed = pld.window(
            tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        main_window = pld.window(
            main_buf, [RECORDS_PER_WINDOW, HEAD_DIM], dtype=pl.BF16
        )
        idx_window = pld.window(
            idx_buf, [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8
        )
        scale_window = pld.window(
            scale_buf, [RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP32
        )
        record_window = pld.window(
            record_buf, [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32
        )
        main_state_window = pld.window(
            main_state_buf,
            [STATE_RECORDS_PER_WINDOW, MAIN_STATE_DIM],
            dtype=pl.FP32,
        )
        main_state_meta_window = pld.window(
            main_state_meta_buf,
            [STATE_RECORDS_PER_WINDOW, STATE_META_DIM],
            dtype=pl.INT32,
        )
        inner_state_window = pld.window(
            inner_state_buf,
            [STATE_RECORDS_PER_WINDOW, INNER_STATE_DIM],
            dtype=pl.FP32,
        )
        inner_state_meta_window = pld.window(
            inner_state_meta_buf,
            [STATE_RECORDS_PER_WINDOW, STATE_META_DIM],
            dtype=pl.INT32,
        )
        compact_ready = pld.window(
            compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        compact_consumed = pld.window(
            compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        # MoE window bindings (baseline bank).
        recv_meta = pld.window(recv_meta_buf, [META_WINDOW_ROWS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [PAYLOAD_WINDOW_ROWS, D], dtype=pl.INT8)
        recv_aux = pld.window(
            recv_aux_buf, [PAYLOAD_WINDOW_ROWS, AUX_PAD], dtype=pl.FP32
        )
        recv_route = pld.window(
            recv_route_buf, [PAYLOAD_WINDOW_ROWS, IDX_PAD], dtype=pl.INT32
        )
        arrived = pld.window(arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32
        )
        routed_y_buf = pld.window(routed_y_buf_buf, [ROUTED_WINDOW_ROWS, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buf, [META_WINDOW_ROWS, 1], dtype=pl.INT32
        )
        stage_done = pld.window(stage_done_buf, [CP_SIZE, 1], dtype=pl.INT32)

        # CSA attention weights are shared across ranks (passed directly, not
        # rank-materialized); persistent state and MoE weights are rank-sliced.
        prefill_layer_cp_csa(
            x_hc[rank],
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
            cmp_wkv,
            cmp_wgate,
            cmp_ape,
            cmp_norm_w,
            hadamard_idx,
            idx_wq_b,
            idx_wq_b_scale,
            idx_weights_proj,
            inner_wkv,
            inner_wgate,
            inner_ape,
            inner_norm_w,
            main_state_workspace0[rank],
            inner_state_workspace0[rank],
            main_state_workspace1[rank],
            inner_state_workspace1[rank],
            compress_state[rank],
            compress_state_block_table[rank],
            inner_compress_state[rank],
            inner_compress_state_block_table[rank],
            kv_cache[rank],
            cmp_kv[rank],
            cmp_block_table[rank],
            idx_kv_cache[rank],
            idx_kv_scale[rank],
            idx_block_table[rank],
            segment_starts_t,
            segment_lengths_t,
            segment_active_lengths[rank],
            owner_segments_t[rank],
            predecessor_segments[rank],
            query_positions[rank],
            query_requests[rank],
            overlay_positions[rank],
            overlay_requests[rank],
            overlay_active_lengths[rank],
            swa_indices[rank],
            final_segment_t,
            reverse_index,
            owner_rank_table,
            final_win_seg_src,
            final_win_row_src,
            final_slot_mapping,
            leaf_positions_input[rank],
            leaf_main_slots_input[rank],
            leaf_idx_slots_input[rank],
            leaf_main_state_slots_input[rank],
            leaf_inner_state_slots_input[rank],
            leaf_num_tokens_input[rank],
            effective_x_workspace[rank],
            hidden_tail_window,
            kv_tail_window,
            tail_ready,
            tail_consumed,
            main_window,
            idx_window,
            scale_window,
            record_window,
            main_state_window,
            main_state_meta_window,
            inner_state_window,
            inner_state_meta_window,
            compact_ready,
            compact_consumed,
            attn_sink,
            wo_a,
            wo_b,
            wo_b_scale,
            # MoE inputs (rank-sliced).
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done,
            x_next[rank],
            layer_id, rank,
            device=rank,
        )



# ---------------------------------------------------------------------------
# TensorSpec composition
# ---------------------------------------------------------------------------
# Host argument order for the SWA layer host (l3_prefill_layer_cp_swa).
SWA_HOST_ARG_ORDER = (
    "x_hc",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "segment_starts_t", "predecessor_segments",
    "query_position_ids", "query_token_to_request",
    "overlay_position_ids", "overlay_token_to_request",
    "overlay_active_lengths", "swa_indices", "reverse_index",
    "owner_rank_table", "final_win_seg_src", "final_win_row_src",
    "final_slot_mapping",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale", "x_next", "layer_id",
)

# Host argument order for the HCA layer host (l3_prefill_layer_cp_hca).
HCA_HOST_ARG_ORDER = (
    "x_hc",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin",
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state", "compress_state_block_table",
    "kv_cache", "cmp_kv", "cmp_block_table",
    "segment_starts_t", "segment_active_lengths",
    "owner_segments_t", "predecessor_segments",
    "query_positions", "query_requests",
    "overlay_positions", "overlay_requests",
    "overlay_active_lengths", "swa_indices", "cmp_indices",
    "segment_tail_positions",
    "snapshot_positions", "snapshot_valid", "final_segment_t",
    "reverse_index", "owner_rank_table", "owner_part_table",
    "final_win_seg_src", "final_win_row_src", "final_slot_mapping",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale", "x_next", "layer_id",
)


def _dedup_specs(specs):
    """Reject duplicate spec names instead of silently replacing a spec."""
    by_name = {}
    for spec in specs:
        if spec.name in by_name:
            raise ValueError(
                f"duplicate TensorSpec name {spec.name!r}; "
                f"resolve the collision explicitly instead of last-wins"
            )
        by_name[spec.name] = spec
    return by_name


def _build_moe_specs(layer_id: int):
    """Append baseline MoE weights/tables, discarding x_hc / x_next /
    input_ids and the two scalar specs (layer_id, num_tokens)."""
    moe_specs = build_moe_tensor_specs(layer_id=layer_id, num_tokens=T)
    moe_keep = []
    for spec in moe_specs:
        if not isinstance(spec, TensorSpec):
            continue  # drop ScalarSpecs (layer_id, num_tokens)
        if spec.name in {"x_hc", "x_next", "input_ids"}:
            continue
        moe_keep.append(spec)
    return moe_keep


def _build_input_ids_spec(cp_size: int, active_lengths_spec, prefix_seed: int):
    """Deterministic structured input_ids [CP_SIZE, 2, 2, T]. Keep inactive
    rows zero so an empty wave's protocol-only token routes deterministically."""
    torch.manual_seed(prefix_seed)
    ids = torch.randint(0, VOCAB, (cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T),
                        dtype=torch.int64)
    active_lengths = active_lengths_spec.create_tensor()
    for rank in range(cp_size):
        for part in range(LOCAL_PARTS):
            for tile in range(MAX_SEGMENT_TILES):
                active = int(active_lengths[rank, part, tile, 1])
                ids[rank, part, tile, active:] = 0
    return TensorSpec(
        "input_ids",
        [cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T],
        torch.int64, init_value=ids,
    )


def _build_swa_specs(layer_id: int, cp_size: int):
    """Compose SWA CP attention specs with baseline MoE specs."""
    swa_specs, ctx = build_swa_tensor_specs(cp_size)
    # Drop the attention x_out (it is an internal layer temporary).
    swa_specs = [s for s in swa_specs if s.name != "x_out"]

    moe_keep = _build_moe_specs(layer_id)

    active_lengths_spec = next(
        spec for spec in swa_specs if spec.name == "overlay_active_lengths"
    )
    input_ids_spec = _build_input_ids_spec(
        cp_size, active_lengths_spec, 4100 + cp_size * 31
    )
    x_next_spec = TensorSpec(
        "x_next",
        [cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        torch.float32, is_output=True,
    )
    layer_id_spec = ScalarSpec("layer_id", torch.int32, layer_id)

    all_specs = list(swa_specs) + moe_keep + [input_ids_spec, x_next_spec, layer_id_spec]
    by_name = _dedup_specs(all_specs)
    missing = [name for name in SWA_HOST_ARG_ORDER if name not in by_name]
    extra = [name for name in by_name if name not in SWA_HOST_ARG_ORDER]
    if missing or extra:
        raise ValueError(
            f"SWA layer CP host ABI mismatch: missing={missing}, extra={extra}"
        )
    ordered = [by_name[name] for name in SWA_HOST_ARG_ORDER]
    return ordered, ctx


def _build_hca_specs(layer_id: int, cp_size: int):
    """Compose HCA CP attention specs with baseline MoE specs."""
    hca_specs = build_hca_tensor_specs(cp_size)
    # HCA build_tensor_specs returns a plain list (no ctx); it installs
    # golden_prefill_cp_hca._ctx internally. Use None as the ctx sentinel.
    ctx = None
    # Drop the attention x_out (it is an internal layer temporary).
    hca_specs = [s for s in hca_specs if s.name != "x_out"]

    moe_keep = _build_moe_specs(layer_id)

    active_lengths_spec = next(
        spec for spec in hca_specs if spec.name == "overlay_active_lengths"
    )
    input_ids_spec = _build_input_ids_spec(
        cp_size, active_lengths_spec, 4100 + cp_size * 31
    )
    x_next_spec = TensorSpec(
        "x_next",
        [cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        torch.float32, is_output=True,
    )
    layer_id_spec = ScalarSpec("layer_id", torch.int32, layer_id)

    all_specs = list(hca_specs) + moe_keep + [input_ids_spec, x_next_spec, layer_id_spec]
    by_name = _dedup_specs(all_specs)
    missing = [name for name in HCA_HOST_ARG_ORDER if name not in by_name]
    extra = [name for name in by_name if name not in HCA_HOST_ARG_ORDER]
    if missing or extra:
        raise ValueError(
            f"HCA layer CP host ABI mismatch: missing={missing}, extra={extra}"
        )
    ordered = [by_name[name] for name in HCA_HOST_ARG_ORDER]
    return ordered, ctx


CSA_HOST_ARG_ORDER = (
    "x_hc", "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "attn_norm_w", "wq_a", "wq_b", "wq_b_scale",
    "wkv", "gamma_cq", "gamma_ckv", "freqs_cos",
    "freqs_sin", "cmp_wkv", "cmp_wgate", "cmp_ape",
    "cmp_norm_w", "hadamard_idx", "idx_wq_b", "idx_wq_b_scale",
    "idx_weights_proj", "inner_wkv", "inner_wgate", "inner_ape",
    "inner_norm_w", "attn_sink", "wo_a", "wo_b",
    "wo_b_scale", "compress_state", "compress_state_block_table", "inner_compress_state",
    "inner_compress_state_block_table", "kv_cache", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table", "segment_starts_t",
    "segment_lengths_t", "segment_active_lengths", "owner_segments_t", "predecessor_segments",
    "query_positions", "query_requests", "overlay_positions", "overlay_requests",
    "overlay_active_lengths", "swa_indices", "final_segment_t", "reverse_index",
    "owner_rank_table", "final_win_seg_src", "final_win_row_src", "final_slot_mapping",
    "leaf_positions_input", "leaf_main_slots_input", "leaf_idx_slots_input", "leaf_main_state_slots_input",
    "leaf_inner_state_slots_input", "leaf_num_tokens_input", "effective_x_workspace", "main_state_workspace0",
    "inner_state_workspace0", "main_state_workspace1", "inner_state_workspace1", "hc_ffn_fn",
    "hc_ffn_scale", "hc_ffn_base", "norm_w", "gate_w",
    "gate_bias", "tid2eid", "input_ids", "routed_w1",
    "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2",
    "routed_w2_scale", "shared_w1", "shared_w1_scale", "shared_w3",
    "shared_w3_scale", "shared_w2", "shared_w2_scale", "x_next",
    "layer_id",
)

def _build_csa_specs(layer_id: int, cp_size: int):
    """Compose CSA CP attention specs with baseline MoE specs."""
    csa_specs = build_csa_tensor_specs(cp_size)
    # CSA build_tensor_specs returns a plain list (no ctx); it installs
    # golden_prefill_cp_csa._ctx internally. Use None as the ctx sentinel.
    ctx = None
    # Drop the attention x_out (it is an internal layer temporary).
    csa_specs = [s for s in csa_specs if s.name != "x_out"]

    moe_keep = _build_moe_specs(layer_id)

    active_lengths_spec = next(
        spec for spec in csa_specs if spec.name == "overlay_active_lengths"
    )
    input_ids_spec = _build_input_ids_spec(
        cp_size, active_lengths_spec, 4100 + cp_size * 31
    )
    x_next_spec = TensorSpec(
        "x_next",
        [cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        torch.float32, is_output=True,
    )
    layer_id_spec = ScalarSpec("layer_id", torch.int32, layer_id)

    all_specs = list(csa_specs) + moe_keep + [input_ids_spec, x_next_spec, layer_id_spec]
    by_name = _dedup_specs(all_specs)
    missing = [name for name in CSA_HOST_ARG_ORDER if name not in by_name]
    extra = [name for name in by_name if name not in CSA_HOST_ARG_ORDER]
    if missing or extra:
        raise ValueError(
            f"CSA layer CP host ABI mismatch: missing={missing}, extra={extra}"
        )
    ordered = [by_name[name] for name in CSA_HOST_ARG_ORDER]
    return ordered, ctx




def build_tensor_specs(layer_id: int = SWA_LAYER_ID, cp_size: int = CP_SIZE):
    """Compose selected CP attention specs with baseline MoE specs by layer kind."""
    _assert_layer_id(layer_id)
    if layer_id == SWA_LAYER_ID:
        return _build_swa_specs(layer_id, cp_size)
    if layer_id == HCA_LAYER_ID:
        return _build_hca_specs(layer_id, cp_size)
    if layer_id == CSA_LAYER_ID:
        return _build_csa_specs(layer_id, cp_size)
    raise RuntimeError(f"unsupported layer_id={layer_id}")


# ---------------------------------------------------------------------------
# Golden orchestration
# ---------------------------------------------------------------------------
def _golden_moe_waves(tensors, x_attn, cp, x_next):
    """Replay four baseline golden MoE waves over the structured attention
    output, copying only active rows into x_next."""
    import torch as _torch

    overlay = tensors["overlay_active_lengths"]
    for wave in range(NUM_MOE_WAVES):
        part = wave // MAX_SEGMENT_TILES
        tile = wave % MAX_SEGMENT_TILES
        # Canonical active count across ranks for this wave.
        actives = [int(overlay[r, part, tile, 1]) for r in range(cp)]
        active = actives[0]
        for a in actives[1:]:
            if a != active:
                raise RuntimeError(
                    f"golden: wave {wave} active count differs across ranks "
                    f"({actives}); fixture must be rank-uniform per wave"
                )
        effective = max(active, 1)

        # Build per-wave MoE tensors: x_hc = this wave's attention slice,
        # input_ids = this wave's slice, num_tokens = effective.
        moe_tensors = dict(tensors)
        moe_tensors["x_hc"] = x_attn[:, part, tile]  # [cp, T, HC_MULT, D]
        ids_tile = _torch.zeros(cp, T, dtype=_torch.int64)
        if active > 0:
            ids_tile[:, :active] = tensors["input_ids"][:, part, tile, :active]
        else:
            # Empty wave: zero first row (matches kernel effective_tokens=1).
            ids_tile[:, 0] = 0
        moe_tensors["input_ids"] = ids_tile
        moe_tensors["num_tokens"] = effective
        x_next_wave = _torch.zeros(cp, T, HC_MULT, D, dtype=_torch.float32)
        moe_tensors["x_next"] = x_next_wave
        golden_moe(moe_tensors)

        # Copy only true active rows into the structured output; inactive
        # rows remain zero.
        if active > 0:
            x_next[:, part, tile, :active] = x_next_wave[:, :active]


def golden_prefill_layer_cp(tensors):
    """Compose selected CP attention golden with baseline MoE golden.
    Dispatches by layer_id scalar."""
    import torch as _torch

    cp = tensors["x_hc"].shape[0]
    layer_id = int(tensors["layer_id"])

    # Temporary structured attention output (NOT a compare target).
    x_attn = _torch.zeros(
        cp, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D, dtype=_torch.float32
    )

    x_next = _torch.zeros(
        cp, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D, dtype=_torch.float32
    )

    if layer_id == SWA_LAYER_ID:
        # SWA golden context is installed by __main__ before run_jit.
        swa_tensors = dict(tensors)
        swa_tensors["x_out"] = x_attn
        golden_prefill_cp_swa(swa_tensors)
        x_attn = swa_tensors["x_out"]
        # SWA golden mutates tensors["kv_cache"] in place; that stands.
    elif layer_id == HCA_LAYER_ID:
        # HCA golden context is installed by __main__ before run_jit.
        hca_tensors = dict(tensors)
        hca_tensors["x_out"] = x_attn
        golden_prefill_cp_hca(hca_tensors)
        x_attn = hca_tensors["x_out"]
        # HCA golden mutates tensors["compress_state"], ["cmp_kv"],
        # ["kv_cache"] in place; those side effects stand.
    elif layer_id == CSA_LAYER_ID:
        # CSA golden context is installed by build_csa_tensor_specs inside
        # build_tensor_specs; no extra install needed here.
        csa_tensors = dict(tensors)
        csa_tensors["x_out"] = x_attn
        golden_prefill_cp_csa(csa_tensors)
        x_attn = csa_tensors["x_out"]
        # CSA golden mutates tensors["kv_cache"], ["cmp_kv"],
        # ["compress_state"], ["inner_compress_state"],
        # ["idx_kv_cache"], ["idx_kv_scale"] in place; those side effects stand.
    else:
        raise RuntimeError(f"golden: unsupported layer_id={layer_id}")

    _golden_moe_waves(tensors, x_attn, cp, x_next)
    tensors["x_next"] = x_next


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek V4 context-parallel prefill single layer (SWA/HCA/CSA)."
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument(
        "-d", "--device", type=str,
        default=",".join(str(i) for i in range(CP_SIZE)),
        help=f"comma-separated device ids; need at least {CP_SIZE}",
    )
    parser.add_argument(
        "--cp", type=int, default=CP_SIZE, choices=list(CP_CHOICES),
        help="context-parallel world size (parsed at import by prefill_cp_zigzag)",
    )
    parser.add_argument(
        "--ep", type=int, default=N_RANKS, choices=[2, 4, 8],
        help="expert-parallel world size (parsed at import by moe)",
    )
    parser.add_argument(
        "--layer-id", type=int, default=SWA_LAYER_ID,
        help="layer 0 = SWA, layer 2 = CSA, layer 3 = HCA",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--enable-dep-gen", action="store_true", default=False)
    parser.add_argument("--no-golden", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    _assert_layer_id(args.layer_id)

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(
            f"CP{args.cp} requires {args.cp} devices, got {device_ids}"
        )

    specs, ctx = build_tensor_specs(layer_id=args.layer_id, cp_size=args.cp)

    if args.layer_id == SWA_LAYER_ID:
        # SWA golden reads its fixture context via this attribute.
        golden_prefill_cp_swa._ctx = ctx
        host_fn = l3_prefill_layer_cp_swa
        compare_fn = {
            "x_next": ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05, max_diff_hd=float("inf")
            ),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1e-2),
        }
    elif args.layer_id == HCA_LAYER_ID:
        # HCA golden context is installed by build_hca_tensor_specs inside
        # build_tensor_specs; no extra install needed here.
        host_fn = l3_prefill_layer_cp_hca
        compare_fn = {
            "x_next": ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05, max_diff_hd=float("inf")
            ),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "cmp_kv": ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005
            ),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3),
        }
    elif args.layer_id == CSA_LAYER_ID:
        # CSA golden context is installed by build_csa_tensor_specs inside
        # build_tensor_specs; no extra install needed here.
        host_fn = l3_prefill_layer_cp_csa
        compare_fn = {
            "x_next": ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05, max_diff_hd=float("inf")
            ),
            "kv_cache": ratio_allclose(atol=1e-4, rtol=1.0 / 128),
            "cmp_kv": ratio_allclose(
                atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.005
            ),
            "compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.005),
            "inner_compress_state": ratio_allclose(atol=1e-3, rtol=1e-3, max_error_ratio=0.005),
            "idx_kv_cache": ratio_allclose(atol=1, rtol=0, max_error_ratio=0.01),
            "idx_kv_scale": ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.01),
        }
    else:
        raise RuntimeError(f"unsupported layer_id={args.layer_id}")

    result = run_jit(
        fn=host_fn,
        specs=specs,
        golden_fn=None if args.no_golden else golden_prefill_layer_cp,
        compile_only=args.compile_only,
        compile_cfg=dict(
            distributed_config=DistributedConfig(
                device_ids=device_ids[:args.cp], num_sub_workers=0
            ),
            dump_passes=args.dump_passes,
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_dep_gen=args.enable_dep_gen,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
