# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: EP2/TP2 fused serving-step run
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim
"""Fused DeepSeek-V4 main decode, token verification, and MTP decode orchestration."""

import argparse
from dataclasses import replace

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig

from decode_fwd import (
    AUX_PAD,
    B,
    BLOCK_SIZE,
    CSA_CMP_BLOCK_NUM,
    CSA_CMP_MAX_BLOCKS,
    CSA_COMPRESS_RATIO,
    CSA_IDX_CACHE_BLOCK_NUM,
    CSA_IDX_CACHE_MAX_BLOCKS,
    CSA_IDX_HEAD_DIM,
    CSA_IDX_N_HEADS,
    CSA_INNER_OUT_DIM,
    CSA_INNER_STATE_BLOCK_NUM,
    CSA_INNER_STATE_BLOCK_SIZE,
    CSA_INNER_STATE_DIM,
    CSA_INNER_STATE_MAX_BLOCKS,
    CSA_MAIN_OUT_DIM,
    CSA_MAIN_STATE_BLOCK_NUM,
    CSA_MAIN_STATE_BLOCK_SIZE,
    CSA_MAIN_STATE_DIM,
    CSA_MAIN_STATE_MAX_BLOCKS,
    CSA_NUM_LAYERS,
    D,
    DECODE_START_POS,
    FWD_CSA_CMP_BLOCK_NUM_DYN,
    FWD_CSA_STATE_BLOCK_NUM_DYN,
    FWD_HCA_CMP_BLOCK_NUM_DYN,
    FWD_HCA_STATE_BLOCK_NUM_DYN,
    FWD_IDX_BLOCK_NUM_DYN,
    FWD_INNER_STATE_BLOCK_NUM_DYN,
    FWD_NUM_LAYERS,
    FWD_ORI_BLOCK_NUM_DYN,
    GROUP_LOGIT_ROWS,
    H,
    HCA_COMPRESS_RATIO,
    HCA_COMPRESS_STATE_BLOCK_NUM,
    HCA_COMPRESS_STATE_BLOCK_SIZE,
    HCA_COMPRESS_STATE_DIM,
    HCA_COMPRESS_STATE_MAX_BLOCKS,
    HCA_CMP_MAX_BLOCKS,
    HCA_CMP_STORAGE_BLOCK_SIZE,
    HCA_MAIN_OUT_DIM,
    HCA_NUM_LAYERS,
    HC_DIM,
    HC_MULT,
    HEAD_DIM,
    IDX_PAD,
    LM_HEAD_TP_SIZE,
    LM_HEAD_VOCAB,
    MAX_LOGIT_ROWS,
    MAX_SEQ_LEN,
    MIX_HC,
    MOE_INTER,
    N_CACHE_GROUPS,
    N_EXPERTS_GLOBAL,
    N_LOCAL,
    N_RANKS,
    N_ROUTES,
    ORI_BLOCK_NUM,
    ORI_TABLE_MAX_BLOCKS,
    O_GROUPS,
    O_GROUP_IN,
    O_LORA,
    PREAMBLE_OUTPUT_NAMES,
    Q_LORA,
    RECV_MAX,
    ROPE_HEAD_DIM,
    SAMPLED_IDS_PAD,
    SWA_WIN,
    CSA_CMP_STORAGE_BLOCK_SIZE,
    T,
    TOPK,
    VOCAB,
    VOCAB_PER_TP,
    build_preamble_tensor_specs as build_decode_fwd_preamble_specs,
    build_tensor_specs as build_decode_fwd_tensor_specs,
    decode_fwd,
)
from decode_mtp import (
    MOE_TOPK,
    MOE_VOCAB,
    ORI_BLOCK_NUM_DYN as MTP_ORI_BLOCK_NUM_DYN,
    build_tensor_specs as build_decode_mtp_tensor_specs,
    decode_mtp,
)
from decode_prepare import (
    VOCAB_DYN as EMBED_VOCAB_DYN,
    build_decode_metadata,
    build_swa_metadata,
    pack_mtp_hidden,
    pack_x_hc,
)
from lookup_embedding import lookup_embedding

# decode_fwd re-exports B and T but not the per-step token count.
from config import DECODE_SEQ as S

# Persistent MTP serving state, indexed by serving-assigned slot rather than by
# the request's transient decode-batch row. For S == 2, flattened rows 2*r and
# 2*r+1 belong to local batch row r.
# Initialization guard: set on first assignment, not cleared on release.
STATE_VALID = 0
# ABA guard: incremented when a freed slot is reassigned.
STATE_GENERATION = 1
STATE_TAIL_POSITION = 2
STATE_COMMITTED_COUNT = 3
STATE_TEMPERATURE_MICROS = 4
STATE_TOP_K = 5
STATE_SEED = 6
STATE_META_WIDTH = 7
TEMPERATURE_SCALE = 1000000.0

STATE_TAIL_TOKEN = 0
STATE_DRAFT_TOKEN = 1
STATE_TOKEN_WIDTH = 2

assert S == 2, "persistent MTP state requires decode_seq=2"
assert MAX_LOGIT_ROWS >= T, "verification reads one sampled row per decode token"


@pl.jit.inline
def prepare_decode_from_device_state(
    state_slot_ids: pl.Tensor[[B], pl.INT32],
    state_generations: pl.Tensor[[B], pl.INT32],
    state_tokens: pl.Tensor[[B, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[B, STATE_META_WIDTH], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    tail_token_ids: pl.Tensor[[B], pl.INT64],
    tail_positions: pl.Tensor[[B], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
):
    """Late-bind recurrent decode fields from stable device slots::

        row0 = 2*r:   input = tail,  position = tail_position + 1
        row1 = 2*r+1: input = draft, position = tail_position + 2
        kv_seq_lens[r] = tail_position + 3

    Positions are zero-based, so tail_position + 3 is the KV length once both
    rows are counted. A row is consumed only when slot_id >= 0, the slot is
    valid, and its generation matches; every other row keeps the caller's
    padded value.
    """
    # One core owns these tightly packed scalar updates.  The metadata volume
    # is tiny, and single ownership avoids adjacent scalar DMA write races.
    for core in pl.spmd(1, name_hint="mtp_state_prepare"):
        default_temperature = pl.cast(0.0, pl.FP32)
        default_top_k = pl.cast(VOCAB, pl.INT32)
        default_seed = pl.cast(0, pl.INT32)
        default_position = pl.cast(0, pl.INT32)
        for request in pl.range(core, B):
            row0 = request * S
            row1 = row0 + 1
            pl.write(sampling_temperatures, [row0], default_temperature)
            pl.write(sampling_temperatures, [row1], default_temperature)
            pl.write(sampling_top_ks, [row0], default_top_k)
            pl.write(sampling_top_ks, [row1], default_top_k)
            pl.write(sampling_seeds, [row0], default_seed)
            pl.write(sampling_seeds, [row1], default_seed)
            pl.write(sampling_positions, [row0], default_position)
            pl.write(sampling_positions, [row1], default_position)
            slot_raw = pl.read(state_slot_ids, [request])
            if slot_raw >= 0:
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    tail_token = pl.read(state_tokens, [slot, STATE_TAIL_TOKEN])
                    draft_token = pl.read(state_tokens, [slot, STATE_DRAFT_TOKEN])
                    tail_position = pl.read(state_meta, [slot, STATE_TAIL_POSITION])
                    temperature_micros = pl.read(state_meta, [slot, STATE_TEMPERATURE_MICROS])
                    temperature_fp32 = pl.cast(temperature_micros, pl.FP32)
                    temperature = pl.div(temperature_fp32, TEMPERATURE_SCALE)
                    top_k = pl.read(state_meta, [slot, STATE_TOP_K])
                    seed = pl.read(state_meta, [slot, STATE_SEED])
                    pl.write(input_ids, [row0], tail_token)
                    pl.write(input_ids, [row1], draft_token)
                    pl.write(
                        position_ids,
                        [row0],
                        pl.cast(tail_position + 1, target_type=pl.INT32),
                    )
                    pl.write(
                        position_ids,
                        [row1],
                        pl.cast(tail_position + 2, target_type=pl.INT32),
                    )
                    pl.write(
                        kv_seq_lens,
                        [request],
                        pl.cast(tail_position + 3, target_type=pl.INT32),
                    )
                    pl.write(tail_token_ids, [request], tail_token)
                    pl.write(tail_positions, [request], tail_position)
                    pl.write(sampling_temperatures, [row0], temperature)
                    pl.write(sampling_temperatures, [row1], temperature)
                    pl.write(sampling_top_ks, [row0], top_k)
                    pl.write(sampling_top_ks, [row1], top_k)
                    pl.write(sampling_seeds, [row0], seed)
                    pl.write(sampling_seeds, [row1], seed)
                    sampling_position0 = pl.cast(tail_position + 2, target_type=pl.INT32)
                    sampling_position1 = pl.cast(tail_position + 3, target_type=pl.INT32)
                    pl.write(sampling_positions, [row0], sampling_position0)
                    pl.write(sampling_positions, [row1], sampling_position1)
    return input_ids, position_ids, kv_seq_lens, tail_token_ids, tail_positions


@pl.jit.inline
def prepare_mtp_sampling_from_device_state(
    state_slot_ids: pl.Tensor[[B], pl.INT32],
    state_generations: pl.Tensor[[B], pl.INT32],
    state_meta: pl.Tensor[[B, STATE_META_WIDTH], pl.INT32],
    position_ids: pl.Tensor[[T], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
):
    """Build next-draft sampling controls from persistent request state."""
    for core in pl.spmd(1, name_hint="mtp_state_sampling"):
        default_temperature = pl.cast(0.0, pl.FP32)
        default_top_k = pl.cast(VOCAB, pl.INT32)
        default_seed = pl.cast(0, pl.INT32)
        default_position = pl.cast(0, pl.INT32)
        for row in pl.range(core, MAX_LOGIT_ROWS):
            pl.write(sampling_temperatures, [row], default_temperature)
            pl.write(sampling_top_ks, [row], default_top_k)
            pl.write(sampling_seeds, [row], default_seed)
            pl.write(sampling_positions, [row], default_position)
        for request in pl.range(core, B):
            slot_raw = pl.read(state_slot_ids, [request])
            if slot_raw >= 0:
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    temperature_micros = pl.read(state_meta, [slot, STATE_TEMPERATURE_MICROS])
                    temperature_fp32 = pl.cast(temperature_micros, pl.FP32)
                    temperature = pl.div(temperature_fp32, TEMPERATURE_SCALE)
                    top_k = pl.read(state_meta, [slot, STATE_TOP_K])
                    seed = pl.read(state_meta, [slot, STATE_SEED])
                    row1 = request * S + 1
                    row1_position = pl.read(position_ids, [row1])
                    sampling_position = pl.cast(row1_position + 1, target_type=pl.INT32)
                    pl.write(sampling_temperatures, [request], temperature)
                    pl.write(sampling_top_ks, [request], top_k)
                    pl.write(sampling_seeds, [request], seed)
                    pl.write(sampling_positions, [request], sampling_position)
    return sampling_temperatures, sampling_top_ks, sampling_seeds, sampling_positions


@pl.jit.inline
def advance_decode_device_state(
    state_slot_ids: pl.Tensor[[B], pl.INT32],
    state_generations: pl.Tensor[[B], pl.INT32],
    state_tokens: pl.Tensor[[B, STATE_TOKEN_WIDTH], pl.INT64],
    state_meta: pl.Tensor[[B, STATE_META_WIDTH], pl.INT32],
    committed_input_ids: pl.Tensor[[T], pl.INT64],
    committed_position_ids: pl.Tensor[[T], pl.INT32],
    next_sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    accepted_counts: pl.Tensor[[B], pl.INT32],
):
    """Commit verifier and draft-model outputs into their stable slots.

    verify_and_pack_mtp_tokens has already normalized both outcomes into a
    two-row committed window for request r::

        draft accepted (accepted=2): [main0, main1]
        draft rejected (accepted=1): [old_tail, main0]

    Row 2*r+1 is the newest committed tail either way. It becomes the next
    step's tail, the MTP sample becomes the next draft, and committed_count
    advances by accepted. Guarded by the same valid bit and generation match
    as the prepare side.
    """
    # Keep the state transition single-owner for the same scalar-write reason
    # as the prepare helper above.
    for core in pl.spmd(1, name_hint="mtp_state_advance"):
        for request in pl.range(core, B):
            slot_raw = pl.read(state_slot_ids, [request])
            if slot_raw >= 0:
                slot = pl.cast(slot_raw, target_type=pl.INDEX)
                valid = pl.read(state_meta, [slot, STATE_VALID])
                generation = pl.read(state_meta, [slot, STATE_GENERATION])
                expected = pl.read(state_generations, [request])
                if valid == 1 and generation == expected:
                    # Verification always packs the newest committed token and
                    # position into the second row, regardless of acceptance.
                    row1 = request * S + 1
                    accepted = pl.read(accepted_counts, [request])
                    next_draft = pl.cast(
                        pl.read(next_sampled_ids, [request, 0]),
                        target_type=pl.INT64,
                    )
                    pl.write(
                        state_tokens,
                        [slot, STATE_TAIL_TOKEN],
                        pl.read(committed_input_ids, [row1]),
                    )
                    pl.write(state_tokens, [slot, STATE_DRAFT_TOKEN], next_draft)
                    pl.write(
                        state_meta,
                        [slot, STATE_TAIL_POSITION],
                        pl.read(committed_position_ids, [row1]),
                    )
                    committed = pl.read(state_meta, [slot, STATE_COMMITTED_COUNT])
                    pl.write(
                        state_meta,
                        [slot, STATE_COMMITTED_COUNT],
                        committed + accepted,
                    )
    return state_tokens, state_meta


@pl.jit.inline
def verify_and_pack_mtp_tokens(
    main_input_ids: pl.Tensor[[T], pl.INT64],
    main_position_ids: pl.Tensor[[T], pl.INT32],
    main_sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
    tail_token_ids: pl.Tensor[[B], pl.INT64],
    tail_positions: pl.Tensor[[B], pl.INT32],
    tail_slot_ids: pl.Tensor[[B], pl.INT32],
    mtp_input_ids: pl.Tensor[[T], pl.INT64],
    mtp_position_ids: pl.Tensor[[T], pl.INT32],
    accepted_counts: pl.Tensor[[B], pl.INT32],
):
    """Verify one draft per row and pack the two-token MTP committed window."""
    # Keep the tightly packed scalar stores on one core. Multiple SPMD cores
    # writing adjacent scalar addresses can race through overlapping DMA units.
    for verify_core in pl.spmd(1, name_hint="mtp_verify_and_pack"):
        for request in pl.range(verify_core, B):
            row0 = request * S
            row1 = row0 + 1
            slot = pl.read(tail_slot_ids, [request])
            if slot >= 0:
                draft = pl.read(main_input_ids, [row1])
                main0 = pl.cast(pl.read(main_sampled_ids, [row0, 0]), pl.INT64)
                main1 = pl.cast(pl.read(main_sampled_ids, [row1, 0]), pl.INT64)
                accepted = pl.cast(1, pl.INT32)
                committed0 = pl.read(tail_token_ids, [request])
                committed1 = main0
                position0 = pl.read(tail_positions, [request])
                position1 = pl.read(main_position_ids, [row0])
                if draft == main0:
                    accepted = pl.cast(2, pl.INT32)
                    committed0 = main0
                    committed1 = main1
                    position0 = pl.read(main_position_ids, [row0])
                    position1 = pl.read(main_position_ids, [row1])
                pl.write(accepted_counts, [request], accepted)
                pl.write(mtp_input_ids, [row0], committed0)
                pl.write(mtp_input_ids, [row1], committed1)
                pl.write(mtp_position_ids, [row0], position0)
                pl.write(mtp_position_ids, [row1], position1)
            else:
                pl.write(accepted_counts, [request], pl.cast(1, pl.INT32))
                pl.write(mtp_input_ids, [row0], pl.read(main_input_ids, [row0]))
                pl.write(mtp_input_ids, [row1], pl.read(main_input_ids, [row1]))
                pl.write(
                    mtp_position_ids,
                    [row0],
                    pl.read(main_position_ids, [row0]),
                )
                pl.write(
                    mtp_position_ids,
                    [row1],
                    pl.read(main_position_ids, [row1]),
                )
    return mtp_input_ids, mtp_position_ids, accepted_counts

@pl.jit(auto_scope=False)
def l2_decode_fwd_mtp(
    embed_weight: pl.Tensor[[EMBED_VOCAB_DYN, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[pl.Tensor[[FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[pl.Tensor[[FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    csa_idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[pl.Tensor[[FWD_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    hca_cmp_kv: pl.InOut[pl.Tensor[[FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    idx_kv_cache: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
    freqs_cos: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.InOut[pl.Tensor[[T], pl.INT32]],
    kv_seq_lens: pl.InOut[pl.Tensor[[B], pl.INT32]],
    hca_compress_state_block_table: pl.Tensor[[B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    block_counts: pl.Tensor[[B, N_CACHE_GROUPS], pl.INT32],
    input_ids: pl.InOut[pl.Tensor[[T], pl.INT64]],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32]],
    sampling_top_ks: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    sampling_seeds: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    sampling_positions: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    hidden_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    mtp_tail_token_ids: pl.InOut[pl.Tensor[[B], pl.INT64]],
    mtp_tail_positions: pl.InOut[pl.Tensor[[B], pl.INT32]],
    mtp_tail_slot_ids: pl.Tensor[[B], pl.INT32],
    mtp_state_generations: pl.Tensor[[B], pl.INT32],
    mtp_state_tokens: pl.InOut[pl.Tensor[[B, STATE_TOKEN_WIDTH], pl.INT64]],
    mtp_state_meta: pl.InOut[pl.Tensor[[B, STATE_META_WIDTH], pl.INT32]],
    mtp_input_ids: pl.Out[pl.Tensor[[T], pl.INT64]],
    mtp_position_ids: pl.Out[pl.Tensor[[T], pl.INT32]],
    mtp_accepted_counts: pl.Out[pl.Tensor[[B], pl.INT32]],
    mtp_tail_pre_hc_pool: pl.InOut[pl.Tensor[[B, HC_MULT, D], pl.FP32]],
    mtp_enorm_w: pl.Tensor[[D], pl.FP32],
    mtp_hnorm_w: pl.Tensor[[D], pl.FP32],
    mtp_e_proj_w: pl.Tensor[[D, D], pl.INT8],
    mtp_e_proj_w_scale: pl.Tensor[[D], pl.FP32],
    mtp_e_proj_smooth: pl.Tensor[[D], pl.FP32],
    mtp_h_proj_w: pl.Tensor[[D, D], pl.INT8],
    mtp_h_proj_w_scale: pl.Tensor[[D], pl.FP32],
    mtp_h_proj_smooth: pl.Tensor[[D], pl.FP32],
    mtp_hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    mtp_hc_attn_scale: pl.Tensor[[3], pl.FP32],
    mtp_hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    mtp_attn_norm_w: pl.Tensor[[D], pl.BF16],
    mtp_wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    mtp_wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    mtp_wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    mtp_wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    mtp_gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    mtp_gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    mtp_kv_cache: pl.InOut[pl.Tensor[[MTP_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    mtp_attn_sink: pl.Tensor[[H], pl.FP32],
    mtp_wo_a: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    mtp_wo_b: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8],
    mtp_wo_b_scale: pl.Tensor[[D], pl.FP32],
    mtp_hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    mtp_hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    mtp_hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    mtp_norm_w: pl.Tensor[[D], pl.BF16],
    mtp_gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    mtp_gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    mtp_tid2eid: pl.Tensor[[MOE_VOCAB, MOE_TOPK], pl.INT32],
    mtp_routed_w1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    mtp_routed_w1_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    mtp_routed_w3: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8],
    mtp_routed_w3_scale: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32],
    mtp_routed_w2: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8],
    mtp_routed_w2_scale: pl.Tensor[[N_LOCAL, D], pl.FP32],
    mtp_shared_w1: pl.Tensor[[MOE_INTER, D], pl.INT8],
    mtp_shared_w1_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    mtp_shared_w3: pl.Tensor[[MOE_INTER, D], pl.INT8],
    mtp_shared_w3_scale: pl.Tensor[[MOE_INTER], pl.FP32],
    mtp_shared_w2: pl.Tensor[[D, MOE_INTER], pl.INT8],
    mtp_shared_w2_scale: pl.Tensor[[D], pl.FP32],
    mtp_mtp_hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    mtp_mtp_hc_head_scale: pl.Tensor[[1], pl.FP32],
    mtp_mtp_hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    mtp_mtp_norm_w: pl.Tensor[[D], pl.BF16],
    mtp_logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    mtp_sampling_temperatures: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32]],
    mtp_sampling_top_ks: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    mtp_sampling_seeds: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    mtp_sampling_positions: pl.InOut[pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32]],
    mtp_hidden_out: pl.Out[pl.Tensor[[T, D], pl.BF16]],
    mtp_next_pre_hc_hidden: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    mtp_logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    mtp_sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    mtp_recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    mtp_recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    mtp_recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    mtp_recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    mtp_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    mtp_data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    mtp_routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    mtp_combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    mtp_lm_head_hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    mtp_lm_head_hidden_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    mtp_lm_head_logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32],
    mtp_lm_head_logits_done: pld.DistributedTensor[[LM_HEAD_TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    mtp_num_tokens: pl.Scalar[pl.INT32],
):
    swa_cos_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(freqs_cos, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0])
    swa_sin_profile: pl.Tensor[[1, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.slice(freqs_sin, [1, MAX_SEQ_LEN, ROPE_HEAD_DIM], [0, 0, 0])
    swa_freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(swa_cos_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    swa_freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16] = pl.reshape(swa_sin_profile, [MAX_SEQ_LEN, ROPE_HEAD_DIM])
    # Recurrent token, position, and length fields arrive as placeholders; resolve
    # them from each request's stable device slot before main decode reads them.
    prepare_decode_from_device_state(
        mtp_tail_slot_ids, mtp_state_generations, mtp_state_tokens, mtp_state_meta,
        input_ids, position_ids, kv_seq_lens,
        mtp_tail_token_ids, mtp_tail_positions,
        sampling_temperatures, sampling_top_ks,
        sampling_seeds, sampling_positions,
    )
    ori_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    swa_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    swa_indices = pl.create_tensor([T, SWA_WIN], dtype=pl.INT32)
    swa_lens = pl.create_tensor([T], dtype=pl.INT32)
    hca_cmp_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    hca_state_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_cmp_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_idx_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_state_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    csa_inner_state_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    build_decode_metadata(
        position_ids,
        block_table,
        hca_cmp_block_table,
        csa_cmp_block_table,
        idx_block_table,
        hca_compress_state_block_table,
        csa_compress_state_block_table,
        csa_inner_compress_state_block_table,
        block_counts,
        ori_slot_mapping,
        swa_slot_mapping,
        swa_indices,
        swa_lens,
        hca_cmp_slot_mapping,
        hca_state_slot_mapping,
        csa_cmp_slot_mapping,
        csa_idx_slot_mapping,
        csa_state_slot_mapping,
        csa_inner_state_slot_mapping,
    )
    x_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    pack_x_hc(input_ids, embed_weight, x_hc)
    decode_fwd(
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale,
        wkv, gamma_cq, gamma_ckv,
        kv_cache,
        attn_sink, wo_a, wo_b, wo_b_scale,
        hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w, hca_compress_state,
        csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w, csa_compress_state,
        csa_idx_wq_b, csa_idx_wq_b_scale, csa_weights_proj, csa_hadamard_idx,
        csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w, csa_inner_compress_state,
        hca_cmp_kv, csa_cmp_kv, idx_kv_cache, idx_kv_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale, routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale, shared_w2, shared_w2_scale,
        freqs_cos, freqs_sin,
        x_hc, position_ids, kv_seq_lens,
        hca_compress_state_block_table, csa_compress_state_block_table, csa_inner_compress_state_block_table,
        hca_cmp_block_table, csa_cmp_block_table, idx_block_table,
        ori_slot_mapping, swa_slot_mapping, swa_indices, swa_lens,
        hca_cmp_slot_mapping, hca_state_slot_mapping,
        csa_cmp_slot_mapping, csa_idx_slot_mapping, csa_state_slot_mapping, csa_inner_state_slot_mapping,
        input_ids,
        hc_head_fn, hc_head_scale, hc_head_base, final_norm_w,
        lm_head_weight, logit_row_indices,
        sampling_temperatures, sampling_top_ks, sampling_seeds, sampling_positions,
        pre_hc_hidden_out, hidden_out, logits, sampled_ids,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        lm_head_hidden_window, lm_head_hidden_done, lm_head_logits_window, lm_head_logits_done,
        num_tokens_per_owner, rank,
    )
    verify_and_pack_mtp_tokens(
        input_ids, position_ids, sampled_ids,
        mtp_tail_token_ids, mtp_tail_positions, mtp_tail_slot_ids,
        mtp_input_ids, mtp_position_ids, mtp_accepted_counts,
    )
    prepare_mtp_sampling_from_device_state(
        mtp_tail_slot_ids, mtp_state_generations, mtp_state_meta,
        mtp_position_ids,
        mtp_sampling_temperatures, mtp_sampling_top_ks,
        mtp_sampling_seeds, mtp_sampling_positions,
    )
    mtp_hidden_states = pl.create_tensor([T, D], dtype=pl.BF16)
    lookup_embedding(mtp_input_ids, embed_weight, mtp_hidden_states)
    mtp_prev_pre_hc_hidden = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32)
    mtp_fallback_hidden = pre_hc_hidden_out[0:S, 0:HC_MULT, 0:D]
    pack_mtp_hidden(
        pre_hc_hidden_out,
        mtp_tail_pre_hc_pool,
        mtp_accepted_counts,
        mtp_tail_slot_ids,
        mtp_fallback_hidden,
        mtp_prev_pre_hc_hidden,
    )
    mtp_swa_slot_mapping = pl.create_tensor([T], dtype=pl.INT64)
    mtp_swa_indices = pl.create_tensor([T, SWA_WIN], dtype=pl.INT32)
    mtp_swa_lens = pl.create_tensor([T], dtype=pl.INT32)
    build_swa_metadata(
        mtp_position_ids,
        block_table,
        mtp_swa_slot_mapping,
        mtp_swa_indices,
        mtp_swa_lens,
    )
    decode_mtp(
        mtp_hidden_states, mtp_prev_pre_hc_hidden, mtp_position_ids,
        mtp_enorm_w, mtp_hnorm_w,
        mtp_e_proj_w, mtp_e_proj_w_scale, mtp_e_proj_smooth,
        mtp_h_proj_w, mtp_h_proj_w_scale, mtp_h_proj_smooth,
        mtp_hc_attn_fn, mtp_hc_attn_scale, mtp_hc_attn_base,
        mtp_attn_norm_w, mtp_wq_a, mtp_wq_b, mtp_wq_b_scale, mtp_wkv, mtp_gamma_cq, mtp_gamma_ckv,
        swa_freqs_cos, swa_freqs_sin,
        mtp_kv_cache, mtp_swa_slot_mapping, mtp_swa_indices, mtp_swa_lens,
        mtp_attn_sink, mtp_wo_a, mtp_wo_b, mtp_wo_b_scale,
        mtp_hc_ffn_fn, mtp_hc_ffn_scale, mtp_hc_ffn_base,
        mtp_norm_w, mtp_gate_w, mtp_gate_bias, mtp_tid2eid, mtp_input_ids,
        mtp_routed_w1, mtp_routed_w1_scale, mtp_routed_w3, mtp_routed_w3_scale, mtp_routed_w2, mtp_routed_w2_scale,
        mtp_shared_w1, mtp_shared_w1_scale, mtp_shared_w3, mtp_shared_w3_scale, mtp_shared_w2, mtp_shared_w2_scale,
        mtp_mtp_hc_head_fn, mtp_mtp_hc_head_scale, mtp_mtp_hc_head_base, mtp_mtp_norm_w,
        lm_head_weight, mtp_logit_row_indices,
        mtp_sampling_temperatures, mtp_sampling_top_ks,
        mtp_sampling_seeds, mtp_sampling_positions,
        mtp_hidden_out, mtp_next_pre_hc_hidden, mtp_logits, mtp_sampled_ids,
        mtp_recv_meta, mtp_recv_x, mtp_recv_aux, mtp_recv_route,
        mtp_arrived, mtp_data_arrived, mtp_routed_y_buf, mtp_combine_arrived,
        mtp_lm_head_hidden_window, mtp_lm_head_hidden_done, mtp_lm_head_logits_window, mtp_lm_head_logits_done,
        rank, mtp_num_tokens,
    )
    # Commit the verified window and the next draft back to the persistent slot.
    advance_decode_device_state(
        mtp_tail_slot_ids, mtp_state_generations, mtp_state_tokens, mtp_state_meta,
        mtp_input_ids, mtp_position_ids,
        mtp_sampled_ids, mtp_accepted_counts,
    )






@pl.jit.host
def l3_decode_fwd_mtp(
    embed_weight: pl.Tensor[[N_RANKS, EMBED_VOCAB_DYN, D], pl.BF16],
    hc_attn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, FWD_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    attn_sink: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_HCA_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_CSA_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32
        ]
    ],
    csa_idx_wq_b: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    csa_idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    csa_weights_proj: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_wkv: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[N_RANKS, CSA_NUM_LAYERS * CSA_IDX_HEAD_DIM], pl.BF16],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32
        ]
    ],
    hca_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    csa_cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, FWD_CSA_CMP_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, FWD_IDX_BLOCK_NUM_DYN, CSA_CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]],
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    routed_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32],
    shared_w1: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w1_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w3: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8],
    shared_w3_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32],
    shared_w2: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8],
    shared_w2_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.FP32],
    freqs_cos: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, 2, MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    block_table: pl.Tensor[[N_RANKS, B, ORI_TABLE_MAX_BLOCKS], pl.INT32],
    position_ids: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT32]],
    kv_seq_lens: pl.InOut[pl.Tensor[[N_RANKS, B], pl.INT32]],
    hca_compress_state_block_table: pl.Tensor[[N_RANKS, B, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    csa_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    csa_inner_compress_state_block_table: pl.Tensor[[N_RANKS, B, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    hca_cmp_block_table: pl.Tensor[[N_RANKS, B, HCA_CMP_MAX_BLOCKS], pl.INT32],
    csa_cmp_block_table: pl.Tensor[[N_RANKS, B, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_block_table: pl.Tensor[[N_RANKS, B, CSA_IDX_CACHE_MAX_BLOCKS], pl.INT32],
    block_counts: pl.Tensor[[N_RANKS, B, N_CACHE_GROUPS], pl.INT32],
    input_ids: pl.InOut[pl.Tensor[[N_RANKS, T], pl.INT64]],
    hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    pre_hc_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    lm_head_weight: pl.Tensor[[N_RANKS, VOCAB_PER_TP, D], pl.BF16],
    sampling_temperatures: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.FP32]],
    sampling_top_ks: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32]],
    sampling_seeds: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32]],
    sampling_positions: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32]],
    hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    mtp_tail_token_ids: pl.InOut[pl.Tensor[[N_RANKS, B], pl.INT64]],
    mtp_tail_positions: pl.InOut[pl.Tensor[[N_RANKS, B], pl.INT32]],
    mtp_tail_pre_hc_pool: pl.InOut[pl.Tensor[[N_RANKS, B, HC_MULT, D], pl.FP32]],
    mtp_accepted_counts: pl.Out[pl.Tensor[[N_RANKS, B], pl.INT32]],
    mtp_tail_slot_ids: pl.Tensor[[N_RANKS, B], pl.INT32],
    mtp_state_generations: pl.Tensor[[N_RANKS, B], pl.INT32],
    mtp_state_tokens: pl.InOut[
        pl.Tensor[[N_RANKS, B, STATE_TOKEN_WIDTH], pl.INT64]
    ],
    mtp_state_meta: pl.InOut[
        pl.Tensor[[N_RANKS, B, STATE_META_WIDTH], pl.INT32]
    ],
    mtp_position_ids: pl.Out[pl.Tensor[[N_RANKS, T], pl.INT32]],
    mtp_enorm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hnorm_w: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_e_proj_w: pl.Tensor[[N_RANKS, D, D], pl.INT8],
    mtp_e_proj_w_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_e_proj_smooth: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_h_proj_w: pl.Tensor[[N_RANKS, D, D], pl.INT8],
    mtp_h_proj_w_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_h_proj_smooth: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    mtp_hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    mtp_hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    mtp_attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    mtp_wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    mtp_wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    mtp_wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    mtp_wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    mtp_gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    mtp_gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    mtp_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, MTP_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    mtp_attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    mtp_wo_a: pl.Tensor[[N_RANKS, O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    mtp_wo_b: pl.Tensor[[N_RANKS, D, O_GROUPS * O_LORA], pl.INT8],
    mtp_wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    mtp_hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    mtp_hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    mtp_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    mtp_gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    mtp_gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    mtp_tid2eid: pl.Tensor[[N_RANKS, MOE_VOCAB, MOE_TOPK], pl.INT32],
    mtp_input_ids: pl.Out[pl.Tensor[[N_RANKS, T], pl.INT64]],
    mtp_routed_w1: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    mtp_routed_w1_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    mtp_routed_w3: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER, D], pl.INT8],
    mtp_routed_w3_scale: pl.Tensor[[N_RANKS, N_LOCAL, MOE_INTER], pl.FP32],
    mtp_routed_w2: pl.Tensor[[N_RANKS, N_LOCAL, D, MOE_INTER], pl.INT8],
    mtp_routed_w2_scale: pl.Tensor[[N_RANKS, N_LOCAL, D], pl.FP32],
    mtp_shared_w1: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    mtp_shared_w1_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    mtp_shared_w3: pl.Tensor[[N_RANKS, MOE_INTER, D], pl.INT8],
    mtp_shared_w3_scale: pl.Tensor[[N_RANKS, MOE_INTER], pl.FP32],
    mtp_shared_w2: pl.Tensor[[N_RANKS, D, MOE_INTER], pl.INT8],
    mtp_shared_w2_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    mtp_mtp_hc_head_fn: pl.Tensor[[N_RANKS, HC_MULT, HC_DIM], pl.FP32],
    mtp_mtp_hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    mtp_mtp_hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    mtp_mtp_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    mtp_sampling_temperatures: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.FP32]],
    mtp_sampling_top_ks: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32]],
    mtp_sampling_seeds: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32]],
    mtp_sampling_positions: pl.InOut[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32]],
    mtp_hidden_out: pl.Out[pl.Tensor[[N_RANKS, T, D], pl.BF16]],
    mtp_next_pre_hc_hidden: pl.Out[pl.Tensor[[N_RANKS, T, HC_MULT, D], pl.FP32]],
    mtp_logits: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]],
    mtp_sampled_ids: pl.Out[pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    mtp_logit_row_indices: pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS], pl.INT32],
    mtp_num_tokens: pl.Scalar[pl.INT32],
):
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer(N_LOCAL * RECV_MAX * D)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    lm_head_hidden_window_buf = pld.alloc_window_buffer(GROUP_LOGIT_ROWS * D * 2)
    lm_head_logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * LM_HEAD_VOCAB * 4)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    mtp_recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    mtp_recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    mtp_recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    mtp_recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    mtp_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    mtp_data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    mtp_routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    mtp_combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    mtp_lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    mtp_lm_head_logits_window_buf = pld.alloc_window_buffer(MAX_LOGIT_ROWS * LM_HEAD_VOCAB * 4)
    mtp_lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    mtp_lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32] = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8] = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32] = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32] = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16] = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32] = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        lm_head_hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        lm_head_hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
        lm_head_logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        mtp_recv_meta = pld.window(mtp_recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        mtp_recv_x = pld.window(mtp_recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        mtp_recv_aux = pld.window(mtp_recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        mtp_recv_route = pld.window(mtp_recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        mtp_arrived = pld.window(mtp_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_data_arrived = pld.window(mtp_data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_routed_y_buf = pld.window(mtp_routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        mtp_combine_arrived = pld.window(mtp_combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        mtp_lm_head_hidden_window = pld.window(mtp_lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        mtp_lm_head_hidden_done = pld.window(mtp_lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        mtp_lm_head_logits_window = pld.window(mtp_lm_head_logits_window_buf, [MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32)
        mtp_lm_head_logits_done = pld.window(mtp_lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        l2_decode_fwd_mtp(
            embed_weight[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank], wq_b_scale[rank],
            wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            kv_cache[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank], hca_cmp_norm_w[rank],
            hca_compress_state[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank], csa_cmp_norm_w[rank],
            csa_compress_state[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank], csa_weights_proj[rank], csa_hadamard_idx[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank], csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_compress_state[rank],
            hca_cmp_kv[rank], csa_cmp_kv[rank], idx_kv_cache[rank], idx_kv_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            routed_w1[rank], routed_w1_scale[rank], routed_w3[rank], routed_w3_scale[rank], routed_w2[rank],
            routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank], shared_w3[rank], shared_w3_scale[rank], shared_w2[rank],
            shared_w2_scale[rank],
            freqs_cos[rank], freqs_sin[rank],
            block_table[rank], position_ids[rank], kv_seq_lens[rank],
            hca_compress_state_block_table[rank], csa_compress_state_block_table[rank],
            csa_inner_compress_state_block_table[rank],
            hca_cmp_block_table[rank], csa_cmp_block_table[rank],
            idx_block_table[rank], block_counts[rank],
            input_ids[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank], final_norm_w[rank],
            lm_head_weight[rank], logit_row_indices[rank],
            sampling_temperatures[rank], sampling_top_ks[rank],
            sampling_seeds[rank], sampling_positions[rank],
            pre_hc_hidden_out[rank], hidden_out[rank], logits[rank], sampled_ids[rank],
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            lm_head_hidden_window, lm_head_hidden_done, lm_head_logits_window, lm_head_logits_done,
            num_tokens_per_owner,
            mtp_tail_token_ids[rank], mtp_tail_positions[rank], mtp_tail_slot_ids[rank],
            mtp_state_generations[rank], mtp_state_tokens[rank], mtp_state_meta[rank],
            mtp_input_ids[rank], mtp_position_ids[rank], mtp_accepted_counts[rank],
            mtp_tail_pre_hc_pool[rank],
            mtp_enorm_w[rank], mtp_hnorm_w[rank],
            mtp_e_proj_w[rank], mtp_e_proj_w_scale[rank], mtp_e_proj_smooth[rank],
            mtp_h_proj_w[rank], mtp_h_proj_w_scale[rank], mtp_h_proj_smooth[rank],
            mtp_hc_attn_fn[rank], mtp_hc_attn_scale[rank], mtp_hc_attn_base[rank],
            mtp_attn_norm_w[rank], mtp_wq_a[rank], mtp_wq_b[rank], mtp_wq_b_scale[rank],
            mtp_wkv[rank], mtp_gamma_cq[rank], mtp_gamma_ckv[rank],
            mtp_kv_cache[rank],
            mtp_attn_sink[rank], mtp_wo_a[rank], mtp_wo_b[rank], mtp_wo_b_scale[rank],
            mtp_hc_ffn_fn[rank], mtp_hc_ffn_scale[rank], mtp_hc_ffn_base[rank],
            mtp_norm_w[rank], mtp_gate_w[rank], mtp_gate_bias[rank], mtp_tid2eid[rank],
            mtp_routed_w1[rank], mtp_routed_w1_scale[rank], mtp_routed_w3[rank], mtp_routed_w3_scale[rank],
            mtp_routed_w2[rank], mtp_routed_w2_scale[rank],
            mtp_shared_w1[rank], mtp_shared_w1_scale[rank], mtp_shared_w3[rank], mtp_shared_w3_scale[rank],
            mtp_shared_w2[rank], mtp_shared_w2_scale[rank],
            mtp_mtp_hc_head_fn[rank], mtp_mtp_hc_head_scale[rank], mtp_mtp_hc_head_base[rank],
            mtp_mtp_norm_w[rank],
            mtp_logit_row_indices[rank],
            mtp_sampling_temperatures[rank], mtp_sampling_top_ks[rank],
            mtp_sampling_seeds[rank], mtp_sampling_positions[rank],
            mtp_hidden_out[rank], mtp_next_pre_hc_hidden[rank], mtp_logits[rank], mtp_sampled_ids[rank],
            mtp_recv_meta, mtp_recv_x, mtp_recv_aux, mtp_recv_route,
            mtp_arrived, mtp_data_arrived, mtp_routed_y_buf, mtp_combine_arrived,
            mtp_lm_head_hidden_window, mtp_lm_head_hidden_done, mtp_lm_head_logits_window,
            mtp_lm_head_logits_done,
            rank, mtp_num_tokens,
            device=rank,
        )


def build_tensor_specs(
    start_pos=DECODE_START_POS,
    num_tokens=T,
    ori_block_num=ORI_BLOCK_NUM,
    cmp_block_num=CSA_CMP_BLOCK_NUM,
    idx_block_num=CSA_IDX_CACHE_BLOCK_NUM,
    hca_state_block_num=HCA_COMPRESS_STATE_BLOCK_NUM,
    csa_state_block_num=CSA_MAIN_STATE_BLOCK_NUM,
    inner_state_block_num=CSA_INNER_STATE_BLOCK_NUM,
):
    import torch
    from golden import ScalarSpec, TensorSpec

    forward_specs = {
        spec.name: spec
        for spec in build_decode_fwd_tensor_specs(
            start_pos=start_pos,
            num_tokens=num_tokens,
            ori_block_num=ori_block_num,
            cmp_block_num=cmp_block_num,
            idx_block_num=idx_block_num,
            hca_state_block_num=hca_state_block_num,
            csa_state_block_num=csa_state_block_num,
            inner_state_block_num=inner_state_block_num,
        )
    }
    mtp_specs = {
        spec.name: spec
        for spec in build_decode_mtp_tensor_specs(
            start_pos=start_pos,
            num_tokens=num_tokens,
            ori_block_num=ori_block_num,
        )
    }

    specs = {
        name: spec
        for name, spec in forward_specs.items()
        if name not in PREAMBLE_OUTPUT_NAMES
    }
    specs.update(
        build_decode_fwd_preamble_specs(
            start_pos=start_pos,
            ori_block_num=ori_block_num,
            cmp_block_num=cmp_block_num,
            idx_block_num=idx_block_num,
            hca_state_block_num=hca_state_block_num,
            csa_state_block_num=csa_state_block_num,
            inner_state_block_num=inner_state_block_num,
        )
    )
    for name in ("input_ids", "position_ids", "kv_seq_lens"):
        specs[name] = replace(specs[name], is_output=True)

    def init_tail_token_ids():
        tokens = torch.arange(B, dtype=torch.int64) + 10
        return tokens.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_tail_positions():
        return torch.full((N_RANKS, B), start_pos - 1, dtype=torch.int32)

    def init_tail_slot_ids():
        slots = torch.arange(B, dtype=torch.int32)
        return slots.unsqueeze(0).expand(N_RANKS, -1).contiguous()

    def init_state_generations():
        return torch.ones(N_RANKS, B, dtype=torch.int32)

    def init_state_tokens():
        tokens = torch.empty(N_RANKS, B, STATE_TOKEN_WIDTH, dtype=torch.int64)
        tokens[:, :, STATE_TAIL_TOKEN] = torch.arange(B, dtype=torch.int64) + 10
        tokens[:, :, STATE_DRAFT_TOKEN] = torch.arange(B, dtype=torch.int64) + 20
        return tokens

    def init_state_meta():
        meta = torch.zeros(N_RANKS, B, STATE_META_WIDTH, dtype=torch.int32)
        meta[:, :, STATE_VALID] = 1
        meta[:, :, STATE_GENERATION] = 1
        meta[:, :, STATE_TAIL_POSITION] = start_pos - 1
        meta[:, :, STATE_COMMITTED_COUNT] = 0
        meta[:, :, STATE_TEMPERATURE_MICROS] = 0
        meta[:, :, STATE_TOP_K] = VOCAB
        meta[:, :, STATE_SEED] = 0
        return meta

    specs.update(
        {
            "mtp_tail_token_ids": TensorSpec(
                "mtp_tail_token_ids",
                [N_RANKS, B],
                torch.int64,
                init_value=init_tail_token_ids,
                is_output=True,
            ),
            "mtp_tail_positions": TensorSpec(
                "mtp_tail_positions",
                [N_RANKS, B],
                torch.int32,
                init_value=init_tail_positions,
                is_output=True,
            ),
            "mtp_tail_slot_ids": TensorSpec(
                "mtp_tail_slot_ids",
                [N_RANKS, B],
                torch.int32,
                init_value=init_tail_slot_ids,
            ),
            "mtp_state_generations": TensorSpec(
                "mtp_state_generations",
                [N_RANKS, B],
                torch.int32,
                init_value=init_state_generations,
            ),
            "mtp_state_tokens": TensorSpec(
                "mtp_state_tokens",
                [N_RANKS, B, STATE_TOKEN_WIDTH],
                torch.int64,
                init_value=init_state_tokens,
                is_output=True,
                resident="stacked",
            ),
            "mtp_state_meta": TensorSpec(
                "mtp_state_meta",
                [N_RANKS, B, STATE_META_WIDTH],
                torch.int32,
                init_value=init_state_meta,
                is_output=True,
                resident="stacked",
            ),
            "mtp_input_ids": replace(
                mtp_specs["input_ids"],
                name="mtp_input_ids",
                init_value=None,
                is_output=True,
            ),
            "mtp_position_ids": replace(
                mtp_specs["position_ids"],
                name="mtp_position_ids",
                init_value=None,
                is_output=True,
            ),
            "mtp_accepted_counts": TensorSpec(
                "mtp_accepted_counts",
                [N_RANKS, B],
                torch.int32,
                is_output=True,
            ),
            "mtp_tail_pre_hc_pool": TensorSpec(
                "mtp_tail_pre_hc_pool",
                [N_RANKS, B, HC_MULT, D],
                torch.float32,
                init_value=lambda: torch.randn(N_RANKS, B, HC_MULT, D),
                is_output=True,
                resident="stacked",
            ),
            "mtp_hidden_out": replace(mtp_specs["hidden_out"], name="mtp_hidden_out"),
            "mtp_next_pre_hc_hidden": replace(
                mtp_specs["next_pre_hc_hidden"],
                name="mtp_next_pre_hc_hidden",
            ),
            "mtp_logits": replace(mtp_specs["logits"], name="mtp_logits"),
            "mtp_sampled_ids": replace(mtp_specs["sampled_ids"], name="mtp_sampled_ids"),
            "mtp_logit_row_indices": replace(
                mtp_specs["logit_row_indices"],
                name="mtp_logit_row_indices",
            ),
            "mtp_num_tokens": ScalarSpec("mtp_num_tokens", torch.int32, num_tokens),
        }
    )

    shared_mtp_names = {
        "freqs_cos",
        "freqs_sin",
        "lm_head_weight",
        "num_tokens",
    }
    # Built on device from the main step's outputs, or built explicitly below.
    custom_mtp_names = {
        "hidden_out",
        "hidden_states",
        "input_ids",
        "logit_row_indices",
        "logits",
        "next_pre_hc_hidden",
        "position_ids",
        "prev_pre_hc_hidden",
        "sampled_ids",
        "swa_indices",
        "swa_lens",
        "swa_slot_mapping",
    }
    for name, spec in mtp_specs.items():
        if name in shared_mtp_names or name in custom_mtp_names:
            continue
        specs[f"mtp_{name}"] = replace(spec, name=f"mtp_{name}")

    sampling_names = {
        "sampling_temperatures",
        "sampling_top_ks",
        "sampling_seeds",
        "sampling_positions",
    }
    for name in sampling_names:
        specs[name] = replace(specs[name], is_output=True)
        mtp_name = f"mtp_{name}"
        specs[mtp_name] = replace(specs[mtp_name], is_output=True)

    param_names = l3_decode_fwd_mtp._param_names()
    missing = set(param_names) - specs.keys()
    extra = specs.keys() - set(param_names)
    if missing or extra:
        raise ValueError(
            f"decode_fwd_mtp fixture mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return [specs[name] for name in param_names]


def main():
    from golden import run_jit

    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 fused main-decode, verification, and MTP decode driver."
    )
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a5"])
    parser.add_argument(
        "--ep",
        type=int,
        default=N_RANKS,
        choices=[2, 4, 8],
        help="EP world size / rank count (parsed at import by moe).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=LM_HEAD_TP_SIZE,
        choices=[2, 4, 8, 16],
        help="LM-head TP world size (parsed at import by lm_head); must divide --ep.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=",".join(str(i) for i in range(N_RANKS)),
        help=f"comma-separated device ids; need at least {N_RANKS}",
    )
    parser.add_argument("--start-pos", type=int, default=DECODE_START_POS)
    parser.add_argument("--num-tokens", type=int, default=T)
    parser.add_argument("--ori-block-num", type=int, default=ORI_BLOCK_NUM)
    parser.add_argument("--cmp-block-num", type=int, default=CSA_CMP_BLOCK_NUM)
    parser.add_argument("--idx-block-num", type=int, default=CSA_IDX_CACHE_BLOCK_NUM)
    parser.add_argument("--hca-state-block-num", type=int, default=HCA_COMPRESS_STATE_BLOCK_NUM)
    parser.add_argument("--csa-state-block-num", type=int, default=CSA_MAIN_STATE_BLOCK_NUM)
    parser.add_argument("--inner-state-block-num", type=int, default=CSA_INNER_STATE_BLOCK_NUM)
    parser.add_argument(
        "--enable-chip-swimlane",
        type=int,
        nargs="?",
        const=1,
        default=0,
        choices=(0, 1, 2),
    )
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    assert args.tp <= args.ep, f"fused decode requires --tp <= --ep, got tp={args.tp}, ep={args.ep}"
    assert args.ep % args.tp == 0, (
        f"grouped LM head needs --ep % --tp == 0, got ep={args.ep}, tp={args.tp}"
    )
    assert LM_HEAD_TP_SIZE == args.tp, (
        f"import-time LM_HEAD_TP_SIZE must match --tp, got {LM_HEAD_TP_SIZE} vs {args.tp}"
    )
    assert N_RANKS == args.ep, f"import-time N_RANKS must match --ep, got {N_RANKS} vs {args.ep}"
    assert args.start_pos >= 1, f"--start-pos must be at least 1, got {args.start_pos}"

    device_ids = [int(device) for device in args.device.split(",")]
    assert len(device_ids) >= N_RANKS, f"need at least {N_RANKS} devices, got {device_ids}"

    result = run_jit(
        fn=l3_decode_fwd_mtp,
        specs=build_tensor_specs(
            start_pos=args.start_pos,
            num_tokens=args.num_tokens,
            ori_block_num=args.ori_block_num,
            cmp_block_num=args.cmp_block_num,
            idx_block_num=args.idx_block_num,
            hca_state_block_num=args.hca_state_block_num,
            csa_state_block_num=args.csa_state_block_num,
            inner_state_block_num=args.inner_state_block_num,
        ),
        golden_fn=None,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        save_data=False,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids[:N_RANKS],
                num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
