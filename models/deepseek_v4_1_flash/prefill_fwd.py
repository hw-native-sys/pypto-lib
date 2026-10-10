# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compose native V4.1 operators into one resident CED prefill per A5 rank.

Use --tp 1 --ep 4. Each rank owns a prompt and 96 routed experts; attention
weights are replicated. Original MXFP8/MXFP4 weights remain resident. Generation
recomputes the complete prefix, including both CED stages, once per new token.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir import DistributedConfig
from pypto.runtime import RunConfig, StackedDeviceTensor
import torch

from models.deepseek_v4_1_flash import config as C

# Native MoE uses a static, padded row capacity shared by every EP peer.
_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--capacity", type=int, default=128)
CAPACITY = _shape_parser.parse_known_args()[0].capacity
C.MOE_TOKENS = CAPACITY
C.MOE_RECV_MAX = C.EP_SIZE * CAPACITY

from models.deepseek_v4_1_flash.decode_common import zero_bf16_padding
from models.deepseek_v4_1_flash.engram import engram_mx
from models.deepseek_v4_1_flash.hc_head import hc_head
from models.deepseek_v4_1_flash.hc_mixes import mhc_mixes
from models.deepseek_v4_1_flash.hc_post import mhc_post
from models.deepseek_v4_1_flash.hc_pre import mhc_pre
from models.deepseek_v4_1_flash.input_pack import gather_rows, pack_x_hc
from models.deepseek_v4_1_flash.lm_head import greedy_sample, lm_head
from models.deepseek_v4_1_flash.metadata import (
    PrefillCheckpoint, ResidentPrefillWeights, prepare_resident_inputs,
)
from models.deepseek_v4_1_flash.moe import moe
from models.deepseek_v4_1_flash.prefill_c1a_common import (
    make_prefill_attention, make_prefill_publish_decoder,
)
from models.deepseek_v4_1_flash.quantization import build_mxfp4_pair_lut
from models.deepseek_v4_1_flash.rmsnorm import rms_norm

RANKS = 4
D = C.D
HC_MULT = C.HC_MULT
VOCAB = C.VOCAB
PAGES = (CAPACITY + 127) // 128 + 1
LOCAL_EXPERTS = C.N_LOCAL_EXPERTS
RECV_MAX = C.MOE_RECV_MAX
AUX_WIDTH = C.AUX_WIDTH
HC_DIM = C.HC_DIM
HEAD_DIM = C.HEAD_DIM
INDEX_DIM = C.INDEX_DIM
INDEX_H = C.INDEX_H
INDEX_TOPK = C.INDEX_TOPK
LOCAL_H = C.LOCAL_H
LOCAL_O_GROUPS = C.LOCAL_O_GROUPS
LOCAL_O_WIDTH = C.LOCAL_O_WIDTH
MIX_HC = C.MIX_HC
MOE_INTER = C.MOE_INTER
N_EXPERTS = C.N_EXPERTS
O_GROUP_IN = C.O_GROUP_IN
O_LORA = C.O_LORA
PREFILL_MAX_TOKENS = C.PREFILL_MAX_TOKENS
Q_LORA = C.Q_LORA
ROUTE_WIDTH = C.ROUTE_WIDTH
TOPK = C.TOPK
TP_SIZE = C.TP_SIZE
B8 = pl.dynamic("PREFILL_FP8_BLOCKS")
B4 = pl.dynamic("PREFILL_FP4_BLOCKS")
S8 = pl.dynamic("PREFILL_SCALE_BLOCKS")
B16 = pl.dynamic("PREFILL_BF16_VALUES")
B32 = pl.dynamic("PREFILL_FP32_VALUES")
attention = make_prefill_attention()
publish_decoder = make_prefill_publish_decoder()

@pl.jit(auto_scope=False)
def prefill_fwd(
    embedding: pl.Tensor[[CAPACITY, D], pl.BF16],
    embedding_ids: pl.Tensor[[CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[CAPACITY], pl.INT32],
    engram_lookup: pl.Tensor[[2, CAPACITY, 6144], pl.BF16],
    bank8: pl.Tensor[[B8, 128], pl.FP8E4M3FN],
    bank4: pl.Tensor[[B4, 128], pl.UINT8],
    scales8: pl.Tensor[[S8, 128], pl.FP8E8M0],
    bf16: pl.Tensor[[B16], pl.BF16],
    dense: pl.Tensor[[B32], pl.FP32],
    fp8_offsets: pl.Tensor[[40, 9], pl.INT64],
    fp8_scale_offsets: pl.Tensor[[40, 9], pl.INT64],
    bf16_offsets: pl.Tensor[[40, 10], pl.INT64],
    dense_offsets: pl.Tensor[[40, 12], pl.INT64],
    expert_offsets: pl.Tensor[[40, 3], pl.INT64],
    expert_scale_offsets: pl.Tensor[[40, 3], pl.INT64],
    mxfp4_pair_lut: pl.Tensor[[2, 256], pl.INT16],
    counts: pl.Tensor[[2], pl.INT32],
    replay_rows: pl.Tensor[[CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[2, CAPACITY], pl.INT32],
    rope_cos: pl.Tensor[[3, CAPACITY, 32], pl.FP32],
    rope_sin: pl.Tensor[[3, CAPACITY, 32], pl.FP32],
    compressed_cos: pl.Tensor[[CAPACITY, 32], pl.FP32],
    compressed_sin: pl.Tensor[[CAPACITY, 32], pl.FP32],
    publisher_cos: pl.Tensor[[CAPACITY, 32], pl.FP32],
    publisher_sin: pl.Tensor[[CAPACITY, 32], pl.FP32],
    window_slots: pl.Tensor[[2, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[2, CAPACITY, 128], pl.INT32],
    compressed_slots: pl.Tensor[[CAPACITY], pl.INT64],
    publisher_slots: pl.Tensor[[CAPACITY], pl.INT64],
    token_requests: pl.Tensor[[2, CAPACITY], pl.INT32],
    compressed_lens: pl.Tensor[[2, CAPACITY], pl.INT32],
    query_starts: pl.Tensor[[2, 2], pl.INT32],
    index_block_tables: pl.Tensor[[2, 1, PAGES], pl.INT32],
    state_block_table: pl.Tensor[[1, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[40, PAGES, 128, 1, 512], pl.FP8E4M3FN]],
    window_scales: pl.InOut[pl.Tensor[[40, PAGES, 128, 1, 16], pl.FP8E8M0]],
    compressed_cache: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 256], pl.UINT8]],
    compressed_scales: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 32], pl.FP8E4M3FN]],
    index_cache: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[4, PAGES, 128, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[3, 2, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32]],
    final_norm: pl.Tensor[[D], pl.BF16],
    head: pl.Tensor[[VOCAB, D], pl.BF16],
    logit_rows: pl.Tensor[[16], pl.INT32],
    logits: pl.Out[pl.Tensor[[16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[16, 8], pl.INT32]],
    recv_meta: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS], pl.INT32],
    recv_x: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, D], pl.INT8],
    recv_scale: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, D // 32], pl.UINT8],
    recv_weights: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, C.AUX_WIDTH], pl.FP32],
    recv_routes: pld.DistributedTensor[[LOCAL_EXPERTS * RECV_MAX, C.ROUTE_WIDTH], pl.INT32],
    arrived: pld.DistributedTensor[[RANKS, 128], pl.INT32],
    data_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    routed_output: pld.DistributedTensor[[CAPACITY * C.TOPK, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[RANKS, LOCAL_EXPERTS, 128], pl.INT32],
    attention_window: pld.DistributedTensor[[C.PREFILL_MAX_TOKENS, D], pl.FP32],
    attention_arrived: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_hidden: pld.DistributedTensor[[16, D], pl.BF16],
    head_hidden_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    head_logits: pld.DistributedTensor[[16, VOCAB], pl.FP32],
    head_logits_done: pld.DistributedTensor[[C.TP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    epoch_base: pl.Scalar[pl.INT32],
):
    """Run encoder, full encoder-cache publication, decoder replay and head."""
    bank8.bind_dynamic(0, B8)
    bank4.bind_dynamic(0, B4)
    scales8.bind_dynamic(0, S8)
    bf16.bind_dynamic(0, B16)
    dense.bind_dynamic(0, B32)
    hidden_pair = pl.create_tensor([4 * CAPACITY, HC_MULT, D], dtype=pl.FP32)
    pre_pair = pl.create_tensor([4 * CAPACITY, HC_MULT], dtype=pl.FP32)
    initial_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
    initial_mix = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
    pack_x_hc(embedding_ids, embedding, initial_hidden)
    gather_rows(initial_pre, identity_rows, initial_mix)
    topk_indices = pl.create_tensor([CAPACITY, INDEX_TOPK], dtype=pl.INT32)
    candidate_mask = pl.create_tensor([CAPACITY, PAGES * 128], dtype=pl.UINT8)
    # Only stage handoff buffers outlive an attention scope. Keep MoE at its
    # existing nesting depth so each expert's scratch can be reclaimed.
    attn_hidden = pl.create_tensor([CAPACITY, HC_MULT, D], dtype=pl.FP32)
    attn_pre = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
    ffn_mixed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    for layer in pl.range(40):
        stage = layer // 20
        active = pl.read(counts, [stage])
        current_slot = (2 * stage + layer % 2) * CAPACITY
        next_slot = (2 * stage + (layer + 1) % 2) * CAPACITY
        hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [current_slot, 0, 0])
        delayed_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [current_slot, 0])
        next_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [next_slot, 0, 0])
        next_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [next_slot, 0])
        # Use 128-element rows: runtime view coordinates are uint32, while
        # their row-stride product and storage offsets are uint64.
        wq_a_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 0]), 0), pl.INDEX)
        wq_a_storage = pl.slice(bank8, [(D * Q_LORA) // 128, 128], [wq_a_offset // 128, 0])
        wq_a = pl.reshape(wq_a_storage, [D, Q_LORA])
        wq_a_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 0]), 0), pl.INDEX)
        wq_a_scale_storage = pl.slice(scales8, [(D // 32 * Q_LORA) // 128, 128], [wq_a_scale_offset // 128, 0])
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wq_a_scale_storage, [D // 32, Q_LORA])
        wq_b_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 1]), 0), pl.INDEX)
        wq_b_storage = pl.slice(bank8, [(Q_LORA * (LOCAL_H * HEAD_DIM)) // 128, 128], [wq_b_offset // 128, 0])
        wq_b = pl.reshape(wq_b_storage, [Q_LORA, LOCAL_H * HEAD_DIM])
        wq_b_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 1]), 0), pl.INDEX)
        wq_b_scale_storage = pl.slice(scales8, [(Q_LORA // 32 * (LOCAL_H * HEAD_DIM)) // 128, 128], [wq_b_scale_offset // 128, 0])
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wq_b_scale_storage, [Q_LORA // 32, LOCAL_H * HEAD_DIM])
        wkv_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 2]), 0), pl.INDEX)
        wkv_storage = pl.slice(bank8, [(D * HEAD_DIM) // 128, 128], [wkv_offset // 128, 0])
        wkv = pl.reshape(wkv_storage, [D, HEAD_DIM])
        wkv_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 2]), 0), pl.INDEX)
        wkv_scale_storage = pl.slice(scales8, [(D // 32 * HEAD_DIM) // 128, 128], [wkv_scale_offset // 128, 0])
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wkv_scale_storage, [D // 32, HEAD_DIM])
        wo_b_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 3]), 0), pl.INDEX)
        wo_b_storage = pl.slice(bank8, [(LOCAL_O_WIDTH * D) // 128, 128], [wo_b_offset // 128, 0])
        wo_b = pl.reshape(wo_b_storage, [LOCAL_O_WIDTH, D])
        wo_b_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 3]), 0), pl.INDEX)
        wo_b_scale_storage = pl.slice(scales8, [(LOCAL_O_WIDTH // 32 * D) // 128, 128], [wo_b_scale_offset // 128, 0])
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(wo_b_scale_storage, [LOCAL_O_WIDTH // 32, D])
        index_wq_b_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 4]), 0), pl.INDEX)
        index_wq_b_storage = pl.slice(bank8, [(Q_LORA * (INDEX_H * INDEX_DIM)) // 128, 128], [index_wq_b_offset // 128, 0])
        index_wq_b = pl.reshape(index_wq_b_storage, [Q_LORA, INDEX_H * INDEX_DIM])
        index_wq_b_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 4]), 0), pl.INDEX)
        index_wq_b_scale_storage = pl.slice(scales8, [(Q_LORA // 32 * (INDEX_H * INDEX_DIM)) // 128, 128], [index_wq_b_scale_offset // 128, 0])
        index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(index_wq_b_scale_storage, [Q_LORA // 32, INDEX_H * INDEX_DIM])
        shared_w1_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 5]), 0), pl.INDEX)
        shared_w1_storage = pl.slice(bank8, [(D * MOE_INTER) // 128, 128], [shared_w1_offset // 128, 0])
        shared_w1 = pl.reshape(shared_w1_storage, [D, MOE_INTER])
        shared_w1_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 5]), 0), pl.INDEX)
        shared_w1_scale_storage = pl.slice(scales8, [(D // 32 * MOE_INTER) // 128, 128], [shared_w1_scale_offset // 128, 0])
        shared_w1_scale: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(shared_w1_scale_storage, [D // 32, MOE_INTER])
        shared_w3_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 6]), 0), pl.INDEX)
        shared_w3_storage = pl.slice(bank8, [(D * MOE_INTER) // 128, 128], [shared_w3_offset // 128, 0])
        shared_w3 = pl.reshape(shared_w3_storage, [D, MOE_INTER])
        shared_w3_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 6]), 0), pl.INDEX)
        shared_w3_scale_storage = pl.slice(scales8, [(D // 32 * MOE_INTER) // 128, 128], [shared_w3_scale_offset // 128, 0])
        shared_w3_scale: pl.Tensor[[D // 32, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(shared_w3_scale_storage, [D // 32, MOE_INTER])
        shared_w2_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 7]), 0), pl.INDEX)
        shared_w2_storage = pl.slice(bank8, [(MOE_INTER * D) // 128, 128], [shared_w2_offset // 128, 0])
        shared_w2 = pl.reshape(shared_w2_storage, [MOE_INTER, D])
        shared_w2_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 7]), 0), pl.INDEX)
        shared_w2_scale_storage = pl.slice(scales8, [(MOE_INTER // 32 * D) // 128, 128], [shared_w2_scale_offset // 128, 0])
        shared_w2_scale: pl.Tensor[[MOE_INTER // 32, D], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(shared_w2_scale_storage, [MOE_INTER // 32, D])
        attn_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 0]), 0), pl.INDEX)
        attn_norm_storage = pl.slice(bf16, [D], [attn_norm_offset])
        attn_norm = pl.reshape(attn_norm_storage, [D])
        ffn_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 1]), 0), pl.INDEX)
        ffn_norm_storage = pl.slice(bf16, [D], [ffn_norm_offset])
        ffn_norm = pl.reshape(ffn_norm_storage, [D])
        q_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 2]), 0), pl.INDEX)
        q_norm_storage = pl.slice(bf16, [Q_LORA], [q_norm_offset])
        q_norm = pl.reshape(q_norm_storage, [Q_LORA])
        kv_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 3]), 0), pl.INDEX)
        kv_norm_storage = pl.slice(bf16, [HEAD_DIM], [kv_norm_offset])
        kv_norm = pl.reshape(kv_norm_storage, [HEAD_DIM])
        wo_a_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 4]), 0), pl.INDEX)
        wo_a_storage = pl.slice(bf16, [LOCAL_O_GROUPS * O_LORA * O_GROUP_IN], [wo_a_offset])
        wo_a = pl.reshape(wo_a_storage, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN])
        publisher_wkv_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 5]), 0), pl.INDEX)
        publisher_wkv_storage = pl.slice(bf16, [D * HEAD_DIM], [publisher_wkv_offset])
        publisher_wkv = pl.reshape(publisher_wkv_storage, [D, HEAD_DIM])
        compressor_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 6]), 0), pl.INDEX)
        compressor_norm_storage = pl.slice(bf16, [HEAD_DIM], [compressor_norm_offset])
        compressor_norm = pl.reshape(compressor_norm_storage, [HEAD_DIM])
        index_wk_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 7]), 0), pl.INDEX)
        index_wk_storage = pl.slice(bf16, [HEAD_DIM * INDEX_DIM], [index_wk_offset])
        index_wk = pl.reshape(index_wk_storage, [HEAD_DIM, INDEX_DIM])
        index_norm_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 8]), 0), pl.INDEX)
        index_norm_storage = pl.slice(bf16, [INDEX_DIM], [index_norm_offset])
        index_norm = pl.reshape(index_norm_storage, [INDEX_DIM])
        index_weights_offset = pl.cast(pl.max(pl.read(bf16_offsets, [layer, 9]), 0), pl.INDEX)
        index_weights_storage = pl.slice(bf16, [D * INDEX_H], [index_weights_offset])
        index_weights = pl.reshape(index_weights_storage, [D, INDEX_H])
        attn_fn_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 0]), 0), pl.INDEX)
        attn_fn_storage = pl.slice(dense, [MIX_HC * HC_DIM], [attn_fn_offset])
        attn_fn = pl.reshape(attn_fn_storage, [MIX_HC, HC_DIM])
        attn_scale_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 1]), 0), pl.INDEX)
        attn_scale_storage = pl.slice(dense, [3], [attn_scale_offset])
        attn_scale = pl.reshape(attn_scale_storage, [3])
        attn_base_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 2]), 0), pl.INDEX)
        attn_base_storage = pl.slice(dense, [MIX_HC], [attn_base_offset])
        attn_base = pl.reshape(attn_base_storage, [MIX_HC])
        ffn_fn_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 3]), 0), pl.INDEX)
        ffn_fn_storage = pl.slice(dense, [MIX_HC * HC_DIM], [ffn_fn_offset])
        ffn_fn = pl.reshape(ffn_fn_storage, [MIX_HC, HC_DIM])
        ffn_scale_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 4]), 0), pl.INDEX)
        ffn_scale_storage = pl.slice(dense, [3], [ffn_scale_offset])
        ffn_scale = pl.reshape(ffn_scale_storage, [3])
        ffn_base_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 5]), 0), pl.INDEX)
        ffn_base_storage = pl.slice(dense, [MIX_HC], [ffn_base_offset])
        ffn_base = pl.reshape(ffn_base_storage, [MIX_HC])
        attn_sink_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 6]), 0), pl.INDEX)
        attn_sink_storage = pl.slice(dense, [LOCAL_H], [attn_sink_offset])
        attn_sink = pl.reshape(attn_sink_storage, [LOCAL_H])
        gate_weight_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 7]), 0), pl.INDEX)
        gate_weight_storage = pl.slice(dense, [N_EXPERTS * D], [gate_weight_offset])
        gate_weight = pl.reshape(gate_weight_storage, [N_EXPERTS, D])
        gate_bias_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 8]), 0), pl.INDEX)
        gate_bias_storage = pl.slice(dense, [N_EXPERTS], [gate_bias_offset])
        gate_bias = pl.reshape(gate_bias_storage, [N_EXPERTS])
        compressor_wkv_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 9]), 0), pl.INDEX)
        compressor_wkv_storage = pl.slice(dense, [D * HEAD_DIM], [compressor_wkv_offset])
        compressor_wkv = pl.reshape(compressor_wkv_storage, [D, HEAD_DIM])
        compressor_wgate_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 10]), 0), pl.INDEX)
        compressor_wgate_storage = pl.slice(dense, [D * HEAD_DIM], [compressor_wgate_offset])
        compressor_wgate = pl.reshape(compressor_wgate_storage, [D, HEAD_DIM])
        routed_w1_offset = pl.cast(pl.max(pl.read(expert_offsets, [layer, 0]), 0), pl.INDEX)
        routed_w1_storage = pl.slice(bank4, [(LOCAL_EXPERTS * (D * MOE_INTER // 256) * 128) // 128, 128], [routed_w1_offset // 128, 0])
        routed_w1 = pl.reshape(routed_w1_storage, [LOCAL_EXPERTS, D * MOE_INTER // 256, 128])
        routed_w1_scale_offset = pl.cast(pl.max(pl.read(expert_scale_offsets, [layer, 0]), 0), pl.INDEX)
        routed_w1_scale_storage = pl.slice(scales8, [(LOCAL_EXPERTS * (D // 32) * MOE_INTER) // 128, 128], [routed_w1_scale_offset // 128, 0])
        routed_w1_scale: pl.Tensor[[LOCAL_EXPERTS * (D // 32), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(routed_w1_scale_storage, [LOCAL_EXPERTS * (D // 32), MOE_INTER])
        routed_w3_offset = pl.cast(pl.max(pl.read(expert_offsets, [layer, 1]), 0), pl.INDEX)
        routed_w3_storage = pl.slice(bank4, [(LOCAL_EXPERTS * (D * MOE_INTER // 256) * 128) // 128, 128], [routed_w3_offset // 128, 0])
        routed_w3 = pl.reshape(routed_w3_storage, [LOCAL_EXPERTS, D * MOE_INTER // 256, 128])
        routed_w3_scale_offset = pl.cast(pl.max(pl.read(expert_scale_offsets, [layer, 1]), 0), pl.INDEX)
        routed_w3_scale_storage = pl.slice(scales8, [(LOCAL_EXPERTS * (D // 32) * MOE_INTER) // 128, 128], [routed_w3_scale_offset // 128, 0])
        routed_w3_scale: pl.Tensor[[LOCAL_EXPERTS * (D // 32), MOE_INTER], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(routed_w3_scale_storage, [LOCAL_EXPERTS * (D // 32), MOE_INTER])
        routed_w2_offset = pl.cast(pl.max(pl.read(expert_offsets, [layer, 2]), 0), pl.INDEX)
        routed_w2_storage = pl.slice(bank4, [(LOCAL_EXPERTS * (D * MOE_INTER // 256) * 128) // 128, 128], [routed_w2_offset // 128, 0])
        routed_w2 = pl.reshape(routed_w2_storage, [LOCAL_EXPERTS, D * MOE_INTER // 256, 128])
        routed_w2_scale_offset = pl.cast(pl.max(pl.read(expert_scale_offsets, [layer, 2]), 0), pl.INDEX)
        routed_w2_scale_storage = pl.slice(scales8, [(LOCAL_EXPERTS * (MOE_INTER // 32) * D) // 128, 128], [routed_w2_scale_offset // 128, 0])
        routed_w2_scale: pl.Tensor[[LOCAL_EXPERTS * (MOE_INTER // 32), D], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(routed_w2_scale_storage, [LOCAL_EXPERTS * (MOE_INTER // 32), D])
        if layer == 20:
            with pl.scope():
                # Publish C1A source KV for ALL encoder rows before selecting replay.
                encoder_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [0, 0, 0])
                encoder_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [0, 0])
                encoder_collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                encoder_normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
                mhc_pre(encoder_hidden, encoder_pre, encoder_collapsed)
                rms_norm(encoder_collapsed, attn_norm, encoder_normalized)
                publisher_ready = publish_decoder(
                    encoder_normalized, publisher_wkv, compressor_norm, index_wk, index_norm,
                    publisher_cos, publisher_sin, publisher_slots,
                    compressed_cache[3], compressed_scales[3], index_cache[3], index_scales[3],
                    pl.read(counts, [0]),
                )
                gathered_hidden = pl.reshape(hidden, [CAPACITY, HC_DIM])
                source_hidden = pl.reshape(encoder_hidden, [CAPACITY, HC_DIM])
                gather_rows(source_hidden, replay_rows, gathered_hidden)
                gather_rows(encoder_pre, replay_rows, delayed_pre)
        has_engram = 0
        if layer == 1:
            has_engram = 1
        elif layer == 14:
            has_engram = 1
        if has_engram == 1:
            with pl.scope():
                engram_weight_offset = pl.cast(pl.max(pl.read(fp8_offsets, [layer, 8]), 0), pl.INDEX)
                engram_weight_storage = pl.slice(bank8, [(6144 * 25600) // 128, 128], [engram_weight_offset // 128, 0])
                engram_weight = pl.reshape(engram_weight_storage, [6144, 25600])
                engram_scale_offset = pl.cast(pl.max(pl.read(fp8_scale_offsets, [layer, 8]), 0), pl.INDEX)
                engram_scale_storage = pl.slice(scales8, [(192 * 25600) // 128, 128], [engram_scale_offset // 128, 0])
                engram_scale: pl.Tensor[[192, 25600], pl.FP8E8M0, pl.MX_B_NN] = pl.reshape(engram_scale_storage, [192, 25600])
                engram_norm_offset = pl.cast(pl.max(pl.read(dense_offsets, [layer, 11]), 0), pl.INDEX)
                engram_norm_storage = pl.slice(dense, [HC_MULT * D], [engram_norm_offset])
                engram_norm = pl.reshape(engram_norm_storage, [HC_MULT, D])
                engram_output = pl.create_tensor([CAPACITY, HC_MULT, D], dtype=pl.FP32)
                engram_slot = layer // 14
                engram_mx(engram_lookup[engram_slot], engram_weight, engram_scale, engram_norm, hidden, engram_output)
                engram_flat = pl.reshape(engram_output, [CAPACITY, HC_DIM])
                hidden_flat = pl.reshape(hidden, [CAPACITY, HC_DIM])
                gather_rows(engram_flat, identity_rows, hidden_flat)
        with pl.scope():
            attn_post = pl.create_tensor([CAPACITY, HC_MULT], dtype=pl.FP32)
            attn_residual = pl.create_tensor([CAPACITY, HC_MULT, HC_MULT], dtype=pl.FP32)
            collapsed = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            normalized = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            attn_output = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
            zero_bf16_padding(attn_output, pl.cast(0, pl.INT32))
            mhc_mixes(hidden, attn_fn, attn_scale, attn_base, attn_pre, attn_post, attn_residual)
            mhc_pre(hidden, delayed_pre, collapsed)
            rms_norm(collapsed, attn_norm, normalized)
            mode = 0
            cache_slot = 0
            rope_slot = 0
            if layer >= 20:
                cache_slot = 3
                rope_slot = 2
                mode = 5
                if layer == 20:
                    mode = 3
                elif layer % 4 == 0:
                    mode = 4
            elif layer >= 2:
                cache_slot = (layer - 2) // 6
                rope_slot = 1
                mode = 2
                if (layer - 2) % 6 == 0:
                    mode = 1
            state_slot = pl.min(cache_slot, 2)
            attention(
                normalized, wq_a, wq_a_scale, q_norm, wq_b, wq_b_scale, wkv, wkv_scale, kv_norm,
                attn_sink, wo_a, wo_b, wo_b_scale, rope_cos[rope_slot], rope_sin[rope_slot],
                window_slots[stage], window_indices[stage], window_cache[layer], window_scales[layer],
                compressed_cache[cache_slot], compressed_scales[cache_slot], token_requests[stage],
                compressed_lens[stage], index_cache[cache_slot], index_scales[cache_slot],
                index_block_tables[stage], position_ids[stage], compressed_cos, compressed_sin,
                compressor_wkv, compressor_wgate, query_starts[stage], state_block_table, state_cache[state_slot],
                compressor_norm, compressed_slots, index_wk, index_norm, index_wq_b, index_wq_b_scale,
                index_weights, candidate_mask, topk_indices, attention_window, attention_arrived,
                attn_output, rank, 0, active, epoch_base + layer + 1, mode,
            )
            mhc_post(attn_output, hidden, attn_post, attn_residual, attn_hidden)
        moe(
            attn_hidden, attn_pre, ffn_fn, ffn_scale, ffn_base, ffn_norm, gate_weight, gate_bias,
            routed_w1, routed_w1_scale, routed_w2, routed_w2_scale, routed_w3, routed_w3_scale,
            mxfp4_pair_lut, shared_w1, shared_w1_scale, shared_w2, shared_w2_scale,
            shared_w3, shared_w3_scale, next_pre, ffn_mixed, next_hidden,
            recv_meta, recv_x, recv_scale, recv_weights, recv_routes, arrived, data_arrived,
            routed_output, combine_arrived, active, rank, epoch_base + layer + 1,
        )
    final_hidden = pl.slice(hidden_pair, [CAPACITY, HC_MULT, D], [2 * CAPACITY, 0, 0])
    final_pre = pl.slice(pre_pair, [CAPACITY, HC_MULT], [2 * CAPACITY, 0])
    collapsed_head = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    normalized_head = pl.create_tensor([CAPACITY, D], dtype=pl.BF16)
    hc_head(final_hidden, final_pre, collapsed_head)
    rms_norm(collapsed_head, final_norm, normalized_head)
    lm_head(normalized_head, head, logit_rows, logits, head_hidden, head_hidden_done,
            head_logits, head_logits_done, rank, 0, epoch_base // 40 + 1)
    greedy_sample(logits, sampled_ids)
    return logits, sampled_ids


@pl.jit.host
def l3_prefill_fwd(
    embedding: pl.Tensor[[RANKS, CAPACITY, D], pl.BF16],
    embedding_ids: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    initial_pre: pl.Tensor[[RANKS, CAPACITY, HC_MULT], pl.FP32],
    identity_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    engram_lookup: pl.Tensor[[RANKS, 2, CAPACITY, 6144], pl.BF16],
    bank8: pl.Tensor[[RANKS, B8, 128], pl.FP8E4M3FN],
    bank4: pl.Tensor[[RANKS, B4, 128], pl.UINT8],
    scales8: pl.Tensor[[RANKS, S8, 128], pl.FP8E8M0],
    bf16: pl.Tensor[[RANKS, B16], pl.BF16],
    dense: pl.Tensor[[RANKS, B32], pl.FP32],
    fp8_offsets: pl.Tensor[[RANKS, 40, 9], pl.INT64],
    fp8_scale_offsets: pl.Tensor[[RANKS, 40, 9], pl.INT64],
    bf16_offsets: pl.Tensor[[RANKS, 40, 10], pl.INT64],
    dense_offsets: pl.Tensor[[RANKS, 40, 12], pl.INT64],
    expert_offsets: pl.Tensor[[RANKS, 40, 3], pl.INT64],
    expert_scale_offsets: pl.Tensor[[RANKS, 40, 3], pl.INT64],
    mxfp4_pair_lut: pl.Tensor[[RANKS, 2, 256], pl.INT16],
    counts: pl.Tensor[[RANKS, 2], pl.INT32],
    replay_rows: pl.Tensor[[RANKS, CAPACITY], pl.INT32],
    position_ids: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    rope_cos: pl.Tensor[[RANKS, 3, CAPACITY, 32], pl.FP32],
    rope_sin: pl.Tensor[[RANKS, 3, CAPACITY, 32], pl.FP32],
    compressed_cos: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    compressed_sin: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    publisher_cos: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    publisher_sin: pl.Tensor[[RANKS, CAPACITY, 32], pl.FP32],
    window_slots: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT64],
    window_indices: pl.Tensor[[RANKS, 2, CAPACITY, 128], pl.INT32],
    compressed_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    publisher_slots: pl.Tensor[[RANKS, CAPACITY], pl.INT64],
    token_requests: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    compressed_lens: pl.Tensor[[RANKS, 2, CAPACITY], pl.INT32],
    query_starts: pl.Tensor[[RANKS, 2, 2], pl.INT32],
    index_block_tables: pl.Tensor[[RANKS, 2, 1, PAGES], pl.INT32],
    state_block_table: pl.Tensor[[RANKS, 1, 1], pl.INT32],
    window_cache: pl.InOut[pl.Tensor[[RANKS, 40, PAGES, 128, 1, 512], pl.FP8E4M3FN]],
    window_scales: pl.InOut[pl.Tensor[[RANKS, 40, PAGES, 128, 1, 16], pl.FP8E8M0]],
    compressed_cache: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 256], pl.UINT8]],
    compressed_scales: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 32], pl.FP8E4M3FN]],
    index_cache: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 64], pl.UINT8]],
    index_scales: pl.InOut[pl.Tensor[[RANKS, 4, PAGES, 128, 1, 4], pl.FP8E8M0]],
    state_cache: pl.InOut[pl.Tensor[[RANKS, 3, 2, C.STATE_CAPACITY, C.STATE_WIDTH], pl.FP32]],
    final_norm: pl.Tensor[[RANKS, D], pl.BF16],
    head: pl.Tensor[[RANKS, VOCAB, D], pl.BF16],
    logit_rows: pl.Tensor[[RANKS, 16], pl.INT32],
    logits: pl.Out[pl.Tensor[[RANKS, 16, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[RANKS, 16, 8], pl.INT32]],
    epoch_base: pl.Scalar[pl.INT32],
):
    """Allocate native communication windows and launch one complete L2 per rank."""
    recv_meta_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
    recv_x_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
    recv_scale_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
    recv_weights_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
    recv_routes_buffer = pld.alloc_window_buffer([LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
    arrived_buffer = pld.alloc_window_buffer([RANKS, 128], dtype=pl.INT32)
    data_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    routed_output_buffer = pld.alloc_window_buffer([CAPACITY * TOPK, D], dtype=pl.BF16)
    combine_arrived_buffer = pld.alloc_window_buffer([RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
    attention_window_buffer = pld.alloc_window_buffer([PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
    attention_arrived_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    head_hidden_buffer = pld.alloc_window_buffer([16, D], dtype=pl.BF16)
    head_hidden_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    head_logits_buffer = pld.alloc_window_buffer([16, VOCAB], dtype=pl.FP32)
    head_logits_done_buffer = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    for rank in pl.range(pld.world_size()):
        recv_meta = pld.window(recv_meta_buffer, [RANKS, LOCAL_EXPERTS], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buffer, [LOCAL_EXPERTS * RECV_MAX, D], dtype=pl.INT8)
        recv_scale = pld.window(recv_scale_buffer, [LOCAL_EXPERTS * RECV_MAX, D // 32], dtype=pl.UINT8)
        recv_weights = pld.window(recv_weights_buffer, [LOCAL_EXPERTS * RECV_MAX, AUX_WIDTH], dtype=pl.FP32)
        recv_routes = pld.window(recv_routes_buffer, [LOCAL_EXPERTS * RECV_MAX, ROUTE_WIDTH], dtype=pl.INT32)
        arrived = pld.window(arrived_buffer, [RANKS, 128], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        routed_output = pld.window(routed_output_buffer, [CAPACITY * TOPK, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buffer, [RANKS, LOCAL_EXPERTS, 128], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buffer, [PREFILL_MAX_TOKENS, D], dtype=pl.FP32)
        attention_arrived = pld.window(attention_arrived_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_hidden = pld.window(head_hidden_buffer, [16, D], dtype=pl.BF16)
        head_hidden_done = pld.window(head_hidden_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        head_logits = pld.window(head_logits_buffer, [16, VOCAB], dtype=pl.FP32)
        head_logits_done = pld.window(head_logits_done_buffer, [TP_SIZE, 1], dtype=pl.INT32)
        prefill_fwd(
            embedding[rank],
            embedding_ids[rank],
            initial_pre[rank],
            identity_rows[rank],
            engram_lookup[rank],
            bank8[rank],
            bank4[rank],
            scales8[rank],
            bf16[rank],
            dense[rank],
            fp8_offsets[rank],
            fp8_scale_offsets[rank],
            bf16_offsets[rank],
            dense_offsets[rank],
            expert_offsets[rank],
            expert_scale_offsets[rank],
            mxfp4_pair_lut[rank],
            counts[rank],
            replay_rows[rank],
            position_ids[rank],
            rope_cos[rank],
            rope_sin[rank],
            compressed_cos[rank],
            compressed_sin[rank],
            publisher_cos[rank],
            publisher_sin[rank],
            window_slots[rank],
            window_indices[rank],
            compressed_slots[rank],
            publisher_slots[rank],
            token_requests[rank],
            compressed_lens[rank],
            query_starts[rank],
            index_block_tables[rank],
            state_block_table[rank],
            window_cache[rank],
            window_scales[rank],
            compressed_cache[rank],
            compressed_scales[rank],
            index_cache[rank],
            index_scales[rank],
            state_cache[rank],
            final_norm[rank],
            head[rank],
            logit_rows[rank],
            logits[rank],
            sampled_ids[rank],
            recv_meta,
            recv_x,
            recv_scale,
            recv_weights,
            recv_routes,
            arrived,
            data_arrived,
            routed_output,
            combine_arrived,
            attention_window,
            attention_arrived,
            head_hidden,
            head_hidden_done,
            head_logits,
            head_logits_done,
            rank, epoch_base, device=rank,
        )


PREFILL_TENSORS = (
    "embedding",
    "embedding_ids",
    "initial_pre",
    "identity_rows",
    "engram_lookup",
    "bank8",
    "bank4",
    "scales8",
    "bf16",
    "dense",
    "fp8_offsets",
    "fp8_scale_offsets",
    "bf16_offsets",
    "dense_offsets",
    "expert_offsets",
    "expert_scale_offsets",
    "mxfp4_pair_lut",
    "counts",
    "replay_rows",
    "position_ids",
    "rope_cos",
    "rope_sin",
    "compressed_cos",
    "compressed_sin",
    "publisher_cos",
    "publisher_sin",
    "window_slots",
    "window_indices",
    "compressed_slots",
    "publisher_slots",
    "token_requests",
    "compressed_lens",
    "query_starts",
    "index_block_tables",
    "state_block_table",
    "window_cache",
    "window_scales",
    "compressed_cache",
    "compressed_scales",
    "index_cache",
    "index_scales",
    "state_cache",
    "final_norm",
    "head",
    "logit_rows",
    "logits",
    "sampled_ids",
)


def generate(checkpoint, compiled, device_ids, prompts, max_new_tokens):
    """Generate four independent continuations using resident checkpoint weights.

    Each forward gets fresh caches and communication windows. Weights and tensor
    allocations survive across forwards; no CPU network reference is evaluated.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(checkpoint.root / "tokenizer.json"))
    spec = importlib.util.spec_from_file_location(
        "v41_checkpoint_encoding", checkpoint.root / "encoding/encoding.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("checkpoint is missing the official prompt encoder")
    encoding = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoding)
    sequences = [
        tokenizer.encode(
            encoding.encode_messages([{"role": "user", "content": prompt}], thinking_mode="chat"),
            add_special_tokens=False,
        ).ids
        for prompt in prompts
    ]
    if any(len(sequence) + max_new_tokens > CAPACITY for sequence in sequences):
        raise ValueError("prompt plus generation exceeds --capacity; compile a larger capacity")
    tokenizer_config = json.loads((checkpoint.root / "tokenizer_config.json").read_text())
    eos = tokenizer_config["eos_token"]
    eos_id = tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else eos)
    if eos_id is None:
        raise ValueError("checkpoint EOS token is missing from its tokenizer")
    weights = ResidentPrefillWeights(checkpoint)
    config = RunConfig(
        platform="a5", ring_heap=(2 * 1024**3,) * 4,
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    outputs = {
        "logits": ((16, VOCAB), torch.float32),
        "sampled_ids": ((16, 8), torch.int32),
    }
    constants = {*weights.tensor_layouts, *weights.offset_tensors, "head", "final_norm", "mxfp4_pair_lut"}
    generated = [[] for _ in prompts]
    finished = [False] * RANKS
    print("CED_RESIDENT_BYTES_PER_RANK", weights.bytes_per_rank, flush=True)
    start = time.monotonic()
    # Fresh CommDomains start epoch 1 for every complete prefix evaluation.
    with compiled.prepare(config, persistent=False) as runtime:
        per_rank = [{} for _ in device_ids]
        for rank, tensors in enumerate(per_rank):
            inputs = prepare_resident_inputs(checkpoint, weights, sequences[rank], CAPACITY)
            inputs["mxfp4_pair_lut"] = build_mxfp4_pair_lut()
            for name in PREFILL_TENSORS:
                if name in weights.tensor_layouts:
                    shape, dtype = weights.tensor_layouts[name]
                    tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                elif name in outputs:
                    shape, dtype = outputs[name]
                    tensors[name] = runtime.alloc_tensor(shape, dtype, worker_id=rank)
                else:
                    value = inputs[name]
                    tensors[name] = runtime.alloc_tensor(value.shape, value.dtype, init=value, worker_id=rank)

        def upload(rank):
            weights.upload(runtime, rank, per_rank[rank])
            print("CED_RANK_WEIGHTS_READY", rank, flush=True)

        with ThreadPoolExecutor(max_workers=RANKS) as executor:
            list(executor.map(upload, range(RANKS)))
        stacked = {
            name: StackedDeviceTensor(
                [rank[name] for rank in per_rank], (RANKS, *per_rank[0][name].shape), tuple(range(RANKS))
            )
            for name in PREFILL_TENSORS
        }
        logits = torch.empty((RANKS, 16, VOCAB), dtype=torch.float32)
        sampled_ids = torch.empty((RANKS, 16, 8), dtype=torch.int32)
        for step in range(max_new_tokens):
            if step:
                for rank, tensors in enumerate(per_rank):
                    inputs = prepare_resident_inputs(checkpoint, weights, sequences[rank], CAPACITY)
                    for name, value in inputs.items():
                        if name not in constants:
                            runtime.copy_to(
                                tensors[name].data_ptr, value.data_ptr(), value.numel() * value.element_size(),
                                worker_id=rank,
                            )
            tick = time.monotonic()
            print("CED_DEVICE_START", step, flush=True)
            runtime(*[stacked[name] for name in PREFILL_TENSORS], 0, config=config)
            runtime.copy_stacked_from(stacked["logits"], logits)
            runtime.copy_stacked_from(stacked["sampled_ids"], sampled_ids)
            if not bool(torch.isfinite(logits[:, 0]).all()):
                raise FloatingPointError(f"nonfinite last-token logits at step {step}")
            for rank in range(RANKS):
                token = int(sampled_ids[rank, 0, 0])
                if token != int(logits[rank, 0].argmax()):
                    raise AssertionError("device greedy sampling disagrees with copied logits")
                if not finished[rank]:
                    generated[rank].append(token)
                    sequences[rank].append(token)
                    finished[rank] = token == eos_id
            print("CED_GENERATION_STEP", json.dumps({
                "step": step, "seconds": time.monotonic() - tick,
                "token_ids": sampled_ids[:, 0, 0].tolist(),
                "text": [tokenizer.decode(tokens, skip_special_tokens=False) for tokens in generated],
            }, ensure_ascii=False), flush=True)
            if all(finished):
                break
    return {
        "platform": "a5", "tp": 1, "ep": RANKS, "dp": RANKS,
        "precision": "official_bf16_mxfp8_mxfp4", "checkpoint": str(checkpoint.root),
        "capacity": CAPACITY, "full_prefix_recomputation": True,
        "layers_per_forward": 40, "l2_invocations_per_rank_per_forward": 1,
        "seconds": time.monotonic() - start,
        "cases": [
            {"prompt": prompt, "token_ids": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=False),
             "eos_reached": done}
            for prompt, tokens, done in zip(prompts, generated, finished)
        ],
    }


def main(argv=None):
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True, help="previously compiled L3 artifact")
    parser.add_argument("--devices", default=os.environ.get("TASK_DEVICE"))
    parser.add_argument("--tp", type=int, default=C.TP_SIZE, choices=[1])
    parser.add_argument("--ep", type=int, default=C.EP_SIZE, choices=[4])
    parser.add_argument("--dp", type=int, default=4, choices=[4])
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    parser.add_argument("--prompt", action="append", help="one prompt per rank; provide exactly four")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--report", type=Path)
    options = parser.parse_args(argv)
    if (C.TP_SIZE, C.EP_SIZE) != (1, 4):
        raise ValueError("compile and run with --tp 1 --ep 4")
    if options.capacity != CAPACITY or not 16 <= CAPACITY <= C.PREFILL_MAX_TOKENS or CAPACITY % 16:
        raise ValueError("--capacity must match compilation and be a positive multiple of 16")
    device_ids = [int(value) for value in (options.devices or "").split(",") if value]
    if len(device_ids) != RANKS or len(set(device_ids)) != RANKS:
        raise ValueError("provide four distinct task-allocated A5 devices")
    prompts = options.prompt or [
        "What is 2 + 3? Reply with just the number.",
        "What is the capital of France? Reply with just the city name.",
        "Translate 'hello' into Chinese. Reply with only the translation.",
        "Continue the sequence with three numbers: 1, 2, 3,",
    ]
    if len(prompts) != RANKS or options.max_new_tokens <= 0:
        raise ValueError("provide four prompts and a positive --max-new-tokens")
    torch.set_num_threads(8)
    checkpoint = PrefillCheckpoint(options.checkpoint)
    compiled = DistributedCompiledProgram.from_dir(
        options.runtime_dir, platform="a5",
        distributed_config=DistributedConfig(device_ids=device_ids, num_sub_workers=0),
    )
    report = generate(checkpoint, compiled, device_ids, prompts, options.max_new_tokens)
    if options.report:
        options.report.parent.mkdir(parents=True, exist_ok=True)
        options.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("CED_GENERATION_COMPLETE", json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    main()
