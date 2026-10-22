# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
# ci: no-sim
"""GLM-5.3-Flash decode backbone and persistent-state validation.

The input is a batch of embeddings already prepared by the caller. The output
is the mHC stream state and the final normalized hidden state. This entry
composes the implemented mainline attention and MLP operators; token lookup,
vocabulary projection, sampling, MTP and serving integration are out of scope.

``--dense-dtype bf16`` selects resolved BF16 Dense weights and activations;
its unused scale arguments retain the W8A8 forward ABI. The mode is fixed when
this module is imported. Attention, caches and MoE retain their existing contracts.

The default validation executes one KDA/dense layer, 16 requests, and two decode
steps with persistent caches. The fixture retains four layers; ``--end-layer 4``
explicitly runs the full three-KDA/dense plus one-DSA/MoE chain outside CI.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
if __name__ == "__main__" and not any(a == "--ep" or a.startswith("--ep=") for a in sys.argv):
    sys.argv += ["--ep", "2"]

import pypto.language as pl
import pypto.language.distributed as pld
import torch

from models.glm5_3_flash.config import (
    D, EP_SIZE, TP_SIZE, HC_DIM, HC_MULT, MIX_HC, KDA_DIM, KDA_CONV_K, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM,
)
from models.glm5_3_flash.attention_tp import tp_all_reduce
from models.glm5_3_flash.kda_projection import BETA_PAD, kda_projection, golden_kda_projection
from models.glm5_3_flash.kda_conv import CONV_DIM, STATE_LEN, kda_conv_decode, golden_kda_conv_decode
from models.glm5_3_flash.decode_kda import decode_kda, golden_decode_kda
from models.glm5_3_flash.kda_output import kda_output, golden_kda_output
from models.glm5_3_flash.mhc import mhc_mixes, mhc_pre, mhc_post, mhc_head
from models.glm5_3_flash.mhc import golden_mhc_mixes, golden_mhc_pre, golden_mhc_post, golden_mhc_head
from models.glm5_3_flash.rmsnorm import rmsnorm, rmsnorm_quant, golden_rmsnorm, golden_rmsnorm_quant
from models.glm5_3_flash.dense_mlp import LOCAL_DENSE_INTER, dense_mlp, dense_mlp_bf16

from models.glm5_3_flash.config import (
    BLOCK_SIZE, INDEX_DIM, INDEX_H, INDEX_KPOOL, INDEX_STATE_WIDTH, KPOOL_SELECT_K,
    KV_LORA, LOCAL_H, Q_LORA, QK_DIM, TOPK_INDEX_WIDTH, N_EXPERTS, N_LOCAL_EXPERTS, MOE_INTER,
)
from models.glm5_3_flash.mla_prolog import mla_prolog, absorb_query
from models.glm5_3_flash.mla_cache import mla_cache_write
from models.glm5_3_flash.decode_sparse_attn import decode_sparse_attn
from models.glm5_3_flash.mla_epilog import mla_epilog_decode
from models.glm5_3_flash.indexer import indexer_proj, indexer_score_token, indexer_topk, indexer_expand
from models.glm5_3_flash.indexer_cache import (
    POOL_SCALE_WIDTH, POOL_VALID_WIDTH, indexer_cache_write, indexer_pool_write,
)
from models.glm5_3_flash.decode_layer import decode_mlp
from models.glm5_3_flash.decode_moe import LANE_ROWS, AUX_PAD, COUNT_PAD, N_ROUTES

_shape_parser = argparse.ArgumentParser(add_help=False)
_shape_parser.add_argument("--layers", type=int, default=4)
_shape_parser.add_argument("--start-layer", type=int, default=0)
_shape_parser.add_argument("--end-layer", type=int)
_shape_parser.add_argument("--steps", type=int, default=2)
_shape_parser.add_argument("--dense-dtype", choices=["int8", "bf16"], default="int8")
_shape_args = _shape_parser.parse_known_args()[0]
N_LAYERS = _shape_args.layers
START_LAYER = _shape_args.start_layer
END_LAYER = START_LAYER + 1 if _shape_args.end_layer is None else _shape_args.end_layer
STEPS = _shape_args.steps
DENSE_BF16 = _shape_args.dense_dtype == "bf16"
DENSE_PL_DTYPE = pl.BF16 if DENSE_BF16 else pl.INT8
DENSE_TORCH_DTYPE = torch.bfloat16 if DENSE_BF16 else torch.int8
T = 16
# Attention output, MLP output, MLP input, normalized attention input.
TRACE_PHASES = 4
STATE_ROWS = T + 4
N_KDA = N_LAYERS - N_LAYERS // 4
N_DSA = max(1, N_LAYERS // 4)
N_DENSE = min(3, N_LAYERS)
N_MOE = max(1, N_LAYERS - 3)
RAW_ROWS = T * BLOCK_SIZE
POOL_ROWS = 2048
if not 0 <= START_LAYER < END_LAYER <= N_LAYERS <= 45 or not 1 <= STEPS <= 8:
    raise ValueError("decode requires 0 <= start < end <= layers <= 45 and 1-8 validation steps")
if TP_SIZE != EP_SIZE:
    raise ValueError("decode forward currently requires TP_SIZE == EP_SIZE")


if DENSE_BF16:
    @pl.jit.inline
    def dense_input_norm(
        mixed: pl.Tensor[[T, D], pl.BF16],
        norm_weight: pl.Tensor[[D], pl.BF16],
        normalized: pl.Tensor[[T, D], pl.BF16],
        dense_input: pl.Tensor[[T, D], pl.INT8],
        input_scale: pl.Tensor[[T, 1], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        """Return BF16 normalized inputs through the shared forward input ABI."""
        rmsnorm(mixed, norm_weight, normalized)
        return normalized, normalized, input_scale

    @pl.jit.inline
    def dense_forward(
        dense_input: pl.Tensor[[T, D], DENSE_PL_DTYPE],
        input_scale: pl.Tensor[[T, 1], pl.FP32],
        dense_gate_up: pl.Tensor[[2 * LOCAL_DENSE_INTER, D], DENSE_PL_DTYPE],
        dense_gate_up_scale: pl.Tensor[[2 * LOCAL_DENSE_INTER], pl.FP32],
        dense_down: pl.Tensor[[D, LOCAL_DENSE_INTER], DENSE_PL_DTYPE],
        dense_down_scale: pl.Tensor[[D], pl.FP32],
        partial: pl.Tensor[[T, D], pl.FP32],
    ):
        bf16_mlp_local = pl.create_tensor([T, D], dtype=pl.BF16)
        dense_mlp_bf16(dense_input, dense_gate_up, dense_down, bf16_mlp_local)
        for row in pl.spmd(T, name_hint="dense_bf16_exchange_input"):
            for col in pl.range(0, D, 512):
                bf16_mlp_tile = bf16_mlp_local[row:row + 1, col:col + 512]
                partial[row:row + 1, col:col + 512] = pl.cast(bf16_mlp_tile, pl.FP32)
        return partial

else:
    dense_input_norm = rmsnorm_quant
    dense_forward = dense_mlp


@pl.jit(auto_scope=False)
def decode_fwd(
    inputs_embeds: pl.Tensor[[STEPS, T, D], pl.BF16],
    state_rows: pl.Tensor[[STEPS, T], pl.INT32],
    hc_function: pl.Tensor[[N_LAYERS, 2, MIX_HC, HC_DIM], pl.BF16],
    hc_scale: pl.Tensor[[N_LAYERS, 2, 3], pl.FP32],
    hc_base: pl.Tensor[[N_LAYERS, 2, MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[N_LAYERS, 2, D], pl.BF16],
    w_q: pl.Tensor[[N_KDA, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[N_KDA, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[N_KDA, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[N_KDA, KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[N_KDA, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[N_KDA, LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[N_KDA, LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[N_KDA, LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[N_KDA, KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[N_KDA, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    conv_weight: pl.Tensor[[N_KDA, KDA_CONV_K, CONV_DIM], pl.BF16],
    out_norm: pl.Tensor[[N_KDA, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[N_KDA, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    dense_gate_up: pl.Tensor[[N_DENSE, 2 * LOCAL_DENSE_INTER, D], DENSE_PL_DTYPE],
    dense_gate_up_scale: pl.Tensor[[N_DENSE, 2 * LOCAL_DENSE_INTER], pl.FP32],
    dense_down: pl.Tensor[[N_DENSE, D, LOCAL_DENSE_INTER], DENSE_PL_DTYPE],
    dense_down_scale: pl.Tensor[[N_DENSE, D], pl.FP32],
    final_norm: pl.Tensor[[D], pl.BF16],
    mla_w_q_a: pl.Tensor[[N_DSA, Q_LORA, D], pl.BF16],
    mla_q_norm: pl.Tensor[[N_DSA, Q_LORA], pl.BF16],
    mla_w_q_b: pl.Tensor[[N_DSA, LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    mla_w_kv_a: pl.Tensor[[N_DSA, KV_LORA, D], pl.BF16],
    mla_kv_norm: pl.Tensor[[N_DSA, KV_LORA], pl.BF16],
    mla_w_k: pl.Tensor[[N_DSA, LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    mla_w_o: pl.Tensor[[N_DSA, D, LOCAL_H * KV_LORA], pl.BF16],
    index_w_q_b: pl.Tensor[[N_DSA, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    index_w_k: pl.Tensor[[N_DSA, D, INDEX_DIM], pl.BF16],
    index_norm: pl.Tensor[[N_DSA, INDEX_DIM], pl.BF16],
    index_bias: pl.Tensor[[N_DSA, INDEX_DIM], pl.BF16],
    index_weights: pl.Tensor[[N_DSA, D, INDEX_H], pl.BF16],
    index_gate: pl.Tensor[[N_DSA, D, INDEX_DIM], pl.BF16],
    compress_ape: pl.Tensor[[N_DSA, INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[INDEX_DIM, INDEX_DIM], pl.BF16],
    gate_weight: pl.Tensor[[N_MOE, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[N_MOE, N_EXPERTS], pl.FP32],
    expert_gate_up: pl.Tensor[[N_MOE, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    expert_gate_up_scale: pl.Tensor[[N_MOE, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    expert_down: pl.Tensor[[N_MOE, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_down_scale: pl.Tensor[[N_MOE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[N_MOE, 2 * MOE_INTER, D], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[N_MOE, 2 * MOE_INTER], pl.FP32],
    shared_down: pl.Tensor[[N_MOE, D, MOE_INTER], pl.INT8],
    shared_down_scale: pl.Tensor[[N_MOE, D], pl.FP32],
    kv_slots: pl.Tensor[[STEPS, T], pl.INT32],
    pool_token_slots: pl.Tensor[[STEPS, T, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[STEPS, T], pl.INT32],
    pool_blocks: pl.Tensor[[POOL_ROWS], pl.INT32],
    seg_start: pl.Tensor[[T], pl.INT32],
    pool_count: pl.Tensor[[STEPS, T], pl.INT32],
    tail_start: pl.Tensor[[STEPS, T], pl.INT32],
    tail_count: pl.Tensor[[STEPS, T], pl.INT32],
    kv_len: pl.Tensor[[STEPS, T], pl.INT32],
    block_table: pl.Tensor[[T, 1], pl.INT32],
    request_ids: pl.Tensor[[T], pl.INT32],
    latent_cache: pl.InOut[pl.Tensor[[N_DSA, RAW_ROWS, KV_LORA], pl.BF16]],
    raw_cache: pl.InOut[pl.Tensor[[N_DSA, RAW_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    pool_cache: pl.InOut[pl.Tensor[[N_DSA, POOL_ROWS, INDEX_DIM], pl.INT8]],
    pool_scale: pl.InOut[pl.Tensor[[N_DSA, POOL_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    conv_state: pl.InOut[pl.Tensor[[N_KDA, STATE_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    recurrent_state: pl.InOut[pl.Tensor[[N_KDA, STATE_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    layer_outputs: pl.InOut[pl.Tensor[[STEPS, N_LAYERS, TRACE_PHASES, T, D], pl.BF16]],
    streams: pl.Out[pl.Tensor[[STEPS, T, HC_MULT, D], pl.BF16]],
    hidden: pl.Out[pl.Tensor[[STEPS, T, D], pl.BF16]],
    lane_x: pld.DistributedTensor[[LANE_ROWS, D], pl.INT8],
    lane_aux: pld.DistributedTensor[[LANE_ROWS, AUX_PAD], pl.FP32],
    lane_count: pld.DistributedTensor[[EP_SIZE, COUNT_PAD], pl.INT32],
    lane_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    slot_out: pld.DistributedTensor[[N_ROUTES, D], pl.FP32],
    return_ready: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    exchange: pld.DistributedTensor[[T, D], pl.FP32],
    arrived: pld.DistributedTensor[[EP_SIZE, 1], pl.INT32],
    rank: pl.Scalar[pl.INT32],
    step: pl.Scalar[pl.INT32],
):
    inputs_embeds_step = inputs_embeds[step]
    state_rows_step = state_rows[step]
    streams_step = streams[step]
    hidden_step = hidden[step]
    x_hc = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
    x_flat = pl.reshape(x_hc, [T, HC_DIM])
    for t in pl.spmd(T, name_hint="decode_seed"):
        for h in pl.unroll(HC_MULT):
            x_flat[t:t + 1, h * D:(h + 1) * D] = inputs_embeds_step[t:t + 1, :]
    for layer in pl.range(START_LAYER, END_LAYER):
        with pl.scope():
            pre_mix = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
            post_mix = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
            residual_mix = pl.create_tensor([T, HC_MULT, HC_MULT], dtype=pl.FP32)
            mixed = pl.create_tensor([T, D], dtype=pl.BF16)
            normed = pl.create_tensor([T, D], dtype=pl.BF16)
            mhc_mixes(x_hc, hc_function[layer, 0], hc_scale[layer, 0], hc_base[layer, 0],
                      pre_mix, post_mix, residual_mix)
            mhc_pre(x_hc, pre_mix, mixed)
            rmsnorm(mixed, norm_weight[layer, 0], normed)
            norm_trace = layer_outputs[step, layer, 3]
            for row in pl.spmd(T, name_hint="attention_input_trace"):
                norm_trace[row:row + 1, :] = normed[row:row + 1, :]
            partial = pl.create_tensor([T, D], dtype=pl.FP32)
            if END_LAYER <= 3 or layer % 4 != 3:
                linear_layer = layer - layer // 4
                qkv = pl.create_tensor([T, CONV_DIM], dtype=pl.BF16)
                decay = pl.create_tensor([T, LOCAL_KDA_H, KDA_DIM], dtype=pl.FP32)
                beta = pl.create_tensor([T, BETA_PAD], dtype=pl.FP32)
                out_gate = pl.create_tensor([T, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
                kda_projection(
                    normed, w_q[linear_layer], w_k[linear_layer], w_v[linear_layer],
                    w_f_a[linear_layer], w_f_b[linear_layer], dt_bias[linear_layer], a_log[linear_layer],
                    w_b[linear_layer], w_g_a[linear_layer], w_g_b[linear_layer],
                    qkv, decay, beta, out_gate,
                )
                query = pl.create_tensor([T, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
                key = pl.create_tensor([T, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
                value = pl.create_tensor([T, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
                kda_conv_decode(qkv, conv_weight[linear_layer], conv_state[linear_layer], state_rows_step, query, key, value)
                core_out = pl.create_tensor([T, LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)
                decode_kda(query, key, value, decay, beta, recurrent_state[linear_layer], state_rows_step, core_out)
                kda_output(core_out, out_norm[linear_layer], out_gate, w_o[linear_layer], partial)
            else:
                sparse_layer = layer // 4
                q_resid = pl.create_tensor([T, Q_LORA], dtype=pl.BF16)
                mla_query = pl.create_tensor([T, LOCAL_H, QK_DIM], dtype=pl.BF16)
                latent = pl.create_tensor([T, KV_LORA], dtype=pl.BF16)
                mla_prolog(normed, mla_w_q_a[sparse_layer], mla_q_norm[sparse_layer],
                           mla_w_q_b[sparse_layer], mla_w_kv_a[sparse_layer], mla_kv_norm[sparse_layer],
                           q_resid, mla_query, latent)
                mla_cache_write(latent, kv_slots[step], latent_cache[sparse_layer])
                index_q = pl.create_tensor([T, INDEX_H, INDEX_DIM], dtype=pl.BF16)
                index_k = pl.create_tensor([T, INDEX_DIM], dtype=pl.BF16)
                head_weights = pl.create_tensor([T, INDEX_H], dtype=pl.FP32)
                gate_scores = pl.create_tensor([T, INDEX_DIM], dtype=pl.FP32)
                indexer_proj(normed, q_resid, index_w_q_b[sparse_layer], index_w_k[sparse_layer],
                             index_norm[sparse_layer], index_bias[sparse_layer], index_weights[sparse_layer],
                             index_gate[sparse_layer], index_q, index_k, head_weights, gate_scores)
                indexer_cache_write(index_k, gate_scores, kv_slots[step], raw_cache[sparse_layer])
                pool_valid = pl.create_tensor([T, POOL_VALID_WIDTH], dtype=pl.INT32)
                indexer_pool_write(raw_cache[sparse_layer], compress_ape[sparse_layer], hadamard,
                                   pool_token_slots[step], pool_slots[step], pool_cache[sparse_layer],
                                   pool_scale[sparse_layer], pool_valid)
                scores = pl.create_tensor([T, POOL_ROWS], dtype=pl.FP32)
                indexer_score_token(index_q, hadamard, pool_cache[sparse_layer], pool_scale[sparse_layer],
                                    pool_blocks, head_weights, seg_start, pool_count[step], scores)
                selected = pl.create_tensor([T, KPOOL_SELECT_K], dtype=pl.INT32)
                valid = pl.create_tensor([T, KPOOL_SELECT_K], dtype=pl.INT32)
                indices = pl.create_tensor([T, TOPK_INDEX_WIDTH], dtype=pl.INT32)
                indexer_topk(scores, seg_start, pool_count[step], selected, valid)
                indexer_expand(selected, valid, tail_start[step], tail_count[step], kv_len[step], indices)
                absorbed = pl.create_tensor([T, LOCAL_H, KV_LORA], dtype=pl.BF16)
                absorb_query(mla_query, mla_w_k[sparse_layer], absorbed)
                context = pl.create_tensor([T, LOCAL_H, KV_LORA], dtype=pl.BF16)
                decode_sparse_attn(absorbed, latent_cache[sparse_layer], block_table, request_ids, indices, context)
                mla_epilog_decode(context, mla_w_o[sparse_layer], partial)
            reduced = pl.create_tensor([T, D], dtype=pl.BF16)
            epoch = pl.cast(step * (2 * (END_LAYER - START_LAYER)) + 2 * (layer - START_LAYER) + 1, pl.INT32)
            tp_all_reduce(partial, exchange, arrived, reduced, T, rank, 0, epoch)
            attention_trace = layer_outputs[step, layer, 0]
            for row in pl.spmd(T, name_hint="attention_trace"):
                attention_trace[row:row + 1, :] = reduced[row:row + 1, :]
            after_attn = pl.create_tensor([T, HC_MULT, D], dtype=pl.BF16)
            mhc_post(reduced, x_hc, post_mix, residual_mix, after_attn)

            mhc_mixes(after_attn, hc_function[layer, 1], hc_scale[layer, 1], hc_base[layer, 1],
                      pre_mix, post_mix, residual_mix)
            mixed, mlp_input_tid = mhc_pre(after_attn, pre_mix, mixed)
            input_trace = layer_outputs[step, layer, 2]
            for row in pl.spmd(T, name_hint="mlp_input_trace"):
                input_trace[row:row + 1, :] = mixed[row:row + 1, :]
            if END_LAYER <= 3 or layer < 3:
                x_int8 = pl.create_tensor([T, D], dtype=pl.INT8)
                input_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
                mlp_normed = pl.create_tensor([T, D], dtype=pl.BF16)
                mlp_normed, dense_input, input_scale = dense_input_norm(
                    mixed, norm_weight[layer, 1], mlp_normed, x_int8, input_scale, T,
                )
                dense_forward(
                    dense_input, input_scale, dense_gate_up[layer], dense_gate_up_scale[layer],
                    dense_down[layer], dense_down_scale[layer], partial,
                )
            else:
                expert_layer = layer - 3
                owned = pl.cast(0, pl.INT32)
                if rank == 0:
                    owned = pl.cast(T, pl.INT32)
                moe_epoch = pl.cast(step * N_MOE + expert_layer + 1, pl.INT32)
                decode_mlp(mixed, norm_weight[layer, 1], gate_weight[expert_layer], correction_bias[expert_layer],
                           expert_gate_up[expert_layer], expert_gate_up_scale[expert_layer],
                           expert_down[expert_layer], expert_down_scale[expert_layer],
                           shared_gate_up[expert_layer], shared_gate_up_scale[expert_layer],
                           shared_down[expert_layer], shared_down_scale[expert_layer], partial,
                           lane_x, lane_aux, lane_count, lane_ready, slot_out, return_ready,
                           owned, EP_SIZE, rank, moe_epoch, mlp_input_tid)
            mlp_reduced = pl.create_tensor([T, D], dtype=pl.BF16)
            tp_all_reduce(partial, exchange, arrived, mlp_reduced, T, rank, 0, epoch + 1)
            mlp_trace = layer_outputs[step, layer, 1]
            for row in pl.spmd(T, name_hint="mlp_trace"):
                mlp_trace[row:row + 1, :] = mlp_reduced[row:row + 1, :]
            mhc_post(mlp_reduced, after_attn, post_mix, residual_mix, x_hc)
    out_flat = pl.reshape(streams_step, [T, HC_DIM])
    for row in pl.spmd(T, name_hint="decode_stream_output"):
        for col in pl.range(0, HC_DIM, D):
            out_flat[row:row + 1, col:col + D] = x_flat[row:row + 1, col:col + D]
    collapsed = pl.create_tensor([T, D], dtype=pl.BF16)
    mhc_head(x_hc, collapsed)
    rmsnorm(collapsed, final_norm, hidden_step)
    return streams_step, hidden_step, conv_state, recurrent_state


@pl.jit.host
def l3_decode_fwd(
    inputs_embeds: pl.Tensor[[EP_SIZE, STEPS, T, D], pl.BF16],
    state_rows: pl.Tensor[[EP_SIZE, STEPS, T], pl.INT32],
    hc_function: pl.Tensor[[EP_SIZE, N_LAYERS, 2, MIX_HC, HC_DIM], pl.BF16],
    hc_scale: pl.Tensor[[EP_SIZE, N_LAYERS, 2, 3], pl.FP32],
    hc_base: pl.Tensor[[EP_SIZE, N_LAYERS, 2, MIX_HC], pl.FP32],
    norm_weight: pl.Tensor[[EP_SIZE, N_LAYERS, 2, D], pl.BF16],
    w_q: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_k: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_v: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_QKV_DIM, D], pl.BF16],
    w_f_a: pl.Tensor[[EP_SIZE, N_KDA, KDA_DIM, D], pl.BF16],
    w_f_b: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    dt_bias: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_QKV_DIM], pl.FP32],
    a_log: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_H], pl.FP32],
    w_b: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_H, D], pl.BF16],
    w_g_a: pl.Tensor[[EP_SIZE, N_KDA, KDA_DIM, D], pl.BF16],
    w_g_b: pl.Tensor[[EP_SIZE, N_KDA, LOCAL_KDA_QKV_DIM, KDA_DIM], pl.BF16],
    conv_weight: pl.Tensor[[EP_SIZE, N_KDA, KDA_CONV_K, CONV_DIM], pl.BF16],
    out_norm: pl.Tensor[[EP_SIZE, N_KDA, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[EP_SIZE, N_KDA, D, LOCAL_KDA_QKV_DIM], pl.BF16],
    dense_gate_up: pl.Tensor[[EP_SIZE, N_DENSE, 2 * LOCAL_DENSE_INTER, D], DENSE_PL_DTYPE],
    dense_gate_up_scale: pl.Tensor[[EP_SIZE, N_DENSE, 2 * LOCAL_DENSE_INTER], pl.FP32],
    dense_down: pl.Tensor[[EP_SIZE, N_DENSE, D, LOCAL_DENSE_INTER], DENSE_PL_DTYPE],
    dense_down_scale: pl.Tensor[[EP_SIZE, N_DENSE, D], pl.FP32],
    final_norm: pl.Tensor[[EP_SIZE, D], pl.BF16],
    mla_w_q_a: pl.Tensor[[EP_SIZE, N_DSA, Q_LORA, D], pl.BF16],
    mla_q_norm: pl.Tensor[[EP_SIZE, N_DSA, Q_LORA], pl.BF16],
    mla_w_q_b: pl.Tensor[[EP_SIZE, N_DSA, LOCAL_H * QK_DIM, Q_LORA], pl.BF16],
    mla_w_kv_a: pl.Tensor[[EP_SIZE, N_DSA, KV_LORA, D], pl.BF16],
    mla_kv_norm: pl.Tensor[[EP_SIZE, N_DSA, KV_LORA], pl.BF16],
    mla_w_k: pl.Tensor[[EP_SIZE, N_DSA, LOCAL_H, QK_DIM, KV_LORA], pl.BF16],
    mla_w_o: pl.Tensor[[EP_SIZE, N_DSA, D, LOCAL_H * KV_LORA], pl.BF16],
    index_w_q_b: pl.Tensor[[EP_SIZE, N_DSA, Q_LORA, INDEX_H * INDEX_DIM], pl.BF16],
    index_w_k: pl.Tensor[[EP_SIZE, N_DSA, D, INDEX_DIM], pl.BF16],
    index_norm: pl.Tensor[[EP_SIZE, N_DSA, INDEX_DIM], pl.BF16],
    index_bias: pl.Tensor[[EP_SIZE, N_DSA, INDEX_DIM], pl.BF16],
    index_weights: pl.Tensor[[EP_SIZE, N_DSA, D, INDEX_H], pl.BF16],
    index_gate: pl.Tensor[[EP_SIZE, N_DSA, D, INDEX_DIM], pl.BF16],
    compress_ape: pl.Tensor[[EP_SIZE, N_DSA, INDEX_KPOOL, INDEX_DIM], pl.BF16],
    hadamard: pl.Tensor[[EP_SIZE, INDEX_DIM, INDEX_DIM], pl.BF16],
    gate_weight: pl.Tensor[[EP_SIZE, N_MOE, N_EXPERTS, D], pl.BF16],
    correction_bias: pl.Tensor[[EP_SIZE, N_MOE, N_EXPERTS], pl.FP32],
    expert_gate_up: pl.Tensor[[EP_SIZE, N_MOE, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], pl.INT8],
    expert_gate_up_scale: pl.Tensor[[EP_SIZE, N_MOE, N_LOCAL_EXPERTS, 2 * MOE_INTER], pl.FP32],
    expert_down: pl.Tensor[[EP_SIZE, N_MOE, N_LOCAL_EXPERTS, D, MOE_INTER], pl.INT8],
    expert_down_scale: pl.Tensor[[EP_SIZE, N_MOE, N_LOCAL_EXPERTS, D], pl.FP32],
    shared_gate_up: pl.Tensor[[EP_SIZE, N_MOE, 2 * MOE_INTER, D], pl.INT8],
    shared_gate_up_scale: pl.Tensor[[EP_SIZE, N_MOE, 2 * MOE_INTER], pl.FP32],
    shared_down: pl.Tensor[[EP_SIZE, N_MOE, D, MOE_INTER], pl.INT8],
    shared_down_scale: pl.Tensor[[EP_SIZE, N_MOE, D], pl.FP32],
    kv_slots: pl.Tensor[[EP_SIZE, STEPS, T], pl.INT32],
    pool_token_slots: pl.Tensor[[EP_SIZE, STEPS, T, INDEX_KPOOL], pl.INT32],
    pool_slots: pl.Tensor[[EP_SIZE, STEPS, T], pl.INT32],
    pool_blocks: pl.Tensor[[EP_SIZE, POOL_ROWS], pl.INT32],
    seg_start: pl.Tensor[[EP_SIZE, T], pl.INT32],
    pool_count: pl.Tensor[[EP_SIZE, STEPS, T], pl.INT32],
    tail_start: pl.Tensor[[EP_SIZE, STEPS, T], pl.INT32],
    tail_count: pl.Tensor[[EP_SIZE, STEPS, T], pl.INT32],
    kv_len: pl.Tensor[[EP_SIZE, STEPS, T], pl.INT32],
    block_table: pl.Tensor[[EP_SIZE, T, 1], pl.INT32],
    request_ids: pl.Tensor[[EP_SIZE, T], pl.INT32],
    latent_cache: pl.InOut[pl.Tensor[[EP_SIZE, N_DSA, RAW_ROWS, KV_LORA], pl.BF16]],
    raw_cache: pl.InOut[pl.Tensor[[EP_SIZE, N_DSA, RAW_ROWS, INDEX_STATE_WIDTH], pl.FP32]],
    pool_cache: pl.InOut[pl.Tensor[[EP_SIZE, N_DSA, POOL_ROWS, INDEX_DIM], pl.INT8]],
    pool_scale: pl.InOut[pl.Tensor[[EP_SIZE, N_DSA, POOL_ROWS, POOL_SCALE_WIDTH], pl.FP32]],
    conv_state: pl.InOut[pl.Tensor[[EP_SIZE, N_KDA, STATE_ROWS, STATE_LEN, CONV_DIM], pl.BF16]],
    recurrent_state: pl.InOut[pl.Tensor[[EP_SIZE, N_KDA, STATE_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], pl.FP32]],
    layer_outputs: pl.InOut[pl.Tensor[[EP_SIZE, STEPS, N_LAYERS, TRACE_PHASES, T, D], pl.BF16]],
    streams: pl.Out[pl.Tensor[[EP_SIZE, STEPS, T, HC_MULT, D], pl.BF16]],
    hidden: pl.Out[pl.Tensor[[EP_SIZE, STEPS, T, D], pl.BF16]],
):
    lane_x_buf = pld.alloc_window_buffer([LANE_ROWS, D], dtype=pl.INT8)
    lane_aux_buf = pld.alloc_window_buffer([LANE_ROWS, AUX_PAD], dtype=pl.FP32)
    lane_count_buf = pld.alloc_window_buffer([EP_SIZE, COUNT_PAD], dtype=pl.INT32)
    lane_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    slot_out_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.FP32)
    return_ready_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    exchange_buf = pld.alloc_window_buffer([T, D], dtype=pl.FP32)
    arrived_buf = pld.alloc_window_buffer([EP_SIZE, 1], dtype=pl.INT32)
    for step in pl.range(STEPS):
        for rank in pl.range(pld.world_size()):
            lane_x = pld.window(lane_x_buf, [LANE_ROWS, D], dtype=pl.INT8)
            lane_aux = pld.window(lane_aux_buf, [LANE_ROWS, AUX_PAD], dtype=pl.FP32)
            lane_count = pld.window(lane_count_buf, [EP_SIZE, COUNT_PAD], dtype=pl.INT32)
            lane_ready = pld.window(lane_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
            slot_out = pld.window(slot_out_buf, [N_ROUTES, D], dtype=pl.FP32)
            return_ready = pld.window(return_ready_buf, [EP_SIZE, 1], dtype=pl.INT32)
            exchange = pld.window(exchange_buf, [T, D], dtype=pl.FP32)
            arrived = pld.window(arrived_buf, [EP_SIZE, 1], dtype=pl.INT32)
            decode_fwd(
                inputs_embeds[rank], state_rows[rank], hc_function[rank],
                hc_scale[rank], hc_base[rank], norm_weight[rank],
                w_q[rank], w_k[rank], w_v[rank],
                w_f_a[rank], w_f_b[rank], dt_bias[rank],
                a_log[rank], w_b[rank], w_g_a[rank],
                w_g_b[rank], conv_weight[rank], out_norm[rank],
                w_o[rank], dense_gate_up[rank], dense_gate_up_scale[rank],
                dense_down[rank], dense_down_scale[rank], final_norm[rank],
                mla_w_q_a[rank], mla_q_norm[rank], mla_w_q_b[rank],
                mla_w_kv_a[rank], mla_kv_norm[rank], mla_w_k[rank],
                mla_w_o[rank], index_w_q_b[rank], index_w_k[rank],
                index_norm[rank], index_bias[rank], index_weights[rank],
                index_gate[rank], compress_ape[rank], hadamard[rank],
                gate_weight[rank], correction_bias[rank], expert_gate_up[rank],
                expert_gate_up_scale[rank], expert_down[rank], expert_down_scale[rank],
                shared_gate_up[rank], shared_gate_up_scale[rank], shared_down[rank],
                shared_down_scale[rank], kv_slots[rank], pool_token_slots[rank],
                pool_slots[rank], pool_blocks[rank], seg_start[rank],
                pool_count[rank], tail_start[rank], tail_count[rank],
                kv_len[rank], block_table[rank], request_ids[rank],
                latent_cache[rank], raw_cache[rank], pool_cache[rank],
                pool_scale[rank],
                conv_state[rank], recurrent_state[rank], layer_outputs[rank], streams[rank],
                hidden[rank],
                lane_x, lane_aux, lane_count, lane_ready, slot_out, return_ready,
                exchange, arrived, rank, pl.cast(step, pl.INT32), device=rank,
            )


_WEIGHT_LAYOUTS = [
    ("hc_function", [N_LAYERS, 2, MIX_HC, HC_DIM], torch.bfloat16),
    ("hc_scale", [N_LAYERS, 2, 3], torch.float32),
    ("hc_base", [N_LAYERS, 2, MIX_HC], torch.float32),
    ("norm_weight", [N_LAYERS, 2, D], torch.bfloat16),
    ("w_q", [N_KDA, LOCAL_KDA_QKV_DIM, D], torch.bfloat16),
    ("w_k", [N_KDA, LOCAL_KDA_QKV_DIM, D], torch.bfloat16),
    ("w_v", [N_KDA, LOCAL_KDA_QKV_DIM, D], torch.bfloat16),
    ("w_f_a", [N_KDA, KDA_DIM, D], torch.bfloat16),
    ("w_f_b", [N_KDA, LOCAL_KDA_QKV_DIM, KDA_DIM], torch.bfloat16),
    ("dt_bias", [N_KDA, LOCAL_KDA_QKV_DIM], torch.float32),
    ("a_log", [N_KDA, LOCAL_KDA_H], torch.float32),
    ("w_b", [N_KDA, LOCAL_KDA_H, D], torch.bfloat16),
    ("w_g_a", [N_KDA, KDA_DIM, D], torch.bfloat16),
    ("w_g_b", [N_KDA, LOCAL_KDA_QKV_DIM, KDA_DIM], torch.bfloat16),
    ("conv_weight", [N_KDA, KDA_CONV_K, CONV_DIM], torch.bfloat16),
    ("out_norm", [N_KDA, KDA_DIM], torch.bfloat16),
    ("w_o", [N_KDA, D, LOCAL_KDA_QKV_DIM], torch.bfloat16),
    ("dense_gate_up", [N_DENSE, 2 * LOCAL_DENSE_INTER, D], DENSE_TORCH_DTYPE),
    ("dense_gate_up_scale", [N_DENSE, 2 * LOCAL_DENSE_INTER], torch.float32),
    ("dense_down", [N_DENSE, D, LOCAL_DENSE_INTER], DENSE_TORCH_DTYPE),
    ("dense_down_scale", [N_DENSE, D], torch.float32),
    ("final_norm", [D], torch.bfloat16),
    ("mla_w_q_a", [N_DSA, Q_LORA, D], torch.bfloat16),
    ("mla_q_norm", [N_DSA, Q_LORA], torch.bfloat16),
    ("mla_w_q_b", [N_DSA, LOCAL_H * QK_DIM, Q_LORA], torch.bfloat16),
    ("mla_w_kv_a", [N_DSA, KV_LORA, D], torch.bfloat16),
    ("mla_kv_norm", [N_DSA, KV_LORA], torch.bfloat16),
    ("mla_w_k", [N_DSA, LOCAL_H, QK_DIM, KV_LORA], torch.bfloat16),
    ("mla_w_o", [N_DSA, D, LOCAL_H * KV_LORA], torch.bfloat16),
    ("index_w_q_b", [N_DSA, Q_LORA, INDEX_H * INDEX_DIM], torch.bfloat16),
    ("index_w_k", [N_DSA, D, INDEX_DIM], torch.bfloat16),
    ("index_norm", [N_DSA, INDEX_DIM], torch.bfloat16),
    ("index_bias", [N_DSA, INDEX_DIM], torch.bfloat16),
    ("index_weights", [N_DSA, D, INDEX_H], torch.bfloat16),
    ("index_gate", [N_DSA, D, INDEX_DIM], torch.bfloat16),
    ("compress_ape", [N_DSA, INDEX_KPOOL, INDEX_DIM], torch.bfloat16),
    ("hadamard", [INDEX_DIM, INDEX_DIM], torch.bfloat16),
    ("gate_weight", [N_MOE, N_EXPERTS, D], torch.bfloat16),
    ("correction_bias", [N_MOE, N_EXPERTS], torch.float32),
    ("expert_gate_up", [N_MOE, N_LOCAL_EXPERTS, 2 * MOE_INTER, D], torch.int8),
    ("expert_gate_up_scale", [N_MOE, N_LOCAL_EXPERTS, 2 * MOE_INTER], torch.float32),
    ("expert_down", [N_MOE, N_LOCAL_EXPERTS, D, MOE_INTER], torch.int8),
    ("expert_down_scale", [N_MOE, N_LOCAL_EXPERTS, D], torch.float32),
    ("shared_gate_up", [N_MOE, 2 * MOE_INTER, D], torch.int8),
    ("shared_gate_up_scale", [N_MOE, 2 * MOE_INTER], torch.float32),
    ("shared_down", [N_MOE, D, MOE_INTER], torch.int8),
    ("shared_down_scale", [N_MOE, D], torch.float32),
]


_METADATA_LAYOUTS = [
    ("kv_slots", [STEPS, T], torch.int32),
    ("pool_token_slots", [STEPS, T, INDEX_KPOOL], torch.int32),
    ("pool_slots", [STEPS, T], torch.int32),
    ("pool_blocks", [POOL_ROWS], torch.int32),
    ("seg_start", [T], torch.int32),
    ("pool_count", [STEPS, T], torch.int32),
    ("tail_start", [STEPS, T], torch.int32),
    ("tail_count", [STEPS, T], torch.int32),
    ("kv_len", [STEPS, T], torch.int32),
    ("block_table", [T, 1], torch.int32),
    ("request_ids", [T], torch.int32),
]
_CACHE_LAYOUTS = [
    ("latent_cache", [N_DSA, RAW_ROWS, KV_LORA], torch.bfloat16),
    ("raw_cache", [N_DSA, RAW_ROWS, INDEX_STATE_WIDTH], torch.float32),
    ("pool_cache", [N_DSA, POOL_ROWS, INDEX_DIM], torch.int8),
    ("pool_scale", [N_DSA, POOL_ROWS, POOL_SCALE_WIDTH], torch.float32),
]


def golden_sparse_step(tensors, normed, rank, layer, step):
    from models.glm5_3_flash.mla_prolog import golden_mla_prolog, golden_absorb_query
    from models.glm5_3_flash.mla_cache import golden_mla_cache_write
    from models.glm5_3_flash.mla_epilog import golden_mla_epilog_decode
    from models.glm5_3_flash.decode_sparse_attn import golden_decode_sparse_attn
    from models.glm5_3_flash.indexer import (
        golden_indexer_proj, golden_indexer_score, golden_indexer_topk, golden_indexer_expand,
    )
    from models.glm5_3_flash.indexer_cache import golden_indexer_cache_write, golden_indexer_pool_write

    weights = [tensors[n][rank, layer] for n in (
        "mla_w_q_a", "mla_q_norm", "mla_w_q_b", "mla_w_kv_a", "mla_kv_norm",
    )]
    q_resid, query, latent = golden_mla_prolog(normed, *weights)
    slots = tensors["kv_slots"][rank, step]
    cache = tensors["latent_cache"][rank, layer]
    cache.copy_(golden_mla_cache_write(cache, latent, slots))
    weights = [tensors[n][rank, layer] for n in (
        "index_w_q_b", "index_w_k", "index_norm", "index_bias", "index_weights", "index_gate",
    )]
    iq, ik, hw, gs = golden_indexer_proj(normed, q_resid, *weights)
    raw = tensors["raw_cache"][rank, layer]
    raw.copy_(golden_indexer_cache_write(raw, ik, gs, slots))
    pc = tensors["pool_cache"][rank, layer]
    ps = tensors["pool_scale"][rank, layer]
    updated, scales, _ = golden_indexer_pool_write(
        pc, ps, raw, tensors["compress_ape"][rank, layer], tensors["hadamard"][rank],
        tensors["pool_token_slots"][rank, step], tensors["pool_slots"][rank, step],
    )
    pc.copy_(updated)
    ps.copy_(scales)
    scores = golden_indexer_score(iq, tensors["hadamard"][rank], pc, ps, tensors["pool_blocks"][rank],
                                  hw, tensors["seg_start"][rank], tensors["pool_count"][rank, step])
    selected, valid = golden_indexer_topk(scores, tensors["seg_start"][rank], tensors["pool_count"][rank, step])
    indices = golden_indexer_expand(selected, valid, tensors["tail_start"][rank, step],
                                    tensors["tail_count"][rank, step], tensors["kv_len"][rank, step])
    absorbed = golden_absorb_query(query, tensors["mla_w_k"][rank, layer])
    context = golden_decode_sparse_attn(absorbed, cache, indices,
                                       tensors["block_table"][rank], tensors["request_ids"][rank])
    return golden_mla_epilog_decode(context, tensors["mla_w_o"][rank, layer])


def golden_decode_fwd(tensors, *, observed=None):
    """Compose child goldens, optionally checking boundaries with captured inputs.

    The default is an independent end-to-end reference. Boundary validation uses
    the recorded attention/MLP inputs for the child goldens and the recorded
    sublayer outputs for the surrounding mHC edges. Each recorded input is also
    checked against the preceding edge, so the reference covers the composition.
    """
    from models.glm5_3_flash.dense_mlp import golden_dense_mlp_fn, golden_dense_mlp_bf16

    for step in range(STEPS):
        x = tensors["inputs_embeds"][0, step, :, None, :].expand(T, HC_MULT, D).clone()
        for layer in range(START_LAYER, END_LAYER):
            pre, post, residual = golden_mhc_mixes(
                x, tensors["hc_function"][0, layer, 0], tensors["hc_scale"][0, layer, 0],
                tensors["hc_base"][0, layer, 0],
            )
            mixed = golden_mhc_pre(x, pre)
            normed = golden_rmsnorm(mixed, tensors["norm_weight"][0, layer, 0])
            tensors["layer_outputs"][:, step, layer, 3].copy_(normed)
            if observed is not None:
                normed = observed["layer_outputs"][0, step, layer, 3]
            partials = []
            if layer % 4 != 3:
                linear_layer = layer - layer // 4
                for rank in range(EP_SIZE):
                    weights = [tensors[n][rank, linear_layer] for n in (
                        "w_q", "w_k", "w_v", "w_f_a", "w_f_b", "dt_bias", "a_log", "w_b", "w_g_a", "w_g_b",
                    )]
                    qkv, decay, beta, out_gate = golden_kda_projection(normed, *weights)
                    rows = tensors["state_rows"][rank, step]
                    q, k, v, conv = golden_kda_conv_decode(
                        qkv, tensors["conv_weight"][rank, linear_layer], tensors["conv_state"][rank, linear_layer], rows,
                    )
                    tensors["conv_state"][rank, linear_layer].copy_(conv)
                    core, state = golden_decode_kda(
                        q, k, v, decay, beta, tensors["recurrent_state"][rank, linear_layer], rows,
                    )
                    tensors["recurrent_state"][rank, linear_layer].copy_(state)
                    partials.append(golden_kda_output(
                        core, tensors["out_norm"][rank, linear_layer], out_gate, tensors["w_o"][rank, linear_layer],
                    ))
            else:
                for rank in range(EP_SIZE):
                    partials.append(golden_sparse_step(tensors, normed, rank, layer // 4, step))
            reduced = torch.stack(partials).sum(0).to(torch.bfloat16)
            tensors["layer_outputs"][:, step, layer, 0].copy_(reduced)
            if observed is not None:
                reduced = observed["layer_outputs"][0, step, layer, 0]
            x = golden_mhc_post(reduced, x, post, residual)
            pre, post, residual = golden_mhc_mixes(
                x, tensors["hc_function"][0, layer, 1], tensors["hc_scale"][0, layer, 1],
                tensors["hc_base"][0, layer, 1],
            )
            mixed = golden_mhc_pre(x, pre)
            tensors["layer_outputs"][:, step, layer, 2].copy_(mixed)
            if observed is not None:
                mixed = observed["layer_outputs"][0, step, layer, 2]
            if layer < 3:
                if DENSE_BF16:
                    dense_input = golden_rmsnorm(mixed, tensors["norm_weight"][0, layer, 1])
                    partials = [golden_dense_mlp_bf16(
                        dense_input, tensors["dense_gate_up"][rank, layer], tensors["dense_down"][rank, layer]
                    ).float() for rank in range(EP_SIZE)]
                    reduced = torch.stack(partials).sum(0).to(torch.bfloat16)
                else:
                    _, x_int8, x_scale = golden_rmsnorm_quant(mixed, tensors["norm_weight"][0, layer, 1])
                    partials = []
                    for rank in range(EP_SIZE):
                        case = {"x_int8": x_int8, "x_scale": x_scale, "output": torch.empty(T, D)}
                        for dst, src in (("w_gate_up", "dense_gate_up"), ("w_gate_up_scale", "dense_gate_up_scale"),
                                         ("w_down", "dense_down"), ("w_down_scale", "dense_down_scale")):
                            case[dst] = tensors[src][rank, layer]
                        golden_dense_mlp_fn(case)
                        partials.append(case["output"])
                    reduced = torch.stack(partials).sum(0).to(torch.bfloat16)
            else:
                from models.glm5_3_flash.decode_layer import golden_decode_layer

                m = layer - 3
                case = {"x_mixed": mixed.unsqueeze(0).expand(EP_SIZE, -1, -1),
                        "norm_weight": tensors["norm_weight"][:, layer, 1], "y": torch.empty(EP_SIZE, T, D)}
                for dest, source in (
                    ("gate_weight", "gate_weight"), ("correction_bias", "correction_bias"),
                    ("w_gate_up", "expert_gate_up"), ("w_gate_up_scale", "expert_gate_up_scale"),
                    ("w_down", "expert_down"), ("w_down_scale", "expert_down_scale"),
                    ("shared_w_gate_up", "shared_gate_up"), ("shared_w_gate_up_scale", "shared_gate_up_scale"),
                    ("shared_w_down", "shared_down"), ("shared_w_down_scale", "shared_down_scale"),
                ):
                    case[dest] = tensors[source][:, m]
                golden_decode_layer(case)
                reduced = case["y"][0].to(torch.bfloat16)
            tensors["layer_outputs"][:, step, layer, 1].copy_(reduced)
            if observed is not None:
                reduced = observed["layer_outputs"][0, step, layer, 1]
            x = golden_mhc_post(reduced, x, post, residual)
        head_input = x if observed is None else observed["streams"][0, step]
        hidden = golden_rmsnorm(golden_mhc_head(head_input), tensors["final_norm"][0])
        tensors["streams"][:, step].copy_(x)
        tensors["hidden"][:, step].copy_(hidden)


def build_forward_metadata():
    request = torch.arange(T, dtype=torch.int32)
    pages = T - 1 - request
    prior = 3 + request % 4
    length = prior[None] + torch.arange(1, STEPS + 1, dtype=torch.int32)[:, None]
    positions = length - 1
    meta = {"kv_slots": pages[None] * BLOCK_SIZE + positions,
            "pool_blocks": torch.cat([torch.arange(0, POOL_ROWS, 4, dtype=torch.int32),
                                      torch.zeros(POOL_ROWS * 3 // 4, dtype=torch.int32)]),
            "seg_start": request * 64, "pool_count": length // 4,
            "tail_start": length // 4 * 4, "tail_count": length % 4, "kv_len": length,
            "block_table": pages[:, None], "request_ids": request}
    closes = length % 4 == 0
    members = pages[None, :, None] * BLOCK_SIZE + length[:, :, None] - 4 + torch.arange(4)[None, None]
    meta["pool_token_slots"] = torch.where(closes[:, :, None], members, -1).to(torch.int32)
    meta["pool_slots"] = torch.where(closes, request[None] * 64 + length // 4 - 1,
                                     request[None] * 64 + 63).to(torch.int32)
    return meta


def build_tensor_specs():
    from golden import TensorSpec

    specs = []
    generator = torch.Generator().manual_seed(533)
    embeds = torch.randn(STEPS, T, D, generator=generator).to(torch.bfloat16)
    mappings = torch.stack([(torch.arange(T) + 2 * step + 1) % STATE_ROWS for step in range(STEPS)])
    specs.append(TensorSpec("inputs_embeds", [EP_SIZE, STEPS, T, D], torch.bfloat16,
                            init_value=lambda: embeds.unsqueeze(0).expand(EP_SIZE, -1, -1, -1).clone(),
                            resident="stacked"))
    specs.append(TensorSpec("state_rows", [EP_SIZE, STEPS, T], torch.int32,
                            init_value=lambda: mappings.unsqueeze(0).expand(EP_SIZE, -1, -1).clone(),
                            resident="stacked"))
    replicas = {"hc_function", "hc_scale", "hc_base", "norm_weight", "final_norm", "w_f_a", "w_g_a",
                "mla_w_q_a", "mla_q_norm", "mla_w_kv_a", "mla_kv_norm", "index_w_q_b", "index_w_k",
                "index_norm", "index_bias", "index_weights", "index_gate", "compress_ape", "hadamard",
                "gate_weight", "correction_bias", "shared_gate_up", "shared_gate_up_scale",
                "shared_down", "shared_down_scale"}
    for index, (name, shape, dtype) in enumerate(_WEIGHT_LAYOUTS):
        def init(name=name, shape=shape, dtype=dtype, seed=1000 + index):
            gen = torch.Generator().manual_seed(seed)
            rank_count = 1 if name in replicas else EP_SIZE
            full_shape = [rank_count, *shape]
            if name == "hadamard":
                from models.glm5_3_flash.indexer_cache import sylvester_hadamard
                value = sylvester_hadamard().to(dtype).unsqueeze(0)
            elif name == "correction_bias":
                from models.glm5_3_flash.gate import make_correction_bias
                value = make_correction_bias(EP_SIZE, N_EXPERTS, 8)[:1, None].expand(1, N_MOE, N_EXPERTS)
            elif dtype == torch.int8:
                value = torch.randint(-16, 17, full_shape, dtype=dtype, generator=gen)
            elif "norm" in name:
                value = (1 + 0.03 * torch.randn(full_shape, generator=gen)).to(dtype)
            elif name.endswith("_scale") and name != "hc_scale":
                value = torch.full(full_shape, 0.002, dtype=dtype)
            elif name == "hc_scale":
                value = torch.full(full_shape, 0.1, dtype=dtype)
            else:
                std = 0.15 if name in ("conv_weight", "hc_base", "dt_bias", "a_log") else shape[-1] ** -0.5
                value = (torch.randn(full_shape, generator=gen) * std).to(dtype)
            return value.expand(EP_SIZE, *shape).contiguous()
        specs.append(TensorSpec(name, [EP_SIZE, *shape], dtype, init_value=init, resident="stacked"))
    meta = build_forward_metadata()
    for name, shape, dtype in _METADATA_LAYOUTS:
        def init_metadata(name=name, shape=shape):
            return meta[name].unsqueeze(0).expand(EP_SIZE, *shape).contiguous()
        specs.append(TensorSpec(name, [EP_SIZE, *shape], dtype, init_value=init_metadata, resident="stacked"))
    for index, (name, shape, dtype) in enumerate(_CACHE_LAYOUTS):
        def init_cache(name=name, shape=shape, dtype=dtype, seed=3000 + index):
            gen = torch.Generator().manual_seed(seed)
            if dtype == torch.int8:
                value = torch.randint(-16, 17, shape, generator=gen, dtype=dtype)
            elif name == "pool_scale":
                value = (0.01 + torch.rand(shape, generator=gen) * 0.02).to(dtype)
            else:
                value = (torch.randn(shape, generator=gen) * 0.02).to(dtype)
            return value.unsqueeze(0).expand(EP_SIZE, *shape).contiguous()
        specs.append(TensorSpec(name, [EP_SIZE, *shape], dtype, init_value=init_cache, resident="stacked"))
    for name, shape, dtype in (
        ("conv_state", [EP_SIZE, N_KDA, STATE_ROWS, STATE_LEN, CONV_DIM], torch.bfloat16),
        ("recurrent_state", [EP_SIZE, N_KDA, STATE_ROWS, LOCAL_KDA_H, KDA_DIM, KDA_DIM], torch.float32),
    ):
        def init_state(shape=shape, dtype=dtype):
            gen = torch.Generator().manual_seed(2001)
            return (torch.randn(shape, generator=gen) * 0.02).to(dtype)
        specs.append(TensorSpec(name, shape, dtype, init_value=init_state, resident="stacked"))
    specs.append(TensorSpec("layer_outputs", [EP_SIZE, STEPS, N_LAYERS, TRACE_PHASES, T, D], torch.bfloat16,
                            resident="stacked"))
    specs.append(TensorSpec("streams", [EP_SIZE, STEPS, T, HC_MULT, D], torch.bfloat16, resident="stacked"))
    specs.append(TensorSpec("hidden", [EP_SIZE, STEPS, T, D], torch.bfloat16, resident="stacked"))
    return specs



def compare_replicated(actual, expected, **kwargs):
    from golden import ratio_allclose

    if not torch.equal(actual, actual[:1].expand_as(actual)):
        return False, "TP replicas diverged"
    compare = ratio_allclose(max_error_ratio=0.05)
    if actual.ndim == 6:
        if not torch.equal(actual[:, :, :START_LAYER], expected[:, :, :START_LAYER]):
            return False, "an unexecuted layer trace changed"
        if not torch.equal(actual[:, :, END_LAYER:], expected[:, :, END_LAYER:]):
            return False, "an unexecuted layer trace changed"
        for step in range(STEPS):
            for layer in range(START_LAYER, END_LAYER):
                for phase in range(TRACE_PHASES):
                    ok, detail = compare(actual[:, step, layer, phase], expected[:, step, layer, phase], **kwargs)
                    if not ok:
                        return False, f"step={step}, layer={layer}, phase={phase}: {detail}"
        return True, ""
    for step in range(STEPS):
        ok, detail = compare(actual[:, step], expected[:, step], **kwargs)
        if not ok:
            return False, f"step={step}: {detail}"
    return True, ""


def compare_state(actual, expected, *, inputs, **kwargs):
    from golden import ratio_allclose

    for rank in range(EP_SIZE):
        used = inputs["state_rows"][rank].flatten().long().unique()
        untouched = torch.ones(STATE_ROWS, dtype=torch.bool)
        untouched[used] = False
        if not torch.equal(actual[rank, :, untouched], expected[rank, :, untouched]):
            return False, "an untouched state slot changed"
    return ratio_allclose(max_error_ratio=0)(actual, expected, inputs=inputs, **kwargs)


def compare_cache_slots(slot_name, *, atol=None, rtol=None, max_error_ratio=0):
    """Require exact preservation of cache rows outside this decode update."""
    from golden import ratio_allclose

    def compare(actual, expected, *, inputs, **kwargs):
        numeric = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_error_ratio)
        for rank in range(EP_SIZE):
            used = inputs[slot_name][rank].flatten().long().unique()
            untouched = torch.ones(actual.shape[2], dtype=torch.bool)
            untouched[used] = False
            if not torch.equal(actual[rank, :, untouched], expected[rank, :, untouched]):
                return False, "an untouched cache slot changed"
            for layer in range(actual.shape[1]):
                got = actual[rank, layer, used]
                want = expected[rank, layer, used]
                ok, detail = numeric(got, want, inputs=inputs, **kwargs)
                if not ok:
                    return False, f"rank={rank}, cache_layer={layer}: {detail}"
        return True, ""

    return compare


def build_comparators(*, validate_boundaries=False, initial_states=None):
    """Build the selected validation gate; cache one boundary replay per run."""
    comparators = {
        "latent_cache": compare_cache_slots("kv_slots"),
        "raw_cache": compare_cache_slots("kv_slots"),
        "pool_cache": compare_cache_slots("pool_slots"),
        "pool_scale": compare_cache_slots("pool_slots"),
        "conv_state": compare_state,
        "recurrent_state": compare_state,
        "layer_outputs": compare_replicated,
        "streams": compare_replicated,
        "hidden": compare_replicated,
    }
    if not validate_boundaries:
        return comparators

    if initial_states is None:
        raise ValueError("boundary validation requires snapshots of the initial mutable tensors")

    # Match indexer_cache.py's local INT8 rounding contract for written rows.
    comparators["pool_cache"] = compare_cache_slots("pool_slots", atol=1.0, rtol=0.0, max_error_ratio=0.005)
    references = {}

    def bind(name, compare):
        def boundary_compare(actual, expected, *, actual_outputs, inputs, **kwargs):
            if not references:
                replay = dict(inputs)
                for output_name, value in actual_outputs.items():
                    initial = initial_states.get(output_name)
                    replay[output_name] = torch.zeros_like(value) if initial is None else initial.clone()
                golden_decode_fwd(replay, observed=actual_outputs)
                references.update({key: replay[key] for key in actual_outputs})
            if name == "hidden":
                from golden import error_distribution

                independent_ok, _detail = compare_replicated(
                    actual, expected, actual_outputs=actual_outputs, inputs=inputs, **kwargs
                )
                status = "PASS" if independent_ok else "FAIL"
                print(f"[decode_fwd] independent end-to-end hidden gate: {status}", flush=True)
                error_distribution()(actual, expected, actual_outputs=actual_outputs, inputs=inputs, **kwargs)
            return compare(actual, references[name], actual_outputs=actual_outputs, inputs=inputs, **kwargs)

        boundary_compare.__name__ = "boundary_" + name
        return boundary_compare

    return {name: bind(name, compare) for name, compare in comparators.items()}


def main():
    from golden import run
    from pypto.ir import DistributedConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--platform", dest="backend_type", default="a2a3", choices=["a2a3"])
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", default="0,1")
    parser.add_argument("--ep", type=int, default=EP_SIZE)
    parser.add_argument("--tp", type=int, default=TP_SIZE)
    parser.add_argument("--layers", type=int, default=N_LAYERS)
    parser.add_argument("--start-layer", type=int, default=START_LAYER)
    parser.add_argument("--end-layer", type=int, default=END_LAYER, help="exclusive execution end (default: start + 1); keep fixture layers fixed")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--dense-dtype", choices=["int8", "bf16"], default=_shape_args.dense_dtype)
    parser.add_argument("--dump-passes", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--validate-boundaries", action="store_true",
                        help="check sublayers, mHC edges and state updates using recorded sublayer inputs")
    args = parser.parse_args()
    devices = [int(v) for v in args.device.split(",")]
    if len(devices) != EP_SIZE or len(set(devices)) != EP_SIZE:
        parser.error(f"need {EP_SIZE} distinct devices")
    torch.set_num_threads(8)
    initial_states = {}

    def golden_with_snapshots(tensors):
        if args.validate_boundaries:
            names = [name for name, _shape, _dtype in _CACHE_LAYOUTS]
            names += ["conv_state", "recurrent_state", "layer_outputs"]
            initial_states.update({name: tensors[name].clone() for name in names})
        golden_decode_fwd(tensors)

    validation = "boundary correctness" if args.validate_boundaries else "independent end-to-end reference"
    print(f"[decode_fwd] validation: {validation}", flush=True)
    result = run(
        fn=l3_decode_fwd,
        specs=build_tensor_specs(),
        golden_fn=golden_with_snapshots,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, dump_passes=args.dump_passes,
                    ring_heap=[0, 512 * 1024 * 1024, 0, 0], enable_scope_stats=True,
                    distributed_config=DistributedConfig(device_ids=devices, num_sub_workers=0)),
        compile_only=args.compile_only,
        compare_fn=build_comparators(validate_boundaries=args.validate_boundaries, initial_states=initial_states),
        rtol=0.02,
        atol=0.02,
    )
    print(result)
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
