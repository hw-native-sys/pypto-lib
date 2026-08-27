# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: EP2/TP2 full decode forward
# ci: no-sim    # CI marker: full multi-layer / multi-card forward — device-only, skip on *sim
"""DeepSeek-V4 D-Spark decode forward."""

import sys

import config


_TP_CHOICES = (1, 2, 4)
_EP_CHOICES = (2, 4, 8, 16)
_TP_DEFAULT = 2
_EP_DEFAULT = 2


def _parse_parallel_arg(name, default):
    flag = f"--{name}"
    prefix = f"{flag}="
    for index, arg in enumerate(sys.argv):
        if arg == flag and index + 1 < len(sys.argv):
            return int(sys.argv[index + 1])
        if arg.startswith(prefix):
            return int(arg.split("=", 1)[1])
    return default


# Parallel configuration for shape-specialized kernel imports.
TP_SIZE = _parse_parallel_arg("tp", _TP_DEFAULT)
EP_SIZE = _parse_parallel_arg("ep", _EP_DEFAULT)
FWD_WEIGHT_BANK_SIZE = _parse_parallel_arg("weight-bank-size", 1)
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")
if EP_SIZE not in _EP_CHOICES:
    raise ValueError(f"--ep must be one of {_EP_CHOICES} (got {EP_SIZE})")
if EP_SIZE % TP_SIZE != 0:
    raise ValueError(f"EP={EP_SIZE} must be divisible by TP={TP_SIZE}")
if FWD_WEIGHT_BANK_SIZE not in (1, 43):
    raise ValueError("--weight-bank-size must be 1 or 43")

FWD_CSA_WEIGHT_BANK_SIZE = 21 if FWD_WEIGHT_BANK_SIZE == 43 else 1
FWD_HCA_WEIGHT_BANK_SIZE = 20 if FWD_WEIGHT_BANK_SIZE == 43 else 1

config.TP = TP_SIZE
config.EP = EP_SIZE

import decode_csa as csa
import decode_hca as hca
import decode_swa as swa
import moe as moe_module
import pypto.language as pl
import pypto.language.distributed as pld
from decode_csa import decode_csa, decode_csa_tp1
from decode_hca import decode_hca, decode_hca_tp1
from decode_swa import decode_swa, decode_swa_tp1
from hc_head import hc_head
from lm_head import (
    GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS,
    SAMPLED_IDS_PAD,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP,
    greedy_sample,
    lm_head,
)
from lookup_embedding import VOCAB_DYN as EMBED_VOCAB_DYN
from lookup_embedding import lookup_embedding
from moe import clear_moe_signals, moe
from rmsnorm import rms_norm


# Dynamic shape variables.
T_DYN = swa.T_DYN
FWD_PACKED_RAW_BLOCKS_DYN = pl.dynamic("FWD_PACKED_RAW_BLOCKS_DYN")
FWD_HCA_STATE_BLOCKS_DYN = pl.dynamic("FWD_HCA_STATE_BLOCKS_DYN")
FWD_HCA_CMP_BLOCKS_DYN = pl.dynamic("FWD_HCA_CMP_BLOCKS_DYN")
FWD_CSA_MAIN_STATE_BLOCKS_DYN = pl.dynamic("FWD_CSA_MAIN_STATE_BLOCKS_DYN")
FWD_CSA_CMP_BLOCKS_DYN = pl.dynamic("FWD_CSA_CMP_BLOCKS_DYN")
FWD_CSA_INNER_STATE_BLOCKS_DYN = pl.dynamic("FWD_CSA_INNER_STATE_BLOCKS_DYN")
FWD_CSA_IDX_BLOCKS_DYN = pl.dynamic("FWD_CSA_IDX_BLOCKS_DYN")

# model config
MODEL_CONFIG = config.FLASH
DECODE_TOKENS = config.DECODE_TOKENS
MAIN_LAYER_COUNT = MODEL_CONFIG.num_hidden_layers
SWA_LAYER_COUNT = 2
CSA_LAYER_COUNT = 21
HCA_LAYER_COUNT = 20
MAX_PUBLIC_TENSOR_DIMS = 5
N_RANKS = moe_module.N_RANKS
MOE_TOKENS = moe_module.T
D = swa.D
HC_MULT = swa.HC_MULT
HC_DIM = swa.HC_DIM
LM_HEAD_COMM_EPOCH = 1
MIX_HC = swa.MIX_HC
Q_LORA = swa.Q_LORA
H = swa.H
HEAD_DIM = swa.HEAD_DIM
ROPE_HEAD_DIM = swa.ROPE_HEAD_DIM
WIN = swa.WIN
O_LORA = swa.O_LORA
O_GROUP_IN = swa.O_GROUP_IN
LOCAL_O_GROUPS = swa.LOCAL_O_GROUPS
LOCAL_O_WIDTH = swa.LOCAL_O_WIDTH
BLOCK_SIZE = swa.BLOCK_SIZE
ATTENTION_WINDOW_ROWS = swa.ATTENTION_WINDOW_ROWS
O_WINDOW_ROWS = swa.O_WINDOW_ROWS
N_EXPERTS_GLOBAL = moe_module.N_EXPERTS_GLOBAL
N_LOCAL = moe_module.N_LOCAL
MOE_INTER = moe_module.MOE_INTER
VOCAB = moe_module.VOCAB
TOPK = moe_module.TOPK
RECV_MAX = moe_module.RECV_MAX
AUX_PAD = moe_module.AUX_PAD
IDX_PAD = moe_module.IDX_PAD
N_ROUTES = moe_module.N_ROUTES

HCA_B_DYN = hca.B_DYN
HCA_CMP_TABLE_BLOCKS_DYN = hca.CMP_TABLE_BLOCKS_DYN
HCA_B = hca.B
HCA_MAIN_OUT_DIM = hca.MAIN_OUT_DIM
HCA_COMPRESS_RATIO = hca.COMPRESS_RATIO
HCA_COMPRESS_STATE_BLOCK_SIZE = hca.COMPRESS_STATE_BLOCK_SIZE
HCA_COMPRESS_STATE_MAX_BLOCKS = hca.COMPRESS_STATE_MAX_BLOCKS
HCA_COMPRESS_STATE_DIM = hca.COMPRESS_STATE_DIM

CSA_B_DYN = csa.B_DYN
CSA_MAIN_OUT_DIM = csa.MAIN_OUT_DIM
CSA_COMPRESS_RATIO = csa.COMPRESS_RATIO
CSA_MAIN_STATE_BLOCK_SIZE = csa.MAIN_STATE_BLOCK_SIZE
CSA_MAIN_STATE_MAX_BLOCKS = csa.MAIN_STATE_MAX_BLOCKS
CSA_MAIN_STATE_DIM = csa.MAIN_STATE_DIM
CSA_IDX_N_HEADS = csa.IDX_N_HEADS
CSA_IDX_HEAD_DIM = csa.IDX_HEAD_DIM
CSA_INNER_OUT_DIM = csa.INNER_OUT_DIM
CSA_INNER_STATE_BLOCK_SIZE = csa.INNER_STATE_BLOCK_SIZE
CSA_INNER_STATE_MAX_BLOCKS = csa.INNER_STATE_MAX_BLOCKS
CSA_INNER_STATE_DIM = csa.INNER_STATE_DIM
CSA_CMP_MAX_BLOCKS = csa.CMP_MAX_BLOCKS
CSA_IDX_MAX_BLOCKS = csa.IDX_MAX_BLOCKS

RUNTIME_TEST_VOCAB = 256
RUNTIME_WEIGHT_BANK = 1
HC_FN_STORAGE_ROWS = 32
DECODE_RING_TASK_WINDOW = 16384
DECODE_RING_HEAP = 1 << 30
DECODE_RING_DEP_POOL = 16384


def _validate_import_contract():
    for module in (swa, hca, csa):
        if module.TP_SIZE != TP_SIZE:
            raise ValueError(f"{module.__name__} TP={module.TP_SIZE} does not match forward TP={TP_SIZE}")
    if moe_module.EP != EP_SIZE:
        raise ValueError(f"MoE EP={moe_module.EP} does not match forward EP={EP_SIZE}")
    if N_RANKS != EP_SIZE:
        raise ValueError(f"MoE world size {N_RANKS} does not match EP={EP_SIZE}")
    if MAIN_LAYER_COUNT != 43:
        raise ValueError(f"D-Spark decode forward expects 43 layers, got {MAIN_LAYER_COUNT}")
    if MODEL_CONFIG.vocab_size % TP_SIZE:
        raise ValueError(f"vocab size {MODEL_CONFIG.vocab_size} must be divisible by TP={TP_SIZE}")
    if LM_HEAD_TP_SIZE != TP_SIZE:
        raise ValueError(f"LM-head TP={LM_HEAD_TP_SIZE} does not match forward TP={TP_SIZE}")
    if LM_HEAD_VOCAB != MODEL_CONFIG.vocab_size:
        raise ValueError(f"LM-head vocab={LM_HEAD_VOCAB} does not match model vocab={MODEL_CONFIG.vocab_size}")
    if MAX_LOGIT_ROWS != DECODE_TOKENS:
        raise ValueError(f"LM-head rows={MAX_LOGIT_ROWS} do not match decode capacity={DECODE_TOKENS}")
    if MOE_TOKENS > MAX_LOGIT_ROWS:
        raise ValueError(f"MoE capacity {MOE_TOKENS} exceeds LM-head rows {MAX_LOGIT_ROWS}")


_validate_import_contract()


PACKED_POOL_LAYER_COUNTS = {
    "raw_kv_pool": MAIN_LAYER_COUNT,
    "hca_compress_state": HCA_LAYER_COUNT,
    "hca_cmp_kv": HCA_LAYER_COUNT,
    "csa_compress_state": CSA_LAYER_COUNT,
    "csa_cmp_kv": CSA_LAYER_COUNT,
    "csa_inner_compress_state": CSA_LAYER_COUNT,
    "csa_idx_kv_cache": CSA_LAYER_COUNT,
    "csa_idx_kv_scale": CSA_LAYER_COUNT,
}


def build_active_logit_row_indices_host(active_tokens):
    """Build the host fixture for the terminal active-prefix row contract."""
    import torch

    active_tokens = int(active_tokens)
    if active_tokens < 0 or active_tokens > min(MOE_TOKENS, MAX_LOGIT_ROWS):
        max_active_tokens = min(MOE_TOKENS, MAX_LOGIT_ROWS)
        raise ValueError(f"active token count must be in [0, {max_active_tokens}], got {active_tokens}")
    indices = torch.full((N_RANKS, MAX_LOGIT_ROWS), -1, dtype=torch.int32)
    if active_tokens:
        active_rows = torch.arange(active_tokens, dtype=torch.int32)
        indices[:, :active_tokens] = active_rows
    return indices


@pl.jit.inline
def decode_embedding_preamble(
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    embed_weight: pl.Tensor[[EMBED_VOCAB_DYN, D], pl.BF16],
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    moe_input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
):
    """Embed active rows and pad their token ids to the fixed MoE capacity."""
    active_tokens = pl.tensor.dim(input_ids, 0)
    for token_idx in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_pack_moe_input_ids"):
        token_id = pl.cast(0, pl.INT64)
        if token_idx < active_tokens:
            token_id = pl.read(input_ids, [token_idx])
        pl.write(moe_input_ids, [token_idx], token_id)
    lookup_embedding(input_ids, embed_weight, hidden_states, x_hc)
    return x_hc


@pl.jit.inline
def mask_inactive_sample_rows(
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampled_ids: pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    """Make inactive terminal rows observable as -1 after greedy sampling."""
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="decode_fwd_sample_mask"):
        if pl.read(logit_row_indices, [row]) < 0:
            sampled_ids[row:row + 1, :] = pl.full([1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=-1)
    return sampled_ids


@pl.jit(auto_scope=False)
def decode_fwd(
    embed_weight: pl.Tensor[[EMBED_VOCAB_DYN, D], pl.BF16],
    hc_attn_fn: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM], pl.FP32
    ],
    hc_attn_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * Q_LORA, H * HEAD_DIM], pl.INT8
    ],
    wq_b_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * H * HEAD_DIM], pl.FP32
    ],
    wkv: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16],
    raw_kv_pool: pl.InOut[
        pl.Tensor[
            [FWD_PACKED_RAW_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    csa_cmp_freqs_cos: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_freqs_sin: pl.Tensor[[T_DYN, ROPE_HEAD_DIM], pl.BF16],
    csa_cmp_wkv: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16
    ],
    csa_cmp_wgate: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16
    ],
    csa_cmp_ape: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    csa_cmp_norm_w: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16
    ],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [
                FWD_CSA_MAIN_STATE_BLOCKS_DYN,
                CSA_MAIN_STATE_BLOCK_SIZE,
                CSA_MAIN_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_compress_state_block_table: pl.Tensor[
        [CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_idx_wq_b: pl.Tensor[
        [
            FWD_CSA_WEIGHT_BANK_SIZE * Q_LORA,
            CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM,
        ],
        pl.INT8,
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        pl.FP32,
    ],
    csa_weights_proj: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * D, CSA_IDX_N_HEADS], pl.BF16
    ],
    csa_hadamard_idx: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
        pl.BF16,
    ],
    csa_inner_wkv: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D], pl.BF16
    ],
    csa_inner_wgate: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D], pl.BF16
    ],
    csa_inner_ape: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
        pl.FP32,
    ],
    csa_inner_norm_w: pl.Tensor[
        [FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM], pl.BF16
    ],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [
                FWD_CSA_INNER_STATE_BLOCKS_DYN,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[
        [CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[
            [FWD_CSA_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    csa_cmp_block_table: pl.Tensor[
        [CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32
    ],
    csa_idx_kv_cache: pl.InOut[
        pl.Tensor[
            [FWD_CSA_IDX_BLOCKS_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
            pl.INT8,
        ]
    ],
    csa_idx_kv_scale: pl.InOut[
        pl.Tensor[
            [FWD_CSA_IDX_BLOCKS_DYN, BLOCK_SIZE, 1, 1], pl.FP32
        ]
    ],
    csa_idx_block_table: pl.Tensor[
        [CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32
    ],
    csa_ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    csa_window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    csa_window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    csa_cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    csa_kv_seq_lens: pl.Tensor[[CSA_B_DYN], pl.INT32],
    hca_cmp_freqs_cos: pl.Tensor[[HCA_B, ROPE_HEAD_DIM // 2], pl.FP32],
    hca_cmp_freqs_sin: pl.Tensor[[HCA_B, ROPE_HEAD_DIM // 2], pl.FP32],
    hca_cmp_wkv: pl.Tensor[
        [FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16
    ],
    hca_cmp_wgate: pl.Tensor[
        [FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16
    ],
    hca_cmp_ape: pl.Tensor[
        [FWD_HCA_WEIGHT_BANK_SIZE * HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
        pl.FP32,
    ],
    hca_cmp_norm_w: pl.Tensor[
        [FWD_HCA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16
    ],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [
                FWD_HCA_STATE_BLOCKS_DYN,
                HCA_COMPRESS_STATE_BLOCK_SIZE,
                HCA_COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[
        [HCA_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32
    ],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[
            [FWD_HCA_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ]
    ],
    hca_cmp_block_table: pl.Tensor[
        [HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32
    ],
    hca_ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    hca_window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    hca_window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    hca_kv_seq_lens: pl.Tensor[[HCA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * H], pl.FP32],
    wo_a: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
        pl.BF16,
    ],
    wo_b: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * D, LOCAL_O_WIDTH], pl.INT8
    ],
    wo_b_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM], pl.FP32
    ],
    hc_ffn_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_w: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL, D], pl.FP32
    ],
    gate_bias: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL], pl.FP32
    ],
    tid2eid: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T_DYN], pl.INT64],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    routed_w1: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * MOE_INTER, D], pl.INT8
    ],
    shared_w1_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32
    ],
    shared_w3: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * MOE_INTER, D], pl.INT8
    ],
    shared_w3_scale: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32
    ],
    shared_w2: pl.Tensor[
        [FWD_WEIGHT_BANK_SIZE * D, MOE_INTER], pl.INT8
    ],
    shared_w2_scale: pl.Tensor[[FWD_WEIGHT_BANK_SIZE * D], pl.FP32],
    hidden_workspace: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    x_ping: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_pong: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_attn_active: pl.InOut[
        pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_moe_next: pl.InOut[
        pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]
    ],
    pre_hc_hidden_out: pl.Out[
        pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_out: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
    logits: pl.Out[
        pl.Tensor[[MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]
    ],
    sampled_ids: pl.Out[
        pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]
    ],
    attention_window: pld.DistributedTensor[
        [ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16
    ],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.FP32],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[
        [N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32
    ],
    recv_route: pld.DistributedTensor[
        [N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32
    ],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    lm_head_hidden_window: pld.DistributedTensor[
        [GROUP_LOGIT_ROWS, D], pl.BF16
    ],
    lm_head_hidden_done: pld.DistributedTensor[
        [LM_HEAD_TP_SIZE, 1], pl.INT32
    ],
    lm_head_logits_window: pld.DistributedTensor[
        [MAX_LOGIT_ROWS * LM_HEAD_VOCAB], pl.FP32
    ],
    lm_head_logits_done: pld.DistributedTensor[
        [LM_HEAD_TP_SIZE, 1], pl.INT32
    ],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the fixed 2-SWA, 21-CSA, 20-HCA model in one rank child."""
    embed_weight.bind_dynamic(0, EMBED_VOCAB_DYN)
    input_ids.bind_dynamic(0, T_DYN)
    hidden_workspace.bind_dynamic(0, T_DYN)
    x_ping.bind_dynamic(0, T_DYN)
    raw_kv_pool.bind_dynamic(0, FWD_PACKED_RAW_BLOCKS_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    swa_slot_mapping.bind_dynamic(0, T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    csa_cmp_freqs_cos.bind_dynamic(0, T_DYN)
    csa_cmp_freqs_sin.bind_dynamic(0, T_DYN)
    csa_compress_state.bind_dynamic(0, FWD_CSA_MAIN_STATE_BLOCKS_DYN)
    csa_compress_state_block_table.bind_dynamic(0, CSA_B_DYN)
    csa_inner_compress_state.bind_dynamic(0, FWD_CSA_INNER_STATE_BLOCKS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(0, CSA_B_DYN)
    csa_cmp_kv.bind_dynamic(0, FWD_CSA_CMP_BLOCKS_DYN)
    csa_cmp_block_table.bind_dynamic(0, CSA_B_DYN)
    csa_idx_kv_cache.bind_dynamic(0, FWD_CSA_IDX_BLOCKS_DYN)
    csa_idx_kv_scale.bind_dynamic(0, FWD_CSA_IDX_BLOCKS_DYN)
    csa_idx_block_table.bind_dynamic(0, CSA_B_DYN)
    csa_ori_slot_mapping.bind_dynamic(0, T_DYN)
    csa_window_swa_indices.bind_dynamic(0, T_DYN)
    csa_window_swa_lens.bind_dynamic(0, T_DYN)
    csa_cmp_slot_mapping.bind_dynamic(0, T_DYN)
    csa_idx_slot_mapping.bind_dynamic(0, T_DYN)
    csa_state_slot_mapping.bind_dynamic(0, T_DYN)
    csa_inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    csa_kv_seq_lens.bind_dynamic(0, CSA_B_DYN)
    hca_compress_state.bind_dynamic(0, FWD_HCA_STATE_BLOCKS_DYN)
    hca_compress_state_block_table.bind_dynamic(0, HCA_B_DYN)
    hca_cmp_kv.bind_dynamic(0, FWD_HCA_CMP_BLOCKS_DYN)
    hca_cmp_block_table.bind_dynamic(0, HCA_B_DYN)
    hca_cmp_block_table.bind_dynamic(1, HCA_CMP_TABLE_BLOCKS_DYN)
    hca_ori_slot_mapping.bind_dynamic(0, T_DYN)
    hca_window_swa_indices.bind_dynamic(0, T_DYN)
    hca_window_swa_lens.bind_dynamic(0, T_DYN)
    hca_cmp_slot_mapping.bind_dynamic(0, T_DYN)
    hca_state_slot_mapping.bind_dynamic(0, T_DYN)
    hca_kv_seq_lens.bind_dynamic(0, HCA_B_DYN)
    x_pong.bind_dynamic(0, T_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    pre_hc_hidden_out.bind_dynamic(0, T_DYN)
    x_out.bind_dynamic(0, T_DYN)

    moe_input_ids = pl.create_tensor([MOE_TOKENS], dtype=pl.INT64)
    with pl.scope():
        decode_embedding_preamble(input_ids, embed_weight, hidden_workspace, x_ping, moe_input_ids)

    local_t = pl.cast(pl.tensor.dim(input_ids, 0), pl.INT32)
    raw_blocks_per_layer = pl.tensor.dim(raw_kv_pool, 0) // MAIN_LAYER_COUNT
    csa_state_blocks_per_layer = pl.tensor.dim(csa_compress_state, 0) // CSA_LAYER_COUNT
    csa_cmp_blocks_per_layer = pl.tensor.dim(csa_cmp_kv, 0) // CSA_LAYER_COUNT
    csa_inner_state_blocks_per_layer = pl.tensor.dim(csa_inner_compress_state, 0) // CSA_LAYER_COUNT
    csa_idx_blocks_per_layer = pl.tensor.dim(csa_idx_kv_cache, 0) // CSA_LAYER_COUNT
    hca_state_blocks_per_layer = pl.tensor.dim(hca_compress_state, 0) // HCA_LAYER_COUNT
    hca_cmp_blocks_per_layer = pl.tensor.dim(hca_cmp_kv, 0) // HCA_LAYER_COUNT

    with pl.scope():
        weight_layer_swa0 = pl.const(0, pl.INT32)
        hc_attn_fn_layer_swa0 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [0, 0])
        hc_ffn_fn_layer_swa0 = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [0, 0])
        wq_a_layer_swa0: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [0, 0])
        wq_b_layer_swa0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [0, 0])
        wq_b_scale_layer_swa0: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [0])
        wkv_layer_swa0: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [0, 0])
        gamma_cq_layer_swa0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [0])
        gamma_ckv_layer_swa0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [0])
        wo_a_layer_swa0: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [weight_layer_swa0 * LOCAL_O_GROUPS, 0, 0])
        routed_w1_layer_swa0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [weight_layer_swa0 * N_LOCAL, 0, 0])
        hc_attn_scale_layer_swa0 = pl.slice(hc_attn_scale, [3], [0])
        hc_attn_base_layer_swa0 = pl.slice(hc_attn_base, [MIX_HC], [0])
        attn_norm_w_layer_swa0 = pl.slice(attn_norm_w, [D], [0])
        attn_sink_layer_swa0 = pl.slice(attn_sink, [H], [0])
        wo_b_layer_swa0 = pl.slice(wo_b, [D, LOCAL_O_WIDTH], [0, 0])
        wo_b_scale_layer_swa0 = pl.slice(wo_b_scale, [D], [0])
        hc_ffn_scale_layer_swa0 = pl.slice(hc_ffn_scale, [3], [0])
        hc_ffn_base_layer_swa0 = pl.slice(hc_ffn_base, [MIX_HC], [0])
        norm_w_layer_swa0 = pl.slice(norm_w, [D], [0])
        gate_w_layer_swa0 = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [0, 0])
        gate_bias_layer_swa0 = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [0])
        tid2eid_layer_swa0 = pl.slice(tid2eid, [VOCAB, TOPK], [0, 0])
        routed_w1_scale_layer_swa0 = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [0, 0])
        routed_w3_scale_layer_swa0 = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [0, 0])
        routed_w2_scale_layer_swa0 = pl.slice(routed_w2_scale, [N_LOCAL, D], [0, 0])
        shared_w1_layer_swa0 = pl.slice(shared_w1, [MOE_INTER, D], [0, 0])
        shared_w1_scale_layer_swa0 = pl.slice(shared_w1_scale, [MOE_INTER], [0])
        shared_w3_layer_swa0 = pl.slice(shared_w3, [MOE_INTER, D], [0, 0])
        shared_w3_scale_layer_swa0 = pl.slice(shared_w3_scale, [MOE_INTER], [0])
        shared_w2_layer_swa0 = pl.slice(shared_w2, [D, MOE_INTER], [0, 0])
        shared_w2_scale_layer_swa0 = pl.slice(shared_w2_scale, [D], [0])
        routed_w3_layer_swa0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [weight_layer_swa0 * N_LOCAL, 0, 0])
        routed_w2_layer_swa0: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [weight_layer_swa0 * N_LOCAL, 0, 0])
        raw_kv_layer_swa0 = pl.slice(raw_kv_pool, [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [0, 0, 0, 0])
        with pl.scope():
            if TP_SIZE == 1:
                decode_swa_tp1(
                    x_ping,
                    hc_attn_fn_layer_swa0, hc_attn_scale_layer_swa0, hc_attn_base_layer_swa0,
                    attn_norm_w_layer_swa0, wq_a_layer_swa0, wq_b_layer_swa0, wq_b_scale_layer_swa0,
                    wkv_layer_swa0, gamma_cq_layer_swa0, gamma_ckv_layer_swa0,
                    freqs_cos, freqs_sin,
                    raw_kv_layer_swa0, swa_slot_mapping, swa_indices, swa_lens, position_ids,
                    attn_sink_layer_swa0, wo_a_layer_swa0, wo_b_layer_swa0, wo_b_scale_layer_swa0,
                    x_attn_active,
                )
            else:
                decode_swa(
                    x_ping,
                    hc_attn_fn_layer_swa0, hc_attn_scale_layer_swa0, hc_attn_base_layer_swa0,
                    attn_norm_w_layer_swa0, wq_a_layer_swa0, wq_b_layer_swa0, wq_b_scale_layer_swa0,
                    wkv_layer_swa0, gamma_cq_layer_swa0, gamma_ckv_layer_swa0,
                    freqs_cos, freqs_sin,
                    raw_kv_layer_swa0, swa_slot_mapping, swa_indices, swa_lens, position_ids,
                    attn_sink_layer_swa0, wo_a_layer_swa0, wo_b_layer_swa0, wo_b_scale_layer_swa0,
                    x_attn_active,
                    attention_window, attention_signal, o_window, o_signal,
                    group_base, tp_rank, local_t,
                )

        with pl.scope():
            x_attn_moe_swa0 = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
            for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_swa0_attn_pack"):
                if token < local_t:
                    x_attn_moe_swa0[token : token + 1, 0 : HC_MULT, 0 : D] = x_attn_active[
                        token : token + 1, 0 : HC_MULT, 0 : D
                    ]
                else:
                    zero_moe_row_swa0 = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
                    x_attn_moe_swa0[token : token + 1, 0 : HC_MULT, 0 : D] = zero_moe_row_swa0
            moe(
                x_attn_moe_swa0,
                hc_ffn_fn_layer_swa0, hc_ffn_scale_layer_swa0, hc_ffn_base_layer_swa0,
                norm_w_layer_swa0, gate_w_layer_swa0, gate_bias_layer_swa0, tid2eid_layer_swa0,
                moe_input_ids,
                routed_w1_layer_swa0, routed_w1_scale_layer_swa0,
                routed_w3_layer_swa0, routed_w3_scale_layer_swa0,
                routed_w2_layer_swa0, routed_w2_scale_layer_swa0,
                shared_w1_layer_swa0, shared_w1_scale_layer_swa0,
                shared_w3_layer_swa0, shared_w3_scale_layer_swa0,
                shared_w2_layer_swa0, shared_w2_scale_layer_swa0,
                x_moe_next,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                pl.const(0, pl.INT32), local_t, my_rank, pl.const(1, pl.INT32),
            )
            for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_swa0_active_trim"):
                if token < local_t:
                    x_pong[token : token + 1, 0 : HC_MULT, 0 : D] = x_moe_next[
                        token : token + 1, 0 : HC_MULT, 0 : D
                    ]

    with pl.scope():
        weight_layer_swa1 = pl.const(1, pl.INT32) % FWD_WEIGHT_BANK_SIZE
        hc_attn_fn_layer_swa1 = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [weight_layer_swa1 * HC_FN_STORAGE_ROWS, 0])
        hc_ffn_fn_layer_swa1 = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [weight_layer_swa1 * HC_FN_STORAGE_ROWS, 0])
        wq_a_layer_swa1: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [weight_layer_swa1 * D, 0])
        wq_b_layer_swa1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [weight_layer_swa1 * Q_LORA, 0])
        wq_b_scale_layer_swa1: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [weight_layer_swa1 * H * HEAD_DIM])
        wkv_layer_swa1: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [weight_layer_swa1 * D, 0])
        gamma_cq_layer_swa1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [weight_layer_swa1 * Q_LORA])
        gamma_ckv_layer_swa1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [weight_layer_swa1 * HEAD_DIM])
        wo_a_layer_swa1: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [weight_layer_swa1 * LOCAL_O_GROUPS, 0, 0])
        routed_w1_layer_swa1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [weight_layer_swa1 * N_LOCAL, 0, 0])
        hc_attn_scale_layer_swa1 = pl.slice(hc_attn_scale, [3], [weight_layer_swa1 * 3])
        hc_attn_base_layer_swa1 = pl.slice(hc_attn_base, [MIX_HC], [weight_layer_swa1 * MIX_HC])
        attn_norm_w_layer_swa1 = pl.slice(attn_norm_w, [D], [weight_layer_swa1 * D])
        attn_sink_layer_swa1 = pl.slice(attn_sink, [H], [weight_layer_swa1 * H])
        wo_b_layer_swa1 = pl.slice(wo_b, [D, LOCAL_O_WIDTH], [weight_layer_swa1 * D, 0])
        wo_b_scale_layer_swa1 = pl.slice(wo_b_scale, [D], [weight_layer_swa1 * D])
        hc_ffn_scale_layer_swa1 = pl.slice(hc_ffn_scale, [3], [weight_layer_swa1 * 3])
        hc_ffn_base_layer_swa1 = pl.slice(hc_ffn_base, [MIX_HC], [weight_layer_swa1 * MIX_HC])
        norm_w_layer_swa1 = pl.slice(norm_w, [D], [weight_layer_swa1 * D])
        gate_w_layer_swa1 = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [weight_layer_swa1 * N_EXPERTS_GLOBAL, 0])
        gate_bias_layer_swa1 = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [weight_layer_swa1 * N_EXPERTS_GLOBAL])
        tid2eid_layer_swa1 = pl.slice(tid2eid, [VOCAB, TOPK], [weight_layer_swa1 * VOCAB, 0])
        routed_w1_scale_layer_swa1 = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [weight_layer_swa1 * N_LOCAL, 0])
        routed_w3_scale_layer_swa1 = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [weight_layer_swa1 * N_LOCAL, 0])
        routed_w2_scale_layer_swa1 = pl.slice(routed_w2_scale, [N_LOCAL, D], [weight_layer_swa1 * N_LOCAL, 0])
        shared_w1_layer_swa1 = pl.slice(shared_w1, [MOE_INTER, D], [weight_layer_swa1 * MOE_INTER, 0])
        shared_w1_scale_layer_swa1 = pl.slice(shared_w1_scale, [MOE_INTER], [weight_layer_swa1 * MOE_INTER])
        shared_w3_layer_swa1 = pl.slice(shared_w3, [MOE_INTER, D], [weight_layer_swa1 * MOE_INTER, 0])
        shared_w3_scale_layer_swa1 = pl.slice(shared_w3_scale, [MOE_INTER], [weight_layer_swa1 * MOE_INTER])
        shared_w2_layer_swa1 = pl.slice(shared_w2, [D, MOE_INTER], [weight_layer_swa1 * D, 0])
        shared_w2_scale_layer_swa1 = pl.slice(shared_w2_scale, [D], [weight_layer_swa1 * D])
        routed_w3_layer_swa1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [weight_layer_swa1 * N_LOCAL, 0, 0])
        routed_w2_layer_swa1: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [weight_layer_swa1 * N_LOCAL, 0, 0])
        raw_kv_layer_swa1 = pl.slice(raw_kv_pool, [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [raw_blocks_per_layer, 0, 0, 0])
        with pl.scope():
            if TP_SIZE == 1:
                decode_swa_tp1(
                    x_pong,
                    hc_attn_fn_layer_swa1, hc_attn_scale_layer_swa1, hc_attn_base_layer_swa1,
                    attn_norm_w_layer_swa1, wq_a_layer_swa1, wq_b_layer_swa1, wq_b_scale_layer_swa1,
                    wkv_layer_swa1, gamma_cq_layer_swa1, gamma_ckv_layer_swa1,
                    freqs_cos, freqs_sin,
                    raw_kv_layer_swa1, swa_slot_mapping, swa_indices, swa_lens, position_ids,
                    attn_sink_layer_swa1, wo_a_layer_swa1, wo_b_layer_swa1, wo_b_scale_layer_swa1,
                    x_attn_active,
                )
            else:
                decode_swa(
                    x_pong,
                    hc_attn_fn_layer_swa1, hc_attn_scale_layer_swa1, hc_attn_base_layer_swa1,
                    attn_norm_w_layer_swa1, wq_a_layer_swa1, wq_b_layer_swa1, wq_b_scale_layer_swa1,
                    wkv_layer_swa1, gamma_cq_layer_swa1, gamma_ckv_layer_swa1,
                    freqs_cos, freqs_sin,
                    raw_kv_layer_swa1, swa_slot_mapping, swa_indices, swa_lens, position_ids,
                    attn_sink_layer_swa1, wo_a_layer_swa1, wo_b_layer_swa1, wo_b_scale_layer_swa1,
                    x_attn_active,
                    attention_window, attention_signal, o_window, o_signal,
                    group_base, tp_rank, local_t,
                )

        with pl.scope():
            x_attn_moe_swa1 = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
            for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_swa1_attn_pack"):
                if token < local_t:
                    x_attn_moe_swa1[token : token + 1, 0 : HC_MULT, 0 : D] = x_attn_active[
                        token : token + 1, 0 : HC_MULT, 0 : D
                    ]
                else:
                    zero_moe_row_swa1 = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
                    x_attn_moe_swa1[token : token + 1, 0 : HC_MULT, 0 : D] = zero_moe_row_swa1
            moe(
                x_attn_moe_swa1,
                hc_ffn_fn_layer_swa1, hc_ffn_scale_layer_swa1, hc_ffn_base_layer_swa1,
                norm_w_layer_swa1, gate_w_layer_swa1, gate_bias_layer_swa1, tid2eid_layer_swa1,
                moe_input_ids,
                routed_w1_layer_swa1, routed_w1_scale_layer_swa1,
                routed_w3_layer_swa1, routed_w3_scale_layer_swa1,
                routed_w2_layer_swa1, routed_w2_scale_layer_swa1,
                shared_w1_layer_swa1, shared_w1_scale_layer_swa1,
                shared_w3_layer_swa1, shared_w3_scale_layer_swa1,
                shared_w2_layer_swa1, shared_w2_scale_layer_swa1,
                x_moe_next,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                pl.const(1, pl.INT32), local_t, my_rank, pl.const(2, pl.INT32),
            )
            for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_swa1_active_trim"):
                if token < local_t:
                    x_ping[token : token + 1, 0 : HC_MULT, 0 : D] = x_moe_next[
                        token : token + 1, 0 : HC_MULT, 0 : D
                    ]

    for ordinal in pl.range(HCA_LAYER_COUNT):
        csa_model_layer = pl.cast(ordinal * 2 + 2, pl.INT32)
        hca_model_layer = pl.cast(ordinal * 2 + 3, pl.INT32)
        csa_weight_layer = csa_model_layer % FWD_WEIGHT_BANK_SIZE
        hca_weight_layer = hca_model_layer % FWD_WEIGHT_BANK_SIZE
        csa_extra_layer = ordinal % FWD_CSA_WEIGHT_BANK_SIZE
        hca_extra_layer = ordinal % FWD_HCA_WEIGHT_BANK_SIZE

        with pl.scope():
            hc_attn_fn_layer_csa = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [csa_weight_layer * HC_FN_STORAGE_ROWS, 0])
            hc_ffn_fn_layer_csa = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [csa_weight_layer * HC_FN_STORAGE_ROWS, 0])
            wq_a_layer_csa: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_weight_layer * D, 0])
            wq_b_layer_csa: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [csa_weight_layer * Q_LORA, 0])
            wq_b_scale_layer_csa: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [csa_weight_layer * H * HEAD_DIM])
            wkv_layer_csa: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_weight_layer * D, 0])
            gamma_cq_layer_csa: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_weight_layer * Q_LORA])
            gamma_ckv_layer_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [csa_weight_layer * HEAD_DIM])
            wo_a_layer_csa: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [csa_weight_layer * LOCAL_O_GROUPS, 0, 0])
            routed_w1_layer_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [csa_weight_layer * N_LOCAL, 0, 0])
            routed_w3_layer_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [csa_weight_layer * N_LOCAL, 0, 0])
            routed_w2_layer_csa: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [csa_weight_layer * N_LOCAL, 0, 0])
            raw_kv_layer_csa = pl.slice(raw_kv_pool, [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [csa_model_layer * raw_blocks_per_layer, 0, 0, 0])
            csa_state_layer_csa = pl.slice(csa_compress_state, [csa_state_blocks_per_layer, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], [ordinal * csa_state_blocks_per_layer, 0, 0])
            csa_cmp_kv_layer_csa = pl.slice(csa_cmp_kv, [csa_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [ordinal * csa_cmp_blocks_per_layer, 0, 0, 0])
            csa_inner_state_layer_csa = pl.slice(csa_inner_compress_state, [csa_inner_state_blocks_per_layer, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], [ordinal * csa_inner_state_blocks_per_layer, 0, 0])
            csa_idx_cache_layer_csa = pl.slice(csa_idx_kv_cache, [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], [ordinal * csa_idx_blocks_per_layer, 0, 0, 0])
            csa_idx_scale_layer_csa = pl.slice(csa_idx_kv_scale, [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, 1], [ordinal * csa_idx_blocks_per_layer, 0, 0, 0])
            hc_attn_scale_layer_csa = pl.slice(hc_attn_scale, [3], [csa_weight_layer * 3])
            hc_attn_base_layer_csa = pl.slice(hc_attn_base, [MIX_HC], [csa_weight_layer * MIX_HC])
            attn_norm_w_layer_csa = pl.slice(attn_norm_w, [D], [csa_weight_layer * D])
            csa_cmp_wkv_layer_csa = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [csa_extra_layer * CSA_MAIN_OUT_DIM, 0])
            csa_cmp_wgate_layer_csa = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [csa_extra_layer * CSA_MAIN_OUT_DIM, 0])
            csa_cmp_ape_layer_csa = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_extra_layer * CSA_COMPRESS_RATIO, 0])
            csa_cmp_norm_w_layer_csa = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [csa_extra_layer * HEAD_DIM])
            csa_idx_wq_b_layer_csa = pl.slice(csa_idx_wq_b, [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [csa_extra_layer * Q_LORA, 0])
            csa_idx_wq_b_scale_layer_csa = pl.slice(csa_idx_wq_b_scale, [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [csa_extra_layer * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM])
            csa_weights_proj_layer_csa = pl.slice(csa_weights_proj, [D, CSA_IDX_N_HEADS], [csa_extra_layer * D, 0])
            csa_hadamard_idx_layer_csa = pl.slice(csa_hadamard_idx, [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], [csa_extra_layer * CSA_IDX_HEAD_DIM, 0])
            csa_inner_wkv_layer_csa = pl.slice(csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [csa_extra_layer * CSA_INNER_OUT_DIM, 0])
            csa_inner_wgate_layer_csa = pl.slice(csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [csa_extra_layer * CSA_INNER_OUT_DIM, 0])
            csa_inner_ape_layer_csa = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [csa_extra_layer * CSA_COMPRESS_RATIO, 0])
            csa_inner_norm_w_layer_csa = pl.slice(csa_inner_norm_w, [CSA_IDX_HEAD_DIM], [csa_extra_layer * CSA_IDX_HEAD_DIM])
            attn_sink_layer_csa = pl.slice(attn_sink, [H], [csa_weight_layer * H])
            wo_b_layer_csa = pl.slice(wo_b, [D, LOCAL_O_WIDTH], [csa_weight_layer * D, 0])
            wo_b_scale_layer_csa = pl.slice(wo_b_scale, [D], [csa_weight_layer * D])
            hc_ffn_scale_layer_csa = pl.slice(hc_ffn_scale, [3], [csa_weight_layer * 3])
            hc_ffn_base_layer_csa = pl.slice(hc_ffn_base, [MIX_HC], [csa_weight_layer * MIX_HC])
            norm_w_layer_csa = pl.slice(norm_w, [D], [csa_weight_layer * D])
            gate_w_layer_csa = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [csa_weight_layer * N_EXPERTS_GLOBAL, 0])
            gate_bias_layer_csa = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [csa_weight_layer * N_EXPERTS_GLOBAL])
            tid2eid_layer_csa = pl.slice(tid2eid, [VOCAB, TOPK], [csa_weight_layer * VOCAB, 0])
            routed_w1_scale_layer_csa = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [csa_weight_layer * N_LOCAL, 0])
            routed_w3_scale_layer_csa = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [csa_weight_layer * N_LOCAL, 0])
            routed_w2_scale_layer_csa = pl.slice(routed_w2_scale, [N_LOCAL, D], [csa_weight_layer * N_LOCAL, 0])
            shared_w1_layer_csa = pl.slice(shared_w1, [MOE_INTER, D], [csa_weight_layer * MOE_INTER, 0])
            shared_w1_scale_layer_csa = pl.slice(shared_w1_scale, [MOE_INTER], [csa_weight_layer * MOE_INTER])
            shared_w3_layer_csa = pl.slice(shared_w3, [MOE_INTER, D], [csa_weight_layer * MOE_INTER, 0])
            shared_w3_scale_layer_csa = pl.slice(shared_w3_scale, [MOE_INTER], [csa_weight_layer * MOE_INTER])
            shared_w2_layer_csa = pl.slice(shared_w2, [D, MOE_INTER], [csa_weight_layer * D, 0])
            shared_w2_scale_layer_csa = pl.slice(shared_w2_scale, [D], [csa_weight_layer * D])
            with pl.scope():
                if TP_SIZE == 1:
                    decode_csa_tp1(
                        x_ping,
                        hc_attn_fn_layer_csa, hc_attn_scale_layer_csa, hc_attn_base_layer_csa,
                        attn_norm_w_layer_csa, wq_a_layer_csa, wq_b_layer_csa, wq_b_scale_layer_csa,
                        wkv_layer_csa, gamma_cq_layer_csa, gamma_ckv_layer_csa,
                        freqs_cos, freqs_sin, csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                        csa_cmp_wkv_layer_csa, csa_cmp_wgate_layer_csa,
                        csa_cmp_ape_layer_csa, csa_cmp_norm_w_layer_csa,
                        csa_state_layer_csa, csa_compress_state_block_table,
                        csa_idx_wq_b_layer_csa, csa_idx_wq_b_scale_layer_csa,
                        csa_weights_proj_layer_csa, csa_hadamard_idx_layer_csa,
                        csa_inner_wkv_layer_csa, csa_inner_wgate_layer_csa,
                        csa_inner_ape_layer_csa, csa_inner_norm_w_layer_csa,
                        csa_inner_state_layer_csa, csa_inner_compress_state_block_table,
                        raw_kv_layer_csa, csa_cmp_kv_layer_csa, csa_cmp_block_table,
                        csa_idx_cache_layer_csa, csa_idx_scale_layer_csa, csa_idx_block_table,
                        csa_ori_slot_mapping, csa_window_swa_indices, csa_window_swa_lens,
                        csa_cmp_slot_mapping, csa_idx_slot_mapping,
                        csa_state_slot_mapping, csa_inner_state_slot_mapping,
                        position_ids, csa_kv_seq_lens,
                        attn_sink_layer_csa, wo_a_layer_csa, wo_b_layer_csa, wo_b_scale_layer_csa,
                        x_attn_active,
                    )
                else:
                    decode_csa(
                        x_ping,
                        hc_attn_fn_layer_csa, hc_attn_scale_layer_csa, hc_attn_base_layer_csa,
                        attn_norm_w_layer_csa, wq_a_layer_csa, wq_b_layer_csa, wq_b_scale_layer_csa,
                        wkv_layer_csa, gamma_cq_layer_csa, gamma_ckv_layer_csa,
                        freqs_cos, freqs_sin, csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                        csa_cmp_wkv_layer_csa, csa_cmp_wgate_layer_csa,
                        csa_cmp_ape_layer_csa, csa_cmp_norm_w_layer_csa,
                        csa_state_layer_csa, csa_compress_state_block_table,
                        csa_idx_wq_b_layer_csa, csa_idx_wq_b_scale_layer_csa,
                        csa_weights_proj_layer_csa, csa_hadamard_idx_layer_csa,
                        csa_inner_wkv_layer_csa, csa_inner_wgate_layer_csa,
                        csa_inner_ape_layer_csa, csa_inner_norm_w_layer_csa,
                        csa_inner_state_layer_csa, csa_inner_compress_state_block_table,
                        raw_kv_layer_csa, csa_cmp_kv_layer_csa, csa_cmp_block_table,
                        csa_idx_cache_layer_csa, csa_idx_scale_layer_csa, csa_idx_block_table,
                        csa_ori_slot_mapping, csa_window_swa_indices, csa_window_swa_lens,
                        csa_cmp_slot_mapping, csa_idx_slot_mapping,
                        csa_state_slot_mapping, csa_inner_state_slot_mapping,
                        position_ids, csa_kv_seq_lens,
                        attn_sink_layer_csa, wo_a_layer_csa, wo_b_layer_csa, wo_b_scale_layer_csa,
                        x_attn_active,
                        attention_window, attention_signal, o_window, o_signal,
                        group_base, tp_rank, local_t,
                    )

            with pl.scope():
                x_attn_moe_csa = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
                for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_csa_attn_pack"):
                    if token < local_t:
                        x_attn_moe_csa[token : token + 1, 0 : HC_MULT, 0 : D] = x_attn_active[
                            token : token + 1, 0 : HC_MULT, 0 : D
                        ]
                    else:
                        zero_moe_row_csa = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
                        x_attn_moe_csa[token : token + 1, 0 : HC_MULT, 0 : D] = zero_moe_row_csa
                moe(
                    x_attn_moe_csa,
                    hc_ffn_fn_layer_csa, hc_ffn_scale_layer_csa, hc_ffn_base_layer_csa,
                    norm_w_layer_csa, gate_w_layer_csa, gate_bias_layer_csa, tid2eid_layer_csa,
                    moe_input_ids,
                    routed_w1_layer_csa, routed_w1_scale_layer_csa,
                    routed_w3_layer_csa, routed_w3_scale_layer_csa,
                    routed_w2_layer_csa, routed_w2_scale_layer_csa,
                    shared_w1_layer_csa, shared_w1_scale_layer_csa,
                    shared_w3_layer_csa, shared_w3_scale_layer_csa,
                    shared_w2_layer_csa, shared_w2_scale_layer_csa,
                    x_moe_next,
                    recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                    routed_y_buf, combine_arrived,
                    csa_model_layer, local_t, my_rank, csa_model_layer + 1,
                )
                for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_csa_active_trim"):
                    if token < local_t:
                        x_pong[token : token + 1, 0 : HC_MULT, 0 : D] = x_moe_next[
                            token : token + 1, 0 : HC_MULT, 0 : D
                        ]

        with pl.scope():
            hc_attn_fn_layer_hca = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [hca_weight_layer * HC_FN_STORAGE_ROWS, 0])
            hc_ffn_fn_layer_hca = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [hca_weight_layer * HC_FN_STORAGE_ROWS, 0])
            wq_a_layer_hca: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [hca_weight_layer * D, 0])
            wq_b_layer_hca: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [hca_weight_layer * Q_LORA, 0])
            wq_b_scale_layer_hca: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [hca_weight_layer * H * HEAD_DIM])
            wkv_layer_hca: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [hca_weight_layer * D, 0])
            gamma_cq_layer_hca: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [hca_weight_layer * Q_LORA])
            gamma_ckv_layer_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [hca_weight_layer * HEAD_DIM])
            wo_a_layer_hca: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [hca_weight_layer * LOCAL_O_GROUPS, 0, 0])
            routed_w1_layer_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [hca_weight_layer * N_LOCAL, 0, 0])
            routed_w3_layer_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [hca_weight_layer * N_LOCAL, 0, 0])
            routed_w2_layer_hca: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [hca_weight_layer * N_LOCAL, 0, 0])
            raw_kv_layer_hca = pl.slice(raw_kv_pool, [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [hca_model_layer * raw_blocks_per_layer, 0, 0, 0])
            hca_state_layer_hca = pl.slice(hca_compress_state, [hca_state_blocks_per_layer, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], [ordinal * hca_state_blocks_per_layer, 0, 0])
            hca_cmp_kv_layer_hca = pl.slice(hca_cmp_kv, [hca_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [ordinal * hca_cmp_blocks_per_layer, 0, 0, 0])
            hc_attn_scale_layer_hca = pl.slice(hc_attn_scale, [3], [hca_weight_layer * 3])
            hc_attn_base_layer_hca = pl.slice(hc_attn_base, [MIX_HC], [hca_weight_layer * MIX_HC])
            attn_norm_w_layer_hca = pl.slice(attn_norm_w, [D], [hca_weight_layer * D])
            hca_cmp_wkv_layer_hca = pl.slice(hca_cmp_wkv, [HCA_MAIN_OUT_DIM, D], [hca_extra_layer * HCA_MAIN_OUT_DIM, 0])
            hca_cmp_wgate_layer_hca = pl.slice(hca_cmp_wgate, [HCA_MAIN_OUT_DIM, D], [hca_extra_layer * HCA_MAIN_OUT_DIM, 0])
            hca_cmp_ape_layer_hca = pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], [hca_extra_layer * HCA_COMPRESS_RATIO, 0])
            hca_cmp_norm_w_layer_hca = pl.slice(hca_cmp_norm_w, [HEAD_DIM], [hca_extra_layer * HEAD_DIM])
            attn_sink_layer_hca = pl.slice(attn_sink, [H], [hca_weight_layer * H])
            wo_b_layer_hca = pl.slice(wo_b, [D, LOCAL_O_WIDTH], [hca_weight_layer * D, 0])
            wo_b_scale_layer_hca = pl.slice(wo_b_scale, [D], [hca_weight_layer * D])
            hc_ffn_scale_layer_hca = pl.slice(hc_ffn_scale, [3], [hca_weight_layer * 3])
            hc_ffn_base_layer_hca = pl.slice(hc_ffn_base, [MIX_HC], [hca_weight_layer * MIX_HC])
            norm_w_layer_hca = pl.slice(norm_w, [D], [hca_weight_layer * D])
            gate_w_layer_hca = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [hca_weight_layer * N_EXPERTS_GLOBAL, 0])
            gate_bias_layer_hca = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [hca_weight_layer * N_EXPERTS_GLOBAL])
            tid2eid_layer_hca = pl.slice(tid2eid, [VOCAB, TOPK], [hca_weight_layer * VOCAB, 0])
            routed_w1_scale_layer_hca = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [hca_weight_layer * N_LOCAL, 0])
            routed_w3_scale_layer_hca = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [hca_weight_layer * N_LOCAL, 0])
            routed_w2_scale_layer_hca = pl.slice(routed_w2_scale, [N_LOCAL, D], [hca_weight_layer * N_LOCAL, 0])
            shared_w1_layer_hca = pl.slice(shared_w1, [MOE_INTER, D], [hca_weight_layer * MOE_INTER, 0])
            shared_w1_scale_layer_hca = pl.slice(shared_w1_scale, [MOE_INTER], [hca_weight_layer * MOE_INTER])
            shared_w3_layer_hca = pl.slice(shared_w3, [MOE_INTER, D], [hca_weight_layer * MOE_INTER, 0])
            shared_w3_scale_layer_hca = pl.slice(shared_w3_scale, [MOE_INTER], [hca_weight_layer * MOE_INTER])
            shared_w2_layer_hca = pl.slice(shared_w2, [D, MOE_INTER], [hca_weight_layer * D, 0])
            shared_w2_scale_layer_hca = pl.slice(shared_w2_scale, [D], [hca_weight_layer * D])
            with pl.scope():
                if TP_SIZE == 1:
                    decode_hca_tp1(
                        x_pong,
                        hc_attn_fn_layer_hca, hc_attn_scale_layer_hca, hc_attn_base_layer_hca,
                        attn_norm_w_layer_hca, wq_a_layer_hca, wq_b_layer_hca, wq_b_scale_layer_hca,
                        wkv_layer_hca, gamma_cq_layer_hca, gamma_ckv_layer_hca,
                        freqs_cos, freqs_sin, hca_cmp_freqs_cos, hca_cmp_freqs_sin,
                        hca_cmp_wkv_layer_hca, hca_cmp_wgate_layer_hca,
                        hca_cmp_ape_layer_hca, hca_cmp_norm_w_layer_hca,
                        hca_state_layer_hca, hca_compress_state_block_table,
                        raw_kv_layer_hca, hca_cmp_kv_layer_hca, hca_cmp_block_table,
                        hca_ori_slot_mapping, hca_window_swa_indices, hca_window_swa_lens,
                        hca_cmp_slot_mapping, hca_state_slot_mapping,
                        position_ids, hca_kv_seq_lens,
                        attn_sink_layer_hca, wo_a_layer_hca, wo_b_layer_hca, wo_b_scale_layer_hca,
                        x_attn_active,
                    )
                else:
                    decode_hca(
                        x_pong,
                        hc_attn_fn_layer_hca, hc_attn_scale_layer_hca, hc_attn_base_layer_hca,
                        attn_norm_w_layer_hca, wq_a_layer_hca, wq_b_layer_hca, wq_b_scale_layer_hca,
                        wkv_layer_hca, gamma_cq_layer_hca, gamma_ckv_layer_hca,
                        freqs_cos, freqs_sin, hca_cmp_freqs_cos, hca_cmp_freqs_sin,
                        hca_cmp_wkv_layer_hca, hca_cmp_wgate_layer_hca,
                        hca_cmp_ape_layer_hca, hca_cmp_norm_w_layer_hca,
                        hca_state_layer_hca, hca_compress_state_block_table,
                        raw_kv_layer_hca, hca_cmp_kv_layer_hca, hca_cmp_block_table,
                        hca_ori_slot_mapping, hca_window_swa_indices, hca_window_swa_lens,
                        hca_cmp_slot_mapping, hca_state_slot_mapping,
                        position_ids, hca_kv_seq_lens,
                        attn_sink_layer_hca, wo_a_layer_hca, wo_b_layer_hca, wo_b_scale_layer_hca,
                        x_attn_active,
                        attention_window, attention_signal, o_window, o_signal,
                        group_base, tp_rank, local_t,
                    )

            with pl.scope():
                x_attn_moe_hca = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
                for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_hca_attn_pack"):
                    if token < local_t:
                        x_attn_moe_hca[token : token + 1, 0 : HC_MULT, 0 : D] = x_attn_active[
                            token : token + 1, 0 : HC_MULT, 0 : D
                        ]
                    else:
                        zero_moe_row_hca = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
                        x_attn_moe_hca[token : token + 1, 0 : HC_MULT, 0 : D] = zero_moe_row_hca
                moe(
                    x_attn_moe_hca,
                    hc_ffn_fn_layer_hca, hc_ffn_scale_layer_hca, hc_ffn_base_layer_hca,
                    norm_w_layer_hca, gate_w_layer_hca, gate_bias_layer_hca, tid2eid_layer_hca,
                    moe_input_ids,
                    routed_w1_layer_hca, routed_w1_scale_layer_hca,
                    routed_w3_layer_hca, routed_w3_scale_layer_hca,
                    routed_w2_layer_hca, routed_w2_scale_layer_hca,
                    shared_w1_layer_hca, shared_w1_scale_layer_hca,
                    shared_w3_layer_hca, shared_w3_scale_layer_hca,
                    shared_w2_layer_hca, shared_w2_scale_layer_hca,
                    x_moe_next,
                    recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                    routed_y_buf, combine_arrived,
                    hca_model_layer, local_t, my_rank, hca_model_layer + 1,
                )
                for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_hca_active_trim"):
                    if token < local_t:
                        x_ping[token : token + 1, 0 : HC_MULT, 0 : D] = x_moe_next[
                            token : token + 1, 0 : HC_MULT, 0 : D
                        ]

    with pl.scope():
        csa_ordinal_last = pl.const(20, pl.INT32)
        model_layer_last = pl.const(42, pl.INT32)
        weight_layer_last = model_layer_last % FWD_WEIGHT_BANK_SIZE
        extra_layer_last = csa_ordinal_last % FWD_CSA_WEIGHT_BANK_SIZE
        hc_attn_fn_layer_last = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [weight_layer_last * HC_FN_STORAGE_ROWS, 0])
        hc_ffn_fn_layer_last = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [weight_layer_last * HC_FN_STORAGE_ROWS, 0])
        wq_a_layer_last: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [weight_layer_last * D, 0])
        wq_b_layer_last: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [weight_layer_last * Q_LORA, 0])
        wq_b_scale_layer_last: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [weight_layer_last * H * HEAD_DIM])
        wkv_layer_last: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [weight_layer_last * D, 0])
        gamma_cq_layer_last: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [weight_layer_last * Q_LORA])
        gamma_ckv_layer_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [weight_layer_last * HEAD_DIM])
        wo_a_layer_last: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [weight_layer_last * LOCAL_O_GROUPS, 0, 0])
        routed_w1_layer_last: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [weight_layer_last * N_LOCAL, 0, 0])
        routed_w3_layer_last: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [weight_layer_last * N_LOCAL, 0, 0])
        routed_w2_layer_last: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [weight_layer_last * N_LOCAL, 0, 0])
        raw_kv_layer_last = pl.slice(raw_kv_pool, [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [model_layer_last * raw_blocks_per_layer, 0, 0, 0])
        csa_state_layer_last = pl.slice(csa_compress_state, [csa_state_blocks_per_layer, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], [csa_ordinal_last * csa_state_blocks_per_layer, 0, 0])
        csa_cmp_kv_layer_last = pl.slice(csa_cmp_kv, [csa_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM], [csa_ordinal_last * csa_cmp_blocks_per_layer, 0, 0, 0])
        csa_inner_state_layer_last = pl.slice(csa_inner_compress_state, [csa_inner_state_blocks_per_layer, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], [csa_ordinal_last * csa_inner_state_blocks_per_layer, 0, 0])
        csa_idx_cache_layer_last = pl.slice(csa_idx_kv_cache, [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], [csa_ordinal_last * csa_idx_blocks_per_layer, 0, 0, 0])
        csa_idx_scale_layer_last = pl.slice(csa_idx_kv_scale, [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, 1], [csa_ordinal_last * csa_idx_blocks_per_layer, 0, 0, 0])
        hc_attn_scale_layer_last = pl.slice(hc_attn_scale, [3], [weight_layer_last * 3])
        hc_attn_base_layer_last = pl.slice(hc_attn_base, [MIX_HC], [weight_layer_last * MIX_HC])
        attn_norm_w_layer_last = pl.slice(attn_norm_w, [D], [weight_layer_last * D])
        csa_cmp_wkv_layer_last = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [extra_layer_last * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_wgate_layer_last = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [extra_layer_last * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_ape_layer_last = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [extra_layer_last * CSA_COMPRESS_RATIO, 0])
        csa_cmp_norm_w_layer_last = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [extra_layer_last * HEAD_DIM])
        csa_idx_wq_b_layer_last = pl.slice(csa_idx_wq_b, [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [extra_layer_last * Q_LORA, 0])
        csa_idx_wq_b_scale_layer_last = pl.slice(csa_idx_wq_b_scale, [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], [extra_layer_last * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM])
        csa_weights_proj_layer_last = pl.slice(csa_weights_proj, [D, CSA_IDX_N_HEADS], [extra_layer_last * D, 0])
        csa_hadamard_idx_layer_last = pl.slice(csa_hadamard_idx, [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], [extra_layer_last * CSA_IDX_HEAD_DIM, 0])
        csa_inner_wkv_layer_last = pl.slice(csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [extra_layer_last * CSA_INNER_OUT_DIM, 0])
        csa_inner_wgate_layer_last = pl.slice(csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [extra_layer_last * CSA_INNER_OUT_DIM, 0])
        csa_inner_ape_layer_last = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [extra_layer_last * CSA_COMPRESS_RATIO, 0])
        csa_inner_norm_w_layer_last = pl.slice(csa_inner_norm_w, [CSA_IDX_HEAD_DIM], [extra_layer_last * CSA_IDX_HEAD_DIM])
        attn_sink_layer_last = pl.slice(attn_sink, [H], [weight_layer_last * H])
        wo_b_layer_last = pl.slice(wo_b, [D, LOCAL_O_WIDTH], [weight_layer_last * D, 0])
        wo_b_scale_layer_last = pl.slice(wo_b_scale, [D], [weight_layer_last * D])
        hc_ffn_scale_layer_last = pl.slice(hc_ffn_scale, [3], [weight_layer_last * 3])
        hc_ffn_base_layer_last = pl.slice(hc_ffn_base, [MIX_HC], [weight_layer_last * MIX_HC])
        norm_w_layer_last = pl.slice(norm_w, [D], [weight_layer_last * D])
        gate_w_layer_last = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [weight_layer_last * N_EXPERTS_GLOBAL, 0])
        gate_bias_layer_last = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [weight_layer_last * N_EXPERTS_GLOBAL])
        tid2eid_layer_last = pl.slice(tid2eid, [VOCAB, TOPK], [weight_layer_last * VOCAB, 0])
        routed_w1_scale_layer_last = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [weight_layer_last * N_LOCAL, 0])
        routed_w3_scale_layer_last = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [weight_layer_last * N_LOCAL, 0])
        routed_w2_scale_layer_last = pl.slice(routed_w2_scale, [N_LOCAL, D], [weight_layer_last * N_LOCAL, 0])
        shared_w1_layer_last = pl.slice(shared_w1, [MOE_INTER, D], [weight_layer_last * MOE_INTER, 0])
        shared_w1_scale_layer_last = pl.slice(shared_w1_scale, [MOE_INTER], [weight_layer_last * MOE_INTER])
        shared_w3_layer_last = pl.slice(shared_w3, [MOE_INTER, D], [weight_layer_last * MOE_INTER, 0])
        shared_w3_scale_layer_last = pl.slice(shared_w3_scale, [MOE_INTER], [weight_layer_last * MOE_INTER])
        shared_w2_layer_last = pl.slice(shared_w2, [D, MOE_INTER], [weight_layer_last * D, 0])
        shared_w2_scale_layer_last = pl.slice(shared_w2_scale, [D], [weight_layer_last * D])
        with pl.scope():
            if TP_SIZE == 1:
                decode_csa_tp1(
                    x_ping,
                    hc_attn_fn_layer_last, hc_attn_scale_layer_last, hc_attn_base_layer_last,
                    attn_norm_w_layer_last, wq_a_layer_last, wq_b_layer_last, wq_b_scale_layer_last,
                    wkv_layer_last, gamma_cq_layer_last, gamma_ckv_layer_last,
                    freqs_cos, freqs_sin, csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                    csa_cmp_wkv_layer_last, csa_cmp_wgate_layer_last,
                    csa_cmp_ape_layer_last, csa_cmp_norm_w_layer_last,
                    csa_state_layer_last, csa_compress_state_block_table,
                    csa_idx_wq_b_layer_last, csa_idx_wq_b_scale_layer_last,
                    csa_weights_proj_layer_last, csa_hadamard_idx_layer_last,
                    csa_inner_wkv_layer_last, csa_inner_wgate_layer_last,
                    csa_inner_ape_layer_last, csa_inner_norm_w_layer_last,
                    csa_inner_state_layer_last, csa_inner_compress_state_block_table,
                    raw_kv_layer_last, csa_cmp_kv_layer_last, csa_cmp_block_table,
                    csa_idx_cache_layer_last, csa_idx_scale_layer_last, csa_idx_block_table,
                    csa_ori_slot_mapping, csa_window_swa_indices, csa_window_swa_lens,
                    csa_cmp_slot_mapping, csa_idx_slot_mapping,
                    csa_state_slot_mapping, csa_inner_state_slot_mapping,
                    position_ids, csa_kv_seq_lens,
                    attn_sink_layer_last, wo_a_layer_last, wo_b_layer_last, wo_b_scale_layer_last,
                    x_attn_active,
                )
            else:
                decode_csa(
                    x_ping,
                    hc_attn_fn_layer_last, hc_attn_scale_layer_last, hc_attn_base_layer_last,
                    attn_norm_w_layer_last, wq_a_layer_last, wq_b_layer_last, wq_b_scale_layer_last,
                    wkv_layer_last, gamma_cq_layer_last, gamma_ckv_layer_last,
                    freqs_cos, freqs_sin, csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                    csa_cmp_wkv_layer_last, csa_cmp_wgate_layer_last,
                    csa_cmp_ape_layer_last, csa_cmp_norm_w_layer_last,
                    csa_state_layer_last, csa_compress_state_block_table,
                    csa_idx_wq_b_layer_last, csa_idx_wq_b_scale_layer_last,
                    csa_weights_proj_layer_last, csa_hadamard_idx_layer_last,
                    csa_inner_wkv_layer_last, csa_inner_wgate_layer_last,
                    csa_inner_ape_layer_last, csa_inner_norm_w_layer_last,
                    csa_inner_state_layer_last, csa_inner_compress_state_block_table,
                    raw_kv_layer_last, csa_cmp_kv_layer_last, csa_cmp_block_table,
                    csa_idx_cache_layer_last, csa_idx_scale_layer_last, csa_idx_block_table,
                    csa_ori_slot_mapping, csa_window_swa_indices, csa_window_swa_lens,
                    csa_cmp_slot_mapping, csa_idx_slot_mapping,
                    csa_state_slot_mapping, csa_inner_state_slot_mapping,
                    position_ids, csa_kv_seq_lens,
                    attn_sink_layer_last, wo_a_layer_last, wo_b_layer_last, wo_b_scale_layer_last,
                    x_attn_active,
                    attention_window, attention_signal, o_window, o_signal,
                    group_base, tp_rank, local_t,
                )

        with pl.scope():
            x_attn_moe_last = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
            for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_last_attn_pack"):
                if token < local_t:
                    x_attn_moe_last[token : token + 1, 0 : HC_MULT, 0 : D] = x_attn_active[
                        token : token + 1, 0 : HC_MULT, 0 : D
                    ]
                else:
                    zero_moe_row_last = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)
                    x_attn_moe_last[token : token + 1, 0 : HC_MULT, 0 : D] = zero_moe_row_last
            moe(
                x_attn_moe_last,
                hc_ffn_fn_layer_last, hc_ffn_scale_layer_last, hc_ffn_base_layer_last,
                norm_w_layer_last, gate_w_layer_last, gate_bias_layer_last, tid2eid_layer_last,
                moe_input_ids,
                routed_w1_layer_last, routed_w1_scale_layer_last,
                routed_w3_layer_last, routed_w3_scale_layer_last,
                routed_w2_layer_last, routed_w2_scale_layer_last,
                shared_w1_layer_last, shared_w1_scale_layer_last,
                shared_w3_layer_last, shared_w3_scale_layer_last,
                shared_w2_layer_last, shared_w2_scale_layer_last,
                x_moe_next,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                model_layer_last, local_t, my_rank, pl.const(43, pl.INT32),
            )
            for token in pl.spmd(MOE_TOKENS, name_hint="decode_fwd_last_active_trim"):
                if token < local_t:
                    pre_hc_hidden_out[token : token + 1, 0 : HC_MULT, 0 : D] = x_moe_next[
                        token : token + 1, 0 : HC_MULT, 0 : D
                    ]
    clear_moe_signals(x_moe_next, arrived, data_arrived, combine_arrived)

    with pl.scope():
        hc_head(pre_hc_hidden_out, hc_head_fn, hc_head_scale, hc_head_base, hidden_workspace)
        final_norm_tid = rms_norm(hidden_workspace, final_norm_w, x_out)
        lm_head(
            x_out,
            lm_head_weight,
            logit_row_indices,
            logits,
            lm_head_hidden_window,
            lm_head_hidden_done,
            lm_head_logits_window,
            lm_head_logits_done,
            group_base,
            tp_rank,
            pl.const(LM_HEAD_COMM_EPOCH, pl.INT32),
            final_norm_tid,
        )
        greedy_sample(logits, sampled_ids)
        mask_inactive_sample_rows(logit_row_indices, sampled_ids)
    return x_out


@pl.jit.host
def l3_decode_fwd(
    embed_weight: pl.Tensor[
        [N_RANKS, EMBED_VOCAB_DYN, D], pl.BF16
    ],
    hc_attn_fn: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM],
        pl.FP32,
    ],
    hc_attn_scale: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * 3], pl.FP32],
    hc_attn_base: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32
    ],
    attn_norm_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    wq_a: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * D, Q_LORA], pl.BF16
    ],
    wq_b: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * Q_LORA, H * HEAD_DIM], pl.INT8
    ],
    wq_b_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * H * HEAD_DIM], pl.FP32
    ],
    wkv: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * D, HEAD_DIM], pl.BF16
    ],
    gamma_cq: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * Q_LORA], pl.BF16
    ],
    gamma_ckv: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16
    ],
    raw_kv_pool: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                FWD_PACKED_RAW_BLOCKS_DYN,
                BLOCK_SIZE,
                1,
                HEAD_DIM,
            ],
            pl.BF16,
        ]
    ],
    freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    swa_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    csa_cmp_freqs_cos: pl.Tensor[
        [N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16
    ],
    csa_cmp_freqs_sin: pl.Tensor[
        [N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16
    ],
    csa_cmp_wkv: pl.Tensor[
        [N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16
    ],
    csa_cmp_wgate: pl.Tensor[
        [N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_MAIN_OUT_DIM, D], pl.BF16
    ],
    csa_cmp_ape: pl.Tensor[
        [
            N_RANKS,
            FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO,
            CSA_MAIN_OUT_DIM,
        ],
        pl.FP32,
    ],
    csa_cmp_norm_w: pl.Tensor[
        [N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16
    ],
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                FWD_CSA_MAIN_STATE_BLOCKS_DYN,
                CSA_MAIN_STATE_BLOCK_SIZE,
                CSA_MAIN_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_compress_state_block_table: pl.Tensor[
        [N_RANKS, CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_idx_wq_b: pl.Tensor[
        [
            N_RANKS,
            FWD_CSA_WEIGHT_BANK_SIZE * Q_LORA,
            CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM,
        ],
        pl.INT8,
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [
            N_RANKS,
            FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM,
        ],
        pl.FP32,
    ],
    csa_weights_proj: pl.Tensor[
        [N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * D, CSA_IDX_N_HEADS], pl.BF16
    ],
    csa_hadamard_idx: pl.Tensor[
        [
            N_RANKS,
            FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM,
            CSA_IDX_HEAD_DIM,
        ],
        pl.BF16,
    ],
    csa_inner_wkv: pl.Tensor[
        [N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D],
        pl.BF16,
    ],
    csa_inner_wgate: pl.Tensor[
        [N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_INNER_OUT_DIM, D],
        pl.BF16,
    ],
    csa_inner_ape: pl.Tensor[
        [
            N_RANKS,
            FWD_CSA_WEIGHT_BANK_SIZE * CSA_COMPRESS_RATIO,
            CSA_INNER_OUT_DIM,
        ],
        pl.FP32,
    ],
    csa_inner_norm_w: pl.Tensor[
        [N_RANKS, FWD_CSA_WEIGHT_BANK_SIZE * CSA_IDX_HEAD_DIM], pl.BF16
    ],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                FWD_CSA_INNER_STATE_BLOCKS_DYN,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state_block_table: pl.Tensor[
        [N_RANKS, CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_cmp_kv: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                FWD_CSA_CMP_BLOCKS_DYN,
                BLOCK_SIZE,
                1,
                HEAD_DIM,
            ],
            pl.BF16,
        ]
    ],
    csa_cmp_block_table: pl.Tensor[
        [N_RANKS, CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32
    ],
    csa_idx_kv_cache: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                FWD_CSA_IDX_BLOCKS_DYN,
                BLOCK_SIZE,
                1,
                CSA_IDX_HEAD_DIM,
            ],
            pl.INT8,
        ]
    ],
    csa_idx_kv_scale: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_CSA_IDX_BLOCKS_DYN, BLOCK_SIZE, 1, 1],
            pl.FP32,
        ]
    ],
    csa_idx_block_table: pl.Tensor[
        [N_RANKS, CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32
    ],
    csa_ori_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    csa_window_swa_indices: pl.Tensor[
        [N_RANKS, T_DYN, WIN], pl.INT32
    ],
    csa_window_swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    csa_cmp_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    csa_idx_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    csa_state_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    csa_inner_state_slot_mapping: pl.Tensor[
        [N_RANKS, T_DYN], pl.INT64
    ],
    csa_kv_seq_lens: pl.Tensor[[N_RANKS, CSA_B_DYN], pl.INT32],
    hca_cmp_freqs_cos: pl.Tensor[
        [N_RANKS, HCA_B, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    hca_cmp_freqs_sin: pl.Tensor[
        [N_RANKS, HCA_B, ROPE_HEAD_DIM // 2], pl.FP32
    ],
    hca_cmp_wkv: pl.Tensor[
        [N_RANKS, FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16
    ],
    hca_cmp_wgate: pl.Tensor[
        [N_RANKS, FWD_HCA_WEIGHT_BANK_SIZE * HCA_MAIN_OUT_DIM, D], pl.BF16
    ],
    hca_cmp_ape: pl.Tensor[
        [
            N_RANKS,
            FWD_HCA_WEIGHT_BANK_SIZE * HCA_COMPRESS_RATIO,
            HCA_MAIN_OUT_DIM,
        ],
        pl.FP32,
    ],
    hca_cmp_norm_w: pl.Tensor[
        [N_RANKS, FWD_HCA_WEIGHT_BANK_SIZE * HEAD_DIM], pl.BF16
    ],
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [
                N_RANKS,
                FWD_HCA_STATE_BLOCKS_DYN,
                HCA_COMPRESS_STATE_BLOCK_SIZE,
                HCA_COMPRESS_STATE_DIM,
            ],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[
        [N_RANKS, HCA_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32
    ],
    hca_cmp_kv: pl.InOut[
        pl.Tensor[
            [N_RANKS, FWD_HCA_CMP_BLOCKS_DYN, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    hca_cmp_block_table: pl.Tensor[
        [N_RANKS, HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32
    ],
    hca_ori_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    hca_window_swa_indices: pl.Tensor[
        [N_RANKS, T_DYN, WIN], pl.INT32
    ],
    hca_window_swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    hca_cmp_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    hca_state_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    hca_kv_seq_lens: pl.Tensor[[N_RANKS, HCA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * H], pl.FP32],
    wo_a: pl.Tensor[
        [
            N_RANKS,
            FWD_WEIGHT_BANK_SIZE * LOCAL_O_GROUPS,
            O_LORA,
            O_GROUP_IN,
        ],
        pl.BF16,
    ],
    wo_b: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * D, LOCAL_O_WIDTH], pl.INT8
    ],
    wo_b_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.FP32
    ],
    hc_ffn_fn: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * HC_FN_STORAGE_ROWS, HC_DIM],
        pl.FP32,
    ],
    hc_ffn_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * 3], pl.FP32
    ],
    hc_ffn_base: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * MIX_HC], pl.FP32
    ],
    norm_w: pl.Tensor[[N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.BF16],
    gate_w: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL, D], pl.FP32
    ],
    gate_bias: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_EXPERTS_GLOBAL], pl.FP32
    ],
    tid2eid: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * VOCAB, TOPK], pl.INT32
    ],
    input_ids: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    hc_head_fn: pl.Tensor[
        [N_RANKS, HC_MULT, HC_DIM], pl.FP32
    ],
    hc_head_scale: pl.Tensor[[N_RANKS, 1], pl.FP32],
    hc_head_base: pl.Tensor[[N_RANKS, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    lm_head_weight: pl.Tensor[
        [N_RANKS, VOCAB_PER_TP, D], pl.BF16
    ],
    logit_row_indices: pl.Tensor[
        [N_RANKS, MAX_LOGIT_ROWS], pl.INT32
    ],
    routed_w1: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * MOE_INTER, D], pl.INT8
    ],
    shared_w1_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32
    ],
    shared_w3: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * MOE_INTER, D], pl.INT8
    ],
    shared_w3_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * MOE_INTER], pl.FP32
    ],
    shared_w2: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * D, MOE_INTER], pl.INT8
    ],
    shared_w2_scale: pl.Tensor[
        [N_RANKS, FWD_WEIGHT_BANK_SIZE * D], pl.FP32
    ],
    hidden_workspace: pl.Out[
        pl.Tensor[[N_RANKS, T_DYN, D], pl.BF16]
    ],
    x_ping: pl.InOut[
        pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_pong: pl.InOut[
        pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_attn_active: pl.InOut[
        pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_moe_next: pl.InOut[
        pl.Tensor[[N_RANKS, MOE_TOKENS, HC_MULT, D], pl.FP32]
    ],
    pre_hc_hidden_out: pl.Out[
        pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_out: pl.Out[
        pl.Tensor[[N_RANKS, T_DYN, D], pl.BF16]
    ],
    logits: pl.Out[
        pl.Tensor[[N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32]
    ],
    sampled_ids: pl.Out[
        pl.Tensor[
            [N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32
        ]
    ],
):
    """Allocate each communication window once and submit one decode forward child."""
    embed_weight.bind_dynamic(1, EMBED_VOCAB_DYN)
    input_ids.bind_dynamic(1, T_DYN)
    hidden_workspace.bind_dynamic(1, T_DYN)
    x_ping.bind_dynamic(1, T_DYN)
    raw_kv_pool.bind_dynamic(1, FWD_PACKED_RAW_BLOCKS_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    swa_slot_mapping.bind_dynamic(1, T_DYN)
    swa_indices.bind_dynamic(1, T_DYN)
    swa_lens.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    csa_cmp_freqs_cos.bind_dynamic(1, T_DYN)
    csa_cmp_freqs_sin.bind_dynamic(1, T_DYN)
    csa_compress_state.bind_dynamic(1, FWD_CSA_MAIN_STATE_BLOCKS_DYN)
    csa_compress_state_block_table.bind_dynamic(1, CSA_B_DYN)
    csa_inner_compress_state.bind_dynamic(1, FWD_CSA_INNER_STATE_BLOCKS_DYN)
    csa_inner_compress_state_block_table.bind_dynamic(1, CSA_B_DYN)
    csa_cmp_kv.bind_dynamic(1, FWD_CSA_CMP_BLOCKS_DYN)
    csa_cmp_block_table.bind_dynamic(1, CSA_B_DYN)
    csa_idx_kv_cache.bind_dynamic(1, FWD_CSA_IDX_BLOCKS_DYN)
    csa_idx_kv_scale.bind_dynamic(1, FWD_CSA_IDX_BLOCKS_DYN)
    csa_idx_block_table.bind_dynamic(1, CSA_B_DYN)
    csa_ori_slot_mapping.bind_dynamic(1, T_DYN)
    csa_window_swa_indices.bind_dynamic(1, T_DYN)
    csa_window_swa_lens.bind_dynamic(1, T_DYN)
    csa_cmp_slot_mapping.bind_dynamic(1, T_DYN)
    csa_idx_slot_mapping.bind_dynamic(1, T_DYN)
    csa_state_slot_mapping.bind_dynamic(1, T_DYN)
    csa_inner_state_slot_mapping.bind_dynamic(1, T_DYN)
    csa_kv_seq_lens.bind_dynamic(1, CSA_B_DYN)
    hca_compress_state.bind_dynamic(1, FWD_HCA_STATE_BLOCKS_DYN)
    hca_compress_state_block_table.bind_dynamic(1, HCA_B_DYN)
    hca_cmp_kv.bind_dynamic(1, FWD_HCA_CMP_BLOCKS_DYN)
    hca_cmp_block_table.bind_dynamic(1, HCA_B_DYN)
    hca_cmp_block_table.bind_dynamic(2, HCA_CMP_TABLE_BLOCKS_DYN)
    hca_ori_slot_mapping.bind_dynamic(1, T_DYN)
    hca_window_swa_indices.bind_dynamic(1, T_DYN)
    hca_window_swa_lens.bind_dynamic(1, T_DYN)
    hca_cmp_slot_mapping.bind_dynamic(1, T_DYN)
    hca_state_slot_mapping.bind_dynamic(1, T_DYN)
    hca_kv_seq_lens.bind_dynamic(1, HCA_B_DYN)
    x_pong.bind_dynamic(1, T_DYN)
    x_attn_active.bind_dynamic(1, T_DYN)
    pre_hc_hidden_out.bind_dynamic(1, T_DYN)
    x_out.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.FP32)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    lm_head_hidden_window_buf = pld.alloc_window_buffer([GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
    lm_head_hidden_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
    lm_head_logits_window_buf = pld.alloc_window_buffer([MAX_LOGIT_ROWS * LM_HEAD_VOCAB], dtype=pl.FP32)
    lm_head_logits_done_buf = pld.alloc_window_buffer([LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        lm_head_hidden_window = pld.window(lm_head_hidden_window_buf, [GROUP_LOGIT_ROWS, D], dtype=pl.BF16)
        lm_head_hidden_done = pld.window(lm_head_hidden_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        lm_head_logits_window = pld.window(lm_head_logits_window_buf, [MAX_LOGIT_ROWS * LM_HEAD_VOCAB], dtype=pl.FP32)
        lm_head_logits_done = pld.window(lm_head_logits_done_buf, [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32)
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        decode_fwd(
            embed_weight[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank],
            wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            raw_kv_pool[rank], freqs_cos[rank], freqs_sin[rank],
            swa_slot_mapping[rank], swa_indices[rank], swa_lens[rank],
            position_ids[rank],
            csa_cmp_freqs_cos[rank], csa_cmp_freqs_sin[rank],
            csa_cmp_wkv[rank], csa_cmp_wgate[rank], csa_cmp_ape[rank],
            csa_cmp_norm_w[rank], csa_compress_state[rank],
            csa_compress_state_block_table[rank],
            csa_idx_wq_b[rank], csa_idx_wq_b_scale[rank],
            csa_weights_proj[rank], csa_hadamard_idx[rank],
            csa_inner_wkv[rank], csa_inner_wgate[rank],
            csa_inner_ape[rank], csa_inner_norm_w[rank],
            csa_inner_compress_state[rank],
            csa_inner_compress_state_block_table[rank],
            csa_cmp_kv[rank], csa_cmp_block_table[rank],
            csa_idx_kv_cache[rank], csa_idx_kv_scale[rank],
            csa_idx_block_table[rank], csa_ori_slot_mapping[rank],
            csa_window_swa_indices[rank], csa_window_swa_lens[rank],
            csa_cmp_slot_mapping[rank], csa_idx_slot_mapping[rank],
            csa_state_slot_mapping[rank],
            csa_inner_state_slot_mapping[rank], csa_kv_seq_lens[rank],
            hca_cmp_freqs_cos[rank], hca_cmp_freqs_sin[rank],
            hca_cmp_wkv[rank], hca_cmp_wgate[rank], hca_cmp_ape[rank],
            hca_cmp_norm_w[rank], hca_compress_state[rank],
            hca_compress_state_block_table[rank],
            hca_cmp_kv[rank], hca_cmp_block_table[rank],
            hca_ori_slot_mapping[rank], hca_window_swa_indices[rank],
            hca_window_swa_lens[rank], hca_cmp_slot_mapping[rank],
            hca_state_slot_mapping[rank], hca_kv_seq_lens[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank],
            wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            input_ids[rank],
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank],
            final_norm_w[rank], lm_head_weight[rank],
            logit_row_indices[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            hidden_workspace[rank],
            x_ping[rank], x_pong[rank],
            x_attn_active[rank], x_moe_next[rank],
            pre_hc_hidden_out[rank], x_out[rank], logits[rank],
            sampled_ids[rank],
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            lm_head_hidden_window, lm_head_hidden_done,
            lm_head_logits_window, lm_head_logits_done,
            group_base, tp_rank, rank,
            device=rank,
        )
    return x_out, logits, sampled_ids


_COMMON_ATTN_WEIGHT_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
)

_HCA_EXTRA_WEIGHT_NAMES = ("cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w")

_CSA_EXTRA_WEIGHT_NAMES = (
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
)

_LAYER_WEIGHT_NAMES = (
    *_COMMON_ATTN_WEIGHT_NAMES,
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base",
    "norm_w", "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale", "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
)

_SWA_METADATA_NAMES = ("freqs_cos", "freqs_sin", "swa_slot_mapping", "swa_indices", "swa_lens", "position_ids")

_CSA_SOURCES = {
    "csa_cmp_freqs_cos": "cmp_freqs_cos",
    "csa_cmp_freqs_sin": "cmp_freqs_sin",
    "csa_cmp_wkv": "cmp_wkv",
    "csa_cmp_wgate": "cmp_wgate",
    "csa_cmp_ape": "cmp_ape",
    "csa_cmp_norm_w": "cmp_norm_w",
    "csa_compress_state": "compress_state",
    "csa_compress_state_block_table": "compress_state_block_table",
    "csa_idx_wq_b": "idx_wq_b",
    "csa_idx_wq_b_scale": "idx_wq_b_scale",
    "csa_weights_proj": "weights_proj",
    "csa_hadamard_idx": "hadamard_idx",
    "csa_inner_wkv": "inner_wkv",
    "csa_inner_wgate": "inner_wgate",
    "csa_inner_ape": "inner_ape",
    "csa_inner_norm_w": "inner_norm_w",
    "csa_inner_compress_state": "inner_compress_state",
    "csa_inner_compress_state_block_table": "inner_compress_state_block_table",
    "csa_cmp_kv": "cmp_kv",
    "csa_cmp_block_table": "cmp_block_table",
    "csa_idx_kv_cache": "idx_kv_cache",
    "csa_idx_kv_scale": "idx_kv_scale",
    "csa_idx_block_table": "idx_block_table",
    "csa_ori_slot_mapping": "ori_slot_mapping",
    "csa_window_swa_indices": "window_swa_indices",
    "csa_window_swa_lens": "window_swa_lens",
    "csa_cmp_slot_mapping": "cmp_slot_mapping",
    "csa_idx_slot_mapping": "idx_slot_mapping",
    "csa_state_slot_mapping": "state_slot_mapping",
    "csa_inner_state_slot_mapping": "inner_state_slot_mapping",
    "csa_kv_seq_lens": "kv_seq_lens",
}

_HCA_SOURCES = {
    "hca_cmp_freqs_cos": "cmp_freqs_cos",
    "hca_cmp_freqs_sin": "cmp_freqs_sin",
    "hca_cmp_wkv": "cmp_wkv",
    "hca_cmp_wgate": "cmp_wgate",
    "hca_cmp_ape": "cmp_ape",
    "hca_cmp_norm_w": "cmp_norm_w",
    "hca_compress_state": "compress_state",
    "hca_compress_state_block_table": "compress_state_block_table",
    "hca_cmp_kv": "cmp_kv",
    "hca_cmp_block_table": "cmp_block_table",
    "hca_ori_slot_mapping": "ori_slot_mapping",
    "hca_window_swa_indices": "window_swa_indices",
    "hca_window_swa_lens": "window_swa_lens",
    "hca_cmp_slot_mapping": "cmp_slot_mapping",
    "hca_state_slot_mapping": "state_slot_mapping",
    "hca_kv_seq_lens": "kv_seq_lens",
}

def _copy_spec(name, source, *, is_output=None):
    from golden import TensorSpec

    output = source.is_output if is_output is None else is_output
    copied = TensorSpec(name, list(source.shape), source.dtype, init_value=source.init_value, is_output=output)
    copied.resident = source.resident
    return copied


def _make_weight_bank_spec(name, source, bank_size, *, compile_only=False):
    """Pack a static layer bank along the first rank-local data axis."""
    from golden import TensorSpec

    storage_shape = list(source.shape[1:])
    pad_hc_fn = name in {"hc_attn_fn", "hc_ffn_fn"}
    if pad_hc_fn:
        if storage_shape[0] != MIX_HC:
            raise ValueError(f"unexpected HC function shape for {name}")
        storage_shape[0] = HC_FN_STORAGE_ROWS
    shape = [N_RANKS, int(bank_size) * int(storage_shape[0]), *storage_shape[1:]]

    def init_value():
        import torch

        value = source.create_tensor()
        if pad_hc_fn:
            padded = torch.zeros(N_RANKS, HC_FN_STORAGE_ROWS, HC_DIM, dtype=value.dtype)
            padded[:, :MIX_HC].copy_(value)
            value = padded
        repeats = [1, int(bank_size)] + [1] * (value.ndim - 2)
        return value.repeat(*repeats).contiguous()

    initializer = 0 if compile_only else init_value
    spec = TensorSpec(name, shape, source.dtype, init_value=initializer)
    spec.resident = "stacked"
    return spec


def _make_packed_pool_spec(name, source, layer_count, *, sentinel=False):
    """Pack identical layer-local allocator pools along the block axis."""
    import torch
    from golden import TensorSpec

    shape = list(source.shape)
    shape[1] *= int(layer_count)

    def init_value():
        value = source.create_tensor()
        if not sentinel:
            repeats = [1, int(layer_count)] + [1] * (value.ndim - 2)
            return value.repeat(*repeats).contiguous()
        packed = torch.empty(shape, dtype=source.dtype)
        extent = int(source.shape[1])
        for ordinal in range(layer_count):
            packed[:, ordinal * extent : (ordinal + 1) * extent].fill_(ordinal + 1)
        return packed

    spec = TensorSpec(name, shape, source.dtype, init_value=init_value, is_output=True)
    spec.resident = "stacked"
    return spec


def build_tensor_specs(start_pos=None, *, weight_bank_size=RUNTIME_WEIGHT_BANK, runtime_case="full_active"):
    """Build the production or bounded-runtime decode forward L3 fixture."""
    import inspect

    import torch
    from golden import TensorSpec

    if weight_bank_size != FWD_WEIGHT_BANK_SIZE:
        raise ValueError(f"weight bank froze at module import as {FWD_WEIGHT_BANK_SIZE}, got {weight_bank_size}")
    compile_only = runtime_case is None
    if not compile_only and weight_bank_size != RUNTIME_WEIGHT_BANK:
        raise ValueError("decode forward runtime witnesses use one reusable weight bank")
    if runtime_case not in {
        None,
        "full_active",
        "packed_pool_sentinel",
        "long_context_tail",
    }:
        raise ValueError(f"unknown decode forward runtime case: {runtime_case!r}")

    if runtime_case == "long_context_tail" and start_pos is None:
        start_pos = [0, 0, 0, 1048568]

    if start_pos is None:
        active_batch = MOE_TOKENS // config.DECODE_SEQ
    elif isinstance(start_pos, int):
        active_batch = 1
    elif isinstance(start_pos, (list, tuple)) and start_pos:
        active_batch = len(start_pos)
    else:
        raise ValueError("start_pos must be None, an int, or a non-empty list/tuple")
    local_t = active_batch * config.DECODE_SEQ

    def attention_specs(module):
        specs = {}
        for source in module.build_distributed_tensor_specs(local_t, start_pos=start_pos):
            if not isinstance(source, TensorSpec):
                continue

            def init_value(source=source):
                value = source.create_tensor()
                repeats = [EP_SIZE // TP_SIZE] + [1] * (value.ndim - 1)
                return value.repeat(*repeats)

            spec = TensorSpec(
                source.name, [N_RANKS, *source.shape[1:]], source.dtype,
                init_value=init_value, is_output=source.is_output,
            )
            spec.resident = source.resident
            specs[spec.name] = spec
        return specs

    swa_specs = attention_specs(swa)
    csa_specs = attention_specs(csa)
    hca_specs = attention_specs(hca)
    for spec in moe_module.build_tensor_specs(layer_id=0, num_tokens=local_t):
        if isinstance(spec, TensorSpec) and spec.name not in {"x_hc", "x_next"}:
            swa_specs.setdefault(spec.name, spec)

    if int(csa_specs["freqs_cos"].shape[1]) != local_t:
        raise ValueError("CSA and SWA decode forward fixtures disagree on active rows")
    if int(hca_specs["freqs_cos"].shape[1]) != local_t:
        raise ValueError("HCA and SWA decode forward fixtures disagree on active rows")

    def zero_active():
        return torch.zeros(N_RANKS, local_t, HC_MULT, D, dtype=torch.float32)

    def init_input_ids():
        value = swa_specs["input_ids"].create_tensor()
        return torch.remainder(value[:, :local_t], RUNTIME_TEST_VOCAB).contiguous()

    embedding_vocab = MODEL_CONFIG.vocab_size if compile_only else RUNTIME_TEST_VOCAB

    def init_embed_weight():
        return (torch.randn(N_RANKS, embedding_vocab, D) * 0.05).to(torch.bfloat16)

    sentinel = runtime_case == "packed_pool_sentinel"
    specs_by_name = {
        "embed_weight": TensorSpec(
            "embed_weight", [N_RANKS, embedding_vocab, D], torch.bfloat16,
            init_value=0 if compile_only else init_embed_weight,
        ),
        "raw_kv_pool": _make_packed_pool_spec(
            "raw_kv_pool", swa_specs["kv_cache"], MAIN_LAYER_COUNT, sentinel=sentinel,
        ),
        "hca_compress_state": _make_packed_pool_spec(
            "hca_compress_state", hca_specs["compress_state"], HCA_LAYER_COUNT, sentinel=sentinel,
        ),
        "hca_cmp_kv": _make_packed_pool_spec("hca_cmp_kv", hca_specs["cmp_kv"], HCA_LAYER_COUNT, sentinel=sentinel),
        "csa_compress_state": _make_packed_pool_spec(
            "csa_compress_state", csa_specs["compress_state"], CSA_LAYER_COUNT, sentinel=sentinel,
        ),
        "csa_cmp_kv": _make_packed_pool_spec("csa_cmp_kv", csa_specs["cmp_kv"], CSA_LAYER_COUNT, sentinel=sentinel),
        "csa_inner_compress_state": _make_packed_pool_spec(
            "csa_inner_compress_state", csa_specs["inner_compress_state"], CSA_LAYER_COUNT, sentinel=sentinel,
        ),
        "csa_idx_kv_cache": _make_packed_pool_spec(
            "csa_idx_kv_cache", csa_specs["idx_kv_cache"], CSA_LAYER_COUNT, sentinel=sentinel,
        ),
        "csa_idx_kv_scale": _make_packed_pool_spec(
            "csa_idx_kv_scale", csa_specs["idx_kv_scale"], CSA_LAYER_COUNT, sentinel=sentinel,
        ),
        "input_ids": TensorSpec("input_ids", [N_RANKS, local_t], torch.int64, init_value=init_input_ids),
        "hc_head_fn": TensorSpec("hc_head_fn", [N_RANKS, HC_MULT, HC_DIM], torch.float32, init_value=0),
        "hc_head_scale": TensorSpec("hc_head_scale", [N_RANKS, 1], torch.float32, init_value=1.0),
        "hc_head_base": TensorSpec("hc_head_base", [N_RANKS, HC_MULT], torch.float32, init_value=0),
        "final_norm_w": TensorSpec("final_norm_w", [N_RANKS, D], torch.bfloat16, init_value=1.0),
        "lm_head_weight": TensorSpec("lm_head_weight", [N_RANKS, VOCAB_PER_TP, D], torch.bfloat16, init_value=0),
        "logit_row_indices": TensorSpec(
            "logit_row_indices", [N_RANKS, MAX_LOGIT_ROWS], torch.int32,
            init_value=lambda: build_active_logit_row_indices_host(local_t),
        ),
        "hidden_workspace": TensorSpec("hidden_workspace", [N_RANKS, local_t, D], torch.bfloat16, is_output=True),
        "x_ping": TensorSpec(
            "x_ping", [N_RANKS, local_t, HC_MULT, D], torch.float32,
            init_value=zero_active, is_output=True,
        ),
        "x_pong": TensorSpec(
            "x_pong", [N_RANKS, local_t, HC_MULT, D], torch.float32,
            init_value=zero_active, is_output=True,
        ),
        "x_attn_active": TensorSpec(
            "x_attn_active", [N_RANKS, local_t, HC_MULT, D], torch.float32,
            init_value=zero_active, is_output=True,
        ),
        "x_moe_next": TensorSpec(
            "x_moe_next", [N_RANKS, MOE_TOKENS, HC_MULT, D], torch.float32,
            init_value=lambda: torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32), is_output=True,
        ),
        "pre_hc_hidden_out": TensorSpec(
            "pre_hc_hidden_out", [N_RANKS, local_t, HC_MULT, D], torch.float32, is_output=True,
        ),
        "x_out": TensorSpec("x_out", [N_RANKS, local_t, D], torch.bfloat16, is_output=True),
        "logits": TensorSpec("logits", [N_RANKS, MAX_LOGIT_ROWS, LM_HEAD_VOCAB], torch.float32, is_output=True),
        "sampled_ids": TensorSpec(
            "sampled_ids", [N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], torch.int32, is_output=True,
        ),
    }

    for name in _LAYER_WEIGHT_NAMES:
        specs_by_name[name] = _make_weight_bank_spec(name, swa_specs[name], weight_bank_size, compile_only=compile_only)
    for name in _SWA_METADATA_NAMES:
        specs_by_name[name] = _copy_spec(name, swa_specs[name])

    csa_weight_names = {f"csa_{name}": name for name in _CSA_EXTRA_WEIGHT_NAMES}
    hca_weight_names = {f"hca_{name}": name for name in _HCA_EXTRA_WEIGHT_NAMES}
    csa_bank_size = CSA_LAYER_COUNT if compile_only else RUNTIME_WEIGHT_BANK
    for public_name, source_name in csa_weight_names.items():
        specs_by_name[public_name] = _make_weight_bank_spec(
            public_name, csa_specs[source_name], csa_bank_size, compile_only=compile_only,
        )
    hca_bank_size = HCA_LAYER_COUNT if compile_only else RUNTIME_WEIGHT_BANK
    for public_name, source_name in hca_weight_names.items():
        specs_by_name[public_name] = _make_weight_bank_spec(
            public_name, hca_specs[source_name], hca_bank_size, compile_only=compile_only,
        )

    packed_names = set(PACKED_POOL_LAYER_COUNTS)
    for public_name, source_name in _CSA_SOURCES.items():
        if public_name in packed_names or public_name in csa_weight_names:
            continue
        specs_by_name[public_name] = _copy_spec(public_name, csa_specs[source_name])
    for public_name, source_name in _HCA_SOURCES.items():
        if public_name in packed_names or public_name in hca_weight_names:
            continue
        specs_by_name[public_name] = _copy_spec(public_name, hca_specs[source_name])

    parameter_names = list(inspect.signature(l3_decode_fwd._func).parameters)
    missing = [name for name in parameter_names if name not in specs_by_name]
    extra = [name for name in specs_by_name if name not in parameter_names]
    if missing or extra:
        raise ValueError(f"decode forward spec/signature mismatch: missing={missing}, extra={extra}")
    specs = [specs_by_name[name] for name in parameter_names]
    for spec in specs:
        if len(spec.shape) > MAX_PUBLIC_TENSOR_DIMS:
            message = f"decode forward tensor {spec.name!r} exceeds {MAX_PUBLIC_TENSOR_DIMS} dimensions: {spec.shape}"
            raise ValueError(message)
    return specs


def golden_decode_fwd(_tensors):
    """decode forward is a topology/isolation witness; layer math is gated earlier."""


def finite_tensor_compare(actual, _expected, **_kwargs):
    """Require a completed finite device result without duplicating 43 goldens."""
    import torch

    if actual.numel() == 0:
        return False, "    decode forward output is empty"
    if actual.is_floating_point() and not bool(torch.isfinite(actual).all()):
        return False, "    decode forward output contains NaN or Inf"
    return True, ""


def sampled_ids_compare(actual, _expected, **kwargs):
    """Validate active greedy ids and the inactive-row -1 contract."""
    import torch

    row_indices = kwargs.get("inputs", {}).get("logit_row_indices")
    if row_indices is None:
        return False, "    missing logit_row_indices input"
    active = row_indices >= 0
    inactive_values = actual.masked_select((~active).unsqueeze(-1).expand_as(actual))
    if inactive_values.numel() and not bool(torch.all(inactive_values == -1)):
        return False, "    inactive sampled-id rows are not -1"
    active_ids = actual[..., 0].masked_select(active)
    if active_ids.numel() and bool(((active_ids < 0) | (active_ids >= LM_HEAD_VOCAB)).any()):
        return False, "    active sampled token id is outside the vocabulary"
    return True, ""


def _packed_pool_compare(layer_mappings, layer_count, block_size, pool_name):
    """Check every packed layer slice and reject writes outside local slots."""

    def compare(actual, expected, **kwargs):
        import torch

        if actual.shape != expected.shape:
            message = f"    {pool_name} shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
            return False, message
        if actual.shape[1] % layer_count:
            return False, f"    {pool_name} packed extent is not divisible"
        if len(layer_mappings) != layer_count:
            return False, f"    {pool_name} mapping count diverged"
        inputs = kwargs.get("inputs", {})
        extent = actual.shape[1] // layer_count
        for ordinal, mapping_name in enumerate(layer_mappings):
            mapping = inputs.get(mapping_name)
            if mapping is None:
                return False, f"    missing input {mapping_name!r}"
            for rank in range(actual.shape[0]):
                begin = ordinal * extent
                actual_rows = actual[rank, begin : begin + extent].reshape(
                    extent * block_size, *actual.shape[3:]
                )
                expected_rows = expected[rank, begin : begin + extent].reshape(
                    extent * block_size, *expected.shape[3:]
                )
                slots = mapping[rank].to(dtype=torch.int64).reshape(-1)
                invalid = (slots < -1) | (slots >= actual_rows.shape[0])
                if bool(invalid.any()):
                    first = int(slots[invalid][0])
                    return False, f"    {pool_name} layer={ordinal} rank={rank} contains out-of-range slot {first}"
                slots = slots[slots >= 0]
                target = torch.zeros(actual_rows.shape[0], dtype=torch.bool, device=actual.device)
                if slots.numel():
                    target[slots] = True
                if not torch.equal(actual_rows[~target], expected_rows[~target]):
                    return False, f"    {pool_name} layer={ordinal} rank={rank} modified an unmapped row"
                selected = actual_rows[target]
                if selected.is_floating_point() and not bool(torch.isfinite(selected).all()):
                    return False, f"    {pool_name} layer={ordinal} rank={rank} wrote NaN or Inf"
                sentinel_value = ordinal + 1
                sentinel_initialized = bool(torch.all(expected_rows == sentinel_value))
                selected_unchanged = torch.equal(selected, expected_rows[target])
                if slots.numel() and sentinel_initialized and selected_unchanged:
                    return False, f"    {pool_name} layer={ordinal} rank={rank} did not update any mapped sentinel row"
        return True, ""

    return compare


def compare_functions():
    """Return the decode forward runtime isolation and completion comparators."""
    raw_mappings = ("swa_slot_mapping",) * SWA_LAYER_COUNT
    alternating_mappings = ("csa_ori_slot_mapping", "hca_ori_slot_mapping")
    raw_mappings += tuple(mapping for _ in range(HCA_LAYER_COUNT) for mapping in alternating_mappings)
    raw_mappings += ("csa_ori_slot_mapping",)
    csa_mappings = ("csa_state_slot_mapping",) * CSA_LAYER_COUNT
    csa_cmp_mappings = ("csa_cmp_slot_mapping",) * CSA_LAYER_COUNT
    csa_inner_mappings = ("csa_inner_state_slot_mapping",) * CSA_LAYER_COUNT
    csa_idx_mappings = ("csa_idx_slot_mapping",) * CSA_LAYER_COUNT
    hca_state_mappings = ("hca_state_slot_mapping",) * HCA_LAYER_COUNT
    hca_cmp_mappings = ("hca_cmp_slot_mapping",) * HCA_LAYER_COUNT
    finite_names = {
        "hidden_workspace", "x_ping", "x_pong", "x_attn_active",
        "x_moe_next", "pre_hc_hidden_out", "x_out", "logits",
    }
    compare = {name: finite_tensor_compare for name in finite_names}
    compare["sampled_ids"] = sampled_ids_compare
    compare.update({
        "raw_kv_pool": _packed_pool_compare(raw_mappings, MAIN_LAYER_COUNT, BLOCK_SIZE, "decode forward raw KV"),
        "hca_compress_state": _packed_pool_compare(
            hca_state_mappings, HCA_LAYER_COUNT, HCA_COMPRESS_STATE_BLOCK_SIZE, "decode forward HCA state",
        ),
        "hca_cmp_kv": _packed_pool_compare(
            hca_cmp_mappings, HCA_LAYER_COUNT, BLOCK_SIZE, "decode forward HCA compressed KV",
        ),
        "csa_compress_state": _packed_pool_compare(
            csa_mappings, CSA_LAYER_COUNT, CSA_MAIN_STATE_BLOCK_SIZE, "decode forward CSA main state",
        ),
        "csa_cmp_kv": _packed_pool_compare(
            csa_cmp_mappings, CSA_LAYER_COUNT, BLOCK_SIZE, "decode forward CSA compressed KV",
        ),
        "csa_inner_compress_state": _packed_pool_compare(
            csa_inner_mappings, CSA_LAYER_COUNT, CSA_INNER_STATE_BLOCK_SIZE, "decode forward CSA inner state",
        ),
        "csa_idx_kv_cache": _packed_pool_compare(
            csa_idx_mappings, CSA_LAYER_COUNT, BLOCK_SIZE, "decode forward CSA index KV",
        ),
        "csa_idx_kv_scale": _packed_pool_compare(
            csa_idx_mappings, CSA_LAYER_COUNT, BLOCK_SIZE, "decode forward CSA index scale",
        ),
    })
    return compare


def _parse_start_pos(raw):
    if raw is None:
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--start-pos must contain at least one integer")
    return values[0] if len(values) == 1 else values


def main():
    import argparse

    from golden import run_jit
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser(description="DeepSeek-V4 D-Spark decode-forward integration")
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=("a2a3", "a2a3sim", "a5", "a5sim"))
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=_TP_CHOICES)
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=_EP_CHOICES)
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help=f"comma-separated device ids; EP={EP_SIZE} needs {EP_SIZE}",
    )
    parser.add_argument(
        "--start-pos", type=str, default=None,
        help="a scalar selects batch=1; a comma-separated list sets the batch",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--weight-bank-size", type=int, default=FWD_WEIGHT_BANK_SIZE, choices=(1, MAIN_LAYER_COUNT),
        help="1 reuses weights at runtime; 43 builds production layer banks",
    )
    parser.add_argument(
        "--runtime-case", type=str, default="full_active",
        choices=("full_active", "packed_pool_sentinel", "long_context_tail"),
    )
    parser.add_argument("--enable-scope-stats", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    if args.tp != TP_SIZE or args.ep != EP_SIZE:
        parser.error(f"parallel sizes froze at import as TP={TP_SIZE}, EP={EP_SIZE}")
    start_pos = _parse_start_pos(args.start_pos)
    weight_bank_size = args.weight_bank_size
    if weight_bank_size != FWD_WEIGHT_BANK_SIZE:
        parser.error(f"weight bank froze at import as {FWD_WEIGHT_BANK_SIZE}, got {weight_bank_size}")
    if not args.compile_only and weight_bank_size != 1:
        parser.error("decode forward runtime requires --weight-bank-size 1")

    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(EP_SIZE))
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if len(device_ids) != EP_SIZE:
        parser.error(f"EP={EP_SIZE} needs exactly {EP_SIZE} devices, got {device_ids}")
    if len(set(device_ids)) != len(device_ids) or any(device < 0 for device in device_ids):
        parser.error(f"device IDs must be distinct and non-negative: {device_ids}")

    runtime_case = None if weight_bank_size == MAIN_LAYER_COUNT else args.runtime_case
    specs = build_tensor_specs(start_pos=start_pos, weight_bank_size=weight_bank_size, runtime_case=runtime_case)
    result = run_jit(
        fn=l3_decode_fwd,
        specs=specs,
        golden_fn=golden_decode_fwd,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids, num_sub_workers=0
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_scope_stats=args.enable_scope_stats,
            log_level=args.log_level,
            ring_task_window=DECODE_RING_TASK_WINDOW,
            ring_heap=DECODE_RING_HEAP,
            ring_dep_pool=DECODE_RING_DEP_POOL,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn=compare_functions(),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
