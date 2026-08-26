# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 D-Spark 43-layer decode-forward integration."""

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


# TP/EP-dependent leaf shapes freeze at import time. Select both worlds before
# importing attention, MoE, or output-projection modules.
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
import decode_layer as layer
import decode_swa as swa
import moe as moe_module
import pypto.language as pl
import pypto.language.distributed as pld
from decode_layer import decode_layer_csa, decode_layer_hca, decode_layer_swa
from lookup_embedding import VOCAB_DYN as EMBED_VOCAB_DYN
from lookup_embedding import lookup_embedding
from moe import clear_moe_signals


MODEL_CONFIG = config.FLASH
DECODE_TOKENS = config.DECODE_TOKENS

MAIN_LAYER_COUNT = MODEL_CONFIG.num_hidden_layers
SWA_LAYER_COUNT = 2
CSA_LAYER_COUNT = 21
HCA_LAYER_COUNT = 20
LAST_MODEL_LAYER = MAIN_LAYER_COUNT - 1
MAX_PUBLIC_TENSOR_DIMS = 5

N_RANKS = layer.N_RANKS
MOE_TOKENS = layer.MOE_TOKENS
D = layer.D
HC_MULT = layer.HC_MULT
HC_DIM = layer.HC_DIM
MAX_LOGIT_ROWS = DECODE_TOKENS
SAMPLED_IDS_PAD = 8
VOCAB_PER_TP = MODEL_CONFIG.vocab_size // TP_SIZE

T_DYN = layer.T_DYN
FWD_PACKED_RAW_BLOCKS_DYN = pl.dynamic("FWD_PACKED_RAW_BLOCKS_DYN")
FWD_HCA_STATE_BLOCKS_DYN = pl.dynamic("FWD_HCA_STATE_BLOCKS_DYN")
FWD_HCA_CMP_BLOCKS_DYN = pl.dynamic("FWD_HCA_CMP_BLOCKS_DYN")
FWD_CSA_MAIN_STATE_BLOCKS_DYN = pl.dynamic("FWD_CSA_MAIN_STATE_BLOCKS_DYN")
FWD_CSA_CMP_BLOCKS_DYN = pl.dynamic("FWD_CSA_CMP_BLOCKS_DYN")
FWD_CSA_INNER_STATE_BLOCKS_DYN = pl.dynamic("FWD_CSA_INNER_STATE_BLOCKS_DYN")
FWD_CSA_IDX_BLOCKS_DYN = pl.dynamic("FWD_CSA_IDX_BLOCKS_DYN")

MIX_HC = layer.MIX_HC
Q_LORA = layer.Q_LORA
H = layer.H
HEAD_DIM = layer.HEAD_DIM
ROPE_HEAD_DIM = layer.ROPE_HEAD_DIM
WIN = layer.WIN
O_LORA = layer.O_LORA
O_GROUP_IN = layer.O_GROUP_IN
LOCAL_O_GROUPS = layer.LOCAL_O_GROUPS
LOCAL_O_WIDTH = layer.LOCAL_O_WIDTH
BLOCK_SIZE = layer.BLOCK_SIZE
ATTENTION_WINDOW_ROWS = layer.ATTENTION_WINDOW_ROWS
O_WINDOW_ROWS = layer.O_WINDOW_ROWS
N_EXPERTS_GLOBAL = layer.N_EXPERTS_GLOBAL
N_LOCAL = layer.N_LOCAL
MOE_INTER = layer.MOE_INTER
VOCAB = layer.VOCAB
TOPK = layer.TOPK
RECV_MAX = layer.RECV_MAX
AUX_PAD = layer.AUX_PAD
IDX_PAD = layer.IDX_PAD
N_ROUTES = layer.N_ROUTES

HCA_B_DYN = layer.HCA_B_DYN
HCA_CMP_TABLE_BLOCKS_DYN = layer.HCA_CMP_TABLE_BLOCKS_DYN
HCA_B = layer.HCA_B
HCA_MAIN_OUT_DIM = layer.HCA_MAIN_OUT_DIM
HCA_COMPRESS_RATIO = layer.HCA_COMPRESS_RATIO
HCA_COMPRESS_STATE_BLOCK_SIZE = layer.HCA_COMPRESS_STATE_BLOCK_SIZE
HCA_COMPRESS_STATE_MAX_BLOCKS = layer.HCA_COMPRESS_STATE_MAX_BLOCKS
HCA_COMPRESS_STATE_DIM = layer.HCA_COMPRESS_STATE_DIM

CSA_B_DYN = layer.CSA_B_DYN
CSA_MAIN_OUT_DIM = layer.CSA_MAIN_OUT_DIM
CSA_COMPRESS_RATIO = layer.CSA_COMPRESS_RATIO
CSA_MAIN_STATE_BLOCK_SIZE = layer.CSA_MAIN_STATE_BLOCK_SIZE
CSA_MAIN_STATE_MAX_BLOCKS = layer.CSA_MAIN_STATE_MAX_BLOCKS
CSA_MAIN_STATE_DIM = layer.CSA_MAIN_STATE_DIM
CSA_IDX_N_HEADS = layer.CSA_IDX_N_HEADS
CSA_IDX_HEAD_DIM = layer.CSA_IDX_HEAD_DIM
CSA_INNER_OUT_DIM = layer.CSA_INNER_OUT_DIM
CSA_INNER_STATE_BLOCK_SIZE = layer.CSA_INNER_STATE_BLOCK_SIZE
CSA_INNER_STATE_MAX_BLOCKS = layer.CSA_INNER_STATE_MAX_BLOCKS
CSA_INNER_STATE_DIM = layer.CSA_INNER_STATE_DIM
CSA_CMP_MAX_BLOCKS = layer.CSA_CMP_MAX_BLOCKS
CSA_IDX_MAX_BLOCKS = layer.CSA_IDX_MAX_BLOCKS

MIXED_PREFIX_LAYER_COUNT = 4
MIXED_PREFIX_CSA_LAYER_COUNT = 1
MIXED_PREFIX_HCA_LAYER_COUNT = 1
MIXED_PREFIX_TEST_VOCAB = 256
FULL43_RUNTIME_WEIGHT_BANK = 1
HC_FN_STORAGE_ROWS = 32


def _validate_import_contract():
    if layer.TP_SIZE != TP_SIZE or layer.EP_SIZE != EP_SIZE:
        raise ValueError(
            "decode_layer parallel configuration diverged from decode_fwd: "
            f"layer TP/EP={layer.TP_SIZE}/{layer.EP_SIZE}, "
            f"forward TP/EP={TP_SIZE}/{EP_SIZE}"
        )
    if N_RANKS != EP_SIZE:
        raise ValueError(f"MoE world size {N_RANKS} does not match EP={EP_SIZE}")
    if MAIN_LAYER_COUNT != 43:
        raise ValueError(
            f"D-Spark decode forward expects 43 layers, got {MAIN_LAYER_COUNT}"
        )
    if MODEL_CONFIG.vocab_size % TP_SIZE:
        raise ValueError(
            f"vocab size {MODEL_CONFIG.vocab_size} must be divisible by TP={TP_SIZE}"
        )
    if MOE_TOKENS > MAX_LOGIT_ROWS:
        raise ValueError(
            f"MoE capacity {MOE_TOKENS} exceeds LM-head rows {MAX_LOGIT_ROWS}"
        )


_validate_import_contract()


def build_full43_layer_plan():
    """Return the fixed model-layer, kind-ordinal, and MoE-epoch mapping."""
    kind_ordinals = {"swa": 0, "csa": 0, "hca": 0}
    plan = []
    for model_layer in range(MAIN_LAYER_COUNT):
        kind = layer.attention_kind_for_layer(model_layer)
        ordinal = kind_ordinals[kind]
        kind_ordinals[kind] += 1
        plan.append(
            {
                "model_layer": model_layer,
                "kind": kind,
                "kind_ordinal": ordinal,
                "moe_epoch": model_layer + 1,
            }
        )

    expected_counts = {
        "swa": SWA_LAYER_COUNT,
        "csa": CSA_LAYER_COUNT,
        "hca": HCA_LAYER_COUNT,
    }
    if kind_ordinals != expected_counts:
        raise ValueError(
            f"full43 attention counts changed: {kind_ordinals} != {expected_counts}"
        )
    if plan[0]["kind"] != "swa" or plan[1]["kind"] != "swa":
        raise ValueError("model layers 0 and 1 must be SWA")
    for model_layer in range(2, MAIN_LAYER_COUNT):
        expected_kind = "csa" if model_layer % 2 == 0 else "hca"
        if plan[model_layer]["kind"] != expected_kind:
            raise ValueError(
                f"model layer {model_layer} must be {expected_kind}, "
                f"got {plan[model_layer]['kind']}"
            )
    if plan[LAST_MODEL_LAYER] != {
        "model_layer": 42,
        "kind": "csa",
        "kind_ordinal": 20,
        "moe_epoch": 43,
    }:
        raise ValueError(f"unexpected terminal layer mapping: {plan[-1]}")
    return tuple(plan)


FULL43_LAYER_PLAN = build_full43_layer_plan()


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

PACKED_POOL_BLOCK_SIZES = {
    "raw_kv_pool": swa.BLOCK_SIZE,
    "hca_compress_state": hca.COMPRESS_STATE_BLOCK_SIZE,
    "hca_cmp_kv": hca.BLOCK_SIZE,
    "csa_compress_state": csa.MAIN_STATE_BLOCK_SIZE,
    "csa_cmp_kv": csa.BLOCK_SIZE,
    "csa_inner_compress_state": csa.INNER_STATE_BLOCK_SIZE,
    "csa_idx_kv_cache": csa.BLOCK_SIZE,
    "csa_idx_kv_scale": csa.BLOCK_SIZE,
}

PACKED_POOL_ELEMENT_BYTES = {
    "raw_kv_pool": 2,
    "hca_compress_state": 4,
    "hca_cmp_kv": 2,
    "csa_compress_state": 4,
    "csa_cmp_kv": 2,
    "csa_inner_compress_state": 4,
    "csa_idx_kv_cache": 1,
    "csa_idx_kv_scale": 4,
}


def build_packed_pool_layout(per_layer_shapes):
    """Pack each persistent pool along its block axis without a layer axis."""
    expected_names = set(PACKED_POOL_LAYER_COUNTS)
    provided_names = set(per_layer_shapes)
    missing = expected_names - provided_names
    unknown = provided_names - expected_names
    if missing or unknown:
        raise ValueError(
            f"packed pool names mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )

    layout = {}
    for name, layer_count in PACKED_POOL_LAYER_COUNTS.items():
        leaf_shape = [int(dim) for dim in per_layer_shapes[name]]
        if len(leaf_shape) not in (4, 5):
            raise ValueError(
                f"{name} rank-stacked leaf pool must be 4D or 5D, "
                f"got {leaf_shape}"
            )
        if leaf_shape[0] != N_RANKS:
            raise ValueError(
                f"{name} must have leading rank extent {N_RANKS}, "
                f"got {leaf_shape}"
            )
        per_layer_extent = leaf_shape[1]
        if per_layer_extent <= 0:
            raise ValueError(
                f"{name} per-layer block extent must be positive, "
                f"got {per_layer_extent}"
            )
        expected_block_size = PACKED_POOL_BLOCK_SIZES[name]
        if leaf_shape[2] != expected_block_size:
            raise ValueError(
                f"{name} block size {leaf_shape[2]} does not match "
                f"{expected_block_size}"
            )

        packed_shape = list(leaf_shape)
        packed_shape[1] = layer_count * per_layer_extent
        elements_per_rank = 1
        for dim in packed_shape[1:]:
            elements_per_rank *= dim
        slices = tuple(
            (ordinal * per_layer_extent, (ordinal + 1) * per_layer_extent)
            for ordinal in range(layer_count)
        )
        layout[name] = {
            "layer_count": layer_count,
            "per_layer_extent": per_layer_extent,
            "total_extent": packed_shape[1],
            "block_size": expected_block_size,
            "leaf_shape": leaf_shape,
            "packed_shape": packed_shape,
            "bytes_per_rank": (
                elements_per_rank * PACKED_POOL_ELEMENT_BYTES[name]
            ),
            "slices": slices,
        }
    validate_packed_pool_layout(layout)
    return layout


def validate_packed_pool_layout(layout):
    """Fail closed on dimension, extent, and layer-slice contract errors."""
    if set(layout) != set(PACKED_POOL_LAYER_COUNTS):
        raise ValueError("packed layout must describe every persistent pool")
    for name, entry in layout.items():
        layer_count = PACKED_POOL_LAYER_COUNTS[name]
        total_extent = int(entry["total_extent"])
        if total_extent % layer_count:
            raise ValueError(
                f"{name} packed extent {total_extent} is not divisible by "
                f"{layer_count}"
            )
        if len(entry["packed_shape"]) > MAX_PUBLIC_TENSOR_DIMS:
            raise ValueError(
                f"{name} has {len(entry['packed_shape'])} dimensions; "
                f"maximum is {MAX_PUBLIC_TENSOR_DIMS}"
            )
        if entry["packed_shape"][1] != total_extent:
            raise ValueError(f"{name} packed shape and extent diverged")

        previous_end = 0
        for begin, end in entry["slices"]:
            if begin != previous_end or begin >= end or end > total_extent:
                raise ValueError(f"{name} layer slices overlap or leave a gap")
            previous_end = end
        if previous_end != total_extent:
            raise ValueError(f"{name} layer slices do not cover the packed axis")
    return layout


def validate_local_write_slots(name, slots, per_layer_capacity, active_mask=None):
    """Validate layer-local write slots and the inactive-row -1 contract."""
    import torch

    values = torch.as_tensor(slots)
    capacity = int(per_layer_capacity)
    if capacity <= 0:
        raise ValueError(f"{name} capacity must be positive, got {capacity}")
    if bool((values < -1).any().item()):
        raise ValueError(f"{name} contains a write slot below -1")
    if bool((values >= capacity).any().item()):
        raise ValueError(
            f"{name} contains a non-local write slot for capacity {capacity}"
        )
    if active_mask is not None:
        active = torch.as_tensor(active_mask, dtype=torch.bool)
        while active.ndim < values.ndim:
            active = active.unsqueeze(-1)
        if bool((values.masked_select(~active) != -1).any().item()):
            raise ValueError(f"{name} retains a write slot for an inactive row")
    return values


def build_active_logit_row_indices_host(active_tokens):
    """Build the host fixture for the terminal active-prefix row contract."""
    import torch

    active_tokens = int(active_tokens)
    if active_tokens < 0 or active_tokens > min(MOE_TOKENS, MAX_LOGIT_ROWS):
        raise ValueError(
            f"active token count must be in [0, {min(MOE_TOKENS, MAX_LOGIT_ROWS)}], "
            f"got {active_tokens}"
        )
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
):
    """Reuse the canonical lookup and emit the active Hyper-Connections rows."""
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
            sampled_ids[row : row + 1, :] = pl.full(
                [1, SAMPLED_IDS_PAD], dtype=pl.INT32, value=-1
            )
    return sampled_ids


@pl.jit(auto_scope=False)
def decode_fwd(
    hc_attn_fn: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * 3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * D], pl.BF16],
    wq_a: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * Q_LORA, H * HEAD_DIM], pl.INT8
    ],
    wq_b_scale: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * H * HEAD_DIM], pl.FP32
    ],
    wkv: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * HEAD_DIM], pl.BF16],
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
    csa_cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[
        [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32
    ],
    csa_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
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
        [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32
    ],
    csa_weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    csa_hadamard_idx: pl.Tensor[
        [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16
    ],
    csa_inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[
        [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32
    ],
    csa_inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
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
    hca_cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[
        [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32
    ],
    hca_cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
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
    attn_sink: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * H], pl.FP32],
    wo_a: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
        pl.BF16,
    ],
    wo_b: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * D, LOCAL_O_WIDTH], pl.INT8
    ],
    wo_b_scale: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * D], pl.FP32],
    hc_ffn_fn: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * MIX_HC, HC_DIM], pl.FP32
    ],
    hc_ffn_scale: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * D], pl.BF16],
    gate_w: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_EXPERTS_GLOBAL, D], pl.FP32
    ],
    gate_bias: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_EXPERTS_GLOBAL], pl.FP32
    ],
    tid2eid: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
    routed_w1: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * MOE_INTER, D], pl.INT8
    ],
    shared_w1_scale: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * MOE_INTER], pl.FP32
    ],
    shared_w3: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * MOE_INTER, D], pl.INT8
    ],
    shared_w3_scale: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * MOE_INTER], pl.FP32
    ],
    shared_w2: pl.Tensor[
        [MIXED_PREFIX_LAYER_COUNT * D, MOE_INTER], pl.INT8
    ],
    shared_w2_scale: pl.Tensor[[MIXED_PREFIX_LAYER_COUNT * D], pl.FP32],
    x_ping: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_pong: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_attn_active: pl.InOut[
        pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_moe_next: pl.InOut[
        pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]
    ],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
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
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run SWA, SWA, CSA, and HCA with shared windows and ping-pong."""
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
    csa_inner_compress_state.bind_dynamic(
        0, FWD_CSA_INNER_STATE_BLOCKS_DYN
    )
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
    x_out.bind_dynamic(0, T_DYN)

    local_t = pl.cast(pl.tensor.dim(x_ping, 0), pl.INT32)
    raw_blocks_per_layer = (
        pl.tensor.dim(raw_kv_pool, 0) // MIXED_PREFIX_LAYER_COUNT
    )
    with pl.scope():
        raw_kv_l0 = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [0, 0, 0, 0],
        )
        hc_attn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [0, 0]
        )
        hc_attn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [0]
        )
        hc_attn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [0]
        )
        attn_norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_w, [D], [0]
        )
        wq_a_l0: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [0, 0]
        )
        wq_b_l0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [0, 0]
        )
        wq_b_scale_l0: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [0]
        )
        wkv_l0: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [0, 0]
        )
        gamma_cq_l0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [0]
        )
        gamma_ckv_l0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [0]
        )
        attn_sink_l0: pl.Tensor[[H], pl.FP32] = pl.slice(
            attn_sink, [H], [0]
        )
        wo_a_l0: pl.Tensor[
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
        ] = pl.slice(
            wo_a, [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], [0, 0, 0]
        )
        wo_b_l0: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8] = pl.slice(
            wo_b, [D, LOCAL_O_WIDTH], [0, 0]
        )
        wo_b_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(
            wo_b_scale, [D], [0]
        )
        hc_ffn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [0, 0]
        )
        hc_ffn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [0]
        )
        hc_ffn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [0]
        )
        norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [0])
        gate_w_l0: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [0, 0]
        )
        gate_bias_l0: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS_GLOBAL], [0]
        )
        tid2eid_l0: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
            tid2eid, [VOCAB, TOPK], [0, 0]
        )
        routed_w1_l0: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [0, 0, 0])
        routed_w1_scale_l0: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [0, 0])
        routed_w3_l0: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [0, 0, 0])
        routed_w3_scale_l0: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [0, 0])
        routed_w2_l0: pl.Tensor[
            [N_LOCAL, D, MOE_INTER], pl.INT8
        ] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [0, 0, 0])
        routed_w2_scale_l0: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [0, 0]
        )
        shared_w1_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w1, [MOE_INTER, D], [0, 0]
        )
        shared_w1_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w1_scale, [MOE_INTER], [0]
        )
        shared_w3_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w3, [MOE_INTER, D], [0, 0]
        )
        shared_w3_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w3_scale, [MOE_INTER], [0]
        )
        shared_w2_l0: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(
            shared_w2, [D, MOE_INTER], [0, 0]
        )
        shared_w2_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(
            shared_w2_scale, [D], [0]
        )
        decode_layer_swa(
            x_ping,
            hc_attn_fn_l0, hc_attn_scale_l0, hc_attn_base_l0,
            attn_norm_w_l0, wq_a_l0, wq_b_l0, wq_b_scale_l0,
            wkv_l0, gamma_cq_l0, gamma_ckv_l0,
            freqs_cos, freqs_sin, raw_kv_l0,
            swa_slot_mapping, swa_indices, swa_lens, position_ids,
            attn_sink_l0, wo_a_l0, wo_b_l0, wo_b_scale_l0,
            hc_ffn_fn_l0, hc_ffn_scale_l0, hc_ffn_base_l0,
            norm_w_l0, gate_w_l0, gate_bias_l0, tid2eid_l0, input_ids,
            routed_w1_l0, routed_w1_scale_l0,
            routed_w3_l0, routed_w3_scale_l0,
            routed_w2_l0, routed_w2_scale_l0,
            shared_w1_l0, shared_w1_scale_l0,
            shared_w3_l0, shared_w3_scale_l0,
            shared_w2_l0, shared_w2_scale_l0,
            x_attn_active, x_moe_next, x_pong,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.const(0, pl.INT32), group_base, tp_rank, local_t, my_rank,
            pl.const(1, pl.INT32),
        )

    with pl.scope():
        raw_kv_l1 = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [raw_blocks_per_layer, 0, 0, 0],
        )
        hc_attn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [MIX_HC, 0]
        )
        hc_attn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [3]
        )
        hc_attn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [MIX_HC]
        )
        attn_norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_w, [D], [D]
        )
        wq_a_l1: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [D, 0]
        )
        wq_b_l1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [Q_LORA, 0]
        )
        wq_b_scale_l1: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [H * HEAD_DIM]
        )
        wkv_l1: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [D, 0]
        )
        gamma_cq_l1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [Q_LORA]
        )
        gamma_ckv_l1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [HEAD_DIM]
        )
        attn_sink_l1: pl.Tensor[[H], pl.FP32] = pl.slice(
            attn_sink, [H], [H]
        )
        wo_a_l1: pl.Tensor[
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
        ] = pl.slice(
            wo_a,
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
            [LOCAL_O_GROUPS, 0, 0],
        )
        wo_b_l1: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8] = pl.slice(
            wo_b, [D, LOCAL_O_WIDTH], [D, 0]
        )
        wo_b_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(
            wo_b_scale, [D], [D]
        )
        hc_ffn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [MIX_HC, 0]
        )
        hc_ffn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [3]
        )
        hc_ffn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [MIX_HC]
        )
        norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [D])
        gate_w_l1: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [N_EXPERTS_GLOBAL, 0]
        )
        gate_bias_l1: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS_GLOBAL], [N_EXPERTS_GLOBAL]
        )
        tid2eid_l1: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
            tid2eid, [VOCAB, TOPK], [VOCAB, 0]
        )
        routed_w1_l1: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [N_LOCAL, 0, 0])
        routed_w1_scale_l1: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [N_LOCAL, 0])
        routed_w3_l1: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [N_LOCAL, 0, 0])
        routed_w3_scale_l1: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [N_LOCAL, 0])
        routed_w2_l1: pl.Tensor[
            [N_LOCAL, D, MOE_INTER], pl.INT8
        ] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [N_LOCAL, 0, 0])
        routed_w2_scale_l1: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [N_LOCAL, 0]
        )
        shared_w1_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w1, [MOE_INTER, D], [MOE_INTER, 0]
        )
        shared_w1_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w1_scale, [MOE_INTER], [MOE_INTER]
        )
        shared_w3_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w3, [MOE_INTER, D], [MOE_INTER, 0]
        )
        shared_w3_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w3_scale, [MOE_INTER], [MOE_INTER]
        )
        shared_w2_l1: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(
            shared_w2, [D, MOE_INTER], [D, 0]
        )
        shared_w2_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(
            shared_w2_scale, [D], [D]
        )
        decode_layer_swa(
            x_pong,
            hc_attn_fn_l1, hc_attn_scale_l1, hc_attn_base_l1,
            attn_norm_w_l1, wq_a_l1, wq_b_l1, wq_b_scale_l1,
            wkv_l1, gamma_cq_l1, gamma_ckv_l1,
            freqs_cos, freqs_sin, raw_kv_l1,
            swa_slot_mapping, swa_indices, swa_lens, position_ids,
            attn_sink_l1, wo_a_l1, wo_b_l1, wo_b_scale_l1,
            hc_ffn_fn_l1, hc_ffn_scale_l1, hc_ffn_base_l1,
            norm_w_l1, gate_w_l1, gate_bias_l1, tid2eid_l1, input_ids,
            routed_w1_l1, routed_w1_scale_l1,
            routed_w3_l1, routed_w3_scale_l1,
            routed_w2_l1, routed_w2_scale_l1,
            shared_w1_l1, shared_w1_scale_l1,
            shared_w3_l1, shared_w3_scale_l1,
            shared_w2_l1, shared_w2_scale_l1,
            x_attn_active, x_moe_next, x_ping,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.const(1, pl.INT32), group_base, tp_rank, local_t, my_rank,
            pl.const(2, pl.INT32),
        )
    csa_state_blocks_per_layer = (
        pl.tensor.dim(csa_compress_state, 0) // MIXED_PREFIX_CSA_LAYER_COUNT
    )
    csa_cmp_blocks_per_layer = (
        pl.tensor.dim(csa_cmp_kv, 0) // MIXED_PREFIX_CSA_LAYER_COUNT
    )
    csa_inner_state_blocks_per_layer = (
        pl.tensor.dim(csa_inner_compress_state, 0)
        // MIXED_PREFIX_CSA_LAYER_COUNT
    )
    csa_idx_blocks_per_layer = (
        pl.tensor.dim(csa_idx_kv_cache, 0) // MIXED_PREFIX_CSA_LAYER_COUNT
    )
    with pl.scope():
        raw_kv_l2 = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [2 * raw_blocks_per_layer, 0, 0, 0],
        )
        csa_compress_state_l0 = pl.slice(
            csa_compress_state,
            [
                csa_state_blocks_per_layer,
                CSA_MAIN_STATE_BLOCK_SIZE,
                CSA_MAIN_STATE_DIM,
            ],
            [0, 0, 0],
        )
        csa_cmp_kv_l0 = pl.slice(
            csa_cmp_kv,
            [csa_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [0, 0, 0, 0],
        )
        csa_inner_compress_state_l0 = pl.slice(
            csa_inner_compress_state,
            [
                csa_inner_state_blocks_per_layer,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            [0, 0, 0],
        )
        csa_idx_kv_cache_l0 = pl.slice(
            csa_idx_kv_cache,
            [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM],
            [0, 0, 0, 0],
        )
        csa_idx_kv_scale_l0 = pl.slice(
            csa_idx_kv_scale,
            [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, 1],
            [0, 0, 0, 0],
        )
        hc_attn_fn_l2: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [2 * MIX_HC, 0]
        )
        hc_attn_scale_l2: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [6]
        )
        hc_attn_base_l2: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [2 * MIX_HC]
        )
        attn_norm_w_l2: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_w, [D], [2 * D]
        )
        wq_a_l2: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [2 * D, 0]
        )
        wq_b_l2: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [2 * Q_LORA, 0]
        )
        wq_b_scale_l2: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [2 * H * HEAD_DIM]
        )
        wkv_l2: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [2 * D, 0]
        )
        gamma_cq_l2: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [2 * Q_LORA]
        )
        gamma_ckv_l2: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [2 * HEAD_DIM]
        )
        attn_sink_l2: pl.Tensor[[H], pl.FP32] = pl.slice(
            attn_sink, [H], [2 * H]
        )
        wo_a_l2: pl.Tensor[
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
        ] = pl.slice(
            wo_a,
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
            [2 * LOCAL_O_GROUPS, 0, 0],
        )
        wo_b_l2: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8] = pl.slice(
            wo_b, [D, LOCAL_O_WIDTH], [2 * D, 0]
        )
        wo_b_scale_l2: pl.Tensor[[D], pl.FP32] = pl.slice(
            wo_b_scale, [D], [2 * D]
        )
        hc_ffn_fn_l2: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [2 * MIX_HC, 0]
        )
        hc_ffn_scale_l2: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [6]
        )
        hc_ffn_base_l2: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [2 * MIX_HC]
        )
        norm_w_l2: pl.Tensor[[D], pl.BF16] = pl.slice(
            norm_w, [D], [2 * D]
        )
        gate_w_l2: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [2 * N_EXPERTS_GLOBAL, 0]
        )
        gate_bias_l2: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS_GLOBAL], [2 * N_EXPERTS_GLOBAL]
        )
        tid2eid_l2: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
            tid2eid, [VOCAB, TOPK], [2 * VOCAB, 0]
        )
        routed_w1_l2: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w1, [N_LOCAL, MOE_INTER, D], [2 * N_LOCAL, 0, 0]
        )
        routed_w1_scale_l2: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(
            routed_w1_scale, [N_LOCAL, MOE_INTER], [2 * N_LOCAL, 0]
        )
        routed_w3_l2: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w3, [N_LOCAL, MOE_INTER, D], [2 * N_LOCAL, 0, 0]
        )
        routed_w3_scale_l2: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(
            routed_w3_scale, [N_LOCAL, MOE_INTER], [2 * N_LOCAL, 0]
        )
        routed_w2_l2: pl.Tensor[
            [N_LOCAL, D, MOE_INTER], pl.INT8
        ] = pl.slice(
            routed_w2, [N_LOCAL, D, MOE_INTER], [2 * N_LOCAL, 0, 0]
        )
        routed_w2_scale_l2: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [2 * N_LOCAL, 0]
        )
        shared_w1_l2: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w1, [MOE_INTER, D], [2 * MOE_INTER, 0]
        )
        shared_w1_scale_l2: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w1_scale, [MOE_INTER], [2 * MOE_INTER]
        )
        shared_w3_l2: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w3, [MOE_INTER, D], [2 * MOE_INTER, 0]
        )
        shared_w3_scale_l2: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w3_scale, [MOE_INTER], [2 * MOE_INTER]
        )
        shared_w2_l2: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(
            shared_w2, [D, MOE_INTER], [2 * D, 0]
        )
        shared_w2_scale_l2: pl.Tensor[[D], pl.FP32] = pl.slice(
            shared_w2_scale, [D], [2 * D]
        )
        decode_layer_csa(
            x_ping,
            hc_attn_fn_l2, hc_attn_scale_l2, hc_attn_base_l2,
            attn_norm_w_l2, wq_a_l2, wq_b_l2, wq_b_scale_l2,
            wkv_l2, gamma_cq_l2, gamma_ckv_l2,
            freqs_cos, freqs_sin,
            csa_cmp_freqs_cos, csa_cmp_freqs_sin,
            csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
            csa_compress_state_l0, csa_compress_state_block_table,
            csa_idx_wq_b, csa_idx_wq_b_scale,
            csa_weights_proj, csa_hadamard_idx,
            csa_inner_wkv, csa_inner_wgate,
            csa_inner_ape, csa_inner_norm_w,
            csa_inner_compress_state_l0,
            csa_inner_compress_state_block_table,
            raw_kv_l2, csa_cmp_kv_l0, csa_cmp_block_table,
            csa_idx_kv_cache_l0, csa_idx_kv_scale_l0,
            csa_idx_block_table,
            csa_ori_slot_mapping, csa_window_swa_indices,
            csa_window_swa_lens, csa_cmp_slot_mapping,
            csa_idx_slot_mapping, csa_state_slot_mapping,
            csa_inner_state_slot_mapping, position_ids,
            csa_kv_seq_lens, attn_sink_l2,
            wo_a_l2, wo_b_l2, wo_b_scale_l2,
            hc_ffn_fn_l2, hc_ffn_scale_l2, hc_ffn_base_l2,
            norm_w_l2, gate_w_l2, gate_bias_l2, tid2eid_l2, input_ids,
            routed_w1_l2, routed_w1_scale_l2,
            routed_w3_l2, routed_w3_scale_l2,
            routed_w2_l2, routed_w2_scale_l2,
            shared_w1_l2, shared_w1_scale_l2,
            shared_w3_l2, shared_w3_scale_l2,
            shared_w2_l2, shared_w2_scale_l2,
            x_attn_active, x_moe_next, x_pong,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.const(2, pl.INT32), group_base, tp_rank, local_t, my_rank,
            pl.const(3, pl.INT32),
        )

    hca_state_blocks_per_layer = (
        pl.tensor.dim(hca_compress_state, 0) // MIXED_PREFIX_HCA_LAYER_COUNT
    )
    hca_cmp_blocks_per_layer = (
        pl.tensor.dim(hca_cmp_kv, 0) // MIXED_PREFIX_HCA_LAYER_COUNT
    )
    with pl.scope():
        raw_kv_l3 = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [3 * raw_blocks_per_layer, 0, 0, 0],
        )
        hca_compress_state_l0 = pl.slice(
            hca_compress_state,
            [
                hca_state_blocks_per_layer,
                HCA_COMPRESS_STATE_BLOCK_SIZE,
                HCA_COMPRESS_STATE_DIM,
            ],
            [0, 0, 0],
        )
        hca_cmp_kv_l0 = pl.slice(
            hca_cmp_kv,
            [hca_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [0, 0, 0, 0],
        )
        hc_attn_fn_l3: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [3 * MIX_HC, 0]
        )
        hc_attn_scale_l3: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_attn_scale, [3], [9]
        )
        hc_attn_base_l3: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_attn_base, [MIX_HC], [3 * MIX_HC]
        )
        attn_norm_w_l3: pl.Tensor[[D], pl.BF16] = pl.slice(
            attn_norm_w, [D], [3 * D]
        )
        wq_a_l3: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [3 * D, 0]
        )
        wq_b_l3: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [3 * Q_LORA, 0]
        )
        wq_b_scale_l3: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [3 * H * HEAD_DIM]
        )
        wkv_l3: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [3 * D, 0]
        )
        gamma_cq_l3: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [3 * Q_LORA]
        )
        gamma_ckv_l3: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [3 * HEAD_DIM]
        )
        attn_sink_l3: pl.Tensor[[H], pl.FP32] = pl.slice(
            attn_sink, [H], [3 * H]
        )
        wo_a_l3: pl.Tensor[
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
        ] = pl.slice(
            wo_a,
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
            [3 * LOCAL_O_GROUPS, 0, 0],
        )
        wo_b_l3: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8] = pl.slice(
            wo_b, [D, LOCAL_O_WIDTH], [3 * D, 0]
        )
        wo_b_scale_l3: pl.Tensor[[D], pl.FP32] = pl.slice(
            wo_b_scale, [D], [3 * D]
        )
        hc_ffn_fn_l3: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [3 * MIX_HC, 0]
        )
        hc_ffn_scale_l3: pl.Tensor[[3], pl.FP32] = pl.slice(
            hc_ffn_scale, [3], [9]
        )
        hc_ffn_base_l3: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(
            hc_ffn_base, [MIX_HC], [3 * MIX_HC]
        )
        norm_w_l3: pl.Tensor[[D], pl.BF16] = pl.slice(
            norm_w, [D], [3 * D]
        )
        gate_w_l3: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [3 * N_EXPERTS_GLOBAL, 0]
        )
        gate_bias_l3: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(
            gate_bias, [N_EXPERTS_GLOBAL], [3 * N_EXPERTS_GLOBAL]
        )
        tid2eid_l3: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(
            tid2eid, [VOCAB, TOPK], [3 * VOCAB, 0]
        )
        routed_w1_l3: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w1, [N_LOCAL, MOE_INTER, D], [3 * N_LOCAL, 0, 0]
        )
        routed_w1_scale_l3: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(
            routed_w1_scale, [N_LOCAL, MOE_INTER], [3 * N_LOCAL, 0]
        )
        routed_w3_l3: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w3, [N_LOCAL, MOE_INTER, D], [3 * N_LOCAL, 0, 0]
        )
        routed_w3_scale_l3: pl.Tensor[
            [N_LOCAL, MOE_INTER], pl.FP32
        ] = pl.slice(
            routed_w3_scale, [N_LOCAL, MOE_INTER], [3 * N_LOCAL, 0]
        )
        routed_w2_l3: pl.Tensor[
            [N_LOCAL, D, MOE_INTER], pl.INT8
        ] = pl.slice(
            routed_w2, [N_LOCAL, D, MOE_INTER], [3 * N_LOCAL, 0, 0]
        )
        routed_w2_scale_l3: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [3 * N_LOCAL, 0]
        )
        shared_w1_l3: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w1, [MOE_INTER, D], [3 * MOE_INTER, 0]
        )
        shared_w1_scale_l3: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w1_scale, [MOE_INTER], [3 * MOE_INTER]
        )
        shared_w3_l3: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(
            shared_w3, [MOE_INTER, D], [3 * MOE_INTER, 0]
        )
        shared_w3_scale_l3: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(
            shared_w3_scale, [MOE_INTER], [3 * MOE_INTER]
        )
        shared_w2_l3: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(
            shared_w2, [D, MOE_INTER], [3 * D, 0]
        )
        shared_w2_scale_l3: pl.Tensor[[D], pl.FP32] = pl.slice(
            shared_w2_scale, [D], [3 * D]
        )
        decode_layer_hca(
            x_pong,
            hc_attn_fn_l3, hc_attn_scale_l3, hc_attn_base_l3,
            attn_norm_w_l3, wq_a_l3, wq_b_l3, wq_b_scale_l3,
            wkv_l3, gamma_cq_l3, gamma_ckv_l3,
            freqs_cos, freqs_sin,
            hca_cmp_freqs_cos, hca_cmp_freqs_sin,
            hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
            hca_compress_state_l0, hca_compress_state_block_table,
            raw_kv_l3, hca_cmp_kv_l0, hca_cmp_block_table,
            hca_ori_slot_mapping, hca_window_swa_indices,
            hca_window_swa_lens, hca_cmp_slot_mapping,
            hca_state_slot_mapping, position_ids, hca_kv_seq_lens,
            attn_sink_l3, wo_a_l3, wo_b_l3, wo_b_scale_l3,
            hc_ffn_fn_l3, hc_ffn_scale_l3, hc_ffn_base_l3,
            norm_w_l3, gate_w_l3, gate_bias_l3, tid2eid_l3, input_ids,
            routed_w1_l3, routed_w1_scale_l3,
            routed_w3_l3, routed_w3_scale_l3,
            routed_w2_l3, routed_w2_scale_l3,
            shared_w1_l3, shared_w1_scale_l3,
            shared_w3_l3, shared_w3_scale_l3,
            shared_w2_l3, shared_w2_scale_l3,
            x_attn_active, x_moe_next, x_out,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.const(3, pl.INT32), group_base, tp_rank, local_t, my_rank,
            pl.const(4, pl.INT32),
        )
        clear_moe_signals(
            x_moe_next, arrived, data_arrived, combine_arrived
        )
    return x_out


@pl.jit.host
def l3_decode_fwd(
    hc_attn_fn: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MIX_HC, HC_DIM], pl.FP32
    ],
    hc_attn_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * 3], pl.FP32
    ],
    hc_attn_base: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MIX_HC], pl.FP32
    ],
    attn_norm_w: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D], pl.BF16
    ],
    wq_a: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D, Q_LORA], pl.BF16
    ],
    wq_b: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * Q_LORA, H * HEAD_DIM], pl.INT8
    ],
    wq_b_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * H * HEAD_DIM], pl.FP32
    ],
    wkv: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D, HEAD_DIM], pl.BF16
    ],
    gamma_cq: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * Q_LORA], pl.BF16
    ],
    gamma_ckv: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * HEAD_DIM], pl.BF16
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
    freqs_cos: pl.Tensor[
        [N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16
    ],
    freqs_sin: pl.Tensor[
        [N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16
    ],
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
    csa_cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[
        [N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16
    ],
    csa_cmp_ape: pl.Tensor[
        [N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32
    ],
    csa_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
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
        [N_RANKS, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8
    ],
    csa_idx_wq_b_scale: pl.Tensor[
        [N_RANKS, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32
    ],
    csa_weights_proj: pl.Tensor[
        [N_RANKS, D, CSA_IDX_N_HEADS], pl.BF16
    ],
    csa_hadamard_idx: pl.Tensor[
        [N_RANKS, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16
    ],
    csa_inner_wkv: pl.Tensor[
        [N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16
    ],
    csa_inner_wgate: pl.Tensor[
        [N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16
    ],
    csa_inner_ape: pl.Tensor[
        [N_RANKS, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32
    ],
    csa_inner_norm_w: pl.Tensor[
        [N_RANKS, CSA_IDX_HEAD_DIM], pl.BF16
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
    hca_cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[
        [N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16
    ],
    hca_cmp_ape: pl.Tensor[
        [N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32
    ],
    hca_cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
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
    attn_sink: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * H], pl.FP32
    ],
    wo_a: pl.Tensor[
        [
            N_RANKS,
            MIXED_PREFIX_LAYER_COUNT * LOCAL_O_GROUPS,
            O_LORA,
            O_GROUP_IN,
        ],
        pl.BF16,
    ],
    wo_b: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D, LOCAL_O_WIDTH], pl.INT8
    ],
    wo_b_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D], pl.FP32
    ],
    hc_ffn_fn: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MIX_HC, HC_DIM], pl.FP32
    ],
    hc_ffn_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * 3], pl.FP32
    ],
    hc_ffn_base: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MIX_HC], pl.FP32
    ],
    norm_w: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D], pl.BF16
    ],
    gate_w: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_EXPERTS_GLOBAL, D], pl.FP32
    ],
    gate_bias: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_EXPERTS_GLOBAL], pl.FP32
    ],
    tid2eid: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * VOCAB, TOPK], pl.INT32
    ],
    input_ids: pl.Tensor[[N_RANKS, MOE_TOKENS], pl.INT64],
    routed_w1: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MOE_INTER, D], pl.INT8
    ],
    shared_w1_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MOE_INTER], pl.FP32
    ],
    shared_w3: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MOE_INTER, D], pl.INT8
    ],
    shared_w3_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * MOE_INTER], pl.FP32
    ],
    shared_w2: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D, MOE_INTER], pl.INT8
    ],
    shared_w2_scale: pl.Tensor[
        [N_RANKS, MIXED_PREFIX_LAYER_COUNT * D], pl.FP32
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
    x_out: pl.Out[
        pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]
    ],
):
    """Allocate shared windows once and launch one four-layer child per rank."""
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
    csa_inner_compress_state.bind_dynamic(
        1, FWD_CSA_INNER_STATE_BLOCKS_DYN
    )
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
    x_out.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer(
        [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
    )
    attention_signal_buf = pld.alloc_window_buffer(
        [TP_SIZE, 1], dtype=pl.INT32
    )
    o_window_buf = pld.alloc_window_buffer(
        [O_WINDOW_ROWS, D], dtype=pl.FP32
    )
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    recv_meta_buf = pld.alloc_window_buffer(
        [N_RANKS, N_LOCAL], dtype=pl.INT32
    )
    recv_x_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
    )
    recv_aux_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, 1], dtype=pl.INT32
    )
    routed_y_buf_buf = pld.alloc_window_buffer(
        [N_ROUTES, D], dtype=pl.BF16
    )
    combine_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, 1], dtype=pl.INT32
    )

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(
            attention_window_buf,
            [ATTENTION_WINDOW_ROWS, O_GROUP_IN],
            dtype=pl.BF16,
        )
        attention_signal = pld.window(
            attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        o_window = pld.window(
            o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32
        )
        o_signal = pld.window(
            o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        recv_meta = pld.window(
            recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32
        )
        recv_x = pld.window(
            recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
        )
        recv_aux = pld.window(
            recv_aux_buf,
            [N_LOCAL * RECV_MAX, AUX_PAD],
            dtype=pl.FP32,
        )
        recv_route = pld.window(
            recv_route_buf,
            [N_LOCAL * RECV_MAX, IDX_PAD],
            dtype=pl.INT32,
        )
        arrived = pld.window(
            arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        data_arrived = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        routed_y_buf = pld.window(
            routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16
        )
        combine_arrived = pld.window(
            combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        decode_fwd(
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
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_ping[rank], x_pong[rank],
            x_attn_active[rank], x_moe_next[rank], x_out[rank],
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            group_base, tp_rank, rank,
            device=rank,
        )
    return x_out


@pl.jit(auto_scope=False)
def decode_fwd_full43(
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
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
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
    x_ping: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_pong: pl.InOut[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_attn_active: pl.InOut[
        pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]
    ],
    x_moe_next: pl.InOut[
        pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]
    ],
    x_out: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
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
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
):
    """Run the fixed 2-SWA, 21-CSA, 20-HCA model in one rank child."""
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
    csa_inner_compress_state.bind_dynamic(
        0, FWD_CSA_INNER_STATE_BLOCKS_DYN
    )
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
    x_out.bind_dynamic(0, T_DYN)

    local_t = pl.cast(pl.tensor.dim(x_ping, 0), pl.INT32)
    raw_blocks_per_layer = (
        pl.tensor.dim(raw_kv_pool, 0) // MAIN_LAYER_COUNT
    )
    csa_state_blocks_per_layer = (
        pl.tensor.dim(csa_compress_state, 0) // CSA_LAYER_COUNT
    )
    csa_cmp_blocks_per_layer = (
        pl.tensor.dim(csa_cmp_kv, 0) // CSA_LAYER_COUNT
    )
    csa_inner_state_blocks_per_layer = (
        pl.tensor.dim(csa_inner_compress_state, 0) // CSA_LAYER_COUNT
    )
    csa_idx_blocks_per_layer = (
        pl.tensor.dim(csa_idx_kv_cache, 0) // CSA_LAYER_COUNT
    )
    hca_state_blocks_per_layer = (
        pl.tensor.dim(hca_compress_state, 0) // HCA_LAYER_COUNT
    )
    hca_cmp_blocks_per_layer = (
        pl.tensor.dim(hca_cmp_kv, 0) // HCA_LAYER_COUNT
    )

    with pl.scope():
        weight_layer_swa0 = pl.const(0, pl.INT32)
        hc_attn_fn_layer_swa0 = pl.slice(
            hc_attn_fn, [MIX_HC, HC_DIM], [0, 0]
        )
        hc_ffn_fn_layer_swa0 = pl.slice(
            hc_ffn_fn, [MIX_HC, HC_DIM], [0, 0]
        )
        wq_a_layer_swa0: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [0, 0]
        )
        wq_b_layer_swa0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [0, 0]
        )
        wq_b_scale_layer_swa0: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale, [H * HEAD_DIM], [0]
        )
        wkv_layer_swa0: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [0, 0]
        )
        gamma_cq_layer_swa0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [0]
        )
        gamma_ckv_layer_swa0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [0]
        )
        wo_a_layer_swa0: pl.Tensor[
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
        ] = pl.slice(
            wo_a,
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
            [weight_layer_swa0 * LOCAL_O_GROUPS, 0, 0],
        )
        routed_w1_layer_swa0: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL, MOE_INTER, D],
            [weight_layer_swa0 * N_LOCAL, 0, 0],
        )
        hc_attn_scale_layer_swa0 = pl.slice(hc_attn_scale, [3], [0])
        hc_attn_base_layer_swa0 = pl.slice(hc_attn_base, [MIX_HC], [0])
        attn_norm_w_layer_swa0 = pl.slice(attn_norm_w, [D], [0])
        attn_sink_layer_swa0 = pl.slice(attn_sink, [H], [0])
        wo_b_layer_swa0 = pl.slice(wo_b, [D, LOCAL_O_WIDTH], [0, 0])
        wo_b_scale_layer_swa0 = pl.slice(wo_b_scale, [D], [0])
        hc_ffn_scale_layer_swa0 = pl.slice(hc_ffn_scale, [3], [0])
        hc_ffn_base_layer_swa0 = pl.slice(hc_ffn_base, [MIX_HC], [0])
        norm_w_layer_swa0 = pl.slice(norm_w, [D], [0])
        gate_w_layer_swa0 = pl.slice(
            gate_w, [N_EXPERTS_GLOBAL, D], [0, 0]
        )
        gate_bias_layer_swa0 = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [0])
        tid2eid_layer_swa0 = pl.slice(tid2eid, [VOCAB, TOPK], [0, 0])
        routed_w1_scale_layer_swa0 = pl.slice(
            routed_w1_scale, [N_LOCAL, MOE_INTER], [0, 0]
        )
        routed_w3_scale_layer_swa0 = pl.slice(
            routed_w3_scale, [N_LOCAL, MOE_INTER], [0, 0]
        )
        routed_w2_scale_layer_swa0 = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [0, 0]
        )
        shared_w1_layer_swa0 = pl.slice(shared_w1, [MOE_INTER, D], [0, 0])
        shared_w1_scale_layer_swa0 = pl.slice(
            shared_w1_scale, [MOE_INTER], [0]
        )
        shared_w3_layer_swa0 = pl.slice(shared_w3, [MOE_INTER, D], [0, 0])
        shared_w3_scale_layer_swa0 = pl.slice(
            shared_w3_scale, [MOE_INTER], [0]
        )
        shared_w2_layer_swa0 = pl.slice(shared_w2, [D, MOE_INTER], [0, 0])
        shared_w2_scale_layer_swa0 = pl.slice(shared_w2_scale, [D], [0])
        routed_w3_layer_swa0: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL, MOE_INTER, D],
            [weight_layer_swa0 * N_LOCAL, 0, 0],
        )
        routed_w2_layer_swa0: pl.Tensor[
            [N_LOCAL, D, MOE_INTER], pl.INT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL, D, MOE_INTER],
            [weight_layer_swa0 * N_LOCAL, 0, 0],
        )
        raw_kv_layer_swa0 = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [0, 0, 0, 0],
        )
        decode_layer_swa(
            x_ping,
            hc_attn_fn_layer_swa0, hc_attn_scale_layer_swa0,
            hc_attn_base_layer_swa0, attn_norm_w_layer_swa0,
            wq_a_layer_swa0, wq_b_layer_swa0, wq_b_scale_layer_swa0, wkv_layer_swa0,
            gamma_cq_layer_swa0, gamma_ckv_layer_swa0,
            freqs_cos, freqs_sin, raw_kv_layer_swa0,
            swa_slot_mapping, swa_indices, swa_lens, position_ids,
            attn_sink_layer_swa0, wo_a_layer_swa0, wo_b_layer_swa0, wo_b_scale_layer_swa0,
            hc_ffn_fn_layer_swa0, hc_ffn_scale_layer_swa0,
            hc_ffn_base_layer_swa0, norm_w_layer_swa0,
            gate_w_layer_swa0, gate_bias_layer_swa0, tid2eid_layer_swa0, input_ids,
            routed_w1_layer_swa0, routed_w1_scale_layer_swa0,
            routed_w3_layer_swa0, routed_w3_scale_layer_swa0,
            routed_w2_layer_swa0, routed_w2_scale_layer_swa0,
            shared_w1_layer_swa0, shared_w1_scale_layer_swa0,
            shared_w3_layer_swa0, shared_w3_scale_layer_swa0,
            shared_w2_layer_swa0, shared_w2_scale_layer_swa0,
            x_attn_active, x_moe_next, x_pong,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.const(0, pl.INT32), group_base, tp_rank, local_t, my_rank,
            pl.const(1, pl.INT32),
        )

    with pl.scope():
        weight_layer_swa1 = pl.const(1, pl.INT32) % FWD_WEIGHT_BANK_SIZE
        hc_attn_fn_layer_swa1 = pl.slice(
            hc_attn_fn,
            [MIX_HC, HC_DIM],
            [weight_layer_swa1 * HC_FN_STORAGE_ROWS, 0],
        )
        hc_ffn_fn_layer_swa1 = pl.slice(
            hc_ffn_fn,
            [MIX_HC, HC_DIM],
            [weight_layer_swa1 * HC_FN_STORAGE_ROWS, 0],
        )
        wq_a_layer_swa1: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [weight_layer_swa1 * D, 0]
        )
        wq_b_layer_swa1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [weight_layer_swa1 * Q_LORA, 0]
        )
        wq_b_scale_layer_swa1: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale,
            [H * HEAD_DIM],
            [weight_layer_swa1 * H * HEAD_DIM],
        )
        wkv_layer_swa1: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [weight_layer_swa1 * D, 0]
        )
        gamma_cq_layer_swa1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [weight_layer_swa1 * Q_LORA]
        )
        gamma_ckv_layer_swa1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [weight_layer_swa1 * HEAD_DIM]
        )
        wo_a_layer_swa1: pl.Tensor[
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
        ] = pl.slice(
            wo_a,
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
            [weight_layer_swa1 * LOCAL_O_GROUPS, 0, 0],
        )
        routed_w1_layer_swa1: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL, MOE_INTER, D],
            [weight_layer_swa1 * N_LOCAL, 0, 0],
        )
        hc_attn_scale_layer_swa1 = pl.slice(
            hc_attn_scale, [3], [weight_layer_swa1 * 3]
        )
        hc_attn_base_layer_swa1 = pl.slice(
            hc_attn_base, [MIX_HC], [weight_layer_swa1 * MIX_HC]
        )
        attn_norm_w_layer_swa1 = pl.slice(
            attn_norm_w, [D], [weight_layer_swa1 * D]
        )
        attn_sink_layer_swa1 = pl.slice(attn_sink, [H], [weight_layer_swa1 * H])
        wo_b_layer_swa1 = pl.slice(
            wo_b, [D, LOCAL_O_WIDTH], [weight_layer_swa1 * D, 0]
        )
        wo_b_scale_layer_swa1 = pl.slice(
            wo_b_scale, [D], [weight_layer_swa1 * D]
        )
        hc_ffn_scale_layer_swa1 = pl.slice(
            hc_ffn_scale, [3], [weight_layer_swa1 * 3]
        )
        hc_ffn_base_layer_swa1 = pl.slice(
            hc_ffn_base, [MIX_HC], [weight_layer_swa1 * MIX_HC]
        )
        norm_w_layer_swa1 = pl.slice(norm_w, [D], [weight_layer_swa1 * D])
        gate_w_layer_swa1 = pl.slice(
            gate_w,
            [N_EXPERTS_GLOBAL, D],
            [weight_layer_swa1 * N_EXPERTS_GLOBAL, 0],
        )
        gate_bias_layer_swa1 = pl.slice(
            gate_bias,
            [N_EXPERTS_GLOBAL],
            [weight_layer_swa1 * N_EXPERTS_GLOBAL],
        )
        tid2eid_layer_swa1 = pl.slice(
            tid2eid, [VOCAB, TOPK], [weight_layer_swa1 * VOCAB, 0]
        )
        routed_w1_scale_layer_swa1 = pl.slice(
            routed_w1_scale,
            [N_LOCAL, MOE_INTER],
            [weight_layer_swa1 * N_LOCAL, 0],
        )
        routed_w3_scale_layer_swa1 = pl.slice(
            routed_w3_scale,
            [N_LOCAL, MOE_INTER],
            [weight_layer_swa1 * N_LOCAL, 0],
        )
        routed_w2_scale_layer_swa1 = pl.slice(
            routed_w2_scale,
            [N_LOCAL, D],
            [weight_layer_swa1 * N_LOCAL, 0],
        )
        shared_w1_layer_swa1 = pl.slice(
            shared_w1, [MOE_INTER, D], [weight_layer_swa1 * MOE_INTER, 0]
        )
        shared_w1_scale_layer_swa1 = pl.slice(
            shared_w1_scale, [MOE_INTER], [weight_layer_swa1 * MOE_INTER]
        )
        shared_w3_layer_swa1 = pl.slice(
            shared_w3, [MOE_INTER, D], [weight_layer_swa1 * MOE_INTER, 0]
        )
        shared_w3_scale_layer_swa1 = pl.slice(
            shared_w3_scale, [MOE_INTER], [weight_layer_swa1 * MOE_INTER]
        )
        shared_w2_layer_swa1 = pl.slice(
            shared_w2, [D, MOE_INTER], [weight_layer_swa1 * D, 0]
        )
        shared_w2_scale_layer_swa1 = pl.slice(
            shared_w2_scale, [D], [weight_layer_swa1 * D]
        )
        routed_w3_layer_swa1: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL, MOE_INTER, D],
            [weight_layer_swa1 * N_LOCAL, 0, 0],
        )
        routed_w2_layer_swa1: pl.Tensor[
            [N_LOCAL, D, MOE_INTER], pl.INT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL, D, MOE_INTER],
            [weight_layer_swa1 * N_LOCAL, 0, 0],
        )
        raw_kv_layer_swa1 = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [raw_blocks_per_layer, 0, 0, 0],
        )
        decode_layer_swa(
            x_pong,
            hc_attn_fn_layer_swa1, hc_attn_scale_layer_swa1,
            hc_attn_base_layer_swa1, attn_norm_w_layer_swa1,
            wq_a_layer_swa1, wq_b_layer_swa1, wq_b_scale_layer_swa1, wkv_layer_swa1,
            gamma_cq_layer_swa1, gamma_ckv_layer_swa1,
            freqs_cos, freqs_sin, raw_kv_layer_swa1,
            swa_slot_mapping, swa_indices, swa_lens, position_ids,
            attn_sink_layer_swa1, wo_a_layer_swa1, wo_b_layer_swa1, wo_b_scale_layer_swa1,
            hc_ffn_fn_layer_swa1, hc_ffn_scale_layer_swa1,
            hc_ffn_base_layer_swa1, norm_w_layer_swa1,
            gate_w_layer_swa1, gate_bias_layer_swa1, tid2eid_layer_swa1, input_ids,
            routed_w1_layer_swa1, routed_w1_scale_layer_swa1,
            routed_w3_layer_swa1, routed_w3_scale_layer_swa1,
            routed_w2_layer_swa1, routed_w2_scale_layer_swa1,
            shared_w1_layer_swa1, shared_w1_scale_layer_swa1,
            shared_w3_layer_swa1, shared_w3_scale_layer_swa1,
            shared_w2_layer_swa1, shared_w2_scale_layer_swa1,
            x_attn_active, x_moe_next, x_ping,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            pl.const(1, pl.INT32), group_base, tp_rank, local_t, my_rank,
            pl.const(2, pl.INT32),
        )

    for ordinal in pl.range(HCA_LAYER_COUNT):
        csa_model_layer = pl.cast(ordinal * 2 + 2, pl.INT32)
        hca_model_layer = pl.cast(ordinal * 2 + 3, pl.INT32)
        csa_weight_layer = csa_model_layer % FWD_WEIGHT_BANK_SIZE
        hca_weight_layer = hca_model_layer % FWD_WEIGHT_BANK_SIZE
        csa_extra_layer = ordinal % FWD_CSA_WEIGHT_BANK_SIZE
        hca_extra_layer = ordinal % FWD_HCA_WEIGHT_BANK_SIZE

        with pl.scope():
            hc_attn_fn_layer_csa = pl.slice(
                hc_attn_fn,
                [MIX_HC, HC_DIM],
                [csa_weight_layer * HC_FN_STORAGE_ROWS, 0],
            )
            hc_ffn_fn_layer_csa = pl.slice(
                hc_ffn_fn,
                [MIX_HC, HC_DIM],
                [csa_weight_layer * HC_FN_STORAGE_ROWS, 0],
            )
            wq_a_layer_csa: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
                wq_a, [D, Q_LORA], [csa_weight_layer * D, 0]
            )
            wq_b_layer_csa: pl.Tensor[
                [Q_LORA, H * HEAD_DIM], pl.INT8
            ] = pl.slice(
                wq_b,
                [Q_LORA, H * HEAD_DIM],
                [csa_weight_layer * Q_LORA, 0],
            )
            wq_b_scale_layer_csa: pl.Tensor[
                [H * HEAD_DIM], pl.FP32
            ] = pl.slice(
                wq_b_scale,
                [H * HEAD_DIM],
                [csa_weight_layer * H * HEAD_DIM],
            )
            wkv_layer_csa: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
                wkv, [D, HEAD_DIM], [csa_weight_layer * D, 0]
            )
            gamma_cq_layer_csa: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
                gamma_cq, [Q_LORA], [csa_weight_layer * Q_LORA]
            )
            gamma_ckv_layer_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
                gamma_ckv, [HEAD_DIM], [csa_weight_layer * HEAD_DIM]
            )
            wo_a_layer_csa: pl.Tensor[
                [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
            ] = pl.slice(
                wo_a,
                [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
                [csa_weight_layer * LOCAL_O_GROUPS, 0, 0],
            )
            routed_w1_layer_csa: pl.Tensor[
                [N_LOCAL, MOE_INTER, D], pl.INT8
            ] = pl.slice(
                routed_w1,
                [N_LOCAL, MOE_INTER, D],
                [csa_weight_layer * N_LOCAL, 0, 0],
            )
            routed_w3_layer_csa: pl.Tensor[
                [N_LOCAL, MOE_INTER, D], pl.INT8
            ] = pl.slice(
                routed_w3,
                [N_LOCAL, MOE_INTER, D],
                [csa_weight_layer * N_LOCAL, 0, 0],
            )
            routed_w2_layer_csa: pl.Tensor[
                [N_LOCAL, D, MOE_INTER], pl.INT8
            ] = pl.slice(
                routed_w2,
                [N_LOCAL, D, MOE_INTER],
                [csa_weight_layer * N_LOCAL, 0, 0],
            )
            raw_kv_layer_csa = pl.slice(
                raw_kv_pool,
                [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
                [csa_model_layer * raw_blocks_per_layer, 0, 0, 0],
            )
            csa_state_layer_csa = pl.slice(
                csa_compress_state,
                [
                    csa_state_blocks_per_layer,
                    CSA_MAIN_STATE_BLOCK_SIZE,
                    CSA_MAIN_STATE_DIM,
                ],
                [ordinal * csa_state_blocks_per_layer, 0, 0],
            )
            csa_cmp_kv_layer_csa = pl.slice(
                csa_cmp_kv,
                [csa_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
                [ordinal * csa_cmp_blocks_per_layer, 0, 0, 0],
            )
            csa_inner_state_layer_csa = pl.slice(
                csa_inner_compress_state,
                [
                    csa_inner_state_blocks_per_layer,
                    CSA_INNER_STATE_BLOCK_SIZE,
                    CSA_INNER_STATE_DIM,
                ],
                [ordinal * csa_inner_state_blocks_per_layer, 0, 0],
            )
            csa_idx_cache_layer_csa = pl.slice(
                csa_idx_kv_cache,
                [
                    csa_idx_blocks_per_layer,
                    BLOCK_SIZE,
                    1,
                    CSA_IDX_HEAD_DIM,
                ],
                [ordinal * csa_idx_blocks_per_layer, 0, 0, 0],
            )
            csa_idx_scale_layer_csa = pl.slice(
                csa_idx_kv_scale,
                [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, 1],
                [ordinal * csa_idx_blocks_per_layer, 0, 0, 0],
            )
            hc_attn_scale_layer_csa = pl.slice(
                hc_attn_scale, [3], [csa_weight_layer * 3]
            )
            hc_attn_base_layer_csa = pl.slice(
                hc_attn_base, [MIX_HC], [csa_weight_layer * MIX_HC]
            )
            attn_norm_w_layer_csa = pl.slice(
                attn_norm_w, [D], [csa_weight_layer * D]
            )
            csa_cmp_wkv_layer_csa = pl.slice(
                csa_cmp_wkv,
                [CSA_MAIN_OUT_DIM, D],
                [csa_extra_layer * CSA_MAIN_OUT_DIM, 0],
            )
            csa_cmp_wgate_layer_csa = pl.slice(
                csa_cmp_wgate,
                [CSA_MAIN_OUT_DIM, D],
                [csa_extra_layer * CSA_MAIN_OUT_DIM, 0],
            )
            csa_cmp_ape_layer_csa = pl.slice(
                csa_cmp_ape,
                [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
                [csa_extra_layer * CSA_COMPRESS_RATIO, 0],
            )
            csa_cmp_norm_w_layer_csa = pl.slice(
                csa_cmp_norm_w,
                [HEAD_DIM],
                [csa_extra_layer * HEAD_DIM],
            )
            csa_idx_wq_b_layer_csa = pl.slice(
                csa_idx_wq_b,
                [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
                [csa_extra_layer * Q_LORA, 0],
            )
            csa_idx_wq_b_scale_layer_csa = pl.slice(
                csa_idx_wq_b_scale,
                [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
                [csa_extra_layer * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            )
            csa_weights_proj_layer_csa = pl.slice(
                csa_weights_proj,
                [D, CSA_IDX_N_HEADS],
                [csa_extra_layer * D, 0],
            )
            csa_hadamard_idx_layer_csa = pl.slice(
                csa_hadamard_idx,
                [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
                [csa_extra_layer * CSA_IDX_HEAD_DIM, 0],
            )
            csa_inner_wkv_layer_csa = pl.slice(
                csa_inner_wkv,
                [CSA_INNER_OUT_DIM, D],
                [csa_extra_layer * CSA_INNER_OUT_DIM, 0],
            )
            csa_inner_wgate_layer_csa = pl.slice(
                csa_inner_wgate,
                [CSA_INNER_OUT_DIM, D],
                [csa_extra_layer * CSA_INNER_OUT_DIM, 0],
            )
            csa_inner_ape_layer_csa = pl.slice(
                csa_inner_ape,
                [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
                [csa_extra_layer * CSA_COMPRESS_RATIO, 0],
            )
            csa_inner_norm_w_layer_csa = pl.slice(
                csa_inner_norm_w,
                [CSA_IDX_HEAD_DIM],
                [csa_extra_layer * CSA_IDX_HEAD_DIM],
            )
            attn_sink_layer_csa = pl.slice(
                attn_sink, [H], [csa_weight_layer * H]
            )
            wo_b_layer_csa = pl.slice(
                wo_b, [D, LOCAL_O_WIDTH], [csa_weight_layer * D, 0]
            )
            wo_b_scale_layer_csa = pl.slice(
                wo_b_scale, [D], [csa_weight_layer * D]
            )
            hc_ffn_scale_layer_csa = pl.slice(
                hc_ffn_scale, [3], [csa_weight_layer * 3]
            )
            hc_ffn_base_layer_csa = pl.slice(
                hc_ffn_base, [MIX_HC], [csa_weight_layer * MIX_HC]
            )
            norm_w_layer_csa = pl.slice(norm_w, [D], [csa_weight_layer * D])
            gate_w_layer_csa = pl.slice(
                gate_w,
                [N_EXPERTS_GLOBAL, D],
                [csa_weight_layer * N_EXPERTS_GLOBAL, 0],
            )
            gate_bias_layer_csa = pl.slice(
                gate_bias,
                [N_EXPERTS_GLOBAL],
                [csa_weight_layer * N_EXPERTS_GLOBAL],
            )
            tid2eid_layer_csa = pl.slice(
                tid2eid, [VOCAB, TOPK], [csa_weight_layer * VOCAB, 0]
            )
            routed_w1_scale_layer_csa = pl.slice(
                routed_w1_scale,
                [N_LOCAL, MOE_INTER],
                [csa_weight_layer * N_LOCAL, 0],
            )
            routed_w3_scale_layer_csa = pl.slice(
                routed_w3_scale,
                [N_LOCAL, MOE_INTER],
                [csa_weight_layer * N_LOCAL, 0],
            )
            routed_w2_scale_layer_csa = pl.slice(
                routed_w2_scale,
                [N_LOCAL, D],
                [csa_weight_layer * N_LOCAL, 0],
            )
            shared_w1_layer_csa = pl.slice(
                shared_w1,
                [MOE_INTER, D],
                [csa_weight_layer * MOE_INTER, 0],
            )
            shared_w1_scale_layer_csa = pl.slice(
                shared_w1_scale,
                [MOE_INTER],
                [csa_weight_layer * MOE_INTER],
            )
            shared_w3_layer_csa = pl.slice(
                shared_w3,
                [MOE_INTER, D],
                [csa_weight_layer * MOE_INTER, 0],
            )
            shared_w3_scale_layer_csa = pl.slice(
                shared_w3_scale,
                [MOE_INTER],
                [csa_weight_layer * MOE_INTER],
            )
            shared_w2_layer_csa = pl.slice(
                shared_w2, [D, MOE_INTER], [csa_weight_layer * D, 0]
            )
            shared_w2_scale_layer_csa = pl.slice(
                shared_w2_scale, [D], [csa_weight_layer * D]
            )
            decode_layer_csa(
                x_ping,
                hc_attn_fn_layer_csa, hc_attn_scale_layer_csa,
                hc_attn_base_layer_csa, attn_norm_w_layer_csa,
                wq_a_layer_csa, wq_b_layer_csa, wq_b_scale_layer_csa, wkv_layer_csa,
                gamma_cq_layer_csa, gamma_ckv_layer_csa,
                freqs_cos, freqs_sin,
                csa_cmp_freqs_cos, csa_cmp_freqs_sin,
                csa_cmp_wkv_layer_csa, csa_cmp_wgate_layer_csa,
                csa_cmp_ape_layer_csa, csa_cmp_norm_w_layer_csa,
                csa_state_layer_csa, csa_compress_state_block_table,
                csa_idx_wq_b_layer_csa, csa_idx_wq_b_scale_layer_csa,
                csa_weights_proj_layer_csa, csa_hadamard_idx_layer_csa,
                csa_inner_wkv_layer_csa, csa_inner_wgate_layer_csa,
                csa_inner_ape_layer_csa, csa_inner_norm_w_layer_csa,
                csa_inner_state_layer_csa,
                csa_inner_compress_state_block_table,
                raw_kv_layer_csa, csa_cmp_kv_layer_csa, csa_cmp_block_table,
                csa_idx_cache_layer_csa, csa_idx_scale_layer_csa,
                csa_idx_block_table,
                csa_ori_slot_mapping, csa_window_swa_indices,
                csa_window_swa_lens, csa_cmp_slot_mapping,
                csa_idx_slot_mapping, csa_state_slot_mapping,
                csa_inner_state_slot_mapping, position_ids,
                csa_kv_seq_lens, attn_sink_layer_csa,
                wo_a_layer_csa, wo_b_layer_csa, wo_b_scale_layer_csa,
                hc_ffn_fn_layer_csa, hc_ffn_scale_layer_csa,
                hc_ffn_base_layer_csa, norm_w_layer_csa,
                gate_w_layer_csa, gate_bias_layer_csa, tid2eid_layer_csa, input_ids,
                routed_w1_layer_csa, routed_w1_scale_layer_csa,
                routed_w3_layer_csa, routed_w3_scale_layer_csa,
                routed_w2_layer_csa, routed_w2_scale_layer_csa,
                shared_w1_layer_csa, shared_w1_scale_layer_csa,
                shared_w3_layer_csa, shared_w3_scale_layer_csa,
                shared_w2_layer_csa, shared_w2_scale_layer_csa,
                x_attn_active, x_moe_next, x_pong,
                attention_window, attention_signal, o_window, o_signal,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                csa_model_layer, group_base, tp_rank, local_t, my_rank,
                csa_model_layer + 1,
            )

        with pl.scope():
            hc_attn_fn_layer_hca = pl.slice(
                hc_attn_fn,
                [MIX_HC, HC_DIM],
                [hca_weight_layer * HC_FN_STORAGE_ROWS, 0],
            )
            hc_ffn_fn_layer_hca = pl.slice(
                hc_ffn_fn,
                [MIX_HC, HC_DIM],
                [hca_weight_layer * HC_FN_STORAGE_ROWS, 0],
            )
            wq_a_layer_hca: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
                wq_a, [D, Q_LORA], [hca_weight_layer * D, 0]
            )
            wq_b_layer_hca: pl.Tensor[
                [Q_LORA, H * HEAD_DIM], pl.INT8
            ] = pl.slice(
                wq_b,
                [Q_LORA, H * HEAD_DIM],
                [hca_weight_layer * Q_LORA, 0],
            )
            wq_b_scale_layer_hca: pl.Tensor[
                [H * HEAD_DIM], pl.FP32
            ] = pl.slice(
                wq_b_scale,
                [H * HEAD_DIM],
                [hca_weight_layer * H * HEAD_DIM],
            )
            wkv_layer_hca: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
                wkv, [D, HEAD_DIM], [hca_weight_layer * D, 0]
            )
            gamma_cq_layer_hca: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
                gamma_cq, [Q_LORA], [hca_weight_layer * Q_LORA]
            )
            gamma_ckv_layer_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
                gamma_ckv, [HEAD_DIM], [hca_weight_layer * HEAD_DIM]
            )
            wo_a_layer_hca: pl.Tensor[
                [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
            ] = pl.slice(
                wo_a,
                [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
                [hca_weight_layer * LOCAL_O_GROUPS, 0, 0],
            )
            routed_w1_layer_hca: pl.Tensor[
                [N_LOCAL, MOE_INTER, D], pl.INT8
            ] = pl.slice(
                routed_w1,
                [N_LOCAL, MOE_INTER, D],
                [hca_weight_layer * N_LOCAL, 0, 0],
            )
            routed_w3_layer_hca: pl.Tensor[
                [N_LOCAL, MOE_INTER, D], pl.INT8
            ] = pl.slice(
                routed_w3,
                [N_LOCAL, MOE_INTER, D],
                [hca_weight_layer * N_LOCAL, 0, 0],
            )
            routed_w2_layer_hca: pl.Tensor[
                [N_LOCAL, D, MOE_INTER], pl.INT8
            ] = pl.slice(
                routed_w2,
                [N_LOCAL, D, MOE_INTER],
                [hca_weight_layer * N_LOCAL, 0, 0],
            )
            raw_kv_layer_hca = pl.slice(
                raw_kv_pool,
                [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
                [hca_model_layer * raw_blocks_per_layer, 0, 0, 0],
            )
            hca_state_layer_hca = pl.slice(
                hca_compress_state,
                [
                    hca_state_blocks_per_layer,
                    HCA_COMPRESS_STATE_BLOCK_SIZE,
                    HCA_COMPRESS_STATE_DIM,
                ],
                [ordinal * hca_state_blocks_per_layer, 0, 0],
            )
            hca_cmp_kv_layer_hca = pl.slice(
                hca_cmp_kv,
                [hca_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
                [ordinal * hca_cmp_blocks_per_layer, 0, 0, 0],
            )
            hc_attn_scale_layer_hca = pl.slice(
                hc_attn_scale, [3], [hca_weight_layer * 3]
            )
            hc_attn_base_layer_hca = pl.slice(
                hc_attn_base, [MIX_HC], [hca_weight_layer * MIX_HC]
            )
            attn_norm_w_layer_hca = pl.slice(
                attn_norm_w, [D], [hca_weight_layer * D]
            )
            hca_cmp_wkv_layer_hca = pl.slice(
                hca_cmp_wkv,
                [HCA_MAIN_OUT_DIM, D],
                [hca_extra_layer * HCA_MAIN_OUT_DIM, 0],
            )
            hca_cmp_wgate_layer_hca = pl.slice(
                hca_cmp_wgate,
                [HCA_MAIN_OUT_DIM, D],
                [hca_extra_layer * HCA_MAIN_OUT_DIM, 0],
            )
            hca_cmp_ape_layer_hca = pl.slice(
                hca_cmp_ape,
                [HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM],
                [hca_extra_layer * HCA_COMPRESS_RATIO, 0],
            )
            hca_cmp_norm_w_layer_hca = pl.slice(
                hca_cmp_norm_w,
                [HEAD_DIM],
                [hca_extra_layer * HEAD_DIM],
            )
            attn_sink_layer_hca = pl.slice(
                attn_sink, [H], [hca_weight_layer * H]
            )
            wo_b_layer_hca = pl.slice(
                wo_b, [D, LOCAL_O_WIDTH], [hca_weight_layer * D, 0]
            )
            wo_b_scale_layer_hca = pl.slice(
                wo_b_scale, [D], [hca_weight_layer * D]
            )
            hc_ffn_scale_layer_hca = pl.slice(
                hc_ffn_scale, [3], [hca_weight_layer * 3]
            )
            hc_ffn_base_layer_hca = pl.slice(
                hc_ffn_base, [MIX_HC], [hca_weight_layer * MIX_HC]
            )
            norm_w_layer_hca = pl.slice(norm_w, [D], [hca_weight_layer * D])
            gate_w_layer_hca = pl.slice(
                gate_w,
                [N_EXPERTS_GLOBAL, D],
                [hca_weight_layer * N_EXPERTS_GLOBAL, 0],
            )
            gate_bias_layer_hca = pl.slice(
                gate_bias,
                [N_EXPERTS_GLOBAL],
                [hca_weight_layer * N_EXPERTS_GLOBAL],
            )
            tid2eid_layer_hca = pl.slice(
                tid2eid, [VOCAB, TOPK], [hca_weight_layer * VOCAB, 0]
            )
            routed_w1_scale_layer_hca = pl.slice(
                routed_w1_scale,
                [N_LOCAL, MOE_INTER],
                [hca_weight_layer * N_LOCAL, 0],
            )
            routed_w3_scale_layer_hca = pl.slice(
                routed_w3_scale,
                [N_LOCAL, MOE_INTER],
                [hca_weight_layer * N_LOCAL, 0],
            )
            routed_w2_scale_layer_hca = pl.slice(
                routed_w2_scale,
                [N_LOCAL, D],
                [hca_weight_layer * N_LOCAL, 0],
            )
            shared_w1_layer_hca = pl.slice(
                shared_w1,
                [MOE_INTER, D],
                [hca_weight_layer * MOE_INTER, 0],
            )
            shared_w1_scale_layer_hca = pl.slice(
                shared_w1_scale,
                [MOE_INTER],
                [hca_weight_layer * MOE_INTER],
            )
            shared_w3_layer_hca = pl.slice(
                shared_w3,
                [MOE_INTER, D],
                [hca_weight_layer * MOE_INTER, 0],
            )
            shared_w3_scale_layer_hca = pl.slice(
                shared_w3_scale,
                [MOE_INTER],
                [hca_weight_layer * MOE_INTER],
            )
            shared_w2_layer_hca = pl.slice(
                shared_w2, [D, MOE_INTER], [hca_weight_layer * D, 0]
            )
            shared_w2_scale_layer_hca = pl.slice(
                shared_w2_scale, [D], [hca_weight_layer * D]
            )
            decode_layer_hca(
                x_pong,
                hc_attn_fn_layer_hca, hc_attn_scale_layer_hca,
                hc_attn_base_layer_hca, attn_norm_w_layer_hca,
                wq_a_layer_hca, wq_b_layer_hca, wq_b_scale_layer_hca, wkv_layer_hca,
                gamma_cq_layer_hca, gamma_ckv_layer_hca,
                freqs_cos, freqs_sin,
                hca_cmp_freqs_cos, hca_cmp_freqs_sin,
                hca_cmp_wkv_layer_hca, hca_cmp_wgate_layer_hca,
                hca_cmp_ape_layer_hca, hca_cmp_norm_w_layer_hca,
                hca_state_layer_hca, hca_compress_state_block_table,
                raw_kv_layer_hca, hca_cmp_kv_layer_hca, hca_cmp_block_table,
                hca_ori_slot_mapping, hca_window_swa_indices,
                hca_window_swa_lens, hca_cmp_slot_mapping,
                hca_state_slot_mapping, position_ids, hca_kv_seq_lens,
                attn_sink_layer_hca, wo_a_layer_hca, wo_b_layer_hca, wo_b_scale_layer_hca,
                hc_ffn_fn_layer_hca, hc_ffn_scale_layer_hca,
                hc_ffn_base_layer_hca, norm_w_layer_hca,
                gate_w_layer_hca, gate_bias_layer_hca, tid2eid_layer_hca, input_ids,
                routed_w1_layer_hca, routed_w1_scale_layer_hca,
                routed_w3_layer_hca, routed_w3_scale_layer_hca,
                routed_w2_layer_hca, routed_w2_scale_layer_hca,
                shared_w1_layer_hca, shared_w1_scale_layer_hca,
                shared_w3_layer_hca, shared_w3_scale_layer_hca,
                shared_w2_layer_hca, shared_w2_scale_layer_hca,
                x_attn_active, x_moe_next, x_ping,
                attention_window, attention_signal, o_window, o_signal,
                recv_meta, recv_x, recv_aux, recv_route,
                arrived, data_arrived, routed_y_buf, combine_arrived,
                hca_model_layer, group_base, tp_rank, local_t, my_rank,
                hca_model_layer + 1,
            )

    with pl.scope():
        csa_ordinal_last = pl.const(20, pl.INT32)
        model_layer_last = pl.const(42, pl.INT32)
        weight_layer_last = model_layer_last % FWD_WEIGHT_BANK_SIZE
        extra_layer_last = csa_ordinal_last % FWD_CSA_WEIGHT_BANK_SIZE
        hc_attn_fn_layer_last = pl.slice(
            hc_attn_fn,
            [MIX_HC, HC_DIM],
            [weight_layer_last * HC_FN_STORAGE_ROWS, 0],
        )
        hc_ffn_fn_layer_last = pl.slice(
            hc_ffn_fn,
            [MIX_HC, HC_DIM],
            [weight_layer_last * HC_FN_STORAGE_ROWS, 0],
        )
        wq_a_layer_last: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(
            wq_a, [D, Q_LORA], [weight_layer_last * D, 0]
        )
        wq_b_layer_last: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(
            wq_b, [Q_LORA, H * HEAD_DIM], [weight_layer_last * Q_LORA, 0]
        )
        wq_b_scale_layer_last: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(
            wq_b_scale,
            [H * HEAD_DIM],
            [weight_layer_last * H * HEAD_DIM],
        )
        wkv_layer_last: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(
            wkv, [D, HEAD_DIM], [weight_layer_last * D, 0]
        )
        gamma_cq_layer_last: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(
            gamma_cq, [Q_LORA], [weight_layer_last * Q_LORA]
        )
        gamma_ckv_layer_last: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(
            gamma_ckv, [HEAD_DIM], [weight_layer_last * HEAD_DIM]
        )
        wo_a_layer_last: pl.Tensor[
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16
        ] = pl.slice(
            wo_a,
            [LOCAL_O_GROUPS, O_LORA, O_GROUP_IN],
            [weight_layer_last * LOCAL_O_GROUPS, 0, 0],
        )
        routed_w1_layer_last: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w1,
            [N_LOCAL, MOE_INTER, D],
            [weight_layer_last * N_LOCAL, 0, 0],
        )
        routed_w3_layer_last: pl.Tensor[
            [N_LOCAL, MOE_INTER, D], pl.INT8
        ] = pl.slice(
            routed_w3,
            [N_LOCAL, MOE_INTER, D],
            [weight_layer_last * N_LOCAL, 0, 0],
        )
        routed_w2_layer_last: pl.Tensor[
            [N_LOCAL, D, MOE_INTER], pl.INT8
        ] = pl.slice(
            routed_w2,
            [N_LOCAL, D, MOE_INTER],
            [weight_layer_last * N_LOCAL, 0, 0],
        )
        raw_kv_layer_last = pl.slice(
            raw_kv_pool,
            [raw_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [model_layer_last * raw_blocks_per_layer, 0, 0, 0],
        )
        csa_state_layer_last = pl.slice(
            csa_compress_state,
            [
                csa_state_blocks_per_layer,
                CSA_MAIN_STATE_BLOCK_SIZE,
                CSA_MAIN_STATE_DIM,
            ],
            [csa_ordinal_last * csa_state_blocks_per_layer, 0, 0],
        )
        csa_cmp_kv_layer_last = pl.slice(
            csa_cmp_kv,
            [csa_cmp_blocks_per_layer, BLOCK_SIZE, 1, HEAD_DIM],
            [csa_ordinal_last * csa_cmp_blocks_per_layer, 0, 0, 0],
        )
        csa_inner_state_layer_last = pl.slice(
            csa_inner_compress_state,
            [
                csa_inner_state_blocks_per_layer,
                CSA_INNER_STATE_BLOCK_SIZE,
                CSA_INNER_STATE_DIM,
            ],
            [csa_ordinal_last * csa_inner_state_blocks_per_layer, 0, 0],
        )
        csa_idx_cache_layer_last = pl.slice(
            csa_idx_kv_cache,
            [
                csa_idx_blocks_per_layer,
                BLOCK_SIZE,
                1,
                CSA_IDX_HEAD_DIM,
            ],
            [csa_ordinal_last * csa_idx_blocks_per_layer, 0, 0, 0],
        )
        csa_idx_scale_layer_last = pl.slice(
            csa_idx_kv_scale,
            [csa_idx_blocks_per_layer, BLOCK_SIZE, 1, 1],
            [csa_ordinal_last * csa_idx_blocks_per_layer, 0, 0, 0],
        )
        hc_attn_scale_layer_last = pl.slice(
            hc_attn_scale, [3], [weight_layer_last * 3]
        )
        hc_attn_base_layer_last = pl.slice(
            hc_attn_base, [MIX_HC], [weight_layer_last * MIX_HC]
        )
        attn_norm_w_layer_last = pl.slice(
            attn_norm_w, [D], [weight_layer_last * D]
        )
        csa_cmp_wkv_layer_last = pl.slice(
            csa_cmp_wkv,
            [CSA_MAIN_OUT_DIM, D],
            [extra_layer_last * CSA_MAIN_OUT_DIM, 0],
        )
        csa_cmp_wgate_layer_last = pl.slice(
            csa_cmp_wgate,
            [CSA_MAIN_OUT_DIM, D],
            [extra_layer_last * CSA_MAIN_OUT_DIM, 0],
        )
        csa_cmp_ape_layer_last = pl.slice(
            csa_cmp_ape,
            [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM],
            [extra_layer_last * CSA_COMPRESS_RATIO, 0],
        )
        csa_cmp_norm_w_layer_last = pl.slice(
            csa_cmp_norm_w, [HEAD_DIM], [extra_layer_last * HEAD_DIM]
        )
        csa_idx_wq_b_layer_last = pl.slice(
            csa_idx_wq_b,
            [Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            [extra_layer_last * Q_LORA, 0],
        )
        csa_idx_wq_b_scale_layer_last = pl.slice(
            csa_idx_wq_b_scale,
            [CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
            [extra_layer_last * CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM],
        )
        csa_weights_proj_layer_last = pl.slice(
            csa_weights_proj,
            [D, CSA_IDX_N_HEADS],
            [extra_layer_last * D, 0],
        )
        csa_hadamard_idx_layer_last = pl.slice(
            csa_hadamard_idx,
            [CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM],
            [extra_layer_last * CSA_IDX_HEAD_DIM, 0],
        )
        csa_inner_wkv_layer_last = pl.slice(
            csa_inner_wkv,
            [CSA_INNER_OUT_DIM, D],
            [extra_layer_last * CSA_INNER_OUT_DIM, 0],
        )
        csa_inner_wgate_layer_last = pl.slice(
            csa_inner_wgate,
            [CSA_INNER_OUT_DIM, D],
            [extra_layer_last * CSA_INNER_OUT_DIM, 0],
        )
        csa_inner_ape_layer_last = pl.slice(
            csa_inner_ape,
            [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM],
            [extra_layer_last * CSA_COMPRESS_RATIO, 0],
        )
        csa_inner_norm_w_layer_last = pl.slice(
            csa_inner_norm_w,
            [CSA_IDX_HEAD_DIM],
            [extra_layer_last * CSA_IDX_HEAD_DIM],
        )
        attn_sink_layer_last = pl.slice(attn_sink, [H], [weight_layer_last * H])
        wo_b_layer_last = pl.slice(
            wo_b, [D, LOCAL_O_WIDTH], [weight_layer_last * D, 0]
        )
        wo_b_scale_layer_last = pl.slice(
            wo_b_scale, [D], [weight_layer_last * D]
        )
        hc_ffn_scale_layer_last = pl.slice(
            hc_ffn_scale, [3], [weight_layer_last * 3]
        )
        hc_ffn_base_layer_last = pl.slice(
            hc_ffn_base, [MIX_HC], [weight_layer_last * MIX_HC]
        )
        norm_w_layer_last = pl.slice(norm_w, [D], [weight_layer_last * D])
        gate_w_layer_last = pl.slice(
            gate_w,
            [N_EXPERTS_GLOBAL, D],
            [weight_layer_last * N_EXPERTS_GLOBAL, 0],
        )
        gate_bias_layer_last = pl.slice(
            gate_bias,
            [N_EXPERTS_GLOBAL],
            [weight_layer_last * N_EXPERTS_GLOBAL],
        )
        tid2eid_layer_last = pl.slice(
            tid2eid, [VOCAB, TOPK], [weight_layer_last * VOCAB, 0]
        )
        routed_w1_scale_layer_last = pl.slice(
            routed_w1_scale,
            [N_LOCAL, MOE_INTER],
            [weight_layer_last * N_LOCAL, 0],
        )
        routed_w3_scale_layer_last = pl.slice(
            routed_w3_scale,
            [N_LOCAL, MOE_INTER],
            [weight_layer_last * N_LOCAL, 0],
        )
        routed_w2_scale_layer_last = pl.slice(
            routed_w2_scale, [N_LOCAL, D], [weight_layer_last * N_LOCAL, 0]
        )
        shared_w1_layer_last = pl.slice(
            shared_w1, [MOE_INTER, D], [weight_layer_last * MOE_INTER, 0]
        )
        shared_w1_scale_layer_last = pl.slice(
            shared_w1_scale, [MOE_INTER], [weight_layer_last * MOE_INTER]
        )
        shared_w3_layer_last = pl.slice(
            shared_w3, [MOE_INTER, D], [weight_layer_last * MOE_INTER, 0]
        )
        shared_w3_scale_layer_last = pl.slice(
            shared_w3_scale, [MOE_INTER], [weight_layer_last * MOE_INTER]
        )
        shared_w2_layer_last = pl.slice(
            shared_w2, [D, MOE_INTER], [weight_layer_last * D, 0]
        )
        shared_w2_scale_layer_last = pl.slice(
            shared_w2_scale, [D], [weight_layer_last * D]
        )
        decode_layer_csa(
            x_ping,
            hc_attn_fn_layer_last, hc_attn_scale_layer_last,
            hc_attn_base_layer_last, attn_norm_w_layer_last,
            wq_a_layer_last, wq_b_layer_last, wq_b_scale_layer_last, wkv_layer_last,
            gamma_cq_layer_last, gamma_ckv_layer_last,
            freqs_cos, freqs_sin,
            csa_cmp_freqs_cos, csa_cmp_freqs_sin,
            csa_cmp_wkv_layer_last, csa_cmp_wgate_layer_last,
            csa_cmp_ape_layer_last, csa_cmp_norm_w_layer_last,
            csa_state_layer_last, csa_compress_state_block_table,
            csa_idx_wq_b_layer_last, csa_idx_wq_b_scale_layer_last,
            csa_weights_proj_layer_last, csa_hadamard_idx_layer_last,
            csa_inner_wkv_layer_last, csa_inner_wgate_layer_last,
            csa_inner_ape_layer_last, csa_inner_norm_w_layer_last,
            csa_inner_state_layer_last,
            csa_inner_compress_state_block_table,
            raw_kv_layer_last, csa_cmp_kv_layer_last, csa_cmp_block_table,
            csa_idx_cache_layer_last, csa_idx_scale_layer_last,
            csa_idx_block_table,
            csa_ori_slot_mapping, csa_window_swa_indices,
            csa_window_swa_lens, csa_cmp_slot_mapping,
            csa_idx_slot_mapping, csa_state_slot_mapping,
            csa_inner_state_slot_mapping, position_ids,
            csa_kv_seq_lens, attn_sink_layer_last,
            wo_a_layer_last, wo_b_layer_last, wo_b_scale_layer_last,
            hc_ffn_fn_layer_last, hc_ffn_scale_layer_last,
            hc_ffn_base_layer_last, norm_w_layer_last,
            gate_w_layer_last, gate_bias_layer_last, tid2eid_layer_last, input_ids,
            routed_w1_layer_last, routed_w1_scale_layer_last,
            routed_w3_layer_last, routed_w3_scale_layer_last,
            routed_w2_layer_last, routed_w2_scale_layer_last,
            shared_w1_layer_last, shared_w1_scale_layer_last,
            shared_w3_layer_last, shared_w3_scale_layer_last,
            shared_w2_layer_last, shared_w2_scale_layer_last,
            x_attn_active, x_moe_next, x_out,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            model_layer_last, group_base, tp_rank, local_t, my_rank,
            pl.const(43, pl.INT32),
        )
        clear_moe_signals(
            x_moe_next, arrived, data_arrived, combine_arrived
        )
    return x_out


@pl.jit.host
def l3_decode_fwd_full43(
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
    input_ids: pl.Tensor[[N_RANKS, MOE_TOKENS], pl.INT64],
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
    x_out: pl.Out[
        pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]
    ],
):
    """Allocate each communication window once and submit one Full43 child."""
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
    csa_inner_compress_state.bind_dynamic(
        1, FWD_CSA_INNER_STATE_BLOCKS_DYN
    )
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
    x_out.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer(
        [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16
    )
    attention_signal_buf = pld.alloc_window_buffer(
        [TP_SIZE, 1], dtype=pl.INT32
    )
    o_window_buf = pld.alloc_window_buffer(
        [O_WINDOW_ROWS, D], dtype=pl.FP32
    )
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer(
        [N_RANKS, N_LOCAL], dtype=pl.INT32
    )
    recv_x_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
    )
    recv_aux_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, 1], dtype=pl.INT32
    )
    routed_y_buf_buf = pld.alloc_window_buffer(
        [N_ROUTES, D], dtype=pl.BF16
    )
    combine_arrived_buf = pld.alloc_window_buffer(
        [N_RANKS, 1], dtype=pl.INT32
    )

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(
            attention_window_buf,
            [ATTENTION_WINDOW_ROWS, O_GROUP_IN],
            dtype=pl.BF16,
        )
        attention_signal = pld.window(
            attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        o_window = pld.window(
            o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.FP32
        )
        o_signal = pld.window(
            o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32
        )
        recv_meta = pld.window(
            recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32
        )
        recv_x = pld.window(
            recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8
        )
        recv_aux = pld.window(
            recv_aux_buf,
            [N_LOCAL * RECV_MAX, AUX_PAD],
            dtype=pl.FP32,
        )
        recv_route = pld.window(
            recv_route_buf,
            [N_LOCAL * RECV_MAX, IDX_PAD],
            dtype=pl.INT32,
        )
        arrived = pld.window(
            arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        data_arrived = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        routed_y_buf = pld.window(
            routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16
        )
        combine_arrived = pld.window(
            combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        decode_fwd_full43(
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
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_ping[rank], x_pong[rank],
            x_attn_active[rank], x_moe_next[rank], x_out[rank],
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            group_base, tp_rank, rank,
            device=rank,
        )
    return x_out


_COMMON_ATTN_WEIGHT_NAMES = (
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)

_HCA_EXTRA_WEIGHT_NAMES = (
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
)

_CSA_EXTRA_WEIGHT_NAMES = (
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "idx_wq_b",
    "idx_wq_b_scale",
    "weights_proj",
    "hadamard_idx",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
)

_MOE_NON_WEIGHT_NAMES = {"x_attn_moe", "x_moe_next", "input_ids"}

_MIXED_LAYERED_WEIGHT_NAMES = (
    *_COMMON_ATTN_WEIGHT_NAMES,
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
)

_SWA_METADATA_NAMES = (
    "freqs_cos",
    "freqs_sin",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "position_ids",
)

_CSA_PREFIX_SOURCES = {
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
    "csa_inner_compress_state_block_table": (
        "inner_compress_state_block_table"
    ),
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

_HCA_PREFIX_SOURCES = {
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

_METADATA_EXCLUDES = {
    "x_hc",
    "x_out",
    "kv_cache",
    "compress_state",
    "cmp_kv",
    "inner_compress_state",
    "idx_kv_cache",
    "idx_kv_scale",
    *_COMMON_ATTN_WEIGHT_NAMES,
    *_HCA_EXTRA_WEIGHT_NAMES,
    *_CSA_EXTRA_WEIGHT_NAMES,
}


def _stack_rank_shape(shape, layer_count):
    stacked = [int(dim) for dim in shape]
    if len(stacked) < 2 or stacked[0] != N_RANKS:
        raise ValueError(f"layer-stacked tensor must start with rank: {stacked}")
    stacked[1] *= int(layer_count)
    return stacked


def _collect_weight_shapes(reports, common_count, hca_count, csa_count):
    swa_shapes = reports["swa"]["distributed_shapes"]
    hca_shapes = reports["hca"]["distributed_shapes"]
    csa_shapes = reports["csa"]["distributed_shapes"]

    common = {}
    for name in _COMMON_ATTN_WEIGHT_NAMES:
        reference = swa_shapes[name]
        if hca_shapes[name] != reference or csa_shapes[name] != reference:
            raise ValueError(f"common attention weight shape diverged for {name}")
        common[name] = _stack_rank_shape(reference, common_count)

    moe_shapes = reports["swa"]["moe_shapes"]
    for name, shape in moe_shapes.items():
        if name not in _MOE_NON_WEIGHT_NAMES:
            common[name] = _stack_rank_shape(shape, common_count)

    return {
        "common": common,
        "hca": {
            name: _stack_rank_shape(hca_shapes[name], hca_count)
            for name in _HCA_EXTRA_WEIGHT_NAMES
        },
        "csa": {
            name: _stack_rank_shape(csa_shapes[name], csa_count)
            for name in _CSA_EXTRA_WEIGHT_NAMES
        },
    }


def _collect_metadata_shapes(reports):
    metadata = {}
    for kind, report in reports.items():
        for name, shape in report["distributed_shapes"].items():
            if name not in _METADATA_EXCLUDES:
                metadata[f"{kind}_{name}"] = list(shape)
    return metadata


def _collect_per_layer_pool_shapes(reports):
    raw_shapes = tuple(
        reports[kind]["distributed_shapes"]["kv_cache"]
        for kind in ("swa", "hca", "csa")
    )
    if raw_shapes[1:] != raw_shapes[:-1]:
        raise ValueError(f"raw KV pool shapes diverged by attention kind: {raw_shapes}")

    hca_shapes = reports["hca"]["distributed_shapes"]
    csa_shapes = reports["csa"]["distributed_shapes"]
    return {
        "raw_kv_pool": list(raw_shapes[0]),
        "hca_compress_state": list(hca_shapes["compress_state"]),
        "hca_cmp_kv": list(hca_shapes["cmp_kv"]),
        "csa_compress_state": list(csa_shapes["compress_state"]),
        "csa_cmp_kv": list(csa_shapes["cmp_kv"]),
        "csa_inner_compress_state": list(csa_shapes["inner_compress_state"]),
        "csa_idx_kv_cache": list(csa_shapes["idx_kv_cache"]),
        "csa_idx_kv_scale": list(csa_shapes["idx_kv_scale"]),
    }


def _validate_public_shapes(sections):
    maximum_dims = 0
    for section_name, shapes in sections.items():
        for name, shape in shapes.items():
            dims = len(shape)
            maximum_dims = max(maximum_dims, dims)
            if dims > MAX_PUBLIC_TENSOR_DIMS:
                raise ValueError(
                    f"{section_name} tensor {name!r} has {dims} dimensions: "
                    f"{shape}"
                )
            if any(int(dim) <= 0 for dim in shape):
                raise ValueError(
                    f"{section_name} tensor {name!r} has an invalid shape: {shape}"
                )
    return maximum_dims


def _make_mixed_layered_spec(name, source_specs):
    """Pack the four real layer fixtures along the rank-local data axis."""
    from golden import TensorSpec

    shape = list(source_specs[0].shape)
    shape[1] = sum(int(spec.shape[1]) for spec in source_specs)
    reference_tail = tuple(source_specs[0].shape[2:])
    for spec in source_specs:
        if spec.shape[0] != N_RANKS or tuple(spec.shape[2:]) != reference_tail:
            raise ValueError(f"mixed layer weight shape diverged for {name}")

    def init_value():
        import torch

        return torch.cat(
            [spec.create_tensor() for spec in source_specs], dim=1
        ).contiguous()

    packed = TensorSpec(name, shape, source_specs[0].dtype, init_value=init_value)
    packed.resident = "stacked"
    return packed


def _copy_prefix_spec(name, source, *, is_output=None):
    from golden import TensorSpec

    copied = TensorSpec(
        name,
        list(source.shape),
        source.dtype,
        init_value=source.init_value,
        is_output=source.is_output if is_output is None else is_output,
    )
    copied.resident = source.resident
    return copied


def build_mixed_prefix_tensor_specs(start_pos=None):
    """Build the rank-stacked SWA/SWA/CSA/HCA fixture in L3 ABI order."""
    import inspect

    import torch
    from golden import TensorSpec

    builders = (
        (layer.build_swa_layer_specs, 0),
        (layer.build_swa_layer_specs, 1),
        (layer.build_csa_layer_specs, 2),
        (layer.build_hca_layer_specs, 3),
    )
    layer_specs = []
    for builder, layer_id in builders:
        layer_specs.append({
            spec.name: spec
            for spec in builder(start_pos=start_pos, layer_id=layer_id)
            if isinstance(spec, TensorSpec)
        })
    swa0_specs, _swa1_specs, csa_specs, hca_specs = layer_specs
    local_t = int(swa0_specs["freqs_cos"].shape[1])
    if any(int(specs["freqs_cos"].shape[1]) != local_t for specs in layer_specs):
        raise ValueError("mixed attention fixtures disagree on active tokens")

    def init_raw_kv_pool():
        layers = []
        for ordinal, specs in enumerate(layer_specs):
            value = specs["kv_cache"].create_tensor()
            value = (value.float() + ordinal * 0.125).to(torch.bfloat16)
            layers.append(value)
        return torch.cat(layers, dim=1).contiguous()

    def init_input_ids():
        value = swa0_specs["input_ids"].create_tensor()
        return torch.remainder(value, MIXED_PREFIX_TEST_VOCAB).contiguous()

    def zero_active():
        return torch.zeros(
            N_RANKS, local_t, HC_MULT, D, dtype=torch.float32
        )

    specs_by_name = {
        "raw_kv_pool": TensorSpec(
            "raw_kv_pool",
            [
                N_RANKS,
                MIXED_PREFIX_LAYER_COUNT * swa0_specs["kv_cache"].shape[1],
                BLOCK_SIZE,
                1,
                HEAD_DIM,
            ],
            torch.bfloat16,
            init_value=init_raw_kv_pool,
            is_output=True,
        ),
        "input_ids": TensorSpec(
            "input_ids",
            [N_RANKS, MOE_TOKENS],
            torch.int64,
            init_value=init_input_ids,
        ),
        "x_ping": TensorSpec(
            "x_ping",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            init_value=swa0_specs["x_hc"].init_value,
            is_output=True,
        ),
        "x_pong": TensorSpec(
            "x_pong",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            init_value=zero_active,
            is_output=True,
        ),
        "x_attn_active": TensorSpec(
            "x_attn_active",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            init_value=zero_active,
            is_output=True,
        ),
        "x_moe_next": TensorSpec(
            "x_moe_next",
            [N_RANKS, MOE_TOKENS, HC_MULT, D],
            torch.float32,
            init_value=lambda: torch.zeros(
                N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32
            ),
            is_output=True,
        ),
        "x_out": TensorSpec(
            "x_out",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            is_output=True,
        ),
    }
    specs_by_name["raw_kv_pool"].resident = "stacked"

    for name in _MIXED_LAYERED_WEIGHT_NAMES:
        specs_by_name[name] = _make_mixed_layered_spec(
            name, [specs[name] for specs in layer_specs]
        )
    for name in _SWA_METADATA_NAMES:
        specs_by_name[name] = _copy_prefix_spec(name, swa0_specs[name])
    for public_name, source_name in _CSA_PREFIX_SOURCES.items():
        specs_by_name[public_name] = _copy_prefix_spec(
            public_name, csa_specs[source_name]
        )
    for public_name, source_name in _HCA_PREFIX_SOURCES.items():
        specs_by_name[public_name] = _copy_prefix_spec(
            public_name, hca_specs[source_name]
        )

    parameter_names = [
        name
        for name, parameter in inspect.signature(
            l3_decode_fwd._func
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in parameter_names if name not in specs_by_name]
    extra = [name for name in specs_by_name if name not in parameter_names]
    if missing or extra:
        raise ValueError(
            f"mixed-prefix spec/signature mismatch: missing={missing}, extra={extra}"
        )
    specs = [specs_by_name[name] for name in parameter_names]
    for spec in specs:
        if isinstance(spec, TensorSpec) and len(spec.shape) > MAX_PUBLIC_TENSOR_DIMS:
            raise ValueError(
                f"mixed-prefix tensor {spec.name!r} exceeds "
                f"{MAX_PUBLIC_TENSOR_DIMS} dimensions: {spec.shape}"
            )
    return specs


def _make_full43_weight_bank_spec(
    name, source, bank_size, *, compile_only=False
):
    """Pack a static layer bank along the first rank-local data axis."""
    from golden import TensorSpec

    storage_shape = list(source.shape[1:])
    pad_hc_fn = name in {"hc_attn_fn", "hc_ffn_fn"}
    if pad_hc_fn:
        if storage_shape[0] != MIX_HC:
            raise ValueError(f"unexpected HC function shape for {name}")
        storage_shape[0] = HC_FN_STORAGE_ROWS
    shape = [
        N_RANKS,
        int(bank_size) * int(storage_shape[0]),
        *storage_shape[1:],
    ]

    def init_value():
        import torch

        value = source.create_tensor()
        if pad_hc_fn:
            padded = torch.zeros(
                N_RANKS, HC_FN_STORAGE_ROWS, HC_DIM, dtype=value.dtype
            )
            padded[:, :MIX_HC].copy_(value)
            value = padded
        repeats = [1, int(bank_size)] + [1] * (value.ndim - 2)
        return value.repeat(*repeats).contiguous()

    spec = TensorSpec(
        name,
        shape,
        source.dtype,
        init_value=0 if compile_only else init_value,
    )
    spec.resident = "stacked"
    return spec


def _make_full43_packed_pool_spec(
    name, source, layer_count, *, sentinel=False
):
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
            packed[:, ordinal * extent : (ordinal + 1) * extent].fill_(
                ordinal + 1
            )
        return packed

    spec = TensorSpec(
        name, shape, source.dtype, init_value=init_value, is_output=True
    )
    spec.resident = "stacked"
    return spec


def build_full43_tensor_specs(
    start_pos=None,
    *,
    weight_bank_size=FULL43_RUNTIME_WEIGHT_BANK,
    runtime_case="full_active",
):
    """Build the production or bounded-runtime Full43 L3 fixture."""
    import inspect

    import torch
    from golden import TensorSpec

    if weight_bank_size != FWD_WEIGHT_BANK_SIZE:
        raise ValueError(
            "weight bank froze at module import as "
            f"{FWD_WEIGHT_BANK_SIZE}, got {weight_bank_size}"
        )
    compile_only = runtime_case is None
    if not compile_only and weight_bank_size != FULL43_RUNTIME_WEIGHT_BANK:
        raise ValueError("Full43 runtime witnesses use one reusable weight bank")
    if runtime_case not in {
        None,
        "full_active",
        "packed_pool_sentinel",
        "long_context_tail",
    }:
        raise ValueError(f"unknown Full43 runtime case: {runtime_case!r}")

    if runtime_case == "long_context_tail" and start_pos is None:
        start_pos = [0, 0, 0, 1048568]

    def tensor_specs(builder, layer_id):
        return {
            spec.name: spec
            for spec in builder(start_pos=start_pos, layer_id=layer_id)
            if isinstance(spec, TensorSpec)
        }

    swa_specs = tensor_specs(layer.build_swa_layer_specs, 0)
    csa_specs = tensor_specs(layer.build_csa_layer_specs, 2)
    hca_specs = tensor_specs(layer.build_hca_layer_specs, 3)
    local_t = int(swa_specs["freqs_cos"].shape[1])
    if int(csa_specs["freqs_cos"].shape[1]) != local_t:
        raise ValueError("CSA and SWA Full43 fixtures disagree on active rows")
    if int(hca_specs["freqs_cos"].shape[1]) != local_t:
        raise ValueError("HCA and SWA Full43 fixtures disagree on active rows")

    def zero_active():
        return torch.zeros(
            N_RANKS, local_t, HC_MULT, D, dtype=torch.float32
        )

    def init_input_ids():
        value = swa_specs["input_ids"].create_tensor()
        return torch.remainder(value, MIXED_PREFIX_TEST_VOCAB).contiguous()

    sentinel = runtime_case == "packed_pool_sentinel"
    specs_by_name = {
        "raw_kv_pool": _make_full43_packed_pool_spec(
            "raw_kv_pool",
            swa_specs["kv_cache"],
            MAIN_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "hca_compress_state": _make_full43_packed_pool_spec(
            "hca_compress_state",
            hca_specs["compress_state"],
            HCA_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "hca_cmp_kv": _make_full43_packed_pool_spec(
            "hca_cmp_kv",
            hca_specs["cmp_kv"],
            HCA_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "csa_compress_state": _make_full43_packed_pool_spec(
            "csa_compress_state",
            csa_specs["compress_state"],
            CSA_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "csa_cmp_kv": _make_full43_packed_pool_spec(
            "csa_cmp_kv",
            csa_specs["cmp_kv"],
            CSA_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "csa_inner_compress_state": _make_full43_packed_pool_spec(
            "csa_inner_compress_state",
            csa_specs["inner_compress_state"],
            CSA_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "csa_idx_kv_cache": _make_full43_packed_pool_spec(
            "csa_idx_kv_cache",
            csa_specs["idx_kv_cache"],
            CSA_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "csa_idx_kv_scale": _make_full43_packed_pool_spec(
            "csa_idx_kv_scale",
            csa_specs["idx_kv_scale"],
            CSA_LAYER_COUNT,
            sentinel=sentinel,
        ),
        "input_ids": TensorSpec(
            "input_ids",
            [N_RANKS, MOE_TOKENS],
            torch.int64,
            init_value=init_input_ids,
        ),
        "x_ping": TensorSpec(
            "x_ping",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            init_value=swa_specs["x_hc"].init_value,
            is_output=True,
        ),
        "x_pong": TensorSpec(
            "x_pong",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            init_value=zero_active,
            is_output=True,
        ),
        "x_attn_active": TensorSpec(
            "x_attn_active",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            init_value=zero_active,
            is_output=True,
        ),
        "x_moe_next": TensorSpec(
            "x_moe_next",
            [N_RANKS, MOE_TOKENS, HC_MULT, D],
            torch.float32,
            init_value=lambda: torch.zeros(
                N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32
            ),
            is_output=True,
        ),
        "x_out": TensorSpec(
            "x_out",
            [N_RANKS, local_t, HC_MULT, D],
            torch.float32,
            is_output=True,
        ),
    }

    for name in _MIXED_LAYERED_WEIGHT_NAMES:
        specs_by_name[name] = _make_full43_weight_bank_spec(
            name,
            swa_specs[name],
            weight_bank_size,
            compile_only=compile_only,
        )
    for name in _SWA_METADATA_NAMES:
        specs_by_name[name] = _copy_prefix_spec(name, swa_specs[name])

    csa_weight_names = {
        f"csa_{name}": name for name in _CSA_EXTRA_WEIGHT_NAMES
    }
    hca_weight_names = {
        f"hca_{name}": name for name in _HCA_EXTRA_WEIGHT_NAMES
    }
    for public_name, source_name in csa_weight_names.items():
        specs_by_name[public_name] = _make_full43_weight_bank_spec(
            public_name,
            csa_specs[source_name],
            CSA_LAYER_COUNT if compile_only else FULL43_RUNTIME_WEIGHT_BANK,
            compile_only=compile_only,
        )
    for public_name, source_name in hca_weight_names.items():
        specs_by_name[public_name] = _make_full43_weight_bank_spec(
            public_name,
            hca_specs[source_name],
            HCA_LAYER_COUNT if compile_only else FULL43_RUNTIME_WEIGHT_BANK,
            compile_only=compile_only,
        )

    packed_names = set(PACKED_POOL_LAYER_COUNTS)
    for public_name, source_name in _CSA_PREFIX_SOURCES.items():
        if public_name in packed_names or public_name in csa_weight_names:
            continue
        specs_by_name[public_name] = _copy_prefix_spec(
            public_name, csa_specs[source_name]
        )
    for public_name, source_name in _HCA_PREFIX_SOURCES.items():
        if public_name in packed_names or public_name in hca_weight_names:
            continue
        specs_by_name[public_name] = _copy_prefix_spec(
            public_name, hca_specs[source_name]
        )

    parameter_names = list(
        inspect.signature(l3_decode_fwd_full43._func).parameters
    )
    missing = [name for name in parameter_names if name not in specs_by_name]
    extra = [name for name in specs_by_name if name not in parameter_names]
    if missing or extra:
        raise ValueError(
            f"Full43 spec/signature mismatch: missing={missing}, extra={extra}"
        )
    specs = [specs_by_name[name] for name in parameter_names]
    for spec in specs:
        if len(spec.shape) > MAX_PUBLIC_TENSOR_DIMS:
            raise ValueError(
                f"Full43 tensor {spec.name!r} exceeds "
                f"{MAX_PUBLIC_TENSOR_DIMS} dimensions: {spec.shape}"
            )
    return specs


def golden_full43_runtime(_tensors):
    """Full43 is a topology/isolation witness; layer math is gated earlier."""


def finite_full43_tensor_compare(actual, _expected, **_kwargs):
    """Require a completed finite device result without duplicating 43 goldens."""
    import torch

    if actual.numel() == 0:
        return False, "    Full43 output is empty"
    if actual.is_floating_point() and not bool(torch.isfinite(actual).all()):
        return False, "    Full43 output contains NaN or Inf"
    return True, ""


def _full43_pool_compare(layer_mappings, layer_count, block_size, pool_name):
    """Check every packed layer slice and reject writes outside local slots."""
    def compare(actual, expected, **kwargs):
        import torch

        if actual.shape != expected.shape:
            return False, (
                f"    {pool_name} shape mismatch: "
                f"actual={tuple(actual.shape)} expected={tuple(expected.shape)}"
            )
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
                    return False, (
                        f"    {pool_name} layer={ordinal} rank={rank} "
                        f"contains out-of-range slot {first}"
                    )
                slots = slots[slots >= 0]
                target = torch.zeros(
                    actual_rows.shape[0], dtype=torch.bool, device=actual.device
                )
                if slots.numel():
                    target[slots] = True
                if not torch.equal(actual_rows[~target], expected_rows[~target]):
                    return False, (
                        f"    {pool_name} layer={ordinal} rank={rank} "
                        "modified an unmapped row"
                    )
                selected = actual_rows[target]
                if selected.is_floating_point() and not bool(
                    torch.isfinite(selected).all()
                ):
                    return False, (
                        f"    {pool_name} layer={ordinal} rank={rank} "
                        "wrote NaN or Inf"
                    )
                sentinel_value = ordinal + 1
                if slots.numel() and bool(
                    torch.all(expected_rows == sentinel_value)
                ) and torch.equal(selected, expected_rows[target]):
                    return False, (
                        f"    {pool_name} layer={ordinal} rank={rank} "
                        "did not update any mapped sentinel row"
                    )
        return True, ""

    return compare


def full43_compare_functions():
    """Return the Full43 runtime isolation and completion comparators."""
    raw_mappings = tuple(
        "swa_slot_mapping"
        if entry["kind"] == "swa"
        else f"{entry['kind']}_ori_slot_mapping"
        for entry in FULL43_LAYER_PLAN
    )
    csa_mappings = ("csa_state_slot_mapping",) * CSA_LAYER_COUNT
    csa_cmp_mappings = ("csa_cmp_slot_mapping",) * CSA_LAYER_COUNT
    csa_inner_mappings = (
        "csa_inner_state_slot_mapping",
    ) * CSA_LAYER_COUNT
    csa_idx_mappings = ("csa_idx_slot_mapping",) * CSA_LAYER_COUNT
    hca_state_mappings = ("hca_state_slot_mapping",) * HCA_LAYER_COUNT
    hca_cmp_mappings = ("hca_cmp_slot_mapping",) * HCA_LAYER_COUNT
    finite_names = {
        "x_ping",
        "x_pong",
        "x_attn_active",
        "x_moe_next",
        "x_out",
    }
    compare = {name: finite_full43_tensor_compare for name in finite_names}
    compare.update({
        "raw_kv_pool": _full43_pool_compare(
            raw_mappings, MAIN_LAYER_COUNT, BLOCK_SIZE, "Full43 raw KV"
        ),
        "hca_compress_state": _full43_pool_compare(
            hca_state_mappings,
            HCA_LAYER_COUNT,
            HCA_COMPRESS_STATE_BLOCK_SIZE,
            "Full43 HCA state",
        ),
        "hca_cmp_kv": _full43_pool_compare(
            hca_cmp_mappings,
            HCA_LAYER_COUNT,
            BLOCK_SIZE,
            "Full43 HCA compressed KV",
        ),
        "csa_compress_state": _full43_pool_compare(
            csa_mappings,
            CSA_LAYER_COUNT,
            CSA_MAIN_STATE_BLOCK_SIZE,
            "Full43 CSA main state",
        ),
        "csa_cmp_kv": _full43_pool_compare(
            csa_cmp_mappings,
            CSA_LAYER_COUNT,
            BLOCK_SIZE,
            "Full43 CSA compressed KV",
        ),
        "csa_inner_compress_state": _full43_pool_compare(
            csa_inner_mappings,
            CSA_LAYER_COUNT,
            CSA_INNER_STATE_BLOCK_SIZE,
            "Full43 CSA inner state",
        ),
        "csa_idx_kv_cache": _full43_pool_compare(
            csa_idx_mappings,
            CSA_LAYER_COUNT,
            BLOCK_SIZE,
            "Full43 CSA index KV",
        ),
        "csa_idx_kv_scale": _full43_pool_compare(
            csa_idx_mappings,
            CSA_LAYER_COUNT,
            BLOCK_SIZE,
            "Full43 CSA index scale",
        ),
    })
    return compare


def build_full43_shape_report(
    start_pos=None, *, weight_bank_size=FWD_WEIGHT_BANK_SIZE
):
    """Return concrete Full43 device ABI shapes without materializing values."""
    runtime_case = "full_active" if weight_bank_size == 1 else None
    specs = build_full43_tensor_specs(
        start_pos,
        weight_bank_size=weight_bank_size,
        runtime_case=runtime_case,
    )
    shapes = {spec.name: list(spec.shape) for spec in specs}
    return {
        "weight_bank_size": weight_bank_size,
        "active_tokens": shapes["freqs_cos"][1],
        "raw_blocks_per_layer": shapes["raw_kv_pool"][1] // MAIN_LAYER_COUNT,
        "maximum_public_tensor_dims": max(len(shape) for shape in shapes.values()),
        "shapes": shapes,
    }


def _mixed_layer_value(value, layer_ordinal):
    extent = value.shape[1] // MIXED_PREFIX_LAYER_COUNT
    begin = layer_ordinal * extent
    return value[:, begin : begin + extent]


def golden_mixed_prefix_decode_fwd(tensors):
    """Compose the unchanged four production layer goldens in model order."""
    import torch

    local_t = int(tensors["freqs_cos"].shape[1])
    current = tensors["x_ping"].clone()
    raw_blocks_per_layer = (
        tensors["raw_kv_pool"].shape[1] // MIXED_PREFIX_LAYER_COUNT
    )
    layer_kinds = ("swa", "swa", "csa", "hca")
    kind_sources = {"csa": _CSA_PREFIX_SOURCES, "hca": _HCA_PREFIX_SOURCES}
    golden_fns = {
        "swa": layer.golden_decode_layer_swa,
        "csa": layer.golden_decode_layer_csa,
        "hca": layer.golden_decode_layer_hca,
    }

    for layer_ordinal, kind in enumerate(layer_kinds):
        raw_begin = layer_ordinal * raw_blocks_per_layer
        x_attn_active = torch.zeros_like(current)
        x_moe_next = torch.zeros(
            N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32
        )
        x_next = torch.zeros_like(current)
        layer_tensors = {
            name: _mixed_layer_value(tensors[name], layer_ordinal)
            for name in _MIXED_LAYERED_WEIGHT_NAMES
        }
        layer_tensors.update({
            "x_hc": current,
            "freqs_cos": tensors["freqs_cos"],
            "freqs_sin": tensors["freqs_sin"],
            "position_ids": tensors["position_ids"],
            "kv_cache": tensors["raw_kv_pool"][
                :, raw_begin : raw_begin + raw_blocks_per_layer
            ],
            "input_ids": tensors["input_ids"],
            "x_attn_active": x_attn_active,
            "x_moe_next": x_moe_next,
            "x_next": x_next,
            "layer_id": layer_ordinal,
            "local_t": local_t,
        })
        if kind == "swa":
            layer_tensors.update({
                name: tensors[name]
                for name in ("swa_slot_mapping", "swa_indices", "swa_lens")
            })
        else:
            layer_tensors.update({
                source_name: tensors[public_name]
                for public_name, source_name in kind_sources[kind].items()
            })
        golden_fns[kind](layer_tensors)
        current = x_next.clone()
        tensors["x_attn_active"].copy_(x_attn_active)
        tensors["x_moe_next"].copy_(x_moe_next)
        if layer_ordinal == 0:
            tensors["x_pong"].copy_(current)
        elif layer_ordinal == 1:
            tensors["x_ping"].copy_(current)
        elif layer_ordinal == 2:
            tensors["x_pong"].copy_(current)
    tensors["x_out"].copy_(current)


def mixed_packed_raw_pool_compare(actual, expected, **kwargs):
    """Check all four local raw mappings and every untouched cache row."""
    from golden import mapped_pool_ratio_allclose

    if actual.shape != expected.shape:
        return False, (
            f"    raw pool shape mismatch: actual={tuple(actual.shape)} "
            f"expected={tuple(expected.shape)}"
        )
    if actual.shape[1] % MIXED_PREFIX_LAYER_COUNT:
        return False, "    packed raw pool is not divisible by four layers"

    per_layer_blocks = actual.shape[1] // MIXED_PREFIX_LAYER_COUNT
    mapping_names = (
        "swa_slot_mapping",
        "swa_slot_mapping",
        "csa_ori_slot_mapping",
        "hca_ori_slot_mapping",
    )
    inputs = kwargs.get("inputs", {})
    for layer_ordinal, mapping_name in enumerate(mapping_names):
        mapping = inputs.get(mapping_name)
        if mapping is None:
            return False, f"    compare_fn missing input {mapping_name!r}"
        compare_layer = mapped_pool_ratio_allclose(
            mapping_name,
            mapping_shape=tuple(mapping.shape),
            block_size=BLOCK_SIZE,
            leading_rank_axis=True,
            pool_name=f"mixed layer {layer_ordinal} raw KV cache",
            atol=1e-4 if layer_ordinal == 0 else 0.01,
            rtol=1.0 / 128 if layer_ordinal == 0 else 0.08,
            max_error_ratio=0.005 if layer_ordinal == 0 else 0.08,
        )
        begin = layer_ordinal * per_layer_blocks
        ok, detail = compare_layer(
            actual[:, begin : begin + per_layer_blocks],
            expected[:, begin : begin + per_layer_blocks],
            **kwargs,
        )
        if not ok:
            return False, f"    layer {layer_ordinal}:\n{detail}"
    return True, ""


def build_mixed_prefix_shape_report(start_pos=None):
    """Return concrete mixed-prefix L3 shapes without materializing tensors."""
    specs = build_mixed_prefix_tensor_specs(start_pos=start_pos)
    shapes = {
        spec.name: list(spec.shape)
        for spec in specs
        if hasattr(spec, "shape")
    }
    return {
        "tp": TP_SIZE,
        "ep": EP_SIZE,
        "active_tokens": shapes["freqs_cos"][1],
        "raw_blocks_per_layer": (
            shapes["raw_kv_pool"][1] // MIXED_PREFIX_LAYER_COUNT
        ),
        "maximum_public_tensor_dims": max(len(shape) for shape in shapes.values()),
        "shapes": shapes,
    }


def build_decode_fwd_shape_report(start_pos=None):
    """Build production/runtime shape skeletons without building a device graph."""
    reports = {
        "swa": layer.build_layer_shape_report(0, start_pos=start_pos),
        "csa": layer.build_layer_shape_report(2, start_pos=start_pos),
        "hca": layer.build_layer_shape_report(3, start_pos=start_pos),
    }
    active_tokens = reports["swa"]["active_tokens"]
    if any(report["active_tokens"] != active_tokens for report in reports.values()):
        raise ValueError("attention kinds disagree on the active-token extent")

    per_layer_pool_shapes = _collect_per_layer_pool_shapes(reports)
    packed_layout = build_packed_pool_layout(per_layer_pool_shapes)
    packed_pool_shapes = {
        name: entry["packed_shape"] for name, entry in packed_layout.items()
    }
    metadata_shapes = _collect_metadata_shapes(reports)

    preamble_shapes = {
        "input_ids": [N_RANKS, MOE_TOKENS],
        "embed_weight": [N_RANKS, MODEL_CONFIG.vocab_size, D],
    }
    workspace_shapes = {
        "embedding_hidden": [N_RANKS, active_tokens, D],
        "x_ping": [N_RANKS, active_tokens, HC_MULT, D],
        "x_pong": [N_RANKS, active_tokens, HC_MULT, D],
        "x_attn_active": [N_RANKS, active_tokens, HC_MULT, D],
        "x_moe_next": [N_RANKS, MOE_TOKENS, HC_MULT, D],
    }
    terminal_shapes = {
        "hc_head_fn": [N_RANKS, HC_MULT, HC_DIM],
        "hc_head_scale": [N_RANKS, 1],
        "hc_head_base": [N_RANKS, HC_MULT],
        "final_norm_w": [N_RANKS, D],
        "lm_head_weight": [N_RANKS, VOCAB_PER_TP, D],
        "hidden_out": [N_RANKS, active_tokens, D],
        "logit_row_indices": [N_RANKS, MAX_LOGIT_ROWS],
        "logits": [N_RANKS, MAX_LOGIT_ROWS, MODEL_CONFIG.vocab_size],
        "sampled_ids": [N_RANKS, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD],
    }

    production_weights = _collect_weight_shapes(
        reports,
        MAIN_LAYER_COUNT,
        HCA_LAYER_COUNT,
        CSA_LAYER_COUNT,
    )
    runtime_weights = _collect_weight_shapes(reports, 1, 1, 1)

    common_sections = {
        "preamble": preamble_shapes,
        "metadata": metadata_shapes,
        "packed_pools": packed_pool_shapes,
        "workspaces": workspace_shapes,
        "terminal": terminal_shapes,
    }
    production_sections = {
        **common_sections,
        "common_weights": production_weights["common"],
        "hca_weights": production_weights["hca"],
        "csa_weights": production_weights["csa"],
    }
    runtime_sections = {
        **common_sections,
        "common_weights": runtime_weights["common"],
        "hca_weights": runtime_weights["hca"],
        "csa_weights": runtime_weights["csa"],
    }
    maximum_dims = max(
        _validate_public_shapes(production_sections),
        _validate_public_shapes(runtime_sections),
    )

    return {
        "tp": TP_SIZE,
        "ep": EP_SIZE,
        "active_batch": reports["swa"]["active_batch"],
        "active_tokens": active_tokens,
        "layer_plan": FULL43_LAYER_PLAN,
        "packed_layout": packed_layout,
        "packed_pool_bytes_per_rank": sum(
            entry["bytes_per_rank"] for entry in packed_layout.values()
        ),
        "weight_bank_layers": {
            "production": {
                "common": MAIN_LAYER_COUNT,
                "hca": HCA_LAYER_COUNT,
                "csa": CSA_LAYER_COUNT,
            },
            "runtime": {"common": 1, "hca": 1, "csa": 1},
        },
        "production": production_sections,
        "runtime": runtime_sections,
        "maximum_public_tensor_dims": maximum_dims,
    }


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

    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 D-Spark decode-forward integration"
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=("a2a3", "a2a3sim", "a5", "a5sim"),
    )
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=_TP_CHOICES)
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=_EP_CHOICES)
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help=f"comma-separated device ids; EP={EP_SIZE} needs {EP_SIZE}",
    )
    parser.add_argument(
        "--start-pos",
        type=str,
        default=None,
        help="a scalar selects batch=1; a comma-separated list sets the batch",
    )
    parser.add_argument("--shape-only", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument(
        "--weight-bank-size",
        type=int,
        default=FWD_WEIGHT_BANK_SIZE,
        choices=(1, MAIN_LAYER_COUNT),
        help="1 reuses weights at runtime; 43 builds production layer banks",
    )
    parser.add_argument(
        "--runtime-case",
        type=str,
        default="full_active",
        choices=(
            "full_active",
            "packed_pool_sentinel",
            "long_context_tail",
        ),
    )
    parser.add_argument(
        "--enable-scope-stats", action="store_true", default=False
    )
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default=None)
    args = parser.parse_args()

    if args.tp != TP_SIZE or args.ep != EP_SIZE:
        parser.error(
            f"parallel sizes froze at import as TP={TP_SIZE}, EP={EP_SIZE}"
        )
    start_pos = _parse_start_pos(args.start_pos)
    weight_bank_size = args.weight_bank_size
    if weight_bank_size != FWD_WEIGHT_BANK_SIZE:
        parser.error(
            "weight bank froze at import as "
            f"{FWD_WEIGHT_BANK_SIZE}, got {weight_bank_size}"
        )
    if not args.compile_only and not args.shape_only and weight_bank_size != 1:
        parser.error("Full43 runtime requires --weight-bank-size 1")
    if args.shape_only:
        report = build_decode_fwd_shape_report(start_pos)
        full43_report = build_full43_shape_report(
            start_pos, weight_bank_size=weight_bank_size
        )
        packed_summary = ", ".join(
            f"{name}={entry['total_extent']}"
            for name, entry in report["packed_layout"].items()
        )
        print(
            f"TP={report['tp']} EP={report['ep']} "
            f"batch={report['active_batch']} active_t={report['active_tokens']} "
            f"layers={len(report['layer_plan'])} "
            f"max_dims={report['maximum_public_tensor_dims']}"
        )
        print(f"packed_blocks: {packed_summary}")
        print(
            "packed_pool_gib_per_rank: "
            f"{report['packed_pool_bytes_per_rank'] / (1024 ** 3):.3f}"
        )
        print(
            "full43_device: "
            f"weight_bank={full43_report['weight_bank_size']} "
            f"active_t={full43_report['active_tokens']} "
            f"raw_blocks_per_layer={full43_report['raw_blocks_per_layer']} "
            f"max_dims={full43_report['maximum_public_tensor_dims']}"
        )
        return

    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(EP_SIZE))
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(
            f"--device must be a comma-separated integer list, got "
            f"{args.device!r}"
        )
    if len(device_ids) != EP_SIZE:
        parser.error(
            f"EP={EP_SIZE} needs exactly {EP_SIZE} devices, got {device_ids}"
        )
    if len(set(device_ids)) != len(device_ids) or any(
        device < 0 for device in device_ids
    ):
        parser.error(f"device IDs must be distinct and non-negative: {device_ids}")

    runtime_case = None if weight_bank_size == MAIN_LAYER_COUNT else args.runtime_case
    specs = build_full43_tensor_specs(
        start_pos=start_pos,
        weight_bank_size=weight_bank_size,
        runtime_case=runtime_case,
    )
    result = run_jit(
        fn=l3_decode_fwd_full43,
        specs=specs,
        golden_fn=golden_full43_runtime,
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
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn=full43_compare_functions(),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
