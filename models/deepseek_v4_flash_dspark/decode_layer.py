# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2  # CI: 2-card run; borrows 2 cards via task-submit --device-num
# ci: no-sim    # CI marker: distributed communication oracle requires real devices
"""DeepSeek-V4 D-Spark decode-layer integration."""

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


# TP/EP-derived leaf shapes freeze at import time. Select both worlds before
# importing attention, output-projection, gate, or expert modules.
TP_SIZE = _parse_parallel_arg("tp", _TP_DEFAULT)
EP_SIZE = _parse_parallel_arg("ep", _EP_DEFAULT)
if TP_SIZE not in _TP_CHOICES:
    raise ValueError(f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})")
if EP_SIZE not in _EP_CHOICES:
    raise ValueError(f"--ep must be one of {_EP_CHOICES} (got {EP_SIZE})")
if EP_SIZE % TP_SIZE != 0:
    raise ValueError(f"EP={EP_SIZE} must be divisible by TP={TP_SIZE}")

config.TP = TP_SIZE
config.EP = EP_SIZE

import decode_csa as csa
import decode_hca as hca
import decode_swa as swa
import moe as moe_module
import pypto.language as pl
import pypto.language.distributed as pld
from decode_hca import decode_hca_attention
from decode_csa import decode_csa, decode_csa_tp1
from decode_swa import decode_swa_attention
from moe import clear_moe_signals, moe


MODEL_CONFIG = config.FLASH
DECODE_SEQ = config.DECODE_SEQ
MOE_TOKENS = moe_module.T
TP_GROUPS = EP_SIZE // TP_SIZE
MAX_PUBLIC_TENSOR_DIMS = 5

# SWA and MoE use the same hidden-state layout. SWA has a dynamic active-token
# extent, while MoE keeps its fixed per-rank capacity and consumes only the
# active prefix selected by ``num_tokens``.
T_DYN = swa.T_DYN
CP_T_DYN = T_DYN if TP_SIZE == 1 else swa.CP_T_DYN
ORI_BLOCK_NUM_DYN = swa.ORI_BLOCK_NUM_DYN
D = swa.D
H = swa.H
HEAD_DIM = swa.HEAD_DIM
ROPE_HEAD_DIM = swa.ROPE_HEAD_DIM
Q_LORA = swa.Q_LORA
WIN = swa.WIN
HC_MULT = swa.HC_MULT
MIX_HC = swa.MIX_HC
HC_DIM = swa.HC_DIM
O_LORA = swa.O_LORA
O_GROUP_IN = swa.O_GROUP_IN
LOCAL_O_GROUPS = swa.LOCAL_O_GROUPS
LOCAL_O_WIDTH = swa.LOCAL_O_WIDTH
BLOCK_SIZE = swa.BLOCK_SIZE
ATTENTION_WINDOW_ROWS = swa.ATTENTION_WINDOW_ROWS
O_WINDOW_ROWS = swa.O_WINDOW_ROWS
DECODE_GROUP_CAP = swa.DECODE_GROUP_CAP

# HCA keeps the common hidden/attention layout above and adds ratio-128
# compressor pools and request-axis metadata.
HCA_B_DYN = hca.B_DYN
HCA_CP_B_DYN = hca.B_DYN if TP_SIZE == 1 else hca.CP_B_DYN
HCA_CP_T_DYN = hca.T_DYN if TP_SIZE == 1 else hca.CP_T_DYN
HCA_ORI_BLOCK_NUM_DYN = hca.ORI_BLOCK_NUM_DYN
HCA_CMP_BLOCK_NUM_DYN = hca.CMP_BLOCK_NUM_DYN
HCA_CMP_TABLE_BLOCKS_DYN = hca.CMP_TABLE_BLOCKS_DYN
HCA_COMPRESS_STATE_BLOCK_NUM_DYN = hca.COMPRESS_STATE_BLOCK_NUM_DYN
HCA_MAIN_OUT_DIM = hca.MAIN_OUT_DIM
HCA_COMPRESS_RATIO = hca.COMPRESS_RATIO
HCA_COMPRESS_STATE_BLOCK_SIZE = hca.COMPRESS_STATE_BLOCK_SIZE
HCA_COMPRESS_STATE_MAX_BLOCKS = hca.COMPRESS_STATE_MAX_BLOCKS
HCA_COMPRESS_STATE_DIM = hca.COMPRESS_STATE_DIM

# CSA keeps the common hidden/attention layout above and adds ratio-4 main
# compressor, indexer compressor, and quantized index-cache pools.
CSA_B_DYN = csa.B_DYN
CSA_CMP_BLOCK_NUM_DYN = csa.CMP_BLOCK_NUM_DYN
CSA_IDX_CACHE_BLOCK_NUM_DYN = csa.IDX_CACHE_BLOCK_NUM_DYN
CSA_MAIN_STATE_BLOCK_NUM_DYN = csa.MAIN_STATE_BLOCK_NUM_DYN
CSA_INNER_STATE_BLOCK_NUM_DYN = csa.INNER_STATE_BLOCK_NUM_DYN
CSA_B = csa.B
CSA_IDX_N_HEADS = csa.IDX_N_HEADS
CSA_IDX_HEAD_DIM = csa.IDX_HEAD_DIM
CSA_COMPRESS_RATIO = csa.COMPRESS_RATIO
CSA_MAIN_OUT_DIM = csa.MAIN_OUT_DIM
CSA_MAIN_STATE_BLOCK_SIZE = csa.MAIN_STATE_BLOCK_SIZE
CSA_MAIN_STATE_MAX_BLOCKS = csa.MAIN_STATE_MAX_BLOCKS
CSA_MAIN_STATE_DIM = csa.MAIN_STATE_DIM
CSA_INNER_OUT_DIM = csa.INNER_OUT_DIM
CSA_INNER_STATE_BLOCK_SIZE = csa.INNER_STATE_BLOCK_SIZE
CSA_INNER_STATE_MAX_BLOCKS = csa.INNER_STATE_MAX_BLOCKS
CSA_INNER_STATE_DIM = csa.INNER_STATE_DIM
CSA_CMP_MAX_BLOCKS = csa.CMP_MAX_BLOCKS
CSA_IDX_MAX_BLOCKS = csa.IDX_MAX_BLOCKS

N_RANKS = moe_module.N_RANKS
N_EXPERTS_GLOBAL = moe_module.N_EXPERTS_GLOBAL
N_LOCAL = moe_module.N_LOCAL
MOE_INTER = moe_module.MOE_INTER
VOCAB = moe_module.VOCAB
TOPK = moe_module.TOPK
RECV_MAX = moe_module.RECV_MAX
AUX_PAD = moe_module.AUX_PAD
IDX_PAD = moe_module.IDX_PAD
N_ROUTES = moe_module.N_ROUTES

_ATTENTION_MODULES = { "swa": swa, "hca": hca, "csa": csa, }

_SWA_INPUT_NAMES = (
    "x_hc",
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
    "freqs_cos",
    "freqs_sin",
    "kv_freqs_cos",
    "kv_freqs_sin",
    "kv_cache",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "position_ids",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)

_HCA_INPUT_NAMES = (
    "x_hc",
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
    "freqs_cos",
    "freqs_sin",
    "kv_freqs_cos",
    "kv_freqs_sin",
    "group_cmp_freqs_cos",
    "group_cmp_freqs_sin",
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "compress_state",
    "compress_state_block_table",
    "kv_cache",
    "cmp_kv",
    "cmp_block_table",
    "group_ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "group_cmp_slot_mapping",
    "group_state_slot_mapping",
    "position_ids",
    "group_position_ids",
    "kv_seq_lens",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)

_CSA_INPUT_NAMES = (
    "x_hc",
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
    "freqs_cos",
    "freqs_sin",
    "cmp_freqs_cos",
    "cmp_freqs_sin",
    "cmp_wkv",
    "cmp_wgate",
    "cmp_ape",
    "cmp_norm_w",
    "compress_state",
    "compress_state_block_table",
    "idx_wq_b",
    "idx_wq_b_scale",
    "weights_proj",
    "hadamard_idx",
    "inner_wkv",
    "inner_wgate",
    "inner_ape",
    "inner_norm_w",
    "inner_compress_state",
    "inner_compress_state_block_table",
    "kv_cache",
    "cmp_kv",
    "cmp_block_table",
    "idx_kv_cache",
    "idx_kv_scale",
    "idx_block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "cmp_slot_mapping",
    "idx_slot_mapping",
    "state_slot_mapping",
    "inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
)


def _validate_import_contract():
    attention_modules = tuple(_ATTENTION_MODULES.values())
    for module in attention_modules:
        if module.TP_SIZE != TP_SIZE:
            raise ValueError(
                f"{module.__name__} froze TP={module.TP_SIZE}, expected TP={TP_SIZE}",
            )
        if module.S != DECODE_SEQ:
            raise ValueError(
                f"{module.__name__} uses S={module.S}, expected S={DECODE_SEQ}",
            )
        if module.T < MOE_TOKENS:
            raise ValueError(
                f"{module.__name__} local capacity {module.T} is smaller than "
                f"MoE capacity {MOE_TOKENS}",
            )
        if module.MAX_SEQ_LEN != 1_048_576:
            raise ValueError(
                f"{module.__name__} context ceiling must be 1M, got "
                f"{module.MAX_SEQ_LEN}",
            )
    if moe_module.EP != EP_SIZE or moe_module.N_RANKS != EP_SIZE:
        raise ValueError(
            f"MoE froze EP={moe_module.EP}, N_RANKS={moe_module.N_RANKS}; "
            f"expected EP={EP_SIZE}",
        )
    if MOE_TOKENS % DECODE_SEQ != 0:
        raise ValueError(
            f"MoE capacity {MOE_TOKENS} must be divisible by S={DECODE_SEQ}",
        )
    for rank in range(EP_SIZE):
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        if group_base % TP_SIZE != 0 or group_base + TP_SIZE > EP_SIZE:
            raise ValueError(
                f"rank {rank} maps outside its TP group: "
                f"group_base={group_base}, tp_rank={tp_rank}",
            )


_validate_import_contract()


def attention_kind_for_layer(layer_id):
    """Return the configured attention kind for one main-model layer."""
    layer_id = int(layer_id)
    if not 0 <= layer_id < MODEL_CONFIG.num_hidden_layers:
        raise ValueError(
            f"layer_id must be in [0, {MODEL_CONFIG.num_hidden_layers - 1}], "
            f"got {layer_id}",
        )
    ratio = MODEL_CONFIG.compress_ratios[layer_id]
    if ratio == 0:
        return "swa"
    if ratio == 4:
        return "csa"
    if ratio == 128:
        return "hca"
    raise ValueError(f"unsupported compression ratio {ratio} for layer {layer_id}")


def tp_group_for_rank(rank):
    """Return ``(group_base, tp_rank)`` for one EP-world rank."""
    rank = int(rank)
    if not 0 <= rank < EP_SIZE:
        raise ValueError(f"rank must be in [0, {EP_SIZE - 1}], got {rank}")
    tp_rank = rank % TP_SIZE
    return rank - tp_rank, tp_rank


def _active_batch(start_pos):
    if start_pos is None or isinstance(start_pos, int):
        return 1 if start_pos is not None else MOE_TOKENS // DECODE_SEQ
    if not isinstance(start_pos, (list, tuple)):
        raise TypeError("start_pos must be None, an int, or a list/tuple of ints")
    if not start_pos:
        raise ValueError("start_pos list must not be empty")
    return len(start_pos)


def _validate_active_tokens(active_tokens):
    if active_tokens <= 0 or active_tokens > MOE_TOKENS:
        raise ValueError(
            f"active token count must be in [1, {MOE_TOKENS}], got {active_tokens}",
        )
    if active_tokens % DECODE_SEQ != 0:
        raise ValueError(
            f"active token count {active_tokens} must be divisible by S={DECODE_SEQ}",
        )


def _distributed_shape(module, name, shape):
    if name == "wo_a":
        return [
            EP_SIZE,
            module.LOCAL_O_GROUPS,
            module.O_LORA,
            module.O_GROUP_IN,
        ]
    if name == "wo_b":
        return [EP_SIZE, module.D, module.LOCAL_O_WIDTH]
    return [EP_SIZE, *shape]


def _moe_shapes():
    m = moe_module
    return {
        "x_attn_moe": [EP_SIZE, MOE_TOKENS, m.HC_MULT, m.D],
        "hc_ffn_fn": [EP_SIZE, m.MIX_HC, m.HC_DIM],
        "hc_ffn_scale": [EP_SIZE, 3],
        "hc_ffn_base": [EP_SIZE, m.MIX_HC],
        "norm_w": [EP_SIZE, m.D],
        "gate_w": [EP_SIZE, m.N_EXPERTS_GLOBAL, m.D],
        "gate_bias": [EP_SIZE, m.N_EXPERTS_GLOBAL],
        "tid2eid": [EP_SIZE, m.VOCAB, m.TOPK],
        "input_ids": [EP_SIZE, MOE_TOKENS],
        "routed_w1": [EP_SIZE, m.N_LOCAL, m.MOE_INTER, m.D],
        "routed_w1_scale": [EP_SIZE, m.N_LOCAL, m.MOE_INTER],
        "routed_w3": [EP_SIZE, m.N_LOCAL, m.MOE_INTER, m.D],
        "routed_w3_scale": [EP_SIZE, m.N_LOCAL, m.MOE_INTER],
        "routed_w2": [EP_SIZE, m.N_LOCAL, m.D, m.MOE_INTER],
        "routed_w2_scale": [EP_SIZE, m.N_LOCAL, m.D],
        "shared_w1": [EP_SIZE, m.MOE_INTER, m.D],
        "shared_w1_scale": [EP_SIZE, m.MOE_INTER],
        "shared_w3": [EP_SIZE, m.MOE_INTER, m.D],
        "shared_w3_scale": [EP_SIZE, m.MOE_INTER],
        "shared_w2": [EP_SIZE, m.D, m.MOE_INTER],
        "shared_w2_scale": [EP_SIZE, m.D],
        "x_moe_next": [EP_SIZE, MOE_TOKENS, m.HC_MULT, m.D],
    }


def build_layer_shape_report(layer_id, start_pos=None):
    """Build current attention specs and audit their future EP-layer shapes."""
    from golden import TensorSpec

    kind = attention_kind_for_layer(layer_id)
    module = _ATTENTION_MODULES[kind]
    batch = _active_batch(start_pos)
    active_tokens = batch * DECODE_SEQ
    _validate_active_tokens(active_tokens)
    if batch > module.B:
        raise ValueError(
            f"{kind} batch {batch} exceeds TP{TP_SIZE} local capacity {module.B}",
        )

    if kind in {"swa", "hca"}:
        specs = module.build_distributed_tensor_specs(
            active_tokens,
            start_pos=start_pos,
        )
    else:
        specs = module.build_tensor_specs(start_pos=start_pos, batch=batch)
    distributed_shapes = {}
    for spec in specs:
        if not isinstance(spec, TensorSpec):
            continue
        if kind in {"swa", "hca"}:
            spec_shape = list(spec.shape)
            if not spec_shape or int(spec_shape[0]) != TP_SIZE:
                raise ValueError(
                    f"distributed {kind.upper()} shape must start with TP={TP_SIZE}, "
                    f"got {spec_shape}",
                )
            shape = [EP_SIZE, *spec_shape[1:]]
        else:
            shape = _distributed_shape(module, spec.name, list(spec.shape))
        if len(shape) > MAX_PUBLIC_TENSOR_DIMS:
            raise ValueError(
                f"{kind} tensor {spec.name!r} would have {len(shape)} dimensions: "
                f"{shape}",
            )
        distributed_shapes[spec.name] = shape

    for name in ("x_hc", "x_out", "freqs_cos", "freqs_sin", "position_ids"):
        if distributed_shapes[name][1] != active_tokens:
            raise ValueError(
                f"{kind} tensor {name!r} must use active token extent "
                f"{active_tokens}, got {distributed_shapes[name]}",
            )
    if kind == "csa":
        for name in ("cmp_freqs_cos", "cmp_freqs_sin"):
            if distributed_shapes[name][1] != active_tokens:
                raise ValueError(
                    f"CSA tensor {name!r} must use active token extent "
                    f"{active_tokens}, got {distributed_shapes[name]}",
                )
    if kind == "swa":
        group_tokens = TP_SIZE * active_tokens
        for name in ("kv_freqs_cos", "kv_freqs_sin", "swa_slot_mapping"):
            if distributed_shapes[name][1] != group_tokens:
                raise ValueError(
                    f"SWA tensor {name!r} must use TP-group token extent "
                    f"{group_tokens}, got {distributed_shapes[name]}",
                )
    if kind == "hca":
        group_tokens = TP_SIZE * active_tokens
        for name in (
            "kv_freqs_cos",
            "kv_freqs_sin",
            "group_ori_slot_mapping",
            "group_cmp_slot_mapping",
            "group_state_slot_mapping",
            "group_position_ids",
        ):
            if distributed_shapes[name][1] != group_tokens:
                raise ValueError(
                    f"HCA tensor {name!r} must use TP-group token extent "
                    f"{group_tokens}, got {distributed_shapes[name]}",
                )
        group_batch = TP_SIZE * batch
        for name in (
            "group_cmp_freqs_cos",
            "group_cmp_freqs_sin",
            "compress_state_block_table",
        ):
            if distributed_shapes[name][1] != group_batch:
                raise ValueError(
                    f"HCA tensor {name!r} must use TP-group batch extent "
                    f"{group_batch}, got {distributed_shapes[name]}",
                )

    bridge_shapes = {
        "x_attn_active": [EP_SIZE, active_tokens, module.HC_MULT, module.D],
        "x_attn_moe": [EP_SIZE, MOE_TOKENS, module.HC_MULT, module.D],
        "x_moe_next": [EP_SIZE, MOE_TOKENS, module.HC_MULT, module.D],
        "x_next": [EP_SIZE, active_tokens, module.HC_MULT, module.D],
    }
    moe_shapes = _moe_shapes()
    for section, shapes in (
        ("bridge", bridge_shapes),
        ("MoE", moe_shapes),
    ):
        for name, shape in shapes.items():
            if len(shape) > MAX_PUBLIC_TENSOR_DIMS:
                raise ValueError(
                    f"{section} tensor {name!r} would have {len(shape)} "
                    f"dimensions: {shape}",
                )

    return {
        "layer_id": int(layer_id),
        "kind": kind,
        "tp": TP_SIZE,
        "ep": EP_SIZE,
        "tp_groups": TP_GROUPS,
        "active_batch": batch,
        "active_tokens": active_tokens,
        "moe_capacity": MOE_TOKENS,
        "distributed_shapes": distributed_shapes,
        "bridge_shapes": bridge_shapes,
        "moe_shapes": moe_shapes,
    }


@pl.jit.inline(auto_scope=False)
def decode_layer_swa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    kv_freqs_cos: pl.Tensor[[CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_sin: pl.Tensor[[CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[CP_T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one SWA attention and one MoE without clearing shared signals."""
    active_t = pl.tensor.dim(x_hc, 0)
    with pl.scope():
        decode_swa_attention(
            x_hc,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
            gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin,
            kv_freqs_cos, kv_freqs_sin,
            kv_cache, swa_slot_mapping, swa_indices, swa_lens,
            position_ids, attn_sink,
            wo_a, wo_b, wo_b_scale,
            x_attn_active,
            gather_window, gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            group_base, tp_rank, local_t,
        )

    with pl.scope():
        x_attn_moe = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_attn_pack"):
            if token < active_t:
                x_attn_moe[token : token + 1, 0 : HC_MULT, 0 : D] = (
                    x_attn_active[token : token + 1, 0 : HC_MULT, 0 : D]
                )
            else:
                x_attn_moe[token : token + 1, 0 : HC_MULT, 0 : D] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)

        moe(
            x_attn_moe,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            x_moe_next,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, local_t, my_rank, moe_epoch,
        )

        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_active_trim"):
            if token < active_t:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = (x_moe_next[token : token + 1, 0 : HC_MULT, 0 : D])
    return x_next


@pl.jit(auto_scope=False)
def decode_layer_swa_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    kv_freqs_cos: pl.Tensor[[CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_sin: pl.Tensor[[CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[CP_T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Standalone child: run one layer and clear its fresh MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    kv_freqs_cos.bind_dynamic(0, CP_T_DYN)
    kv_freqs_sin.bind_dynamic(0, CP_T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(0, CP_T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)

    decode_layer_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        kv_freqs_cos, kv_freqs_sin,
        kv_cache, swa_slot_mapping, swa_indices, swa_lens, position_ids,
        attn_sink,
        wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_attn_active, x_moe_next, x_next,
        gather_window, gather_signal,
        attention_window, attention_signal, o_window, o_signal,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, group_base, tp_rank, local_t, my_rank, moe_epoch,
    )
    clear_moe_signals(x_moe_next, arrived, data_arrived, combine_arrived)
    return x_next


@pl.jit.host
def l3_decode_layer_swa(
    x_hc: pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_cos: pl.Tensor[[N_RANKS, CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_sin: pl.Tensor[[N_RANKS, CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[N_RANKS, CP_T_DYN], pl.INT64],
    swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32],
    swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[N_RANKS, MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch one standalone SWA+MoE layer child on every EP rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    kv_freqs_cos.bind_dynamic(1, CP_T_DYN)
    kv_freqs_sin.bind_dynamic(1, CP_T_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(1, CP_T_DYN)
    swa_indices.bind_dynamic(1, T_DYN)
    swa_lens.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    x_attn_active.bind_dynamic(1, T_DYN)
    x_next.bind_dynamic(1, T_DYN)

    gather_window_buf = pld.alloc_window_buffer([DECODE_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [DECODE_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        decode_layer_swa_test(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank],
            wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            kv_freqs_cos[rank], kv_freqs_sin[rank],
            kv_cache[rank], swa_slot_mapping[rank], swa_indices[rank],
            swa_lens[rank], position_ids[rank], attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank],
            tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_attn_active[rank], x_moe_next[rank], x_next[rank],
            gather_window, gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, group_base, tp_rank, local_t, rank,
            pl.const(1, pl.INT32),
            device=rank,
        )
    return x_next


@pl.jit.inline(auto_scope=False)
def decode_layer_hca(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    kv_freqs_cos: pl.Tensor[[HCA_CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_sin: pl.Tensor[[HCA_CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    group_cmp_freqs_cos: pl.Tensor[[HCA_CP_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    group_cmp_freqs_sin: pl.Tensor[[HCA_CP_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[HCA_COMPRESS_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[HCA_CP_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[HCA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[HCA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    group_ori_slot_mapping: pl.Tensor[[HCA_CP_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    group_cmp_slot_mapping: pl.Tensor[[HCA_CP_T_DYN], pl.INT64],
    group_state_slot_mapping: pl.Tensor[[HCA_CP_T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    group_position_ids: pl.Tensor[[HCA_CP_T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[HCA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one HCA attention and one MoE without clearing shared signals."""
    active_t = pl.tensor.dim(x_hc, 0)
    with pl.scope():
        decode_hca_attention(
            x_hc,
            hc_attn_fn, hc_attn_scale, hc_attn_base,
            attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
            gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin, kv_freqs_cos, kv_freqs_sin,
            group_cmp_freqs_cos, group_cmp_freqs_sin,
            cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
            compress_state, compress_state_block_table,
            kv_cache, cmp_kv, cmp_block_table,
            group_ori_slot_mapping, window_swa_indices, window_swa_lens,
            group_cmp_slot_mapping, group_state_slot_mapping,
            position_ids, group_position_ids, kv_seq_lens, attn_sink,
            wo_a, wo_b, wo_b_scale,
            x_attn_active,
            gather_window, gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            group_base, tp_rank, local_t,
        )

    with pl.scope():
        x_attn_moe = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_attn_pack"):
            if token < active_t:
                x_attn_moe[token : token + 1, 0 : HC_MULT, 0 : D] = (
                    x_attn_active[token : token + 1, 0 : HC_MULT, 0 : D]
                )
            else:
                x_attn_moe[token : token + 1, 0 : HC_MULT, 0 : D] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)

        moe(
            x_attn_moe,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            x_moe_next,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, local_t, my_rank, moe_epoch,
        )

        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_active_trim"):
            if token < active_t:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = (x_moe_next[token : token + 1, 0 : HC_MULT, 0 : D])
    return x_next


@pl.jit(auto_scope=False)
def decode_layer_hca_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    kv_freqs_cos: pl.Tensor[[HCA_CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_sin: pl.Tensor[[HCA_CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    group_cmp_freqs_cos: pl.Tensor[[HCA_CP_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    group_cmp_freqs_sin: pl.Tensor[[HCA_CP_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[HCA_COMPRESS_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[HCA_CP_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[HCA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[HCA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    group_ori_slot_mapping: pl.Tensor[[HCA_CP_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    group_cmp_slot_mapping: pl.Tensor[[HCA_CP_T_DYN], pl.INT64],
    group_state_slot_mapping: pl.Tensor[[HCA_CP_T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    group_position_ids: pl.Tensor[[HCA_CP_T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[HCA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    gather_window: pld.DistributedTensor[[DECODE_GROUP_CAP, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Standalone child: run one HCA layer and clear its fresh MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    kv_freqs_cos.bind_dynamic(0, HCA_CP_T_DYN)
    kv_freqs_sin.bind_dynamic(0, HCA_CP_T_DYN)
    group_cmp_freqs_cos.bind_dynamic(0, HCA_CP_B_DYN)
    group_cmp_freqs_sin.bind_dynamic(0, HCA_CP_B_DYN)
    compress_state.bind_dynamic(0, HCA_COMPRESS_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, HCA_CP_B_DYN)
    kv_cache.bind_dynamic(0, HCA_ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, HCA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, HCA_B_DYN)
    cmp_block_table.bind_dynamic(1, HCA_CMP_TABLE_BLOCKS_DYN)
    group_ori_slot_mapping.bind_dynamic(0, HCA_CP_T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    group_cmp_slot_mapping.bind_dynamic(0, HCA_CP_T_DYN)
    group_state_slot_mapping.bind_dynamic(0, HCA_CP_T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    group_position_ids.bind_dynamic(0, HCA_CP_T_DYN)
    kv_seq_lens.bind_dynamic(0, HCA_B_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)

    decode_layer_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, kv_freqs_cos, kv_freqs_sin,
        group_cmp_freqs_cos, group_cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        group_ori_slot_mapping, window_swa_indices, window_swa_lens,
        group_cmp_slot_mapping, group_state_slot_mapping,
        position_ids, group_position_ids, kv_seq_lens, attn_sink,
        wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_attn_active, x_moe_next, x_next,
        gather_window, gather_signal,
        attention_window, attention_signal, o_window, o_signal,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, group_base, tp_rank, local_t, my_rank, moe_epoch,
    )
    clear_moe_signals(x_moe_next, arrived, data_arrived, combine_arrived)
    return x_next


@pl.jit.host
def l3_decode_layer_hca(
    x_hc: pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_cos: pl.Tensor[[N_RANKS, HCA_CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    kv_freqs_sin: pl.Tensor[[N_RANKS, HCA_CP_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    group_cmp_freqs_cos: pl.Tensor[[N_RANKS, HCA_CP_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    group_cmp_freqs_sin: pl.Tensor[[N_RANKS, HCA_CP_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[N_RANKS, HCA_COMPRESS_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[N_RANKS, HCA_CP_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, HCA_ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, HCA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    group_ori_slot_mapping: pl.Tensor[[N_RANKS, HCA_CP_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    group_cmp_slot_mapping: pl.Tensor[[N_RANKS, HCA_CP_T_DYN], pl.INT64],
    group_state_slot_mapping: pl.Tensor[[N_RANKS, HCA_CP_T_DYN], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    group_position_ids: pl.Tensor[[N_RANKS, HCA_CP_T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[N_RANKS, HCA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[N_RANKS, MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch one standalone HCA+MoE layer child on every EP rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    kv_freqs_cos.bind_dynamic(1, HCA_CP_T_DYN)
    kv_freqs_sin.bind_dynamic(1, HCA_CP_T_DYN)
    group_cmp_freqs_cos.bind_dynamic(1, HCA_CP_B_DYN)
    group_cmp_freqs_sin.bind_dynamic(1, HCA_CP_B_DYN)
    compress_state.bind_dynamic(1, HCA_COMPRESS_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, HCA_CP_B_DYN)
    kv_cache.bind_dynamic(1, HCA_ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, HCA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, HCA_B_DYN)
    cmp_block_table.bind_dynamic(2, HCA_CMP_TABLE_BLOCKS_DYN)
    group_ori_slot_mapping.bind_dynamic(1, HCA_CP_T_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    window_swa_lens.bind_dynamic(1, T_DYN)
    group_cmp_slot_mapping.bind_dynamic(1, HCA_CP_T_DYN)
    group_state_slot_mapping.bind_dynamic(1, HCA_CP_T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    group_position_ids.bind_dynamic(1, HCA_CP_T_DYN)
    kv_seq_lens.bind_dynamic(1, HCA_B_DYN)
    x_attn_active.bind_dynamic(1, T_DYN)
    x_next.bind_dynamic(1, T_DYN)

    gather_window_buf = pld.alloc_window_buffer([DECODE_GROUP_CAP, D], dtype=pl.BF16)
    gather_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)

    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        gather_window = pld.window(gather_window_buf, [DECODE_GROUP_CAP, D], dtype=pl.BF16)
        gather_signal = pld.window(gather_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        decode_layer_hca_test(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank],
            wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            kv_freqs_cos[rank], kv_freqs_sin[rank],
            group_cmp_freqs_cos[rank], group_cmp_freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            compress_state[rank], compress_state_block_table[rank],
            kv_cache[rank], cmp_kv[rank], cmp_block_table[rank],
            group_ori_slot_mapping[rank], window_swa_indices[rank],
            window_swa_lens[rank], group_cmp_slot_mapping[rank],
            group_state_slot_mapping[rank], position_ids[rank],
            group_position_ids[rank], kv_seq_lens[rank],
            attn_sink[rank], wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank],
            tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_attn_active[rank], x_moe_next[rank], x_next[rank],
            gather_window, gather_signal,
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, group_base, tp_rank, local_t, rank,
            pl.const(1, pl.INT32),
            device=rank,
        )
    return x_next


@pl.jit.inline(auto_scope=False)
def decode_layer_csa(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[CSA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one CSA attention and one MoE without clearing shared signals."""
    active_t = pl.tensor.dim(x_hc, 0)
    with pl.scope():
        if TP_SIZE == 1:
            decode_csa_tp1(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
                gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin,
                cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
                compress_state, compress_state_block_table,
                idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
                inner_wkv, inner_wgate, inner_ape, inner_norm_w,
                inner_compress_state, inner_compress_state_block_table,
                kv_cache, cmp_kv, cmp_block_table,
                idx_kv_cache, idx_kv_scale, idx_block_table,
                ori_slot_mapping, window_swa_indices, window_swa_lens,
                cmp_slot_mapping, idx_slot_mapping,
                state_slot_mapping, inner_state_slot_mapping,
                position_ids, kv_seq_lens, attn_sink,
                wo_a, wo_b, wo_b_scale,
                x_attn_active,
            )
        else:
            decode_csa(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
                gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin,
                cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
                compress_state, compress_state_block_table,
                idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
                inner_wkv, inner_wgate, inner_ape, inner_norm_w,
                inner_compress_state, inner_compress_state_block_table,
                kv_cache, cmp_kv, cmp_block_table,
                idx_kv_cache, idx_kv_scale, idx_block_table,
                ori_slot_mapping, window_swa_indices, window_swa_lens,
                cmp_slot_mapping, idx_slot_mapping,
                state_slot_mapping, inner_state_slot_mapping,
                position_ids, kv_seq_lens, attn_sink,
                wo_a, wo_b, wo_b_scale,
                x_attn_active,
                attention_window, attention_signal, o_window, o_signal,
                group_base, tp_rank, local_t,
            )

    with pl.scope():
        x_attn_moe = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_attn_pack"):
            if token < active_t:
                x_attn_moe[token : token + 1, 0 : HC_MULT, 0 : D] = (
                    x_attn_active[token : token + 1, 0 : HC_MULT, 0 : D]
                )
            else:
                x_attn_moe[token : token + 1, 0 : HC_MULT, 0 : D] = pl.full([1, HC_MULT, D], dtype=pl.FP32, value=0.0)

        moe(
            x_attn_moe,
            hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, input_ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            x_moe_next,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, local_t, my_rank, moe_epoch,
        )

        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_active_trim"):
            if token < active_t:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = (x_moe_next[token : token + 1, 0 : HC_MULT, 0 : D])
    return x_next


@pl.jit(auto_scope=False)
def decode_layer_csa_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
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
    cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T_DYN], pl.INT64],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[CSA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
    attention_window: pld.DistributedTensor[[ATTENTION_WINDOW_ROWS, O_GROUP_IN], pl.BF16],
    attention_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    o_window: pld.DistributedTensor[[O_WINDOW_ROWS, D], pl.BF16],
    o_signal: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one standalone CSA+MoE layer and clear its MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, CSA_MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, CSA_B_DYN)
    inner_compress_state.bind_dynamic(0, CSA_INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(0, CSA_B_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CSA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, CSA_B_DYN)
    idx_kv_cache.bind_dynamic(0, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(0, CSA_B_DYN)
    ori_slot_mapping.bind_dynamic(0, T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, T_DYN)
    idx_slot_mapping.bind_dynamic(0, T_DYN)
    state_slot_mapping.bind_dynamic(0, T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    kv_seq_lens.bind_dynamic(0, CSA_B_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)

    decode_layer_csa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        inner_compress_state, inner_compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, idx_slot_mapping,
        state_slot_mapping, inner_state_slot_mapping,
        position_ids, kv_seq_lens, attn_sink,
        wo_a, wo_b, wo_b_scale,
        hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        norm_w, gate_w, gate_bias, tid2eid, input_ids,
        routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
        routed_w2, routed_w2_scale,
        shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
        shared_w2, shared_w2_scale,
        x_attn_active, x_moe_next, x_next,
        attention_window, attention_signal, o_window, o_signal,
        recv_meta, recv_x, recv_aux, recv_route,
        arrived, data_arrived, routed_y_buf, combine_arrived,
        layer_id, group_base, tp_rank, local_t, my_rank, moe_epoch,
    )
    clear_moe_signals(x_moe_next, arrived, data_arrived, combine_arrived)
    return x_next


@pl.jit.host
def l3_decode_layer_csa(
    x_hc: pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_attn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    wq_a: pl.Tensor[[N_RANKS, D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[N_RANKS, Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[N_RANKS, H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[N_RANKS, D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[N_RANKS, Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_cos: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[N_RANKS, T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[N_RANKS, CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[N_RANKS, D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[N_RANKS, T_DYN], pl.INT64],
    position_ids: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    kv_seq_lens: pl.Tensor[[N_RANKS, CSA_B_DYN], pl.INT32],
    attn_sink: pl.Tensor[[N_RANKS, H], pl.FP32],
    wo_a: pl.Tensor[[N_RANKS, LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[N_RANKS, D, LOCAL_O_WIDTH], pl.INT8],
    wo_b_scale: pl.Tensor[[N_RANKS, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[N_RANKS, MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[N_RANKS, MOE_TOKENS], pl.INT64],
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
    x_attn_active: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    x_moe_next: pl.Out[pl.Tensor[[N_RANKS, MOE_TOKENS, HC_MULT, D], pl.FP32]],
    x_next: pl.Out[pl.Tensor[[N_RANKS, T_DYN, HC_MULT, D], pl.FP32]],
    layer_id: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch one complete CSA+MoE child per EP-world rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    cmp_freqs_cos.bind_dynamic(1, T_DYN)
    cmp_freqs_sin.bind_dynamic(1, T_DYN)
    compress_state.bind_dynamic(1, CSA_MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, CSA_B_DYN)
    inner_compress_state.bind_dynamic(1, CSA_INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(1, CSA_B_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, CSA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, CSA_B_DYN)
    idx_kv_cache.bind_dynamic(1, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(1, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(1, CSA_B_DYN)
    ori_slot_mapping.bind_dynamic(1, T_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    window_swa_lens.bind_dynamic(1, T_DYN)
    cmp_slot_mapping.bind_dynamic(1, T_DYN)
    idx_slot_mapping.bind_dynamic(1, T_DYN)
    state_slot_mapping.bind_dynamic(1, T_DYN)
    inner_state_slot_mapping.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, T_DYN)
    kv_seq_lens.bind_dynamic(1, CSA_B_DYN)
    x_attn_active.bind_dynamic(1, T_DYN)
    x_next.bind_dynamic(1, T_DYN)

    attention_window_buf = pld.alloc_window_buffer([ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
    attention_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    o_window_buf = pld.alloc_window_buffer([O_WINDOW_ROWS, D], dtype=pl.BF16)
    o_signal_buf = pld.alloc_window_buffer([TP_SIZE, 1], dtype=pl.INT32)
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
    recv_route_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    for rank in pl.range(pld.world_size()):
        attention_window = pld.window(attention_window_buf, [ATTENTION_WINDOW_ROWS, O_GROUP_IN], dtype=pl.BF16)
        attention_signal = pld.window(attention_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        o_window = pld.window(o_window_buf, [O_WINDOW_ROWS, D], dtype=pl.BF16)
        o_signal = pld.window(o_signal_buf, [TP_SIZE, 1], dtype=pl.INT32)
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32)
        recv_route = pld.window(recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32)
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        tp_rank = rank % TP_SIZE
        group_base = rank - tp_rank
        decode_layer_csa_test(
            x_hc[rank],
            hc_attn_fn[rank], hc_attn_scale[rank], hc_attn_base[rank],
            attn_norm_w[rank], wq_a[rank], wq_b[rank],
            wq_b_scale[rank], wkv[rank], gamma_cq[rank], gamma_ckv[rank],
            freqs_cos[rank], freqs_sin[rank],
            cmp_freqs_cos[rank], cmp_freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            compress_state[rank], compress_state_block_table[rank],
            idx_wq_b[rank], idx_wq_b_scale[rank],
            weights_proj[rank], hadamard_idx[rank],
            inner_wkv[rank], inner_wgate[rank],
            inner_ape[rank], inner_norm_w[rank],
            inner_compress_state[rank],
            inner_compress_state_block_table[rank],
            kv_cache[rank], cmp_kv[rank], cmp_block_table[rank],
            idx_kv_cache[rank], idx_kv_scale[rank], idx_block_table[rank],
            ori_slot_mapping[rank], window_swa_indices[rank],
            window_swa_lens[rank], cmp_slot_mapping[rank],
            idx_slot_mapping[rank], state_slot_mapping[rank],
            inner_state_slot_mapping[rank], position_ids[rank],
            kv_seq_lens[rank], attn_sink[rank],
            wo_a[rank], wo_b[rank], wo_b_scale[rank],
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank],
            tid2eid[rank], input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            x_attn_active[rank], x_moe_next[rank], x_next[rank],
            attention_window, attention_signal, o_window, o_signal,
            recv_meta, recv_x, recv_aux, recv_route,
            arrived, data_arrived, routed_y_buf, combine_arrived,
            layer_id, group_base, tp_rank, local_t, rank,
            pl.const(1, pl.INT32),
            device=rank,
        )
    return x_next


def _expand_swa_spec(spec):
    """Repeat one TP-group fixture across the EP world's TP groups."""
    from golden import TensorSpec

    def init_value():
        value = spec.create_tensor()
        repeats = [TP_GROUPS] + [1] * (value.ndim - 1)
        return value.repeat(*repeats)

    expanded = TensorSpec(
        spec.name,
        [N_RANKS, *spec.shape[1:]],
        spec.dtype,
        init_value=init_value,
        is_output=spec.is_output,
    )
    expanded.resident = spec.resident
    return expanded


def build_swa_layer_specs(start_pos=None, layer_id=0):
    """Build one current-SWA fixture followed by the current MoE fixture."""
    import inspect

    import torch
    from golden import ScalarSpec, TensorSpec

    if attention_kind_for_layer(layer_id) != "swa":
        raise ValueError(f"layer {layer_id} is not an SWA layer")
    batch = _active_batch(start_pos)
    local_t = batch * DECODE_SEQ
    _validate_active_tokens(local_t)
    if batch > swa.B:
        raise ValueError(
            f"SWA batch {batch} exceeds TP{TP_SIZE} local capacity {swa.B}",
        )

    specs = []
    swa_specs = swa.build_distributed_tensor_specs(local_t, start_pos=start_pos)
    for spec in swa_specs:
        if isinstance(spec, TensorSpec) and spec.name != "x_out":
            specs.append(_expand_swa_spec(spec))

    existing = {spec.name for spec in specs}
    for spec in moe_module.build_tensor_specs(
        layer_id=layer_id, num_tokens=local_t,
    ):
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next"} or spec.name in existing:
            continue
        specs.append(spec)
        existing.add(spec.name)

    specs.extend(
        [
            TensorSpec(
                "x_attn_active",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            TensorSpec(
                "x_moe_next",
                [N_RANKS, MOE_TOKENS, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            TensorSpec(
                "x_next",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            ScalarSpec("layer_id", torch.int32, layer_id),
            ScalarSpec("local_t", torch.int32, local_t),
        ],
    )

    specs_by_name = {spec.name: spec for spec in specs}
    parameter_names = [
        name
        for name, parameter in inspect.signature(
            l3_decode_layer_swa._func,
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in parameter_names if name not in specs_by_name]
    extra = [name for name in specs_by_name if name not in parameter_names]
    if missing or extra:
        raise ValueError(
            f"SWA layer spec/signature mismatch: missing={missing}, extra={extra}",
        )
    return [specs_by_name[name] for name in parameter_names]


def golden_decode_layer_swa(tensors):
    """Compose the unchanged SWA and MoE goldens through the bounded bridge."""
    import torch

    local_t = int(tensors["local_t"])
    tensors["x_attn_active"].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = { name: tensors[name][group_begin:group_end] for name in _SWA_INPUT_NAMES }
        group_tensors["x_out"] = tensors["x_attn_active"][group_begin:group_end]
        group_tensors["local_t"] = local_t
        swa.golden_decode_swa(group_tensors)

    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    x_attn_moe[:, :local_t].copy_(tensors["x_attn_active"])
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn_moe
    moe_tensors["x_next"] = tensors["x_moe_next"]
    moe_tensors["num_tokens"] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors["x_next"].copy_(tensors["x_moe_next"][:, :local_t])


def _expand_hca_spec(spec):
    """Repeat one TP-group HCA fixture across the EP world's TP groups."""
    from golden import TensorSpec

    def init_value():
        value = spec.create_tensor()
        repeats = [TP_GROUPS] + [1] * (value.ndim - 1)
        return value.repeat(*repeats)

    expanded = TensorSpec(
        spec.name,
        [N_RANKS, *spec.shape[1:]],
        spec.dtype,
        init_value=init_value,
        is_output=spec.is_output,
    )
    expanded.resident = spec.resident
    return expanded


def build_hca_layer_specs(start_pos=None, layer_id=3):
    """Build one current-HCA fixture followed by the current MoE fixture."""
    import inspect

    import torch
    from golden import ScalarSpec, TensorSpec

    if attention_kind_for_layer(layer_id) != "hca":
        raise ValueError(f"layer {layer_id} is not an HCA layer")
    batch = _active_batch(start_pos)
    local_t = batch * DECODE_SEQ
    _validate_active_tokens(local_t)
    if batch > hca.B:
        raise ValueError(
            f"HCA batch {batch} exceeds TP{TP_SIZE} local capacity {hca.B}",
        )

    specs = []
    hca_specs = hca.build_distributed_tensor_specs(local_t, start_pos=start_pos)
    for spec in hca_specs:
        if isinstance(spec, TensorSpec) and spec.name != "x_out":
            specs.append(_expand_hca_spec(spec))

    existing = {spec.name for spec in specs}
    for spec in moe_module.build_tensor_specs(
        layer_id=layer_id, num_tokens=local_t,
    ):
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next"} or spec.name in existing:
            continue
        specs.append(spec)
        existing.add(spec.name)

    specs.extend(
        [
            TensorSpec(
                "x_attn_active",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            TensorSpec(
                "x_moe_next",
                [N_RANKS, MOE_TOKENS, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            TensorSpec(
                "x_next",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            ScalarSpec("layer_id", torch.int32, layer_id),
            ScalarSpec("local_t", torch.int32, local_t),
        ],
    )

    specs_by_name = {spec.name: spec for spec in specs}
    parameter_names = [
        name
        for name, parameter in inspect.signature(
            l3_decode_layer_hca._func,
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in parameter_names if name not in specs_by_name]
    extra = [name for name in specs_by_name if name not in parameter_names]
    if missing or extra:
        raise ValueError(
            f"HCA layer spec/signature mismatch: missing={missing}, extra={extra}",
        )
    return [specs_by_name[name] for name in parameter_names]


def golden_decode_layer_hca(tensors):
    """Compose the unchanged HCA and MoE goldens through the bounded bridge."""
    import torch

    local_t = int(tensors["local_t"])
    tensors["x_attn_active"].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = { name: tensors[name][group_begin:group_end] for name in _HCA_INPUT_NAMES }
        group_tensors["x_out"] = tensors["x_attn_active"][group_begin:group_end]
        group_tensors["local_t"] = local_t
        hca.golden_decode_hca(group_tensors)

    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    x_attn_moe[:, :local_t].copy_(tensors["x_attn_active"])
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn_moe
    moe_tensors["x_next"] = tensors["x_moe_next"]
    moe_tensors["num_tokens"] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors["x_next"].copy_(tensors["x_moe_next"][:, :local_t])


def _expand_csa_spec(spec):
    """Repeat one TP-group CSA fixture across the EP world's TP groups."""
    from golden import TensorSpec

    def init_value():
        value = spec.create_tensor()
        repeats = [TP_GROUPS] + [1] * (value.ndim - 1)
        return value.repeat(*repeats)

    expanded = TensorSpec(
        spec.name,
        [N_RANKS, *spec.shape[1:]],
        spec.dtype,
        init_value=init_value,
        is_output=spec.is_output,
    )
    expanded.resident = spec.resident
    return expanded


def build_csa_layer_specs(start_pos=None, layer_id=2):
    """Build one current-CSA fixture followed by the current MoE fixture."""
    import inspect

    import torch
    from golden import ScalarSpec, TensorSpec

    if attention_kind_for_layer(layer_id) != "csa":
        raise ValueError(f"layer {layer_id} is not a CSA layer")
    batch = _active_batch(start_pos)
    local_t = batch * DECODE_SEQ
    _validate_active_tokens(local_t)
    if batch > csa.B:
        raise ValueError(
            f"CSA batch {batch} exceeds TP{TP_SIZE} local capacity {csa.B}",
        )

    specs = []
    csa_specs = csa.build_distributed_tensor_specs(local_t, start_pos=start_pos)
    for spec in csa_specs:
        if isinstance(spec, TensorSpec) and spec.name != "x_out":
            specs.append(_expand_csa_spec(spec))

    existing = {spec.name for spec in specs}
    for spec in moe_module.build_tensor_specs(
        layer_id=layer_id, num_tokens=local_t,
    ):
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next"} or spec.name in existing:
            continue
        specs.append(spec)
        existing.add(spec.name)

    specs.extend(
        [
            TensorSpec(
                "x_attn_active",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            TensorSpec(
                "x_moe_next",
                [N_RANKS, MOE_TOKENS, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            TensorSpec(
                "x_next",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
                is_output=True,
            ),
            ScalarSpec("layer_id", torch.int32, layer_id),
            ScalarSpec("local_t", torch.int32, local_t),
        ],
    )

    specs_by_name = {spec.name: spec for spec in specs}
    parameter_names = [
        name
        for name, parameter in inspect.signature(
            l3_decode_layer_csa._func,
        ).parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]
    missing = [name for name in parameter_names if name not in specs_by_name]
    extra = [name for name in specs_by_name if name not in parameter_names]
    if missing or extra:
        raise ValueError(
            f"CSA layer spec/signature mismatch: missing={missing}, extra={extra}",
        )
    return [specs_by_name[name] for name in parameter_names]


def golden_decode_layer_csa(tensors):
    """Compose the unchanged CSA and MoE goldens through the bounded bridge."""
    import torch

    local_t = int(tensors["local_t"])
    tensors["x_attn_active"].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = { name: tensors[name][group_begin:group_end] for name in _CSA_INPUT_NAMES }
        group_tensors["x_out"] = tensors["x_attn_active"][group_begin:group_end]
        group_tensors["local_t"] = local_t
        csa.golden_decode_csa(group_tensors)

    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    x_attn_moe[:, :local_t].copy_(tensors["x_attn_active"])
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn_moe
    moe_tensors["x_next"] = tensors["x_moe_next"]
    moe_tensors["num_tokens"] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors["x_next"].copy_(tensors["x_moe_next"][:, :local_t])


def _parse_start_pos(raw):
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--start-pos must contain at least one integer")
    values = [int(part) for part in parts]
    return values[0] if len(values) == 1 else values


def main():
    import argparse

    from golden import (
        mapped_pool_ratio_allclose,
        ratio_reldiff,
        run_jit,
    )
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=("a2a3", "a2a3sim", "a5", "a5sim"),
    )
    parser.add_argument("--tp", type=int, default=TP_SIZE, choices=list(_TP_CHOICES))
    parser.add_argument("--ep", type=int, default=EP_SIZE, choices=list(_EP_CHOICES))
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help=f"comma-separated device ids; EP={EP_SIZE} needs exactly {EP_SIZE}",
    )
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument(
        "--start-pos",
        type=str,
        default=None,
        help="a scalar selects batch=1; a comma-separated list sets batch to its length",
    )
    parser.add_argument(
        "--enable-chip-swimlane", type=int, default=0, choices=range(5),
    )
    parser.add_argument(
        "--enable-scope-stats", action="store_true", default=False,
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument("--log-level", type=str, default=None)
    parser.add_argument("--shape-only", action="store_true", default=False)
    args = parser.parse_args()

    if args.tp != TP_SIZE or args.ep != EP_SIZE:
        parser.error(f"parallel sizes froze at import as TP={TP_SIZE}, EP={EP_SIZE}")
    start_pos = _parse_start_pos(args.start_pos)
    kind = attention_kind_for_layer(args.layer_id)

    report = build_layer_shape_report(args.layer_id, start_pos=start_pos)
    if args.shape_only:
        print(
            f"layer={report['layer_id']} kind={report['kind']} "
            f"TP={report['tp']} EP={report['ep']} "
            f"groups={report['tp_groups']} "
            f"batch={report['active_batch']} "
            f"active_t={report['active_tokens']} "
            f"moe_capacity={report['moe_capacity']}",
        )
        return

    if args.device is None:
        args.device = ",".join(str(rank) for rank in range(EP_SIZE))
    try:
        device_ids = [int(device) for device in args.device.split(",")]
    except ValueError:
        parser.error(f"--device must be a comma-separated integer list, got {args.device!r}")
    if len(device_ids) != EP_SIZE or len(set(device_ids)) != EP_SIZE:
        parser.error(f"EP={EP_SIZE} needs exactly {EP_SIZE} distinct devices, " f"got {device_ids}")
    if any(device < 0 for device in device_ids):
        parser.error(f"device IDs must be non-negative, got {device_ids}")

    local_t = report["active_tokens"]
    if kind == "swa":
        layer_fn = l3_decode_layer_swa
        specs = build_swa_layer_specs(start_pos=start_pos, layer_id=args.layer_id)
        golden_fn = golden_decode_layer_swa
        compare_fn = {
            "kv_cache": mapped_pool_ratio_allclose(
                "swa_slot_mapping",
                mapping_shape=(N_RANKS, TP_SIZE * local_t),
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="KV cache",
                atol=1e-4,
                rtol=1.0 / 128,
                max_error_ratio=0.005,
            ),
            "x_attn_active": ratio_reldiff(
                diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1,
            ),
            "x_moe_next": ratio_reldiff(
                diff_thd=0.01,
                pct_thd=0.05,
                valid_rows=local_t,
                valid_axis=1,
            ),
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        }
    elif kind == "hca":
        layer_fn = l3_decode_layer_hca
        specs = build_hca_layer_specs(start_pos=start_pos, layer_id=args.layer_id)
        golden_fn = golden_decode_layer_hca
        mapping_shape = (N_RANKS, TP_SIZE * local_t)
        compare_fn = {
            "compress_state": mapped_pool_ratio_allclose(
                "group_state_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=HCA_COMPRESS_STATE_BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="compressor state",
                atol=1e-3,
                rtol=1.0 / 128,
            ),
            "kv_cache": mapped_pool_ratio_allclose(
                "group_ori_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="original KV cache",
                atol=1e-3,
                rtol=1.0 / 128,
            ),
            "cmp_kv": mapped_pool_ratio_allclose(
                "group_cmp_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="compressed KV cache",
                atol=1e-3,
                rtol=1.0 / 128,
            ),
            "x_attn_active": ratio_reldiff(
                diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1,
            ),
            "x_moe_next": ratio_reldiff(
                diff_thd=0.01,
                pct_thd=0.05,
                valid_rows=local_t,
                valid_axis=1,
            ),
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        }
    else:
        layer_fn = l3_decode_layer_csa
        specs = build_csa_layer_specs(start_pos=start_pos, layer_id=args.layer_id)
        golden_fn = golden_decode_layer_csa
        mapping_shape = (N_RANKS, local_t)
        compare_fn = {
            "compress_state": mapped_pool_ratio_allclose(
                "state_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=CSA_MAIN_STATE_BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="main compressor state",
                atol=1e-3,
                rtol=1e-3,
            ),
            "inner_compress_state": mapped_pool_ratio_allclose(
                "inner_state_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=CSA_INNER_STATE_BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="inner compressor state",
                atol=1e-3,
                rtol=1e-3,
            ),
            "kv_cache": mapped_pool_ratio_allclose(
                "ori_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="original KV cache",
                atol=1e-4,
                rtol=1.0 / 128,
            ),
            "cmp_kv": mapped_pool_ratio_allclose(
                "cmp_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="compressed KV cache",
                atol=1e-4,
                rtol=1.0 / 128,
            ),
            "idx_kv_cache": mapped_pool_ratio_allclose(
                "idx_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="indexer KV cache",
                atol=1,
                rtol=0,
                max_error_ratio=0.01,
            ),
            "idx_kv_scale": mapped_pool_ratio_allclose(
                "idx_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="indexer KV scale",
                atol=1e-4,
                rtol=1.0 / 128,
                max_error_ratio=0.01,
            ),
            "x_attn_active": ratio_reldiff(
                diff_thd=4e-3, pct_thd=0.008, max_diff_hd=2,
            ),
            "x_moe_next": ratio_reldiff(
                diff_thd=0.01,
                pct_thd=0.05,
                valid_rows=local_t,
                valid_axis=1,
            ),
            "x_next": ratio_reldiff(diff_thd=0.01, pct_thd=0.05),
        }

    result = run_jit(
        fn=layer_fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        runtime_dir=args.runtime_dir,
        compile_cfg=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids, num_sub_workers=0,
            ),
        ),
        runtime_cfg=dict(
            platform=args.platform,
            enable_chip_swimlane=args.enable_chip_swimlane,
            enable_scope_stats=args.enable_scope_stats,
            log_level=args.log_level,
            ring_task_window=16_384,
            ring_heap=1_073_741_824,
            ring_dep_pool=16_384,
        ),
        rtol=1e-2,
        atol=1e-2,
        compare_fn=compare_fn,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
