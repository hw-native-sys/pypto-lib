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
from decode_cp_allgather import (
    KV_B_DYN,
    KV_T_DYN,
    DECODE_GROUP_CAP,
)
from decode_hca import decode_hca, decode_hca_tp1
from decode_csa import decode_csa, decode_csa_tp1
from decode_swa import decode_swa, decode_swa_tp1
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
HCA_CMP_STORAGE_BLOCK_SIZE = hca.CMP_STORAGE_BLOCK_SIZE
ATTENTION_WINDOW_ROWS = swa.ATTENTION_WINDOW_ROWS
O_WINDOW_ROWS = swa.O_WINDOW_ROWS

# HCA keeps the common hidden/attention layout above and adds ratio-128
# compressor pools and request-axis metadata.
HCA_B_DYN = hca.B_DYN
HCA_CMP_BLOCK_NUM_DYN = hca.CMP_BLOCK_NUM_DYN
HCA_CMP_TABLE_BLOCKS_DYN = hca.CMP_TABLE_BLOCKS_DYN
HCA_COMPRESS_STATE_BLOCK_NUM_DYN = hca.COMPRESS_STATE_BLOCK_NUM_DYN
HCA_B = hca.B
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
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin",
    "kv_cache", "swa_slot_mapping", "swa_indices", "swa_lens",
    "position_ids", "attn_sink",
    "wo_a", "wo_b", "wo_b_scale",
)

_HCA_INPUT_NAMES = (
    "x_hc",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state", "compress_state_block_table",
    "kv_cache", "cmp_kv", "cmp_block_table",
    "ori_slot_mapping", "window_swa_indices", "window_swa_lens",
    "cmp_slot_mapping", "state_slot_mapping",
    "position_ids_local", "position_ids", "kv_seq_lens",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
)

_CSA_INPUT_NAMES = (
    "x_hc",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base",
    "attn_norm_w", "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "cmp_freqs_cos", "cmp_freqs_sin",
    "cmp_wkv", "cmp_wgate", "cmp_ape", "cmp_norm_w",
    "compress_state", "compress_state_block_table",
    "idx_wq_b", "idx_wq_b_scale", "weights_proj", "hadamard_idx",
    "inner_wkv", "inner_wgate", "inner_ape", "inner_norm_w",
    "inner_compress_state", "inner_compress_state_block_table",
    "kv_cache", "cmp_kv", "cmp_block_table",
    "idx_kv_cache", "idx_kv_scale", "idx_block_table",
    "ori_slot_mapping", "window_swa_indices", "window_swa_lens",
    "cmp_slot_mapping", "idx_slot_mapping",
    "state_slot_mapping", "inner_state_slot_mapping",
    "position_ids_local", "position_ids", "kv_seq_lens",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
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
        if module.MAX_SEQ_LEN != MODEL_CONFIG.max_position_embeddings:
            raise ValueError(
                f"{module.__name__} context ceiling {module.MAX_SEQ_LEN} "
                f"does not match model max_position_embeddings="
                f"{MODEL_CONFIG.max_position_embeddings}",
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


def _resolve_owner_fixture(start_pos, num_tokens_per_owner):
    """Resolve the physical local batch and optional owner-major positions."""
    import torch

    owner_start_positions = None
    requested_counts = None
    if num_tokens_per_owner is not None:
        requested_counts = torch.as_tensor(num_tokens_per_owner, dtype=torch.int32).reshape(-1)
        if requested_counts.numel() != N_RANKS:
            raise ValueError(
                f"num_tokens_per_owner needs {N_RANKS} entries, got {requested_counts.numel()}",
            )
        if (
            bool((requested_counts < 0).any())
            or bool((requested_counts % DECODE_SEQ != 0).any())
        ):
            raise ValueError(
                f"num_tokens_per_owner values must be non-negative multiples of "
                f"S={DECODE_SEQ}: {requested_counts.tolist()}",
            )
        requests_per_owner = requested_counts // DECODE_SEQ
        if isinstance(start_pos, (list, tuple)) and len(start_pos) == int(requests_per_owner.sum()):
            batch = int(requests_per_owner.max())
            if batch == 0:
                raise ValueError("packed per-owner start positions need at least one active request")
            owner_start_positions = torch.zeros(N_RANKS, batch, dtype=torch.int32)
            offset = 0
            for rank, request_count in enumerate(requests_per_owner.tolist()):
                owner_start_positions[rank, :request_count] = torch.tensor(
                    start_pos[offset : offset + request_count], dtype=torch.int32,
                )
                offset += request_count

    if owner_start_positions is None:
        batch = _active_batch(start_pos)
    local_t = batch * DECODE_SEQ
    _validate_active_tokens(local_t)
    if requested_counts is None:
        requested_counts = torch.full((N_RANKS,), local_t, dtype=torch.int32)
    if bool((requested_counts > local_t).any()):
        raise ValueError(
            f"num_tokens_per_owner values must be at most physical local_t={local_t}: "
            f"{requested_counts.tolist()}",
        )
    return batch, local_t, requested_counts.contiguous(), owner_start_positions


def _distributed_shape(module, name, shape):
    if name == "wo_a":
        return [EP_SIZE, module.LOCAL_O_GROUPS, module.O_LORA, module.O_GROUP_IN]
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


def build_layer_shape_report(layer_id, start_pos=None, num_tokens_per_owner=None):
    """Build current attention specs and audit their future EP-layer shapes."""
    from golden import TensorSpec

    kind = attention_kind_for_layer(layer_id)
    module = _ATTENTION_MODULES[kind]
    batch, active_tokens, _, owner_start_positions = _resolve_owner_fixture(
        start_pos, num_tokens_per_owner,
    )
    if batch > module.B:
        raise ValueError(
            f"{kind} batch {batch} exceeds TP{TP_SIZE} local capacity {module.B}",
        )

    report_start_pos = start_pos
    if owner_start_positions is not None:
        report_start_pos = [int(owner_start_positions.max())] * batch
    specs = module.build_tensor_specs(start_pos=report_start_pos, batch=batch)
    distributed_shapes = {}
    for spec in specs:
        if not isinstance(spec, TensorSpec):
            continue
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
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
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
    owner_tokens: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one SWA attention and one MoE without clearing shared signals."""
    with pl.scope():
        if TP_SIZE == 1:
            decode_swa_tp1(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
                gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                kv_cache, swa_slot_mapping, swa_indices, swa_lens,
                position_ids, attn_sink,
                wo_a, wo_b, wo_b_scale,
                x_attn_active,
            )
        else:
            decode_swa(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
                gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
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
            if token < owner_tokens:
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
            layer_id, owner_tokens, my_rank, moe_epoch,
        )

        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_active_trim"):
            if token < owner_tokens:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = (x_moe_next[token : token + 1, 0 : HC_MULT, 0 : D])
            elif token < local_t:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = pl.full(
                    [1, HC_MULT, D], dtype=pl.FP32, value=0.0,
                )
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
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Standalone child: run one layer and clear its fresh MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(0, KV_T_DYN)
    swa_indices.bind_dynamic(0, T_DYN)
    swa_lens.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, T_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)

    local_t = pl.cast(pl.tensor.dim(x_hc, 0), pl.INT32)
    owner_tokens = pl.read(num_tokens_per_owner, [my_rank])
    if owner_tokens < 0:
        owner_tokens = pl.cast(0, pl.INT32)
    if owner_tokens > local_t:
        owner_tokens = local_t

    decode_layer_swa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
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
        layer_id, group_base, tp_rank, owner_tokens, local_t, my_rank, moe_epoch,
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
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    swa_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch one standalone SWA+MoE layer child on every EP rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    swa_slot_mapping.bind_dynamic(1, KV_T_DYN)
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
            num_tokens_per_owner, layer_id, group_base, tp_rank, rank,
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
    cmp_freqs_cos: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[HCA_COMPRESS_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[KV_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
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
    owner_tokens: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one HCA attention and one MoE without clearing shared signals."""
    with pl.scope():
        if TP_SIZE == 1:
            decode_hca_tp1(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
                gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin, cmp_freqs_cos, cmp_freqs_sin,
                cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
                compress_state, compress_state_block_table,
                kv_cache, cmp_kv, cmp_block_table,
                ori_slot_mapping, window_swa_indices, window_swa_lens,
                cmp_slot_mapping, state_slot_mapping,
                position_ids_local, kv_seq_lens, attn_sink,
                wo_a, wo_b, wo_b_scale,
                x_attn_active,
            )
        else:
            decode_hca(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
                gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                cmp_freqs_cos, cmp_freqs_sin,
                cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
                compress_state, compress_state_block_table,
                kv_cache, cmp_kv, cmp_block_table,
                ori_slot_mapping, window_swa_indices, window_swa_lens,
                cmp_slot_mapping, state_slot_mapping,
                position_ids_local, position_ids, kv_seq_lens, attn_sink,
                wo_a, wo_b, wo_b_scale,
                x_attn_active,
                gather_window, gather_signal,
                attention_window, attention_signal, o_window, o_signal,
                group_base, tp_rank, local_t,
            )

    with pl.scope():
        x_attn_moe = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_attn_pack"):
            if token < owner_tokens:
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
            layer_id, owner_tokens, my_rank, moe_epoch,
        )

        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_active_trim"):
            if token < owner_tokens:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = (x_moe_next[token : token + 1, 0 : HC_MULT, 0 : D])
            elif token < local_t:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = pl.full(
                    [1, HC_MULT, D], dtype=pl.FP32, value=0.0,
                )
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
    cmp_freqs_cos: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[HCA_COMPRESS_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[KV_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Standalone child: run one HCA layer and clear its fresh MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, KV_B_DYN)
    cmp_freqs_sin.bind_dynamic(0, KV_B_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, HCA_COMPRESS_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, KV_B_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, HCA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, HCA_B_DYN)
    cmp_block_table.bind_dynamic(1, HCA_CMP_TABLE_BLOCKS_DYN)
    ori_slot_mapping.bind_dynamic(0, KV_T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, KV_T_DYN)
    state_slot_mapping.bind_dynamic(0, KV_T_DYN)
    position_ids_local.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, KV_T_DYN)
    kv_seq_lens.bind_dynamic(0, HCA_B_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)

    local_t = pl.cast(pl.tensor.dim(x_hc, 0), pl.INT32)
    owner_tokens = pl.read(num_tokens_per_owner, [my_rank])
    if owner_tokens < 0:
        owner_tokens = pl.cast(0, pl.INT32)
    if owner_tokens > local_t:
        owner_tokens = local_t

    decode_layer_hca(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_freqs_cos, cmp_freqs_sin,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        compress_state, compress_state_block_table,
        kv_cache, cmp_kv, cmp_block_table,
        ori_slot_mapping, window_swa_indices, window_swa_lens,
        cmp_slot_mapping, state_slot_mapping,
        position_ids_local, position_ids, kv_seq_lens, attn_sink,
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
        layer_id, group_base, tp_rank, owner_tokens, local_t, my_rank, moe_epoch,
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
    cmp_freqs_cos: pl.Tensor[[N_RANKS, KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_freqs_sin: pl.Tensor[[N_RANKS, KV_B_DYN, ROPE_HEAD_DIM // 2], pl.FP32],
    cmp_wkv: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[N_RANKS, HCA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[N_RANKS, HCA_COMPRESS_RATIO, HCA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[N_RANKS, HCA_COMPRESS_STATE_BLOCK_NUM_DYN, HCA_COMPRESS_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[N_RANKS, KV_B_DYN, HCA_COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, HCA_CMP_BLOCK_NUM_DYN, HCA_CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, HCA_B_DYN, HCA_CMP_TABLE_BLOCKS_DYN], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT32],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch one standalone HCA+MoE layer child on every EP rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    cmp_freqs_cos.bind_dynamic(1, KV_B_DYN)
    cmp_freqs_sin.bind_dynamic(1, KV_B_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    compress_state.bind_dynamic(1, HCA_COMPRESS_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, KV_B_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, HCA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, HCA_B_DYN)
    cmp_block_table.bind_dynamic(2, HCA_CMP_TABLE_BLOCKS_DYN)
    ori_slot_mapping.bind_dynamic(1, KV_T_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    window_swa_lens.bind_dynamic(1, T_DYN)
    cmp_slot_mapping.bind_dynamic(1, KV_T_DYN)
    state_slot_mapping.bind_dynamic(1, KV_T_DYN)
    position_ids_local.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, KV_T_DYN)
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
            cmp_freqs_cos[rank], cmp_freqs_sin[rank],
            cmp_wkv[rank], cmp_wgate[rank], cmp_ape[rank], cmp_norm_w[rank],
            compress_state[rank], compress_state_block_table[rank],
            kv_cache[rank], cmp_kv[rank], cmp_block_table[rank],
            ori_slot_mapping[rank], window_swa_indices[rank],
            window_swa_lens[rank], cmp_slot_mapping[rank],
            state_slot_mapping[rank], position_ids_local[rank], position_ids[rank],
            kv_seq_lens[rank],
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
            num_tokens_per_owner, layer_id, group_base, tp_rank, rank,
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
    cmp_freqs_cos: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[KV_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[KV_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
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
    owner_tokens: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one CSA attention and one MoE without clearing shared signals."""
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
                position_ids_local, kv_seq_lens, attn_sink,
                wo_a, wo_b, wo_b_scale,
                x_attn_active,
            )
        else:
            decode_csa(
                x_hc,
                hc_attn_fn, hc_attn_scale, hc_attn_base,
                attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
                gamma_cq, gamma_ckv,
                freqs_cos, freqs_sin,
                cmp_freqs_cos, cmp_freqs_sin,
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
                position_ids_local, position_ids, kv_seq_lens, attn_sink,
                wo_a, wo_b, wo_b_scale,
                x_attn_active,
                gather_window, gather_signal,
                attention_window, attention_signal, o_window, o_signal,
                group_base, tp_rank, local_t,
            )

    with pl.scope():
        x_attn_moe = pl.create_tensor([MOE_TOKENS, HC_MULT, D], dtype=pl.FP32)
        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_attn_pack"):
            if token < owner_tokens:
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
            layer_id, owner_tokens, my_rank, moe_epoch,
        )

        for token in pl.spmd(MOE_TOKENS, name_hint="decode_layer_active_trim"):
            if token < owner_tokens:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = (x_moe_next[token : token + 1, 0 : HC_MULT, 0 : D])
            elif token < local_t:
                x_next[token : token + 1, 0 : HC_MULT, 0 : D] = pl.full(
                    [1, HC_MULT, D], dtype=pl.FP32, value=0.0,
                )
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
    cmp_freqs_cos: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[KV_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[[CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[KV_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[T_DYN], pl.INT32],
    position_ids: pl.Tensor[[KV_T_DYN], pl.INT32],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Run one standalone CSA+MoE layer and clear its MoE signals."""
    x_hc.bind_dynamic(0, T_DYN)
    freqs_cos.bind_dynamic(0, T_DYN)
    cmp_freqs_cos.bind_dynamic(0, KV_T_DYN)
    cmp_freqs_sin.bind_dynamic(0, KV_T_DYN)
    freqs_sin.bind_dynamic(0, T_DYN)
    compress_state.bind_dynamic(0, CSA_MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(0, KV_B_DYN)
    inner_compress_state.bind_dynamic(0, CSA_INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(0, KV_B_DYN)
    kv_cache.bind_dynamic(0, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(0, CSA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(0, CSA_B_DYN)
    idx_kv_cache.bind_dynamic(0, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(0, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(0, CSA_B_DYN)
    ori_slot_mapping.bind_dynamic(0, KV_T_DYN)
    window_swa_indices.bind_dynamic(0, T_DYN)
    window_swa_lens.bind_dynamic(0, T_DYN)
    cmp_slot_mapping.bind_dynamic(0, KV_T_DYN)
    idx_slot_mapping.bind_dynamic(0, KV_T_DYN)
    state_slot_mapping.bind_dynamic(0, KV_T_DYN)
    inner_state_slot_mapping.bind_dynamic(0, KV_T_DYN)
    position_ids_local.bind_dynamic(0, T_DYN)
    position_ids.bind_dynamic(0, KV_T_DYN)
    kv_seq_lens.bind_dynamic(0, CSA_B_DYN)
    x_attn_active.bind_dynamic(0, T_DYN)
    x_next.bind_dynamic(0, T_DYN)

    local_t = pl.cast(pl.tensor.dim(x_hc, 0), pl.INT32)
    owner_tokens = pl.read(num_tokens_per_owner, [my_rank])
    if owner_tokens < 0:
        owner_tokens = pl.cast(0, pl.INT32)
    if owner_tokens > local_t:
        owner_tokens = local_t

    decode_layer_csa(
        x_hc,
        hc_attn_fn, hc_attn_scale, hc_attn_base,
        attn_norm_w, wq_a, wq_b, wq_b_scale, wkv,
        gamma_cq, gamma_ckv,
        freqs_cos, freqs_sin,
        cmp_freqs_cos, cmp_freqs_sin,
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
        position_ids_local, position_ids, kv_seq_lens, attn_sink,
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
        layer_id, group_base, tp_rank, owner_tokens, local_t, my_rank, moe_epoch,
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
    cmp_freqs_cos: pl.Tensor[[N_RANKS, KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_freqs_sin: pl.Tensor[[N_RANKS, KV_T_DYN, ROPE_HEAD_DIM], pl.BF16],
    cmp_wkv: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[N_RANKS, CSA_MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[N_RANKS, HEAD_DIM], pl.BF16],
    compress_state: pl.InOut[pl.Tensor[[N_RANKS, CSA_MAIN_STATE_BLOCK_NUM_DYN, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32]],
    compress_state_block_table: pl.Tensor[[N_RANKS, KV_B_DYN, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[N_RANKS, Q_LORA, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[N_RANKS, CSA_IDX_N_HEADS * CSA_IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[N_RANKS, D, CSA_IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[N_RANKS, CSA_INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[N_RANKS, CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[N_RANKS, CSA_IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.InOut[pl.Tensor[[N_RANKS, CSA_INNER_STATE_BLOCK_NUM_DYN, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32]],
    inner_compress_state_block_table: pl.Tensor[[N_RANKS, KV_B_DYN, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.InOut[pl.Tensor[[N_RANKS, ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_kv: pl.InOut[pl.Tensor[[N_RANKS, CSA_CMP_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16]],
    cmp_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.InOut[pl.Tensor[[N_RANKS, CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, CSA_IDX_HEAD_DIM], pl.INT8]],
    idx_kv_scale: pl.InOut[pl.Tensor[[N_RANKS, CSA_IDX_CACHE_BLOCK_NUM_DYN, BLOCK_SIZE, 1, 1], pl.FP32]],
    idx_block_table: pl.Tensor[[N_RANKS, CSA_B_DYN, CSA_IDX_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    window_swa_indices: pl.Tensor[[N_RANKS, T_DYN, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    idx_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    state_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT64],
    position_ids_local: pl.Tensor[[N_RANKS, T_DYN], pl.INT32],
    position_ids: pl.Tensor[[N_RANKS, KV_T_DYN], pl.INT32],
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
    num_tokens_per_owner: pl.Tensor[[N_RANKS], pl.INT32],
    layer_id: pl.Scalar[pl.INT32],
    local_t: pl.Scalar[pl.INT32],
):
    """Launch one complete CSA+MoE child per EP-world rank."""
    x_hc.bind_dynamic(1, T_DYN)
    freqs_cos.bind_dynamic(1, T_DYN)
    freqs_sin.bind_dynamic(1, T_DYN)
    cmp_freqs_cos.bind_dynamic(1, KV_T_DYN)
    cmp_freqs_sin.bind_dynamic(1, KV_T_DYN)
    compress_state.bind_dynamic(1, CSA_MAIN_STATE_BLOCK_NUM_DYN)
    compress_state_block_table.bind_dynamic(1, KV_B_DYN)
    inner_compress_state.bind_dynamic(1, CSA_INNER_STATE_BLOCK_NUM_DYN)
    inner_compress_state_block_table.bind_dynamic(1, KV_B_DYN)
    kv_cache.bind_dynamic(1, ORI_BLOCK_NUM_DYN)
    cmp_kv.bind_dynamic(1, CSA_CMP_BLOCK_NUM_DYN)
    cmp_block_table.bind_dynamic(1, CSA_B_DYN)
    idx_kv_cache.bind_dynamic(1, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_kv_scale.bind_dynamic(1, CSA_IDX_CACHE_BLOCK_NUM_DYN)
    idx_block_table.bind_dynamic(1, CSA_B_DYN)
    ori_slot_mapping.bind_dynamic(1, KV_T_DYN)
    window_swa_indices.bind_dynamic(1, T_DYN)
    window_swa_lens.bind_dynamic(1, T_DYN)
    cmp_slot_mapping.bind_dynamic(1, KV_T_DYN)
    idx_slot_mapping.bind_dynamic(1, KV_T_DYN)
    state_slot_mapping.bind_dynamic(1, KV_T_DYN)
    inner_state_slot_mapping.bind_dynamic(1, KV_T_DYN)
    position_ids_local.bind_dynamic(1, T_DYN)
    position_ids.bind_dynamic(1, KV_T_DYN)
    kv_seq_lens.bind_dynamic(1, CSA_B_DYN)
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
            inner_state_slot_mapping[rank], position_ids_local[rank],
            position_ids[rank], kv_seq_lens[rank], attn_sink[rank],
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
            num_tokens_per_owner, layer_id, group_base, tp_rank, rank,
            pl.const(1, pl.INT32),
            device=rank,
        )
    return x_next


def _build_attention_specs(module, local_t, start_pos, owner_start_positions, expand_spec):
    """Build one repeated or independently positioned fixture per TP group."""
    from golden import TensorSpec

    if owner_start_positions is None:
        return [
            expand_spec(spec)
            for spec in module.build_distributed_tensor_specs(local_t, start_pos=start_pos)
            if isinstance(spec, TensorSpec) and spec.name != "x_out"
        ]

    group_sources = []
    for group_base in range(0, N_RANKS, TP_SIZE):
        group_starts = (
            owner_start_positions[group_base : group_base + TP_SIZE]
            .reshape(-1)
            .tolist()
        )
        group_sources.append({
            source.name: source
            for source in module.build_distributed_tensor_specs(local_t, start_pos=group_starts)
            if isinstance(source, TensorSpec) and source.name != "x_out"
        })

    source_names = set(group_sources[0])
    if any(set(sources) != source_names for sources in group_sources[1:]):
        raise ValueError(f"{module.__name__} distributed specs differ across DP groups")

    specs = []
    for name, source in group_sources[0].items():
        expected = (source.dtype, len(source.shape))
        if any(
            (sources[name].dtype, len(sources[name].shape)) != expected
            for sources in group_sources[1:]
        ):
            raise ValueError(f"{module.__name__} spec {name!r} differs across DP groups")
        shapes = [list(sources[name].shape) for sources in group_sources]
        if any(shape[0] != TP_SIZE for shape in shapes):
            raise ValueError(f"{module.__name__} spec {name!r} lacks its TP rank axis")
        trailing_shape = [
            max(shape[axis] for shape in shapes)
            for axis in range(1, len(shapes[0]))
        ]

        def init_value(name=name, group_sources=group_sources, trailing_shape=trailing_shape):
            result = None
            for group, sources in enumerate(group_sources):
                value = sources[name].create_tensor()
                if result is None:
                    result = value.new_zeros([N_RANKS, *trailing_shape])
                slices = (slice(group * TP_SIZE, (group + 1) * TP_SIZE),) + tuple(
                    slice(0, extent) for extent in value.shape[1:]
                )
                result[slices] = value
            return result.contiguous()

        spec = TensorSpec(
            name, [N_RANKS, *trailing_shape], source.dtype,
            init_value=init_value,
        )
        spec.resident = source.resident
        specs.append(spec)
    return specs


def _mask_inactive_specs(specs, owner_token_counts, local_t, kind):
    """Make inactive owner rows and cache writes fail closed in the fixture."""
    from golden import TensorSpec

    specs_by_name = {spec.name: spec for spec in specs}

    def transform(name, fn):
        source = specs_by_name.get(name)
        if source is None or not isinstance(source, TensorSpec):
            return

        def init_value():
            return fn(source.create_tensor())

        transformed = TensorSpec(name, list(source.shape), source.dtype, init_value=init_value)
        transformed.resident = source.resident
        specs_by_name[name] = transformed

    def mask_local_rows(value, fill_value):
        value = value.clone()
        for rank, active_tokens in enumerate(owner_token_counts.tolist()):
            value[rank, active_tokens:] = fill_value
        return value.contiguous()

    def mask_group_slots(value):
        value = value.clone()
        for rank in range(N_RANKS):
            group_base = rank - rank % TP_SIZE
            for owner_offset in range(TP_SIZE):
                owner_rank = group_base + owner_offset
                active_tokens = int(owner_token_counts[owner_rank])
                segment_begin = owner_offset * local_t
                value[rank, segment_begin + active_tokens : segment_begin + local_t] = -1
        return value.contiguous()

    def mask_request_rows(value):
        value = value.clone()
        for rank, active_tokens in enumerate(owner_token_counts.tolist()):
            value[rank, active_tokens // DECODE_SEQ :] = 0
        return value.contiguous()

    transform("x_hc", lambda value: mask_local_rows(value, 0))
    transform("input_ids", lambda value: mask_local_rows(value, 0))
    if kind == "swa":
        slot_names = ("swa_slot_mapping",)
        index_name, lens_name = "swa_indices", "swa_lens"
    elif kind == "hca":
        slot_names = ("ori_slot_mapping", "cmp_slot_mapping", "state_slot_mapping")
        index_name, lens_name = "window_swa_indices", "window_swa_lens"
    else:
        slot_names = (
            "ori_slot_mapping", "cmp_slot_mapping", "idx_slot_mapping",
            "state_slot_mapping", "inner_state_slot_mapping",
        )
        index_name, lens_name = "window_swa_indices", "window_swa_lens"
    for name in slot_names:
        transform(name, mask_group_slots)
    transform(index_name, lambda value: mask_local_rows(value, -1))
    transform(lens_name, lambda value: mask_local_rows(value, 0))
    if kind != "swa":
        transform("kv_seq_lens", mask_request_rows)
    return [specs_by_name[spec.name] for spec in specs]


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
    )
    expanded.resident = spec.resident
    return expanded


def build_swa_layer_specs(start_pos=None, layer_id=0, num_tokens_per_owner=None):
    """Build one current-SWA fixture followed by the current MoE fixture."""
    import inspect

    import torch
    from golden import ScalarSpec, TensorSpec

    if attention_kind_for_layer(layer_id) != "swa":
        raise ValueError(f"layer {layer_id} is not an SWA layer")
    batch, local_t, owner_token_counts, owner_start_positions = _resolve_owner_fixture(
        start_pos, num_tokens_per_owner,
    )
    if batch > swa.B:
        raise ValueError(
            f"SWA batch {batch} exceeds TP{TP_SIZE} local capacity {swa.B}",
        )

    specs = _build_attention_specs(
        swa, local_t, start_pos, owner_start_positions, _expand_swa_spec,
    )

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
            ),
            TensorSpec(
                "x_moe_next",
                [N_RANKS, MOE_TOKENS, HC_MULT, D],
                torch.float32,
            ),
            TensorSpec(
                "x_next",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
            ),
            TensorSpec(
                "num_tokens_per_owner", [N_RANKS], torch.int32,
                init_value=lambda: owner_token_counts.clone(),
            ),
            ScalarSpec("layer_id", torch.int32, layer_id),
            ScalarSpec("local_t", torch.int32, local_t),
        ],
    )

    specs = _mask_inactive_specs(specs, owner_token_counts, local_t, "swa")
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
    owner_token_counts = tensors["num_tokens_per_owner"]
    tensors["x_attn_active"].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = { name: tensors[name][group_begin:group_end] for name in _SWA_INPUT_NAMES }
        group_tensors["x_out"] = tensors["x_attn_active"][group_begin:group_end]
        group_tensors["local_t"] = local_t
        swa.golden_decode_swa(group_tensors)

    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    for rank, active_tokens in enumerate(owner_token_counts.tolist()):
        tensors["x_attn_active"][rank, active_tokens:].zero_()
        x_attn_moe[rank, :active_tokens].copy_(tensors["x_attn_active"][rank, :active_tokens])
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn_moe
    moe_tensors["x_next"] = tensors["x_moe_next"]
    moe_tensors["num_tokens"] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors["x_next"].zero_()
    for rank, active_tokens in enumerate(owner_token_counts.tolist()):
        tensors["x_next"][rank, :active_tokens].copy_(tensors["x_moe_next"][rank, :active_tokens])


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
    )
    expanded.resident = spec.resident
    return expanded


def build_hca_layer_specs(start_pos=None, layer_id=3, num_tokens_per_owner=None):
    """Build one current-HCA fixture followed by the current MoE fixture."""
    import inspect

    import torch
    from golden import ScalarSpec, TensorSpec

    if attention_kind_for_layer(layer_id) != "hca":
        raise ValueError(f"layer {layer_id} is not an HCA layer")
    batch, local_t, owner_token_counts, owner_start_positions = _resolve_owner_fixture(
        start_pos, num_tokens_per_owner,
    )
    if batch > hca.B:
        raise ValueError(
            f"HCA batch {batch} exceeds TP{TP_SIZE} local capacity {hca.B}",
        )

    specs = _build_attention_specs(
        hca, local_t, start_pos, owner_start_positions, _expand_hca_spec,
    )

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
            ),
            TensorSpec(
                "x_moe_next",
                [N_RANKS, MOE_TOKENS, HC_MULT, D],
                torch.float32,
            ),
            TensorSpec(
                "x_next",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
            ),
            TensorSpec(
                "num_tokens_per_owner", [N_RANKS], torch.int32,
                init_value=lambda: owner_token_counts.clone(),
            ),
            ScalarSpec("layer_id", torch.int32, layer_id),
            ScalarSpec("local_t", torch.int32, local_t),
        ],
    )

    specs = _mask_inactive_specs(specs, owner_token_counts, local_t, "hca")
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
    owner_token_counts = tensors["num_tokens_per_owner"]
    tensors["x_attn_active"].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = { name: tensors[name][group_begin:group_end] for name in _HCA_INPUT_NAMES }
        group_tensors["x_out"] = tensors["x_attn_active"][group_begin:group_end]
        group_tensors["local_t"] = local_t
        hca.golden_decode_hca(group_tensors)

    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    for rank, active_tokens in enumerate(owner_token_counts.tolist()):
        tensors["x_attn_active"][rank, active_tokens:].zero_()
        x_attn_moe[rank, :active_tokens].copy_(tensors["x_attn_active"][rank, :active_tokens])
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn_moe
    moe_tensors["x_next"] = tensors["x_moe_next"]
    moe_tensors["num_tokens"] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors["x_next"].zero_()
    for rank, active_tokens in enumerate(owner_token_counts.tolist()):
        tensors["x_next"][rank, :active_tokens].copy_(tensors["x_moe_next"][rank, :active_tokens])


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
    )
    expanded.resident = spec.resident
    return expanded


def build_csa_layer_specs(start_pos=None, layer_id=2, num_tokens_per_owner=None):
    """Build one current-CSA fixture followed by the current MoE fixture."""
    import inspect

    import torch
    from golden import ScalarSpec, TensorSpec

    if attention_kind_for_layer(layer_id) != "csa":
        raise ValueError(f"layer {layer_id} is not a CSA layer")
    batch, local_t, owner_token_counts, owner_start_positions = _resolve_owner_fixture(
        start_pos, num_tokens_per_owner,
    )
    if batch > csa.B:
        raise ValueError(
            f"CSA batch {batch} exceeds TP{TP_SIZE} local capacity {csa.B}",
        )

    specs = _build_attention_specs(
        csa, local_t, start_pos, owner_start_positions, _expand_csa_spec,
    )

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
            ),
            TensorSpec(
                "x_moe_next",
                [N_RANKS, MOE_TOKENS, HC_MULT, D],
                torch.float32,
            ),
            TensorSpec(
                "x_next",
                [N_RANKS, local_t, HC_MULT, D],
                torch.float32,
            ),
            TensorSpec(
                "num_tokens_per_owner", [N_RANKS], torch.int32,
                init_value=lambda: owner_token_counts.clone(),
            ),
            ScalarSpec("layer_id", torch.int32, layer_id),
            ScalarSpec("local_t", torch.int32, local_t),
        ],
    )

    specs = _mask_inactive_specs(specs, owner_token_counts, local_t, "csa")
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
    owner_token_counts = tensors["num_tokens_per_owner"]
    tensors["x_attn_active"].zero_()
    for group in range(TP_GROUPS):
        group_begin = group * TP_SIZE
        group_end = group_begin + TP_SIZE
        group_tensors = { name: tensors[name][group_begin:group_end] for name in _CSA_INPUT_NAMES }
        group_tensors["x_out"] = tensors["x_attn_active"][group_begin:group_end]
        group_tensors["local_t"] = local_t
        csa.golden_decode_csa(group_tensors)

    x_attn_moe = torch.zeros(N_RANKS, MOE_TOKENS, HC_MULT, D, dtype=torch.float32)
    for rank, active_tokens in enumerate(owner_token_counts.tolist()):
        tensors["x_attn_active"][rank, active_tokens:].zero_()
        x_attn_moe[rank, :active_tokens].copy_(tensors["x_attn_active"][rank, :active_tokens])
    moe_tensors = dict(tensors)
    moe_tensors["x_hc"] = x_attn_moe
    moe_tensors["x_next"] = tensors["x_moe_next"]
    moe_tensors["num_tokens"] = local_t
    moe_module.golden_moe(moe_tensors)
    tensors["x_next"].zero_()
    for rank, active_tokens in enumerate(owner_token_counts.tolist()):
        tensors["x_next"][rank, :active_tokens].copy_(tensors["x_moe_next"][rank, :active_tokens])


def _parse_start_pos(raw):
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--start-pos must contain at least one integer")
    values = [int(part) for part in parts]
    return values[0] if len(values) == 1 else values


def _parse_num_tokens_per_owner(raw):
    if raw is None:
        return None
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--num-tokens-per-owner must contain at least one integer")
    return values[0] if len(values) == 1 else values


def _active_owner_rows_compare(compare):
    """Apply an existing tensor comparator only to each owner's active prefix."""
    def compare_active(actual, expected, **kwargs):
        import torch

        owner_counts = kwargs.get("inputs", {}).get("num_tokens_per_owner")
        if owner_counts is None:
            return False, "    missing num_tokens_per_owner for active-row comparison"
        actual_rows = []
        expected_rows = []
        for rank, active_tokens in enumerate(owner_counts.tolist()):
            if active_tokens:
                actual_rows.append(actual[rank, :active_tokens])
                expected_rows.append(expected[rank, :active_tokens])
        if not actual_rows:
            return True, ""
        return compare(
            torch.cat(actual_rows, dim=0),
            torch.cat(expected_rows, dim=0),
            **kwargs,
        )

    return compare_active


def main():
    import argparse

    from golden import (
        mapped_pool_ratio_allclose,
        ratio_reldiff,
        run,
    )
    from pypto.ir import DistributedConfig

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
        "--start-pos", type=str, default=None,
        help=(
            "a scalar selects batch=1; a comma-separated list sets the batch; "
            "with per-owner token counts, a compact list maps active requests in rank order"
        ),
    )
    parser.add_argument(
        "--num-tokens-per-owner", type=str, default=None,
        help=f"one broadcast count or {N_RANKS} comma-separated active-prefix lengths",
    )
    parser.add_argument(
        "--enable-chip-swimlane", type=int, default=0, choices=range(5),
    )
    parser.add_argument(
        "--enable-scope-stats", action="store_true", default=False,
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--golden-only", action="store_true", default=False,
                        help="compute and persist the golden, then stop before the device run")
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
    num_tokens_per_owner = _parse_num_tokens_per_owner(args.num_tokens_per_owner)
    kind = attention_kind_for_layer(args.layer_id)

    report = build_layer_shape_report(
        args.layer_id, start_pos=start_pos,
        num_tokens_per_owner=num_tokens_per_owner,
    )
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
        specs = build_swa_layer_specs(
            start_pos=start_pos, layer_id=args.layer_id,
            num_tokens_per_owner=num_tokens_per_owner,
        )
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
            "x_attn_active": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1,
            )),
            "x_moe_next": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05,
            )),
            "x_next": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05,
            )),
        }
    elif kind == "hca":
        layer_fn = l3_decode_layer_hca
        specs = build_hca_layer_specs(
            start_pos=start_pos, layer_id=args.layer_id,
            num_tokens_per_owner=num_tokens_per_owner,
        )
        golden_fn = golden_decode_layer_hca
        mapping_shape = (N_RANKS, TP_SIZE * local_t)
        compare_fn = {
            "compress_state": mapped_pool_ratio_allclose(
                "state_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=HCA_COMPRESS_STATE_BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="compressor state",
                atol=1e-3,
                rtol=1.0 / 128,
            ),
            "kv_cache": mapped_pool_ratio_allclose(
                "ori_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="original KV cache",
                atol=1e-3,
                rtol=1.0 / 128,
            ),
            "cmp_kv": mapped_pool_ratio_allclose(
                "cmp_slot_mapping",
                mapping_shape=mapping_shape,
                block_size=HCA_CMP_STORAGE_BLOCK_SIZE,
                leading_rank_axis=True,
                pool_name="compressed KV cache",
                atol=1e-3,
                rtol=1.0 / 128,
            ),
            "x_attn_active": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=3e-3, pct_thd=0.008, max_diff_hd=1,
            )),
            "x_moe_next": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05,
            )),
            "x_next": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05,
            )),
        }
    else:
        layer_fn = l3_decode_layer_csa
        specs = build_csa_layer_specs(
            start_pos=start_pos, layer_id=args.layer_id,
            num_tokens_per_owner=num_tokens_per_owner,
        )
        golden_fn = golden_decode_layer_csa
        mapping_shape = (N_RANKS, TP_SIZE * local_t)
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
            "x_attn_active": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=4e-3, pct_thd=0.008, max_diff_hd=2,
            )),
            "x_moe_next": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05,
            )),
            "x_next": _active_owner_rows_compare(ratio_reldiff(
                diff_thd=0.01, pct_thd=0.05,
            )),
        }

    result = run(
        fn=layer_fn,
        specs=specs,
        golden_fn=golden_fn,
        golden_data=args.golden_data,
        save_data=args.save_data,
        compile_only=args.compile_only,
        golden_only=args.golden_only,
        runtime_dir=args.runtime_dir,
        config=dict(
            dump_passes=args.dump_passes,
            distributed_config=DistributedConfig(
                device_ids=device_ids, num_sub_workers=0,
            ),
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
