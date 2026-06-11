# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Host-side weight loader for the step3p5 8-card TP=EP=8 deployment.

This module maps the HuggingFace-format safetensors checkpoint shipped at
``/mnt/chensiyu-jfs/multi-hardware/models/step3p5_flash_release_hf_mtp3_bf16/``
into the per-card weight bundles consumed by the Wave-3 TP/EP kernels.

Per-rank bundle layout (consolidated from the kernel signatures in
``decode_fwd.py`` / ``decode_layer.py`` / ``mtp.py`` / ``rms_lm_head.py``):

REPLICATED tensors — identical bytes on every rank:
  * ``embed_tokens``                shape ``[VOCAB, HIDDEN]``      BF16
  * ``final_norm_weight``           shape ``[HIDDEN]``             BF16
  * ``input_rms_weight``            shape ``[NUM_HIDDEN_LAYERS, HIDDEN]`` BF16
  * ``post_attn_rms_weight``        shape ``[NUM_HIDDEN_LAYERS, HIDDEN]`` BF16
  * ``q_norm_weight``               shape ``[NUM_HIDDEN_LAYERS, HEAD_DIM]`` BF16
  * ``k_norm_weight``               shape ``[NUM_HIDDEN_LAYERS, HEAD_DIM]`` BF16
  * ``moe_gate_w``                  shape ``[NUM_MOE_LAYERS, HIDDEN, MOE_NUM_EXPERTS]`` FP32
  * ``moe_router_bias``             shape ``[NUM_MOE_LAYERS, MOE_NUM_EXPERTS]`` FP32

TP-SLICED tensors — each rank holds 1/TP_WORLD_SIZE:
  * ``wq_full``                     ``[NUM_FULL_LAYERS, HIDDEN, NUM_HEADS_FULL_LOCAL * HEAD_DIM]`` BF16
  * ``wk_full`` / ``wv_full``       ``[NUM_FULL_LAYERS, HIDDEN, KV_HEADS_LOCAL * HEAD_DIM]`` BF16
  * ``wo_full``                     ``[NUM_FULL_LAYERS, NUM_HEADS_FULL_LOCAL * HEAD_DIM, HIDDEN]`` BF16
  * ``w_g_full``                    ``[NUM_FULL_LAYERS, HIDDEN, NUM_HEADS_FULL_LOCAL_PAD]`` BF16 (zero-padded last dim)
  * ``wq_swa`` / ``wk_swa`` / ``wv_swa`` / ``wo_swa`` / ``w_g_swa`` — SWA equivalents (w_g_swa last dim = NUM_HEADS_SWA_LOCAL_PAD)
  * ``dense_w_gate`` / ``dense_w_up`` ``[NUM_DENSE_LAYERS, HIDDEN, INTERMEDIATE_LOCAL]`` BF16
  * ``dense_w_down``                ``[NUM_DENSE_LAYERS, INTERMEDIATE_LOCAL, HIDDEN]`` BF16
  * ``moe_w_gate_s`` / ``moe_w_up_s`` ``[NUM_MOE_LAYERS, HIDDEN, SHARE_EXPERT_DIM_LOCAL]`` BF16
  * ``moe_w_down_s``                ``[NUM_MOE_LAYERS, SHARE_EXPERT_DIM_LOCAL, HIDDEN]`` BF16
  * ``lm_head_weight``              ``[VOCAB_LOCAL, HIDDEN]`` BF16

EP-SLICED tensors — each rank hosts MOE_NUM_EXPERTS_LOCAL of MOE_NUM_EXPERTS experts:
  * ``moe_w_gate_r`` / ``moe_w_up_r`` ``[NUM_MOE_LAYERS, MOE_NUM_EXPERTS_LOCAL, HIDDEN, MOE_INTERMEDIATE]`` BF16
  * ``moe_w_down_r`` ``[NUM_MOE_LAYERS, MOE_NUM_EXPERTS_LOCAL, MOE_INTERMEDIATE, HIDDEN]`` BF16

MTP tensors (3 next-N-predict layers; LAYER_TYPES[45..47] all SWA):
  * ``mtp_enorm_weight`` / ``mtp_hnorm_weight``  ``[NUM_MTP, HIDDEN]`` BF16 (replicated)
  * ``mtp_eh_proj_weight``                       ``[NUM_MTP, HIDDEN_LOCAL, 2 * HIDDEN]`` BF16 (row slice)
  * ``mtp_shared_head_norm_weight``              ``[NUM_MTP, HIDDEN]`` BF16 (replicated)
  * ``mtp_shared_head_output_weight``            ``[NUM_MTP, VOCAB_LOCAL, HIDDEN]`` BF16 (vocab slice)
  * MTP self-attention/MLP weights mirror the main SWA + dense slicing.

The entry point ``load_step3p5_weights_for_rank`` returns a flat dict of
named ``torch.Tensor``s ready to be passed to the kernel @pl.program
signatures (the caller stacks them along the leading rank axis when
building an 8-rank decode).

The loader is host-side only — it has no pypto runtime dependency and
can be imported / exercised in a CPU-only test environment.

IMPORTANT mapping quirks (see Phase 1.5 ``checkpoint-verifier`` report):
  * ``g_proj`` is stored in HF as ``[NUM_HEADS, HIDDEN]`` but the kernel
    wants ``[HIDDEN, NUM_HEADS_LOCAL]`` — the loader transposes after the
    head-axis slice.
  * MoE keys carry ``.weight`` suffix; ``moe.router_bias`` does NOT and
    is FP32.
  * MTP weights live under ``model.layers.{45, 46, 47}.<member>``.
    The vLLM-style ``.mtp_block.`` infix is stripped at load time.
  * ``shared_head.output`` keeps its checkpoint name (vLLM renames to
    ``shared_head.head`` at load; we don't).
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only import
    import torch

from .config import (
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    LAYER_TYPES,
    LAYER_TYPE_FULL,
    LAYER_TYPE_SWA,
    MOE_INTERMEDIATE,
    MOE_LAYER_INDICES,
    MOE_NUM_EXPERTS,
    NUM_HEADS_FULL,
    NUM_HEADS_FULL_LOCAL_PAD,
    NUM_HEADS_SWA,
    NUM_HEADS_SWA_LOCAL_PAD,
    NUM_HIDDEN_LAYERS,
    NUM_KV_HEADS,
    NUM_NEXTN_PREDICT_LAYERS,
    SHARE_EXPERT_DIM,
    TP_WORLD_SIZE,
    VOCAB,
    ep_global_expert_id,
    is_full_attention,
    is_moe_layer,
)


log = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Per-card derived counts.
# -----------------------------------------------------------------------------
NUM_FULL_LAYERS = sum(
    1 for t in LAYER_TYPES[:NUM_HIDDEN_LAYERS] if t == LAYER_TYPE_FULL
)
NUM_SWA_LAYERS = NUM_HIDDEN_LAYERS - NUM_FULL_LAYERS
NUM_MOE_LAYERS = len(MOE_LAYER_INDICES)
NUM_DENSE_LAYERS = NUM_HIDDEN_LAYERS - NUM_MOE_LAYERS
NUM_MTP = NUM_NEXTN_PREDICT_LAYERS
MTP_HIDDEN_LOCAL = HIDDEN // TP_WORLD_SIZE  # 512 — eh_proj row-slice output dim
EH_IN = 2 * HIDDEN

# Default checkpoint location on the Ascend network share.
DEFAULT_CKPT_DIR = (
    "/mnt/chensiyu-jfs/multi-hardware/models/"
    "step3p5_flash_release_hf_mtp3_bf16"
)

# Bundle key prefixes — kept centralised so verify_bundle_shapes and any
# downstream packer agree on the dictionary layout.
KEY_EMBED = "embed_tokens"
KEY_FINAL_NORM = "final_norm_weight"
KEY_LM_HEAD = "lm_head_weight"
KEY_INPUT_RMS = "input_rms_weight"
KEY_POST_ATTN_RMS = "post_attn_rms_weight"
KEY_Q_NORM = "q_norm_weight"
KEY_K_NORM = "k_norm_weight"

KEY_WQ_FULL = "wq_full"
KEY_WK_FULL = "wk_full"
KEY_WV_FULL = "wv_full"
KEY_WO_FULL = "wo_full"
KEY_WG_FULL = "w_g_full"

KEY_WQ_SWA = "wq_swa"
KEY_WK_SWA = "wk_swa"
KEY_WV_SWA = "wv_swa"
KEY_WO_SWA = "wo_swa"
KEY_WG_SWA = "w_g_swa"

KEY_DENSE_GATE = "dense_w_gate"
KEY_DENSE_UP = "dense_w_up"
KEY_DENSE_DOWN = "dense_w_down"

KEY_MOE_GATE_W = "moe_gate_w"
KEY_MOE_ROUTER_BIAS = "moe_router_bias"
KEY_MOE_W_GATE_R = "moe_w_gate_r"
KEY_MOE_W_UP_R = "moe_w_up_r"
KEY_MOE_W_DOWN_R = "moe_w_down_r"
KEY_MOE_W_GATE_S = "moe_w_gate_s"
KEY_MOE_W_UP_S = "moe_w_up_s"
KEY_MOE_W_DOWN_S = "moe_w_down_s"

KEY_MTP_ENORM = "mtp_enorm_weight"
KEY_MTP_HNORM = "mtp_hnorm_weight"
KEY_MTP_EH_PROJ = "mtp_eh_proj_weight"
KEY_MTP_SH_NORM = "mtp_shared_head_norm_weight"
KEY_MTP_SH_OUT = "mtp_shared_head_output_weight"
KEY_MTP_INPUT_RMS = "mtp_input_rms_weight"
KEY_MTP_POST_ATTN_RMS = "mtp_post_attn_rms_weight"
KEY_MTP_Q_NORM = "mtp_q_norm_weight"
KEY_MTP_K_NORM = "mtp_k_norm_weight"
KEY_MTP_WQ = "mtp_wq_swa"
KEY_MTP_WK = "mtp_wk_swa"
KEY_MTP_WV = "mtp_wv_swa"
KEY_MTP_WO = "mtp_wo_swa"
KEY_MTP_WG = "mtp_w_g_swa"
KEY_MTP_DENSE_GATE = "mtp_dense_w_gate"
KEY_MTP_DENSE_UP = "mtp_dense_w_up"
KEY_MTP_DENSE_DOWN = "mtp_dense_w_down"


# =============================================================================
# Expected per-rank bundle shape table — single source of truth.
# Used by both ``verify_bundle_shapes`` and the bundle constructor.
# =============================================================================
def expected_shapes(tp_world_size: int = TP_WORLD_SIZE) -> dict[str, tuple[int, ...]]:
    """Per-rank expected shape for every bundle key."""
    num_heads_full_local = NUM_HEADS_FULL // tp_world_size
    num_heads_swa_local = NUM_HEADS_SWA // tp_world_size
    # Gate weights are zero-padded to the kernel tile width (NUM_HEADS_*_LOCAL_PAD).
    # The pad values are fixed at 16 for both full and SWA (from config.py).
    num_heads_full_local_pad = NUM_HEADS_FULL_LOCAL_PAD
    num_heads_swa_local_pad = NUM_HEADS_SWA_LOCAL_PAD
    kv_heads_local = NUM_KV_HEADS // tp_world_size
    intermediate_local = INTERMEDIATE // tp_world_size
    share_expert_dim_local = SHARE_EXPERT_DIM // tp_world_size
    vocab_local = VOCAB // tp_world_size
    moe_num_experts_local = MOE_NUM_EXPERTS // tp_world_size

    hidden_q_full_local = num_heads_full_local * HEAD_DIM
    hidden_q_swa_local = num_heads_swa_local * HEAD_DIM
    kv_hidden_local = kv_heads_local * HEAD_DIM
    hidden_local = HIDDEN // tp_world_size

    return {
        # Replicated
        KEY_EMBED: (VOCAB, HIDDEN),
        KEY_FINAL_NORM: (HIDDEN,),
        KEY_INPUT_RMS: (NUM_HIDDEN_LAYERS, HIDDEN),
        KEY_POST_ATTN_RMS: (NUM_HIDDEN_LAYERS, HIDDEN),
        KEY_Q_NORM: (NUM_HIDDEN_LAYERS, HEAD_DIM),
        KEY_K_NORM: (NUM_HIDDEN_LAYERS, HEAD_DIM),
        KEY_MOE_GATE_W: (NUM_MOE_LAYERS, HIDDEN, MOE_NUM_EXPERTS),
        KEY_MOE_ROUTER_BIAS: (NUM_MOE_LAYERS, MOE_NUM_EXPERTS),
        # TP-sliced (attention — full)
        KEY_WQ_FULL: (NUM_FULL_LAYERS, HIDDEN, hidden_q_full_local),
        KEY_WK_FULL: (NUM_FULL_LAYERS, HIDDEN, kv_hidden_local),
        KEY_WV_FULL: (NUM_FULL_LAYERS, HIDDEN, kv_hidden_local),
        KEY_WO_FULL: (NUM_FULL_LAYERS, hidden_q_full_local, HIDDEN),
        KEY_WG_FULL: (NUM_FULL_LAYERS, HIDDEN, num_heads_full_local_pad),
        # TP-sliced (attention — SWA)
        KEY_WQ_SWA: (NUM_SWA_LAYERS, HIDDEN, hidden_q_swa_local),
        KEY_WK_SWA: (NUM_SWA_LAYERS, HIDDEN, kv_hidden_local),
        KEY_WV_SWA: (NUM_SWA_LAYERS, HIDDEN, kv_hidden_local),
        KEY_WO_SWA: (NUM_SWA_LAYERS, hidden_q_swa_local, HIDDEN),
        KEY_WG_SWA: (NUM_SWA_LAYERS, HIDDEN, num_heads_swa_local_pad),
        # TP-sliced (dense MLP, layers 0/1/2)
        KEY_DENSE_GATE: (NUM_DENSE_LAYERS, HIDDEN, intermediate_local),
        KEY_DENSE_UP: (NUM_DENSE_LAYERS, HIDDEN, intermediate_local),
        KEY_DENSE_DOWN: (NUM_DENSE_LAYERS, intermediate_local, HIDDEN),
        # TP-sliced (MoE shared expert)
        KEY_MOE_W_GATE_S: (NUM_MOE_LAYERS, HIDDEN, share_expert_dim_local),
        KEY_MOE_W_UP_S: (NUM_MOE_LAYERS, HIDDEN, share_expert_dim_local),
        KEY_MOE_W_DOWN_S: (NUM_MOE_LAYERS, share_expert_dim_local, HIDDEN),
        # EP-sliced routed experts
        KEY_MOE_W_GATE_R: (
            NUM_MOE_LAYERS, moe_num_experts_local, HIDDEN, MOE_INTERMEDIATE,
        ),
        KEY_MOE_W_UP_R: (
            NUM_MOE_LAYERS, moe_num_experts_local, HIDDEN, MOE_INTERMEDIATE,
        ),
        KEY_MOE_W_DOWN_R: (
            NUM_MOE_LAYERS, moe_num_experts_local, MOE_INTERMEDIATE, HIDDEN,
        ),
        # TP-sliced LM head
        KEY_LM_HEAD: (vocab_local, HIDDEN),
        # MTP
        KEY_MTP_ENORM: (NUM_MTP, HIDDEN),
        KEY_MTP_HNORM: (NUM_MTP, HIDDEN),
        KEY_MTP_EH_PROJ: (NUM_MTP, hidden_local, EH_IN),
        KEY_MTP_SH_NORM: (NUM_MTP, HIDDEN),
        KEY_MTP_SH_OUT: (NUM_MTP, vocab_local, HIDDEN),
        # MTP per-layer (all SWA + dense MLP)
        KEY_MTP_INPUT_RMS: (NUM_MTP, HIDDEN),
        KEY_MTP_POST_ATTN_RMS: (NUM_MTP, HIDDEN),
        KEY_MTP_Q_NORM: (NUM_MTP, HEAD_DIM),
        KEY_MTP_K_NORM: (NUM_MTP, HEAD_DIM),
        KEY_MTP_WQ: (NUM_MTP, HIDDEN, hidden_q_swa_local),
        KEY_MTP_WK: (NUM_MTP, HIDDEN, kv_hidden_local),
        KEY_MTP_WV: (NUM_MTP, HIDDEN, kv_hidden_local),
        KEY_MTP_WO: (NUM_MTP, hidden_q_swa_local, HIDDEN),
        KEY_MTP_WG: (NUM_MTP, HIDDEN, num_heads_swa_local_pad),
        KEY_MTP_DENSE_GATE: (NUM_MTP, HIDDEN, intermediate_local),
        KEY_MTP_DENSE_UP: (NUM_MTP, HIDDEN, intermediate_local),
        KEY_MTP_DENSE_DOWN: (NUM_MTP, intermediate_local, HIDDEN),
    }


# =============================================================================
# Safetensors index loader (lazy import).
# =============================================================================
def _read_index(ckpt_dir: str) -> dict[str, str]:
    """Map tensor-name -> shard-file by reading ``model.safetensors.index.json``.

    Falls back to a single-shard ``model.safetensors`` if the index file is
    absent (some converted checkpoints are stored as one file).
    """
    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index.get("weight_map")
        if weight_map is None:
            raise RuntimeError(
                f"{index_path} does not contain a 'weight_map' field.",
            )
        return dict(weight_map)
    single = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(single):
        raise FileNotFoundError(
            f"Neither {index_path} nor {single} found.",
        )
    # Single-file ckpt: enumerate the tensor names lazily via safetensors.
    from safetensors import safe_open  # noqa: PLC0415 — lazy import

    with safe_open(single, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


class _ShardCache:
    """Lazy per-shard safetensors reader.

    Opens each shard file at most once per ``load_step3p5_weights_for_rank``
    call. The HF index can have ~12 shards for a 290B-parameter model; we
    keep them all open for the duration of a single rank's bundle build.
    """

    def __init__(self, ckpt_dir: str, weight_map: dict[str, str]):
        self.ckpt_dir = ckpt_dir
        self.weight_map = weight_map
        self._handles: dict[str, object] = {}

    def get(self, name: str) -> "torch.Tensor":
        from safetensors import safe_open  # noqa: PLC0415

        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(
                f"Tensor {name!r} not present in checkpoint index "
                f"(have {len(self.weight_map)} known tensors).",
            )
        handle = self._handles.get(shard)
        if handle is None:
            path = os.path.join(self.ckpt_dir, shard)
            handle = safe_open(path, framework="pt")
            handle.__enter__()
            self._handles[shard] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        for h in self._handles.values():
            try:
                h.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 — best effort
                log.warning("shard handle close failed", exc_info=True)
        self._handles.clear()

    def __enter__(self) -> "_ShardCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# =============================================================================
# HF name templates — single source of truth for checkpoint key strings.
#
# These mirror the names produced by the upstream HF conversion script for
# the step3p5_flash_release_hf_mtp3_bf16 ckpt. The MoE layer index is the
# raw checkpoint layer id (not the MoE-only position), so the caller uses
# ``layer_idx`` directly.
# =============================================================================
def _hf_layer_prefix(layer_idx: int) -> str:
    return f"model.layers.{layer_idx}"


def _hf_attn_keys(layer_idx: int) -> dict[str, str]:
    p = _hf_layer_prefix(layer_idx)
    return {
        "q_proj": f"{p}.self_attn.q_proj.weight",
        "k_proj": f"{p}.self_attn.k_proj.weight",
        "v_proj": f"{p}.self_attn.v_proj.weight",
        "o_proj": f"{p}.self_attn.o_proj.weight",
        "q_norm": f"{p}.self_attn.q_norm.weight",
        "k_norm": f"{p}.self_attn.k_norm.weight",
        "g_proj": f"{p}.self_attn.g_proj.weight",
        "input_rms": f"{p}.input_layernorm.weight",
        "post_attn_rms": f"{p}.post_attention_layernorm.weight",
    }


def _hf_dense_mlp_keys(layer_idx: int) -> dict[str, str]:
    p = _hf_layer_prefix(layer_idx)
    return {
        "gate_proj": f"{p}.mlp.gate_proj.weight",
        "up_proj": f"{p}.mlp.up_proj.weight",
        "down_proj": f"{p}.mlp.down_proj.weight",
    }


def _hf_moe_keys(layer_idx: int) -> dict[str, str]:
    p = _hf_layer_prefix(layer_idx)
    return {
        "gate_w": f"{p}.moe.gate.weight",
        "router_bias": f"{p}.moe.router_bias",
        "gate_proj": f"{p}.moe.gate_proj.weight",
        "up_proj": f"{p}.moe.up_proj.weight",
        "down_proj": f"{p}.moe.down_proj.weight",
        "share_gate": f"{p}.share_expert.gate_proj.weight",
        "share_up": f"{p}.share_expert.up_proj.weight",
        "share_down": f"{p}.share_expert.down_proj.weight",
    }


def _hf_mtp_keys(layer_idx: int) -> dict[str, str]:
    """Build HF keys for an MTP layer.

    The on-disk checkpoint stores MTP layers as standard decoder layers
    under ``model.layers.{45, 46, 47}.<member>`` (no extra ``mtp_block``
    infix) plus the MTP-specific projections.
    """
    p = _hf_layer_prefix(layer_idx)
    keys = _hf_attn_keys(layer_idx)
    keys.update(_hf_dense_mlp_keys(layer_idx))
    keys.update({
        "enorm": f"{p}.enorm.weight",
        "hnorm": f"{p}.hnorm.weight",
        "eh_proj": f"{p}.eh_proj.weight",
        "shared_head_norm": f"{p}.transformer.shared_head.norm.weight",
        # Keep the ckpt's ``shared_head.output`` name — vLLM renames at
        # load time, we do not.
        "shared_head_output": f"{p}.transformer.shared_head.output.weight",
    })
    return keys


# =============================================================================
# Slicing helpers.
# =============================================================================
def _to_bf16(t: "torch.Tensor") -> "torch.Tensor":
    import torch  # noqa: PLC0415

    return t.to(torch.bfloat16) if t.dtype != torch.bfloat16 else t


def _to_fp32(t: "torch.Tensor") -> "torch.Tensor":
    import torch  # noqa: PLC0415

    return t.to(torch.float32) if t.dtype != torch.float32 else t


def _slice_q_proj(
    q_full: "torch.Tensor", rank: int, num_heads_local: int,
) -> "torch.Tensor":
    """HF q_proj is ``[num_heads * HEAD_DIM, HIDDEN]``; pypto wants
    ``[HIDDEN, num_heads_local * HEAD_DIM]`` per rank.
    """
    head_lo = rank * num_heads_local
    head_hi = head_lo + num_heads_local
    out = q_full[head_lo * HEAD_DIM:head_hi * HEAD_DIM, :]   # [Q_LOCAL, HIDDEN]
    return _to_bf16(out.transpose(0, 1).contiguous())        # [HIDDEN, Q_LOCAL]


def _slice_kv_proj(
    kv_full: "torch.Tensor", rank: int, kv_heads_local: int,
) -> "torch.Tensor":
    """HF k_proj/v_proj is ``[NUM_KV_HEADS * HEAD_DIM, HIDDEN]``; pypto wants
    ``[HIDDEN, kv_heads_local * HEAD_DIM]``.
    """
    head_lo = rank * kv_heads_local
    head_hi = head_lo + kv_heads_local
    out = kv_full[head_lo * HEAD_DIM:head_hi * HEAD_DIM, :]
    return _to_bf16(out.transpose(0, 1).contiguous())


def _slice_o_proj(
    o_full: "torch.Tensor", rank: int, num_heads_local: int,
) -> "torch.Tensor":
    """HF o_proj is ``[HIDDEN, num_heads * HEAD_DIM]``; pypto wants
    ``[num_heads_local * HEAD_DIM, HIDDEN]`` (per-rank column slice).
    """
    head_lo = rank * num_heads_local
    head_hi = head_lo + num_heads_local
    out = o_full[:, head_lo * HEAD_DIM:head_hi * HEAD_DIM]   # [HIDDEN, Q_LOCAL]
    return _to_bf16(out.transpose(0, 1).contiguous())        # [Q_LOCAL, HIDDEN]


def _slice_g_proj(
    g_full: "torch.Tensor", rank: int, num_heads_local: int, pad_to: int = 0,
) -> "torch.Tensor":
    """HF g_proj is ``[NUM_HEADS, HIDDEN]``; pypto wants
    ``[HIDDEN, pad_to]`` — head-axis slice + transpose + zero-pad.

    ``pad_to`` must be >= ``num_heads_local``.  The extra columns are filled
    with zeros so padded gate logits contribute nothing to the softmax gate.
    When ``pad_to == 0`` (legacy / unpadded callers) no padding is applied.
    """
    import torch  # noqa: PLC0415

    head_lo = rank * num_heads_local
    head_hi = head_lo + num_heads_local
    out = g_full[head_lo:head_hi, :]                          # [NH_LOCAL, HIDDEN]
    result = _to_bf16(out.transpose(0, 1).contiguous())       # [HIDDEN, NH_LOCAL]
    if pad_to > num_heads_local:
        # Zero-pad the last dimension (head axis) so the kernel tile width is met.
        pad_cols = pad_to - num_heads_local
        result = torch.cat(
            [result, torch.zeros(result.shape[0], pad_cols, dtype=result.dtype)],
            dim=1,
        )
    return result


def _slice_mlp_col(
    w_full: "torch.Tensor", rank: int, dim_local: int,
) -> "torch.Tensor":
    """HF gate_proj/up_proj is ``[INTERMEDIATE, HIDDEN]``; pypto wants
    ``[HIDDEN, INTERMEDIATE_LOCAL]`` — column slice + transpose.
    """
    lo = rank * dim_local
    hi = lo + dim_local
    out = w_full[lo:hi, :]                                     # [DIM_LOCAL, HIDDEN]
    return _to_bf16(out.transpose(0, 1).contiguous())          # [HIDDEN, DIM_LOCAL]


def _slice_mlp_row(
    w_full: "torch.Tensor", rank: int, dim_local: int,
) -> "torch.Tensor":
    """HF down_proj is ``[HIDDEN, INTERMEDIATE]``; pypto wants
    ``[INTERMEDIATE_LOCAL, HIDDEN]`` — row slice (column-slice of the
    HF input) + transpose.
    """
    lo = rank * dim_local
    hi = lo + dim_local
    out = w_full[:, lo:hi]                                     # [HIDDEN, DIM_LOCAL]
    return _to_bf16(out.transpose(0, 1).contiguous())          # [DIM_LOCAL, HIDDEN]


def _slice_lm_head(
    lm_full: "torch.Tensor", rank: int, vocab_local: int,
) -> "torch.Tensor":
    """HF lm_head is ``[VOCAB, HIDDEN]``; pypto wants
    ``[VOCAB_LOCAL, HIDDEN]`` — pure row slice (no transpose).
    """
    lo = rank * vocab_local
    hi = lo + vocab_local
    return _to_bf16(lm_full[lo:hi, :].contiguous())


def _slice_eh_proj(
    eh_full: "torch.Tensor", rank: int, hidden_local: int,
) -> "torch.Tensor":
    """HF eh_proj is ``[HIDDEN, 2 * HIDDEN]``; pypto wants
    ``[HIDDEN_LOCAL, 2 * HIDDEN]`` per card (row-slice on the output dim).
    """
    lo = rank * hidden_local
    hi = lo + hidden_local
    return _to_bf16(eh_full[lo:hi, :].contiguous())


def _transpose_routed_block(t: "torch.Tensor") -> "torch.Tensor":
    """Routed gate/up HF block is ``[E_local, MOE_INTERMEDIATE, HIDDEN]``;
    the kernel wants ``[E_local, HIDDEN, MOE_INTERMEDIATE]``. Routed-down
    is the reverse — HF ``[E_local, HIDDEN, MOE_INTERMEDIATE]`` -> kernel
    ``[E_local, MOE_INTERMEDIATE, HIDDEN]``.
    """
    return t.transpose(-2, -1).contiguous()


# =============================================================================
# Public entry — build a per-rank bundle from the on-disk checkpoint.
# =============================================================================
def load_step3p5_weights_for_rank(
    ckpt_dir: str,
    rank: int,
    tp_world_size: int = TP_WORLD_SIZE,
) -> dict[str, "torch.Tensor"]:
    """Construct rank ``rank``'s weight bundle from the HF safetensors ckpt.

    All TP-sliced tensors are sliced by ``rank`` along the appropriate
    axis; all EP-sliced expert weights are sliced by the contiguous expert
    block owned by this rank (``rank * MOE_NUM_EXPERTS_LOCAL..
    (rank + 1) * MOE_NUM_EXPERTS_LOCAL``). Replicated tensors are copied
    verbatim.

    Returns a flat dict of named ``torch.Tensor``s. Caller stacks them
    along a leading rank axis when constructing a full 8-rank decode
    invocation (see ``step3p5_decode.py``).
    """
    if not 0 <= rank < tp_world_size:
        raise ValueError(
            f"rank {rank} out of range [0, {tp_world_size})",
        )
    if tp_world_size != TP_WORLD_SIZE:
        log.warning(
            "tp_world_size=%d does not match config.TP_WORLD_SIZE=%d; "
            "slicing math uses the supplied value.",
            tp_world_size, TP_WORLD_SIZE,
        )

    import torch  # noqa: PLC0415

    num_heads_full_local = NUM_HEADS_FULL // tp_world_size
    num_heads_swa_local = NUM_HEADS_SWA // tp_world_size
    # Gate head-pad widths come from config; fixed at 16 for both full and SWA.
    num_heads_full_local_pad = NUM_HEADS_FULL_LOCAL_PAD
    num_heads_swa_local_pad = NUM_HEADS_SWA_LOCAL_PAD
    kv_heads_local = NUM_KV_HEADS // tp_world_size
    intermediate_local = INTERMEDIATE // tp_world_size
    share_expert_dim_local = SHARE_EXPERT_DIM // tp_world_size
    vocab_local = VOCAB // tp_world_size
    moe_num_experts_local = MOE_NUM_EXPERTS // tp_world_size
    hidden_local = HIDDEN // tp_world_size

    weight_map = _read_index(ckpt_dir)
    bundle: dict[str, torch.Tensor] = {}

    with _ShardCache(ckpt_dir, weight_map) as cache:
        # ── Replicated top-level: embed_tokens + final norm + LM head ──
        embed = cache.get("model.embed_tokens.weight")
        bundle[KEY_EMBED] = _to_bf16(embed.contiguous())

        bundle[KEY_FINAL_NORM] = _to_bf16(
            cache.get("model.norm.weight").contiguous(),
        )

        lm_head_full = cache.get("lm_head.weight")
        bundle[KEY_LM_HEAD] = _slice_lm_head(
            lm_head_full, rank, vocab_local,
        )

        # ── Per-layer attention / RMSNorm tables. ──────────────────────
        input_rms_rows: list[torch.Tensor] = []
        post_attn_rms_rows: list[torch.Tensor] = []
        q_norm_rows: list[torch.Tensor] = []
        k_norm_rows: list[torch.Tensor] = []

        wq_full_rows: list[torch.Tensor] = []
        wk_full_rows: list[torch.Tensor] = []
        wv_full_rows: list[torch.Tensor] = []
        wo_full_rows: list[torch.Tensor] = []
        wg_full_rows: list[torch.Tensor] = []

        wq_swa_rows: list[torch.Tensor] = []
        wk_swa_rows: list[torch.Tensor] = []
        wv_swa_rows: list[torch.Tensor] = []
        wo_swa_rows: list[torch.Tensor] = []
        wg_swa_rows: list[torch.Tensor] = []

        for li in range(NUM_HIDDEN_LAYERS):
            attn = _hf_attn_keys(li)
            input_rms_rows.append(_to_bf16(cache.get(attn["input_rms"])))
            post_attn_rms_rows.append(
                _to_bf16(cache.get(attn["post_attn_rms"])),
            )
            q_norm_rows.append(_to_bf16(cache.get(attn["q_norm"])))
            k_norm_rows.append(_to_bf16(cache.get(attn["k_norm"])))

            q_full = cache.get(attn["q_proj"])
            k_full = cache.get(attn["k_proj"])
            v_full = cache.get(attn["v_proj"])
            o_full = cache.get(attn["o_proj"])
            g_full = cache.get(attn["g_proj"])

            if is_full_attention(li):
                wq_full_rows.append(_slice_q_proj(
                    q_full, rank, num_heads_full_local,
                ))
                wk_full_rows.append(_slice_kv_proj(
                    k_full, rank, kv_heads_local,
                ))
                wv_full_rows.append(_slice_kv_proj(
                    v_full, rank, kv_heads_local,
                ))
                wo_full_rows.append(_slice_o_proj(
                    o_full, rank, num_heads_full_local,
                ))
                wg_full_rows.append(_slice_g_proj(
                    g_full, rank, num_heads_full_local,
                    pad_to=num_heads_full_local_pad,
                ))
            else:
                wq_swa_rows.append(_slice_q_proj(
                    q_full, rank, num_heads_swa_local,
                ))
                wk_swa_rows.append(_slice_kv_proj(
                    k_full, rank, kv_heads_local,
                ))
                wv_swa_rows.append(_slice_kv_proj(
                    v_full, rank, kv_heads_local,
                ))
                wo_swa_rows.append(_slice_o_proj(
                    o_full, rank, num_heads_swa_local,
                ))
                wg_swa_rows.append(_slice_g_proj(
                    g_full, rank, num_heads_swa_local,
                    pad_to=num_heads_swa_local_pad,
                ))

        bundle[KEY_INPUT_RMS] = torch.stack(input_rms_rows, dim=0)
        bundle[KEY_POST_ATTN_RMS] = torch.stack(post_attn_rms_rows, dim=0)
        bundle[KEY_Q_NORM] = torch.stack(q_norm_rows, dim=0)
        bundle[KEY_K_NORM] = torch.stack(k_norm_rows, dim=0)

        bundle[KEY_WQ_FULL] = torch.stack(wq_full_rows, dim=0)
        bundle[KEY_WK_FULL] = torch.stack(wk_full_rows, dim=0)
        bundle[KEY_WV_FULL] = torch.stack(wv_full_rows, dim=0)
        bundle[KEY_WO_FULL] = torch.stack(wo_full_rows, dim=0)
        bundle[KEY_WG_FULL] = torch.stack(wg_full_rows, dim=0)

        bundle[KEY_WQ_SWA] = torch.stack(wq_swa_rows, dim=0)
        bundle[KEY_WK_SWA] = torch.stack(wk_swa_rows, dim=0)
        bundle[KEY_WV_SWA] = torch.stack(wv_swa_rows, dim=0)
        bundle[KEY_WO_SWA] = torch.stack(wo_swa_rows, dim=0)
        bundle[KEY_WG_SWA] = torch.stack(wg_swa_rows, dim=0)

        # ── Dense MLP (layers 0..2). ────────────────────────────────────
        dense_gate_rows, dense_up_rows, dense_down_rows = [], [], []
        for li in range(NUM_HIDDEN_LAYERS):
            if is_moe_layer(li):
                continue
            mlp = _hf_dense_mlp_keys(li)
            dense_gate_rows.append(_slice_mlp_col(
                cache.get(mlp["gate_proj"]), rank, intermediate_local,
            ))
            dense_up_rows.append(_slice_mlp_col(
                cache.get(mlp["up_proj"]), rank, intermediate_local,
            ))
            dense_down_rows.append(_slice_mlp_row(
                cache.get(mlp["down_proj"]), rank, intermediate_local,
            ))
        bundle[KEY_DENSE_GATE] = torch.stack(dense_gate_rows, dim=0)
        bundle[KEY_DENSE_UP] = torch.stack(dense_up_rows, dim=0)
        bundle[KEY_DENSE_DOWN] = torch.stack(dense_down_rows, dim=0)

        # ── MoE layers (3..44): replicated gate_w + router_bias, EP-sliced
        #    routed experts, TP-sliced share expert. ─────────────────────
        gate_w_rows: list[torch.Tensor] = []
        router_bias_rows: list[torch.Tensor] = []
        routed_gate_rows: list[torch.Tensor] = []
        routed_up_rows: list[torch.Tensor] = []
        routed_down_rows: list[torch.Tensor] = []
        share_gate_rows: list[torch.Tensor] = []
        share_up_rows: list[torch.Tensor] = []
        share_down_rows: list[torch.Tensor] = []

        ep_lo = ep_global_expert_id(rank, 0)
        ep_hi = ep_lo + moe_num_experts_local

        for li in MOE_LAYER_INDICES:
            moe = _hf_moe_keys(li)
            # gate matmul is stored as ``[NUM_EXPERTS, HIDDEN]`` and is
            # used in FP32; we transpose to ``[HIDDEN, NUM_EXPERTS]`` to
            # match the gate kernel's signature.
            gate_w = _to_fp32(cache.get(moe["gate_w"]))
            if gate_w.shape[0] == MOE_NUM_EXPERTS and gate_w.shape[1] == HIDDEN:
                gate_w = gate_w.transpose(0, 1).contiguous()
            gate_w_rows.append(gate_w)
            router_bias_rows.append(_to_fp32(cache.get(moe["router_bias"])))

            # Routed experts: full HF block is [NUM_EXPERTS, *, *]; we
            # slice the per-rank expert window and transpose to kernel
            # orientation.
            gate_full = cache.get(moe["gate_proj"])    # [E, MOE_INTER, HIDDEN]
            up_full = cache.get(moe["up_proj"])
            down_full = cache.get(moe["down_proj"])    # [E, HIDDEN, MOE_INTER]

            gate_slab = gate_full[ep_lo:ep_hi].contiguous()
            up_slab = up_full[ep_lo:ep_hi].contiguous()
            down_slab = down_full[ep_lo:ep_hi].contiguous()

            routed_gate_rows.append(_to_bf16(_transpose_routed_block(gate_slab)))
            routed_up_rows.append(_to_bf16(_transpose_routed_block(up_slab)))
            routed_down_rows.append(_to_bf16(_transpose_routed_block(down_slab)))

            # Shared expert: TP-sliced like dense MLP.
            share_gate_rows.append(_slice_mlp_col(
                cache.get(moe["share_gate"]), rank, share_expert_dim_local,
            ))
            share_up_rows.append(_slice_mlp_col(
                cache.get(moe["share_up"]), rank, share_expert_dim_local,
            ))
            share_down_rows.append(_slice_mlp_row(
                cache.get(moe["share_down"]), rank, share_expert_dim_local,
            ))

        bundle[KEY_MOE_GATE_W] = torch.stack(gate_w_rows, dim=0)
        bundle[KEY_MOE_ROUTER_BIAS] = torch.stack(router_bias_rows, dim=0)
        bundle[KEY_MOE_W_GATE_R] = torch.stack(routed_gate_rows, dim=0)
        bundle[KEY_MOE_W_UP_R] = torch.stack(routed_up_rows, dim=0)
        bundle[KEY_MOE_W_DOWN_R] = torch.stack(routed_down_rows, dim=0)
        bundle[KEY_MOE_W_GATE_S] = torch.stack(share_gate_rows, dim=0)
        bundle[KEY_MOE_W_UP_S] = torch.stack(share_up_rows, dim=0)
        bundle[KEY_MOE_W_DOWN_S] = torch.stack(share_down_rows, dim=0)

        # ── MTP layers (45..47). ────────────────────────────────────────
        mtp_input_rms: list[torch.Tensor] = []
        mtp_post_attn_rms: list[torch.Tensor] = []
        mtp_q_norm: list[torch.Tensor] = []
        mtp_k_norm: list[torch.Tensor] = []
        mtp_wq: list[torch.Tensor] = []
        mtp_wk: list[torch.Tensor] = []
        mtp_wv: list[torch.Tensor] = []
        mtp_wo: list[torch.Tensor] = []
        mtp_wg: list[torch.Tensor] = []
        mtp_dense_gate: list[torch.Tensor] = []
        mtp_dense_up: list[torch.Tensor] = []
        mtp_dense_down: list[torch.Tensor] = []
        mtp_enorm: list[torch.Tensor] = []
        mtp_hnorm: list[torch.Tensor] = []
        mtp_eh: list[torch.Tensor] = []
        mtp_sh_norm: list[torch.Tensor] = []
        mtp_sh_out: list[torch.Tensor] = []

        for offset in range(NUM_MTP):
            li = NUM_HIDDEN_LAYERS + offset
            assert LAYER_TYPES[li] == LAYER_TYPE_SWA, (
                f"MTP layer {li} must be SWA per LAYER_TYPES"
            )

            mtp = _hf_mtp_keys(li)

            mtp_input_rms.append(_to_bf16(cache.get(mtp["input_rms"])))
            mtp_post_attn_rms.append(_to_bf16(cache.get(mtp["post_attn_rms"])))
            mtp_q_norm.append(_to_bf16(cache.get(mtp["q_norm"])))
            mtp_k_norm.append(_to_bf16(cache.get(mtp["k_norm"])))

            mtp_wq.append(_slice_q_proj(
                cache.get(mtp["q_proj"]), rank, num_heads_swa_local,
            ))
            mtp_wk.append(_slice_kv_proj(
                cache.get(mtp["k_proj"]), rank, kv_heads_local,
            ))
            mtp_wv.append(_slice_kv_proj(
                cache.get(mtp["v_proj"]), rank, kv_heads_local,
            ))
            mtp_wo.append(_slice_o_proj(
                cache.get(mtp["o_proj"]), rank, num_heads_swa_local,
            ))
            mtp_wg.append(_slice_g_proj(
                cache.get(mtp["g_proj"]), rank, num_heads_swa_local,
                pad_to=num_heads_swa_local_pad,
            ))

            mtp_dense_gate.append(_slice_mlp_col(
                cache.get(mtp["gate_proj"]), rank, intermediate_local,
            ))
            mtp_dense_up.append(_slice_mlp_col(
                cache.get(mtp["up_proj"]), rank, intermediate_local,
            ))
            mtp_dense_down.append(_slice_mlp_row(
                cache.get(mtp["down_proj"]), rank, intermediate_local,
            ))

            mtp_enorm.append(_to_bf16(cache.get(mtp["enorm"])))
            mtp_hnorm.append(_to_bf16(cache.get(mtp["hnorm"])))
            mtp_eh.append(_slice_eh_proj(
                cache.get(mtp["eh_proj"]), rank, hidden_local,
            ))
            mtp_sh_norm.append(_to_bf16(cache.get(mtp["shared_head_norm"])))
            mtp_sh_out.append(_slice_lm_head(
                cache.get(mtp["shared_head_output"]), rank, vocab_local,
            ))

        bundle[KEY_MTP_INPUT_RMS] = torch.stack(mtp_input_rms, dim=0)
        bundle[KEY_MTP_POST_ATTN_RMS] = torch.stack(mtp_post_attn_rms, dim=0)
        bundle[KEY_MTP_Q_NORM] = torch.stack(mtp_q_norm, dim=0)
        bundle[KEY_MTP_K_NORM] = torch.stack(mtp_k_norm, dim=0)
        bundle[KEY_MTP_WQ] = torch.stack(mtp_wq, dim=0)
        bundle[KEY_MTP_WK] = torch.stack(mtp_wk, dim=0)
        bundle[KEY_MTP_WV] = torch.stack(mtp_wv, dim=0)
        bundle[KEY_MTP_WO] = torch.stack(mtp_wo, dim=0)
        bundle[KEY_MTP_WG] = torch.stack(mtp_wg, dim=0)
        bundle[KEY_MTP_DENSE_GATE] = torch.stack(mtp_dense_gate, dim=0)
        bundle[KEY_MTP_DENSE_UP] = torch.stack(mtp_dense_up, dim=0)
        bundle[KEY_MTP_DENSE_DOWN] = torch.stack(mtp_dense_down, dim=0)
        bundle[KEY_MTP_ENORM] = torch.stack(mtp_enorm, dim=0)
        bundle[KEY_MTP_HNORM] = torch.stack(mtp_hnorm, dim=0)
        bundle[KEY_MTP_EH_PROJ] = torch.stack(mtp_eh, dim=0)
        bundle[KEY_MTP_SH_NORM] = torch.stack(mtp_sh_norm, dim=0)
        bundle[KEY_MTP_SH_OUT] = torch.stack(mtp_sh_out, dim=0)

    return bundle


# =============================================================================
# Bundle shape verification.
# =============================================================================
def verify_bundle_shapes(
    bundle: dict[str, "torch.Tensor"],
    tp_world_size: int = TP_WORLD_SIZE,
) -> None:
    """Assert every expected key is present with the right shape.

    Raises ``ValueError`` if a key is missing or a shape mismatches the
    table in ``expected_shapes``. Does not check dtypes (the loader
    promotes/demotes via ``_to_bf16`` / ``_to_fp32`` already).
    """
    expected = expected_shapes(tp_world_size)
    missing = sorted(set(expected) - set(bundle))
    extra = sorted(set(bundle) - set(expected))
    if missing:
        raise ValueError(
            f"bundle is missing {len(missing)} expected keys: "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}",
        )
    if extra:
        log.warning(
            "bundle has %d unexpected keys: %s",
            len(extra), extra[:5],
        )
    mismatches = []
    for key, want in expected.items():
        got = tuple(bundle[key].shape)
        if got != want:
            mismatches.append((key, got, want))
    if mismatches:
        msg = "; ".join(
            f"{k}: got {g} want {w}" for k, g, w in mismatches[:5]
        )
        more = f" (+{len(mismatches) - 5} more)" if len(mismatches) > 5 else ""
        raise ValueError(f"bundle shape mismatches: {msg}{more}")


# =============================================================================
# Optional: bundle a synthetic per-rank dictionary for smoke tests.
#
# Useful when the on-disk checkpoint is unreachable (CPU-only dev box).
# Returns a bundle whose shapes match ``expected_shapes`` (or a caller-
# supplied compact shape table) filled with small random values, suitable
# for exercising the dispatcher and shape-verification paths without
# touching the network share.
#
# WARNING: at production scale (tp_world_size=8), one bundle is roughly
# 72 GB of BF16 tensors — far above any sane host RAM. Pass a compact
# shape table via ``shape_overrides=`` to scale dimensions down for
# smoke tests on CPU-only dev boxes (see ``build_compact_shape_table``).
# =============================================================================
COMPACT_DEFAULTS: dict[str, int] = {
    # Caller-friendly axis-scale knobs for compact CPU-only smoke runs.
    # All scale-down values must divide their respective full constant.
    #
    # NOTE: layer counts (num_hidden_layers, num_full_layers,
    # num_swa_layers, num_dense_layers, num_moe_layers, num_mtp) are kept
    # at the production values — the dispatcher in ``decode_layer.py``
    # walks the compile-time ``LAYER_TYPES`` table from ``config.py``,
    # so the torch reference in ``step3p5_decode.py`` needs one bundle
    # row per real layer for ``is_full_attention`` / ``is_moe_layer`` to
    # line up with the indexed weight slabs.
    "hidden": 256,
    "intermediate": 128,
    "moe_intermediate": 64,
    "vocab": 512,
    "head_dim": 32,
    "num_heads_full_local": 2,
    "num_heads_swa_local": 2,
    # Gate head-pad values are fixed tile-alignment constants (cannot scale
    # below 16); kept at production value even in compact runs.
    "num_heads_full_local_pad": NUM_HEADS_FULL_LOCAL_PAD,
    "num_heads_swa_local_pad": NUM_HEADS_SWA_LOCAL_PAD,
    "kv_heads_local": 1,
    "moe_num_experts": 16,
    "moe_num_experts_local": 2,
    "num_hidden_layers": NUM_HIDDEN_LAYERS,
    "num_full_layers": NUM_FULL_LAYERS,
    "num_swa_layers": NUM_SWA_LAYERS,
    "num_dense_layers": NUM_DENSE_LAYERS,
    "num_moe_layers": NUM_MOE_LAYERS,
    "num_mtp": NUM_MTP,
}


def build_compact_shape_table(
    tp_world_size: int = TP_WORLD_SIZE,
    overrides: dict[str, int] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Construct a shape table with scaled-down axes for smoke testing.

    Mirrors the layout of ``expected_shapes`` but uses small dimensions so
    a full per-rank bundle fits in <100 MB. The default values keep TP/EP
    divisibility invariants (KV head count, etc.) so the per-rank slicing
    math still applies.
    """
    cfg = dict(COMPACT_DEFAULTS)
    if overrides:
        cfg.update(overrides)
    H = cfg["hidden"]
    I_LOCAL = cfg["intermediate"]
    MI = cfg["moe_intermediate"]
    V = cfg["vocab"]
    HD = cfg["head_dim"]
    NHFL = cfg["num_heads_full_local"]
    NHSL = cfg["num_heads_swa_local"]
    NHFL_PAD = cfg["num_heads_full_local_pad"]
    NHSL_PAD = cfg["num_heads_swa_local_pad"]
    KVL = cfg["kv_heads_local"]
    EXP = cfg["moe_num_experts"]
    EXPL = cfg["moe_num_experts_local"]
    L = cfg["num_hidden_layers"]
    LF = cfg["num_full_layers"]
    LS = cfg["num_swa_layers"]
    LD = cfg["num_dense_layers"]
    LM = cfg["num_moe_layers"]
    MTP = cfg["num_mtp"]

    HQF_L = NHFL * HD
    HQS_L = NHSL * HD
    KVH_L = KVL * HD
    H_LOCAL = H // tp_world_size
    V_LOCAL = V // tp_world_size

    return {
        KEY_EMBED: (V, H),
        KEY_FINAL_NORM: (H,),
        KEY_INPUT_RMS: (L, H),
        KEY_POST_ATTN_RMS: (L, H),
        KEY_Q_NORM: (L, HD),
        KEY_K_NORM: (L, HD),
        KEY_MOE_GATE_W: (LM, H, EXP),
        KEY_MOE_ROUTER_BIAS: (LM, EXP),
        KEY_WQ_FULL: (LF, H, HQF_L),
        KEY_WK_FULL: (LF, H, KVH_L),
        KEY_WV_FULL: (LF, H, KVH_L),
        KEY_WO_FULL: (LF, HQF_L, H),
        KEY_WG_FULL: (LF, H, NHFL_PAD),
        KEY_WQ_SWA: (LS, H, HQS_L),
        KEY_WK_SWA: (LS, H, KVH_L),
        KEY_WV_SWA: (LS, H, KVH_L),
        KEY_WO_SWA: (LS, HQS_L, H),
        KEY_WG_SWA: (LS, H, NHSL_PAD),
        KEY_DENSE_GATE: (LD, H, I_LOCAL),
        KEY_DENSE_UP: (LD, H, I_LOCAL),
        KEY_DENSE_DOWN: (LD, I_LOCAL, H),
        KEY_MOE_W_GATE_S: (LM, H, I_LOCAL),
        KEY_MOE_W_UP_S: (LM, H, I_LOCAL),
        KEY_MOE_W_DOWN_S: (LM, I_LOCAL, H),
        KEY_MOE_W_GATE_R: (LM, EXPL, H, MI),
        KEY_MOE_W_UP_R: (LM, EXPL, H, MI),
        KEY_MOE_W_DOWN_R: (LM, EXPL, MI, H),
        KEY_LM_HEAD: (V_LOCAL, H),
        KEY_MTP_ENORM: (MTP, H),
        KEY_MTP_HNORM: (MTP, H),
        KEY_MTP_EH_PROJ: (MTP, H_LOCAL, 2 * H),
        KEY_MTP_SH_NORM: (MTP, H),
        KEY_MTP_SH_OUT: (MTP, V_LOCAL, H),
        KEY_MTP_INPUT_RMS: (MTP, H),
        KEY_MTP_POST_ATTN_RMS: (MTP, H),
        KEY_MTP_Q_NORM: (MTP, HD),
        KEY_MTP_K_NORM: (MTP, HD),
        KEY_MTP_WQ: (MTP, H, HQS_L),
        KEY_MTP_WK: (MTP, H, KVH_L),
        KEY_MTP_WV: (MTP, H, KVH_L),
        KEY_MTP_WO: (MTP, HQS_L, H),
        KEY_MTP_WG: (MTP, H, NHSL_PAD),
        KEY_MTP_DENSE_GATE: (MTP, H, I_LOCAL),
        KEY_MTP_DENSE_UP: (MTP, H, I_LOCAL),
        KEY_MTP_DENSE_DOWN: (MTP, I_LOCAL, H),
    }


def build_synthetic_bundle(
    rank: int,
    tp_world_size: int = TP_WORLD_SIZE,
    seed: int = 0,
    shape_overrides: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, "torch.Tensor"]:
    """Build a per-rank bundle filled with deterministic random values.

    Shapes default to ``expected_shapes(tp_world_size)`` (production-size,
    ~72 GB per bundle); pass ``shape_overrides`` for compact smoke runs.
    Dtypes are BF16 for weight tensors and FP32 for the two router fields.
    """
    import torch  # noqa: PLC0415

    shapes = shape_overrides or expected_shapes(tp_world_size)
    gen = torch.Generator().manual_seed(seed * 8191 + rank * 17)
    bundle: dict[str, torch.Tensor] = {}
    fp32_keys = {KEY_MOE_GATE_W, KEY_MOE_ROUTER_BIAS}
    for key, shape in shapes.items():
        dtype = torch.float32 if key in fp32_keys else torch.bfloat16
        t = (torch.rand(*shape, generator=gen, dtype=torch.float32) - 0.5) * 0.1
        bundle[key] = t.to(dtype).contiguous()
    return bundle


__all__ = [
    "DEFAULT_CKPT_DIR",
    "NUM_FULL_LAYERS",
    "NUM_SWA_LAYERS",
    "NUM_MOE_LAYERS",
    "NUM_DENSE_LAYERS",
    "NUM_MTP",
    "MTP_HIDDEN_LOCAL",
    "EH_IN",
    # Bundle keys
    "KEY_EMBED",
    "KEY_FINAL_NORM",
    "KEY_LM_HEAD",
    "KEY_INPUT_RMS",
    "KEY_POST_ATTN_RMS",
    "KEY_Q_NORM",
    "KEY_K_NORM",
    "KEY_WQ_FULL",
    "KEY_WK_FULL",
    "KEY_WV_FULL",
    "KEY_WO_FULL",
    "KEY_WG_FULL",
    "KEY_WQ_SWA",
    "KEY_WK_SWA",
    "KEY_WV_SWA",
    "KEY_WO_SWA",
    "KEY_WG_SWA",
    "KEY_DENSE_GATE",
    "KEY_DENSE_UP",
    "KEY_DENSE_DOWN",
    "KEY_MOE_GATE_W",
    "KEY_MOE_ROUTER_BIAS",
    "KEY_MOE_W_GATE_R",
    "KEY_MOE_W_UP_R",
    "KEY_MOE_W_DOWN_R",
    "KEY_MOE_W_GATE_S",
    "KEY_MOE_W_UP_S",
    "KEY_MOE_W_DOWN_S",
    "KEY_MTP_ENORM",
    "KEY_MTP_HNORM",
    "KEY_MTP_EH_PROJ",
    "KEY_MTP_SH_NORM",
    "KEY_MTP_SH_OUT",
    "KEY_MTP_INPUT_RMS",
    "KEY_MTP_POST_ATTN_RMS",
    "KEY_MTP_Q_NORM",
    "KEY_MTP_K_NORM",
    "KEY_MTP_WQ",
    "KEY_MTP_WK",
    "KEY_MTP_WV",
    "KEY_MTP_WO",
    "KEY_MTP_WG",
    "KEY_MTP_DENSE_GATE",
    "KEY_MTP_DENSE_UP",
    "KEY_MTP_DENSE_DOWN",
    # Entry points
    "expected_shapes",
    "load_step3p5_weights_for_rank",
    "verify_bundle_shapes",
    "build_synthetic_bundle",
    "build_compact_shape_table",
    "COMPACT_DEFAULTS",
]
