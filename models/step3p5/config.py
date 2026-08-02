# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 model configuration.

Source of truth for the
``step3p5_flash_release_hf_mtp3_bf16`` checkpoint shipped under
``/mnt/chensiyu-jfs/multi-hardware/models/``.

Step3p5 has the following distinguishing features:
- 45 main layers + 3 MTP next-n-predict layers
- mixed full-attention (64 heads) and sliding-attention (96 heads, win=512)
- per-layer RoPE theta and partial rotary factor (0.5 / 1.0)
- llama3 yarn rope scaling on full-attention layers only
- zero-centered q/k RMSNorm (effective gamma = stored_gamma + 1.0)
- head-wise attention gate (g_proj per-head sigmoid)
- MoE on layers 3..44 (288 experts, top-8, sigmoid routing + router_bias,
  renormalize, 1280-dim shared expert); dense MLP on layers 0..2
- SwigluStep with limit=7 on the routed-MoE active path of two specific layers
  and limit=16 on layer 44 share-expert; plain SiLU elsewhere

See ``MIGRATION_PLAN.md`` for the multi-phase migration plan.
"""

from __future__ import annotations

import pypto.language as pl

# -----------------------------------------------------------------------------
# Dynamic dimensions used by the JIT/program signatures.
# -----------------------------------------------------------------------------
USER_BATCH_DYN = pl.dynamic("USER_BATCH_DYN")
KV_CACHE_ROWS_DYN = pl.dynamic("KV_CACHE_ROWS_DYN")
BLOCK_TABLE_FLAT_DYN = pl.dynamic("BLOCK_TABLE_FLAT_DYN")
ROPE_SEQ_DYN = pl.dynamic("ROPE_SEQ_DYN")
LAYER_DYN = pl.dynamic("LAYER_DYN")
LAYER_HIDDEN_ROWS_DYN = pl.dynamic("LAYER_HIDDEN_ROWS_DYN")
LAYER_INTER_ROWS_DYN = pl.dynamic("LAYER_INTER_ROWS_DYN")
LAYER_EXPERTS_DYN = pl.dynamic("LAYER_EXPERTS_DYN")  # n_layers * num_experts
LAYER_EXPERT_ROWS_DYN = pl.dynamic("LAYER_EXPERT_ROWS_DYN")
LAYER_SHARE_ROWS_DYN = pl.dynamic("LAYER_SHARE_ROWS_DYN")

# -----------------------------------------------------------------------------
# Top-level model shape (matches checkpoint config.json verbatim).
# -----------------------------------------------------------------------------
HIDDEN = 4096
INTERMEDIATE = 11264                  # dense MLP hidden
VOCAB = 128896
NUM_HIDDEN_LAYERS = 45
NUM_NEXTN_PREDICT_LAYERS = 3          # MTP layers (indices 45..47 in the ckpt)
NUM_TOTAL_LAYERS = NUM_HIDDEN_LAYERS + NUM_NEXTN_PREDICT_LAYERS

MAX_POSITION_EMBEDDINGS = 262144
MAX_SEQ_DEFAULT = 4096                # default for kernel-level golden harness
                                      # (the ckpt supports up to 262144)

# -----------------------------------------------------------------------------
# Attention shape — two variants, selected per-layer by ``LAYER_TYPES``.
# Both variants share the same kv-head count, kv hidden, and head dim.
# -----------------------------------------------------------------------------
HEAD_DIM = 128
NUM_KV_HEADS = 8                       # ``num_attention_groups``
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM    # 1024

# Full attention (``layer_type == "full_attention"``):
NUM_HEADS_FULL = 64                    # 64 heads * 128 = 8192 q hidden
HIDDEN_Q_FULL = NUM_HEADS_FULL * HEAD_DIM
Q_PER_KV_FULL = NUM_HEADS_FULL // NUM_KV_HEADS  # 8

# Sliding attention (``layer_type == "sliding_attention"``):
# attention_other_setting overrides the head count for SWA layers.
NUM_HEADS_SWA = 96                     # 96 heads * 128 = 12288 q hidden
HIDDEN_Q_SWA = NUM_HEADS_SWA * HEAD_DIM
Q_PER_KV_SWA = NUM_HEADS_SWA // NUM_KV_HEADS  # 12

SLIDING_WINDOW = 512                   # SWA context window (in tokens)

# -----------------------------------------------------------------------------
# MoE shape.
# -----------------------------------------------------------------------------
MOE_NUM_EXPERTS = 288
MOE_TOP_K = 8
MOE_INTERMEDIATE = 1280                # routed expert hidden
SHARE_EXPERT_DIM = 1280                # shared expert hidden
MOE_ROUTER_SCALING_FACTOR = 3.0
MOE_ROUTER_ACTIVATION = "sigmoid"      # sigmoid-gated routing with learned bias
NORM_EXPERT_WEIGHT = True              # renormalize top-k weights
USE_MOE_ROUTER_BIAS = True             # additive learned bias on router logits
NEED_FP32_GATE = True                  # gate matmul runs in FP32

# -----------------------------------------------------------------------------
# Step3p5-specific scalars.
# -----------------------------------------------------------------------------
ZERO_CENTERED_NORM = True              # RMSNorm: gamma_eff = stored_gamma + 1.0
USE_HEAD_WISE_ATTN_GATE = True         # per-head sigmoid gate (g_proj)
USE_QK_NORM = True                     # per-head q_norm / k_norm; treat as
                                       # always-on regardless of the
                                       # ``use_qk_norm`` json flag (vllm reads
                                       # the per-head norm weights anyway when
                                       # ``use_optimus_qknorm`` is true).

# -----------------------------------------------------------------------------
# Numeric constants.
# -----------------------------------------------------------------------------
EPS = 1e-6
HIDDEN_INV = 1.0 / HIDDEN
HEAD_DIM_INV = 1.0 / HEAD_DIM
ATTN_SCALE = 1.0 / (HEAD_DIM ** 0.5)
HALF_DIM = HEAD_DIM // 2

# Partial-rotary half-dim for the two layer types. The "rotary_dim" is
# ``head_dim * partial_rotary_factor``; the half is what gets split into
# (lo, hi) for the RoPE rotation, the rest of the head_dim is pass-through.
ROTARY_HALF_FULL = (HEAD_DIM // 2) // 2     # partial = 0.5 -> rotary_dim=64 -> half=32
ROTARY_HALF_SWA = HEAD_DIM // 2             # partial = 1.0 -> rotary_dim=128 -> half=64

# -----------------------------------------------------------------------------
# RoPE per-layer tables (extracted from the checkpoint config.json).
#
# Pattern repeats every 4 layers: [full, sliding, sliding, sliding].
# Full-attention layers use theta=5e6 with llama3 yarn rope scaling and
# partial_rotary_factor=0.5; sliding layers use theta=1e4, no scaling, and
# partial_rotary_factor=1.0. ``yarn_only_types=["full_attention"]`` means
# scaling is only applied on full-attention layers.
#
# These tables are 48-long (45 main + 3 MTP). The 4-cycle pattern
# [full, sliding, sliding, sliding] continues uninterrupted through the MTP
# layers: index 44 is full (cycle start), so 45/46/47 are sliding/sliding/
# sliding. This matches the ckpt's partial_rotary_factors[45..47] == 1.0,
# 1.0, 1.0 (verified 2026-06-03 against config.json on the
# step3p5_flash_release_hf_mtp3_bf16 checkpoint). The numeric LAYER_TYPES /
# LAYER_ROPE_THETA / LAYER_PARTIAL_ROTARY_FACTOR tables below evaluate to
# exactly that.
# -----------------------------------------------------------------------------
LAYER_TYPE_FULL = "full_attention"
LAYER_TYPE_SWA = "sliding_attention"

# fmt: off
LAYER_TYPES: tuple[str, ...] = (
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
    LAYER_TYPE_FULL, LAYER_TYPE_SWA, LAYER_TYPE_SWA, LAYER_TYPE_SWA,
)
assert len(LAYER_TYPES) == NUM_TOTAL_LAYERS, (
    f"LAYER_TYPES has {len(LAYER_TYPES)} entries, expected {NUM_TOTAL_LAYERS}"
)

# Per-layer RoPE theta. Full-attention layers use 5e6, sliding use 1e4.
LAYER_ROPE_THETA: tuple[float, ...] = tuple(
    5_000_000.0 if t == LAYER_TYPE_FULL else 10_000.0 for t in LAYER_TYPES
)

# Per-layer partial rotary factor. Full-attention layers use 0.5, sliding 1.0.
LAYER_PARTIAL_ROTARY_FACTOR: tuple[float, ...] = tuple(
    0.5 if t == LAYER_TYPE_FULL else 1.0 for t in LAYER_TYPES
)
# fmt: on

# yarn rope scaling parameters (only applied on full-attention layers per
# ``yarn_only_types=["full_attention"]``).
ROPE_SCALING = {
    "rope_type": "llama3",
    "factor": 2.0,
    "original_max_position_embeddings": 131072,
    "low_freq_factor": 1.0,
    "high_freq_factor": 32.0,
}
YARN_ONLY_TYPES = (LAYER_TYPE_FULL,)

# -----------------------------------------------------------------------------
# MoE / dense MLP layer membership.
# -----------------------------------------------------------------------------
# moe_layers_enum = "3,4,...,44" -- layers 0,1,2 are dense MLP (the "use_mfa"
# / "moe_layer_offset" knobs are not used by step3p5 once moe_layers_enum is
# present, see vllm modeling_step3p5).
MOE_LAYER_INDICES: tuple[int, ...] = tuple(range(3, NUM_HIDDEN_LAYERS))  # 3..44
DENSE_LAYER_INDICES: tuple[int, ...] = tuple(
    i for i in range(NUM_HIDDEN_LAYERS) if i not in MOE_LAYER_INDICES
)  # (0, 1, 2)

# -----------------------------------------------------------------------------
# SwigluStep tables (per-layer activation limit). 0.0 means plain SiLU.
# Only the routed-MoE active path uses ``swiglu_limits``; the share/dense MLP
# uses ``swiglu_limits_shared``. Lengths cover the 48 total layers.
# -----------------------------------------------------------------------------
# fmt: off
SWIGLU_LIMITS: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0, 0.0,
)
assert len(SWIGLU_LIMITS) == NUM_TOTAL_LAYERS

SWIGLU_LIMITS_SHARED: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 16.0, 0.0, 0.0, 0.0,
)
assert len(SWIGLU_LIMITS_SHARED) == NUM_TOTAL_LAYERS
# fmt: on


# -----------------------------------------------------------------------------
# Helpers for per-layer compile-time selection.
# -----------------------------------------------------------------------------
def is_full_attention(layer_idx: int) -> bool:
    return LAYER_TYPES[layer_idx] == LAYER_TYPE_FULL


def is_moe_layer(layer_idx: int) -> bool:
    return layer_idx in MOE_LAYER_INDICES


def num_heads_for_layer(layer_idx: int) -> int:
    return NUM_HEADS_FULL if is_full_attention(layer_idx) else NUM_HEADS_SWA


def hidden_q_for_layer(layer_idx: int) -> int:
    return HIDDEN_Q_FULL if is_full_attention(layer_idx) else HIDDEN_Q_SWA


def rotary_half_for_layer(layer_idx: int) -> int:
    """Half-dim of the rotary slice for partial RoPE.

    Full-attention layers rotate the leading 0.5 * head_dim lanes, so the
    sin/cos pair operates on (head_dim*0.5)/2 = 32 lanes.
    Sliding layers rotate the full head_dim, so the half is 64.
    """
    return ROTARY_HALF_FULL if is_full_attention(layer_idx) else ROTARY_HALF_SWA


# -----------------------------------------------------------------------------
# Tiling defaults shared across all step3p5 kernels.
# Per-kernel kernels may override locally; these are the safe starting points.
# -----------------------------------------------------------------------------
BATCH = 16                              # default kernel-level batch
BATCH_TILE = 16
BLOCK_SIZE = 128                        # paged-cache block (also K/V SEQ_TILE)
SEQ_TILE = 128

# Scope 1 tiling (input proj).
INPUT_PROJ_K_CHUNK = 256
KV_PROJ_K_CHUNK = INPUT_PROJ_K_CHUNK // 2  # 128 — keeps K/V L0B within 512 KB at TP=1
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 256

# Scope 3 tiling (output proj + MLP / MoE).
K_CHUNK = 256
OUT_PROJ_K_CHUNK = 256
OUT_PROJ_N_CHUNK = 256
# MLP_OUT_CHUNK must divide BOTH the world-level INTERMEDIATE (11264, used
# by the historical single-card drafts) AND the per-card TP-sliced
# INTERMEDIATE_LOCAL=1408 (used by decode_layer's _dense_mlp_body_tp).
# gcd(11264, 1408) = 1408 with many divisors; 128 is the largest power
# of 2 that divides 1408 (1408 = 128 * 11) while still aligning with
# the cube's 128B / 16-row friendly tiling.
MLP_OUT_CHUNK = 128
MLP_SPMD_INNER = 2
MLP_GROUP_CHUNK = MLP_SPMD_INNER * MLP_OUT_CHUNK
DOWN_MLP_CHUNK = 256
DOWN_OUT_CHUNK = 256
FINAL_RMS_K_CHUNK = 128
LM_HEAD_K_CHUNK = 128
# VOCAB_CHUNK must divide BOTH the world-level VOCAB (128896, used by the
# historical single-card drafts) AND the per-card TP-sliced
# VOCAB_LOCAL = 128896 // 8 = 16112 (used by rms_lm_head's vocab-sliced
# matmul). 16112 = 16 * 19 * 53, so the largest power-of-2 divisor is 16,
# matching the cube's 32B BF16 row alignment. 16 also divides 128896 cleanly
# (128896 = 16 * 8056).
VOCAB_CHUNK = 16

# fa_fused decode tiling for the full-attention path. Step3p5 pairs Q heads in
# batches of (Q_PER_KV) per KV head: q_per_kv = 8 for full-attention layers and
# 12 for sliding layers — both clean factors of the kv-head count and >= the
# cube's minimum row count, so the fa_fused pattern fits without re-tuning.
# Q_HEAD_BATCH = q_per_kv keeps one Q-group per KV head.
Q_HEAD_BATCH_FULL = Q_PER_KV_FULL       # 8
Q_HEAD_BATCH_SWA = Q_PER_KV_SWA         # 12
# Q_HEAD_PAD: padded Q row count fa_fused operates on; needs to be a multiple
# of 4 with Q_HEAD_PAD//2 >= Q_HEAD_BATCH.
Q_HEAD_PAD_FULL = 16                    # 16 % 4 == 0, 16/2 == 8 >= 8 (full)
Q_HEAD_PAD_SWA = 24                     # 24 % 4 == 0, 24/2 == 12 >= 12 (swa)

MAX_BLOCKS_PER_SEQ = (MAX_SEQ_DEFAULT + BLOCK_SIZE - 1) // BLOCK_SIZE


# -----------------------------------------------------------------------------
# Distributed topology (Phase 9 — single-node, 8 cards, one process per card).
#
# Step3p5 inference deployment is a single 8-card node. The TP and EP groups
# are co-located on the same world (typical inference layout — one process
# per card, world_size = TP_WORLD_SIZE = EP_WORLD_SIZE).
#
# Tensor-Parallel (TP) sharding axes:
#   - Attention Q/K/V/O      sliced by HEAD count
#                            (NUM_HEADS_FULL / NUM_HEADS_SWA / NUM_KV_HEADS)
#   - lm_head                sliced by VOCAB
#   - shared_expert / dense  sliced by INTERMEDIATE / SHARE_EXPERT_DIM
#     MLP
#
# Expert-Parallel (EP) sharding:
#   - Routed experts         288 experts evenly partitioned, 36 per card
#   - Token routing          all_to_all dispatch + combine
#                            (per-token send_counts)
#
# IMPORTANT for downstream code in this package:
#   The single-card constants HIDDEN_Q_FULL, HIDDEN_Q_SWA, KV_HIDDEN,
#   INTERMEDIATE, SHARE_EXPERT_DIM, VOCAB, MOE_NUM_EXPERTS defined above are
#   now WORLD-LEVEL totals. Per-card kernel shapes must use the *_LOCAL
#   forms below (Phase 9 Wave 2 refactor). The world-level totals stay
#   defined for host-side weight-loading and golden references.
# -----------------------------------------------------------------------------
TP_WORLD_SIZE = 8
EP_WORLD_SIZE = 8

# Per-card attention head / feature counts (sliced by TP_WORLD_SIZE).
NUM_HEADS_FULL_LOCAL = NUM_HEADS_FULL // TP_WORLD_SIZE   # 64 // 8 == 8
NUM_HEADS_SWA_LOCAL = NUM_HEADS_SWA // TP_WORLD_SIZE     # 96 // 8 == 12
KV_HEADS_LOCAL = NUM_KV_HEADS // TP_WORLD_SIZE           # 8  // 8 == 1

HIDDEN_Q_FULL_LOCAL = NUM_HEADS_FULL_LOCAL * HEAD_DIM    # 8  * 128 == 1024
HIDDEN_Q_SWA_LOCAL = NUM_HEADS_SWA_LOCAL * HEAD_DIM      # 12 * 128 == 1536
KV_HIDDEN_LOCAL = KV_HEADS_LOCAL * HEAD_DIM              # 1  * 128 ==  128

# Adaptive K-chunk for K/V projection: at TP=1 KV_HIDDEN_LOCAL (1024) exceeds
# INPUT_PROJ_K_CHUNK (256), which would make L0B = 1024*256*2 = 512 KB — at the
# limit. Use the halved KV_PROJ_K_CHUNK (128) in that case.
# At TP=8 KV_HIDDEN_LOCAL=128 ≤ 256 → falls back to INPUT_PROJ_K_CHUNK=256.
if KV_HIDDEN_LOCAL > INPUT_PROJ_K_CHUNK:
    KV_PROJ_K_CHUNK_LOCAL = KV_PROJ_K_CHUNK      # 128 — TP=1 / large KV_HIDDEN_LOCAL
else:
    KV_PROJ_K_CHUNK_LOCAL = INPUT_PROJ_K_CHUNK   # 256 — TP=8 / normal KV_HIDDEN_LOCAL

# PTOAS A2/A3 cube unit requires bf16 matmul N (output cols) to be a
# multiple of 16. NUM_HEADS_*_LOCAL after TP=8 sharding falls below that
# threshold (8 / 12), so the per-head gate weight ``w_g`` is padded out
# to NUM_HEADS_*_LOCAL_PAD on its column axis. The host weight loader
# zero-pads the upper columns; downstream consumers only index the first
# NUM_HEADS_*_LOCAL columns so the pad is read-only ignored.
NUM_HEADS_FULL_LOCAL_PAD = 16   # ceil(8  / 16) * 16 == 16
NUM_HEADS_SWA_LOCAL_PAD = 16    # ceil(12 / 16) * 16 == 16

# Per-card MLP / shared-expert / lm_head dims (sliced by TP_WORLD_SIZE).
INTERMEDIATE_LOCAL = INTERMEDIATE // TP_WORLD_SIZE       # 11264 // 8 == 1408
SHARE_EXPERT_DIM_LOCAL = SHARE_EXPERT_DIM // TP_WORLD_SIZE  # 1280 // 8 == 160
VOCAB_LOCAL = VOCAB // TP_WORLD_SIZE                     # 128896 // 8 == 16112

# Per-card routed-expert count (sliced by EP_WORLD_SIZE).
MOE_NUM_EXPERTS_LOCAL = MOE_NUM_EXPERTS // EP_WORLD_SIZE  # 288 // 8 == 36

# Note on Q_HEAD_BATCH / Q_HEAD_PAD:
#   Q_HEAD_BATCH_FULL/SWA == Q_PER_KV_FULL/SWA, and Q_PER_KV is invariant
#   under TP (numerator and denominator are sliced by the same factor:
#   NUM_HEADS_FULL/NUM_KV_HEADS == NUM_HEADS_FULL_LOCAL/KV_HEADS_LOCAL).
#   With TP=8 each rank sees KV_HEADS_LOCAL=1 KV bucket × Q_PER_KV Q-rows,
#   and the fa_fused tile constraints stay the same. The Q_HEAD_PAD values
#   (16 for full, 24 for SWA) likewise stay valid.

# Sanity: every TP / EP sliced dim must divide cleanly.
assert NUM_HEADS_FULL % TP_WORLD_SIZE == 0, (
    f"NUM_HEADS_FULL={NUM_HEADS_FULL} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert NUM_HEADS_SWA % TP_WORLD_SIZE == 0, (
    f"NUM_HEADS_SWA={NUM_HEADS_SWA} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert NUM_KV_HEADS % TP_WORLD_SIZE == 0, (
    f"NUM_KV_HEADS={NUM_KV_HEADS} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert INTERMEDIATE % TP_WORLD_SIZE == 0, (
    f"INTERMEDIATE={INTERMEDIATE} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert SHARE_EXPERT_DIM % TP_WORLD_SIZE == 0, (
    f"SHARE_EXPERT_DIM={SHARE_EXPERT_DIM} must be a multiple of "
    f"TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert VOCAB % TP_WORLD_SIZE == 0, (
    f"VOCAB={VOCAB} must be a multiple of TP_WORLD_SIZE={TP_WORLD_SIZE}"
)
assert MOE_NUM_EXPERTS % EP_WORLD_SIZE == 0, (
    f"MOE_NUM_EXPERTS={MOE_NUM_EXPERTS} must be a multiple of "
    f"EP_WORLD_SIZE={EP_WORLD_SIZE}"
)
# The Q_PER_KV invariant.
assert NUM_HEADS_FULL_LOCAL // KV_HEADS_LOCAL == Q_PER_KV_FULL
assert NUM_HEADS_SWA_LOCAL // KV_HEADS_LOCAL == Q_PER_KV_SWA


def ep_expert_owner(expert_id: int) -> int:
    """Return the EP rank that hosts the given global routed-expert id.

    Experts ``0..MOE_NUM_EXPERTS_LOCAL-1`` belong to rank 0,
    ``MOE_NUM_EXPERTS_LOCAL..2*MOE_NUM_EXPERTS_LOCAL-1`` to rank 1, and so
    on. Mirrors vllm's contiguous block-cyclic expert sharding.
    """
    if not 0 <= expert_id < MOE_NUM_EXPERTS:
        raise ValueError(
            f"expert_id {expert_id} out of range [0, {MOE_NUM_EXPERTS})"
        )
    return expert_id // MOE_NUM_EXPERTS_LOCAL


def ep_local_expert_id(expert_id: int) -> int:
    """Return the local index within the owning card for a global expert id.

    The owner rank is :func:`ep_expert_owner`; the local index is the
    in-shard slot ``[0, MOE_NUM_EXPERTS_LOCAL)``.
    """
    if not 0 <= expert_id < MOE_NUM_EXPERTS:
        raise ValueError(
            f"expert_id {expert_id} out of range [0, {MOE_NUM_EXPERTS})"
        )
    return expert_id % MOE_NUM_EXPERTS_LOCAL


def ep_global_expert_id(rank: int, local_id: int) -> int:
    """Inverse of (:func:`ep_expert_owner`, :func:`ep_local_expert_id`)."""
    if not 0 <= rank < EP_WORLD_SIZE:
        raise ValueError(f"rank {rank} out of range [0, {EP_WORLD_SIZE})")
    if not 0 <= local_id < MOE_NUM_EXPERTS_LOCAL:
        raise ValueError(
            f"local_id {local_id} out of range [0, {MOE_NUM_EXPERTS_LOCAL})"
        )
    return rank * MOE_NUM_EXPERTS_LOCAL + local_id


__all__ = [
    # dynamic dims
    "USER_BATCH_DYN",
    "KV_CACHE_ROWS_DYN",
    "BLOCK_TABLE_FLAT_DYN",
    "ROPE_SEQ_DYN",
    "LAYER_DYN",
    "LAYER_HIDDEN_ROWS_DYN",
    "LAYER_INTER_ROWS_DYN",
    "LAYER_EXPERTS_DYN",
    "LAYER_EXPERT_ROWS_DYN",
    "LAYER_SHARE_ROWS_DYN",
    # top-level shape
    "HIDDEN",
    "INTERMEDIATE",
    "VOCAB",
    "NUM_HIDDEN_LAYERS",
    "NUM_NEXTN_PREDICT_LAYERS",
    "NUM_TOTAL_LAYERS",
    "MAX_POSITION_EMBEDDINGS",
    "MAX_SEQ_DEFAULT",
    # attention
    "HEAD_DIM",
    "NUM_KV_HEADS",
    "KV_HIDDEN",
    "NUM_HEADS_FULL",
    "HIDDEN_Q_FULL",
    "Q_PER_KV_FULL",
    "NUM_HEADS_SWA",
    "HIDDEN_Q_SWA",
    "Q_PER_KV_SWA",
    "SLIDING_WINDOW",
    "HALF_DIM",
    "ROTARY_HALF_FULL",
    "ROTARY_HALF_SWA",
    "ATTN_SCALE",
    # moe
    "MOE_NUM_EXPERTS",
    "MOE_TOP_K",
    "MOE_INTERMEDIATE",
    "SHARE_EXPERT_DIM",
    "MOE_ROUTER_SCALING_FACTOR",
    "MOE_ROUTER_ACTIVATION",
    "NORM_EXPERT_WEIGHT",
    "USE_MOE_ROUTER_BIAS",
    "NEED_FP32_GATE",
    # step3p5 flags
    "ZERO_CENTERED_NORM",
    "USE_HEAD_WISE_ATTN_GATE",
    "USE_QK_NORM",
    # numeric
    "EPS",
    "HIDDEN_INV",
    "HEAD_DIM_INV",
    # per-layer tables
    "LAYER_TYPE_FULL",
    "LAYER_TYPE_SWA",
    "LAYER_TYPES",
    "LAYER_ROPE_THETA",
    "LAYER_PARTIAL_ROTARY_FACTOR",
    "ROPE_SCALING",
    "YARN_ONLY_TYPES",
    "MOE_LAYER_INDICES",
    "DENSE_LAYER_INDICES",
    "SWIGLU_LIMITS",
    "SWIGLU_LIMITS_SHARED",
    # helpers
    "is_full_attention",
    "is_moe_layer",
    "num_heads_for_layer",
    "hidden_q_for_layer",
    "rotary_half_for_layer",
    # tiling
    "BATCH",
    "BATCH_TILE",
    "BLOCK_SIZE",
    "SEQ_TILE",
    "INPUT_PROJ_K_CHUNK",
    "KV_PROJ_K_CHUNK",
    "KV_PROJ_K_CHUNK_LOCAL",
    "Q_OUT_CHUNK",
    "KV_OUT_CHUNK",
    "K_CHUNK",
    "OUT_PROJ_K_CHUNK",
    "OUT_PROJ_N_CHUNK",
    "MLP_OUT_CHUNK",
    "MLP_SPMD_INNER",
    "MLP_GROUP_CHUNK",
    "DOWN_MLP_CHUNK",
    "DOWN_OUT_CHUNK",
    "FINAL_RMS_K_CHUNK",
    "LM_HEAD_K_CHUNK",
    "VOCAB_CHUNK",
    "Q_HEAD_BATCH_FULL",
    "Q_HEAD_BATCH_SWA",
    "Q_HEAD_PAD_FULL",
    "Q_HEAD_PAD_SWA",
    "MAX_BLOCKS_PER_SEQ",
    # distributed topology
    "TP_WORLD_SIZE",
    "EP_WORLD_SIZE",
    "NUM_HEADS_FULL_LOCAL",
    "NUM_HEADS_SWA_LOCAL",
    "NUM_HEADS_FULL_LOCAL_PAD",
    "NUM_HEADS_SWA_LOCAL_PAD",
    "KV_HEADS_LOCAL",
    "HIDDEN_Q_FULL_LOCAL",
    "HIDDEN_Q_SWA_LOCAL",
    "KV_HIDDEN_LOCAL",
    "INTERMEDIATE_LOCAL",
    "SHARE_EXPERT_DIM_LOCAL",
    "VOCAB_LOCAL",
    "MOE_NUM_EXPERTS_LOCAL",
    "ep_expert_owner",
    "ep_local_expert_id",
    "ep_global_expert_id",
]
