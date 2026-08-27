# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: devices=2
"""DeepSeek-V4 context-parallel prefill multi-layer forward.

Phase 6E implements a generic ``SWA, SWA, repeated CSA/HCA pairs, optional
final CSA`` schedule selected at import time by ``--num-layers`` (choices
6, 8, 10, 43; default 8)::

    rank-local x_hc
      -> SWA layer 0 attention  (tail ep 0)
      -> four MoE waves          (epochs 1..4,  stages 1..5)  -> hidden_a
      -> SWA layer 1 attention   (tail ep 1)
      -> four MoE waves          (epochs 5..8,  stages 6..10) -> hidden_b
      -> for pair_i in range(PAIR_COUNT):           # (num_layers - 2) // 2
            CSA layer (2 + 2*pair_i) attention  (tail ep = layer,
                                                 CSA compact ep = pair_i)
            -> four MoE waves  -> hidden_a (ping-pong)
            HCA layer (3 + 2*pair_i) attention  (tail ep = layer,
                                                 HCA compact ep = pair_i)
            -> four MoE waves  -> hidden_b (ping-pong)
      -> [odd num_layers] final CSA layer attention  (tail ep = layer,
            CSA compact ep = CSA_NUM_LAYERS-1) -> four MoE waves -> hidden_a
      -> clear MoE signals once + SPMD publish last hidden -> pre_hc_hidden_out
      -> inlined hc_head + final rms_norm -> hidden_out (BF16)
      -> host-launched lm_head_test over hidden_out (selected rows -> logits)

43 is the production schedule (20 pairs + final CSA L42); 6/8/10 are even
diagnostic modes whose schedule ends at the last HCA pair (no final CSA;
publish reads hidden_b). L0/L1 stay explicit SWA stages; the remaining layers
are one ``pl.range(PAIR_COUNT)`` CSA/HCA pair loop plus a static (import-time)
final-CSA branch when ``HAS_FINAL_CSA``. ``pl.unroll`` and Python layer
expansion are forbidden. The loop reuses the outer ``hidden_a``/``hidden_b``
ping-pong roots plus one CSA attention output, one HCA attention output, and
one anchor per kind (pairs are serialized by coarse scopes; each producer
fully overwrites its output before the next consumer).

The forward calls ``prefill_cp_swa_core``, ``prefill_cp_csa_core``,
``prefill_cp_hca_core`` and ``moe`` directly (never the single-layer
``@pl.jit`` children). One communication bank per domain is reused across
layers, protected by monotonically increasing epochs; payload_epoch stays
0 (EPOCHS == 1). HCA and CSA compact signal banks are distinct from
tail/stage/MoE.
"""

import sys
import torch
import pypto.language as pl
import pypto.language.distributed as pld
from pypto.ir.distributed_compiled_program import DistributedConfig


def _parse_static_int(name: str, default: int) -> int:
    """Read ``--<name> VALUE`` or ``--<name>=VALUE`` from ``sys.argv`` at import.

    Mirrors the pattern in prefill_cp_zigzag: argparse is not yet available
    when module-level constants freeze, so the flag is scanned manually. Both
    whitespace-separated and ``=``-joined forms are accepted.
    """
    flag = f"--{name}"
    for i, token in enumerate(sys.argv):
        if token == flag and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if token.startswith(f"{flag}="):
            return int(token.split("=", 1)[1])
    return default


def _parse_static_bool(name: str, default: bool = False) -> bool:
    """Read a bare ``--<name>`` flag (no value) from ``sys.argv`` at import.

    Like :func:`_parse_static_int` but for boolean store-true flags; the token
    may appear bare (``--name``) or as ``--name=1``/``--name=0``. Used for
    diagnostic plumbing that must be visible at JIT-graph-build time (the
    ``--fwd-only`` guard omits the LM-head child launch from the compiled host
    graph), so argparse (which runs after import) would be too late.
    """
    flag = f"--{name}"
    for token in sys.argv:
        if token == flag:
            return True
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1] not in ("0", "", "false", "False")
    return default

# The prefill path routes PREFILL_TOKENS tokens. Set MOE_TOKENS before importing
# moe (which freezes recv shapes and derives RECV_MAX = EP * MOE_TOKENS at import).
import config
config.MOE_TOKENS = config.PREFILL_TOKENS

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
    moe,
)
from prefill_cp_swa_draft import (
    BLOCK_ROWS,
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
    prefill_cp_swa_core,
)
# HCA / CSA inline cores and their type-specific constants. The FWD child
# calls the cores directly (never @pl.jit children); the constants are used
# only for static child-side shape annotations and typed pl.slice offsets.
from prefill_cp_hca_draft import (
    COMPRESS_RATIO as HCA_COMPRESS_RATIO,
    COMPRESS_STATE_DIM as HCA_COMPRESS_STATE_DIM,
    HCA_STATE_BLOCK_SIZE,
    HCA_STATE_MAX_BLOCKS,
    HCA_STATE_PHYSICAL_BLOCKS,
    IDX_TOPK,
    build_tensor_specs as build_hca_tensor_specs,
    prefill_cp_hca_core,
)
from prefill_cp_csa_draft import (
    COMPRESS_RATIO as CSA_COMPRESS_RATIO,
    IDX_HEAD_DIM,
    IDX_N_HEADS,
    INNER_OUT_DIM as CSA_INNER_OUT_DIM,
    INNER_STATE_BLOCK_SIZE as CSA_INNER_STATE_BLOCK_SIZE,
    INNER_STATE_DIM as CSA_INNER_STATE_DIM,
    INNER_STATE_MAX_BLOCKS as CSA_INNER_STATE_MAX_BLOCKS,
    LOCAL_LEAVES as CSA_LOCAL_LEAVES,
    MAIN_OUT_DIM as CSA_MAIN_OUT_DIM,
    MAIN_STATE_BLOCK_SIZE as CSA_MAIN_STATE_BLOCK_SIZE,
    MAIN_STATE_DIM as CSA_MAIN_STATE_DIM,
    MAIN_STATE_MAX_BLOCKS as CSA_MAIN_STATE_MAX_BLOCKS,
    MAX_COMPRESS_LEAVES as CSA_MAX_COMPRESS_LEAVES,
    build_tensor_specs as build_csa_tensor_specs,
    prefill_cp_csa_core,
)
from config import (
    BLOCK_SIZE,
    CSA_INNER_STATE_PHYSICAL_BLOCKS,
    CSA_STATE_PHYSICAL_BLOCKS,
    IDX_CACHE_MAX_BLOCKS,
    PREFILL_CMP_BLOCK_NUM,
    PREFILL_CMP_MAX_BLOCKS,
    PREFILL_IDX_BLOCK_NUM,
)
from prefill_cp_exchange import (
    CMP_META_DIM,
    CMP_WINDOW_ROWS,
    META_DIM,
    RECORDS_PER_WINDOW,
    SCALE_TILE_COLS,
    STATE_META_DIM,
    STATE_RECORDS_PER_WINDOW,
    STATE_WINDOW_ROWS,
)
from golden import TensorSpec, run_jit
# Phase 3 final tail: HC head + final RMSNorm (inlined in the FWD child) and
# the LM head (host-launched per rank). All are accepted leaf math; only the
# composition is added here.
from hc_head import hc_head
from rmsnorm import rms_norm
from lm_head import (
    DONE_VALUE as LM_HEAD_DONE_VALUE,
    GROUP_LOGIT_ROWS as LM_HEAD_GROUP_LOGIT_ROWS,
    MAX_LOGIT_ROWS as LM_HEAD_MAX_LOGIT_ROWS,
    TP_SIZE as LM_HEAD_TP_SIZE,
    VOCAB as LM_HEAD_VOCAB,
    VOCAB_PER_TP as LM_HEAD_VOCAB_PER_TP,
    lm_head_test,
)

# ---------------------------------------------------------------------------
# Static CP/EP contract
# ---------------------------------------------------------------------------
assert CP_SIZE in CP_CHOICES, f"--cp must be one of {CP_CHOICES} (got {CP_SIZE})"
assert CP_SIZE == N_RANKS, (
    f"CP FWD requires CP == EP == pld.world_size() (got CP={CP_SIZE}, EP={N_RANKS})"
)
assert T == 128, f"CP FWD requires T == 128 (got {T})"
assert LOCAL_PARTS == 2
assert MAX_SEGMENT_TILES == 2
# Phase 6E schedule: SWA L0 -> SWA L1 -> [CSA, HCA] * PAIR_COUNT pairs, then
# an optional final CSA layer when the total layer count is odd. The
# TensorSpec stack sizes and JIT annotations depend on the layer count, so
# ``--num-layers`` must be parsed at import time before those definitions are
# frozen (same pattern as ``--cp`` in prefill_cp_zigzag). ``pl.range`` drives
# the pair loop; ``pl.unroll`` and Python layer expansion are forbidden.
# 43 is the production schedule (20 pairs + final CSA L42); 6/8/10 are even
# diagnostic modes whose schedule ends at an HCA layer (no final CSA). The
# default stays 8 during 6E staging; the final cleanup phase flips it to 43.
FWD_LAYER_CHOICES = (6, 8, 10, 43)
FWD_DEFAULT_LAYERS = 8
FWD_NUM_LAYERS = _parse_static_int("num-layers", FWD_DEFAULT_LAYERS)
if FWD_NUM_LAYERS not in FWD_LAYER_CHOICES:
    raise SystemExit(
        f"--num-layers={FWD_NUM_LAYERS} not in {FWD_LAYER_CHOICES}"
    )
if FWD_NUM_LAYERS < 4:
    raise SystemExit(
        f"--num-layers must be >= 4 (got {FWD_NUM_LAYERS})"
    )
# L0/L1 stay explicit SWA; the remaining layers are CSA/HCA pairs.
PAIR_COUNT = (FWD_NUM_LAYERS - 2) // 2
# Odd layer counts (e.g. 43) append one final CSA layer after the pair loop;
# even counts (6, 8, 10) end at the last HCA pair and skip that branch. The
# final-CSA branch is a static import-time decision, not a runtime attention-
# kind branch.
HAS_FINAL_CSA = (FWD_NUM_LAYERS % 2) == 1
HCA_NUM_LAYERS = PAIR_COUNT
CSA_NUM_LAYERS = PAIR_COUNT + int(HAS_FINAL_CSA)
# Global layer index of the final CSA (only meaningful when HAS_FINAL_CSA).
FINAL_CSA_LAYER = 2 * PAIR_COUNT + 2 if HAS_FINAL_CSA else -1
# §8.17.2 diagnostic plumbing. --fwd-only is parsed at import so the JIT
# host graph can conditionally omit the LM-head child launch/allocation
# (Domain 6 buffers + second rank loop) while reusing the exact 43-layer
# rank core. --enable-scope-stats is the standard DFX flag forwarded into
# runtime_cfg; parsed at import only to keep argparse and the host-graph
# guard consistent. Both default off and are no-ops for production runs.
FWD_ONLY = _parse_static_bool("fwd-only", False)
ENABLE_SCOPE_STATS = _parse_static_bool("enable-scope-stats", False)
NUM_MOE_WAVES = LOCAL_PARTS * MAX_SEGMENT_TILES
assert NUM_MOE_WAVES == 4
assert RECV_MAX == N_RANKS * T, (
    f"CP FWD requires RECV_MAX == N_RANKS * T (got {RECV_MAX} vs {N_RANKS * T})"
)

# Phase 3 final-tail ABI (§8.7). The CP-local flattened row count fed to the
# HC head / RMSNorm / LM head: each rank owns LOCAL_PARTS * MAX_SEGMENT_TILES
# segment-tiles of T tokens each.
LOCAL_ROWS = LOCAL_PARTS * MAX_SEGMENT_TILES * T
# LM head owns its barrier counters independently of the MoE epoch protocol,
# so its done epoch restarts at 1 rather than continuing the MoE numbering.
LM_HEAD_COMM_EPOCH = LM_HEAD_DONE_VALUE
assert LM_HEAD_TP_SIZE <= CP_SIZE, (
    f"TP={LM_HEAD_TP_SIZE} must not exceed CP={CP_SIZE} (ranks are cut into "
    f"CP // TP tensor-parallel groups)"
)
assert CP_SIZE % LM_HEAD_TP_SIZE == 0, (
    f"CP={CP_SIZE} must be a multiple of TP={LM_HEAD_TP_SIZE} (ranks partition "
    f"into CP // TP tensor-parallel groups)"
)

# Coarse stage offsets within one layer (stage_in_layer, one-based).
_ATTN_STAGE_OFFSET = 1   # attention-complete boundary
_WAVE0_STAGE_OFFSET = 2  # MoE wave 0 complete

# FP32 copy tile: 4 tokens x 1 HC lane x D = 64 KiB.
COPY_TOKEN_TILE = 4
assert T % COPY_TOKEN_TILE == 0

# Epoch formulas (§4 + §5.4 + §5.1):
#   tail_comm_epoch     = layer_ordinal          (0,1,2,3)
#   moe_epoch_base      = 4 * layer_ordinal      (0,4,8,12)
#   stage_base          = 5 * layer_ordinal      (0,5,10,15)
#   compact_comm_epoch  = 0 for the first CSA/HCA layer (type-order index 0)
# Cross-layer windows reuse one bank per communication domain under these
# monotonic counters; payload_epoch stays 0 (EPOCHS == 1) in this phase.


# ---------------------------------------------------------------------------
# FWD-local coarse stage synchronization
# ---------------------------------------------------------------------------
@pl.jit.inline
def _fwd_attention_stage_barrier_from_x_attn(
    x_attn_flat: pl.Tensor[[NUM_MOE_WAVES * T, HC_MULT, D], pl.FP32],
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    stage_base: pl.Scalar[pl.INT32],
    stage_token: pl.Out[pl.Tensor[[1, 1, 8], pl.FP32]],
):
    """Align SWA/HCA ranks when wave 0 attention output is ready."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_stage_barrier"):
        one = pl.cast(1, pl.INT32)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=stage_done, peer=peer, offsets=[my_rank, 0],
                    value=one, op=pld.NotifyOp.AtomicAdd,
                )
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(
                    signal=stage_done, offsets=[peer, 0],
                    expected=stage_base + pl.cast(_ATTN_STAGE_OFFSET, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
        stage_token[0:1, 0:1, 0:8] = pl.slice(
            x_attn_flat, [1, 1, 8], [0, 0, 0]
        )
    return stage_token


@pl.jit.inline
def _fwd_attention_stage_barrier_from_completion(
    completion_token: pl.Tensor[[NUM_MOE_WAVES, 1, 8], pl.FP32],
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    stage_base: pl.Scalar[pl.INT32],
    stage_token: pl.Out[pl.Tensor[[1, 1, 8], pl.FP32]],
):
    """Align CSA ranks after the leaf publishes its complete-stage token."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="attn_stage_barrier"):
        one = pl.cast(1, pl.INT32)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=stage_done, peer=peer, offsets=[my_rank, 0],
                    value=one, op=pld.NotifyOp.AtomicAdd,
                )
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(
                    signal=stage_done, offsets=[peer, 0],
                    expected=stage_base + pl.cast(_ATTN_STAGE_OFFSET, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )
        stage_token[0:1, 0:1, 0:8] = pl.slice(
            completion_token, [1, 1, 8], [0, 0, 0]
        )
    return stage_token


@pl.jit.inline
def _fwd_wave_stage_barrier(
    wave_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    expected_stage: pl.Scalar[pl.INT32],
    stage_token: pl.Out[pl.Tensor[[1, 1, 8], pl.FP32]],
):
    """Wave boundary: the whole wave's output completes before the next wave
    reuses the shared MoE window bank."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="wave_stage_barrier") as wave_tid:
        _anchor = pl.read(wave_out, [0, 0, 0])
        one = pl.cast(1, pl.INT32)
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.notify(
                    target=stage_done, peer=peer, offsets=[my_rank, 0],
                    value=one, op=pld.NotifyOp.AtomicAdd,
                )
        for peer in pl.range(CP_SIZE):
            if peer != my_rank:
                pld.system.wait(
                    signal=stage_done, offsets=[peer, 0],
                    expected=expected_stage,
                    cmp=pld.WaitCmp.Ge,
                )
        stage_token[0:1, 0:1, 0:8] = pl.slice(
            wave_out, [1, 1, 8], [0, 0, 0]
        )
    return stage_token


@pl.jit.inline
def _fwd_publish_hidden(
    x_next_work: pl.Tensor[[NUM_MOE_WAVES * T, HC_MULT, D], pl.FP32],
    active_flat: pl.Tensor[[NUM_MOE_WAVES, OVERLAY_SOURCES], pl.INT32],
    stage_token: pl.Tensor[[1, 1, 8], pl.FP32],
    hidden_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
) -> pl.Tensor[
    [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
]:
    """Publish active rows and zero inactive rows through one HBM copy task.

    Reads the final ``stage_token`` (the all-rank wave barrier) so the returned
    ``hidden_out`` explicitly depends on the global layer-completion boundary,
    not just on local MoE wave outputs. This edge is required for the cross-layer
    chain: final wave/barrier -> hidden publication -> next-layer attention."""
    hidden_flat = pl.reshape(
        hidden_out, [NUM_MOE_WAVES * T, HC_MULT, D]
    )
    wave_blocks = (T // COPY_TOKEN_TILE) * HC_MULT
    with pl.spmd(
        NUM_MOE_WAVES * wave_blocks,
        name_hint="publish_hidden",
    ):
        block = pl.tile.get_block_idx()
        # Order publication after the final all-rank stage barrier.
        _stage_tok = pl.read(stage_token, [0, 0, 0])
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
                hidden_flat[
                    row : row + 1,
                    hc_lane : hc_lane + 1,
                    0:D,
                ] = pl.slice(
                    x_next_work,
                    [1, 1, D],
                    [row, hc_lane, 0],
                )
            else:
                hidden_flat[
                    row : row + 1,
                    hc_lane : hc_lane + 1,
                    0:D,
                ] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)
    return pl.reshape(
        hidden_flat,
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
    )


@pl.jit.inline
def _fwd_moe_tail(
    x_attn: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
    ],
    overlay_active_lengths: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, OVERLAY_SOURCES], pl.INT32
    ],
    input_ids: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64
    ],
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
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hidden_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    completion_anchor: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    stage_token: pl.Tensor[[1, 1, 8], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    moe_epoch_base: pl.Scalar[pl.INT32],
    stage_base: pl.Scalar[pl.INT32],
) -> pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32]:
    """FWD-local MoE tail: attention boundary -> four MoE waves -> publish
    hidden_out and completion_anchor. Epoch values are offset by
    moe_epoch_base / stage_base so the same communication bank is reused
    across layers.

    ``completion_anchor`` is a rank-3 tensor written from the final
    ``stage_token`` and the final wave's data. The caller passes it to
    ``clear_moe_signals`` so signal cleanup is ordered after the final
    all-rank wave boundary (§10.8.2-4)."""
    x_attn_flat = pl.reshape(x_attn, [NUM_MOE_WAVES * T, HC_MULT, D])

    x_moe_ready = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32, init_value=0.0)
    x_next_work = pl.create_tensor(
        [NUM_MOE_WAVES * T, HC_MULT, D], dtype=pl.FP32, init_value=0.0
    )
    active_flat = pl.reshape(
        overlay_active_lengths, [NUM_MOE_WAVES, OVERLAY_SOURCES]
    )
    input_ids_flat = pl.reshape(input_ids, [NUM_MOE_WAVES * T])

    one = pl.cast(1, pl.INT32)
    for wave in pl.range(NUM_MOE_WAVES):
        row0 = wave * T
        active = pl.read(active_flat, [wave, 1])
        effective_tokens = pl.max(active, one)
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
            _anchor = pl.read(stage_token, [0, 0, 0])
            x_moe_ready[
                token0 : token0 + COPY_TOKEN_TILE,
                hc_lane : hc_lane + 1,
                0:D,
            ] = pl.slice(
                x_src,
                [COPY_TOKEN_TILE, 1, D],
                [token0, hc_lane, 0],
            )
        wave_out = pl.create_tensor([T, HC_MULT, D], dtype=pl.FP32, init_value=0.0)
        moe_epoch = moe_epoch_base + pl.cast(wave + 1, pl.INT32)
        moe(
            x_moe_ready, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
            norm_w, gate_w, gate_bias, tid2eid, ids,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            shared_w1, shared_w1_scale, shared_w3, shared_w3_scale,
            shared_w2, shared_w2_scale,
            wave_out,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            layer_id, effective_tokens, my_rank, moe_epoch,
        )
        x_next_work = pl.assemble(x_next_work, wave_out, [row0, 0, 0])
        expected_stage = stage_base + pl.cast(_WAVE0_STAGE_OFFSET + wave, pl.INT32)
        stage_token = _fwd_wave_stage_barrier(
            wave_out, stage_done, my_rank, expected_stage, stage_token,
        )

    # Materialize a real rank-3 completion anchor from the final stage_token
    # and the final wave's data. x_next_work is [NUM_MOE_WAVES * T, HC_MULT, D],
    # so final_row = (NUM_MOE_WAVES - 1) * T is a valid index on axis 0
    # (size 4*T = 512). pl.slice on a direct tensor argument lowers to
    # tile.load (copy), which FlattenTileNdTo2D accepts (§10.8.2-4).
    final_row = (NUM_MOE_WAVES - 1) * T
    with pl.spmd(
        1,
        name_hint="final_completion_anchor",
    ):
        _block_idx = pl.tile.get_block_idx()
        # Order cleanup after the final all-rank wave boundary.
        _anchor = pl.read(stage_token, [0, 0, 0])
        final_element = pl.slice(
            x_next_work, [1, 1, 8], [final_row, 0, 0]
        )
        completion_anchor[0:1, 0:1, 0:8] = final_element

    hidden_out = _fwd_publish_hidden(
        x_next_work, active_flat, stage_token, hidden_out
    )
    return hidden_out


# ---------------------------------------------------------------------------
# Rank-local forward child
# ---------------------------------------------------------------------------
@pl.jit(auto_scope=False)
def prefill_cp_fwd(
    x_hc: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D], pl.FP32
    ],
    # SWA attention weights (layer-stacked: FWD_NUM_LAYERS * <unit>).
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    # kv_cache layers are concatenated on dim 0 (FWD_NUM_LAYERS * ORI_MAX_BLOCKS
    # blocks total) so a 4D slice matches the source rank and the inline SWA
    # core receives a statically shaped [ORI_MAX_BLOCKS, ...] InOut whose
    # pl.reshape([ORI_CACHE_ROWS, HEAD_DIM]) is statically inferable. Mirrors
    # baseline prefill_fwd.py's layer-stacked kv_cache convention.
    kv_cache: pl.InOut[
        pl.Tensor[
            [FWD_NUM_LAYERS * ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    # Compressed KV cache: FWD_NUM_LAYERS per-layer pools (every attention
    # layer owns a compressed-KV slice); sliced by the global layer index.
    cmp_kv: pl.InOut[
        pl.Tensor[
            [FWD_NUM_LAYERS * PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
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
    # --- Shared CP metadata needed by CSA/HCA cores (beyond SWA) ----------
    # These are content-identical across all four layers (same zero-history
    # fixture layout); CSA and HCA cores require them as separate arguments.
    segment_active_lengths: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    # --- HCA type-specific (layers 3 and 5) -----------------------------
    # HCA compact compressor weights (ratio-128: OUT_DIM == HEAD_DIM, so
    # cmp_wkv/cmp_wgate are [HEAD_DIM, D] and cmp_ape is [RATIO, HEAD_DIM]).
    # Stacked by HCA_NUM_LAYERS on axis 0 -> [HCA_NUM_LAYERS * unit, ...];
    # the child slices its type ordinal (0 for L3, 1 for L5) per layer.
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    # HCA persistent compressor state (rank-local InOut root; stacked by
    # HCA_NUM_LAYERS on axis 0 -> [HCA_NUM_LAYERS * unit, ...]).
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [HCA_NUM_LAYERS * HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[
        [HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    # HCA-specific metadata.
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    snapshot_positions: pl.Tensor[[LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[LOCAL_PARTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    cmp_indices: pl.Tensor[
        [LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    # --- CSA type-specific (layers 2 and 4) -----------------------------
    # CSA main compressor weights (ratio-4: MAIN_OUT_DIM = 2*HEAD_DIM).
    # Stacked by CSA_NUM_LAYERS on axis 0 -> [CSA_NUM_LAYERS * unit, ...];
    # the child slices its type ordinal (0 for L2, 1 for L4) per layer.
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    # CSA indexer weights (stacked by CSA_NUM_LAYERS on axis 0).
    hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    # CSA inner compressor weights (ratio-4: INNER_OUT_DIM = 2*IDX_HEAD_DIM).
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    # CSA persistent state/caches (rank-local InOut roots; stacked by
    # CSA_NUM_LAYERS on axis 0 -> [CSA_NUM_LAYERS * unit, ...]).
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_NUM_LAYERS * CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [CSA_NUM_LAYERS * CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[[CSA_NUM_LAYERS * PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[CSA_NUM_LAYERS * PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    csa_compress_state_block_table: pl.Tensor[
        [CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_inner_compress_state_block_table: pl.Tensor[
        [CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    idx_block_table: pl.Tensor[[IDX_CACHE_MAX_BLOCKS], pl.INT32],
    # CSA reusable rank-local workspaces (serial layers reuse them).
    main_state_workspace0: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace0: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_workspace1: pl.Tensor[
        [CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace1: pl.Tensor[
        [CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    effective_x_workspace: pl.Tensor[
        [CSA_LOCAL_LEAVES * T, D], pl.BF16
    ],
    # CSA-specific metadata.
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[
        [LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES], pl.INT32
    ],
    # Shared compressed-KV block table (HCA and CSA compact both index it).
    cmp_block_table: pl.Tensor[[PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    # --- Communication windows ------------------------------------------
    # Domain 1: shared tail exchange (SWA + CSA + HCA reuse one bank under
    # monotonic tail_comm_epoch). The dual-tail exchange also needs a
    # hidden-tail window; SWA's zigzag exchange uses kv_tail_window only.
    kv_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    hidden_tail_window: pld.DistributedTensor[
        [CP_TAIL_WINDOW_ROWS, D], pl.BF16
    ],
    tail_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    tail_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Domain 2: HCA compact (one bank reused by HCA layers; compact_comm_epoch).
    cmp_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, HEAD_DIM], pl.BF16
    ],
    cmp_meta_window: pld.DistributedTensor[
        [CMP_WINDOW_ROWS, CMP_META_DIM], pl.INT32
    ],
    state_window: pld.DistributedTensor[
        [STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM], pl.FP32
    ],
    state_meta_window: pld.DistributedTensor[
        [CP_SIZE, STATE_META_DIM], pl.INT32
    ],
    hca_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    hca_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Domain 3: CSA compact/index/state (one bank reused by CSA layers).
    main_window: pld.DistributedTensor[
        [RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM], pl.BF16
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
        [STATE_RECORDS_PER_WINDOW, CSA_MAIN_STATE_DIM], pl.FP32
    ],
    main_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    inner_state_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, CSA_INNER_STATE_DIM], pl.FP32
    ],
    inner_state_meta_window: pld.DistributedTensor[
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], pl.INT32
    ],
    csa_compact_ready: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    csa_compact_consumed: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # MoE weights (layer-stacked).
    hc_ffn_fn: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64],
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
    # MoE distributed windows (reused across layers; monotonic epoch).
    recv_meta: pld.DistributedTensor[[N_RANKS, N_LOCAL], pl.INT32],
    recv_x: pld.DistributedTensor[[N_LOCAL * RECV_MAX, D], pl.INT8],
    recv_aux: pld.DistributedTensor[[N_LOCAL * RECV_MAX, AUX_PAD], pl.FP32],
    recv_route: pld.DistributedTensor[[N_LOCAL * RECV_MAX, IDX_PAD], pl.INT32],
    arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    data_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    routed_y_buf: pld.DistributedTensor[[N_ROUTES, D], pl.BF16],
    combine_arrived: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # Layer stage synchronization window.
    stage_done: pld.DistributedTensor[[CP_SIZE, 1], pl.INT32],
    # Phase 3 final-tail weights (HC head + final RMSNorm). The HC head
    # projects the [HC_MULT, D] hyper-connection mix to a single [D] row; the
    # final RMSNorm normalizes it into hidden_out for the LM head.
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[D], pl.BF16],
    # Final outputs. pre_hc_hidden_out is the FP32 pre-HC MoE result (one
    # [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D] slab per rank);
    # hidden_out is the BF16 post-RMSNorm final hidden ([LOCAL_ROWS, D]) fed
    # to the host-launched lm_head_test. Both are pl.Out so the host ties them
    # to host-level output slots.
    pre_hc_hidden_out: pl.Out[
        pl.Tensor[
            [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    hidden_out: pl.Out[pl.Tensor[[LOCAL_ROWS, D], pl.BF16]],
    # Scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    my_rank: pl.Scalar[pl.INT32],
) -> None:
    """Generic CP forward child: SWA, SWA, repeated CSA/HCA pairs, optional
    final CSA tail, then clear + publish + HC head + final RMSNorm.

    Layer 0 (SWA): attention (tail ep 0) -> 4 MoE waves (epochs 1..4,
    stages 1..5) -> hidden_a.
    Layer 1 (SWA): attention (tail ep 1) -> 4 MoE waves (epochs 5..8,
    stages 6..10) -> hidden_b.
    For each pair_i in ``pl.range(PAIR_COUNT)``:
      CSA layer (2 + 2*pair_i): attention (tail ep = layer, CSA compact ep
      = pair_i) -> 4 MoE waves (epoch base 4*layer, stage base 5*layer) ->
      hidden_a (ping-pong).
      HCA layer (3 + 2*pair_i): attention (tail ep = layer, HCA compact ep
      = pair_i) -> 4 MoE waves (epoch base 4*layer, stage base 5*layer) ->
      hidden_b (ping-pong).
    If HAS_FINAL_CSA (odd num_layers, e.g. 43): final CSA layer (FINAL_CSA_LAYER)
    attention (tail ep = layer, CSA compact ep = CSA_NUM_LAYERS-1) -> 4 MoE
    waves (epoch base 4*layer, stage base 5*layer) -> hidden_a (ping-pong).
    Final: clear MoE signals once, publish the last hidden (hidden_a for odd
    tail, hidden_b for even tail) to pre_hc_hidden_out, then inline hc_head ->
    final rms_norm into hidden_out (mirrors baseline prefill_fwd.py). The host
    separately launches lm_head_test over hidden_out.
    """

    # --- Layer 0: SWA attention -> 4 MoE waves -> hidden_a ---------------
    x_attn_l0 = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    completion_token_l0 = pl.create_tensor(
        [NUM_MOE_WAVES, 1, 8], dtype=pl.FP32, init_value=0.0,
    )
    kv_cache_l0: pl.Tensor[
        [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16
    ] = pl.slice(
        kv_cache, [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], [0, 0, 0, 0]
    )
    hc_attn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0])
    hc_attn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [0 * 3])
    hc_attn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [0 * MIX_HC])
    attn_norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [0 * D])
    wq_a_l0: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [0 * D, 0])
    wq_b_l0: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [0 * Q_LORA, 0])
    wq_b_scale_l0: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [0 * H * HEAD_DIM])
    wkv_l0: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [0 * D, 0])
    gamma_cq_l0: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [0 * Q_LORA])
    gamma_ckv_l0: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [0 * HEAD_DIM])
    attn_sink_l0: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [0 * H])
    wo_a_l0: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [0 * O_GROUPS, 0, 0])
    wo_b_l0: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [0 * D, 0])
    wo_b_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [0 * D])

    with pl.scope():
        prefill_cp_swa_core(
            x_hc, hc_attn_fn_l0, hc_attn_scale_l0, hc_attn_base_l0, attn_norm_w_l0,
            wq_a_l0, wq_b_l0, wq_b_scale_l0, wkv_l0, gamma_cq_l0, gamma_ckv_l0,
            freqs_cos, freqs_sin, kv_cache_l0,
            attn_sink_l0, wo_a_l0, wo_b_l0, wo_b_scale_l0,
            segment_starts_t, predecessor_segments,
            query_position_ids, query_token_to_request,
            overlay_position_ids, overlay_token_to_request,
            overlay_active_lengths, swa_indices,
            reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            kv_tail_window, tail_ready, tail_consumed,
            x_attn_l0, completion_token_l0,
            my_rank, pl.cast(0, pl.INT32),
        )

    hidden_a = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    # Layer 0 completion anchor: produced but not consumed by clear_moe_signals
    # (intermediate layers retain monotonically increasing MoE signals; §10.8.6).
    completion_anchor_l0 = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32, init_value=0.0,
    )
    hc_ffn_fn_l0: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [0 * MIX_HC, 0])
    hc_ffn_scale_l0: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [0 * 3])
    hc_ffn_base_l0: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [0 * MIX_HC])
    norm_w_l0: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [0 * D])
    gate_w_l0: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [0 * N_EXPERTS_GLOBAL, 0])
    gate_bias_l0: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [0 * N_EXPERTS_GLOBAL])
    tid2eid_l0: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [0 * VOCAB, 0])
    routed_w1_l0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [0 * N_LOCAL, 0, 0])
    routed_w1_scale_l0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [0 * N_LOCAL, 0])
    routed_w3_l0: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [0 * N_LOCAL, 0, 0])
    routed_w3_scale_l0: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [0 * N_LOCAL, 0])
    routed_w2_l0: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [0 * N_LOCAL, 0, 0])
    routed_w2_scale_l0: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [0 * N_LOCAL, 0])
    shared_w1_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [0 * MOE_INTER, 0])
    shared_w1_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [0 * MOE_INTER])
    shared_w3_l0: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [0 * MOE_INTER, 0])
    shared_w3_scale_l0: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [0 * MOE_INTER])
    shared_w2_l0: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [0 * D, 0])
    shared_w2_scale_l0: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [0 * D])

    with pl.scope():
        stage_token_l0 = pl.create_tensor([1, 1, 8], dtype=pl.FP32, init_value=0.0)
        stage_token_l0 = _fwd_attention_stage_barrier_from_completion(
            completion_token_l0,
            stage_done, my_rank, pl.cast(0, pl.INT32), stage_token_l0,
        )
        _fwd_moe_tail(
            x_attn_l0, overlay_active_lengths, input_ids,
            hc_ffn_fn_l0, hc_ffn_scale_l0, hc_ffn_base_l0,
            norm_w_l0, gate_w_l0, gate_bias_l0, tid2eid_l0,
            routed_w1_l0, routed_w1_scale_l0,
            routed_w3_l0, routed_w3_scale_l0,
            routed_w2_l0, routed_w2_scale_l0,
            shared_w1_l0, shared_w1_scale_l0,
            shared_w3_l0, shared_w3_scale_l0,
            shared_w2_l0, shared_w2_scale_l0,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done, hidden_a, completion_anchor_l0,
            stage_token_l0,
            pl.cast(0, pl.INT32), my_rank,
            pl.cast(0, pl.INT32), pl.cast(0, pl.INT32),
        )

    # --- Layer 1: SWA attention -> 4 MoE waves -> hidden_b ---------------
    x_attn_l1 = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    completion_token_l1 = pl.create_tensor(
        [NUM_MOE_WAVES, 1, 8], dtype=pl.FP32, init_value=0.0,
    )
    kv_cache_l1: pl.Tensor[
        [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16
    ] = pl.slice(
        kv_cache, [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM],
        [1 * ORI_MAX_BLOCKS, 0, 0, 0],
    )
    hc_attn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0])
    hc_attn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [1 * 3])
    hc_attn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [1 * MIX_HC])
    attn_norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [1 * D])
    wq_a_l1: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [1 * D, 0])
    wq_b_l1: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [1 * Q_LORA, 0])
    wq_b_scale_l1: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [1 * H * HEAD_DIM])
    wkv_l1: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [1 * D, 0])
    gamma_cq_l1: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [1 * Q_LORA])
    gamma_ckv_l1: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [1 * HEAD_DIM])
    attn_sink_l1: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [1 * H])
    wo_a_l1: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [1 * O_GROUPS, 0, 0])
    wo_b_l1: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [1 * D, 0])
    wo_b_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [1 * D])

    with pl.scope():
        prefill_cp_swa_core(
            hidden_a, hc_attn_fn_l1, hc_attn_scale_l1, hc_attn_base_l1, attn_norm_w_l1,
            wq_a_l1, wq_b_l1, wq_b_scale_l1, wkv_l1, gamma_cq_l1, gamma_ckv_l1,
            freqs_cos, freqs_sin, kv_cache_l1,
            attn_sink_l1, wo_a_l1, wo_b_l1, wo_b_scale_l1,
            segment_starts_t, predecessor_segments,
            query_position_ids, query_token_to_request,
            overlay_position_ids, overlay_token_to_request,
            overlay_active_lengths, swa_indices,
            reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            kv_tail_window, tail_ready, tail_consumed,
            x_attn_l1, completion_token_l1,
            my_rank, pl.cast(1, pl.INT32),
        )

    hidden_b = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    # Layer 1 completion anchor: produced by _fwd_moe_tail, consumed by
    # clear_moe_signals so signal cleanup is ordered after the final
    # all-rank wave boundary (§10.8.2-4, §10.8.6).
    completion_anchor_l1 = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32, init_value=0.0,
    )
    hc_ffn_fn_l1: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [1 * MIX_HC, 0])
    hc_ffn_scale_l1: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [1 * 3])
    hc_ffn_base_l1: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [1 * MIX_HC])
    norm_w_l1: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [1 * D])
    gate_w_l1: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [1 * N_EXPERTS_GLOBAL, 0])
    gate_bias_l1: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [1 * N_EXPERTS_GLOBAL])
    tid2eid_l1: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [1 * VOCAB, 0])
    routed_w1_l1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [1 * N_LOCAL, 0, 0])
    routed_w1_scale_l1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [1 * N_LOCAL, 0])
    routed_w3_l1: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [1 * N_LOCAL, 0, 0])
    routed_w3_scale_l1: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [1 * N_LOCAL, 0])
    routed_w2_l1: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [1 * N_LOCAL, 0, 0])
    routed_w2_scale_l1: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [1 * N_LOCAL, 0])
    shared_w1_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [1 * MOE_INTER, 0])
    shared_w1_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [1 * MOE_INTER])
    shared_w3_l1: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [1 * MOE_INTER, 0])
    shared_w3_scale_l1: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [1 * MOE_INTER])
    shared_w2_l1: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [1 * D, 0])
    shared_w2_scale_l1: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [1 * D])

    # Reshape hidden_b/pre_hc_hidden_out views in enclosing scope so they
    # remain live across the L0/L1 SWA scopes, the pair loop, and the final
    # publish scope; only the publish SPMD task retires at the last scope exit.
    active_flat = pl.reshape(
        overlay_active_lengths, [NUM_MOE_WAVES, OVERLAY_SOURCES]
    )
    hidden_b_flat = pl.reshape(
        hidden_b, [NUM_MOE_WAVES * T, HC_MULT, D]
    )
    pre_hc_hidden_out_flat = pl.reshape(
        pre_hc_hidden_out, [NUM_MOE_WAVES * T, HC_MULT, D]
    )
    # hidden_a_flat is only read by the publish in the odd-tail branch, but it
    # must be created in the enclosing scope (not inside the ``if
    # HAS_FINAL_CSA`` branch) so the SSA verifier sees one definition visible to
    # the post-loop publish scope. ``hidden_a`` itself is allocated above as the
    # L0 ping-pong root, so this reshape is always valid.
    hidden_a_flat = pl.reshape(
        hidden_a, [NUM_MOE_WAVES * T, HC_MULT, D]
    )

    with pl.scope():
        stage_token_l1 = pl.create_tensor([1, 1, 8], dtype=pl.FP32, init_value=0.0)
        stage_token_l1 = _fwd_attention_stage_barrier_from_completion(
            completion_token_l1,
            stage_done, my_rank, pl.cast(5, pl.INT32), stage_token_l1,
        )
        _fwd_moe_tail(
            x_attn_l1, overlay_active_lengths, input_ids,
            hc_ffn_fn_l1, hc_ffn_scale_l1, hc_ffn_base_l1,
            norm_w_l1, gate_w_l1, gate_bias_l1, tid2eid_l1,
            routed_w1_l1, routed_w1_scale_l1,
            routed_w3_l1, routed_w3_scale_l1,
            routed_w2_l1, routed_w2_scale_l1,
            shared_w1_l1, shared_w1_scale_l1,
            shared_w3_l1, shared_w3_scale_l1,
            shared_w2_l1, shared_w2_scale_l1,
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done, hidden_b, completion_anchor_l1,
            stage_token_l1,
            pl.cast(1, pl.INT32), my_rank,
            pl.cast(4, pl.INT32), pl.cast(5, pl.INT32),
        )
        # L1 is an intermediate layer — do NOT clear MoE signals or publish
        # pre_hc_hidden_out here. Final clear + publish run after the pair loop (§7.4).

    # --- CSA/HCA pair loop (§7.4) ---------------------------------------
    # L0/L1 stay explicit SWA above; the remaining layers are CSA/HCA pairs
    # driven by one ``pl.range(PAIR_COUNT)`` loop. The pair loop reuses the
    # outer ``hidden_a``/``hidden_b`` ping-pong roots and one CSA attention
    # output, one HCA attention output, one CSA anchor, and one HCA anchor
    # (the pairs are serialized by coarse scopes; each producer fully
    # overwrites its output before the next consumer — §7.5). The HCA anchor
    # carries the final-layer completion out to the post-loop publish scope.
    x_attn_csa = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    x_attn_hca = pl.create_tensor(
        [LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        dtype=pl.FP32, init_value=0.0,
    )
    completion_anchor_csa = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32, init_value=0.0,
    )
    completion_anchor_hca = pl.create_tensor(
        [T, HC_MULT, D], dtype=pl.FP32, init_value=0.0,
    )
    # Loop invariant: at pair entry ``hidden_b`` holds the previous layer
    # result; CSA reads hidden_b -> x_attn_csa -> MoE -> hidden_a; HCA reads
    # hidden_a -> x_attn_hca -> MoE -> hidden_b. No parity branch is needed.
    for pair_i in pl.range(PAIR_COUNT):
        csa_layer: pl.Scalar[pl.INT32] = pl.cast(2 + 2 * pair_i, pl.INT32)
        hca_layer: pl.Scalar[pl.INT32] = pl.cast(3 + 2 * pair_i, pl.INT32)
        # Common attention/MoE epoch bases derive from the global layer index
        # (§7.6); the compact comm epoch is the pair index (second use, etc.).
        csa_moe_epoch_base: pl.Scalar[pl.INT32] = pl.cast(4 * csa_layer, pl.INT32)
        csa_stage_base: pl.Scalar[pl.INT32] = pl.cast(5 * csa_layer, pl.INT32)
        hca_moe_epoch_base: pl.Scalar[pl.INT32] = pl.cast(4 * hca_layer, pl.INT32)
        hca_stage_base: pl.Scalar[pl.INT32] = pl.cast(5 * hca_layer, pl.INT32)
        compact_ep: pl.Scalar[pl.INT32] = pl.cast(pair_i, pl.INT32)

        # --- CSA attention weight slices (global layer index csa_layer) ---
        kv_cache_csa: pl.Tensor[
            [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16
        ] = pl.slice(
            kv_cache, [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM],
            [csa_layer * ORI_MAX_BLOCKS, 0, 0, 0],
        )
        cmp_kv_csa: pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ] = pl.slice(
            cmp_kv, [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            [csa_layer * PREFILL_CMP_BLOCK_NUM, 0, 0, 0],
        )
        hc_attn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0])
        hc_attn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [csa_layer * 3])
        hc_attn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [csa_layer * MIX_HC])
        attn_norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [csa_layer * D])
        wq_a_csa: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [csa_layer * D, 0])
        wq_b_csa: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [csa_layer * Q_LORA, 0])
        wq_b_scale_csa: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [csa_layer * H * HEAD_DIM])
        wkv_csa: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [csa_layer * D, 0])
        gamma_cq_csa: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [csa_layer * Q_LORA])
        gamma_ckv_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [csa_layer * HEAD_DIM])
        attn_sink_csa: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [csa_layer * H])
        wo_a_csa: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [csa_layer * O_GROUPS, 0, 0])
        wo_b_csa: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [csa_layer * D, 0])
        wo_b_scale_csa: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [csa_layer * D])

        # CSA type-ordinal slices (type ordinal = pair_i). The CSA weights
        # and persistent state are stacked by CSA_NUM_LAYERS on axis 0.
        csa_cmp_wkv_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [pair_i * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_wgate_csa: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [pair_i * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [pair_i * CSA_COMPRESS_RATIO, 0])
        csa_cmp_norm_w_csa: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [pair_i * HEAD_DIM])
        hadamard_idx_csa: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = pl.slice(hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [pair_i * IDX_HEAD_DIM, 0])
        idx_wq_b_csa: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8] = pl.slice(idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [pair_i * Q_LORA, 0])
        idx_wq_b_scale_csa: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = pl.slice(idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [pair_i * IDX_N_HEADS * IDX_HEAD_DIM])
        idx_weights_proj_csa: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(idx_weights_proj, [D, IDX_N_HEADS], [pair_i * D, 0])
        csa_inner_wkv_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [pair_i * CSA_INNER_OUT_DIM, 0])
        csa_inner_wgate_csa: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [pair_i * CSA_INNER_OUT_DIM, 0])
        csa_inner_ape_csa: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [pair_i * CSA_COMPRESS_RATIO, 0])
        csa_inner_norm_w_csa: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [pair_i * IDX_HEAD_DIM])
        csa_compress_state_csa: pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32] = pl.slice(csa_compress_state, [CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], [pair_i * CSA_STATE_PHYSICAL_BLOCKS, 0, 0])
        csa_inner_compress_state_csa: pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32] = pl.slice(csa_inner_compress_state, [CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], [pair_i * CSA_INNER_STATE_PHYSICAL_BLOCKS, 0, 0])
        idx_kv_cache_csa: pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8] = pl.slice(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], [pair_i * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])
        idx_kv_scale_csa: pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32] = pl.slice(idx_kv_scale, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], [pair_i * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])

        # §8.17.8e.2 CSA leaf-capture completion token: created outside the
        # attention scope so the sibling MoE scope (consumes it via
        # _fwd_attention_stage_barrier) can see it. Published atomically by
        # prefill_cp_csa_core's terminal cp_csa_rank_complete task.
        completion_token_csa = pl.create_tensor(
            [NUM_MOE_WAVES, 1, 8], dtype=pl.FP32, init_value=0.0
        )
        # scope: CSA attention (tail epoch = csa_layer, CSA compact ep =
        # pair_i). CSA reads hidden_b (previous layer result).
        with pl.scope():
            prefill_cp_csa_core(
                hidden_b,
                hc_attn_fn_csa, hc_attn_scale_csa, hc_attn_base_csa, attn_norm_w_csa,
                wq_a_csa, wq_b_csa, wq_b_scale_csa, wkv_csa, gamma_cq_csa, gamma_ckv_csa,
                freqs_cos, freqs_sin,
                csa_cmp_wkv_csa, csa_cmp_wgate_csa, csa_cmp_ape_csa, csa_cmp_norm_w_csa,
                hadamard_idx_csa, idx_wq_b_csa, idx_wq_b_scale_csa, idx_weights_proj_csa,
                csa_inner_wkv_csa, csa_inner_wgate_csa, csa_inner_ape_csa, csa_inner_norm_w_csa,
                main_state_workspace0, inner_state_workspace0,
                main_state_workspace1, inner_state_workspace1,
                csa_compress_state_csa, csa_compress_state_block_table,
                csa_inner_compress_state_csa, csa_inner_compress_state_block_table,
                kv_cache_csa, cmp_kv_csa, cmp_block_table,
                idx_kv_cache_csa, idx_kv_scale_csa, idx_block_table,
                segment_starts_t, segment_lengths_t,
                segment_active_lengths, owner_segments_t, predecessor_segments,
                query_position_ids, query_token_to_request,
                overlay_position_ids, overlay_token_to_request,
                overlay_active_lengths, swa_indices,
                final_segment_t, reverse_index, owner_rank_table,
                final_win_seg_src, final_win_row_src, final_slot_mapping,
                leaf_positions_input, leaf_main_slots_input,
                leaf_idx_slots_input, leaf_main_state_slots_input,
                leaf_inner_state_slots_input, leaf_num_tokens_input,
                effective_x_workspace,
                hidden_tail_window, kv_tail_window, tail_ready, tail_consumed,
                main_window, idx_window, scale_window, record_window,
                main_state_window, main_state_meta_window,
                inner_state_window, inner_state_meta_window,
                csa_compact_ready, csa_compact_consumed,
                attn_sink_csa, wo_a_csa, wo_b_csa, wo_b_scale_csa,
                x_attn_csa,
                completion_token_csa,
                my_rank,
                csa_layer,  # tail_comm_epoch
                compact_ep,  # compact_comm_epoch_base
            )

        # CSA MoE weight slices (global layer index csa_layer).
        hc_ffn_fn_csa: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [csa_layer * MIX_HC, 0])
        hc_ffn_scale_csa: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [csa_layer * 3])
        hc_ffn_base_csa: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [csa_layer * MIX_HC])
        norm_w_csa: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [csa_layer * D])
        gate_w_csa: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [csa_layer * N_EXPERTS_GLOBAL, 0])
        gate_bias_csa: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [csa_layer * N_EXPERTS_GLOBAL])
        tid2eid_csa: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [csa_layer * VOCAB, 0])
        routed_w1_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [csa_layer * N_LOCAL, 0, 0])
        routed_w1_scale_csa: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [csa_layer * N_LOCAL, 0])
        routed_w3_csa: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [csa_layer * N_LOCAL, 0, 0])
        routed_w3_scale_csa: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [csa_layer * N_LOCAL, 0])
        routed_w2_csa: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [csa_layer * N_LOCAL, 0, 0])
        routed_w2_scale_csa: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [csa_layer * N_LOCAL, 0])
        shared_w1_csa: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [csa_layer * MOE_INTER, 0])
        shared_w1_scale_csa: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [csa_layer * MOE_INTER])
        shared_w3_csa: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [csa_layer * MOE_INTER, 0])
        shared_w3_scale_csa: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [csa_layer * MOE_INTER])
        shared_w2_csa: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [csa_layer * D, 0])
        shared_w2_scale_csa: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [csa_layer * D])

        # scope: CSA MoE tail (epoch base 4*csa_layer, stage base 5*csa_layer)
        # -> hidden_a (ping-pong). No clear after intermediate pairs.
        with pl.scope():
            stage_token_csa = pl.create_tensor(
                [1, 1, 8], dtype=pl.FP32, init_value=0.0
            )
            stage_token_csa = _fwd_attention_stage_barrier_from_completion(
                completion_token_csa, stage_done, my_rank, csa_stage_base,
                stage_token_csa,
            )
            _fwd_moe_tail(
                x_attn_csa, overlay_active_lengths, input_ids,
                hc_ffn_fn_csa, hc_ffn_scale_csa, hc_ffn_base_csa,
                norm_w_csa, gate_w_csa, gate_bias_csa, tid2eid_csa,
                routed_w1_csa, routed_w1_scale_csa,
                routed_w3_csa, routed_w3_scale_csa,
                routed_w2_csa, routed_w2_scale_csa,
                shared_w1_csa, shared_w1_scale_csa,
                shared_w3_csa, shared_w3_scale_csa,
                shared_w2_csa, shared_w2_scale_csa,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                stage_done, hidden_a, completion_anchor_csa,
                stage_token_csa,
                csa_layer, my_rank,
                csa_moe_epoch_base, csa_stage_base,
            )

        # --- HCA attention weight slices (global layer index hca_layer) ---
        kv_cache_hca: pl.Tensor[
            [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16
        ] = pl.slice(
            kv_cache, [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM],
            [hca_layer * ORI_MAX_BLOCKS, 0, 0, 0],
        )
        cmp_kv_hca: pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ] = pl.slice(
            cmp_kv, [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            [hca_layer * PREFILL_CMP_BLOCK_NUM, 0, 0, 0],
        )
        hc_attn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0])
        hc_attn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [hca_layer * 3])
        hc_attn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [hca_layer * MIX_HC])
        attn_norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [hca_layer * D])
        wq_a_hca: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [hca_layer * D, 0])
        wq_b_hca: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [hca_layer * Q_LORA, 0])
        wq_b_scale_hca: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [hca_layer * H * HEAD_DIM])
        wkv_hca: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [hca_layer * D, 0])
        gamma_cq_hca: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [hca_layer * Q_LORA])
        gamma_ckv_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [hca_layer * HEAD_DIM])
        attn_sink_hca: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [hca_layer * H])
        wo_a_hca: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [hca_layer * O_GROUPS, 0, 0])
        wo_b_hca: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [hca_layer * D, 0])
        wo_b_scale_hca: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [hca_layer * D])

        # HCA type-ordinal slices (type ordinal = pair_i). The HCA weights
        # and persistent state are stacked by HCA_NUM_LAYERS on axis 0.
        hca_cmp_wkv_hca: pl.Tensor[[HEAD_DIM, D], pl.BF16] = pl.slice(hca_cmp_wkv, [HEAD_DIM, D], [pair_i * HEAD_DIM, 0])
        hca_cmp_wgate_hca: pl.Tensor[[HEAD_DIM, D], pl.BF16] = pl.slice(hca_cmp_wgate, [HEAD_DIM, D], [pair_i * HEAD_DIM, 0])
        hca_cmp_ape_hca: pl.Tensor[[HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32] = pl.slice(hca_cmp_ape, [HCA_COMPRESS_RATIO, HEAD_DIM], [pair_i * HCA_COMPRESS_RATIO, 0])
        hca_cmp_norm_w_hca: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(hca_cmp_norm_w, [HEAD_DIM], [pair_i * HEAD_DIM])
        hca_compress_state_hca: pl.Tensor[[HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], pl.FP32] = pl.slice(hca_compress_state, [HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM], [pair_i * HCA_STATE_PHYSICAL_BLOCKS, 0, 0])

        # scope: HCA attention (tail epoch = hca_layer, HCA compact ep =
        # pair_i). HCA reads hidden_a (CSA output) and writes the HCA anchor.
        with pl.scope():
            prefill_cp_hca_core(
                hidden_a,
                hc_attn_fn_hca, hc_attn_scale_hca, hc_attn_base_hca, attn_norm_w_hca,
                wq_a_hca, wq_b_hca, wq_b_scale_hca, wkv_hca, gamma_cq_hca, gamma_ckv_hca,
                freqs_cos, freqs_sin,
                hca_cmp_wkv_hca, hca_cmp_wgate_hca, hca_cmp_ape_hca, hca_cmp_norm_w_hca,
                hca_compress_state_hca, hca_compress_state_block_table,
                kv_cache_hca, cmp_kv_hca, cmp_block_table,
                segment_starts_t, segment_active_lengths, owner_segments_t,
                predecessor_segments,
                query_position_ids, query_token_to_request,
                overlay_position_ids, overlay_token_to_request,
                overlay_active_lengths, swa_indices,
                cmp_indices,
                segment_tail_positions, snapshot_positions, snapshot_valid,
                final_segment_t,
                reverse_index, owner_rank_table, owner_part_table,
                final_win_seg_src, final_win_row_src, final_slot_mapping,
                hidden_tail_window, kv_tail_window, tail_ready, tail_consumed,
                cmp_window, cmp_meta_window, state_window, state_meta_window,
                hca_compact_ready, hca_compact_consumed,
                attn_sink_hca, wo_a_hca, wo_b_hca, wo_b_scale_hca,
                x_attn_hca,
                my_rank,
                hca_layer,  # tail_comm_epoch
                compact_ep,  # compact_comm_epoch_base
            )

        # HCA MoE weight slices (global layer index hca_layer).
        hc_ffn_fn_hca: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [hca_layer * MIX_HC, 0])
        hc_ffn_scale_hca: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [hca_layer * 3])
        hc_ffn_base_hca: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [hca_layer * MIX_HC])
        norm_w_hca: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [hca_layer * D])
        gate_w_hca: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [hca_layer * N_EXPERTS_GLOBAL, 0])
        gate_bias_hca: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [hca_layer * N_EXPERTS_GLOBAL])
        tid2eid_hca: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [hca_layer * VOCAB, 0])
        routed_w1_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [hca_layer * N_LOCAL, 0, 0])
        routed_w1_scale_hca: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [hca_layer * N_LOCAL, 0])
        routed_w3_hca: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [hca_layer * N_LOCAL, 0, 0])
        routed_w3_scale_hca: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [hca_layer * N_LOCAL, 0])
        routed_w2_hca: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [hca_layer * N_LOCAL, 0, 0])
        routed_w2_scale_hca: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [hca_layer * N_LOCAL, 0])
        shared_w1_hca: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [hca_layer * MOE_INTER, 0])
        shared_w1_scale_hca: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [hca_layer * MOE_INTER])
        shared_w3_hca: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [hca_layer * MOE_INTER, 0])
        shared_w3_scale_hca: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [hca_layer * MOE_INTER])
        shared_w2_hca: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [hca_layer * D, 0])
        shared_w2_scale_hca: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [hca_layer * D])

        # scope: HCA MoE tail (epoch base 4*hca_layer, stage base
        # 5*hca_layer) -> hidden_b (ping-pong). The HCA anchor produced here
        # is the last-pair completion anchor (§7.8); final clear + publish
        # runs after the loop.
        with pl.scope():
            x_attn_hca_flat = pl.reshape(
                x_attn_hca, [NUM_MOE_WAVES * T, HC_MULT, D]
            )
            stage_token_hca = pl.create_tensor(
                [1, 1, 8], dtype=pl.FP32, init_value=0.0
            )
            stage_token_hca = _fwd_attention_stage_barrier_from_x_attn(
                x_attn_hca_flat,
                stage_done, my_rank, hca_stage_base, stage_token_hca,
            )
            _fwd_moe_tail(
                x_attn_hca, overlay_active_lengths, input_ids,
                hc_ffn_fn_hca, hc_ffn_scale_hca, hc_ffn_base_hca,
                norm_w_hca, gate_w_hca, gate_bias_hca, tid2eid_hca,
                routed_w1_hca, routed_w1_scale_hca,
                routed_w3_hca, routed_w3_scale_hca,
                routed_w2_hca, routed_w2_scale_hca,
                shared_w1_hca, shared_w1_scale_hca,
                shared_w3_hca, shared_w3_scale_hca,
                shared_w2_hca, shared_w2_scale_hca,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                stage_done, hidden_b, completion_anchor_hca,
                stage_token_hca,
                hca_layer, my_rank,
                hca_moe_epoch_base, hca_stage_base,
            )

    # --- Final tail: optional final CSA layer, then clear + publish + HC head ---
    # Even num_layers (6/8/10): the pair loop exits with hidden_b = last HCA
    # MoE result; the HCA anchor carries the final wave boundary; publish
    # reads hidden_b. Odd num_layers (43): a final CSA layer (L42) runs after
    # the loop; it reads hidden_b -> x_attn_csa -> MoE -> hidden_a (ping-pong,
    # since L42 is an even global layer index like every CSA), so publish
    # reads hidden_a and the final-CSA anchor carries the wave boundary. The
    # branch is a static import-time decision (HAS_FINAL_CSA), not a runtime
    # attention-kind branch.
    if HAS_FINAL_CSA:
        final_csa_layer: pl.Scalar[pl.INT32] = pl.cast(FINAL_CSA_LAYER, pl.INT32)
        final_csa_compact_ep: pl.Scalar[pl.INT32] = pl.cast(CSA_NUM_LAYERS - 1, pl.INT32)
        final_csa_moe_epoch_base: pl.Scalar[pl.INT32] = pl.cast(4 * FINAL_CSA_LAYER, pl.INT32)
        final_csa_stage_base: pl.Scalar[pl.INT32] = pl.cast(5 * FINAL_CSA_LAYER, pl.INT32)
        completion_anchor_final_csa = pl.create_tensor(
            [T, HC_MULT, D], dtype=pl.FP32, init_value=0.0,
        )

        # --- Final CSA L42 attention weight slices (global layer index) ----
        kv_cache_final: pl.Tensor[
            [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM], pl.BF16
        ] = pl.slice(
            kv_cache, [ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM],
            [final_csa_layer * ORI_MAX_BLOCKS, 0, 0, 0],
        )
        cmp_kv_final: pl.Tensor[
            [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16
        ] = pl.slice(
            cmp_kv, [PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            [final_csa_layer * PREFILL_CMP_BLOCK_NUM, 0, 0, 0],
        )
        hc_attn_fn_final: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_attn_fn, [MIX_HC, HC_DIM], [final_csa_layer * MIX_HC, 0])
        hc_attn_scale_final: pl.Tensor[[3], pl.FP32] = pl.slice(hc_attn_scale, [3], [final_csa_layer * 3])
        hc_attn_base_final: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_attn_base, [MIX_HC], [final_csa_layer * MIX_HC])
        attn_norm_w_final: pl.Tensor[[D], pl.BF16] = pl.slice(attn_norm_w, [D], [final_csa_layer * D])
        wq_a_final: pl.Tensor[[D, Q_LORA], pl.BF16] = pl.slice(wq_a, [D, Q_LORA], [final_csa_layer * D, 0])
        wq_b_final: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8] = pl.slice(wq_b, [Q_LORA, H * HEAD_DIM], [final_csa_layer * Q_LORA, 0])
        wq_b_scale_final: pl.Tensor[[H * HEAD_DIM], pl.FP32] = pl.slice(wq_b_scale, [H * HEAD_DIM], [final_csa_layer * H * HEAD_DIM])
        wkv_final: pl.Tensor[[D, HEAD_DIM], pl.BF16] = pl.slice(wkv, [D, HEAD_DIM], [final_csa_layer * D, 0])
        gamma_cq_final: pl.Tensor[[Q_LORA], pl.BF16] = pl.slice(gamma_cq, [Q_LORA], [final_csa_layer * Q_LORA])
        gamma_ckv_final: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(gamma_ckv, [HEAD_DIM], [final_csa_layer * HEAD_DIM])
        attn_sink_final: pl.Tensor[[H], pl.FP32] = pl.slice(attn_sink, [H], [final_csa_layer * H])
        wo_a_final: pl.Tensor[[O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16] = pl.slice(wo_a, [O_GROUPS, O_LORA, O_GROUP_IN], [final_csa_layer * O_GROUPS, 0, 0])
        wo_b_final: pl.Tensor[[D, O_GROUPS * O_LORA], pl.INT8] = pl.slice(wo_b, [D, O_GROUPS * O_LORA], [final_csa_layer * D, 0])
        wo_b_scale_final: pl.Tensor[[D], pl.FP32] = pl.slice(wo_b_scale, [D], [final_csa_layer * D])

        # Final-CSA type-ordinal slices (CSA ordinal = CSA_NUM_LAYERS - 1).
        # The CSA weights/state are stacked by CSA_NUM_LAYERS on axis 0; the
        # final CSA is the last type slot (PAIR_COUNT for the 43-layer case).
        csa_ordinal_final = final_csa_compact_ep
        csa_cmp_wkv_final: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wkv, [CSA_MAIN_OUT_DIM, D], [csa_ordinal_final * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_wgate_final: pl.Tensor[[CSA_MAIN_OUT_DIM, D], pl.BF16] = pl.slice(csa_cmp_wgate, [CSA_MAIN_OUT_DIM, D], [csa_ordinal_final * CSA_MAIN_OUT_DIM, 0])
        csa_cmp_ape_final: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32] = pl.slice(csa_cmp_ape, [CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], [csa_ordinal_final * CSA_COMPRESS_RATIO, 0])
        csa_cmp_norm_w_final: pl.Tensor[[HEAD_DIM], pl.BF16] = pl.slice(csa_cmp_norm_w, [HEAD_DIM], [csa_ordinal_final * HEAD_DIM])
        hadamard_idx_final: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16] = pl.slice(hadamard_idx, [IDX_HEAD_DIM, IDX_HEAD_DIM], [csa_ordinal_final * IDX_HEAD_DIM, 0])
        idx_wq_b_final: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8] = pl.slice(idx_wq_b, [Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], [csa_ordinal_final * Q_LORA, 0])
        idx_wq_b_scale_final: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32] = pl.slice(idx_wq_b_scale, [IDX_N_HEADS * IDX_HEAD_DIM], [csa_ordinal_final * IDX_N_HEADS * IDX_HEAD_DIM])
        idx_weights_proj_final: pl.Tensor[[D, IDX_N_HEADS], pl.BF16] = pl.slice(idx_weights_proj, [D, IDX_N_HEADS], [csa_ordinal_final * D, 0])
        csa_inner_wkv_final: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wkv, [CSA_INNER_OUT_DIM, D], [csa_ordinal_final * CSA_INNER_OUT_DIM, 0])
        csa_inner_wgate_final: pl.Tensor[[CSA_INNER_OUT_DIM, D], pl.BF16] = pl.slice(csa_inner_wgate, [CSA_INNER_OUT_DIM, D], [csa_ordinal_final * CSA_INNER_OUT_DIM, 0])
        csa_inner_ape_final: pl.Tensor[[CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32] = pl.slice(csa_inner_ape, [CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], [csa_ordinal_final * CSA_COMPRESS_RATIO, 0])
        csa_inner_norm_w_final: pl.Tensor[[IDX_HEAD_DIM], pl.BF16] = pl.slice(csa_inner_norm_w, [IDX_HEAD_DIM], [csa_ordinal_final * IDX_HEAD_DIM])
        csa_compress_state_final: pl.Tensor[[CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], pl.FP32] = pl.slice(csa_compress_state, [CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM], [csa_ordinal_final * CSA_STATE_PHYSICAL_BLOCKS, 0, 0])
        csa_inner_compress_state_final: pl.Tensor[[CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], pl.FP32] = pl.slice(csa_inner_compress_state, [CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM], [csa_ordinal_final * CSA_INNER_STATE_PHYSICAL_BLOCKS, 0, 0])
        idx_kv_cache_final: pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8] = pl.slice(idx_kv_cache, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], [csa_ordinal_final * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])
        idx_kv_scale_final: pl.Tensor[[PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32] = pl.slice(idx_kv_scale, [PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], [csa_ordinal_final * PREFILL_IDX_BLOCK_NUM, 0, 0, 0])

        # §8.17.8e.2 final CSA L42 leaf-capture completion token (see in-pair CSA).
        completion_token_final_csa = pl.create_tensor(
            [NUM_MOE_WAVES, 1, 8], dtype=pl.FP32, init_value=0.0
        )
        # scope: final CSA L42 attention (tail ep = layer, CSA compact ep =
        # CSA_NUM_LAYERS-1). Reads hidden_b (last HCA pair result).
        with pl.scope():
            prefill_cp_csa_core(
                hidden_b,
                hc_attn_fn_final, hc_attn_scale_final, hc_attn_base_final, attn_norm_w_final,
                wq_a_final, wq_b_final, wq_b_scale_final, wkv_final, gamma_cq_final, gamma_ckv_final,
                freqs_cos, freqs_sin,
                csa_cmp_wkv_final, csa_cmp_wgate_final, csa_cmp_ape_final, csa_cmp_norm_w_final,
                hadamard_idx_final, idx_wq_b_final, idx_wq_b_scale_final, idx_weights_proj_final,
                csa_inner_wkv_final, csa_inner_wgate_final, csa_inner_ape_final, csa_inner_norm_w_final,
                main_state_workspace0, inner_state_workspace0,
                main_state_workspace1, inner_state_workspace1,
                csa_compress_state_final, csa_compress_state_block_table,
                csa_inner_compress_state_final, csa_inner_compress_state_block_table,
                kv_cache_final, cmp_kv_final, cmp_block_table,
                idx_kv_cache_final, idx_kv_scale_final, idx_block_table,
                segment_starts_t, segment_lengths_t,
                segment_active_lengths, owner_segments_t, predecessor_segments,
                query_position_ids, query_token_to_request,
                overlay_position_ids, overlay_token_to_request,
                overlay_active_lengths, swa_indices,
                final_segment_t, reverse_index, owner_rank_table,
                final_win_seg_src, final_win_row_src, final_slot_mapping,
                leaf_positions_input, leaf_main_slots_input,
                leaf_idx_slots_input, leaf_main_state_slots_input,
                leaf_inner_state_slots_input, leaf_num_tokens_input,
                effective_x_workspace,
                hidden_tail_window, kv_tail_window, tail_ready, tail_consumed,
                main_window, idx_window, scale_window, record_window,
                main_state_window, main_state_meta_window,
                inner_state_window, inner_state_meta_window,
                csa_compact_ready, csa_compact_consumed,
                attn_sink_final, wo_a_final, wo_b_final, wo_b_scale_final,
                x_attn_csa,
                completion_token_final_csa,
                my_rank,
                final_csa_layer,  # tail_comm_epoch
                final_csa_compact_ep,  # compact_comm_epoch_base
            )

        # Final CSA L42 MoE weight slices (global layer index).
        hc_ffn_fn_final: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32] = pl.slice(hc_ffn_fn, [MIX_HC, HC_DIM], [final_csa_layer * MIX_HC, 0])
        hc_ffn_scale_final: pl.Tensor[[3], pl.FP32] = pl.slice(hc_ffn_scale, [3], [final_csa_layer * 3])
        hc_ffn_base_final: pl.Tensor[[MIX_HC], pl.FP32] = pl.slice(hc_ffn_base, [MIX_HC], [final_csa_layer * MIX_HC])
        norm_w_final: pl.Tensor[[D], pl.BF16] = pl.slice(norm_w, [D], [final_csa_layer * D])
        gate_w_final: pl.Tensor[[N_EXPERTS_GLOBAL, D], pl.FP32] = pl.slice(gate_w, [N_EXPERTS_GLOBAL, D], [final_csa_layer * N_EXPERTS_GLOBAL, 0])
        gate_bias_final: pl.Tensor[[N_EXPERTS_GLOBAL], pl.FP32] = pl.slice(gate_bias, [N_EXPERTS_GLOBAL], [final_csa_layer * N_EXPERTS_GLOBAL])
        tid2eid_final: pl.Tensor[[VOCAB, TOPK], pl.INT32] = pl.slice(tid2eid, [VOCAB, TOPK], [final_csa_layer * VOCAB, 0])
        routed_w1_final: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w1, [N_LOCAL, MOE_INTER, D], [final_csa_layer * N_LOCAL, 0, 0])
        routed_w1_scale_final: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w1_scale, [N_LOCAL, MOE_INTER], [final_csa_layer * N_LOCAL, 0])
        routed_w3_final: pl.Tensor[[N_LOCAL, MOE_INTER, D], pl.INT8] = pl.slice(routed_w3, [N_LOCAL, MOE_INTER, D], [final_csa_layer * N_LOCAL, 0, 0])
        routed_w3_scale_final: pl.Tensor[[N_LOCAL, MOE_INTER], pl.FP32] = pl.slice(routed_w3_scale, [N_LOCAL, MOE_INTER], [final_csa_layer * N_LOCAL, 0])
        routed_w2_final: pl.Tensor[[N_LOCAL, D, MOE_INTER], pl.INT8] = pl.slice(routed_w2, [N_LOCAL, D, MOE_INTER], [final_csa_layer * N_LOCAL, 0, 0])
        routed_w2_scale_final: pl.Tensor[[N_LOCAL, D], pl.FP32] = pl.slice(routed_w2_scale, [N_LOCAL, D], [final_csa_layer * N_LOCAL, 0])
        shared_w1_final: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w1, [MOE_INTER, D], [final_csa_layer * MOE_INTER, 0])
        shared_w1_scale_final: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w1_scale, [MOE_INTER], [final_csa_layer * MOE_INTER])
        shared_w3_final: pl.Tensor[[MOE_INTER, D], pl.INT8] = pl.slice(shared_w3, [MOE_INTER, D], [final_csa_layer * MOE_INTER, 0])
        shared_w3_scale_final: pl.Tensor[[MOE_INTER], pl.FP32] = pl.slice(shared_w3_scale, [MOE_INTER], [final_csa_layer * MOE_INTER])
        shared_w2_final: pl.Tensor[[D, MOE_INTER], pl.INT8] = pl.slice(shared_w2, [D, MOE_INTER], [final_csa_layer * D, 0])
        shared_w2_scale_final: pl.Tensor[[D], pl.FP32] = pl.slice(shared_w2_scale, [D], [final_csa_layer * D])

        # scope: final CSA L42 MoE tail (epoch base 4*layer, stage base
        # 5*layer) -> hidden_a (ping-pong: even global layer writes hidden_a).
        with pl.scope():
            stage_token_final_csa = pl.create_tensor(
                [1, 1, 8], dtype=pl.FP32, init_value=0.0
            )
            stage_token_final_csa = _fwd_attention_stage_barrier_from_completion(
                completion_token_final_csa, stage_done, my_rank,
                final_csa_stage_base, stage_token_final_csa,
            )
            _fwd_moe_tail(
                x_attn_csa, overlay_active_lengths, input_ids,
                hc_ffn_fn_final, hc_ffn_scale_final, hc_ffn_base_final,
                norm_w_final, gate_w_final, gate_bias_final, tid2eid_final,
                routed_w1_final, routed_w1_scale_final,
                routed_w3_final, routed_w3_scale_final,
                routed_w2_final, routed_w2_scale_final,
                shared_w1_final, shared_w1_scale_final,
                shared_w3_final, shared_w3_scale_final,
                shared_w2_final, shared_w2_scale_final,
                recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
                routed_y_buf, combine_arrived,
                stage_done, hidden_a, completion_anchor_final_csa,
                stage_token_final_csa,
                final_csa_layer, my_rank,
                final_csa_moe_epoch_base, final_csa_stage_base,
            )

        # Odd tail: publish reads hidden_a (the L42 MoE result); the final-CSA
        # anchor carries the final all-rank wave boundary.
        publish_src_flat = hidden_a_flat
        publish_anchor = completion_anchor_final_csa
    else:
        # Even tail: publish reads hidden_b (last HCA pair result); the HCA
        # anchor from the last pair carries the final wave boundary.
        publish_src_flat = hidden_b_flat
        publish_anchor = completion_anchor_hca

    # --- Clear MoE signals once + publish -> pre_hc_hidden_out, then HC head +
    # final RMSNorm -> hidden_out (§7.5 + §8.7, after the loop). Mirrors the
    # baseline prefill_fwd.py tail: the publish SPMD writes the last hidden
    # (hidden_a for odd tail, hidden_b for even tail) into pre_hc_hidden_out,
    # then the inlined hc_head collapses the [HC_MULT, D] mix into one [D] row
    # and the final rms_norm normalizes it into the BF16 hidden_out that the
    # host's lm_head_test consumes. Both pl.Out tensors are written in place.
    with pl.scope():
        clear_moe_signals(publish_anchor, arrived, data_arrived, combine_arrived)

        wave_blocks = (T // COPY_TOKEN_TILE) * HC_MULT
        with pl.spmd(
            NUM_MOE_WAVES * wave_blocks,
            name_hint="publish_pre_hc_hidden_out",
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
                    pre_hc_hidden_out_flat[
                        row : row + 1,
                        hc_lane : hc_lane + 1,
                        0:D,
                    ] = pl.slice(
                        publish_src_flat,
                        [1, 1, D],
                        [row, hc_lane, 0],
                    )
                else:
                    pre_hc_hidden_out_flat[
                        row : row + 1,
                        hc_lane : hc_lane + 1,
                        0:D,
                    ] = pl.full([1, 1, D], dtype=pl.FP32, value=0.0)

        # HC head + final RMSNorm: collapse the [HC_MULT, D] hyper-connection
        # mix to one [D] row and normalize into the BF16 hidden_out. The
        # pre_hc_hidden_out slab is NUM_MOE_WAVES * T rows = LOCAL_ROWS, so the
        # hc_head's T_DYN extent binds to LOCAL_ROWS. The intermediate
        # hidden_head is the hc_head BF16 output and the rms_norm input.
        pre_hc_view = pl.reshape(
            pre_hc_hidden_out_flat, [LOCAL_ROWS, HC_MULT, D]
        )
        hidden_head = pl.create_tensor([LOCAL_ROWS, D], dtype=pl.BF16)
        with pl.scope():
            hc_head(
                pre_hc_view, hc_head_fn, hc_head_scale, hc_head_base, hidden_head
            )
            rms_norm(hidden_head, final_norm_w, hidden_out)


# ---------------------------------------------------------------------------
# Host launcher
# ---------------------------------------------------------------------------
@pl.jit.host
def l3_prefill_cp_fwd(
    x_hc: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, HC_MULT, D],
        pl.FP32,
    ],
    # SWA attention weights (layer-stacked, shared across ranks).
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
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, FWD_NUM_LAYERS * ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    # Compressed KV cache: FWD_NUM_LAYERS per-layer pools (rank-leading).
    cmp_kv: pl.InOut[
        pl.Tensor[
            [CP_SIZE, FWD_NUM_LAYERS * PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
            pl.BF16,
        ]
    ],
    attn_sink: pl.Tensor[[FWD_NUM_LAYERS * H], pl.FP32],
    wo_a: pl.Tensor[[FWD_NUM_LAYERS * O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[FWD_NUM_LAYERS * D, O_GROUPS * O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[FWD_NUM_LAYERS * D], pl.FP32],
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
    # Shared CP metadata needed by CSA/HCA cores (beyond SWA's set).
    segment_active_lengths: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    owner_segments_t: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    final_segment_t: pl.Tensor[[1], pl.INT32],
    # --- HCA type-specific (layers 3 and 5) -----------------------------
    # Compact compressor weights: stacked by HCA_NUM_LAYERS on axis 0
    # (no rank axis). Host passes the full stacked root; child slices type.
    hca_cmp_wkv: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_wgate: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM, D], pl.BF16],
    hca_cmp_ape: pl.Tensor[[HCA_NUM_LAYERS * HCA_COMPRESS_RATIO, HEAD_DIM], pl.FP32],
    hca_cmp_norm_w: pl.Tensor[[HCA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    # Persistent state: stacked by HCA_NUM_LAYERS on axis 1 (rank-leading).
    hca_compress_state: pl.InOut[
        pl.Tensor[
            [CP_SIZE, HCA_NUM_LAYERS * HCA_STATE_PHYSICAL_BLOCKS, HCA_STATE_BLOCK_SIZE, HCA_COMPRESS_STATE_DIM],
            pl.FP32,
        ]
    ],
    hca_compress_state_block_table: pl.Tensor[
        [CP_SIZE, HCA_STATE_MAX_BLOCKS], pl.INT32
    ],
    segment_tail_positions: pl.Tensor[[NUM_SEGMENTS, TAIL_ROWS], pl.INT32],
    snapshot_positions: pl.Tensor[[CP_SIZE, LOCAL_PARTS, TAIL_ROWS], pl.INT32],
    snapshot_valid: pl.Tensor[[CP_SIZE, LOCAL_PARTS], pl.INT32],
    owner_part_table: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    cmp_indices: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, TAIL_ROWS, IDX_TOPK], pl.INT32
    ],
    # --- CSA type-specific (layers 2 and 4) -----------------------------
    # Compact/indexer weights: stacked by CSA_NUM_LAYERS on axis 0
    # (no rank axis). Host passes the full stacked root; child slices type.
    csa_cmp_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_MAIN_OUT_DIM, D], pl.BF16],
    csa_cmp_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_MAIN_OUT_DIM], pl.FP32],
    csa_cmp_norm_w: pl.Tensor[[CSA_NUM_LAYERS * HEAD_DIM], pl.BF16],
    hadamard_idx: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    idx_wq_b: pl.Tensor[[CSA_NUM_LAYERS * Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[CSA_NUM_LAYERS * IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    idx_weights_proj: pl.Tensor[[CSA_NUM_LAYERS * D, IDX_N_HEADS], pl.BF16],
    csa_inner_wkv: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_wgate: pl.Tensor[[CSA_NUM_LAYERS * CSA_INNER_OUT_DIM, D], pl.BF16],
    csa_inner_ape: pl.Tensor[[CSA_NUM_LAYERS * CSA_COMPRESS_RATIO, CSA_INNER_OUT_DIM], pl.FP32],
    csa_inner_norm_w: pl.Tensor[[CSA_NUM_LAYERS * IDX_HEAD_DIM], pl.BF16],
    # Persistent state/caches: stacked by CSA_NUM_LAYERS on axis 1.
    csa_compress_state: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_NUM_LAYERS * CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
            pl.FP32,
        ]
    ],
    csa_inner_compress_state: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_NUM_LAYERS * CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
            pl.FP32,
        ]
    ],
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_NUM_LAYERS * PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[
            [CP_SIZE, CSA_NUM_LAYERS * PREFILL_IDX_BLOCK_NUM, BLOCK_SIZE, 1, 1], pl.FP32
        ]
    ],
    csa_compress_state_block_table: pl.Tensor[
        [CP_SIZE, CSA_MAIN_STATE_MAX_BLOCKS], pl.INT32
    ],
    csa_inner_compress_state_block_table: pl.Tensor[
        [CP_SIZE, CSA_INNER_STATE_MAX_BLOCKS], pl.INT32
    ],
    idx_block_table: pl.Tensor[[CP_SIZE, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    main_state_workspace0: pl.Tensor[
        [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace0: pl.Tensor[
        [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    main_state_workspace1: pl.Tensor[
        [CP_SIZE, CSA_STATE_PHYSICAL_BLOCKS, CSA_MAIN_STATE_BLOCK_SIZE, CSA_MAIN_STATE_DIM],
        pl.FP32,
    ],
    inner_state_workspace1: pl.Tensor[
        [CP_SIZE, CSA_INNER_STATE_PHYSICAL_BLOCKS, CSA_INNER_STATE_BLOCK_SIZE, CSA_INNER_STATE_DIM],
        pl.FP32,
    ],
    effective_x_workspace: pl.Tensor[
        [CP_SIZE, CSA_LOCAL_LEAVES * T, D], pl.BF16
    ],
    segment_lengths_t: pl.Tensor[[NUM_SEGMENTS], pl.INT32],
    leaf_positions_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT32
    ],
    leaf_main_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_idx_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_main_state_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_inner_state_slots_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES, T], pl.INT64
    ],
    leaf_num_tokens_input: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, CSA_MAX_COMPRESS_LEAVES], pl.INT32
    ],
    cmp_block_table: pl.Tensor[[CP_SIZE, PREFILL_CMP_MAX_BLOCKS], pl.INT32],
    # MoE weights (layer-stacked, rank-sliced at launch).
    hc_ffn_fn: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * 3], pl.FP32],
    hc_ffn_base: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * D], pl.BF16],
    gate_w: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL, D], pl.FP32],
    gate_bias: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * N_EXPERTS_GLOBAL], pl.FP32],
    tid2eid: pl.Tensor[[N_RANKS, FWD_NUM_LAYERS * VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[
        [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T], pl.INT64
    ],
    routed_w1: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w1_scale: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w3: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER, D], pl.INT8
    ],
    routed_w3_scale: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * N_LOCAL, MOE_INTER], pl.FP32
    ],
    routed_w2: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D, MOE_INTER], pl.INT8
    ],
    routed_w2_scale: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * N_LOCAL, D], pl.FP32
    ],
    shared_w1: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8
    ],
    shared_w1_scale: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32
    ],
    shared_w3: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * MOE_INTER, D], pl.INT8
    ],
    shared_w3_scale: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * MOE_INTER], pl.FP32
    ],
    shared_w2: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * D, MOE_INTER], pl.INT8
    ],
    shared_w2_scale: pl.Tensor[
        [N_RANKS, FWD_NUM_LAYERS * D], pl.FP32
    ],
    # Phase 3 final-tail weights and outputs. The HC head + final RMSNorm run
    # inlined in the FWD child; lm_head_test is host-launched per rank over
    # hidden_out. hc_head_fn/scale/base and final_norm_w are replicated
    # (rank-leading [CP_SIZE]); lm_head_weight is vocab-sharded
    # (rank owns shard rank % LM_HEAD_TP_SIZE); logit_row_indices selects the
    # rows each rank projects (§8.8).
    hc_head_fn: pl.Tensor[[CP_SIZE, HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[CP_SIZE, 1], pl.FP32],
    hc_head_base: pl.Tensor[[CP_SIZE, HC_MULT], pl.FP32],
    final_norm_w: pl.Tensor[[CP_SIZE, D], pl.BF16],
    lm_head_weight: pl.Tensor[
        [CP_SIZE, LM_HEAD_VOCAB_PER_TP, D], pl.BF16
    ],
    logit_row_indices: pl.Tensor[
        [CP_SIZE, LM_HEAD_MAX_LOGIT_ROWS], pl.INT32
    ],
    pre_hc_hidden_out: pl.Out[
        pl.Tensor[
            [CP_SIZE, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D], pl.FP32
        ]
    ],
    hidden_out: pl.Out[pl.Tensor[[CP_SIZE, LOCAL_ROWS, D], pl.BF16]],
    logits: pl.Out[
        pl.Tensor[
            [CP_SIZE, LM_HEAD_MAX_LOGIT_ROWS, LM_HEAD_VOCAB], pl.FP32
        ]
    ],
):
    """Launch one CP FWD child per rank owning five window domains
    (shared tail exchange, baseline MoE, layer stage synchronization, HCA
    compact, and CSA compact/index/state), then one LM-head child per rank
    over the FWD child's hidden_out. The FWD child inlines the HC head and
    final RMSNorm; only lm_head_test is a separate host-to-child launch in a
    second rank loop (mirrors baseline prefill_fwd.py)."""
    # Domain 1: shared tail exchange (SWA + CSA + HCA reuse one bank under
    # monotonic tail_comm_epoch). SWA's zigzag uses kv_tail_window only; the
    # dual-tail exchange also needs a hidden-tail window.
    kv_tail_window_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    hidden_tail_window_buf = pld.alloc_window_buffer(
        [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
    )
    tail_ready_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)
    tail_consumed_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    # Domain 2: baseline MoE window bank (reused across layers; monotonic
    # epoch).
    recv_meta_buf = pld.alloc_window_buffer([N_RANKS, N_LOCAL], dtype=pl.INT32)
    recv_x_buf = pld.alloc_window_buffer([N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
    recv_aux_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
    )
    recv_route_buf = pld.alloc_window_buffer(
        [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
    )
    arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    data_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)
    routed_y_buf_buf = pld.alloc_window_buffer([N_ROUTES, D], dtype=pl.BF16)
    combine_arrived_buf = pld.alloc_window_buffer([N_RANKS, 1], dtype=pl.INT32)

    # Domain 3: layer stage synchronization (monotonic counter, 1..20).
    stage_done_buf = pld.alloc_window_buffer([CP_SIZE, 1], dtype=pl.INT32)

    # Domain 4: HCA compact exchange (one bank reused by HCA layers; distinct
    # from tail/stage/MoE signals; compact_comm_epoch).
    cmp_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
    )
    cmp_meta_window_buf = pld.alloc_window_buffer(
        [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32
    )
    state_window_buf = pld.alloc_window_buffer(
        [STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM], dtype=pl.FP32
    )
    state_meta_window_buf = pld.alloc_window_buffer(
        [CP_SIZE, STATE_META_DIM], dtype=pl.INT32
    )
    hca_compact_ready_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )
    hca_compact_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )

    # Domain 5: CSA compact/index/state exchange (one bank reused by CSA
    # layers; distinct from HCA/tail/stage/MoE signals; compact_comm_epoch).
    main_window_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM], dtype=pl.BF16
    )
    idx_window_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8
    )
    scale_window_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, SCALE_TILE_COLS], dtype=pl.FP32
    )
    record_window_buf = pld.alloc_window_buffer(
        [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32
    )
    main_state_window_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, CSA_MAIN_STATE_DIM], dtype=pl.FP32
    )
    main_state_meta_window_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
    )
    inner_state_window_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, CSA_INNER_STATE_DIM], dtype=pl.FP32
    )
    inner_state_meta_window_buf = pld.alloc_window_buffer(
        [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
    )
    csa_compact_ready_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )
    csa_compact_consumed_buf = pld.alloc_window_buffer(
        [CP_SIZE, 1], dtype=pl.INT32
    )

    # Domain 6: LM-head TP gather/combine (mirrors baseline prefill_fwd.py
    # Domain "lm_head"). The LM head owns every window and counter it touches:
    # a peer routes into logits_window while still reading its own
    # hidden_window, and the barrier counters stay independent of the MoE
    # epoch protocol (done_epoch = LM_HEAD_COMM_EPOCH, independent of MoE).
    # Windows are group-local: hidden_window holds one slot per group member
    # ([GROUP_LOGIT_ROWS, D]), and every card receives only its own
    # full-vocabulary logits ([MAX_LOGIT_ROWS, VOCAB]).
    #
    # §8.17.2: the buffers are always allocated (cheap metadata at graph
    # build time; the SSA verifier traces the dead `if not FWD_ONLY` branch
    # and requires every referenced name to be defined). Only the second
    # rank loop (the lm_head_test launch) is guarded out under --fwd-only;
    # the buffers simply go unused.
    lm_head_hidden_window_buf = pld.alloc_window_buffer(
        LM_HEAD_GROUP_LOGIT_ROWS * D * 2
    )
    lm_head_logits_window_buf = pld.alloc_window_buffer(
        LM_HEAD_MAX_LOGIT_ROWS * LM_HEAD_VOCAB * 4
    )
    lm_head_hidden_done_buf = pld.alloc_window_buffer(
        [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
    )
    lm_head_logits_done_buf = pld.alloc_window_buffer(
        [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
    )

    for rank in pl.range(pld.world_size()):
        kv_tail_window = pld.window(
            kv_tail_window_buf, [CP_TAIL_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        hidden_tail_window = pld.window(
            hidden_tail_window_buf, [CP_TAIL_WINDOW_ROWS, D], dtype=pl.BF16
        )
        tail_ready = pld.window(tail_ready_buf, [CP_SIZE, 1], dtype=pl.INT32)
        tail_consumed = pld.window(
            tail_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        recv_meta = pld.window(recv_meta_buf, [N_RANKS, N_LOCAL], dtype=pl.INT32)
        recv_x = pld.window(recv_x_buf, [N_LOCAL * RECV_MAX, D], dtype=pl.INT8)
        recv_aux = pld.window(
            recv_aux_buf, [N_LOCAL * RECV_MAX, AUX_PAD], dtype=pl.FP32
        )
        recv_route = pld.window(
            recv_route_buf, [N_LOCAL * RECV_MAX, IDX_PAD], dtype=pl.INT32
        )
        arrived = pld.window(arrived_buf, [N_RANKS, 1], dtype=pl.INT32)
        data_arrived = pld.window(
            data_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        routed_y_buf = pld.window(routed_y_buf_buf, [N_ROUTES, D], dtype=pl.BF16)
        combine_arrived = pld.window(
            combine_arrived_buf, [N_RANKS, 1], dtype=pl.INT32
        )
        stage_done = pld.window(stage_done_buf, [CP_SIZE, 1], dtype=pl.INT32)
        # Domain 4: HCA compact windows.
        cmp_window = pld.window(
            cmp_window_buf, [CMP_WINDOW_ROWS, HEAD_DIM], dtype=pl.BF16
        )
        cmp_meta_window = pld.window(
            cmp_meta_window_buf, [CMP_WINDOW_ROWS, CMP_META_DIM], dtype=pl.INT32
        )
        state_window = pld.window(
            state_window_buf, [STATE_WINDOW_ROWS, HCA_COMPRESS_STATE_DIM],
            dtype=pl.FP32
        )
        state_meta_window = pld.window(
            state_meta_window_buf, [CP_SIZE, STATE_META_DIM], dtype=pl.INT32
        )
        hca_compact_ready = pld.window(
            hca_compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        hca_compact_consumed = pld.window(
            hca_compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        # Domain 5: CSA compact windows.
        main_window = pld.window(
            main_window_buf, [RECORDS_PER_WINDOW, CSA_MAIN_OUT_DIM],
            dtype=pl.BF16
        )
        idx_window = pld.window(
            idx_window_buf, [RECORDS_PER_WINDOW, IDX_HEAD_DIM], dtype=pl.INT8
        )
        scale_window = pld.window(
            scale_window_buf, [RECORDS_PER_WINDOW, SCALE_TILE_COLS],
            dtype=pl.FP32
        )
        record_window = pld.window(
            record_window_buf, [RECORDS_PER_WINDOW, META_DIM], dtype=pl.INT32
        )
        main_state_window = pld.window(
            main_state_window_buf,
            [STATE_RECORDS_PER_WINDOW, CSA_MAIN_STATE_DIM], dtype=pl.FP32
        )
        main_state_meta_window = pld.window(
            main_state_meta_window_buf,
            [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
        )
        inner_state_window = pld.window(
            inner_state_window_buf,
            [STATE_RECORDS_PER_WINDOW, CSA_INNER_STATE_DIM], dtype=pl.FP32
        )
        inner_state_meta_window = pld.window(
            inner_state_meta_window_buf,
            [STATE_RECORDS_PER_WINDOW, STATE_META_DIM], dtype=pl.INT32
        )
        csa_compact_ready = pld.window(
            csa_compact_ready_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        csa_compact_consumed = pld.window(
            csa_compact_consumed_buf, [CP_SIZE, 1], dtype=pl.INT32
        )
        # Shared weights are passed directly; rank-local tensors are
        # [rank]-sliced; rank-local InOut roots are [rank]-sliced.
        prefill_cp_fwd(
            x_hc[rank],
            hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w,
            wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv,
            freqs_cos, freqs_sin, kv_cache[rank], cmp_kv[rank],
            attn_sink, wo_a, wo_b, wo_b_scale,
            segment_starts_t, predecessor_segments[rank],
            query_position_ids[rank], query_token_to_request[rank],
            overlay_position_ids[rank], overlay_token_to_request[rank],
            overlay_active_lengths[rank], swa_indices[rank],
            reverse_index, owner_rank_table,
            final_win_seg_src, final_win_row_src, final_slot_mapping,
            segment_active_lengths[rank], owner_segments_t[rank],
            final_segment_t,
            # HCA type-specific (layers 3 and 5).
            hca_cmp_wkv, hca_cmp_wgate, hca_cmp_ape, hca_cmp_norm_w,
            hca_compress_state[rank], hca_compress_state_block_table[rank],
            segment_tail_positions, snapshot_positions[rank],
            snapshot_valid[rank], owner_part_table, cmp_indices[rank],
            # CSA type-specific (layers 2 and 4).
            csa_cmp_wkv, csa_cmp_wgate, csa_cmp_ape, csa_cmp_norm_w,
            hadamard_idx, idx_wq_b, idx_wq_b_scale, idx_weights_proj,
            csa_inner_wkv, csa_inner_wgate, csa_inner_ape, csa_inner_norm_w,
            csa_compress_state[rank], csa_inner_compress_state[rank],
            idx_kv_cache[rank], idx_kv_scale[rank],
            csa_compress_state_block_table[rank],
            csa_inner_compress_state_block_table[rank],
            idx_block_table[rank],
            main_state_workspace0[rank], inner_state_workspace0[rank],
            main_state_workspace1[rank], inner_state_workspace1[rank],
            effective_x_workspace[rank],
            segment_lengths_t,
            leaf_positions_input[rank], leaf_main_slots_input[rank],
            leaf_idx_slots_input[rank], leaf_main_state_slots_input[rank],
            leaf_inner_state_slots_input[rank], leaf_num_tokens_input[rank],
            cmp_block_table[rank],
            # Communication windows.
            kv_tail_window, hidden_tail_window, tail_ready, tail_consumed,
            cmp_window, cmp_meta_window, state_window, state_meta_window,
            hca_compact_ready, hca_compact_consumed,
            main_window, idx_window, scale_window, record_window,
            main_state_window, main_state_meta_window,
            inner_state_window, inner_state_meta_window,
            csa_compact_ready, csa_compact_consumed,
            # MoE weights (rank-sliced).
            hc_ffn_fn[rank], hc_ffn_scale[rank], hc_ffn_base[rank],
            norm_w[rank], gate_w[rank], gate_bias[rank], tid2eid[rank],
            input_ids[rank],
            routed_w1[rank], routed_w1_scale[rank],
            routed_w3[rank], routed_w3_scale[rank],
            routed_w2[rank], routed_w2_scale[rank],
            shared_w1[rank], shared_w1_scale[rank],
            shared_w3[rank], shared_w3_scale[rank],
            shared_w2[rank], shared_w2_scale[rank],
            # MoE comm windows.
            recv_meta, recv_x, recv_aux, recv_route, arrived, data_arrived,
            routed_y_buf, combine_arrived,
            stage_done,
            # Phase 3 final-tail weights (rank-sliced).
            hc_head_fn[rank], hc_head_scale[rank], hc_head_base[rank],
            final_norm_w[rank],
            # Phase 3 final-tail outputs (rank-sliced pl.Out).
            pre_hc_hidden_out[rank], hidden_out[rank],
            rank,
            device=rank,
        )

    # Phase 3 second rank loop: host-launched lm_head_test over each rank's
    # hidden_out (mirrors baseline prefill_fwd.py). The four LM-head windows
    # are group-local; group_base/tp_rank/done_epoch partition the CP world
    # into CP // TP tensor-parallel groups. Each card holds vocab shard
    # rank % TP and serves only its own group.
    for rank in pl.range(pld.world_size()):
        hidden_window = pld.window(
            lm_head_hidden_window_buf,
            [LM_HEAD_GROUP_LOGIT_ROWS, D], dtype=pl.BF16
        )
        hidden_done = pld.window(
            lm_head_hidden_done_buf,
            [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        logits_window = pld.window(
            lm_head_logits_window_buf,
            [LM_HEAD_MAX_LOGIT_ROWS, LM_HEAD_VOCAB], dtype=pl.FP32
        )
        logits_done = pld.window(
            lm_head_logits_done_buf,
            [LM_HEAD_TP_SIZE, 1], dtype=pl.INT32
        )
        lm_head_test(
            hidden_out[rank], lm_head_weight[rank], logit_row_indices[rank],
            logits[rank],
            hidden_window, hidden_done, logits_window, logits_done,
            rank // LM_HEAD_TP_SIZE * LM_HEAD_TP_SIZE,  # group_base
            rank % LM_HEAD_TP_SIZE,                      # tp_rank
            LM_HEAD_COMM_EPOCH,                          # done_epoch
            device=rank,
        )


# ---------------------------------------------------------------------------
# TensorSpec composition
# ---------------------------------------------------------------------------
# Names of SWA attention weights that must be layer-stacked for the FWD child.
_SWA_ATTN_WEIGHT_NAMES = (
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
)
# Names of SWA shared metadata (not layer-stacked; one copy for all layers).
_SWA_SHARED_NAMES = (
    "freqs_cos", "freqs_sin", "segment_starts_t",
    "predecessor_segments", "query_position_ids", "query_token_to_request",
    "overlay_position_ids", "overlay_token_to_request",
    "overlay_active_lengths", "swa_indices",
    "reverse_index", "owner_rank_table",
    "final_win_seg_src", "final_win_row_src", "final_slot_mapping",
)
# MoE weight names to layer-stack (drop x_hc/x_next/input_ids/scalars).
_MOE_WEIGHT_NAMES = (
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
)
# HCA type-specific compact weights and state (stacked by HCA_NUM_LAYERS).
_HCA_COMPACT_WEIGHT_NAMES = (
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
)
# HCA persistent state and block table (one root, sliced by type order).
_HCA_STATE_NAMES = (
    "hca_compress_state", "hca_compress_state_block_table",
)
# HCA-specific metadata (single copy; not stacked).
_HCA_METADATA_NAMES = (
    "segment_tail_positions", "snapshot_positions", "snapshot_valid",
    "owner_part_table", "cmp_indices",
)
# CSA type-specific compact weights and state (stacked by CSA_NUM_LAYERS).
_CSA_COMPACT_WEIGHT_NAMES = (
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "hadamard_idx", "idx_wq_b", "idx_wq_b_scale", "idx_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
)
# CSA persistent state, cache, and block tables (one root per kind).
# The CSA builder emits bare names (compress_state, inner_compress_state,
# compress_state_block_table, inner_compress_state_block_table); the FWD
# child uses csa_ prefixes to avoid colliding with HCA's compress_state.
# idx_kv_cache/idx_kv_scale/idx_block_table keep their bare names.
_CSA_STATE_NAMES = (
    "csa_compress_state", "csa_inner_compress_state",
    "idx_kv_cache", "idx_kv_scale",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table", "idx_block_table",
)
# Map FWD child name -> CSA builder spec name (for the 4 renamed state specs).
_CSA_STATE_RENAME_MAP = {
    "csa_compress_state": "compress_state",
    "csa_inner_compress_state": "inner_compress_state",
    "csa_compress_state_block_table": "compress_state_block_table",
    "csa_inner_compress_state_block_table": "inner_compress_state_block_table",
}
# CSA rank-local reusable workspaces (not stacked; serial layers reuse them).
_CSA_WORKSPACE_NAMES = (
    "main_state_workspace0", "inner_state_workspace0",
    "main_state_workspace1", "inner_state_workspace1",
    "effective_x_workspace",
)
# CSA-specific metadata (single copy; not stacked).
_CSA_METADATA_NAMES = (
    "segment_lengths_t",
    "leaf_positions_input", "leaf_main_slots_input",
    "leaf_idx_slots_input", "leaf_main_state_slots_input",
    "leaf_inner_state_slots_input", "leaf_num_tokens_input",
)

# Host argument order (must match l3_prefill_cp_fwd signature).
# Order convention: x_hc -> common attn weights -> freqs/kv_cache/cmp_kv ->
# common attn tail -> shared CP metadata -> HCA compact+state+metadata ->
# CSA compact+state+workspaces+metadata -> cmp_block_table -> MoE weights ->
# input_ids -> Phase 3 tail weights + outputs. Scalars are passed by the host
# rank loop, not here.
FWD_HOST_ARG_ORDER = (
    "x_hc",
    # common attention weights (layer-stacked, FWD_NUM_LAYERS).
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "cmp_kv",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    # shared CP metadata (one copy).
    "segment_starts_t", "predecessor_segments",
    "query_position_ids", "query_token_to_request",
    "overlay_position_ids", "overlay_token_to_request",
    "overlay_active_lengths", "swa_indices",
    "reverse_index", "owner_rank_table",
    "final_win_seg_src", "final_win_row_src", "final_slot_mapping",
    # shared CP metadata needed by CSA/HCA cores (beyond SWA's set).
    "segment_active_lengths", "owner_segments_t", "final_segment_t",
    # HCA type-specific (layer 3): compact weights + state + metadata.
    "hca_cmp_wkv", "hca_cmp_wgate", "hca_cmp_ape", "hca_cmp_norm_w",
    "hca_compress_state", "hca_compress_state_block_table",
    "segment_tail_positions", "snapshot_positions", "snapshot_valid",
    "owner_part_table", "cmp_indices",
    # CSA type-specific (layer 2): compact weights + state + workspaces + metadata.
    "csa_cmp_wkv", "csa_cmp_wgate", "csa_cmp_ape", "csa_cmp_norm_w",
    "hadamard_idx", "idx_wq_b", "idx_wq_b_scale", "idx_weights_proj",
    "csa_inner_wkv", "csa_inner_wgate", "csa_inner_ape", "csa_inner_norm_w",
    "csa_compress_state", "csa_inner_compress_state",
    "idx_kv_cache", "idx_kv_scale",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table", "idx_block_table",
    "main_state_workspace0", "inner_state_workspace0",
    "main_state_workspace1", "inner_state_workspace1",
    "effective_x_workspace",
    "segment_lengths_t",
    "leaf_positions_input", "leaf_main_slots_input",
    "leaf_idx_slots_input", "leaf_main_state_slots_input",
    "leaf_inner_state_slots_input", "leaf_num_tokens_input",
    # shared cmp block table (HCA and CSA compact both index it).
    "cmp_block_table",
    # MoE weights (layer-stacked, FWD_NUM_LAYERS; rank-sliced at launch).
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale",
    "shared_w1", "shared_w1_scale", "shared_w3", "shared_w3_scale",
    "shared_w2", "shared_w2_scale",
    # Phase 3 final-tail weights and outputs (§8.7 + §8.8).
    "hc_head_fn", "hc_head_scale", "hc_head_base", "final_norm_w",
    "lm_head_weight", "logit_row_indices",
    "pre_hc_hidden_out", "hidden_out", "logits",
)


def _stack_spec(spec: TensorSpec, num_layers: int) -> TensorSpec:
    """Stack a single-layer weight spec along a new leading layer axis.

    For a spec with shape [N_RANKS, unit, ...] the result is
    [N_RANKS, num_layers * unit, ...]. For a spec with shape [unit, ...]
    (no leading N_RANKS) the result is [num_layers * unit, ...]."""
    shape = list(spec.shape)
    if shape[0] == N_RANKS:
        shape[1] = num_layers * shape[1]
    else:
        shape[0] = num_layers * shape[0]
    return TensorSpec(
        spec.name, shape, spec.dtype,
        init_value=spec.init_value, is_output=spec.is_output,
        resident=spec.resident,
    )


def _make_stacked_swa_attn_spec(name: str, base_spec: TensorSpec,
                                 num_layers: int) -> TensorSpec:
    """Build a layer-stacked SWA attention weight spec from the single-layer
    base. The SWA base specs do NOT have a leading N_RANKS axis, so stacking
    is [num_layers * unit, ...]."""
    raw = base_spec.create_tensor()
    stacked = raw.clone()
    for _ in range(num_layers - 1):
        stacked = torch.cat([stacked, raw.clone()], dim=0)
    shape = [num_layers * base_spec.shape[0]] + list(base_spec.shape[1:])
    return TensorSpec(
        name, shape, base_spec.dtype, init_value=stacked,
        is_output=base_spec.is_output, resident=base_spec.resident,
    )


def _make_stacked_moe_spec(name: str, base_spec: TensorSpec,
                            num_layers: int) -> TensorSpec:
    """Build a layer-stacked MoE weight spec from the single-layer base.
    The MoE base specs have shape [N_RANKS, unit, ...]; stacking produces
    [N_RANKS, num_layers * unit, ...]."""
    raw = base_spec.create_tensor()
    stacked = raw.clone()
    for _ in range(num_layers - 1):
        stacked = torch.cat([stacked, raw.clone()], dim=1)
    shape = [base_spec.shape[0], num_layers * base_spec.shape[1]] + list(
        base_spec.shape[2:]
    )
    return TensorSpec(
        name, shape, base_spec.dtype, init_value=stacked,
        is_output=base_spec.is_output, resident=base_spec.resident,
    )


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


def _rename_spec(base_spec: TensorSpec, new_name: str) -> TensorSpec:
    """Return a copy of ``base_spec`` under a new name, preserving shape,
    dtype, init_value, is_output, and resident."""
    return TensorSpec(
        new_name, list(base_spec.shape), base_spec.dtype,
        init_value=base_spec.init_value, is_output=base_spec.is_output,
        resident=base_spec.resident,
    )


def _stack_type_spec(
    name: str, base_spec: TensorSpec, num_type_layers: int,
    *, rank_leading: bool,
) -> TensorSpec:
    """Stack a single-layer type-specific spec along the type-ordinal axis.

    ``rank_leading`` is decided by the caller: a persistent per-rank root
    (leading ``N_RANKS``/CP_SIZE axis) is stacked on axis 1 ->
    ``[N_RANKS, num_type_layers * unit, ...]``; a compact compressor weight
    (no rank axis) is stacked on axis 0 ->
    ``[num_type_layers * unit, ...]``. The init value is tiled so each
    type-ordinal slice is an independent copy of the base.

    The rank-leading flag is explicit because the previous ``shape[0] ==
    N_RANKS`` heuristic collided with CSA_COMPRESS_RATIO (==4) at CP4,
    mis-stacking ``cmp_ape`` on axis 1 and leaving axis 0 at 4."""
    raw = base_spec.create_tensor()
    stacked = raw.clone()
    for _ in range(num_type_layers - 1):
        if rank_leading:
            stacked = torch.cat([stacked, raw.clone()], dim=1)
        else:
            stacked = torch.cat([stacked, raw.clone()], dim=0)
    if rank_leading:
        shape = [base_spec.shape[0], num_type_layers * base_spec.shape[1]] + list(
            base_spec.shape[2:]
        )
    else:
        shape = [num_type_layers * base_spec.shape[0]] + list(
            base_spec.shape[1:]
        )
    return TensorSpec(
        name, shape, base_spec.dtype, init_value=stacked,
        is_output=base_spec.is_output, resident=base_spec.resident,
    )


def build_tensor_specs(cp_size: int = CP_SIZE):
    """Compose the CP FWD attention and MoE specs for the generic schedule.

    Common attention/MoE weights and the raw/compressed KV caches are stacked
    by the FWD layer index (``FWD_NUM_LAYERS``); type-specific compressor
    weights and persistent state are stacked by the type order index
    (``HCA_NUM_LAYERS`` / ``CSA_NUM_LAYERS`` == ``PAIR_COUNT``, one copy per
    CSA/HCA pair). Shared CP metadata, block tables, and RoPE tables are
    single copies.
    """
    swa_specs, ctx = build_swa_tensor_specs(cp_size)
    swa_by_name = {s.name: s for s in swa_specs}

    # HCA / CSA standalone spec builders return a flat list (no ctx). The
    # type-specific names below are renamed with hca_/csa_ prefixes so they
    # do not collide with the common SWA weight names.
    hca_specs = build_hca_tensor_specs(cp_size)
    hca_by_name = {s.name: s for s in hca_specs}
    csa_specs = build_csa_tensor_specs(cp_size)
    csa_by_name = {s.name: s for s in csa_specs}

    specs_by_name = {}

    # x_hc: the input hidden (single-layer shape, rank-leading).
    specs_by_name["x_hc"] = swa_by_name["x_hc"]

    # Common attention weights: layer-stack the single-layer SWA base. These
    # weights are shared by all four attention layers (SWA/CSA/HCA use the
    # same QKV/RoPE/output-projection weight layout), so a single FWD-stacked
    # pool serves every layer; the child slices by the global layer index.
    for name in _SWA_ATTN_WEIGHT_NAMES:
        base = swa_by_name[name]
        specs_by_name[name] = _make_stacked_swa_attn_spec(
            name, base, FWD_NUM_LAYERS
        )

    # Shared CP metadata: pass through unchanged (rank-leading where needed).
    # CSA/HCA standalone fixtures share the same zero-history segment/query/
    # overlay/swa_indices values as SWA; one copy serves all four layers.
    for name in _SWA_SHARED_NAMES:
        specs_by_name[name] = swa_by_name[name]

    # Shared CP metadata needed by CSA/HCA cores but absent from SWA's spec
    # set (segment_active_lengths, owner_segments_t, final_segment_t). HCA
    # and CSA builders emit identical-shaped, identical-value specs for
    # these; take from the HCA builder (one copy serves both layer 2 and 3).
    for name in ("segment_active_lengths", "owner_segments_t", "final_segment_t"):
        specs_by_name[name] = hca_by_name[name]

    # kv_cache: FWD_NUM_LAYERS per-layer pools concatenated on the per-rank
    # block axis so the child can slice a 4D [ORI_MAX_BLOCKS, ...] view per
    # layer and the inline cores' pl.reshape([ORI_CACHE_ROWS, HEAD_DIM]) stays
    # statically inferable. Layer 0 is initialised from the SWA fixture; the
    # remaining layers are zero slabs (InOut scratch the kernel fills).
    base_cache = swa_by_name["kv_cache"].create_tensor()
    cache_slabs = [base_cache] + [
        torch.zeros_like(base_cache) for _ in range(FWD_NUM_LAYERS - 1)
    ]
    cache_fwd = torch.cat(cache_slabs, dim=1)
    specs_by_name["kv_cache"] = TensorSpec(
        "kv_cache",
        [cp_size, FWD_NUM_LAYERS * ORI_MAX_BLOCKS, BLOCK_ROWS, 1, HEAD_DIM],
        torch.bfloat16, init_value=cache_fwd, is_output=True,
    )

    # cmp_kv: compressed KV cache, FWD_NUM_LAYERS per-layer pools (every
    # attention layer owns a compressed-KV slice). Stacked on the per-rank
    # block axis the same way as kv_cache. Mirrors baseline prefill_fwd.py.
    base_cmp_kv = hca_by_name["cmp_kv"].create_tensor()
    cmp_kv_slabs = [base_cmp_kv] + [
        torch.zeros_like(base_cmp_kv) for _ in range(FWD_NUM_LAYERS - 1)
    ]
    cmp_kv_fwd = torch.cat(cmp_kv_slabs, dim=1)
    specs_by_name["cmp_kv"] = TensorSpec(
        "cmp_kv",
        [cp_size, FWD_NUM_LAYERS * PREFILL_CMP_BLOCK_NUM, BLOCK_SIZE, 1, HEAD_DIM],
        torch.bfloat16, init_value=cmp_kv_fwd, is_output=True,
    )

    # --- HCA type-specific (layers 3 and 5) --------------------------------
    # Compact compressor weights (no leading rank axis): stacked by
    # HCA_NUM_LAYERS on axis 0 -> [HCA_NUM_LAYERS * unit, ...]. Each HCA
    # layer slices its type ordinal (0 for L3, 1 for L5).
    for name in _HCA_COMPACT_WEIGHT_NAMES:
        base_name = name[len("hca_"):]  # e.g. cmp_wkv
        specs_by_name[name] = _stack_type_spec(
            name, hca_by_name[base_name], HCA_NUM_LAYERS,
            rank_leading=False,
        )
    # Persistent HCA compressor state (rank-leading InOut root): stacked by
    # HCA_NUM_LAYERS on axis 1 -> [cp_size, HCA_NUM_LAYERS * unit, ...]. The
    # state block table maps logical positions to physical ids in
    # [0, HCA_STATE_PHYSICAL_BLOCKS), which is relative to each sliced
    # per-layer root, so one shared copy serves both layers (§6.4).
    for name in _HCA_STATE_NAMES:
        base_name = name[len("hca_"):]
        specs_by_name[name] = _stack_type_spec(
            name, hca_by_name[base_name], HCA_NUM_LAYERS,
            rank_leading=True,
        )
    # HCA-specific metadata (single copy).
    for name in _HCA_METADATA_NAMES:
        specs_by_name[name] = hca_by_name[name]

    # --- CSA type-specific (layers 2 and 4) --------------------------------
    # CSA main/inner compressor + indexer weights. The CSA builder names
    # cmp_wkv/cmp_wgate/cmp_ape/cmp_norm_w (main) and inner_wkv/inner_wgate/
    # inner_ape/inner_norm_w; hadamard_idx/idx_wq_b/idx_wq_b_scale/
    # idx_weights_proj keep their bare names. Stacked by CSA_NUM_LAYERS so
    # each CSA layer slices its type ordinal (0 for L2, 1 for L4).
    _csa_main_compact_map = {
        "csa_cmp_wkv": "cmp_wkv",
        "csa_cmp_wgate": "cmp_wgate",
        "csa_cmp_ape": "cmp_ape",
        "csa_cmp_norm_w": "cmp_norm_w",
        "csa_inner_wkv": "inner_wkv",
        "csa_inner_wgate": "inner_wgate",
        "csa_inner_ape": "inner_ape",
        "csa_inner_norm_w": "inner_norm_w",
    }
    for new_name, base_name in _csa_main_compact_map.items():
        specs_by_name[new_name] = _stack_type_spec(
            new_name, csa_by_name[base_name], CSA_NUM_LAYERS,
            rank_leading=False,
        )
    for name in ("hadamard_idx", "idx_wq_b", "idx_wq_b_scale", "idx_weights_proj"):
        specs_by_name[name] = _stack_type_spec(
            name, csa_by_name[name], CSA_NUM_LAYERS,
            rank_leading=False,
        )
    # Persistent CSA state/caches: stacked by CSA_NUM_LAYERS on the
    # per-rank block axis. The 4 renamed state specs keep their csa_ prefix;
    # idx_kv_cache/idx_kv_scale keep bare names. Block tables remain a
    # single shared copy (physical ids are relative to the sliced root).
    for new_name, base_name in _CSA_STATE_RENAME_MAP.items():
        specs_by_name[new_name] = _stack_type_spec(
            new_name, csa_by_name[base_name], CSA_NUM_LAYERS,
            rank_leading=True,
        )
    for name in ("idx_kv_cache", "idx_kv_scale"):
        specs_by_name[name] = _stack_type_spec(
            name, csa_by_name[name], CSA_NUM_LAYERS,
            rank_leading=True,
        )
    # idx_block_table: logical slot -> physical block id in
    # [0, PREFILL_IDX_BLOCK_NUM), relative to each sliced root; single copy.
    specs_by_name["idx_block_table"] = csa_by_name["idx_block_table"]
    # CSA reusable rank-local workspaces (serial layers reuse them).
    for name in _CSA_WORKSPACE_NAMES:
        specs_by_name[name] = csa_by_name[name]
    # CSA-specific metadata (single copy).
    for name in _CSA_METADATA_NAMES:
        specs_by_name[name] = csa_by_name[name]

    # Shared compressed-KV block table (HCA and CSA compact both index it).
    specs_by_name["cmp_block_table"] = hca_by_name["cmp_block_table"]

    # MoE weights: layer-stack the single-layer base (rank-leading).
    moe_specs = build_moe_tensor_specs(layer_id=0, num_tokens=T)
    moe_by_name = {}
    for spec in moe_specs:
        if not isinstance(spec, TensorSpec):
            continue
        if spec.name in {"x_hc", "x_next", "input_ids"}:
            continue
        moe_by_name[spec.name] = spec
    for name in _MOE_WEIGHT_NAMES:
        base = moe_by_name[name]
        specs_by_name[name] = _make_stacked_moe_spec(
            name, base, FWD_NUM_LAYERS
        )

    # input_ids: rank-local structured, reused by all layers.
    active_lengths_spec = swa_by_name["overlay_active_lengths"]
    specs_by_name["input_ids"] = _build_input_ids_spec(
        cp_size, active_lengths_spec, 4100 + cp_size * 31
    )

    # Phase 3 final-tail weights (replicated; rank-leading [cp_size]).
    # hc_head_fn/scale/base init values mirror hc_head.py:188-197; final_norm_w
    # mirrors rmsnorm.py:102. They are one shared tensor per rank (not stacked
    # by layer); the FWD child reads them directly per rank.
    specs_by_name["hc_head_fn"] = TensorSpec(
        "hc_head_fn", [cp_size, HC_MULT, HC_DIM], torch.float32,
        init_value=lambda: torch.randn(cp_size, HC_MULT, HC_DIM) * 0.0519,
    )
    specs_by_name["hc_head_scale"] = TensorSpec(
        "hc_head_scale", [cp_size, 1], torch.float32,
        init_value=lambda: torch.tensor([0.076099]).repeat(cp_size, 1),
    )
    specs_by_name["hc_head_base"] = TensorSpec(
        "hc_head_base", [cp_size, HC_MULT], torch.float32,
        init_value=lambda: torch.tensor(
            [5.9166, -3.6223, -2.9324, -3.3124]
        ).repeat(cp_size, 1),
    )
    specs_by_name["final_norm_w"] = TensorSpec(
        "final_norm_w", [cp_size, D], torch.bfloat16,
        init_value=lambda: (torch.randn(cp_size, D) * 0.1 + 1.0).to(
            torch.bfloat16
        ),
    )

    # lm_head_weight: each card owns vocab shard rank % LM_HEAD_TP_SIZE
    # (resident="stacked"). Mirrors lm_head.py:535-537: one shard per TP rank,
    # then stacked per CP rank so card r carries a copy of shard r % TP.
    def _init_lm_head_weight():
        shards = (
            torch.randn(LM_HEAD_TP_SIZE, LM_HEAD_VOCAB_PER_TP, D) / (D ** 0.5)
        ).to(torch.bfloat16)
        return torch.stack(
            [shards[r % LM_HEAD_TP_SIZE] for r in range(cp_size)], dim=0
        )

    specs_by_name["lm_head_weight"] = TensorSpec(
        "lm_head_weight", [cp_size, LM_HEAD_VOCAB_PER_TP, D], torch.bfloat16,
        init_value=_init_lm_head_weight, resident="stacked",
    )

    # logit_row_indices: single-prompt final-token selection (§8.8). All unused
    # entries stay -1 (lm_head.py masks them so the corresponding logit rows
    # stay zero). For the canonical single-prompt CP fixture, every segment
    # carries TAIL_ROWS tokens, so only tile 0 is active (tile 1 is empty):
    # the final active row of a segment is tile 0 row T-1. The owning rank/part
    # of each segment come from ctx["owner_segments"] (a list of length
    # cp_size, each a LOCAL_PARTS list of segment ids) and the segment length
    # from ctx["lengths"]. To exercise every TP group under grouped smoke
    # (§8.8), select the last segment owned by each TP group and place its
    # final active row at index 0 of that group's owning rank; all other
    # entries stay -1. For CP2/CP4 with TP=2 this is one (CP2) or two (CP4)
    # selected rows; for CP8 TP=4 it is two rows, one per TP group.
    def _init_logit_row_indices():
        indices = torch.full(
            (cp_size, LM_HEAD_MAX_LOGIT_ROWS), -1, dtype=torch.int32
        )
        owner_parts = ctx.get("owner_segments", None)
        seg_lengths = ctx.get("lengths", None)
        if owner_parts is None or seg_lengths is None:
            indices[0, 0] = LOCAL_ROWS - 1
            return indices
        # group -> (segment_id, owner_rank, local_part); keep the highest
        # segment id per TP group.
        n_seg = cp_size * LOCAL_PARTS
        group_last: dict = {}
        for r in range(cp_size):
            if r >= len(owner_parts):
                continue
            for local_part, seg in enumerate(owner_parts[r]):
                if seg < 0:
                    continue
                grp = r // LM_HEAD_TP_SIZE
                cur = group_last.get(grp)
                if cur is None or seg > cur[0]:
                    group_last[grp] = (seg, r, local_part)
        for grp, (seg, r, local_part) in group_last.items():
            seg_len = int(seg_lengths[seg]) if seg < len(seg_lengths) else 0
            # last tile with active tokens: active_tile(len, tile) =
            # max(0, min(T, len - tile*T)); scan tiles high->low.
            tile = MAX_SEGMENT_TILES - 1
            while tile >= 0:
                active = max(0, min(T, seg_len - tile * T))
                if active > 0:
                    break
                tile -= 1
            if tile < 0:
                continue
            row = active - 1
            flat = (local_part * MAX_SEGMENT_TILES + tile) * T + row
            indices[r, 0] = flat
        return indices

    specs_by_name["logit_row_indices"] = TensorSpec(
        "logit_row_indices", [cp_size, LM_HEAD_MAX_LOGIT_ROWS], torch.int32,
        init_value=_init_logit_row_indices,
    )

    # Phase 3 final-tail outputs (§8.7).
    specs_by_name["pre_hc_hidden_out"] = TensorSpec(
        "pre_hc_hidden_out",
        [cp_size, LOCAL_PARTS, MAX_SEGMENT_TILES, T, HC_MULT, D],
        torch.float32, is_output=True,
    )
    specs_by_name["hidden_out"] = TensorSpec(
        "hidden_out", [cp_size, LOCAL_ROWS, D], torch.bfloat16,
        is_output=True,
    )
    specs_by_name["logits"] = TensorSpec(
        "logits", [cp_size, LM_HEAD_MAX_LOGIT_ROWS, LM_HEAD_VOCAB],
        torch.float32, is_output=True,
    )

    # Verify the host ABI matches.
    missing = [n for n in FWD_HOST_ARG_ORDER if n not in specs_by_name]
    extra = [n for n in specs_by_name if n not in FWD_HOST_ARG_ORDER]
    if missing or extra:
        raise ValueError(
            f"FWD host ABI mismatch: missing={missing}, extra={extra}"
        )
    ordered = [specs_by_name[n] for n in FWD_HOST_ARG_ORDER]
    return ordered, ctx


# ---------------------------------------------------------------------------
# Harness-only Phase-3 tail sanity comparators (§8.9).
#
# The multi-layer FWD has no mathematical golden. Under --check-outputs the
# harness supplies a no-op golden_fn (so every is_output spec gets a
# zero-filled expected tensor) and a per-output compare_fn: persistent outputs
# pass through _accept_output, and the three Phase-3 tail outputs
# (pre_hc_hidden_out, hidden_out, logits) are inspected for finite + nonzero
# active values (§8.9). --check-x-out is kept as an alias. This stays in the
# Python test harness; the FWD kernel ABI is unchanged.
# ---------------------------------------------------------------------------
def _accept_output(_actual, _expected, **_kwargs):
    """Pass-through comparator for persistent cache/state outputs."""
    return True, ""


def _check_pre_hc_hidden_out(actual, _expected, *, inputs, **_kwargs):
    """Sanity comparator for pre_hc_hidden_out: active values must be finite
    and not all zero on every active rank. Prints per-rank active/nonzero
    counts and min/max/absmax. Not a golden compare."""
    import torch

    active_lengths = inputs["overlay_active_lengths"][..., 1]  # [R, P, S]
    cp_size, n_parts, n_segs, _t, hc_mult, _d = actual.shape
    token_ids = torch.arange(_t).view(1, 1, 1, _t, 1, 1)
    active_mask = token_ids < active_lengths.view(
        cp_size, n_parts, n_segs, 1, 1, 1
    )
    active_mask = active_mask.expand(cp_size, n_parts, n_segs, _t, hc_mult, _d)
    failures = []
    for rank in range(cp_size):
        rank_mask = active_mask[rank]
        if not rank_mask.any():
            print(f"[pre_hc_hidden_out] rank {rank}: no active tokens (skipped)")
            continue
        rank_values = actual[rank][rank_mask]
        n_active = int(rank_mask.sum().item() // hc_mult // _d)
        finite_mask = torch.isfinite(rank_values)
        n_nonzero = int((rank_values != 0).sum().item())
        if not bool(finite_mask.all().item()):
            n_bad = int((~finite_mask).sum().item())
            failures.append(
                f"rank {rank}: {n_bad} non-finite values in active "
                f"pre_hc_hidden_out"
            )
            print(
                f"[pre_hc_hidden_out] rank {rank}: FAIL active={n_active} "
                f"non-finite={n_bad} (of {rank_values.numel()})"
            )
            continue
        all_zero = n_nonzero == 0
        absmax = float(rank_values.abs().max().item()) if rank_values.numel() else 0.0
        mn = float(rank_values.min().item()) if rank_values.numel() else 0.0
        mx = float(rank_values.max().item()) if rank_values.numel() else 0.0
        status = "FAIL (all zero)" if all_zero else "OK"
        print(
            f"[pre_hc_hidden_out] rank {rank}: {status} active={n_active} "
            f"nonzero={n_nonzero} min={mn:.6f} max={mx:.6f} absmax={absmax:.6f}"
        )
        if all_zero:
            failures.append(
                f"rank {rank}: all active pre_hc_hidden_out values are zero"
            )
    if failures:
        return False, "\n".join(failures)
    return True, "pre_hc_hidden_out sanity OK"


def _check_hidden_out(actual, _expected, *, inputs, **_kwargs):
    """Sanity comparator for hidden_out ([CP_SIZE, LOCAL_ROWS, D] BF16): the
    rows selected by logit_row_indices must be finite and not all zero;
    invalid (-1) selected rows must stay deterministic zero."""
    import torch

    row_indices = inputs["logit_row_indices"]  # [CP_SIZE, MAX_LOGIT_ROWS]
    cp_size = actual.shape[0]
    failures = []
    for rank in range(cp_size):
        rank_rows = row_indices[rank]
        rank_hidden = actual[rank]
        valid_mask = row_indices[rank] >= 0
        n_valid = int(valid_mask.sum().item())
        # Whole-tensor diagnostic: distinguishes "hc_head/rms_norm ran but the
        # selected row fell in a dead (zero-published) tile" from "the inlined
        # tail never wrote hidden_out at all". Without this, a wrong
        # logit_row_indices lowering and a missing tail both surface as "all
        # zero" on the selected row only.
        whole_absmax = (
            float(rank_hidden.float().abs().max().item())
            if rank_hidden.numel() else 0.0
        )
        whole_nonzero = int((rank_hidden != 0).sum().item())
        if n_valid == 0:
            print(
                f"[hidden_out] rank {rank}: no valid selected rows (skipped) "
                f"whole_absmax={whole_absmax:.6f} whole_nonzero={whole_nonzero}"
            )
            continue
        selected = rank_hidden[row_indices[rank][valid_mask]]
        finite_mask = torch.isfinite(selected.float())
        n_nonzero = int((selected != 0).sum().item())
        if not bool(finite_mask.all().item()):
            n_bad = int((~finite_mask).sum().item())
            failures.append(
                f"rank {rank}: {n_bad} non-finite values in selected "
                f"hidden_out rows"
            )
            print(
                f"[hidden_out] rank {rank}: FAIL selected={n_valid} "
                f"non-finite={n_bad}"
            )
            continue
        absmax = float(selected.abs().max().item()) if selected.numel() else 0.0
        all_zero = n_nonzero == 0
        status = "FAIL (all zero)" if all_zero else "OK"
        print(
            f"[hidden_out] rank {rank}: {status} selected={n_valid} "
            f"nonzero={n_nonzero} absmax={absmax:.6f} "
            f"whole_absmax={whole_absmax:.6f} whole_nonzero={whole_nonzero}"
        )
        if all_zero:
            failures.append(
                f"rank {rank}: selected hidden_out rows are all zero"
            )
    if failures:
        return False, "\n".join(failures)
    return True, "hidden_out sanity OK"


def _check_logits(actual, _expected, *, inputs, **_kwargs):
    """Sanity comparator for logits ([CP_SIZE, MAX_LOGIT_ROWS, VOCAB] FP32):
    each valid selected row (logit_row_indices >= 0) must be finite and not
    all zero; invalid (-1) rows stay deterministic zero; every TP vocab shard
    must carry nonzero values in a valid full-vocabulary row."""
    import torch

    row_indices = inputs["logit_row_indices"]  # [CP_SIZE, MAX_LOGIT_ROWS]
    cp_size, _max_rows, vocab = actual.shape
    failures = []
    for rank in range(cp_size):
        valid_mask = row_indices[rank] >= 0
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0:
            print(f"[logits] rank {rank}: no valid selected rows (skipped)")
            continue
        valid_rows = actual[rank][valid_mask]
        finite_mask = torch.isfinite(valid_rows)
        if not bool(finite_mask.all().item()):
            n_bad = int((~finite_mask).sum().item())
            failures.append(
                f"rank {rank}: {n_bad} non-finite values in valid logits rows"
            )
            print(
                f"[logits] rank {rank}: FAIL selected={n_valid} "
                f"non-finite={n_bad}"
            )
            continue
        n_nonzero = int((valid_rows != 0).sum().item())
        absmax = float(valid_rows.abs().max().item()) if valid_rows.numel() else 0.0
        # Every TP vocab shard must carry nonzero values in a valid row.
        n_shards = LM_HEAD_TP_SIZE
        shard_size = vocab // n_shards
        per_shard_nonzero = [
            int((valid_rows[:, s * shard_size:(s + 1) * shard_size] != 0)
                .sum().item())
            for s in range(n_shards)
        ]
        empty_shards = [s for s, n in enumerate(per_shard_nonzero) if n == 0]
        all_zero = n_nonzero == 0
        invalid_rows = actual[rank][~valid_mask]
        invalid_nonzero = int((invalid_rows != 0).sum().item())
        status = "FAIL (all zero)" if all_zero else "OK"
        print(
            f"[logits] rank {rank}: {status} selected={n_valid} "
            f"nonzero={n_nonzero} absmax={absmax:.6f} "
            f"per_shard_nonzero={per_shard_nonzero} "
            f"invalid_row_nonzero={invalid_nonzero}"
        )
        if all_zero:
            failures.append(f"rank {rank}: valid logits rows are all zero")
        if empty_shards:
            failures.append(
                f"rank {rank}: empty vocab shards {empty_shards} in valid "
                f"logits rows"
            )
        if invalid_nonzero:
            failures.append(
                f"rank {rank}: {invalid_nonzero} nonzero values in "
                f"invalid (-1) logits rows (must stay zero)"
            )
    if failures:
        return False, "\n".join(failures)
    return True, "logits sanity OK"


def _build_outputs_compare_fn(specs):
    """Build a compare_fn dict: persistent outputs pass through
    _accept_output; the three Phase-3 tail outputs use dedicated sanity
    comparators (§8.9)."""
    compare_fn = {}
    for spec in specs:
        if not isinstance(spec, TensorSpec):
            continue
        if not spec.is_output:
            continue
        if spec.name == "pre_hc_hidden_out":
            compare_fn[spec.name] = _check_pre_hc_hidden_out
        elif spec.name == "hidden_out":
            compare_fn[spec.name] = _check_hidden_out
        elif spec.name == "logits":
            compare_fn[spec.name] = _check_logits
        else:
            compare_fn[spec.name] = _accept_output
    return compare_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DeepSeek V4 context-parallel prefill multi-layer (SWA, SWA, repeated CSA/HCA pairs) forward."
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
        "--tp", type=int, default=LM_HEAD_TP_SIZE, choices=[1, 2, 4],
        help=(
            "tensor-parallel world size for the LM head (parsed at import by "
            "lm_head; declared here so --tp survives argparse's "
            "unrecognized-argument check). default reflects the import-time "
            "value so a plain run with no --tp keeps TP=2."
        ),
    )
    parser.add_argument(
        "--num-layers", type=int, default=FWD_NUM_LAYERS,
        choices=list(FWD_LAYER_CHOICES),
        help=f"number of FWD layers (parsed at import; choices {FWD_LAYER_CHOICES}; 43 is the production schedule)",
    )
    parser.add_argument("--enable-chip-swimlane", action="store_true", default=False)
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    parser.add_argument(
        "--check-outputs", "--check-x-out", action="store_true", default=False,
        dest="check_outputs",
        help="harness-only: inspect the Phase-3 tail outputs "
             "(pre_hc_hidden_out, hidden_out, logits) for finite + nonzero "
             "active values (no mathematical golden). §8.9. "
             "--check-x-out is kept as a backward alias.",
    )
    parser.add_argument(
        "--enable-scope-stats", action="store_true", default=ENABLE_SCOPE_STATS,
        help="§8.17.2 diagnostic: enable per-scope resource-peak tracking "
             "(task-window/heap/dep-pool/tensormap). Writes "
             "dfx_outputs/**/scope_stats/scope_stats.jsonl. Parsed at import "
             "so the flag survives argparse's unrecognized-argument check.",
    )
    parser.add_argument(
        "--fwd-only", action="store_true", default=FWD_ONLY,
        help="§8.17.2 diagnostic: omit the LM-head child launch/allocation "
             "from the host graph (no Domain 6 buffers, no second rank "
             "loop) while reusing the exact 43-layer rank core. hidden_out "
             "is still produced by the FWD child; logits stays zero and the "
             "--check-outputs comparator accepts it. Parsed at import.",
    )
    args = parser.parse_args()

    if args.num_layers != FWD_NUM_LAYERS:
        raise SystemExit(
            f"--num-layers={args.num_layers} does not match import-time "
            f"FWD_NUM_LAYERS={FWD_NUM_LAYERS} (parsed from sys.argv)."
        )
    if args.fwd_only != FWD_ONLY:
        raise SystemExit(
            f"--fwd-only={args.fwd_only} does not match import-time "
            f"FWD_ONLY={FWD_ONLY} (parsed from sys.argv)."
        )
    if args.fwd_only:
        # §8.17.2: the FWD-only host variant omits the LM-head child
        # launch/allocation. A Python ``if not FWD_ONLY`` guard on the second
        # rank loop fails the JIT (SSA traces the dead branch and either flags
        # free variables or dead allocations), so the variant needs a separate
        # trimmed host callable + filtered spec list. A1 already proved the
        # standalone LM head passes and A3 proved CP8 6-layer full graph fails
        # in ~1 s, so the FWD-only run is not required to localize the hot
        # scope — --enable-scope-stats on the full CP8 reproducer suffices.
        raise SystemExit(
            "--fwd-only is declared (so the flag survives argparse) but the "
            "FWD-only host variant is not implemented; run the full graph with "
            "--enable-scope-stats instead (§8.17.4 step 2)."
        )
    if args.enable_scope_stats != ENABLE_SCOPE_STATS:
        raise SystemExit(
            f"--enable-scope-stats={args.enable_scope_stats} does not match "
            f"import-time ENABLE_SCOPE_STATS={ENABLE_SCOPE_STATS} "
            f"(parsed from sys.argv)."
        )

    device_ids = [int(d) for d in args.device.split(",")]
    if len(device_ids) < args.cp:
        raise SystemExit(
            f"CP{args.cp} requires {args.cp} devices, got {device_ids}"
        )

    specs, ctx = build_tensor_specs(cp_size=args.cp)

    if args.check_outputs:
        # No-op golden fills every is_output spec with zeros; the compare_fn
        # ignores expected for the tail outputs and pass-throughs for the
        # persistent caches.
        golden_fn = lambda _scratch: None
        compare_fn = _build_outputs_compare_fn(specs)
    else:
        golden_fn = None
        compare_fn = None

    result = run_jit(
        fn=l3_prefill_cp_fwd,
        specs=specs,
        golden_fn=golden_fn,
        compare_fn=compare_fn,
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
            enable_scope_stats=args.enable_scope_stats,
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
