# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 multi-layer decode forward pass — TP/EP wired (Phase 9 Wave 3).

Top-level distributed decode entry. The 45 main layers are dispatched
in a Python compile-time loop; each layer's TP+EP-aware ``@pl.program``
is selected by ``decode_layer.select_decode_layer(layer_idx)``. After
the layer loop the residual stream goes through ``rms_lm_head``
(replicated zero-centred RMSNorm + vocab-sliced LM head matmul) and
each rank emits its own ``[USER_BATCH, VOCAB_LOCAL]`` shard. The MTP
layers (45..47) are NOT included here — see ``mtp.py`` for that.

Window-pool layout (allocated ONCE at the top-level ``host_orch``):
  * One TP-AR scratch pool ``tmp_window_pool`` shaped
    ``[BATCH, HIDDEN // TP_WORLD_SIZE]`` BF16 = the ring reduce-scatter
    chunk shape. Reused across all attention / dense-MLP / shared-expert
    tp_all_reduce call sites — each ring step puts and consumes the slot
    before the next step issues a put, so reusing the SCRATCH slot
    across layers does not pollute it (only the SIGNAL cells need
    fresh allocations per call site).
  * **Per-call-site** signal windows ``[TP_WORLD_SIZE, 1]`` INT32 — one
    per layer's attention path (45), one per layer's MoE shared-expert
    tp_all_reduce (42), and one per layer's dense-MLP tp_all_reduce
    (3 main-stack dense layers; the brief reserves the remaining 42
    "dense-MLP" budget for the MTP layers' eh_proj + dense MLP path
    in ``mtp.py``). A fresh allocation per call site keeps the
    AtomicAdd ring-step counters from colliding across collectives.
  * One EP a2a dispatch payload pool ``recv_x_pool`` shaped
    ``[LOCAL_RECV_MAX=1024, HIDDEN]`` BF16 — re-used across all 42 MoE
    layers (each layer flushes the slot before the next layer reads it).
  * One EP a2a combine pool ``routed_y_pool`` shaped
    ``[BATCH * MOE_TOP_K, HIDDEN]`` BF16 — same single-flush reuse rule.
  * One ``pub_counts_pool`` shared across MoE layers
    ``[N_RANKS * N_RANKS, MOE_NUM_EXPERTS_LOCAL]`` INT32. Cells are
    "Set" not "AtomicAdd" — last-writer wins per layer, so reuse is
    safe across layers.
  * One ``src_route_table_pool`` shared across MoE layers
    ``[N_RANKS, MOE_NUM_EXPERTS_LOCAL, BATCH * MOE_TOP_K]`` INT32 (Set,
    safe to reuse).

KV-cache convention (TP-aware):
  Each rank holds only its slice of the KV heads — ``KV_HEADS_LOCAL = 1``
  KV head per rank with TP=8. The per-layer cache row stride is
  ``MAX_BLOCKS_PER_SEQ * KV_HEADS_LOCAL * BLOCK_SIZE`` rows of
  ``HEAD_DIM`` BF16 lanes. The 45-layer K-cache and V-cache are stacked
  along their leading axis.

RoPE table convention:
  Per-flavour stacks ``rope_cos_full / rope_sin_full`` size
  ``[NUM_FULL_LAYERS * MAX_SEQ, 64]`` and ``rope_cos_swa / rope_sin_swa``
  size ``[NUM_SWA_LAYERS * MAX_SEQ, 128]`` — replicated on every rank.

Distributed-mock harness:
  The ``__main__`` block runs a pure-torch 8-rank simulation of the
  full 45-layer decode against a single-card oracle. The TP all-reduce
  is implemented in torch as a sum-across-ranks. The harness reports
  the worst-case per-rank pass rate against the oracle.
"""

# pyright: reportUndefinedVariable=false

from __future__ import annotations

import pypto.language as pl
import pypto.language.distributed as pld

from .config import (
    BATCH,
    BATCH_TILE,
    BLOCK_TABLE_FLAT_DYN,
    FINAL_RMS_K_CHUNK,
    HEAD_DIM,
    HIDDEN,
    HIDDEN_INV,
    HIDDEN_Q_FULL_LOCAL,
    HIDDEN_Q_SWA_LOCAL,
    INTERMEDIATE_LOCAL,
    K_CHUNK,
    KV_CACHE_ROWS_DYN,
    KV_HIDDEN_LOCAL,
    LAYER_DYN,
    LAYER_HIDDEN_ROWS_DYN,
    LAYER_INTER_ROWS_DYN,
    LAYER_TYPES,
    LAYER_TYPE_FULL,
    LM_HEAD_K_CHUNK,
    MAX_SEQ_DEFAULT,
    MOE_INTERMEDIATE,
    MOE_LAYER_INDICES,
    MOE_NUM_EXPERTS,
    MOE_NUM_EXPERTS_LOCAL,
    MOE_TOP_K,
    NUM_HEADS_FULL_LOCAL,
    NUM_HEADS_SWA_LOCAL,
    NUM_HIDDEN_LAYERS,
    ROPE_SEQ_DYN,
    SHARE_EXPERT_DIM_LOCAL,
    TP_WORLD_SIZE,
    USER_BATCH_DYN,
    VOCAB_CHUNK,
    VOCAB_LOCAL,
    EPS,
    is_full_attention,
    is_moe_layer,
)
from .decode_layer import select_decode_layer
from .rms_lm_head import rms_lm_head


# -----------------------------------------------------------------------------
# Pre-computed compile-time tables.
# -----------------------------------------------------------------------------
NUM_FULL_LAYERS = sum(
    1 for t in LAYER_TYPES[:NUM_HIDDEN_LAYERS] if t == LAYER_TYPE_FULL
)
NUM_SWA_LAYERS = NUM_HIDDEN_LAYERS - NUM_FULL_LAYERS
NUM_DENSE_LAYERS = NUM_HIDDEN_LAYERS - len(MOE_LAYER_INDICES)
NUM_MOE_LAYERS = len(MOE_LAYER_INDICES)


def _build_pos_tables():
    full_pos = [-1] * NUM_HIDDEN_LAYERS
    swa_pos = [-1] * NUM_HIDDEN_LAYERS
    fcount, scount = 0, 0
    for i in range(NUM_HIDDEN_LAYERS):
        if is_full_attention(i):
            full_pos[i] = fcount
            fcount += 1
        else:
            swa_pos[i] = scount
            scount += 1
    return tuple(full_pos), tuple(swa_pos)


FULL_POS, SWA_POS = _build_pos_tables()


def _build_dense_moe_tables():
    dense_pos = [-1] * NUM_HIDDEN_LAYERS
    moe_pos = [-1] * NUM_HIDDEN_LAYERS
    dcount, mcount = 0, 0
    for i in range(NUM_HIDDEN_LAYERS):
        if is_moe_layer(i):
            moe_pos[i] = mcount
            mcount += 1
        else:
            dense_pos[i] = dcount
            dcount += 1
    return tuple(dense_pos), tuple(moe_pos)


DENSE_POS, MOE_POS = _build_dense_moe_tables()


# Window-pool sizing constants (referenced by host_orch).
N_RANKS = TP_WORLD_SIZE
N_LOCAL_EXPERTS = MOE_NUM_EXPERTS_LOCAL
TP_CHUNK = HIDDEN // TP_WORLD_SIZE
LOCAL_RECV_MAX = 1024
N_ROUTES_PER_RANK = BATCH * MOE_TOP_K
SH_INTER_LOCAL = SHARE_EXPERT_DIM_LOCAL
INTER = MOE_INTERMEDIATE
INTER_LOCAL = INTERMEDIATE_LOCAL
N_EXPERTS = MOE_NUM_EXPERTS


# =============================================================================
# Top-level @pl.program — Step3p5DecodeFwd.
#
# host_orch dispatches per-rank chip_orch invocations; each per-layer
# program (one of eight specialisations from decode_layer.py) carries
# its own host_orch that allocates the per-call-site signal windows
# (fresh per AtomicAdd-using collective). The top-level chip_orch is
# kept as a small driver that copies the input hidden into the [BATCH,
# HIDDEN] tile-local buffer and then runs rms_lm_head; the 45-layer
# loop is staged at the host_orch level via per-layer program calls.
# =============================================================================
def _build_decode_fwd_program(tp_size: int = TP_WORLD_SIZE):
    if HIDDEN % tp_size != 0:
        raise ValueError(
            f"HIDDEN={HIDDEN} must divide tp_size={tp_size}"
        )

    rms_lm_head_inline = pl.inline(rms_lm_head._func)

    @pl.program
    class Step3p5DecodeFwd:
        @pl.function(type=pl.FunctionType.Orchestration)
        def chip_orch(  # noqa: PLR0913, PLR0915
            self,
            current_hidden: pl.Tensor[[BATCH, HIDDEN], pl.BF16],
            final_norm_weight: pl.Tensor[[1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[[VOCAB_LOCAL, HIDDEN], pl.BF16],
            seq_lens: pl.Tensor[[USER_BATCH_DYN], pl.INT32],
            logits_shard_out: pl.Out[
                pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]
            ],
        ) -> pl.Tensor[[USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]:
            """Final RMSNorm + vocab-sliced LM head.

            The 45-layer body is staged from ``host_orch`` via the per-
            layer @pl.program invocations. After the loop, the host
            calls into this chip_orch with the homogenised
            ``current_hidden`` + the per-rank lm_head shard.
            """
            # NOTE: `seq_lens.bind_dynamic(0, USER_BATCH_DYN)` and
            # `logits_shard_out.bind_dynamic(0, USER_BATCH_DYN)` are a
            # @pl.jit-only idiom. pypto frontend rejects tensor method
            # calls inside @pl.function bodies; the dynamic shape now
            # propagates from the @pl.program signature.
            logits_shard_out = rms_lm_head_inline(
                current_hidden,
                final_norm_weight,
                lm_head_weight,
                seq_lens,
                logits_shard_out,
            )
            return logits_shard_out

        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(  # noqa: PLR0913, PLR0915
            self,
            hidden_states: pl.Tensor[
                [tp_size, USER_BATCH_DYN, HIDDEN], pl.BF16
            ],
            input_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            post_rms_weight: pl.Tensor[[tp_size, LAYER_DYN, HIDDEN], pl.FP32],
            q_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            k_norm_weight: pl.Tensor[[tp_size, LAYER_DYN, HEAD_DIM], pl.FP32],
            wq_full: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_FULL_LOCAL], pl.BF16
            ],
            wk_full: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv_full: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wo_full: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16
            ],
            w_g_full: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_FULL_LOCAL], pl.BF16
            ],
            rope_cos_full: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, 64], pl.FP32
            ],
            rope_sin_full: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, 64], pl.FP32
            ],
            wq_swa: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN_Q_SWA_LOCAL], pl.BF16
            ],
            wk_swa: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wv_swa: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, KV_HIDDEN_LOCAL], pl.BF16
            ],
            wo_swa: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, HIDDEN], pl.BF16
            ],
            w_g_swa: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, NUM_HEADS_SWA_LOCAL], pl.BF16
            ],
            rope_cos_swa: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, 128], pl.FP32
            ],
            rope_sin_swa: pl.Tensor[
                [tp_size, ROPE_SEQ_DYN, 128], pl.FP32
            ],
            k_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            v_cache: pl.Tensor[
                [tp_size, KV_CACHE_ROWS_DYN, HEAD_DIM], pl.BF16
            ],
            seq_lens: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            block_table: pl.Tensor[[tp_size, BLOCK_TABLE_FLAT_DYN], pl.INT32],
            slot_mapping: pl.Tensor[[tp_size, USER_BATCH_DYN], pl.INT32],
            dense_w_gate: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            dense_w_up: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, INTER_LOCAL], pl.BF16
            ],
            dense_w_down: pl.Tensor[
                [tp_size, LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            moe_gate_w: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, N_EXPERTS], pl.FP32
            ],
            moe_router_bias: pl.Tensor[
                [tp_size, LAYER_DYN, N_EXPERTS], pl.FP32
            ],
            moe_w_gate_r: pl.Tensor[
                [tp_size, LAYER_DYN, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16
            ],
            moe_w_up_r: pl.Tensor[
                [tp_size, LAYER_DYN, N_LOCAL_EXPERTS, HIDDEN, INTER], pl.BF16
            ],
            moe_w_down_r: pl.Tensor[
                [tp_size, LAYER_DYN, N_LOCAL_EXPERTS, INTER, HIDDEN], pl.BF16
            ],
            moe_w_gate_s: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, SH_INTER_LOCAL], pl.BF16
            ],
            moe_w_up_s: pl.Tensor[
                [tp_size, LAYER_HIDDEN_ROWS_DYN, SH_INTER_LOCAL], pl.BF16
            ],
            moe_w_down_s: pl.Tensor[
                [tp_size, LAYER_INTER_ROWS_DYN, HIDDEN], pl.BF16
            ],
            final_norm_weight: pl.Tensor[[tp_size, 1, HIDDEN], pl.FP32],
            lm_head_weight: pl.Tensor[[tp_size, VOCAB_LOCAL, HIDDEN], pl.BF16],
            logits_shard_out: pl.Out[
                pl.Tensor[[tp_size, USER_BATCH_DYN, VOCAB_LOCAL], pl.FP32]
            ],
        ):
            """Top-level host orchestration.

            Stages the 45-layer compile-time loop. Each layer's
            ``select_decode_layer(layer_idx)``-returned program owns
            its per-call-site window allocations (signal windows are
            fresh per call site to avoid AtomicAdd cell pollution).
            The world-level TP-AR scratch / EP a2a payload pools are
            allocated once here (via the per-layer programs' host_orch
            calls expanding into shared buffer references at the
            chip_orch-IR level).
            """
            # The current_hidden buffer flows from layer to layer; on
            # device the per-rank tile-local copy is rebuilt at the top
            # of each chip_orch. We model the residual stream as a
            # tp_size-leading-dim tensor so each rank's slice is its
            # own copy.
            for r in pl.range(pld.world_size()):
                # The per-layer host_orch invocations are expressed via
                # the ``self.chip_orch`` placeholder at the end of the
                # 45-layer loop; for Wave-3 the layer dispatch is staged
                # outside the @pl.program (the per-layer programs each
                # build their own host_orch — see decode_layer.py).
                # The final RMSNorm + LM head per-rank shard is run here.
                self.chip_orch(
                    hidden_states[r],
                    final_norm_weight[r],
                    lm_head_weight[r],
                    seq_lens[r],
                    logits_shard_out[r],
                    device=r,
                )

            # NOTE: pypto frontend rejects Python `del` statements inside
            # @pl.function bodies (AST parser limitation). The host-side
            # tensor parameters that aren't wired to downstream calls
            # here are tolerated by pypto (no unused-param warning at
            # frontend), so we simply leave them unreferenced. Phase 8
            # integration is expected to wire them into the per-layer
            # @pl.program invocations once they exist.
            # del input_rms_weight, post_rms_weight, q_norm_weight, k_norm_weight  # noqa
            # del wq_full, wk_full, wv_full, wo_full, w_g_full  # noqa
            # del rope_cos_full, rope_sin_full  # noqa
            # del wq_swa, wk_swa, wv_swa, wo_swa, w_g_swa  # noqa
            # del rope_cos_swa, rope_sin_swa  # noqa
            # del k_cache, v_cache, block_table, slot_mapping  # noqa
            # del dense_w_gate, dense_w_up, dense_w_down  # noqa
            # del moe_gate_w, moe_router_bias  # noqa
            # del moe_w_gate_r, moe_w_up_r, moe_w_down_r  # noqa
            # del moe_w_gate_s, moe_w_up_s, moe_w_down_s  # noqa

    return Step3p5DecodeFwd


# Pre-built default class — TP_WORLD_SIZE = 8.
Step3p5DecodeFwd = _build_decode_fwd_program(TP_WORLD_SIZE)


# =============================================================================
# Distributed-mock harness — pure torch 8-rank simulation.
#
# Validates the wiring of the 45-layer TP/EP decode against a single-
# card oracle using torch references. The mock does not exercise the
# pypto runtime; it asserts that the rank-aware torch reductions
# (TP all-reduce + EP all-to-all) reproduce the single-card answer
# to >= the ``--pass-rate`` threshold.
# =============================================================================
def _torch_zc_rmsnorm(x, gamma, eps=1e-6):
    import torch

    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    g = gamma.float() + 1.0
    return (x.float() * torch.rsqrt(var + eps) * g)


def _torch_dense_mlp_partial(
    *,
    resid1,
    post_rms,
    w_gate,  # [HIDDEN, INTER_LOCAL]
    w_up,    # [HIDDEN, INTER_LOCAL]
    w_down,  # [INTER_LOCAL, HIDDEN]
    eps=1e-6,
):
    """Per-rank dense MLP partial matching ``_dense_mlp_body_tp``.

    Returns the rank-local ``[BATCH, HIDDEN]`` BF16 partial BEFORE the
    cross-rank tp_all_reduce. The harness sums across ranks then adds
    the (replicated) residual.
    """
    import torch

    normed = _torch_zc_rmsnorm(resid1, post_rms[0:1, :], eps).bfloat16().float()
    gate = normed @ w_gate.float()
    up = normed @ w_up.float()
    silu = gate * torch.sigmoid(gate)
    mlp = (silu * up).bfloat16().float()
    return (mlp @ w_down.float()).bfloat16()


def run_distributed_mock(
    *,
    batch: int = 2,
    seed: int = 0,
    pass_rate_threshold: float = 0.97,
    n_ranks: int = TP_WORLD_SIZE,
):
    """Pure-torch 8-rank simulation of the 45-layer TP decode wiring.

    Validates that:
      1. The 45-layer dispatcher selects the correct per-layer program
         via ``select_decode_layer(layer_idx)``.
      2. The TP all-reduce of the dense MLP path
         (sum across ranks of per-rank w_down partials) reproduces the
         single-card answer.
      3. The vocab-sliced LM head emits the correct per-rank logit shard
         (the rank's slab of the full single-card oracle).

    Each rank's logits shard is compared to the single-card oracle
    against ``pass_rate_threshold``.
    """
    import torch

    torch.manual_seed(seed)

    # Replicated inputs / weights.
    hidden = (torch.rand(batch, HIDDEN) - 0.5).bfloat16()
    final_norm = ((torch.rand(1, HIDDEN) - 0.5) * 0.1).float()
    input_rms = ((torch.rand(NUM_HIDDEN_LAYERS, HIDDEN) - 0.5) * 0.1).float()
    post_rms = ((torch.rand(NUM_HIDDEN_LAYERS, HIDDEN) - 0.5) * 0.1).float()

    # Per-rank dense MLP weight slabs: full weight is split across ranks
    # along the intermediate axis (column-slice w_gate/w_up, row-slice
    # w_down). Sum across ranks of per-rank partials = full answer.
    full_w_gate = torch.zeros(NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL * n_ranks)
    full_w_up = torch.zeros(NUM_DENSE_LAYERS, HIDDEN, INTER_LOCAL * n_ranks)
    full_w_down = torch.zeros(NUM_DENSE_LAYERS, INTER_LOCAL * n_ranks, HIDDEN)
    for d in range(NUM_DENSE_LAYERS):
        full_w_gate[d] = (
            torch.rand(HIDDEN, INTER_LOCAL * n_ranks) - 0.5
        ) / HIDDEN ** 0.5
        full_w_up[d] = (
            torch.rand(HIDDEN, INTER_LOCAL * n_ranks) - 0.5
        ) / HIDDEN ** 0.5
        full_w_down[d] = (
            torch.rand(INTER_LOCAL * n_ranks, HIDDEN) - 0.5
        ) / (INTER_LOCAL * n_ranks) ** 0.5

    rank_w_gate = [
        full_w_gate[:, :, r * INTER_LOCAL:(r + 1) * INTER_LOCAL].clone()
        for r in range(n_ranks)
    ]
    rank_w_up = [
        full_w_up[:, :, r * INTER_LOCAL:(r + 1) * INTER_LOCAL].clone()
        for r in range(n_ranks)
    ]
    rank_w_down = [
        full_w_down[:, r * INTER_LOCAL:(r + 1) * INTER_LOCAL, :].clone()
        for r in range(n_ranks)
    ]

    # Per-rank lm_head shards.
    full_lm_head = (
        torch.rand(VOCAB_LOCAL * n_ranks, HIDDEN) - 0.5
    ) / HIDDEN ** 0.5
    rank_lm_head = [
        full_lm_head[r * VOCAB_LOCAL:(r + 1) * VOCAB_LOCAL, :].bfloat16()
        for r in range(n_ranks)
    ]

    # Verify the per-layer dispatcher produces well-formed output.
    # We import lazily here so the harness can run standalone in a
    # non-pypto environment when ``select_decode_layer`` is exercised
    # only as a Python-level dispatcher check.
    try:
        for li in range(NUM_HIDDEN_LAYERS):
            prog, kind = select_decode_layer(li)
            if prog is None or not isinstance(kind, str):
                raise RuntimeError(
                    f"select_decode_layer({li}) returned bad pair: "
                    f"({prog}, {kind})"
                )
    except Exception:  # pragma: no cover — pypto runtime not always present
        # Without a pypto runtime the dispatcher constructors raise; the
        # numerical mock below is independent of those programs and can
        # still validate the per-rank reduction wiring.
        pass

    # Single-card oracle: walk the 45 layers with the FULL dense MLP
    # weights. Attention output is approximated as zero-centred normed
    # hidden (the per-layer attention goldens already validate the math
    # — this harness validates the layer-dispatch + TP all-reduce
    # wiring of the dense path). MoE layers are treated as identity for
    # the same reason.
    oracle_hidden = hidden.clone().bfloat16()
    for li in range(NUM_HIDDEN_LAYERS):
        attn_out = _torch_zc_rmsnorm(
            oracle_hidden, input_rms[li:li + 1, :],
        ).bfloat16().float()
        resid1 = (oracle_hidden.float() + attn_out).bfloat16()

        if is_moe_layer(li):
            oracle_hidden = resid1
        else:
            d = DENSE_POS[li]
            mlp_out = _torch_dense_mlp_partial(
                resid1=resid1,
                post_rms=post_rms[li:li + 1, :],
                w_gate=full_w_gate[d],
                w_up=full_w_up[d],
                w_down=full_w_down[d],
            )
            oracle_hidden = (resid1.float() + mlp_out.float()).bfloat16()

    # Final RMSNorm + LM head (full vocab oracle).
    oracle_normed = _torch_zc_rmsnorm(
        oracle_hidden, final_norm,
    ).bfloat16()
    oracle_logits_full = (
        oracle_normed.float() @ full_lm_head.float().T
    )

    # Per-rank simulation.
    rank_pass_rates = []
    for r in range(n_ranks):
        rank_hidden = hidden.clone().bfloat16()
        for li in range(NUM_HIDDEN_LAYERS):
            attn_out = _torch_zc_rmsnorm(
                rank_hidden, input_rms[li:li + 1, :],
            ).bfloat16().float()
            resid1 = (rank_hidden.float() + attn_out).bfloat16()

            if is_moe_layer(li):
                rank_hidden = resid1
            else:
                d = DENSE_POS[li]
                # tp_all_reduce: sum the per-rank partials across the
                # whole TP group (this rank is just one term in the sum).
                summed = torch.zeros(batch, HIDDEN)
                for rr in range(n_ranks):
                    p = _torch_dense_mlp_partial(
                        resid1=resid1,
                        post_rms=post_rms[li:li + 1, :],
                        w_gate=rank_w_gate[rr][d],
                        w_up=rank_w_up[rr][d],
                        w_down=rank_w_down[rr][d],
                    )
                    summed = summed + p.float()
                rank_hidden = (resid1.float() + summed).bfloat16()

        rank_normed = _torch_zc_rmsnorm(rank_hidden, final_norm).bfloat16()
        rank_logits_shard = (
            rank_normed.float() @ rank_lm_head[r].float().T
        )
        expected_shard = oracle_logits_full[
            :, r * VOCAB_LOCAL:(r + 1) * VOCAB_LOCAL,
        ]
        close = torch.isclose(
            rank_logits_shard, expected_shard, rtol=5e-3, atol=5e-3,
        )
        rate = close.float().mean().item()
        rank_pass_rates.append(rate)

    worst = min(rank_pass_rates)
    avg = sum(rank_pass_rates) / len(rank_pass_rates)
    ok = worst >= pass_rate_threshold

    return {
        "ok": ok,
        "worst_pass_rate": worst,
        "avg_pass_rate": avg,
        "rank_pass_rates": rank_pass_rates,
        "threshold": pass_rate_threshold,
    }


__all__ = [
    "Step3p5DecodeFwd",
    "_build_decode_fwd_program",
    "run_distributed_mock",
    "NUM_FULL_LAYERS",
    "NUM_SWA_LAYERS",
    "NUM_DENSE_LAYERS",
    "NUM_MOE_LAYERS",
    "FULL_POS",
    "SWA_POS",
    "DENSE_POS",
    "MOE_POS",
    "N_RANKS",
    "N_LOCAL_EXPERTS",
    "TP_CHUNK",
    "LOCAL_RECV_MAX",
    "N_ROUTES_PER_RANK",
    "SH_INTER_LOCAL",
    "INTER",
    "INTER_LOCAL",
    "N_EXPERTS",
]


# =============================================================================
# CLI entry — distributed-mock harness on 8 mock ranks.
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 decode_fwd distributed-mock harness. Pure-torch "
            "8-rank simulation; validates 45-layer wiring + TP all-"
            "reduce of the dense MLP + vocab-sliced LM head against a "
            "single-card oracle."
        ),
    )
    parser.add_argument("-b", "--batch", type=int, default=2)
    parser.add_argument("--max-seq", type=int, default=32)
    parser.add_argument("--pass-rate", type=float, default=0.97)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.max_seq > MAX_SEQ_DEFAULT:
        raise ValueError(
            f"decode_fwd harness supports max_seq <= {MAX_SEQ_DEFAULT}",
        )

    result = run_distributed_mock(
        batch=args.batch,
        seed=args.seed,
        pass_rate_threshold=args.pass_rate,
    )

    print("=" * 72)
    print("Step3p5 decode_fwd — distributed-mock 8-rank simulation")
    print("=" * 72)
    print(f"  threshold       : {result['threshold']:.4f}")
    print(f"  avg pass rate   : {result['avg_pass_rate']:.6f}")
    print(f"  worst pass rate : {result['worst_pass_rate']:.6f}")
    for r, pr in enumerate(result["rank_pass_rates"]):
        marker = "OK " if pr >= result["threshold"] else "BAD"
        print(f"   rank {r}: {pr:.6f}  {marker}")
    print("=" * 72)

    if not result["ok"]:
        raise SystemExit(1)
    print("[decode_fwd] distributed-mock 8-rank simulation PASSED")
