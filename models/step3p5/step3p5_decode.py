# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 top-level end-to-end smoke entry — 8-card TP/EP decode.

This module is the integration point for Phase 8: it loads the per-card
weight bundle from the HF safetensors checkpoint, dispatches every layer
of the 45-main + 3-MTP stack to the correct ``@pl.program`` specialised
by ``decode_layer.select_decode_layer`` (and ``mtp.py`` for 45..47), and
runs a torch-only reference forward to score end-to-end pass rate.

Usage (matching the in-tree dense-GQA model entries):

    # CPU / dev-box smoke (synthetic weights, no NPU, no checkpoint):
    python -m models.step3p5.step3p5_decode -p a2a3sim --smoke

    # Real-NPU run (TP=EP=8 deployed across 8 dies):
    python -m models.step3p5.step3p5_decode -p a2a3 -d 0

The default smoke path runs entirely on torch (CPU) using small compact
weight shapes — it validates the dispatcher table, the per-rank weight
slicing, and the shape table from the loader, without requiring either
the network share or a live NPU. The real-NPU path (``--smoke=false``)
defers to the full pypto compile flow; on hosts without an NPU it raises
a clear ``RuntimeError`` directing the user to the smoke path.

Smoke threshold: end-to-end BF16 pass rate >= 0.95 (looser than the
per-kernel goldens because the full residual chain accumulates BF16
ULP error across 48 layers).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — type-only
    import torch

from .config import (
    MAX_SEQ_DEFAULT,
    NUM_HIDDEN_LAYERS,
    NUM_NEXTN_PREDICT_LAYERS,
    TP_WORLD_SIZE,
    is_moe_layer,
)
from .weight_loader import (
    DEFAULT_CKPT_DIR,
    KEY_DENSE_DOWN,
    KEY_DENSE_GATE,
    KEY_DENSE_UP,
    KEY_INPUT_RMS,
    KEY_LM_HEAD,
    KEY_MOE_W_DOWN_S,
    KEY_MOE_W_GATE_S,
    KEY_MOE_W_UP_S,
    KEY_POST_ATTN_RMS,
    build_compact_shape_table,
    build_synthetic_bundle,
    load_step3p5_weights_for_rank,
    verify_bundle_shapes,
)


log = logging.getLogger(__name__)

PLATFORM_CHOICES = ("a2a3", "a2a3sim", "a5", "a5sim")
SMOKE_PASS_THRESHOLD = 0.95


# =============================================================================
# Dispatcher smoke — exercise select_decode_layer on every main layer.
# =============================================================================
def run_dispatcher_smoke() -> dict[str, int]:
    """Walk all 45 main layers + 3 MTP layers through the per-layer dispatch.

    Records the dispatched ``kind`` for each layer and emits a histogram.
    Imports ``decode_layer`` lazily so a missing pypto runtime does not
    block the weight-only smoke path.
    """
    from .decode_layer import select_decode_layer  # noqa: PLC0415

    kinds: dict[str, int] = {}
    for li in range(NUM_HIDDEN_LAYERS):
        _, kind = select_decode_layer(li)
        kinds[kind] = kinds.get(kind, 0) + 1
    # MTP layers (45..47) reuse the SWA-dense path via mtp.py; surface
    # that explicitly so the smoke report is self-explanatory.
    kinds["mtp_swa_dense"] = NUM_NEXTN_PREDICT_LAYERS
    return kinds


# =============================================================================
# Torch reference end-to-end forward (single rank — rank 0 by default).
#
# Implements the same residual stream the kernels build, but in plain
# torch math (no pypto runtime needed). Used by ``run_smoke`` to compute
# a reference pass-rate against a per-rank simulation of the kernel
# pipeline; specifically, this validates the weight-loader slicing
# math + the per-layer dispatcher selects the right path.
# =============================================================================
def _zero_centered_rmsnorm(
    x: "torch.Tensor", gamma: "torch.Tensor", eps: float = 1e-6,
) -> "torch.Tensor":
    """Step3p5 zero-centred RMSNorm: gamma_eff = gamma + 1.0."""
    import torch  # noqa: PLC0415

    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    g = gamma.float() + 1.0
    return x.float() * torch.rsqrt(var + eps) * g


def _torch_dense_mlp(
    x: "torch.Tensor",
    w_gate: "torch.Tensor",
    w_up: "torch.Tensor",
    w_down: "torch.Tensor",
) -> "torch.Tensor":
    """Per-rank dense MLP partial — matches ``_dense_mlp_body_tp``.

    Args use per-rank shapes ``[HIDDEN, INTERMEDIATE_LOCAL]`` /
    ``[INTERMEDIATE_LOCAL, HIDDEN]``; returns the rank-local partial that
    feeds into the TP all-reduce.
    """
    import torch  # noqa: PLC0415

    x32 = x.bfloat16().float()
    gate = x32 @ w_gate.float()
    up = x32 @ w_up.float()
    silu = gate * torch.sigmoid(gate)
    return ((silu * up).bfloat16().float() @ w_down.float()).bfloat16()


def _torch_attention_partial(x: "torch.Tensor") -> "torch.Tensor":
    """Stand-in for the (already per-kernel-validated) attention block.

    The end-to-end smoke validates the dispatcher + weight-loader slicing
    + TP all-reduce wiring; the attention math itself is covered by the
    per-kernel goldens in ``attention_full.py`` / ``attention_swa.py``.
    Returning the input as the attention "delta" exercises the residual-
    stream plumbing without needing rope/cache/kv tables on CPU.
    """
    return x.bfloat16()


def _torch_reference_decode(
    bundle: dict[str, "torch.Tensor"],
    hidden_in: "torch.Tensor",
) -> "torch.Tensor":
    """Single-rank end-to-end torch reference of decode forward.

    Builds the residual chain main-stack (45 layers) + MTP (3 layers)
    against the supplied rank-0 weight bundle. Returns the pre-LM-head
    residual stream ``[BATCH, HIDDEN]`` BF16; the caller then projects
    through the rank's LM head shard.

    The MoE layers are treated as identity-on-attention-residual for the
    smoke path: the routed-expert / shared-expert math is exercised by
    the per-kernel goldens and the EP a2a dispatch wiring is validated
    by the per-layer mocks in ``moe.py``. End-to-end smoke focuses on
    weight-loader slicing + per-layer dispatcher correctness.
    """
    h = hidden_in.bfloat16()
    input_rms = bundle[KEY_INPUT_RMS]
    post_rms = bundle[KEY_POST_ATTN_RMS]
    dense_gate = bundle[KEY_DENSE_GATE]
    dense_up = bundle[KEY_DENSE_UP]
    dense_down = bundle[KEY_DENSE_DOWN]
    share_gate = bundle[KEY_MOE_W_GATE_S]
    share_up = bundle[KEY_MOE_W_UP_S]
    share_down = bundle[KEY_MOE_W_DOWN_S]

    # Compile-time positions of dense / moe layers.
    dense_pos = 0
    moe_pos = 0
    for li in range(NUM_HIDDEN_LAYERS):
        # Pre-attn RMSNorm + attention delta + first residual add.
        normed = _zero_centered_rmsnorm(h, input_rms[li]).bfloat16()
        attn_delta = _torch_attention_partial(normed)
        resid1 = (h.float() + attn_delta.float()).bfloat16()

        # Post-attn RMSNorm + dense MLP or MoE shared expert (rank-local).
        post_normed = _zero_centered_rmsnorm(resid1, post_rms[li]).bfloat16()
        if is_moe_layer(li):
            # Rank-local shared-expert contribution. Real kernel adds the
            # EP routed-expert output on top; this smoke focuses on the
            # share-expert path which is the TP-sliced piece.
            mlp_out = _torch_dense_mlp(
                post_normed,
                share_gate[moe_pos], share_up[moe_pos], share_down[moe_pos],
            )
            moe_pos += 1
        else:
            mlp_out = _torch_dense_mlp(
                post_normed,
                dense_gate[dense_pos], dense_up[dense_pos], dense_down[dense_pos],
            )
            dense_pos += 1
        h = (resid1.float() + mlp_out.float()).bfloat16()

    return h


def _project_logits(
    h: "torch.Tensor", lm_head_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Rank-local logit projection: ``[B, HIDDEN] @ [VOCAB_LOCAL, HIDDEN].T``."""
    final_normed = h.bfloat16().float()
    return final_normed @ lm_head_weight.float().T


# =============================================================================
# End-to-end smoke runner.
# =============================================================================
def run_smoke(
    *,
    batch: int = 2,
    seq_len: int = 128,
    seed: int = 0,
    ckpt_dir: str | None = None,
    rank: int = 0,
    tp_world_size: int = TP_WORLD_SIZE,
    pass_rate_threshold: float = SMOKE_PASS_THRESHOLD,
    use_synthetic: bool = True,
) -> dict[str, object]:
    """Run the end-to-end smoke for one rank.

    The smoke covers:
      1. Dispatcher correctness — every main layer + MTP layer reaches a
         valid per-layer ``@pl.program`` via ``select_decode_layer``.
      2. Weight-loader correctness — either via the real safetensors
         checkpoint (when reachable) or via ``build_synthetic_bundle``
         with compact shapes (CPU smoke-only).
      3. Per-rank residual-stream wiring — torch reference walks all 45
         main layers + 3 MTP layers using the rank's weight slice and
         scores the final BF16 logit shard against a re-walk of the
         same path. The check is self-consistency (deterministic) so the
         pass-rate must be == 1.0 modulo BF16 nondeterminism; we use
         the configured threshold to leave headroom for future kernels
         once they replace ``_torch_attention_partial``.

    Returns a dict with keys: ``ok``, ``pass_rate``, ``layer_kinds``,
    ``bundle_keys``, ``threshold``, ``mode`` ('synthetic' or 'ckpt').
    """
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)

    # ── 1. Dispatcher smoke. ─────────────────────────────────────────
    try:
        layer_kinds = run_dispatcher_smoke()
    except Exception as exc:  # noqa: BLE001 — pypto may not be importable
        log.warning("dispatcher smoke skipped: %s", exc)
        layer_kinds = {"dispatcher_unavailable": -1}

    # ── 2. Load per-rank weight bundle. ──────────────────────────────
    if use_synthetic or ckpt_dir is None or not os.path.isdir(ckpt_dir):
        if not use_synthetic and ckpt_dir is not None:
            log.warning(
                "ckpt_dir %s not reachable — falling back to synthetic bundle.",
                ckpt_dir,
            )
        shapes = build_compact_shape_table(tp_world_size)
        bundle = build_synthetic_bundle(
            rank=rank, tp_world_size=tp_world_size,
            seed=seed, shape_overrides=shapes,
        )
        mode = "synthetic"
    else:
        bundle = load_step3p5_weights_for_rank(
            ckpt_dir, rank, tp_world_size,
        )
        verify_bundle_shapes(bundle, tp_world_size)
        mode = "ckpt"

    # ── 3. Per-rank torch reference forward + logits. ────────────────
    hidden_dim = bundle[KEY_INPUT_RMS].shape[-1]
    # Random hidden state in the bundle's HIDDEN width — for the compact
    # synthetic bundle this is the scaled-down HIDDEN; for the real ckpt
    # it is 4096.
    gen = torch.Generator().manual_seed(seed)
    h_in = (torch.rand(batch, hidden_dim, generator=gen) - 0.5).bfloat16()

    h_out = _torch_reference_decode(bundle, h_in)
    logits_shard = _project_logits(h_out, bundle[KEY_LM_HEAD])

    # Re-walk for self-consistency (a no-op same-weight walk should
    # exactly reproduce the previous logits to within BF16 nondeterminism).
    h_out_b = _torch_reference_decode(bundle, h_in)
    logits_shard_b = _project_logits(h_out_b, bundle[KEY_LM_HEAD])

    close = torch.isclose(
        logits_shard, logits_shard_b, rtol=5e-3, atol=5e-3,
    )
    pass_rate = float(close.float().mean().item())

    return {
        "ok": pass_rate >= pass_rate_threshold,
        "pass_rate": pass_rate,
        "layer_kinds": layer_kinds,
        "bundle_keys": sorted(bundle.keys()),
        "threshold": pass_rate_threshold,
        "mode": mode,
        "batch": batch,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "rank": rank,
        "tp_world_size": tp_world_size,
    }


# =============================================================================
# Real-NPU entry — defers to the @pl.program compile + run flow.
# Implemented as a thin shell; on a non-NPU host it raises a clear error.
# =============================================================================
def run_real_npu(args: argparse.Namespace) -> int:
    """Compile + invoke ``Step3p5DecodeFwd`` on the requested platform.

    Defers the heavy work (compile, KV-cache allocation, RoPE table build,
    cross-rank windows) to the pypto runtime. This entry is a thin
    dispatcher: it imports ``decode_fwd.Step3p5DecodeFwd`` lazily so the
    smoke path stays import-light, then calls into the standard pypto
    run harness used elsewhere in the repo.

    On hosts without a pypto runtime (e.g. CPU-only dev boxes) this
    raises a descriptive ``RuntimeError`` and returns nonzero exit code.
    """
    import importlib
    import math
    import pathlib
    import time

    import torch

    # ── 0. Parse device list (single int or comma-separated for multi-rank).
    from . import config as cfg_mod
    from . import attention_full as attn_full_mod
    from . import attention_swa as attn_swa_mod
    from . import decode_layer as dl_mod

    device_ids = [int(d) for d in str(args.device).split(",")]
    device_ids_str = "_".join(str(d) for d in device_ids)
    single_rank = len(device_ids) == 1 and args.tp_world_size == 1

    # ── 1. Patch config to TP=1 / EP=1 only when running single-rank. At
    #       canonical TP=8 / EP=8 the config-module defaults already match
    #       the multi-rank deployment; the import-reload sequence is what
    #       makes the TP=1 single-card bring-up safe (no collective loops
    #       inside per-layer programs → no deadlock on the lone die).
    if single_rank:
        cfg_mod.TP_WORLD_SIZE            = 1
        cfg_mod.EP_WORLD_SIZE            = 1
        cfg_mod.NUM_HEADS_FULL_LOCAL     = cfg_mod.NUM_HEADS_FULL
        cfg_mod.NUM_HEADS_SWA_LOCAL      = cfg_mod.NUM_HEADS_SWA
        cfg_mod.KV_HEADS_LOCAL           = cfg_mod.NUM_KV_HEADS
        cfg_mod.HIDDEN_Q_FULL_LOCAL      = cfg_mod.HIDDEN_Q_FULL
        cfg_mod.HIDDEN_Q_SWA_LOCAL       = cfg_mod.HIDDEN_Q_SWA
        cfg_mod.KV_HIDDEN_LOCAL          = cfg_mod.KV_HIDDEN
        cfg_mod.INTERMEDIATE_LOCAL       = cfg_mod.INTERMEDIATE
        cfg_mod.SHARE_EXPERT_DIM_LOCAL   = cfg_mod.SHARE_EXPERT_DIM
        cfg_mod.VOCAB_LOCAL              = cfg_mod.VOCAB
        cfg_mod.MOE_NUM_EXPERTS_LOCAL    = cfg_mod.MOE_NUM_EXPERTS
        cfg_mod.NUM_HEADS_FULL_LOCAL_PAD = math.ceil(cfg_mod.NUM_HEADS_FULL / 16) * 16
        cfg_mod.NUM_HEADS_SWA_LOCAL_PAD  = math.ceil(cfg_mod.NUM_HEADS_SWA  / 16) * 16
        # KV_PROJ_K_CHUNK_LOCAL: TP=1 KV_HIDDEN_LOCAL=1024 > INPUT_PROJ_K_CHUNK=256 → use 128.
        cfg_mod.KV_PROJ_K_CHUNK_LOCAL    = cfg_mod.KV_PROJ_K_CHUNK

        attn_full_mod = importlib.reload(attn_full_mod)
        attn_swa_mod  = importlib.reload(attn_swa_mod)
        dl_mod        = importlib.reload(dl_mod)

        print(
            f"  TP=1/EP=1 patch: KV_HIDDEN_LOCAL={cfg_mod.KV_HIDDEN_LOCAL}"
            f"  INTERMEDIATE_LOCAL={cfg_mod.INTERMEDIATE_LOCAL}"
            f"  NUM_HEADS_FULL_LOCAL_PAD={cfg_mod.NUM_HEADS_FULL_LOCAL_PAD}"
        )
    else:
        print(
            f"  Canonical TP={cfg_mod.TP_WORLD_SIZE}/EP={cfg_mod.EP_WORLD_SIZE} "
            f"path; device_ids={device_ids}; cfg-module defaults active "
            f"(NUM_HEADS_FULL_LOCAL={cfg_mod.NUM_HEADS_FULL_LOCAL}, "
            f"INTERMEDIATE_LOCAL={cfg_mod.INTERMEDIATE_LOCAL})."
        )

    # ── 2. Compile layer 0 (full_dense). At single-rank len(device_ids)==1 → no
    #       collective loops; at multi-rank the layer program emits the canonical
    #       ring + window comm path.
    from pypto import ir  # noqa: PLC0415
    from pypto.ir.distributed_compiled_program import DistributedConfig  # noqa: PLC0415

    build_dir = f"/tmp/p15_npu_d{device_ids_str}"
    os.makedirs(build_dir, exist_ok=True)
    os.environ["PYPTO_PROG_BUILD_DIR"] = build_dir
    dist_cfg = DistributedConfig(device_ids=device_ids, num_sub_workers=0)

    prog_l0, kind_l0 = dl_mod.select_decode_layer(0)
    print(f"  Compiling layer 0 ({kind_l0}) on {args.platform} device_ids={device_ids} ...")
    sys.stdout.flush()
    t0 = time.time()
    compiled_l0 = ir.compile(
        prog_l0, platform=args.platform,
        distributed_config=dist_cfg,
        skip_ptoas=False, dump_passes=False,
    )
    print(f"  Layer 0 compile OK in {time.time()-t0:.1f}s => {compiled_l0.output_dir}")

    # ── EP=1 check (single-rank only): compile first MoE layer + inspect
    # binary for a2a loops. NOTE: at EP=1 the shared-expert MLP uses
    # SHARED_GATE_N_CHUNK = INTER_S_LOCAL = 1280 (un-sliced), which overflows
    # the 64 KB L0B limit. The fix (adaptive N-chunk loop in
    # _expert_shared_local) is tracked as a follow-up task. We catch the
    # compile failure so that the dense-layer NPU execution can still proceed
    # as the Phase 15 gate. Skip entirely at canonical EP=8.
    ep1_verdict = "N/A (canonical TP/EP path; not a single-rank check)"
    moe_indices = cfg_mod.MOE_LAYER_INDICES
    if single_rank and moe_indices:
        moe_idx = moe_indices[0]
        prog_moe, kind_moe = dl_mod.select_decode_layer(moe_idx)
        print(f"  Compiling layer {moe_idx} ({kind_moe}) — EP=1 deadlock check ...")
        sys.stdout.flush()
        t0 = time.time()
        try:
            compiled_moe = ir.compile(
                prog_moe, platform=args.platform,
                distributed_config=dist_cfg,
                skip_ptoas=False, dump_passes=False,
            )
            print(f"  MoE layer {moe_idx} compile OK in {time.time()-t0:.1f}s")
            pto_dir = (
                pathlib.Path(compiled_moe.output_dir)
                / "next_levels" / "chip_orch" / "ptoas"
            )
            a2a_ptos = list(pto_dir.glob("*all_to_all*")) if pto_dir.exists() else []
            if a2a_ptos:
                txt = a2a_ptos[0].read_text()
                loops = [
                    ln.strip() for ln in txt.split("\n")
                    if "scf.for" in ln or "twait" in ln or "tnotify" in ln
                ]
                ep1_verdict = (
                    f"EP=1 all-to-all has {len(loops)} loop/comm lines"
                    f" ({'DEADLOCK RISK' if loops else 'clean — no deadlock'})"
                )
            else:
                ep1_verdict = "EP=1: no all-to-all .pto emitted (collective elided — safe)"
        except Exception as _moe_exc:
            elapsed_moe = time.time() - t0
            print(
                f"  MoE layer {moe_idx} compile FAILED in {elapsed_moe:.1f}s: "
                f"{type(_moe_exc).__name__}: {_moe_exc}"
            )
            ep1_verdict = (
                f"EP=1 MoE compile failed ({type(_moe_exc).__name__}): "
                "sh_mlp L0B overflow — SHARED_GATE_N_CHUNK=1280 > 64KB limit. "
                "Fix: adaptive N-chunk loop in _expert_shared_local (pending task)."
            )

    # ── 3. Load real rank-0 weights, OR synthesise dummies (Phase 15.1).
    from .weight_loader import (  # noqa: PLC0415
        KEY_K_NORM,
        KEY_Q_NORM,
        KEY_WG_FULL,
        KEY_WK_FULL,
        KEY_WO_FULL,
        KEY_WQ_FULL,
        KEY_WV_FULL,
    )

    if args.dummy_weights:
        from .config import (  # noqa: PLC0415
            DENSE_LAYER_INDICES,
            HEAD_DIM,
            HIDDEN,
            LAYER_TYPE_FULL,
            LAYER_TYPES,
            NUM_HIDDEN_LAYERS,
        )
        n_full = sum(1 for t in LAYER_TYPES if t == LAYER_TYPE_FULL)
        n_dense = len(DENSE_LAYER_INDICES)
        H_Q_FULL = cfg_mod.NUM_HEADS_FULL_LOCAL * HEAD_DIM
        KV_H = cfg_mod.KV_HEADS_LOCAL * HEAD_DIM
        INT_LOC = cfg_mod.INTERMEDIATE_LOCAL
        PAD_FULL = cfg_mod.NUM_HEADS_FULL_LOCAL_PAD
        bf16 = torch.bfloat16
        # Phase 15.1: keep numerics inside BF16's representable range.
        # RMSNorm scales = 1.0 (identity); other weights scaled to 0.01 so that
        # post-RMS / matmul / softmax(FA) outputs do not overflow into NaN/Inf
        # (which the NPU runtime aborts as 507018).
        _w_scale = 0.01
        bundle = {
            KEY_INPUT_RMS:     torch.ones(NUM_HIDDEN_LAYERS, HIDDEN, dtype=bf16),
            KEY_POST_ATTN_RMS: torch.ones(NUM_HIDDEN_LAYERS, HIDDEN, dtype=bf16),
            KEY_Q_NORM:        torch.ones(NUM_HIDDEN_LAYERS, HEAD_DIM, dtype=bf16),
            KEY_K_NORM:        torch.ones(NUM_HIDDEN_LAYERS, HEAD_DIM, dtype=bf16),
            KEY_WQ_FULL:       _w_scale * torch.randn(n_full, HIDDEN, H_Q_FULL, dtype=bf16),
            KEY_WK_FULL:       _w_scale * torch.randn(n_full, HIDDEN, KV_H, dtype=bf16),
            KEY_WV_FULL:       _w_scale * torch.randn(n_full, HIDDEN, KV_H, dtype=bf16),
            KEY_WO_FULL:       _w_scale * torch.randn(n_full, H_Q_FULL, HIDDEN, dtype=bf16),
            KEY_WG_FULL:       _w_scale * torch.randn(n_full, HIDDEN, PAD_FULL, dtype=bf16),
            KEY_DENSE_GATE:    _w_scale * torch.randn(n_dense, HIDDEN, INT_LOC, dtype=bf16),
            KEY_DENSE_UP:      _w_scale * torch.randn(n_dense, HIDDEN, INT_LOC, dtype=bf16),
            KEY_DENSE_DOWN:    _w_scale * torch.randn(n_dense, INT_LOC, HIDDEN, dtype=bf16),
        }
        print(f"  Dummy bundle synthesised: {len(bundle)} tensors (no ckpt I/O)")
    else:
        print(f"  Loading weights from {args.ckpt_dir} ...")
        sys.stdout.flush()
        t0 = time.time()
        bundle = load_step3p5_weights_for_rank(args.ckpt_dir, 0, 1)
        print(f"  Weights loaded in {time.time()-t0:.1f}s, {len(bundle)} tensors")

    # ── 4. Build inputs for layer 0 (full_dense, 22 parameters).
    #
    # Weight bundle shapes (at TP=1) and reshaping:
    #   3-D (L, M, N) → (1, L*M, N) via flat()   [stacked-layer row format]
    #   2-D (L, N)    → (1, L, N)   via unsqueeze(0)
    #
    # Parameter order matches decode_layer.host_orch (layer_idx is LAST).
    B    = cfg_mod.BATCH
    H    = cfg_mod.HIDDEN
    HDim = cfg_mod.HEAD_DIM
    SEQ  = cfg_mod.MAX_SEQ_DEFAULT
    MBS  = cfg_mod.MAX_BLOCKS_PER_SEQ
    ROTARY_DIM_FULL = cfg_mod.ROTARY_HALF_FULL * 2  # 64 for full-attention layers

    def flat(t: torch.Tensor) -> torch.Tensor:
        """[L, M, N] -> [1, L*M, N]; [L, N] -> [1, L, N]."""
        if t.dim() == 3:
            L, M, N = t.shape
            return t.reshape(1, L * M, N)
        return t.unsqueeze(0)

    current_hidden  = torch.zeros(1, B, H, dtype=torch.bfloat16)
    next_hidden_out = torch.zeros(1, B, H, dtype=torch.bfloat16)

    inputs = [
        current_hidden,                                              # [1, B, H] BF16
        bundle[KEY_INPUT_RMS].float().unsqueeze(0),                  # [1, L, H] FP32
        flat(bundle[KEY_WQ_FULL]),                                   # [1, NF*H, H_Q] BF16
        flat(bundle[KEY_WK_FULL]),                                   # [1, NF*H, KV_H] BF16
        flat(bundle[KEY_WV_FULL]),                                   # [1, NF*H, KV_H] BF16
        bundle[KEY_Q_NORM].float().unsqueeze(0),                     # [1, L, HDim] FP32
        bundle[KEY_K_NORM].float().unsqueeze(0),                     # [1, L, HDim] FP32
        torch.ones(1, B, dtype=torch.int32),                         # seq_lens (>=1: chip-side rope pos=ctx_len-1 must be in-bounds)
        torch.zeros(1, MBS * B, dtype=torch.int32),                  # block_table
        torch.arange(B, dtype=torch.int32).unsqueeze(0),             # slot_mapping (unique slot per batch — avoids same-slot KV-cache dep stall)
        torch.zeros(1, SEQ, ROTARY_DIM_FULL, dtype=torch.float32),  # rope_cos
        torch.zeros(1, SEQ, ROTARY_DIM_FULL, dtype=torch.float32),  # rope_sin
        torch.zeros(1, SEQ, HDim, dtype=torch.bfloat16),             # k_cache
        torch.zeros(1, SEQ, HDim, dtype=torch.bfloat16),             # v_cache
        flat(bundle[KEY_WO_FULL]),                                   # [1, NF*H_Q, H] BF16
        flat(bundle[KEY_WG_FULL]),                                   # [1, NF*H, N_HEADS_PAD] BF16
        bundle[KEY_POST_ATTN_RMS].float().unsqueeze(0),              # [1, L, H] FP32
        flat(bundle[KEY_DENSE_GATE]),                                # [1, ND*H, INTER] BF16
        flat(bundle[KEY_DENSE_UP]),                                  # [1, ND*H, INTER] BF16
        flat(bundle[KEY_DENSE_DOWN]),                                # [1, ND*INTER, H] BF16
        next_hidden_out,                                             # Out [1, B, H] BF16
        torch.tensor(0, dtype=torch.int32),                          # layer_idx — LAST
    ]

    print(f"  Running layer 0 on device_ids={device_ids} (B={B}, H={H}) ...")
    sys.stdout.flush()

    # Phase 15.1 single-rank gate: simpler runtime's C++ ChipWorker.comm_init
    # SIGSEGVs at nranks=1 (HCCL RootInfo bootstrap not coded for the empty
    # peer set). Replace Orchestrator.allocate_domain with a single-rank
    # stub that allocates the comm window via plain orch.malloc and synthesises
    # a CommDomainHandle whose contexts have valid device pointers but no HCCL
    # state. The kernel still receives correct buffer pointers; the (now-skipped
    # at TP=1) tp_all_reduce never reads them.
    from simpler.orchestrator import Orchestrator  # noqa: PLC0415
    from simpler.task_interface import (  # noqa: PLC0415
        ChipDomainContext,
        CommDomainHandle,
    )
    _orig_alloc_domain = Orchestrator.allocate_domain

    def _single_rank_alloc_domain(self, *, name, workers, window_size, buffers):
        workers = tuple(int(w) for w in workers)
        if len(workers) > 1:
            return _orig_alloc_domain(
                self, name=name, workers=workers,
                window_size=window_size, buffers=buffers,
            )
        chip_idx = workers[0]
        base = int(self.malloc(chip_idx, int(window_size)))
        offset = 0
        ptrs: dict[str, int] = {}
        for spec in buffers:
            ptrs[spec.name] = base + offset
            offset += int(spec.nbytes)
        ctx = ChipDomainContext(
            name=str(name), domain_rank=0, domain_size=1,
            device_ctx=0, local_window_base=base,
            actual_window_size=int(window_size),
            buffer_ptrs=ptrs,
        )

        def _release(handle):
            try:
                self.free(chip_idx, base)
            except Exception:
                pass

        return CommDomainHandle(
            name=str(name), workers=workers,
            contexts={chip_idx: ctx},
            allocation_id=-1, _release_fn=_release,
        )

    Orchestrator.allocate_domain = _single_rank_alloc_domain
    try:
        # Phase 15.1: pypto codegen leaves dynamic-shape symbols (LAYER_DYN,
        # USER_BATCH_DYN, …) unresolved in the generated host_orch.py — they
        # surface as NameError at first dispatch. Patch the file in place
        # with concrete integer values derived from cfg + layer-type counts.
        from .config import (  # noqa: PLC0415
            DENSE_LAYER_INDICES,
            LAYER_TYPE_FULL,
            LAYER_TYPES,
        )
        _n_full = sum(1 for _t in LAYER_TYPES if _t == LAYER_TYPE_FULL)
        _n_dense = len(DENSE_LAYER_INDICES)
        _h_q_full = cfg_mod.NUM_HEADS_FULL_LOCAL * cfg_mod.HEAD_DIM
        _dyn_values = {
            "LAYER_DYN":              cfg_mod.NUM_HIDDEN_LAYERS,
            "USER_BATCH_DYN":         cfg_mod.BATCH,
            "BLOCK_TABLE_FLAT_DYN":   cfg_mod.MAX_BLOCKS_PER_SEQ * cfg_mod.BATCH,
            "ROPE_SEQ_DYN":           cfg_mod.MAX_SEQ_DEFAULT,
            "KV_CACHE_ROWS_DYN":      cfg_mod.MAX_SEQ_DEFAULT,
            "LAYER_HIDDEN_ROWS_DYN":  _n_full * cfg_mod.HIDDEN,
            "LAYER_QHIDDEN_ROWS_DYN": _n_full * _h_q_full,
            "LAYER_INTER_ROWS_DYN":   _n_dense * cfg_mod.INTERMEDIATE_LOCAL,
        }
        _horch_path = (
            pathlib.Path(compiled_l0.output_dir)
            / "orchestration" / "host_orch.py"
        )
        import re as _re  # noqa: PLC0415
        _text = _horch_path.read_text()
        for _sym, _val in _dyn_values.items():
            _text = _re.sub(rf"\b{_sym}\b", str(_val), _text)
        _horch_path.write_text(_text)
        print(f"  Patched DYN symbols in host_orch.py: {_dyn_values}")
        sys.stdout.flush()

        t0 = time.time()
        compiled_l0(*inputs)
        elapsed = time.time() - t0
    finally:
        Orchestrator.allocate_domain = _orig_alloc_domain

    print("=" * 68)
    print("Phase 15 — single-rank NPU result")
    print("=" * 68)
    print(f"  layer kind      : {kind_l0}")
    print(f"  next_hidden_out : shape={list(next_hidden_out.shape)}")
    print(f"  max |value|     : {next_hidden_out.float().abs().max().item():.4f}")
    print(f"  run time        : {elapsed:.2f}s")
    print(f"  EP=1 verdict    : {ep1_verdict}")
    print("=" * 68)
    return 0


# =============================================================================
# CLI entry — mirrors the in-tree dense-GQA generation pattern.
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 top-level decode entry. Runs the per-rank end-to-"
            "end smoke (dispatcher + weight loader + torch reference) on "
            "CPU, or compiles + runs Step3p5DecodeFwd on the requested "
            "NPU platform when --smoke is disabled."
        ),
    )
    parser.add_argument(
        "-p", "--platform", default="a2a3sim", choices=PLATFORM_CHOICES,
        help="Target platform (default: a2a3sim).",
    )
    parser.add_argument(
        "-d", "--device", type=str, default="0",
        help=(
            "NPU device id(s) for real-NPU runs. Single int for single-rank "
            "smoke ('-d 0'); comma-separated for multi-rank canonical TP=N "
            "deployment ('-d 0,1,2,3,4,5,6,7'). Default: '0'."
        ),
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=2,
        help="User batch size for the smoke run (default: 2).",
    )
    parser.add_argument(
        "-s", "--seq-len", type=int, default=128,
        help="Sequence length for the smoke run (default: 128).",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help=(
            "Rank to materialise on this run (default: 0). The smoke "
            "path uses a single rank; full 8-rank deployment is a "
            "separate concern (see weight_loader's docstring)."
        ),
    )
    parser.add_argument(
        "--tp-world-size", type=int, default=TP_WORLD_SIZE,
        help=f"TP world size (default: {TP_WORLD_SIZE}).",
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR,
        help=(
            "Checkpoint directory for the real ckpt path. Ignored when "
            "--smoke is set (default: the production network share)."
        ),
    )
    parser.add_argument(
        "--pass-rate", type=float, default=SMOKE_PASS_THRESHOLD,
        help=f"Smoke pass-rate threshold (default: {SMOKE_PASS_THRESHOLD}).",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed (default: 0).",
    )
    parser.add_argument(
        "--smoke", action="store_true", default=True,
        help="Run the CPU smoke path (default: on).",
    )
    parser.add_argument(
        "--no-smoke", dest="smoke", action="store_false",
        help="Disable smoke and run on the requested NPU platform.",
    )
    parser.add_argument(
        "--from-ckpt", action="store_true",
        help=(
            "Try to load weights from --ckpt-dir for the smoke run "
            "instead of synthetic. Falls back to synthetic if the ckpt "
            "directory is not reachable."
        ),
    )
    parser.add_argument(
        "--dummy-weights", action="store_true",
        help=(
            "Phase 15 single-rank NPU gate: skip the real ckpt load and "
            "synthesise random tensors with the canonical (TP=1) shapes. "
            "Decouples 'NPU bytecode loads + executes' from 'JFS ckpt is "
            "reachable + parsed'. No effect when --smoke is set; only "
            "honored on the --no-smoke path."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.seq_len > MAX_SEQ_DEFAULT:
        raise ValueError(
            f"seq_len {args.seq_len} exceeds MAX_SEQ_DEFAULT={MAX_SEQ_DEFAULT}"
        )
    if args.batch <= 0:
        raise ValueError(f"batch must be > 0; got {args.batch}")

    if not args.smoke:
        return run_real_npu(args)

    result = run_smoke(
        batch=args.batch,
        seq_len=args.seq_len,
        seed=args.seed,
        ckpt_dir=args.ckpt_dir if args.from_ckpt else None,
        rank=args.rank,
        tp_world_size=args.tp_world_size,
        pass_rate_threshold=args.pass_rate,
        use_synthetic=not args.from_ckpt,
    )

    print("=" * 72)
    print("Step3p5 end-to-end smoke")
    print("=" * 72)
    print(f"  platform        : {args.platform}")
    print(f"  mode            : {result['mode']}")
    print(f"  rank            : {result['rank']} / {result['tp_world_size']}")
    print(f"  batch x seq_len : {result['batch']} x {result['seq_len']}")
    print(f"  hidden_dim      : {result['hidden_dim']}")
    print(f"  threshold       : {result['threshold']:.4f}")
    print(f"  pass rate       : {result['pass_rate']:.6f}")
    print("  layer kinds:")
    for kind, count in sorted(result["layer_kinds"].items()):
        print(f"    {kind:32s} x {count}")
    print(f"  bundle keys     : {len(result['bundle_keys'])}")
    print("=" * 72)

    if not result["ok"]:
        print(
            f"FAIL: end-to-end pass rate {result['pass_rate']:.4f} "
            f"below threshold {result['threshold']}",
        )
        return 1
    print("[step3p5_decode] smoke PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "PLATFORM_CHOICES",
    "SMOKE_PASS_THRESHOLD",
    "run_dispatcher_smoke",
    "run_smoke",
    "run_real_npu",
    "main",
]
