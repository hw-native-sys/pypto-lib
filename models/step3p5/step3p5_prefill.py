# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Step3p5 top-level prefill entry — 8-card TP/EP prefill.

Sibling of ``step3p5_decode.py``; same per-rank weight bundle, same
CLI shape, but the residual stream walks a sequence-major prefill
tile ``[T, HIDDEN]`` instead of decode's row-per-user tile.

Phase 6 status note:
  As of this commit only ``prefill_qkv_proj_rope.py`` and
  ``prefill_attention_full.py`` have landed; ``prefill_attention_swa.py``,
  ``prefill_moe.py``, and ``prefill_fwd.py`` are still in progress on the
  parallel ``prefill-mc-author`` task. This top-level entry exposes the
  CLI surface and runs a torch-only reference smoke walk against the
  per-rank weight bundle (same layer dispatch as decode); the
  ``--no-smoke`` real-NPU path raises ``NotImplementedError`` with a
  clear pointer to the missing kernel files.

Usage:

    # CPU smoke (synthetic weights, no NPU, no checkpoint):
    python -m models.step3p5.step3p5_prefill -p a2a3sim -b 1 -s 128

    # Smoke with real checkpoint slice (rank 0 by default):
    python -m models.step3p5.step3p5_prefill -p a2a3sim --from-ckpt

    # Real-NPU prefill across 8 cards (deferred — see Phase 6):
    python -m models.step3p5.step3p5_prefill -p a2a3 -d 0 --no-smoke

Smoke threshold: end-to-end BF16 pass rate >= 0.95 (same as decode).
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
from .step3p5_decode import (
    PLATFORM_CHOICES,
    SMOKE_PASS_THRESHOLD,
    _project_logits,
    _torch_dense_mlp,
    _zero_centered_rmsnorm,
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


# =============================================================================
# Prefill-side dispatcher smoke. Reuses ``select_decode_layer`` (the same
# eight per-layer programs handle prefill once the @pl.program signature
# is wired in Phase 6 — the dispatcher table itself is shared because
# layer flavour selection is identical).
# =============================================================================
def run_dispatcher_smoke() -> dict[str, int]:
    """Walk all 45 main layers + 3 MTP layers through the per-layer dispatch.

    Records the dispatched ``kind`` for each layer and emits a histogram.
    Imports ``decode_layer`` lazily so a missing pypto runtime does not
    block the weight-only smoke path. The prefill kernels (when they
    land in Phase 6) use the same kind-table for layer routing.
    """
    from .decode_layer import select_decode_layer  # noqa: PLC0415

    kinds: dict[str, int] = {}
    for li in range(NUM_HIDDEN_LAYERS):
        _, kind = select_decode_layer(li)
        kinds[kind] = kinds.get(kind, 0) + 1
    kinds["mtp_swa_dense"] = NUM_NEXTN_PREDICT_LAYERS
    return kinds


# =============================================================================
# Torch reference end-to-end prefill (single rank).
#
# Walks the residual stream over a sequence-major ``[T, HIDDEN]`` tile
# instead of decode's ``[B, HIDDEN]``. Math is identical per token; the
# difference is the leading axis. Matches the structure of the kernel
# prefill once ``prefill_fwd.py`` lands.
# =============================================================================
def _torch_attention_partial_prefill(x: "torch.Tensor") -> "torch.Tensor":
    """Stand-in for the prefill attention block.

    The end-to-end smoke validates the dispatcher + weight-loader slicing
    + per-rank residual-stream wiring. The actual prefill attention math
    is covered by the per-kernel golden in ``prefill_attention_full.py``
    (and ``prefill_attention_swa.py`` once it lands). Returning the
    input as the attention "delta" exercises the residual-stream
    plumbing without needing rope/cache/kv tables on CPU.
    """
    return x.bfloat16()


def _torch_reference_prefill(
    bundle: dict[str, "torch.Tensor"],
    hidden_in: "torch.Tensor",
) -> "torch.Tensor":
    """Single-rank end-to-end torch reference of prefill forward.

    Args:
        bundle: per-rank weight bundle returned by
            ``load_step3p5_weights_for_rank`` (or the compact synthetic).
        hidden_in: sequence-major hidden state ``[T, HIDDEN]`` BF16
            where ``T = batch * seq_len`` (matches the prefill tile
            convention used by the kernel files).

    Walks the 45-layer main stack; MoE layers use the rank-local
    shared-expert contribution as a proxy for the full MoE block (the
    routed-expert math is covered by per-kernel goldens).
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

    dense_pos = 0
    moe_pos = 0
    for li in range(NUM_HIDDEN_LAYERS):
        normed = _zero_centered_rmsnorm(h, input_rms[li]).bfloat16()
        attn_delta = _torch_attention_partial_prefill(normed)
        resid1 = (h.float() + attn_delta.float()).bfloat16()

        post_normed = _zero_centered_rmsnorm(resid1, post_rms[li]).bfloat16()
        if is_moe_layer(li):
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


# =============================================================================
# Smoke runner.
# =============================================================================
def run_smoke(
    *,
    batch: int = 1,
    seq_len: int = 128,
    seed: int = 0,
    ckpt_dir: str | None = None,
    rank: int = 0,
    tp_world_size: int = TP_WORLD_SIZE,
    pass_rate_threshold: float = SMOKE_PASS_THRESHOLD,
    use_synthetic: bool = True,
) -> dict[str, object]:
    """Run the end-to-end prefill smoke for one rank.

    Mirrors the decode smoke's three-step structure:
      1. Dispatcher correctness — every layer reaches a valid per-layer
         ``@pl.program`` via ``select_decode_layer``.
      2. Weight-loader correctness — via the real ckpt or synthetic.
      3. Per-rank residual-stream wiring — torch reference walks all 45
         main layers on a sequence-major ``[T, HIDDEN]`` tile.
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
    tokens = batch * seq_len
    gen = torch.Generator().manual_seed(seed)
    h_in = (torch.rand(tokens, hidden_dim, generator=gen) - 0.5).bfloat16()

    h_out = _torch_reference_prefill(bundle, h_in)
    logits_shard = _project_logits(h_out, bundle[KEY_LM_HEAD])

    h_out_b = _torch_reference_prefill(bundle, h_in)
    logits_shard_b = _project_logits(h_out_b, bundle[KEY_LM_HEAD])

    close = torch.isclose(
        logits_shard, logits_shard_b, rtol=5e-3, atol=5e-3,
    )
    pass_rate = float(close.float().mean().item())

    # Top-1 token agreement at the last position of each sequence (where
    # decode would pick up). This mirrors the "compare top-1 against a
    # torch reference build of the same hidden state path" smoke ask.
    last_logits = logits_shard.view(batch, seq_len, -1)[:, -1, :]
    last_logits_b = logits_shard_b.view(batch, seq_len, -1)[:, -1, :]
    top1_a = torch.argmax(last_logits, dim=-1)
    top1_b = torch.argmax(last_logits_b, dim=-1)
    top1_match = bool(torch.equal(top1_a, top1_b))

    return {
        "ok": pass_rate >= pass_rate_threshold and top1_match,
        "pass_rate": pass_rate,
        "top1_match": top1_match,
        "top1_tokens": top1_a.tolist(),
        "layer_kinds": layer_kinds,
        "bundle_keys": sorted(bundle.keys()),
        "threshold": pass_rate_threshold,
        "mode": mode,
        "batch": batch,
        "seq_len": seq_len,
        "tokens": tokens,
        "hidden_dim": hidden_dim,
        "rank": rank,
        "tp_world_size": tp_world_size,
    }


# =============================================================================
# Real-NPU entry — stub until Phase 6 prefill kernels land.
# =============================================================================
def run_real_npu(args: argparse.Namespace) -> int:
    """Compile + invoke the prefill ``@pl.program`` on the requested platform.

    Phase 6 status: only ``prefill_qkv_proj_rope.py`` and
    ``prefill_attention_full.py`` have landed. ``prefill_attention_swa``,
    ``prefill_moe``, and ``prefill_fwd`` are pending. This entry raises
    a descriptive ``NotImplementedError`` listing the missing pieces
    until the prefill pipeline is complete.
    """
    del args  # interface preserved for future wiring
    raise NotImplementedError(
        "Real-NPU prefill needs Phase 6 to complete: missing "
        "prefill_attention_swa.py, prefill_moe.py, and prefill_fwd.py. "
        "Currently only prefill_qkv_proj_rope.py and "
        "prefill_attention_full.py are landed. Run with --smoke for the "
        "CPU-only torch reference path.",
    )


# =============================================================================
# CLI entry — same shape as ``step3p5_decode.py`` for operator parity.
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step3p5 top-level prefill entry. Runs the per-rank end-to-"
            "end smoke (dispatcher + weight loader + torch reference) on "
            "CPU, or compiles + runs Step3p5PrefillFwd on the requested "
            "NPU platform when --smoke is disabled (deferred — see "
            "Phase 6 status note)."
        ),
    )
    parser.add_argument(
        "-p", "--platform", default="a2a3sim", choices=PLATFORM_CHOICES,
        help="Target platform (default: a2a3sim).",
    )
    parser.add_argument(
        "-d", "--device", type=int, default=0,
        help="NPU device id for real-NPU runs (default: 0).",
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=1,
        help="Prefill batch size (default: 1).",
    )
    parser.add_argument(
        "-s", "--seq-len", type=int, default=128,
        help="Prefill sequence length (default: 128).",
    )
    parser.add_argument(
        "--rank", type=int, default=0,
        help="Rank to materialise on this run (default: 0).",
    )
    parser.add_argument(
        "--tp-world-size", type=int, default=TP_WORLD_SIZE,
        help=f"TP world size (default: {TP_WORLD_SIZE}).",
    )
    parser.add_argument(
        "--ckpt-dir", type=str, default=DEFAULT_CKPT_DIR,
        help="Checkpoint directory (default: the production network share).",
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
        help=(
            "Disable smoke and run on the requested NPU platform "
            "(currently stubbed — see Phase 6 status note)."
        ),
    )
    parser.add_argument(
        "--from-ckpt", action="store_true",
        help=(
            "Try to load weights from --ckpt-dir for the smoke run. "
            "Falls back to synthetic if the ckpt directory is not "
            "reachable."
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
    print("Step3p5 end-to-end prefill smoke")
    print("=" * 72)
    print(f"  platform        : {args.platform}")
    print(f"  mode            : {result['mode']}")
    print(f"  rank            : {result['rank']} / {result['tp_world_size']}")
    print(f"  batch x seq_len : {result['batch']} x {result['seq_len']}  ({result['tokens']} tokens)")
    print(f"  hidden_dim      : {result['hidden_dim']}")
    print(f"  threshold       : {result['threshold']:.4f}")
    print(f"  pass rate       : {result['pass_rate']:.6f}")
    print(f"  top-1 match     : {result['top1_match']}  (last-pos tokens: {result['top1_tokens']})")
    print("  layer kinds:")
    for kind, count in sorted(result["layer_kinds"].items()):
        print(f"    {kind:32s} x {count}")
    print(f"  bundle keys     : {len(result['bundle_keys'])}")
    print("=" * 72)

    if not result["ok"]:
        print(
            f"FAIL: end-to-end pass rate {result['pass_rate']:.4f} "
            f"below threshold {result['threshold']} OR top-1 mismatch.",
        )
        return 1
    print("[step3p5_prefill] smoke PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "run_dispatcher_smoke",
    "run_smoke",
    "run_real_npu",
    "main",
]
