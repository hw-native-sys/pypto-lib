# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The block reference against the model's own module, `Qwen3_5GatedDeltaNet`.

`reference.block` is what every kernel of the block is scored against, so it has
to be right before anything is built on it. This runs the `transformers` module
on the same hidden states and weights and compares the two chains at every point
a forward hook can reach: the four projections, the delta rule's output, the
gated norm's output and the block's output.

Two weight sets, both to be kept for every later step:

    python models/qwen3_8_27b/test_block_reference.py            # random, the module's own init
    python models/qwen3_8_27b/test_block_reference.py --weights build_output/qwen3_8_27b/linear_attn_layer0.pt

The second is one real layer (`weights.py` fetches it). Under the module's own
init every chunk's decay underflows, so the random set never carries state from
one chunk to the next; a trained layer's `A_log` and `dt_bias` keep `g` in a
range where about a third of the chunks do, and where a wrong scale, a wrong
eps or a saturating gate shows.

What "match" can mean here. Our chain is float64 end to end. The module, even
with float64 parameters and inputs, computes the gate `g`, the whole delta rule
and the norm's statistics in fp32 by its own code (`.float()` and
`.to(torch.float32)` in `modeling_qwen3_5.py`), so the projections agree to
float64 rounding and everything after the gate agrees to fp32 rounding. The gate
is set just above that floor. `--module-dtype bfloat16` runs the module the way
the model is served; that number is reported, not gated -- it measures the
model's own rounding against the float64 truth, and it is the yardstick for what
the kernels' block-level gate can be.

`--quant` adds the second reference: the block with the W8A8 arithmetic the
projection kernels will implement (`config.GDN_QUANT`; int8 per-channel weights,
per-token int8 hidden states and normed output). It prints what each part of
the scheme costs against the float64 truth -- weights alone, each activation
alone, the whole scheme, and the whole scheme with `in_proj_a/b` in int8 too,
which is the number the decision to keep those bf16 rests on -- and, when a
bf16 module run is requested, against the model as served. Nothing to gate:
the numbers are the finding, and the kernels of Q4b-e are gated against this
chain rather than the truth because of them.

Host only: no NPU, no compile. Needs `transformers` (pinned at 5.17.0 in the
environment; the `qwen3_5` modeling code it carries is byte-identical to `main`
as of 2026-09-11) and runs the module's pure-torch path -- with `fla` or
`causal_conv1d` installed the module would call those instead, and this says so.
"""
from __future__ import annotations

import argparse
import importlib.util
import time

import torch

import reference
import weights as weights_mod
from config import GDN_QUANT, GDN_TILING, QWEN3_8_27B

CHUNK = GDN_TILING.chunk

# The W8A8 ablation: what is int8 in each variant. `weights` names the weight
# set to quantise, or None for bf16.
QUANT_VARIANTS = (
    ("int8 weights only, activations exact", GDN_QUANT.weights, False, False),
    ("per-token int8 x only, weights exact", None, True, False),
    ("per-token int8 y only, weights exact", None, False, True),
    ("W8A8 -- the scheme", GDN_QUANT.weights, True, True),
    ("W8A8 with in_proj_a/b int8 too",
     GDN_QUANT.weights + ("in_proj_a.weight", "in_proj_b.weight"), True, True),
)

# torch's CPU depthwise conv opens a parallel region per channel group, and the
# reference's per-(chunk, head) loops one per small matmul; on a 192-core host
# that overhead is 40 s for a T = 512 conv in float64 against 0.2 s at 16
# threads, and 3x on the reference. Capped, not fixed: a smaller host keeps its
# count.
MAX_THREADS = 32

# Relative Frobenius gate for a float64 (or float32) module. The floor at
# T = 8192 is 1.3e-7 to 2.7e-7 on `o`, `y` and `out` for both weight sets, set
# by the module's own fp32 delta rule; the projections are exact. Dropping the
# qk-norm eps moves the real layer's output by 2.5e-6 and dropping `D^-0.5` by
# 0.27, so the gate sits between the floor and the smallest mistake it is for.
TOL = 1e-6

# Comparison points, in data-flow order: our key, the module's, and how ours is
# laid out to match. The module's z is `[T, H*D]` at the projection and
# `[T*H, D]` at the norm; ours is `[T, H, D]` throughout.
POINTS = (
    ("qkv", "in_proj_qkv", lambda st, t: st["qkv"]),
    ("z", "in_proj_z", lambda st, t: st["z"].reshape(t, -1)),
    ("a", "in_proj_a", lambda st, t: st["a_proj"]),
    ("b", "in_proj_b", lambda st, t: st["b_proj"]),
    ("o", "delta rule", lambda st, t: st["o"].reshape(-1, st["o"].shape[-1])),
    ("y", "norm", lambda st, t: st["y"].reshape(-1, st["y"].shape[-1])),
    ("out", "out_proj", lambda st, t: st["out"]),
)


def module_forward(x: torch.Tensor, w: dict[str, torch.Tensor], dtype: torch.dtype,
                   ) -> dict[str, torch.Tensor]:
    """`Qwen3_5GatedDeltaNet` on `x[None]`, with every comparison point captured."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg = QWEN3_8_27B
    text_config = Qwen3_5TextConfig(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        rms_norm_eps=cfg.rms_norm_eps,
        hidden_act="silu",
        linear_num_value_heads=cfg.linear_num_value_heads,
        linear_num_key_heads=cfg.linear_num_key_heads,
        linear_key_head_dim=cfg.linear_key_head_dim,
        linear_value_head_dim=cfg.linear_value_head_dim,
        linear_conv_kernel_dim=cfg.linear_conv_kernel_dim,
        full_attention_interval=cfg.full_attention_interval,
    )
    module = Qwen3_5GatedDeltaNet(text_config, layer_idx=0)
    module.load_state_dict(w)
    module = module.to(dtype).eval()

    got = {}

    def keep(name):
        def hook(_mod, _inputs, output):
            got[name] = output.detach()[0]
        return hook

    def keep_norm(_mod, inputs, output):
        got["o"] = inputs[0].detach()
        got["y"] = output.detach()

    for name in ("qkv", "z", "a", "b"):
        getattr(module, "in_proj_" + name).register_forward_hook(keep(name))
    module.norm.register_forward_hook(keep_norm)
    with torch.no_grad():
        got["out"] = module(x[None].to(dtype))[0]
    return got


def compare(ours: torch.Tensor, theirs: torch.Tensor) -> tuple[float, float]:
    """Relative Frobenius distance and max absolute difference, in float64."""
    a = ours.to(torch.float64)
    b = theirs.to(torch.float64)
    frob = float((a - b).norm() / b.norm())
    return frob, float((a - b).abs().max())


def quant_table(t: int, x: torch.Tensor, w: dict[str, torch.Tensor],
                truth: dict[str, torch.Tensor], bf16_out: torch.Tensor | None) -> None:
    """Every variant of the W8A8 scheme against the truth (and the bf16 model, if run)."""
    print(f"\n[quant] W8A8 chain, T={t}: relative Frobenius against the float64 truth"
          + ("; last column against the bf16 module" if bf16_out is not None else ""))
    head = f"  {'variant':<42} {'o (delta rule)':>15} {'out':>10}"
    print(head + (f" {'out vs bf16':>12}" if bf16_out is not None else ""))
    for name, names, quant_x, quant_y in QUANT_VARIANTS:
        started = time.time()
        wq = reference.quantize_weights(w, names) if names else w
        st = reference.block(x, wq, QWEN3_8_27B, CHUNK, quant_x=quant_x, quant_y=quant_y)
        row = f"  {name:<42} {compare(st['o'], truth['o'])[0]:>15.3e} {compare(st['out'], truth['out'])[0]:>10.3e}"
        if bf16_out is not None:
            row += f" {compare(st['out'], bf16_out)[0]:>12.3e}"
        print(row + f"   ({time.time() - started:.0f}s)")
    y = truth["y"].reshape(t, -1)
    by_channel = y.abs().amax(dim=0)
    by_token = y.abs().amax(dim=1)
    print(f"  y before out_proj: rms {float(y.pow(2).mean().sqrt()):.3g}; amax per channel "
          f"median {float(by_channel.median()):.3g}, max {float(by_channel.max()):.3g}; "
          f"amax per token median {float(by_token.median()):.3g} -- the per-token step is "
          f"amax/{GDN_QUANT.scale_max:g}")


def check(t: int, w: dict[str, torch.Tensor], dtypes: list[torch.dtype],
          seed: int, tol: float, quant: bool = False) -> bool:
    for name in ("fla", "causal_conv1d", "kernels"):
        if importlib.util.find_spec(name) is not None:
            print(f"[warn] `{name}` is installed: the module may not run its "
                  f"pure-torch path, and this compares against whatever it runs")

    x = reference.make_block_inputs(t, QWEN3_8_27B, seed)
    started = time.time()
    ours = reference.block(x, w, QWEN3_8_27B, CHUNK)
    print(f"[ref] float64 block at T={t}: {time.time() - started:.1f}s; "
          f"g in [{float(ours['g'].min()):.3g}, {float(ours['g'].max()):.3g}], "
          f"chunk g_sum min {float(ours['g_sum'].min()):.3g}, "
          f"|out| max {float(ours['out'].abs().max()):.3g}")

    passed = True
    bf16_out = None
    for dtype in dtypes:
        started = time.time()
        theirs = module_forward(x, w, dtype)
        if dtype == torch.bfloat16:
            bf16_out = theirs["out"]
        gated = dtype != torch.bfloat16
        print(f"\n[module] {str(dtype).removeprefix('torch.')}: "
              f"{time.time() - started:.1f}s"
              + (f"; gate rel-frob <= {tol:g} on o, y, out" if gated
                 else "; the model's own dtype, reported not gated"))
        print(f"  {'point':<6} {'module op':<12} {'rel frob':>10} {'max |diff|':>12}")
        for key, op, layout in POINTS:
            frob, max_diff = compare(layout(ours, t), theirs[key])
            verdict = ""
            if gated and key in ("o", "y", "out"):
                ok = frob <= tol
                passed &= ok
                verdict = "  ok" if ok else "  FAIL"
            print(f"  {key:<6} {op:<12} {frob:>10.3e} {max_diff:>12.3e}{verdict}")
    if quant:
        quant_table(t, x, w, ours, bf16_out)
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--weights", type=str, default=None,
                        help="a real layer from weights.py; default: random weights "
                             "from the module's own init")
    parser.add_argument("--seed", type=int, default=42,
                        help="hidden states, and the random weights")
    parser.add_argument("--module-dtype", type=str, default="float64",
                        help="comma-separated dtypes to run the module in; float64 "
                             "and float32 are gated, bfloat16 is reported (and slow "
                             "on a CPU without bf16 GEMM kernels: ~12 min at T = 8192)")
    parser.add_argument("--tol", type=float, default=TOL)
    parser.add_argument("--quant", action="store_true",
                        help="also run the W8A8 chain in its variants and report each "
                             "against the truth (and the bf16 module, if requested)")
    args = parser.parse_args()

    torch.set_num_threads(min(torch.get_num_threads(), MAX_THREADS))
    # oneDNN's CPU matmul takes a 100x slower path when an operand is column-major,
    # which is what `solve_triangular` hands the module's chunk loop (1 s per
    # 25 MFLOP bmm on aarch64). A backend switch, not a change to the maths.
    torch.backends.mkldnn.enabled = False
    if args.weights:
        w = weights_mod.load_layer(args.weights)
        print(f"[weights] real layer from {args.weights}")
    else:
        w = reference.make_block_weights(QWEN3_8_27B, args.seed)
        print(f"[weights] random, the module's own init, seed {args.seed}")
    dtypes = [getattr(torch, name.strip()) for name in args.module_dtype.split(",")]

    passed = check(args.seq_len, w, dtypes, args.seed, args.tol, quant=args.quant)
    print(f"\n{'PASS' if passed else 'FAIL'}: T={args.seq_len} "
          f"H={QWEN3_8_27B.linear_num_value_heads} Hg={QWEN3_8_27B.linear_num_key_heads} "
          f"D={QWEN3_8_27B.linear_value_head_dim} chunk={CHUNK}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
