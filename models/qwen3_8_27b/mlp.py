# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3.8-27B dense SwiGLU MLP, `down(silu(gate(x)) * up(x))`, in W8A8.

Five operators as one program: the per-token quantiser, `gate_proj` and
`up_proj` against the same quantised activation, the fused SwiGLU and its
quantiser, then `down_proj`. 5120 -> 17408 -> 5120 on every one of the 64
layers, which is the largest compute block in the model.

The three projections are `a8w8_linear` at two shapes; only the intermediate
tensors and the activation are new here.
"""
import pypto.language as pl

import a8w8_linear
import gate_up_swiglu
from act_quant import qwen3_act_quant
from config import QWEN3_8_27B
from quant_x import gdn_quant_x
from swiglu_quant import qwen3_swiglu_quant

# model config
HIDDEN = QWEN3_8_27B.hidden_size
INTER = QWEN3_8_27B.intermediate_size
# validation
DEFAULT_CASES = (128, 256)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# ci: no-sim

# The per-ring heap holds task outputs and intermediates, and its compile
# default is 256 MiB. This program's intermediates are `gate` and `up` at
# [T, 17408] BF16 plus `act_q`: 180 MiB at T = 2048, which fits, and 360 MiB at
# T = 4096, which does not -- both measured, the overflow arriving as
# HEAP_RING_DEADLOCK. 720 MiB at T = 8192.
#
# The heap holds what is in flight, not the whole intermediate set, so it does
# not scale with that 720 MiB: measured at T = 8192, 512 MiB runs and 256 MiB
# does not. One power of two above the floor, the runtime taking powers of two.
# Bytes per ring, one entry per scope depth, 0 leaving a ring at its default. The stage
# scopes allocate on ring 1; a scalar here would reserve the same bytes on all four.
MLP_RING_HEAP = [0, 1024 * 1024 * 1024, 0, 0]

# acceptance: the gate is the distance to the W8A8 golden, as a fraction of
# what quantization itself costs. At T = 8192 the kernel measures 0.161 of the
# floor whole-tensor and 0.297 on the worst row; an output mirrored across the
# truth would sit at 2.0.
MAX_VS_GOLDEN = 0.25            # whole tensor
MAX_ROW_VS_GOLDEN = 0.50        # worst token row
ABS_VS_GOLDEN = 1e-2            # fallback when the harness supplies no inputs

# tiling; the two projection shapes get their own knobs
# 128 x 128 x 512 is the tile for both, and it is not a free choice: two INT32
# accumulators cap `m_tile * n_tile` at 16384, and the double-buffered L1
# operands cap `(m_tile + 2 * n_tile) * k_tile` at 512 KiB. Measured at
# T = 8192, every other tiling that fits is 30-60% slower -- n_tile 64 costs
# 46% on `gate_proj` and k_tile 1024 another 10% on top.
UP_M_TILE, UP_N_TILE, UP_K_TILE = 128, 128, 512
DOWN_M_TILE, DOWN_N_TILE, DOWN_K_TILE = 128, 128, 512

# The patch a block walks, in output tiles, per projection and per axis. The
# two shapes want opposite things and the reason is how many blocks each ends
# up with against 24 cube cores. Measured standalone at T = 8192:
#
#   gate/up  64 x 136 tiles   8x8 -> 136 blocks 3066.3 us   16x8 -> 68  3018.4
#   down     64 x  40 tiles   8x8 ->  40 blocks 3216.3 us    4x4 -> 160 2856.8
#
# `down` is a third as wide, so the patch that suits `gate_proj` leaves it 40
# blocks -- 1.67 waves, a third of the second wave idle. Shrinking its patch
# to 4x4 is worth 11%. Going the other way on `gate_proj`, where blocks are
# plentiful, is worth another 1.6%. Note 80 blocks (4x8 or 8x4) is NOT the
# answer for `down`: only 4x4 helped, so this is operand reuse as well as
# occupancy.
# The M lists end at 1 and the N lists at 2. That asymmetry is not cosmetic:
# the token axis can be as short as one output tile -- T = 128, the smallest
# shape every callee here accepts -- and only a patch of 1 divides it, while
# `a8w8_linear` pairs n-tiles and rejects an odd `n_group`. The two real
# widths are 136 and 40 output tiles, both even, so the N axis never needs
# the case.
UP_M_PREFER, UP_N_PREFER = (16, 8, 4, 2, 1), (8, 4, 2)
DOWN_M_PREFER, DOWN_N_PREFER = (4, 2, 1), (4, 2)
# The fused kernel runs two matmuls per output tile, so it wants a smaller patch
# than the pair of plain projections does: 4 x 4 measures 5866.9 us against
# 6011.7 at the unfused form's 16 x 8. The unfused constants above stay as they
# are, so the two forms remain comparable on one grant.
FUSED_M_PREFER, FUSED_N_PREFER = (4, 2, 1), (4, 2)
K_STAGGER = 1               # rotate each block's K start; see `a8w8_linear`
# GM transport depth for the two projections this file ships; the on-chip
# reserve stays at 1. The fused pair is neutral to it standalone -- 5724.8
# against 5727.8 us -- but the composed MLP measures 9833.0 -> 9126.2 us at
# T = 8192 with both at 2, which is also the form validated on both weight
# sets. The unfused arm keeps the default 1, so a fused/unfused comparison
# has to force the two depths to match.
RING_SLOTS = 2


def _patch(blocks: int, prefer: tuple[int, ...]) -> int:
    """The first side in `prefer` that divides `blocks`, else the last.

    The patch cannot simply be the tuned constant: a shorter sequence has
    fewer output tiles down the token axis than the patch wants, and a patch
    that does not divide is rejected outright. Walking down `prefer` keeps a
    small T building at whatever patch does divide.
    """
    for side in prefer:
        if blocks % side == 0:
            return side
    return prefer[-1]


def build_kernel(t: int = BENCH_SEQ, hidden: int = HIDDEN, inter: int = INTER,
                 fused: bool = True, inline: bool = False):
    """The MLP at one shape, composing its operators as inlined callees.

    `fused` computes `gate`, `up` and the activation in one kernel writing a
    single `act` tensor, against the unfused form's two projections and a
    quantiser that reads both. The fused form streams the quantised activation
    once instead of twice and halves the intermediates, at the cost of one
    BF16 rounding the unfused chain does not perform -- see
    `gate_up_swiglu.py`. The unfused path is kept selectable so the two can be
    measured against each other on one grant.

    Static shape only: a packed varlen caller needs a tail, which every
    operator in this directory still owes.
    """
    up_m_group = _patch(t // UP_M_TILE, UP_M_PREFER)
    up_n_group = _patch(inter // UP_N_TILE, UP_N_PREFER)
    # One M tile per patch at the 2048-token serving shape.
    down_m_group = 1 if t == 2048 else _patch(t // DOWN_M_TILE, DOWN_M_PREFER)
    down_n_group = _patch(hidden // DOWN_N_TILE, DOWN_N_PREFER)

    op_gate = op_up = op_gate_up = None
    if fused:
        fused_m_group = 1 if t == 2048 else _patch(t // UP_M_TILE, FUSED_M_PREFER)
        fused_n_group = _patch(inter // UP_N_TILE, FUSED_N_PREFER)
        op_gate_up = gate_up_swiglu.build_kernel(m=t, k=hidden, n=inter, inline=True,
                                                 m_tile=UP_M_TILE, n_tile=UP_N_TILE,
                                                 k_tile=UP_K_TILE, m_group=fused_m_group,
                                                 n_group=fused_n_group,
                                                 k_stagger=K_STAGGER,
                                                 ring_slots=RING_SLOTS)
        op_act = qwen3_act_quant
    else:
        op_gate = a8w8_linear.build_kernel(m=t, k=hidden, n=inter, name="gate_proj", inline=True,
                                           m_tile=UP_M_TILE, n_tile=UP_N_TILE, k_tile=UP_K_TILE,
                                           m_group=up_m_group, n_group=up_n_group)
        op_up = a8w8_linear.build_kernel(m=t, k=hidden, n=inter, name="up_proj", inline=True,
                                         m_tile=UP_M_TILE, n_tile=UP_N_TILE, k_tile=UP_K_TILE,
                                         m_group=up_m_group, n_group=up_n_group)
        op_act = qwen3_swiglu_quant
    op_down = a8w8_linear.build_kernel(m=t, k=inter, n=hidden, name="down_proj", inline=True,
                                       m_tile=DOWN_M_TILE, n_tile=DOWN_N_TILE, k_tile=DOWN_K_TILE,
                                       m_group=down_m_group, n_group=down_n_group,
                                       ring_slots=RING_SLOTS)

    jit = pl.jit.inline if inline else pl.jit

    # Two bodies rather than one with a flag: a `@pl.jit` body traces a Python
    # `if` as IR control flow rather than evaluating it, so the choice has to
    # be made out here. Same reason `gated_rmsnorm` and `gate_up_swiglu` carry
    # two.
    @jit
    def qwen3_mlp(
        x: pl.Tensor[[t, hidden], pl.BF16],
        w_gate: pl.Tensor[[inter, hidden], pl.INT8],
        w_gate_scale: pl.Tensor[[inter], pl.FP32],
        w_up: pl.Tensor[[inter, hidden], pl.INT8],
        w_up_scale: pl.Tensor[[inter], pl.FP32],
        w_down: pl.Tensor[[hidden, inter], pl.INT8],
        w_down_scale: pl.Tensor[[hidden], pl.FP32],
        out: pl.Out[pl.Tensor[[t, hidden], pl.BF16]],
    ):
        x_q = pl.create_tensor([t, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)

        act_q = pl.create_tensor([t, inter], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        # one pass over the quantised activation, one intermediate
        act = pl.create_tensor([t, inter], dtype=pl.BF16)
        op_gate_up(x_q, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, act)
        op_act(act, act_q, act_scale)

        op_down(act_q, act_scale, w_down, w_down_scale, out)
        return out

    @jit
    def qwen3_mlp_unfused(
        x: pl.Tensor[[t, hidden], pl.BF16],
        w_gate: pl.Tensor[[inter, hidden], pl.INT8],
        w_gate_scale: pl.Tensor[[inter], pl.FP32],
        w_up: pl.Tensor[[inter, hidden], pl.INT8],
        w_up_scale: pl.Tensor[[inter], pl.FP32],
        w_down: pl.Tensor[[hidden, inter], pl.INT8],
        w_down_scale: pl.Tensor[[hidden], pl.FP32],
        out: pl.Out[pl.Tensor[[t, hidden], pl.BF16]],
    ):
        x_q = pl.create_tensor([t, hidden], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        gdn_quant_x(x, x_q, x_scale)

        act_q = pl.create_tensor([t, inter], dtype=pl.INT8)
        act_scale = pl.create_tensor([1, t], dtype=pl.FP32)
        # gate and up read the same quantised activation at the same shape
        gate = pl.create_tensor([t, inter], dtype=pl.BF16)
        up = pl.create_tensor([t, inter], dtype=pl.BF16)
        op_gate(x_q, x_scale, w_gate, w_gate_scale, gate)
        op_up(x_q, x_scale, w_up, w_up_scale, up)
        op_act(gate, up, act_q, act_scale)

        op_down(act_q, act_scale, w_down, w_down_scale, out)
        return out

    if not fused:
        return qwen3_mlp_unfused
    return qwen3_mlp


qwen3_mlp = build_kernel()


def build_tensor_specs(t: int = BENCH_SEQ, hidden: int = HIDDEN, inter: int = INTER,
                       weights: str | None = None):
    import torch
    from golden import TensorSpec

    import reference

    def draw(name):
        def make():
            return reference.mlp_inputs(name, t, hidden, inter, weights=weights)
        return make

    return [
        TensorSpec("x", [t, hidden], torch.bfloat16, init_value=draw("x")),
        TensorSpec("w_gate", [inter, hidden], torch.int8, init_value=draw("w_gate")),
        TensorSpec("w_gate_scale", [inter], torch.float32, init_value=draw("w_gate_scale")),
        TensorSpec("w_up", [inter, hidden], torch.int8, init_value=draw("w_up")),
        TensorSpec("w_up_scale", [inter], torch.float32, init_value=draw("w_up_scale")),
        TensorSpec("w_down", [hidden, inter], torch.int8, init_value=draw("w_down")),
        TensorSpec("w_down_scale", [hidden], torch.float32, init_value=draw("w_down_scale")),
        TensorSpec("out", [t, hidden], torch.bfloat16),
    ]


def compare_out():
    """Gate the MLP against its own W8A8 golden; score it against the truth beside.

    The **gate is the distance to `expected`**, the fake-quant chain the
    kernel implements. Not element-wise: `down_proj` sums 17408 terms that
    cancel heavily, so an output near zero carries a large relative difference
    for an absolute one far below the int8 step, and an element-wise 1% bound
    calls 28% of the tensor wrong on a correct kernel. Whole-tensor and
    worst-row norms against `expected` are the right shape of bound, and they
    are expressed as fractions of the quantisation floor so neither carries a
    tolerance invented from nothing.

    The truth ratios are **diagnostics only**. Gating on them alone is unsound:
    an output mirrored across the truth, `2 * truth - fq`, sits exactly at the
    floor distance from the truth and passes such a gate at 1.000x while being
    twice the floor away from the answer it should have produced.

    Inputs come from the harness rather than being redrawn, so a frozen golden
    replayed with `--golden-data` is scored against the tensors it was
    captured from.
    """
    import reference

    def compare(actual, expected, inputs=None, **_kwargs):
        dev, fq = actual.double(), expected.double()

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            return float(((a - b).norm(dim=-1) / b.norm(dim=-1)).max())

        names = ("x", "w_gate", "w_gate_scale", "w_up", "w_up_scale", "w_down", "w_down_scale")
        args = [inputs[n].cpu() for n in names] if inputs else None
        truth = reference.mlp_truth(*args) if args else None
        floor = rel(fq, truth) if truth is not None else None
        row_floor = worst_row(fq, truth) if truth is not None else None

        dev_fq, dev_row_fq = rel(dev, fq), worst_row(dev, fq)
        lines = []
        if floor:
            lines.append(f"[stats] out vs W8A8 golden: {dev_fq:.3e} = {dev_fq / floor:.3f} of the "
                         f"quantisation floor {floor:.3e} (gate {MAX_VS_GOLDEN})")
            lines.append(f"[stats] worst token row vs golden: {dev_row_fq:.3e} = "
                         f"{dev_row_fq / row_floor:.3f} of its floor {row_floor:.3e} "
                         f"(gate {MAX_ROW_VS_GOLDEN})")
            lines.append(f"[stats] diagnostic, vs float64 truth: kernel {rel(dev, truth):.3e}, "
                         f"W8A8 {floor:.3e} -> {rel(dev, truth) / floor:.3f}x")
            ok = dev_fq <= MAX_VS_GOLDEN * floor and dev_row_fq <= MAX_ROW_VS_GOLDEN * row_floor
        else:
            # No inputs to build the truth from: fall back to an absolute bound
            # rather than skipping the gate entirely.
            lines.append(f"[stats] out vs W8A8 golden: {dev_fq:.3e}, worst row {dev_row_fq:.3e} "
                         f"(no inputs; absolute bound {ABS_VS_GOLDEN})")
            ok = dev_fq <= ABS_VS_GOLDEN and dev_row_fq <= ABS_VS_GOLDEN
        detail = "\n".join(lines)
        return ok, "" if ok else detail

    return compare


def golden_qwen3_mlp(fused: bool = True):
    """The oracle for whichever chain was built.

    The fused form writes a BF16 `act` tensor the unfused form never
    materialises, so the two chains differ by that rounding and a golden that
    ignored it would be scoring the wrong one.
    """
    import torch

    import reference

    def golden(tensors):
        ref = reference.mlp_w8a8(
            tensors["x"],
            tensors["w_gate"], tensors["w_gate_scale"],
            tensors["w_up"], tensors["w_up_scale"],
            tensors["w_down"], tensors["w_down_scale"], act_bf16=fused,
        )
        tensors["out"].copy_(ref.to(torch.bfloat16))

    return golden


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--unfused", action="store_true", default=False,
                        help="two projections and swiglu_quant, the form Q6g replaced")
    parser.add_argument("--weights", type=str, default=None,
                        help="one layer's BF16 mlp tensors from weights.py; default: random weights")
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % UP_M_TILE:
        parser.error(f"--seq-len must be a multiple of {UP_M_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")

    for t in cases:
        print(f"[case] mlp T={t}", flush=True)
        result = run(
            fn=build_kernel(t=t, fused=not args.unfused),
            specs=build_tensor_specs(t=t, weights=args.weights),
            golden_fn=golden_qwen3_mlp(fused=not args.unfused),
            compare_fn={"out": compare_out()},
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
                ring_heap=MLP_RING_HEAP,
            ),
            rtol=1e-2,
            atol=1e-5,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={t}")
