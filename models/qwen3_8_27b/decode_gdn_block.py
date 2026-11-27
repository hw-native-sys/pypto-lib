# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# ci: no-sim
"""Qwen3.8-27B's Gated DeltaNet block at a decode batch: one token per sequence.

    x_q, x_scale  = quant_x(x)                      per-token int8 of the hidden states
    qkv           = in_proj_qkv(x_q, x_scale)       5120 -> 10240, A8W8
    z             = in_proj_z(x_q, x_scale)         5120 -> 6144,  A8W8
    a, b          = in_proj_ab(x)                   5120 -> 48 each, bf16
    q, k, v       = short_conv(qkv, conv_state)     one window step, then silu
    beta, g       = gate(a, b)                      sigmoid, softplus
    o             = delta_rule(q, k, v, g, beta, state)   one recurrence step
    y             = gated_rmsnorm(o, z)             the gated norm, bf16 out
    out           = out_proj(y)                     6144 -> 5120, BF16 both sides

Nine operators as one program, the decode counterpart of `gdn_block.py`'s fourteen. The
count differs because decode replaces the six-stage chunked delta rule with one recurrence
step and folds q/k normalisation into it, so `qk_norm_gate` becomes `decode_gate`.

**Two pools enter and leave.** `conv_state` carries the `K - 1` pre-convolution stream rows
per sequence and `state` the `[H, D, D]` recurrent state; both are `InOut` and both are
addressed by the same `state_rows`, so a sequence occupies the same row index in each. That
is a choice this file makes rather than one the operators impose -- they take `state_rows`
independently -- and it is what a serving layer would do, since the two pools are allocated
per sequence together.

Input matrices use a batch padded to a positive multiple of 16. `active_rows` can
restrict the recurrence to a live prefix; its inactive outputs are zero and its
inactive state rows remain untouched. Convolution and matrix stages retain the padded
extent, so callers must still provide distinct scratch rows for inactive tokens.

What differs from the prefill block beyond the two recurrences: there is no convolution halo,
because the window's history comes from `conv_state` rather than from zero rows above a
padded projection output; and `in_proj_ab` still reads `x` in bf16 while everything else
reads the quantised copy, which remains the one place the block reads the hidden states
twice.
"""
import pypto.language as pl

import decode_delta_rule
import decode_gate
import decode_projection
import decode_short_conv
import in_proj_ab
import quant_x
import utils
from config import QWEN3_8_27B
import gated_rmsnorm
from decode_projection import check_batch, check_tile, padded_batch

# model config
C = QWEN3_8_27B.hidden_size                 # the residual stream
CQKV = QWEN3_8_27B.qkv_width                # what in_proj_qkv writes
CV = QWEN3_8_27B.value_width                # what in_proj_z writes, out_proj reads
H = QWEN3_8_27B.linear_num_value_heads
HG = QWEN3_8_27B.linear_num_key_heads
D = QWEN3_8_27B.linear_value_head_dim
KCONV = QWEN3_8_27B.linear_conv_kernel_dim  # convolution taps
STATE_LEN = KCONV - 1

# validation
# one padded decode batch, then two; the inlined tiles are checked against it.
DEFAULT_CASES = (16, 32)

# tiling, each the decode value its own module measured or `decode_projection` carries
QUANT_GROUPS = 4        # quant_x column groups
AB_M_TILE = 16          # in_proj_ab at a padded batch (decode_projection)
AB_K_TILE = 1024
CONV_C_TILE = 2048      # decode_short_conv
CONV_TOK_TILE = 4
GATE_TOK_TILE = 16
HEAD_TILE = 8           # decode_delta_rule head-states per task

# `gated_rmsnorm`'s decode form, one task per token. A module constant rather than a
# build argument so both schedules can be built in one process; the two are bit-identical.
GATED_NORM_FANNED = True


def run_config(backend_type: str = "a2a3", sim: bool = False, device: int = 0,
               swimlane: int = 0) -> dict:
    """The config `golden.run` needs for this block."""
    return dict(backend_type=backend_type, execution_mode=sim, device_id=device,
                enable_chip_swimlane=swimlane)


def build_kernel(batch: int, head_tile: int = HEAD_TILE, inline: bool = False, active_rows: int = 0):
    """The block at one decode batch, composing nine operators as inlined callees."""
    m = padded_batch(batch)
    check_batch(m)
    rows = active_rows or m
    if not 1 <= rows <= m:
        raise ValueError("active_rows must be zero or within the padded batch")
    # Both branches are parsed; even the inactive padding loop needs a positive extent.
    pad_rows = max(1, m - rows)
    check_tile("AB_M_TILE", AB_M_TILE, m)
    check_tile("GATE_TOK_TILE", GATE_TOK_TILE, m)

    quant = quant_x.build_fanned_kernel(m, col_groups=QUANT_GROUPS, inline=True)
    op_qkv = decode_projection.build_kernel("in_proj_qkv", m, inline=True)
    op_z = decode_projection.build_kernel("in_proj_z", m, inline=True)
    op_ab = in_proj_ab.build_kernel(m_tile=AB_M_TILE, k_tile=AB_K_TILE, inline=True)
    conv, _ = decode_short_conv.build_kernel(c_tile=CONV_C_TILE, tok_tile=CONV_TOK_TILE)
    gate = decode_gate.build_kernel(tok_tile=GATE_TOK_TILE, inline=True)
    delta, _ = decode_delta_rule.build_kernel(head_tile=head_tile)
    op_out = decode_projection.build_kernel("out_proj", m, inline=True)
    norm = (gated_rmsnorm.build_fanned_bf16_kernel(m, inline=True) if GATED_NORM_FANNED
            else gated_rmsnorm.gdn_gated_rmsnorm_bf16)

    @(pl.jit.inline if inline else pl.jit)
    def decode_gdn_block(
        x: pl.Tensor[[m, C], pl.BF16],
        w_qkv: pl.Tensor[[CQKV, C], pl.INT8],
        w_qkv_scale: pl.Tensor[[CQKV], pl.FP32],
        w_z: pl.Tensor[[CV, C], pl.INT8],
        w_z_scale: pl.Tensor[[CV], pl.FP32],
        w_ab: pl.Tensor[[2 * H, C], pl.BF16],
        conv_w: pl.Tensor[[KCONV, CQKV], pl.FP32],
        a_log: pl.Tensor[[H], pl.BF16],
        dt_bias: pl.Tensor[[H], pl.BF16],
        norm_w: pl.Tensor[[D], pl.BF16],
        w_out: pl.Tensor[[C, CV], pl.BF16],
        state_rows: pl.Tensor[[m], pl.INT32],
        conv_state: pl.InOut[pl.Tensor[[decode_short_conv.CSTATE_DYN, STATE_LEN, CQKV], pl.BF16]],
        state: pl.InOut[pl.Tensor[[decode_delta_rule.DSTATE_DYN, H, D, D], pl.FP32]],
        out: pl.Out[pl.Tensor[[m, C], pl.BF16]],
    ):
        """`state_rows` names each sequence's row in BOTH pools; they must be distinct."""
        x_q = pl.create_tensor([m, C], dtype=pl.INT8)
        x_scale = pl.create_tensor([1, m], dtype=pl.FP32)
        quant(x, x_q, x_scale)

        qkv = pl.create_tensor([m, CQKV], dtype=pl.BF16)
        op_qkv(x_q, x_scale, w_qkv, w_qkv_scale, qkv)

        z = pl.create_tensor([m, CV], dtype=pl.BF16)
        op_z(x_q, x_scale, w_z, w_z_scale, z)

        a_proj = pl.create_tensor([m, H], dtype=pl.BF16)
        b_proj = pl.create_tensor([m, H], dtype=pl.BF16)
        op_ab(x, w_ab, a_proj, b_proj)

        q_conv = pl.create_tensor([m, HG, D], dtype=pl.BF16)
        k_conv = pl.create_tensor([m, HG, D], dtype=pl.BF16)
        v_conv = pl.create_tensor([m, H, D], dtype=pl.BF16)
        conv(qkv, conv_w, conv_state, state_rows, q_conv, k_conv, v_conv)

        beta = pl.create_tensor([m, H], dtype=pl.FP32)
        g = pl.create_tensor([m, H], dtype=pl.FP32)
        gate(a_proj, b_proj, a_log, dt_bias, beta, g)

        o = pl.create_tensor([m, H, D], dtype=pl.FP16)
        if rows < m:
            o_rows = pl.reshape(o, [m, CV])
            for row in pl.spmd(pad_rows, name_hint="decode_recurrence_pad"):
                start = rows + row
                o_rows[start : start + 1, 0:CV] = pl.full([1, CV], value=0.0, dtype=pl.FP16)
        q_live = q_conv[0:rows, 0:HG, 0:D]
        k_live = k_conv[0:rows, 0:HG, 0:D]
        v_live = v_conv[0:rows, 0:H, 0:D]
        beta_live = beta[0:rows, 0:H]
        g_live = g[0:rows, 0:H]
        state_rows_live = state_rows[0:rows]
        o_live = o[0:rows, 0:H, 0:D]
        delta(q_live, k_live, v_live, beta_live, g_live, state, state_rows_live, o_live)

        y = pl.create_tensor([m, CV], dtype=pl.BF16)
        norm(o, z, norm_w, y)

        op_out(y, w_out, out)
        return out

    return decode_gdn_block


def build_tensor_specs(batch: int = 16, pool_rows: int | None = None,
                       weights: str | None = None, first_row: int = 2, seed: int = 11):
    """Every weight the block reads, the two state pools, and a scattered row mapping.

    `first_row` offsets every sequence's pool row so a kernel that ignored `state_rows` and
    used the token index would land on the wrong state and the gate would catch it. The
    pools are seeded non-zero for the same reason: a zero state hides a dropped read.
    """
    import torch
    from golden import TensorSpec

    import reference
    from weights import quantized_weight

    m = padded_batch(batch)
    rows = pool_rows if pool_rows is not None else m + first_row

    def block_weights():
        if weights is None:
            return reference.make_block_weights(QWEN3_8_27B)
        return torch.load(weights, weights_only=True)

    def quantized(module):
        return quantized_weight(QWEN3_8_27B, module, weights)

    w_qkv, w_qkv_scale = quantized("in_proj_qkv")
    w_z, w_z_scale = quantized("in_proj_z")

    def stacked_ab():
        w = block_weights()
        return torch.cat([w["in_proj_a.weight"], w["in_proj_b.weight"]]).to(torch.bfloat16)

    def taps():
        return block_weights()["conv1d.weight"].reshape(CQKV, KCONV).t().contiguous().float()

    def param(name):
        return lambda: block_weights()[name].to(torch.bfloat16)

    def rowmap():
        gen = torch.Generator().manual_seed(seed)
        return (torch.randperm(rows - first_row, generator=gen)[:m] + first_row).to(torch.int32)

    def conv_pool():
        gen = torch.Generator().manual_seed(seed + 1)
        return (torch.randn(rows, STATE_LEN, CQKV, generator=gen) * 0.5).to(torch.bfloat16)

    def state_pool():
        gen = torch.Generator().manual_seed(seed + 2)
        return torch.randn(rows, H, D, D, generator=gen) * 0.1

    return [
        TensorSpec("x", [m, C], torch.bfloat16,
                   init_value=lambda: reference.make_block_inputs(m, QWEN3_8_27B)),
        TensorSpec("w_qkv", [CQKV, C], torch.int8, init_value=w_qkv),
        TensorSpec("w_qkv_scale", [CQKV], torch.float32, init_value=w_qkv_scale),
        TensorSpec("w_z", [CV, C], torch.int8, init_value=w_z),
        TensorSpec("w_z_scale", [CV], torch.float32, init_value=w_z_scale),
        TensorSpec("w_ab", [2 * H, C], torch.bfloat16, init_value=stacked_ab),
        TensorSpec("conv_w", [KCONV, CQKV], torch.float32, init_value=taps),
        TensorSpec("a_log", [H], torch.bfloat16, init_value=param("A_log")),
        TensorSpec("dt_bias", [H], torch.bfloat16, init_value=param("dt_bias")),
        TensorSpec("norm_w", [D], torch.bfloat16, init_value=param("norm.weight")),
        TensorSpec("w_out", [C, CV], torch.bfloat16, init_value=param("out_proj.weight")),
        TensorSpec("state_rows", [m], torch.int32, init_value=rowmap),
        TensorSpec("conv_state", [rows, STATE_LEN, CQKV], torch.bfloat16, init_value=conv_pool),
        TensorSpec("state", [rows, H, D, D], torch.float32, init_value=state_pool),
        TensorSpec("out", [m, C], torch.bfloat16),
    ]


def golden_decode_gdn_block(weights: str | None = None, active_rows: int = 0):
    """The fake-quant chain in float64: `reference.recurrent_block`, one call per sequence.

    `quant_y` is off and `out_proj` is not in the quantised set, because the deployed
    checkpoint leaves that one projection in BF16 and the model matches it: the block's
    last hand-off is bf16 both sides, so there is no per-token quantisation of `y` to model
    and no weight to dequantise.

    Each sequence reads its own row of both pools, which is what the kernel does in
    parallel, and both pools are written back so the gate covers them as well as `out`.
    Rows `state_rows` does not name must come back untouched.

    **The two pools do not share a layout convention and only one needs converting.**
    `reference.recurrent_short_conv` takes the conv window time-major, `[K - 1, C]`, which is
    the kernel's own layout, so it passes straight through. `recurrent_delta_rule` takes
    `[H, K, V]` while the kernel's pool is `[H, V, K]`, so every row turns through
    `decode_delta_rule.golden_state_to_kv` on the way in and back on the way out. Both
    extents are `D`, so a missed turn cannot be caught by any shape check -- it surfaces only
    as wrong numbers, which is why this is spelled out rather than inlined.
    """
    import torch

    import reference

    def golden(tensors):
        # Quantised, as the device's weights are. `build_tensor_specs` loads `in_proj_qkv`
        # and `in_proj_z` through `weights.quantized_weight`, which is this same
        # `quantize_weights` one module at a time, so the two sides hold the same INT8
        # values and the same per-row scales. Without it `reference.weight` would widen
        # the float weight instead of rounding it, and the `in_proj_qkv` difference would
        # reach `conv_state` and `state`, where `compare_pool` would absorb it as
        # tolerance rather than check the state transition.
        w = reference.quantize_weights(
            reference.make_block_weights(QWEN3_8_27B) if weights is None
            else torch.load(weights, weights_only=True))
        x = tensors["x"]
        rows = tensors["state_rows"].tolist()
        conv_out = tensors["conv_state"].double().clone()
        state_out = tensors["state"].double().clone()
        out = torch.empty(x.shape[0], C, dtype=torch.float64)
        for i, row in enumerate(rows):
            r = reference.recurrent_block(
                x[i : i + 1], w, QWEN3_8_27B,
                conv_state=tensors["conv_state"][row].double(),
                state=decode_delta_rule.golden_state_to_kv(tensors["state"][row].double()),
                quant_x=True, quant_y=False,
            )
            conv_out[row] = r["conv_state_out"]
            if active_rows == 0 or i < active_rows:
                out[i] = r["out"][0]
                state_out[row] = decode_delta_rule.golden_state_from_kv(r["state_out"])
            else:
                out[i].zero_()
        tensors["out"].copy_(out.to(tensors["out"].dtype))
        tensors["conv_state"].copy_(conv_out.to(tensors["conv_state"].dtype))
        tensors["state"].copy_(state_out.to(tensors["state"].dtype))

    return golden


__all__ = ["build_kernel", "build_tensor_specs",
           "golden_decode_gdn_block", "run_config"]


_CHAIN_CACHE: dict[tuple, dict] = {}


def _truth(batch: int, weights: str | None, quant: bool, seed: int = 11, first_row: int = 2):
    """The float64 decode step per sequence, cached. *quant* picks the fake-quant chain.

    Rebuilds the same fixtures `build_tensor_specs` seeds, so the quantisation floor is
    measured on the inputs the kernel actually saw.
    """
    import torch

    import reference

    reference.host_torch_setup()
    m = padded_batch(batch)
    key = (m, weights, quant, seed)
    if key in _CHAIN_CACHE:
        return _CHAIN_CACHE[key]
    w = (reference.make_block_weights(QWEN3_8_27B) if weights is None
         else torch.load(weights, weights_only=True))
    if quant:
        w = reference.quantize_weights(w)
    rows = m + first_row
    gen = torch.Generator().manual_seed(seed)
    rowmap = (torch.randperm(rows - first_row, generator=gen)[:m] + first_row).tolist()
    gen1 = torch.Generator().manual_seed(seed + 1)
    conv_pool = (torch.randn(rows, STATE_LEN, CQKV, generator=gen1) * 0.5).to(torch.bfloat16)
    gen2 = torch.Generator().manual_seed(seed + 2)
    state_pool = torch.randn(rows, H, D, D, generator=gen2) * 0.1
    x = reference.make_block_inputs(m, QWEN3_8_27B)
    out = torch.empty(m, C, dtype=torch.float64)
    for i, row in enumerate(rowmap):
        r = reference.recurrent_block(
            x[i : i + 1], w, QWEN3_8_27B,
            conv_state=conv_pool[row].double(),
            state=decode_delta_rule.golden_state_to_kv(state_pool[row].double()),
            quant_x=quant, quant_y=False)
        out[i] = r["out"][0]
    _CHAIN_CACHE[key] = {"out": out}
    return _CHAIN_CACHE[key]


# Acceptance, as ratios against the quantisation floor rather than fixed numbers, the same
# shape `gdn_block.compare_out` uses and for the same reason: `out_proj` reads INT8, so an
# element-wise gate after a quantisation step measures which side of a rounding boundary a
# value landed on rather than whether the block is right.
# **`MAX_OVER_FLOOR` and `MAX_VS_GOLDEN` cannot both bind.** If the
# kernel may sit a whole floor away from the fake-quant chain, its distance to the truth can
# reach sqrt(2) floors when the two errors are orthogonal. The 1.05 this started at held only
# while the floor was dominated by a term the kernel reproduces exactly; taking the GDN tail to
# BF16 shrank that floor 2.8x, the kernel's own rounding is 0.36 to 0.42 of what is left, and
# the expected ratio is therefore sqrt(1 + 0.42^2) = 1.08. Three of four configurations read
# 1.042x to 1.050x against a 1.05 bound, which is a mis-calibrated gate rather than a kernel
# that moved. The binding gate is the distance to the chain the kernel implements; the truth
# ratio is a sanity bound, and a real regression shows up there as a multiple, not a percent.
MAX_OVER_FLOOR = 1.15
MAX_ROW_OVER_FLOOR = 1.25
MAX_VS_GOLDEN = 1.00
MAX_ROW_VS_GOLDEN = 1.50


def compare_out(batch: int, weights: str | None, active_rows: int = 0):
    """The composed decode block against the float64 truth, beside what W8A8 alone costs."""
    def compare(actual, expected, **_kwargs):
        truth = _truth(batch, weights, quant=False)["out"]
        fq = _truth(batch, weights, quant=True)["out"]
        dev = actual.double()
        if active_rows:
            if bool(actual[active_rows:].count_nonzero()):
                return False, "inactive GDN output rows must be zero"
            truth, fq, dev = truth[:active_rows], fq[:active_rows], dev[:active_rows]

        def rel(a, b):
            return float((a - b).norm() / b.norm())

        def worst_row(a, b):
            num = (a - b).norm(dim=-1)
            den = b.norm(dim=-1).clamp_min(1e-30)
            return float((num / den).max())

        floor, floor_row = rel(fq, truth), worst_row(fq, truth)
        got, got_row = rel(dev, truth), worst_row(dev, truth)
        vs_fq, vs_fq_row = rel(dev, fq), worst_row(dev, fq)
        ratio = got / floor if floor else float("inf")
        ratio_row = got_row / floor_row if floor_row else float("inf")
        fq_ratio = vs_fq / floor if floor else float("inf")
        fq_ratio_row = vs_fq_row / floor_row if floor_row else float("inf")
        line = (f"[stats] out: rel frob {got:.3e} vs the float64 truth against a "
                f"quantisation floor of {floor:.3e} ({ratio:.3f}x); worst token row "
                f"{got_row:.3e} against {floor_row:.3e} ({ratio_row:.3f}x); {vs_fq:.3e} vs "
                f"the fake-quant chain ({fq_ratio:.3f} of the floor, worst row "
                f"{fq_ratio_row:.3f})")
        ok = (ratio <= MAX_OVER_FLOOR and ratio_row <= MAX_ROW_OVER_FLOOR
              and fq_ratio <= MAX_VS_GOLDEN and fq_ratio_row <= MAX_ROW_VS_GOLDEN)
        return ok, "" if ok else line

    return compare


def compare_pool(name: str, live: int = 0, threshold: float = 2e-2):
    """A state pool matches the oracle where this step wrote and is untouched elsewhere.

    `state_rows` names each sequence's row in both pools. `live` is the kernel's
    `active_rows`, and the two pools do not take the same value: the convolution runs over
    the padded batch and writes every named row of `conv_state`, while the delta rule takes
    `state_rows[0:active_rows]` and leaves the padding lanes' rows of `state` alone. `live`
    of zero means every named row, which is what `active_rows` of zero builds.
    """
    def written(inputs, pool_rows: int):
        import torch

        rows = inputs["state_rows"]
        named = rows if live == 0 else rows[:live]
        mask = torch.zeros(pool_rows, dtype=torch.bool)
        mask[named.long()] = True
        return mask

    return utils.in_place_pool(name, written, threshold)


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--head-tile", type=int, default=HEAD_TILE)
    parser.add_argument("--active-rows", type=int, default=0, help="live recurrence rows; zero uses all rows")
    parser.add_argument("--weights", type=str, default=None,
                        help="one layer's BF16 linear_attn tensors; default: random weights")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--enable-chip-swimlane", type=int, nargs="?", const=1, default=0,
                        choices=range(5))
    args = parser.parse_args()
    cases = [args.batch] if args.batch else list(DEFAULT_CASES)

    for batch in cases:
        print(f"[case] decode_gdn_block batch={batch}", flush=True)
        result = run(
            fn=build_kernel(batch, head_tile=args.head_tile, active_rows=args.active_rows),
            specs=build_tensor_specs(batch=batch, weights=args.weights),
            golden_fn=golden_decode_gdn_block(args.weights, active_rows=args.active_rows),
            compare_fn={"out": compare_out(batch, args.weights, active_rows=args.active_rows),
                        "conv_state": compare_pool("conv_state"),
                        "state": compare_pool("state", live=args.active_rows)},
            config=run_config(args.backend_type, args.execution_mode, args.device,
                              args.enable_chip_swimlane),
            rtol=1e-2,
            atol=1e-2,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at batch={batch}")
