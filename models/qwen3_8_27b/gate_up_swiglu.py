# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The MLP's two wide projections and its activation, as one kernel:

    act[m, n] = silu( bf16(gate[m, n]) ) * bf16(up[m, n])

with `gate = a @ w_gate^T` and `up = a @ w_up^T`, both A8W8 against the same
quantised activation. Replaces two `a8w8_linear` instances plus the projection
half of `swiglu_quant`; the per-token quantiser stays in `act_quant.py`,
because its scale spans all of `n` and no output tile sees a whole row.

`a8w8_linear` runs two INT32 accumulators for two n-tiles of one weight; this
runs two for one n-tile of two weights. Same 128 KiB of Acc and the same three
L1 operands per K step.

**AIV lanes split the epilogue by token rows.** Both dequantised tiles have to
be live at once to be combined, where `a8w8_linear` finished one accumulator
before starting the next. At 128 x 128 that wants 262656 B of a 188416 B vector
buffer; halving the lane's rows fits it.

**It adds a rounding the unfused chain does not have.** `swiglu_quant` holds a
token's row and goes from the two BF16 projections straight to int8, never
materialising the activation; writing `act` rounds it. `act_dtype=pl.FP32`
builds the control that adds no rounding and saves no storage.
"""
import pypto.language as pl

from config import QWEN3_8_27B

# model config
HIDDEN = QWEN3_8_27B.hidden_size
INTER = QWEN3_8_27B.intermediate_size
# validation
# one M_TILE of tokens, then two; the entry picks a patch side that divides.
# A block's work scales with the patch side, so the simulator leg is slow: 256
# tokens measured 42 s there, and 43.7 s on the pinned toolchain, where it does
# finish. It once tripped the simulators' 20 s stall detector; it no longer does.
DEFAULT_CASES = (128, 256)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling; the accumulator pair is (gate, up), so n_group has no parity rule
M_TILE = 128            # token rows per output tile
N_TILE = 128            # intermediate channels per output tile
K_TILE = 512            # contraction step; 512 INT8 columns is one L2 line
M_GROUP = 16            # output tiles a block walks down the token axis
N_GROUP = 8             # and across the channel axis
# Fallbacks for the standalone entry, which has to build at whatever `--seq-len`
# it is given. `mlp.py` passes the patch explicitly and carries the measured
# preferences; these repeat the M axis rather than import the consumer.
M_PREFER = (16, 8, 4, 2, 1)


# scheduling: see `rmsnorm.EARLY_RESOLVE`. This kernel is shared with prefill, where a
# saturated engine leaves no early window for the hint to win, so it should be inert there;
# that is measured rather than assumed.
EARLY_RESOLVE = True
AIV_LANES = 2



def build_kernel(m: int = BENCH_SEQ, k: int = HIDDEN, n: int = INTER, name: str = "gate_up_swiglu",
                 inline: bool = False, m_tile: int = M_TILE, n_tile: int = N_TILE,
                 k_tile: int = K_TILE, m_group: int = M_GROUP, n_group: int = N_GROUP,
                 act_dtype=pl.BF16, k_stagger: int = 0, ring_slots: int = 1,
                 workers: int = 0):
    """Both projections and the activation at one shape.

    `act_dtype` is `pl.BF16` for the shipping form and `pl.FP32` for the
    control: same fusion, no added rounding, and no storage saving, so a
    disagreement between the two is the rounding and nothing else.

    `workers` splits the block count from the patch count; see `a8w8_linear.build_kernel`.
    """
    if m % m_tile or n % n_tile or k % k_tile:
        raise ValueError(f"m={m} must be a multiple of m_tile={m_tile}, "
                         f"n={n} of n_tile={n_tile}, k={k} of k_tile={k_tile}")
    mb, nb = m // m_tile, n // n_tile
    if mb % m_group or nb % n_group:
        raise ValueError(f"the {mb}x{nb} output tiles must divide by the patch "
                         f"{m_group}x{n_group}")
    if workers < 0:
        raise ValueError(f"workers={workers} must be zero or a positive block count")
    mg, ng = mb // m_group, nb // n_group
    patches = mg * ng
    blocks = min(patches, workers) if workers else patches
    row_tile = m_tile // AIV_LANES

    jit = pl.jit.inline if inline else pl.jit

    # Two bodies rather than one with a flag, the same shape of duplication
    # `gated_rmsnorm` carries and for the same reason: a `@pl.jit` body cannot
    # branch on a Python flag, and the FP32 form has to omit a cast the DSL
    # rejects as a no-op rather than perform it.
    @jit
    def gdn_gate_up_swiglu(
        a_q: pl.Tensor[[m, k], pl.INT8],
        a_scale: pl.Tensor[[1, m], pl.FP32],
        w_gate: pl.Tensor[[n, k], pl.INT8],
        w_gate_scale: pl.Tensor[[n], pl.FP32],
        w_up: pl.Tensor[[n, k], pl.INT8],
        w_up_scale: pl.Tensor[[n], pl.FP32],
        act: pl.Out[pl.Tensor[[m, n], pl.BF16]],
    ):
        for blk in pl.spmd(blocks, name_hint=name,
                           optimizations=[pl.cross_core_slot(slot_num=ring_slots, local_slot_num=1)],
                           allow_early_resolve=EARLY_RESOLVE):
            for patch in pl.range(blk, patches, blocks):
                pm = (patch % mg) * m_group
                pn = (patch // mg) * n_group
                for gm in pl.range(m_group):
                    m0 = (pm + gm) * m_tile
                    for gn in pl.range(n_group):
                        n0 = (pn + gn) * n_tile
                        acc_gate = pl.create_tensor([m_tile, n_tile], dtype=pl.INT32)
                        acc_up = pl.create_tensor([m_tile, n_tile], dtype=pl.INT32)
                        for kb in pl.pipeline(0, k // k_tile, stage=2):
                            # `k_stagger`: see `a8w8_linear`, same rotation and same reason
                            k0 = ((kb + patch * k_stagger) % (k // k_tile)) * k_tile
                            at = a_q[m0 : m0 + m_tile, k0 : k0 + k_tile]
                            acc_gate = pl.matmul_acc(acc_gate, at,
                                                     w_gate[n0 : n0 + n_tile, k0 : k0 + k_tile],
                                                     b_trans=True, init_cond=(kb == 0))
                            acc_up = pl.matmul_acc(acc_up, at,
                                                   w_up[n0 : n0 + n_tile, k0 : k0 + k_tile],
                                                   b_trans=True, init_cond=(kb == 0))
                        for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                            row = m0 + lane * row_tile
                            xs = pl.reshape(a_scale[0:1, row : row + row_tile], [row_tile, 1])
                            ws_gate = pl.reshape(w_gate_scale[n0 : n0 + n_tile], [1, n_tile])
                            gate_f = pl.cast(pl.aiv_shard(acc_gate), target_type=pl.FP32)
                            gate_scaled = pl.col_expand_mul(gate_f, ws_gate)
                            deq_gate = pl.row_expand_mul(gate_scaled, xs)
                            gate_bf16 = pl.cast(deq_gate, target_type=pl.BF16, mode="rint")
                            gate_f = pl.cast(gate_bf16, target_type=pl.FP32)
                            gate_act = pl.div(gate_f, pl.add(pl.exp(pl.neg(gate_f)), 1.0))
                            ws_up = pl.reshape(w_up_scale[n0 : n0 + n_tile], [1, n_tile])
                            up_f = pl.cast(pl.aiv_shard(acc_up), target_type=pl.FP32)
                            up_scaled = pl.col_expand_mul(up_f, ws_up)
                            deq_up = pl.row_expand_mul(up_scaled, xs)
                            up_bf16 = pl.cast(deq_up, target_type=pl.BF16, mode="rint")
                            up_f = pl.cast(up_bf16, target_type=pl.FP32)
                            product = pl.mul(gate_act, up_f)
                            product = pl.cast(product, target_type=pl.BF16, mode="rint")
                            act[row : row + row_tile, n0 : n0 + n_tile] = product
        return act

    @jit
    def gdn_gate_up_swiglu_fp32(
        a_q: pl.Tensor[[m, k], pl.INT8],
        a_scale: pl.Tensor[[1, m], pl.FP32],
        w_gate: pl.Tensor[[n, k], pl.INT8],
        w_gate_scale: pl.Tensor[[n], pl.FP32],
        w_up: pl.Tensor[[n, k], pl.INT8],
        w_up_scale: pl.Tensor[[n], pl.FP32],
        act: pl.Out[pl.Tensor[[m, n], pl.FP32]],
    ):
        for blk in pl.spmd(blocks, name_hint=name,
                           optimizations=[pl.cross_core_slot(slot_num=ring_slots, local_slot_num=1)],
                           allow_early_resolve=EARLY_RESOLVE):
            for patch in pl.range(blk, patches, blocks):
                pm = (patch % mg) * m_group
                pn = (patch // mg) * n_group
                for gm in pl.range(m_group):
                    m0 = (pm + gm) * m_tile
                    for gn in pl.range(n_group):
                        n0 = (pn + gn) * n_tile
                        acc_gate = pl.create_tensor([m_tile, n_tile], dtype=pl.INT32)
                        acc_up = pl.create_tensor([m_tile, n_tile], dtype=pl.INT32)
                        for kb in pl.pipeline(0, k // k_tile, stage=2):
                            # `k_stagger`: see `a8w8_linear`, same rotation and same reason
                            k0 = ((kb + patch * k_stagger) % (k // k_tile)) * k_tile
                            at = a_q[m0 : m0 + m_tile, k0 : k0 + k_tile]
                            acc_gate = pl.matmul_acc(acc_gate, at,
                                                     w_gate[n0 : n0 + n_tile, k0 : k0 + k_tile],
                                                     b_trans=True, init_cond=(kb == 0))
                            acc_up = pl.matmul_acc(acc_up, at,
                                                   w_up[n0 : n0 + n_tile, k0 : k0 + k_tile],
                                                   b_trans=True, init_cond=(kb == 0))
                        for lane in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                            row = m0 + lane * row_tile
                            xs = pl.reshape(a_scale[0:1, row : row + row_tile], [row_tile, 1])
                            ws_gate = pl.reshape(w_gate_scale[n0 : n0 + n_tile], [1, n_tile])
                            gate_f = pl.cast(pl.aiv_shard(acc_gate), target_type=pl.FP32)
                            gate_scaled = pl.col_expand_mul(gate_f, ws_gate)
                            deq_gate = pl.row_expand_mul(gate_scaled, xs)
                            gate_bf16 = pl.cast(deq_gate, target_type=pl.BF16, mode="rint")
                            gate_f = pl.cast(gate_bf16, target_type=pl.FP32)
                            gate_act = pl.div(gate_f, pl.add(pl.exp(pl.neg(gate_f)), 1.0))
                            ws_up = pl.reshape(w_up_scale[n0 : n0 + n_tile], [1, n_tile])
                            up_f = pl.cast(pl.aiv_shard(acc_up), target_type=pl.FP32)
                            up_scaled = pl.col_expand_mul(up_f, ws_up)
                            deq_up = pl.row_expand_mul(up_scaled, xs)
                            up_bf16 = pl.cast(deq_up, target_type=pl.BF16, mode="rint")
                            up_f = pl.cast(up_bf16, target_type=pl.FP32)
                            product = pl.mul(gate_act, up_f)
                            act[row : row + row_tile, n0 : n0 + n_tile] = product
        return act

    if act_dtype is pl.FP32:
        return gdn_gate_up_swiglu_fp32
    return gdn_gate_up_swiglu


gdn_gate_up_swiglu = build_kernel()


def build_tensor_specs(t: int = BENCH_SEQ, hidden: int = HIDDEN, inter: int = INTER,
                       act_dtype=None, weights: str | None = None):
    import torch
    from golden import TensorSpec

    import reference

    def draw(name):
        def make():
            return reference.mlp_inputs(name, t, hidden, inter, weights=weights)
        return make

    torch_act = torch.float32 if act_dtype is pl.FP32 else torch.bfloat16
    return [
        TensorSpec("a_q", [t, hidden], torch.int8, init_value=draw("x_q")),
        TensorSpec("a_scale", [1, t], torch.float32, init_value=draw("x_scale")),
        TensorSpec("w_gate", [inter, hidden], torch.int8, init_value=draw("w_gate")),
        TensorSpec("w_gate_scale", [inter], torch.float32, init_value=draw("w_gate_scale")),
        TensorSpec("w_up", [inter, hidden], torch.int8, init_value=draw("w_up")),
        TensorSpec("w_up_scale", [inter], torch.float32, init_value=draw("w_up_scale")),
        TensorSpec("act", [t, inter], torch_act),
    ]


# acceptance: the activation is the product of two dequantised INT32 sums, so
# a relative bound on the whole tensor and the worst token row is the right
# shape; an element-wise one measures cancellation in the contraction.
MAX_REL = 5e-3
MAX_ROW_REL = 2e-2


def golden_gdn_gate_up_swiglu(tensors):
    import reference

    ref = reference.gate_up_swiglu(
        tensors["a_q"], tensors["a_scale"],
        tensors["w_gate"], tensors["w_gate_scale"],
        tensors["w_up"], tensors["w_up_scale"],
    )
    tensors["act"].copy_(ref.to(tensors["act"].dtype))


def compare_act(actual, expected, **_kwargs):
    """Whole-tensor and worst-row relative distance to the float64 golden."""
    dev, ref = actual.double(), expected.double()
    rel = float((dev - ref).norm() / ref.norm())
    row = float(((dev - ref).norm(dim=-1) / ref.norm(dim=-1)).max())
    detail = (f"[stats] act vs float64 golden: rel frob {rel:.3e} (gate {MAX_REL}), "
              f"worst row {row:.3e} (gate {MAX_ROW_REL})")
    ok = rel <= MAX_REL and row <= MAX_ROW_REL
    return ok, "" if ok else detail


if __name__ == "__main__":
    import argparse
    import os

    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--fp32-act", action="store_true", default=False,
                        help="the control: no added rounding, and no storage saving")
    parser.add_argument("--weights", type=str, default=None,
                        help="one layer's BF16 mlp tensors from weights.py; default: random weights")
    parser.add_argument("--m-group", type=int, default=M_GROUP)
    parser.add_argument("--n-group", type=int, default=N_GROUP)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    if args.rounds:
        os.environ["PYPTO_BENCH"] = "1"
        os.environ["PYPTO_BENCH_ROUNDS"] = str(args.rounds)
        os.environ["PYPTO_BENCH_WARMUP"] = "5"

    act_dtype = pl.FP32 if args.fp32_act else pl.BF16
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if args.seq_len and args.seq_len % M_TILE:
        parser.error(f"--seq-len must be a multiple of M_TILE={M_TILE}")
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    for seq_len in cases:
        print(f"[case] gate_up_swiglu T={seq_len}", flush=True)
        m_group = args.m_group
        if (seq_len // M_TILE) % m_group:
            m_group = next(side for side in M_PREFER if (seq_len // M_TILE) % side == 0)
        result = run(
            fn=build_kernel(m=seq_len, m_group=m_group, n_group=args.n_group,
                            act_dtype=act_dtype),
            specs=build_tensor_specs(t=seq_len, act_dtype=act_dtype, weights=args.weights),
            golden_fn=golden_gdn_gate_up_swiglu,
            compare_fn={"act": compare_act},
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode,
                        device_id=args.device),
            rtol=1e-2,
            atol=1e-2,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={seq_len}")
