# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_o -- the chunk output.

    inter = exp(g_i) * (Q @ S)                       inter-chunk, from the state
    intra = (Q @ K^T * exp(min(g_i - g_j, 0)) * causal) @ V_new    intra-chunk
    O     = inter + intra

Stage 7 of the GDN pipeline, and the last one to consume `g_sum`. `S` is the
state snapshot entering the chunk and `V_new` the residual-corrected values,
both produced by chunk_h.

The causal mask here includes the diagonal (`i >= j`), unlike scaled_dot_kkt's
strictly-lower one: a token attends to itself in the output but not in the
key-key matrix.

The intra term is blocked over its CONTRACTION axis. Column block j of the gated
score matrix multiplies rows j of V and accumulates in the cube via matmul_acc,
so the full [C, C] score matrix is never resident -- only one [C, COL_TILE]
block at a time. That keeps the vector buffer inside budget at C = 128 without
touching the arithmetic.
"""
import pypto.language as pl

# Model
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads (= key heads; no GQA in the default build)
D = 128                 # head dimension
CHUNK = 128             # chunk size in tokens
NCHUNK = T // CHUNK     # state snapshots, one per chunk

# Tiling
COL_TILE = 64           # contraction block of the intra term. With the row split
                        # below: 32 measures 681.5 us against 467.0 at 64 -- half
                        # as many cube<->vector round trips per head.
D_TILE = D // 2         # output columns per accumulator. The tile that crosses
                        # back to the vector unit is [CHUNK, D_TILE] FP32, and
                        # the cross-core ring reserves two of them, so halving D
                        # halves a 128 KB reserve that alone exceeded the budget.


@pl.jit
def gdn_chunk_o(
    q: pl.Tensor[[T, H, D], pl.FP16],
    k: pl.Tensor[[T, H, D], pl.FP16],
    v: pl.Tensor[[T, H, D], pl.FP16],
    state: pl.Tensor[[NCHUNK * H * D, D], pl.FP16],
    g_sum: pl.Tensor[[H, T], pl.FP32],
    mask: pl.Tensor[[CHUNK, CHUNK], pl.FP32],
    o_out: pl.Out[pl.Tensor[[T, H, D], pl.FP16]],
):
    q_flat = pl.reshape(q, [T, H * D])
    k_flat = pl.reshape(k, [T, H * D])
    v_flat = pl.reshape(v, [T, H * D])
    o_flat = pl.reshape(o_out, [T, H * D])
    for t0 in pl.parallel(0, T, CHUNK):
        # An a2a3 core has two vector sub-cores, and an unsplit mixed region
        # runs its vector work on lane 0 only. Splitting the rows across both
        # halves every vector tile, which also frees the 8704 B that a 2-slot
        # ring needs at COL_TILE=64 -- so the wider tile keeps its double
        # buffer. Measured: 561.0 (1-slot, no split) -> 467.0 us, -16.8%.
        # COL_TILE=128 still does not fit (229888 B), so the reference's
        # whole-[C,C] layout remains out of reach.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="chunk_o",
                   optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
            for h in pl.range(H):
                qc = pl.slice(q_flat, [CHUNK, D], [t0, h * D])
                g_col = pl.reshape(pl.slice(g_sum, [1, CHUNK], [h, t0]), [CHUNK, 1])
                # exp on the ROW vector, reshaped after -- NOT pl.exp(g_col).
                # Under pl.split, an elementwise op *following* a [1,C]->[C,1]
                # reshape loses the split tracking and lane 1 reads the wrong
                # half (max abs diff 36.86). Either ingredient alone is fine;
                # it is the order that matters. Repro + isolation matrix:
                # devtools/split-investigation/repro/.
                eg = pl.reshape(pl.exp(pl.slice(g_sum, [1, CHUNK], [h, t0])), [CHUNK, 1])

                # inter = exp(g_i) * (Q @ S), split over D so the tile crossing
                # back is [CHUNK, D_TILE] rather than [CHUNK, D].
                # (Folding exp(g_i) into Q instead would let one accumulator
                # carry both terms -- it is exact, and FP16-safe by measurement
                # -- but row_expand_mul cannot write a cube-operand layout:
                # "expects dst to use row-major layout".)
                row = (t0 // CHUNK) * (H * D) + h * D
                inter_lo = pl.row_expand_mul(
                    pl.matmul(qc, pl.slice(state, [D, D_TILE], [row, 0])), eg)
                inter_hi = pl.row_expand_mul(
                    pl.matmul(qc, pl.slice(state, [D, D_TILE], [row, D_TILE])), eg)

                # intra, blocked over its contraction axis: column block j of
                # the gated scores multiplies rows j of V and accumulates in the
                # cube, so the full [C, C] score matrix is never resident.
                #
                # Block 0 is peeled to SEED the accumulators. The accumulator has
                # to be a matmul result -- a tile the matrix unit owns, private to
                # this core group. pl.create_tensor looks like the way to declare
                # one, but it allocates a runtime TENSOR: a single buffer shared by
                # every chunk running in parallel, which corrupts intermittently
                # (clean at 16 concurrent chunks, ~40% wrong at 32).
                kj = pl.slice(k_flat, [COL_TILE, D], [t0, h * D])
                qk = pl.matmul(qc, kj, b_trans=True)
                g_row = pl.slice(g_sum, [1, COL_TILE], [h, t0])
                diff = pl.full([CHUNK, COL_TILE], dtype=pl.FP32, value=0.0)
                diff = pl.row_expand_add(diff, g_col)
                diff = pl.col_expand_sub(diff, g_row)
                decay = pl.exp(pl.minimum(diff, 0.0))
                gate = pl.mul(decay, pl.slice(mask, [CHUNK, COL_TILE], [0, 0]))
                # A matmul's result dtype follows its CONSUMER, not its operands
                # (see the note in the loop below).
                gated = pl.cast(pl.mul(qk, gate), target_type=pl.FP16, mode="rint")
                acc_lo = pl.matmul(gated,
                                   pl.slice(v_flat, [COL_TILE, D_TILE], [t0, h * D]),
                                   out_dtype=pl.FP32)
                acc_hi = pl.matmul(gated,
                                   pl.slice(v_flat, [COL_TILE, D_TILE], [t0, h * D + D_TILE]),
                                   out_dtype=pl.FP32)

                for jj in pl.unroll(CHUNK // COL_TILE - 1):
                    j0 = (jj + 1) * COL_TILE
                    kj = pl.slice(k_flat, [COL_TILE, D], [t0 + j0, h * D])
                    qk = pl.matmul(qc, kj, b_trans=True)
                    g_row = pl.slice(g_sum, [1, COL_TILE], [h, t0 + j0])
                    diff = pl.full([CHUNK, COL_TILE], dtype=pl.FP32, value=0.0)
                    diff = pl.row_expand_add(diff, g_col)
                    diff = pl.col_expand_sub(diff, g_row)
                    decay = pl.exp(pl.minimum(diff, 0.0))
                    gate = pl.mul(decay, pl.slice(mask, [CHUNK, COL_TILE], [0, j0]))
                    gated = pl.cast(pl.mul(qk, gate), target_type=pl.FP16, mode="rint")
                    acc_lo = pl.matmul_acc(
                        acc_lo, gated,
                        pl.slice(v_flat, [COL_TILE, D_TILE], [t0 + j0, h * D]))
                    acc_hi = pl.matmul_acc(
                        acc_hi, gated,
                        pl.slice(v_flat, [COL_TILE, D_TILE], [t0 + j0, h * D + D_TILE]))

                o_flat = pl.assemble(
                    o_flat,
                    pl.cast(pl.add(inter_lo, acc_lo), target_type=pl.FP16, mode="rint"),
                    [t0, h * D])
                o_flat = pl.assemble(
                    o_flat,
                    pl.cast(pl.add(inter_hi, acc_hi), target_type=pl.FP16, mode="rint"),
                    [t0, h * D + D_TILE])
    return o_out


# ---------------------------------------------------------------------------
# Test data. The state snapshots and V_new must come from the real chunk_h
# recurrence, not from randn: a state that has decayed through NCHUNK chunks has
# a magnitude structure that noise does not reproduce, and Q @ S in FP16 is
# exactly where a wrong magnitude would show up. Same lesson as synthesising A
# for wy_fast by an actual inverse rather than sampling one.
# ---------------------------------------------------------------------------
_CACHE = {}


def _pipeline_inputs(t: int, h: int, d: int, chunk: int):
    """Correlated inputs: q, k, and the chunk_h outputs (states, v_new) for them."""
    key = (t, h, d, chunk)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    q = F.normalize(torch.randn(t, h, d, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(t, h, d, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(t, h, d, dtype=torch.float16)
    u = torch.randn(t, h, d, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(t, h, dtype=torch.float32))

    g_cumsum = torch.zeros_like(g_in)
    for t0 in range(0, t, chunk):
        g_cumsum[t0 : t0 + chunk] = g_in[t0 : t0 + chunk].cumsum(dim=0)

    # Upstream's _ref_chunk_h (test_gdn_chunk_o.py), the stage-6 recurrence.
    nc = t // chunk
    states = torch.zeros(nc, h, d, d, dtype=torch.float32)
    v_new = torch.zeros(t, h, d, dtype=torch.float32)
    kf, wf, uf = k.float(), w.float(), u.float()
    for hh in range(h):
        S = torch.zeros(d, d, dtype=torch.float32)
        for ci in range(nc):
            s0, e0 = ci * chunk, (ci + 1) * chunk
            gc = g_cumsum[s0:e0, hh]
            gl = gc[-1]
            states[ci, hh] = S
            vc = uf[s0:e0, hh, :] - wf[s0:e0, hh, :] @ S
            v_new[s0:e0, hh, :] = vc
            S = torch.exp(gl) * S + kf[s0:e0, hh, :].T @ (vc * torch.exp(gl - gc)[:, None])

    out = dict(
        q=q, k=k,
        v=v_new.half(),
        state=states.half().reshape(nc * h * d, d),
        g_sum=g_cumsum.t().contiguous(),
    )
    _CACHE[key] = out
    return out


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    data = _pipeline_inputs(t, h, d, chunk)
    nc = t // chunk

    def init_mask():
        rows = torch.arange(chunk)[:, None]
        cols = torch.arange(chunk)[None, :]
        return (rows >= cols).float()      # inclusive diagonal

    return [
        TensorSpec("q", [t, h, d], torch.float16, init_value=lambda: data["q"]),
        TensorSpec("k", [t, h, d], torch.float16, init_value=lambda: data["k"]),
        TensorSpec("v", [t, h, d], torch.float16, init_value=lambda: data["v"]),
        TensorSpec("state", [nc * h * d, d], torch.float16,
                   init_value=lambda: data["state"]),
        TensorSpec("g_sum", [h, t], torch.float32, init_value=lambda: data["g_sum"]),
        TensorSpec("mask", [chunk, chunk], torch.float32, init_value=init_mask),
        TensorSpec("o_out", [t, h, d], torch.float16, is_output=True),
    ]


def golden_gdn_chunk_o(tensors):
    """Upstream's ref_chunk_o (test_gdn_chunk_o.py), in FP32."""
    import torch

    q = tensors["q"].float()
    k = tensors["k"].float()
    v = tensors["v"].float()
    g = tensors["g_sum"].float()
    state = tensors["state"].float()
    out = tensors["o_out"]
    t, h = q.shape[0], q.shape[1]
    rows = torch.arange(CHUNK)[:, None]
    cols = torch.arange(CHUNK)[None, :]
    causal = (rows >= cols).float()
    for ci, t0 in enumerate(range(0, t, CHUNK)):
        for hh in range(h):
            qc = q[t0 : t0 + CHUNK, hh, :]
            kc = k[t0 : t0 + CHUNK, hh, :]
            vc = v[t0 : t0 + CHUNK, hh, :]
            gc = g[hh, t0 : t0 + CHUNK]
            s_blk = state[(ci * h + hh) * D : (ci * h + hh) * D + D, :]
            inter = (qc @ s_blk) * torch.exp(gc)[:, None]
            gate = torch.exp(torch.minimum(gc[:, None] - gc[None, :],
                                           torch.zeros(CHUNK, CHUNK)))
            intra = ((qc @ kc.T) * gate * causal) @ vc
            out[t0 : t0 + CHUNK, hh, :] = (inter + intra).half()


def _stats_ok(actual, expected, **_kwargs):
    """Upstream's acceptance criterion for this kernel (test_gdn_chunk_o.py)."""
    import numpy as np
    import torch

    diff = (actual.float() - expected.float()).abs()
    if diff.max().item() > 1.0:
        return False, f"max abs diff {diff.max().item():.4g} exceeds hard fail 1.0"
    if bool((diff <= 1e-5 + 1e-2 * expected.float().abs()).all()):
        return True, ""
    mean_abs = float(expected.float().abs().mean())
    rmse = float(torch.sqrt((diff.flatten() ** 2).mean()))
    if mean_abs < 1e-9:
        return rmse < 5e-4, f"rmse={rmse:.4g}"
    ratio = rmse / max(mean_abs, 1e-15)
    ref = expected.float().flatten().numpy().astype(np.float64)
    pred = actual.float().flatten().numpy().astype(np.float64)
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = float("nan") if ss_tot < 1e-30 else 1.0 - float(np.sum((ref - pred) ** 2)) / ss_tot
    ok = ratio <= 0.05 and np.isfinite(r2) and r2 >= 0.99
    return ok, f"rmse/mean={ratio:.4g} (<=0.05), r2={r2:.6f} (>=0.99)"


if __name__ == "__main__":
    import argparse
    from golden import run_jit

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run_jit(
        fn=gdn_chunk_o,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_chunk_o,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"o_out": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
