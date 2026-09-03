# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet chunk_h -- the inter-chunk state recurrence.

Per chunk c, per head, with S the state entering the chunk:

    snapshot_c = S
    V_new      = U - W @ S                                  the delta-rule correction
    S          = exp(g_last) * S + K^T (V_new * exp(g_last - g))

Stage 6 of the GDN pipeline. It produces both inputs chunk_o consumes: the
per-chunk state snapshots and V_new.

This is the only stage with a cross-chunk carry, so chunks CANNOT run in
parallel. Work is parallel over heads and the chunks run sequentially within a
head -- the reference's partition (`megagdn-pto/kernels/pto/chunk_h.cpp`:
`total_work = batch_size * H`, then a sequential `for ci`). At B=1, H=16 that is
16 work items of 64 steps each.

The decay is applied to V_new inside the contraction, where the reference applies
it to K; the two are identical arithmetic, K^T (V c) == (K c)^T V. That choice is
forced -- see the note at the contraction.

The decay is rebased on the chunk's last gate: `coeff = exp(g_last - g_i)` is at
most 1, where the algebraically equal `exp(g_last) * exp(-g_i)` reaches e^90 on a
128-token chunk and overflows FP32. Hence the state decay cannot be factored out
of the contraction, and `g_last` is read as a scalar.
"""
import pypto.language as pl

# Model
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads (= key heads; no GQA in the default build)
D = 128                 # head dimension
CHUNK = 128             # chunk size in tokens
NCHUNK = T // CHUNK     # sequential steps per head

# No tiling: the state, W @ S and K^T @ V_new are all carried at full [., D]
# width, so the chunk body is two matmuls and one cube<->vector round trip.
# Splitting the state into column halves -- two narrower matmuls per side, which
# halves the [., D] FP32 crossing tile and the ring reserve with it -- was
# measured and is 45% SLOWER (577.0 us against 397.8, bit-identical output): the
# extra round trip costs far more than the buffer it saves, and the full width
# fits at 78% of Vec anyway. Same result chunk_o got from widening COL_TILE.


@pl.jit
def gdn_chunk_h(
    k: pl.Tensor[[T, H, D], pl.FP16],
    w: pl.Tensor[[T, H, D], pl.FP16],
    u: pl.Tensor[[T, H, D], pl.FP16],
    g_sum: pl.Tensor[[H, T], pl.FP32],
    state: pl.Out[pl.Tensor[[NCHUNK * H * D, D], pl.FP16]],
    v_new: pl.Out[pl.Tensor[[T, H, D], pl.FP16]],
):
    k_flat = pl.reshape(k, [T, H * D])
    w_flat = pl.reshape(w, [T, H * D])
    u_flat = pl.reshape(u, [T, H * D])
    v_flat = pl.reshape(v_new, [T, H * D])
    for h in pl.parallel(0, H, 1):
        # An a2a3 core has two vector sub-cores, and an unsplit mixed region runs
        # its vector work on lane 0 only. Splitting rows across both halves every
        # vector tile -- including the carried state, which the reference also
        # holds half per sub-block (`s_ub` is [HalfC, D]). Here the split is not
        # a tuning choice: without it the kernel needs 197632 B of a 188416 B
        # vector buffer and does not build at all.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="chunk_h",
                   optimizations=[pl.cross_core_slot(slot_num=1),
                                  pl.split(pl.SplitMode.UP_DOWN)]):
            s0 = pl.full([D, D], dtype=pl.FP32, value=0.0)
            for c, (s_cur, st, vf) in pl.range(
                    NCHUNK, init_values=(s0, state, v_flat)):
                t0 = c * CHUNK
                row = (c * H + h) * D

                # snapshot the state ENTERING this chunk -- chunk_o reads these
                s16 = pl.cast(s_cur, target_type=pl.FP16, mode="rint")
                st_next = pl.assemble(st, s16, [row, 0])

                # coeff[i] = exp(g_last - g_i), decay = exp(g_last). Both exp
                # arguments are <= 0. The unary exp comes BEFORE the [1,C]->[C,1]
                # reshape: an elementwise op after that reshape loses pl.split's
                # tracking (see chunk_o's note).
                g_last = pl.read(g_sum, [h, t0 + CHUNK - 1])
                g_row = pl.slice(g_sum, [1, CHUNK], [h, t0])
                neg_g = pl.sub(pl.full([1, CHUNK], dtype=pl.FP32, value=0.0), g_row)
                coeff = pl.reshape(pl.exp(pl.add(neg_g, g_last)), [CHUNK, 1])
                decay = pl.reshape(
                    pl.exp(pl.add(pl.full([1, D], dtype=pl.FP32, value=0.0), g_last)),
                    [D, 1])

                # V_new = U - W @ S, in FP32 as the reference does
                wc = pl.slice(w_flat, [CHUNK, D], [t0, h * D])
                ws = pl.matmul(wc, s16, out_dtype=pl.FP32)
                uc = pl.slice(u_flat, [CHUNK, D], [t0, h * D])
                vc = pl.sub(pl.cast(uc, target_type=pl.FP32), ws)
                vc16 = pl.cast(vc, target_type=pl.FP16, mode="rint")

                # S = exp(g_last) * S + K^T @ (coeff * V_new).
                # The decay is applied to V rather than to K -- identical
                # arithmetic, K^T (V c) == (K c)^T V, and it is the spelling that
                # builds. Under the lane split above, a matmul whose TRANSPOSED
                # operand was produced on the vector unit cannot be lowered
                # (ptoas: "'pto.tmov' op expects a supported tmov address-space
                # pair"); a vector-scaled K would be exactly that, while K read
                # straight from GM is fine. The same matmul compiles without the
                # split, and the split is not optional here. It is also one FP32
                # [CHUNK, D] tile and one cast cheaper than the reference's
                # scaled-K form. Isolation matrix:
                # KNOWN_PYPTO_ISSUES/split-transposed-vector-operand.py
                kc = pl.slice(k_flat, [CHUNK, D], [t0, h * D])
                vs = pl.row_expand_mul(vc, coeff)
                vs16 = pl.cast(vs, target_type=pl.FP16, mode="rint")
                kv = pl.matmul(kc, vs16, a_trans=True, out_dtype=pl.FP32)
                s_next = pl.add(pl.row_expand_mul(s_cur, decay), kv)

                vf_next = pl.assemble(vf, vc16, [t0, h * D])

                s_end, st_end, vf_end = pl.yield_(s_next, st_next, vf_next)
            state = st_end
            v_flat = vf_end
    return state, v_new


# ---------------------------------------------------------------------------
# Test data. The draw order matches chunk_o's `_pipeline_inputs` (q is drawn and
# discarded here) so the two stages see the same k / w / u / gates and can be
# chained end to end.
# ---------------------------------------------------------------------------
_CACHE = {}


def _inputs(t: int, h: int, d: int, chunk: int):
    key = (t, h, d, chunk)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    import torch.nn.functional as F

    torch.manual_seed(42)
    _q = F.normalize(torch.randn(t, h, d, dtype=torch.float16), dim=-1, p=2)
    k = F.normalize(torch.randn(t, h, d, dtype=torch.float16), dim=-1, p=2)
    w = torch.randn(t, h, d, dtype=torch.float16)
    u = torch.randn(t, h, d, dtype=torch.float16)
    g_in = F.logsigmoid(torch.randn(t, h, dtype=torch.float32))

    g_cumsum = torch.zeros_like(g_in)
    for t0 in range(0, t, chunk):
        g_cumsum[t0 : t0 + chunk] = g_in[t0 : t0 + chunk].cumsum(dim=0)

    out = dict(k=k, w=w, u=u, g_sum=g_cumsum.t().contiguous())
    _CACHE[key] = out
    return out


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    data = _inputs(t, h, d, chunk)
    nc = t // chunk

    return [
        TensorSpec("k", [t, h, d], torch.float16, init_value=lambda: data["k"]),
        TensorSpec("w", [t, h, d], torch.float16, init_value=lambda: data["w"]),
        TensorSpec("u", [t, h, d], torch.float16, init_value=lambda: data["u"]),
        TensorSpec("g_sum", [h, t], torch.float32, init_value=lambda: data["g_sum"]),
        TensorSpec("state", [nc * h * d, d], torch.float16, is_output=True),
        TensorSpec("v_new", [t, h, d], torch.float16, is_output=True),
    ]


def golden_gdn_chunk_h(tensors):
    """Upstream's ref_chunk_h (pto-kernels/tests/test_gdn_chunk_h.py), in FP32."""
    import torch

    k = tensors["k"].float()
    w = tensors["w"].float()
    u = tensors["u"].float()
    g = tensors["g_sum"].float()
    state = tensors["state"]
    v_new = tensors["v_new"]
    t, h, d = k.shape
    nc = t // CHUNK
    for hh in range(h):
        s = torch.zeros(d, d, dtype=torch.float32)
        for ci in range(nc):
            t0 = ci * CHUNK
            gc = g[hh, t0 : t0 + CHUNK]
            gl = gc[-1]
            row = (ci * h + hh) * d
            state[row : row + d, :] = s.half()
            vc = u[t0 : t0 + CHUNK, hh, :] - w[t0 : t0 + CHUNK, hh, :] @ s
            v_new[t0 : t0 + CHUNK, hh, :] = vc.half()
            kv = k[t0 : t0 + CHUNK, hh, :].T @ (vc * torch.exp(gl - gc)[:, None])
            s = torch.exp(gl) * s + kv


def _stats_ok(actual, expected, **_kwargs):
    """Upstream's acceptance criterion for this kernel (test_gdn_chunk_h.py).

    Upstream's absolute hard fail (max abs diff <= 1.0) presumes an output FP16
    can resolve to better than 1.0, which holds for the sequence lengths it tests
    (T <= 512). Here W is unnormalised noise, so W @ S drives V_new to ~4.5e3 by
    chunk 64, and FP16's spacing at that magnitude is already 4 -- no
    implementation can meet an absolute 1.0, and the two nearest representable
    values straddle it. The check is therefore applied only where the dtype can
    express it, and the max diff is reported either way.
    """
    import numpy as np
    import torch

    diff = (actual.float() - expected.float()).abs()
    peak = float(expected.float().abs().max())
    ulp16 = float(np.spacing(np.float16(peak))) if peak > 0 else 0.0
    mean_abs = float(expected.float().abs().mean())
    rmse = float(torch.sqrt((diff.flatten() ** 2).mean()))
    ratio = rmse / max(mean_abs, 1e-15)
    # The harness prints a comparator's detail only on FAIL, and at this
    # magnitude the elementwise tolerance passes on its own -- so report the
    # statistics here, where they are always visible.
    print(f"[stats] rmse/mean={ratio:.4g}  max diff={diff.max().item():.4g}  "
          f"peak |ref|={peak:.4g}  fp16 ulp there={ulp16:.4g}", flush=True)
    if peak <= 1024.0 and diff.max().item() > 1.0:
        return False, f"max abs diff {diff.max().item():.4g} exceeds hard fail 1.0"
    if bool((diff <= 1e-5 + 1e-2 * expected.float().abs()).all()):
        # Report the statistics anyway: at this magnitude the elementwise
        # tolerance is loose enough to pass on its own, which says little.
        return True, (f"elementwise rtol/atol met; rmse/mean={ratio:.4g}, "
                      f"max diff={diff.max().item():.4g}, peak |ref|={peak:.4g}, "
                      f"fp16 ulp there={ulp16:.4g}")
    if mean_abs < 1e-9:
        return rmse < 5e-4, f"rmse={rmse:.4g}"
    ref = expected.float().flatten().numpy().astype(np.float64)
    pred = actual.float().flatten().numpy().astype(np.float64)
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    r2 = float("nan") if ss_tot < 1e-30 else 1.0 - float(np.sum((ref - pred) ** 2)) / ss_tot
    ok = ratio <= 0.05 and np.isfinite(r2) and r2 >= 0.99
    return ok, (f"rmse/mean={ratio:.4g} (<=0.05), r2={r2:.6f} (>=0.99), "
                f"max diff={diff.max().item():.4g} (peak |ref|={peak:.4g}, "
                f"fp16 ulp there={ulp16:.4g})")


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
        fn=gdn_chunk_h,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_chunk_h,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"state": _stats_ok, "v_new": _stats_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
