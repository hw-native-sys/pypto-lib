# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet solve_tril -- the unit-triangular inverse `(I + A)^-1`.

`A` is strictly lower triangular per (chunk, head), so `I + A` is unit lower
triangular and its inverse is again unit lower triangular. Stage 4 of the GDN
pipeline: it sits between scaled_dot_kkt, which produces `A`, and wy_fast, which
consumes the inverse.

Straight port of `megagdn-pto/kernels/pto/tri_inverse_impl.cpp`
(`InvertSingleTile`), which is two phases:

  phase 1, the "inv trick", on the FRACTAL-sized diagonal blocks only:
      N = blockdiag_F(A);  X = I - N;  Y = N @ N
      NDOUBLE x:  X = X + X @ Y;  Y = Y @ Y   (Y not updated on the last pass)
  which is the alternating Neumann sum (I-N)(I+N^2)(I+N^4)...(I+N^(F/2)) --
  EXACT, not truncated, because N is nilpotent with N^F = 0 inside an F-block.

  phase 2, the unrolled recursion, doubling the block size:
      for bs in F, 2F, 4F ... < CHUNK:
          E = even bs-diagonal blocks of X;  O = odd bs-diagonal blocks
          X = O + (I - O @ A) @ E
  which is the 2x2 block inverse [[P,0],[-Q A21 P, Q]] applied to every pair of
  blocks at once. Off-block garbage accumulates in X and is harmless: the next
  level's mask discards it, and at the last level there is no next level.

  (The reference calls this branch `swap_parity=true`. Its default branch is the
  upper-triangular mirror, `X = E + (I - E @ A) @ O`. `A` here is lower.)

FRACTAL is a tuned knob, and it is set to CHUNK
----------------------------------------------
Phase 1 *is* the doubling algorithm; FRACTAL says how much of the matrix it
covers. Every doubling of F removes one phase-2 level -- 4 matmuls and one round
of cube<->vector crossings -- and adds one phase-1 step, which is 2. At
F == CHUNK there is no phase 2 at all, no mask, and therefore **not a single
vector op in the kernel**: no cross-core ring, no split, Vec usage 0.

Measured on a2a3, T = 8192 (megagdn-pto's own tri_inverse is 512.1 us on the
same card in the same grant, 418.5 us with the two improvements below applied
to it):

    F    matmuls  phase-2 levels    us      frob (real A)   frob (synthetic A)
    16      20          3         1049.3         --            7.544e-05
    32      18          2          953.0     3.625e-06         7.500e-05
    128     14          0          468.3     2.338e-07         8.807e-04   <- shipped

**Which accuracy contract applies decides this, and it is not a tuning question.**
There are two upstream criteria:

  * `pto-kernels/tests/test_tri_inv_rec_unroll.py` -- `tri_inverse` as a
    GENERAL-PURPOSE triangular inverse, on a synthetic `0.1 * rand` input
    (cond ~6.9): `allclose(5e-5, 0.1)` AND frob <= 1e-4.
  * `megagdn-pto/tests/test_gdn_single_kernels.py::test_solve_tril` -- the GDN
    STAGE, on real `A` from `kkt` (cond ~1.25): frob <= 1e-3.

This is a GDN stage, so the second applies, and F = 128 passes it: megagdn's own
test suite runs **19/19** with this kernel, fixed and varlen shapes alike. On
real `A` it is also 15x MORE accurate than F = 32 (2.338e-07 against 3.625e-06,
which is 0.06x the FP16 floor against 1.00x), and end to end the whole pipeline
scores 4.0738e-04 against F = 32's 4.0737e-04 -- identical to four figures.
F = 128 misses the general-kernel bar only, on an input this pipeline cannot
generate.

Two changes carry it, and both also improve megagdn-pto's own kernel (measured:
512.1 -> 480.8 us at frob 7.711e-05 against 8.081e-05, strictly better on both
axes):

  1. **F = CHUNK**, above.
  2. **Accumulator reuse.** The reference spends a matmul per level copying X
     into a fresh accumulator (`TMATMUL(c, X, I)`) because its L0C buffers are
     needed for other things in between; ours are not, and `xa` already holds X.
     Same arithmetic in one matmul instead of two -- and X then keeps its FP32
     accumulator across levels rather than round-tripping through the FP16
     operand copy, so it is slightly more accurate as well.

Both of the reference's structural tricks are now moot here, and neither ever
cost it anything. Ablating megagdn-pto's own kernel (`../devtools/mega_ablation/`)
showed `TEXTRACT` block selection is worth **0.0%** to it and holding constants
resident in L1 **0.25%** -- it is bound by its serial chain of `[C,C]` matmuls
with everything else hidden underneath. At F = CHUNK there is no block selection
to express at all, so PyPTO having no cube-side gather (`pl.tile.extract` exists
but is unreachable from `@pl.jit`; see KNOWN_PYPTO_ISSUES.md) no longer matters.

Output is FP32, as the reference's `tri_inverse` is. That is not only faithful,
it is free: storing a matmul result straight to GM uses no vector buffer, while
narrowing it to FP16 on the way out is a VECTOR op and drags the whole [C,C] FP32
tile across (measured: Vec 0 and no ring against Vec 163840 and a 131072 ring).
The reference pipeline narrows to FP16 separately before wy_fast
(`mega_kernel.py`: `A_inv_f32` -> `A_inv`), and so must ours -- that narrowing
belongs to the pipeline, not to this stage.
"""
import pypto.language as pl

# Model
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads
CHUNK = 128             # chunk size in tokens; A is [CHUNK, CHUNK] per head
FRACTAL = 128           # doubling block size == CHUNK: pure doubling, no recursion
NDOUBLE = 6             # X updates after X = I - N, log2(FRACTAL) - 1
NLEVEL = 0              # block-recursion levels, log2(CHUNK / FRACTAL) -- none

# Constant cube operands. At FRACTAL == CHUNK only the identities are used: the
# block mask is all ones and there is no recursion to mask for, so the whole
# kernel is cube-only and nothing crosses to the vector unit.
C_I = 0                 # identity (unused at FRACTAL == CHUNK, kept for the layout)
C_NEG_I = 1             # negative identity
NCONST = 2


@pl.jit
def gdn_solve_tril(
    a_in: pl.Tensor[[T, H, CHUNK], pl.FP16],
    consts: pl.Tensor[[NCONST * CHUNK, CHUNK], pl.FP16],
    t_out: pl.Out[pl.Tensor[[T, H, CHUNK], pl.FP32]],
):
    a_flat = pl.reshape(a_in, [T, H * CHUNK])
    t_flat = pl.reshape(t_out, [T, H * CHUNK])
    neg_i = pl.slice(consts, [CHUNK, CHUNK], [C_NEG_I * CHUNK, 0])
    for t0 in pl.parallel(0, T, CHUNK):
        # No optimizations: at FRACTAL == CHUNK there is not a single vector op
        # in this kernel, so there is no cross-core ring to size and nothing for
        # pl.split to halve. Vec usage is 0.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="solve_tril"):
            for h in pl.range(H):
                col = h * CHUNK
                a16 = pl.slice(a_flat, [CHUNK, CHUNK], [t0, col])

                # N is A itself: the FRACTAL-block mask is all ones here.
                n16 = a16
                xa = pl.matmul(n16, neg_i, out_dtype=pl.FP32)
                xa = pl.matmul_acc(xa, neg_i, neg_i)            # X = I - N
                x16 = pl.cast(xa, target_type=pl.FP16, mode="rint")
                y16 = pl.cast(pl.matmul(n16, n16, out_dtype=pl.FP32),
                              target_type=pl.FP16, mode="rint")
                for it in pl.unroll(NDOUBLE - 1):
                    # Accumulate onto the FP32 X directly -- it already holds
                    # X, so no matmul is spent copying it into a fresh
                    # accumulator, and X keeps FP32 across levels instead of
                    # round-tripping through the FP16 operand copy.
                    xa = pl.matmul_acc(xa, x16, y16)            # X = X + X @ Y
                    x16 = pl.cast(xa, target_type=pl.FP16, mode="rint")
                    y16 = pl.cast(pl.matmul(y16, y16, out_dtype=pl.FP32),
                                  target_type=pl.FP16, mode="rint")
                # Last factor; Y is not needed again, and N^CHUNK = 0 makes
                # the alternating product exact at this point.
                xn = pl.matmul_acc(xa, x16, y16)

                t_flat = pl.assemble(t_flat, xn, [t0, col])
    return t_out


# ---------------------------------------------------------------------------
# Test data. `A` comes from upstream's own generator (`random_tri_matrix`,
# `0.1 * rand` strictly lower) rather than `randn`: with N(0,1) entries the true
# inverse of a 128x128 unit-triangular matrix reaches ~1e14 and the problem is
# too ill-conditioned for any dtype to say anything useful. Real GDN `A` is
# `-tril(diag(beta) K K^T, -1)` with row-normalised K, which is tamer still --
# validating against that is a separate step, since it is the conditioning, not
# the algorithm, that this stage is sensitive to.
# ---------------------------------------------------------------------------
_CACHE = {}


def _consts(chunk: int = CHUNK):
    """FP16 cube constants: identity and negative identity."""
    import torch

    eye = torch.eye(chunk)
    return torch.cat([eye, -eye], dim=0).to(torch.float16)


def _inputs(t: int, h: int, chunk: int):
    """`A` as scaled_dot_kkt produces it, which is the only `A` this stage ever
    sees: `-tril(beta_i * exp(g_i - g_j) * (K K^T), -1)` with K row-normalised
    and g a per-chunk cumulative sum of log-sigmoid gates.

    Not upstream's synthetic `0.1 * rand`. That input belongs to pto-kernels'
    STANDALONE triangular-inverse test, where `tri_inverse` is a general-purpose
    kernel; it is roughly 6x worse conditioned than anything this pipeline can
    generate (cond ~6.9 against ~1.25) and it is not what a GDN stage is
    contracted against. It is still worth knowing: this kernel scores 8.807e-04
    on it, inside megagdn's ftol but outside pto-kernels' stricter 1e-4 fp16 bar.
    `../devtools/d1/sweep.py` measures that case.
    """
    key = (t, h, chunk)
    if key in _CACHE:
        return _CACHE[key]
    import torch
    import torch.nn.functional as F

    torch.manual_seed(11)
    k = F.normalize(torch.randn(t, h, chunk, dtype=torch.float16).float(), dim=-1, p=2)
    beta = torch.rand(h, t)
    g = F.logsigmoid(torch.randn(h, t))
    g_sum = torch.zeros_like(g)
    for t0 in range(0, t, chunk):
        g_sum[:, t0 : t0 + chunk] = g[:, t0 : t0 + chunk].cumsum(dim=1)

    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    strict_lower = (rows > cols).float()
    a = torch.zeros(t, h, chunk)
    for t0 in range(0, t, chunk):
        for hh in range(h):
            kc = k[t0 : t0 + chunk, hh, :]
            gc = g_sum[hh, t0 : t0 + chunk]
            bc = beta[hh, t0 : t0 + chunk]
            # exp(min(g_i - g_j, 0)); on the strict lower triangle g is
            # decreasing so the clamp never binds, but keep it for fidelity
            decay = torch.exp(torch.minimum(gc[:, None] - gc[None, :],
                                            torch.zeros(chunk, chunk)))
            a[t0 : t0 + chunk, hh, :] = -(kc @ kc.T) * decay * bc[:, None] * strict_lower

    out = dict(a_in=a.to(torch.float16).contiguous())
    _CACHE[key] = out
    return out


def build_tensor_specs(t: int = T, h: int = H, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    data = _inputs(t, h, chunk)
    return [
        TensorSpec("a_in", [t, h, chunk], torch.float16, init_value=lambda: data["a_in"]),
        TensorSpec("consts", [NCONST * chunk, chunk], torch.float16, init_value=_consts),
        TensorSpec("t_out", [t, h, chunk], torch.float32, is_output=True),
    ]


def golden_gdn_solve_tril(tensors):
    """Upstream's `linalg_inv` (pto-kernels/tests/test_tri_inv_rec_unroll.py):
    the exact inverse of `I + A` in float64, per (chunk, head)."""
    import torch

    a = tensors["a_in"].double()
    out = tensors["t_out"]
    t, h, chunk = a.shape
    eye = torch.eye(chunk, dtype=torch.float64)
    for ci, t0 in enumerate(range(0, t, chunk)):
        for hh in range(h):
            blk = a[t0 : t0 + chunk, hh, :]
            out[t0 : t0 + chunk, hh, :] = torch.linalg.inv(eye + blk).float()


def _tri_inv_ok(actual, expected, **_kwargs):
    """megagdn-pto's acceptance criterion for this stage: relative Frobenius
    error <= 1e-3 (`tests/utils.py: NumericalAccuracy.ftol`), against `A` as the
    model produces it (`tests/test_gdn_single_kernels.py: test_solve_tril`).

    NOT pto-kernels' `test_tri_inv_rec_unroll.py` bar of 1e-4 with
    `allclose(5e-5, 0.1)`. That test covers `tri_inverse` as a general-purpose
    triangular inverse on a synthetic input, and this is a GDN stage. The
    distinction is not academic: it is what separates FRACTAL = 32 from
    FRACTAL = 128, and megagdn's own test suite passes 19/19 with the kernel as
    it stands here.

    The FP16 floor is reported alongside -- the error of the exact inverse merely
    rounded to FP16 -- because the pipeline narrows this stage's FP32 output to
    FP16 before wy_fast (`mega_kernel.py`: `A_inv_f32` -> `A_inv`), so the floor
    is how much of any budget that later step spends on its own.
    """
    import torch

    act = actual.double()
    exp = expected.double()

    def frob_of(x):
        return float(torch.sqrt(((exp - x) ** 2).sum() / (exp ** 2).sum()))

    frob = frob_of(act)
    floor = frob_of(exp.half().double())
    diff = (act - exp).abs()
    print(f"[stats] frob={frob:.4g} (<=1e-3, megagdn ftol)  fp16 floor={floor:.4g} "
          f"({frob / max(floor, 1e-30):.2f}x)  max diff={diff.max().item():.4g}  "
          f"peak |ref|={exp.abs().max().item():.4g}", flush=True)
    ok = frob <= 1e-3
    return ok, f"frob={frob:.4g} (<=1e-3), fp16 floor={floor:.4g}"


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
        fn=gdn_solve_tril,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_solve_tril,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"t_out": _tri_inv_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
