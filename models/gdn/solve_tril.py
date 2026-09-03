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
      3x:  X = X + X @ Y;  Y = Y @ Y        (Y not updated on the last pass)
  which is the alternating Neumann sum (I-N)(I+N^2)(I+N^4)(I+N^8) -- EXACT, not
  truncated, because N is nilpotent with N^F = 0 inside an F-block.

  phase 2, the unrolled recursion, doubling the block size:
      for bs in F, 2F, 4F ... < CHUNK:
          E = even bs-diagonal blocks of X;  O = odd bs-diagonal blocks
          X = O + (I - O @ A) @ E
  which is the 2x2 block inverse [[P,0],[-Q A21 P, Q]] applied to every pair of
  blocks at once. Off-block garbage accumulates in X and is harmless: the next
  level's mask discards it, and at the last level there is no next level.

  (The reference calls this branch `swap_parity=true`. Its default branch is the
  upper-triangular mirror, `X = E + (I - E @ A) @ O`. `A` here is lower.)

Two deviations from the reference, both forced by what PyPTO can express.

1. **Block selection is a vector mask multiply.** The reference selects diagonal
   blocks with `TEXTRACT`, a gather inside the cube's own operand memory, so its
   whole kernel is cube-only. PyPTO has no cube-side gather, so `E` and `O` come
   from an elementwise multiply by a constant 0/1 mask.
2. **X is carried as a tile across the phase-2 levels**, where the reference
   holds it in L1 and extracts sub-blocks from there. Staging it through GM
   instead -- store the level's result, read it back, mask it -- would be the
   closer analogue and would cost no cross-core ring at all, but PyPTO does not
   order an `assemble` of a MATMUL result against a later `slice` of the same
   region: the read comes back as zeros and the kernel silently produces zero
   output. (A vector tile written the same way reads back correctly, so it is
   specific to storing an Acc tile.) Carrying X in a tile means masking a matmul
   result, so every level pays one `[C,C]` FP32 cube->vector crossing.

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
FRACTAL = 16            # reference's FractalSize: the block the Neumann sum is exact on
NLEVEL = 3              # phase-2 levels, log2(CHUNK / FRACTAL)

# Constant cube operands and the phase-2 block masks, all FP16: X is narrowed on
# the way out of the FP32 staging buffer, before it is masked.
C_I = 0                 # identity
C_NEG_I = 1             # negative identity
C_FRAC = 2              # 1 inside the FRACTAL-sized diagonal blocks; multiplies A, so FP16
C_NEG_ONES = 3          # all -1: negates A on the vector unit, see below
NCONST = 4
M_EVEN = 0              # even bs-diagonal blocks, bs = 16, 32, 64
M_ODD = 3               # odd  bs-diagonal blocks, bs = 16, 32, 64
NMASK = 6


@pl.jit
def gdn_solve_tril(
    a_in: pl.Tensor[[T, H, CHUNK], pl.FP16],
    consts: pl.Tensor[[NCONST * CHUNK, CHUNK], pl.FP16],
    masks: pl.Tensor[[NMASK * CHUNK, CHUNK], pl.FP16],
    t_out: pl.Out[pl.Tensor[[T, H, CHUNK], pl.FP32]],
):
    a_flat = pl.reshape(a_in, [T, H * CHUNK])
    t_flat = pl.reshape(t_out, [T, H * CHUNK])
    ident = pl.slice(consts, [CHUNK, CHUNK], [C_I * CHUNK, 0])
    neg_i = pl.slice(consts, [CHUNK, CHUNK], [C_NEG_I * CHUNK, 0])
    m_frac = pl.slice(consts, [CHUNK, CHUNK], [C_FRAC * CHUNK, 0])
    neg_ones = pl.slice(consts, [CHUNK, CHUNK], [C_NEG_ONES * CHUNK, 0])
    for t0 in pl.parallel(0, T, CHUNK):
        # Every vector tile here is a [CHUNK, CHUNK] FP16 mask or masked operand
        # (32768 B each, and there are six live); splitting rows across the two
        # AIV lanes halves all of them. No matmul here transposes an operand, so
        # the split's transposed-operand defect does not apply.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="solve_tril",
                   optimizations=[pl.cross_core_slot(slot_num=1),
                                  pl.split(pl.SplitMode.UP_DOWN)]):
            for h in pl.range(H):
                col = h * CHUNK
                a16 = pl.slice(a_flat, [CHUNK, CHUNK], [t0, col])

                # phase 1 -- inv trick on the FRACTAL-sized diagonal blocks
                n16 = pl.mul(a16, m_frac)
                x_acc = pl.matmul(n16, neg_i, out_dtype=pl.FP32)
                x_acc = pl.matmul_acc(x_acc, neg_i, neg_i)      # X = I - N
                x16 = pl.cast(x_acc, target_type=pl.FP16, mode="rint")
                y16 = pl.cast(pl.matmul(n16, n16, out_dtype=pl.FP32),
                              target_type=pl.FP16, mode="rint")
                for it in pl.unroll(NLEVEL - 1):
                    xa = pl.matmul(x16, ident, out_dtype=pl.FP32)
                    xa = pl.matmul_acc(xa, x16, y16)            # X = X + X @ Y
                    x16 = pl.cast(xa, target_type=pl.FP16, mode="rint")
                    y16 = pl.cast(pl.matmul(y16, y16, out_dtype=pl.FP32),
                                  target_type=pl.FP16, mode="rint")
                # last pass leaves Y alone -- the reference does the same, and
                # N^16 = 0 makes the sum exact at this point anyway.
                xa = pl.matmul(x16, ident, out_dtype=pl.FP32)
                xa = pl.matmul_acc(xa, x16, y16)
                # -A on the VECTOR unit, not as the reference's (-I) @ A matmul.
                # A tensor slice cannot be both a vector operand and a cube
                # operand -- codegen rejects the second use with "'pto.tpush' op
                # tile type must map to a supported producer pipe" -- and `a16`
                # is already the vector operand of the mask multiply above. This
                # spelling keeps it vector-only and drops a matmul.
                neg_a = pl.mul(a16, neg_ones)
                xg = pl.cast(xa, target_type=pl.FP16, mode="rint")

                # phase 2 -- unrolled recursion, bs = FRACTAL, 2F, 4F
                for lv in pl.unroll(NLEVEL):
                    e16 = pl.mul(xg, pl.slice(masks, [CHUNK, CHUNK],
                                              [(M_EVEN + lv) * CHUNK, 0]))
                    o16 = pl.mul(xg, pl.slice(masks, [CHUNK, CHUNK],
                                              [(M_ODD + lv) * CHUNK, 0]))
                    z = pl.matmul(ident, ident, out_dtype=pl.FP32)
                    z = pl.matmul_acc(z, o16, neg_a)            # I - O @ A
                    y = pl.cast(z, target_type=pl.FP16, mode="rint")
                    xn = pl.matmul(o16, ident, out_dtype=pl.FP32)
                    xn = pl.matmul_acc(xn, y, e16)              # O + (I - O A) @ E
                    xg = pl.cast(xn, target_type=pl.FP16, mode="rint")

                # Only the final result reaches GM, in FP32 and uncast, so the
                # store itself costs nothing.
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


def _consts(chunk: int = CHUNK, fractal: int = FRACTAL):
    """FP16 constants: identity, -identity, the FRACTAL-block mask, all -1."""
    import torch

    idx = torch.arange(chunk)
    eye = torch.eye(chunk)
    frac = (idx[:, None] // fractal == idx[None, :] // fractal).float()
    neg_ones = -torch.ones(chunk, chunk)
    return torch.cat([eye, -eye, frac, neg_ones], dim=0).to(torch.float16)


def _masks(chunk: int = CHUNK, fractal: int = FRACTAL):
    """FP16 phase-2 block masks: even then odd, one per level."""
    import torch

    idx = torch.arange(chunk)
    blocks = []
    for parity in (0, 1):                                # M_EVEN then M_ODD
        for lv in range(NLEVEL):
            bs = fractal << lv
            bi = idx[:, None] // bs
            bj = idx[None, :] // bs
            blocks.append(((bi == bj) & (bi % 2 == parity)).float())
    return torch.cat(blocks, dim=0).to(torch.float16)


def _inputs(t: int, h: int, chunk: int):
    key = (t, h, chunk)
    if key in _CACHE:
        return _CACHE[key]
    import torch

    torch.manual_seed(42)
    a = 0.1 * torch.rand(t, h, chunk)
    rows = torch.arange(chunk)[:, None]
    cols = torch.arange(chunk)[None, :]
    strict_lower = (rows > cols).float()
    # `a` is [T, H, CHUNK]; row (t0 + i) column j of head hh is entry (i, j) of
    # that chunk's matrix, so the mask applies per chunk.
    a = a.view(t // chunk, chunk, h, chunk) * strict_lower[None, :, None, :]
    out = dict(a_in=a.reshape(t, h, chunk).to(torch.float16).contiguous())
    _CACHE[key] = out
    return out


def build_tensor_specs(t: int = T, h: int = H, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    data = _inputs(t, h, chunk)
    return [
        TensorSpec("a_in", [t, h, chunk], torch.float16, init_value=lambda: data["a_in"]),
        TensorSpec("consts", [NCONST * chunk, chunk], torch.float16, init_value=_consts),
        TensorSpec("masks", [NMASK * chunk, chunk], torch.float16, init_value=_masks),
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
    """Upstream's acceptance criterion for this kernel: np.allclose at
    atol=5e-5 / rtol=0.1, AND a relative Frobenius error at most 1e-4.

    Reported alongside it is the FP16 FLOOR -- the error of the exact inverse
    merely rounded to FP16. This stage emits FP32, as upstream's does, so the
    criterion is applied to the same dtype it was calibrated on; the floor is
    quoted because the pipeline narrows to FP16 before wy_fast
    (`mega_kernel.py`: `A_inv_f32` -> `A_inv`), and it says how much of the
    budget that later step will consume on its own.
    """
    import numpy as np
    import torch

    act = actual.double()
    exp = expected.double()

    def frob_of(x):
        return float(torch.sqrt(((exp - x) ** 2).sum() / (exp ** 2).sum()))

    frob = frob_of(act)
    floor = frob_of(exp.half().double())
    close = bool(np.allclose(act.numpy(), exp.numpy(), atol=5e-5, rtol=0.1))
    diff = (act - exp).abs()
    print(f"[stats] frob={frob:.4g} (<=1e-4)  fp16 floor={floor:.4g} "
          f"({frob / max(floor, 1e-30):.2f}x)  allclose(5e-5, 0.1)={close}  "
          f"max diff={diff.max().item():.4g}  peak |ref|={exp.abs().max().item():.4g}",
          flush=True)
    ok = close and frob <= 1e-4
    return ok, (f"frob={frob:.4g} (<=1e-4), fp16 floor={floor:.4g}, "
                f"allclose(atol=5e-5, rtol=0.1)={close}")


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
