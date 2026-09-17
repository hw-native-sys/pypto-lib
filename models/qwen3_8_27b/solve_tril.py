# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet solve_tril: the unit-triangular inverse (I + A)^-1 per
(chunk, head), by a block-diagonal split plus the alternating Neumann product.

Write `A = D + N`, D block-diagonal in blocks of BLOCK and N strictly
block-lower. Then

    I + A = (I + D)(I + Xd N),   Xd = (I + D)^-1
    (I + A)^-1 = (I + M)^-1 Xd,  M = Xd N

`Xd` comes from the doubling `X = I - D; Y = D @ D; X += X @ Y; Y = Y @ Y`,
which is exact because D is strictly lower triangular within each block, so
`D^BLOCK = 0`. `M` is strictly block-lower, so `M^(chunk/BLOCK) = 0` and
`(I + M)^-1` terminates after log2(chunk/BLOCK) - 1 further doubling levels.

**Why the split, rather than doubling on the whole chunk.** The doubling holds
`A^(2^j)` as FP16 matmul operands, which is safe only while those powers
shrink. On synthetic keys they do -- `F.normalize(randn)` makes `k_i . k_j`
concentrate near zero and the row sums of A reach only ~0.10. A real layer's
keys are correlated: the row sums reach **37**, the powers stay ~1 instead of
shrinking, the intermediates pass **7e+05** against FP16's 65504, and the
kernel returned NaN -- silently, because the composed block's int8 hand-off
turned it into finite garbage. Splitting bounds each block's powers
independently while keeping every matmul full-tile, so the fix costs 16 matmuls
against 13 rather than the 52 that shrinking the tiles would take.

Measured at T = 8192, H = 48: **1622.1 us against 1269.6 for the unsplit form**,
and on the real layer 2.287e-04 against a 1e-3 gate where the unsplit form is
non-finite. BLOCK = 16 rather than 32 because the split makes the block size
free -- 16 matmuls either way -- so it takes the one with the most margin.

Output is FP32, as the reference's is. The pipeline narrows it to FP16 before
wy_fast; doing that here would be a vector op and would drag the whole [C, C]
FP32 tile across the cube/vector boundary.
"""
import pypto.language as pl

from config import GDN_TILING, QWEN3_8_27B

# model shape
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension; unused here, drawn inputs match the pipeline
CHUNK = GDN_TILING.chunk                    # chunk size in tokens, our tiling choice
BLOCK = 16              # doubling block; see the split above. 32 also passes, 64 does not

# case shape
T = 8192                # tokens (single sequence, B = 1)


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                 block: int = BLOCK, inline: bool = False, out_dtype=None):
    """The stage kernel at one shape; `d` is unused and accepted for a uniform signature.

    `out_dtype` is FP32 standalone, which is what the stage test scores. wy_fast
    reads A_inv as FP16, so `gdn_layer` asks for FP16 here and the narrowing rides
    on the last matmul instead of costing a second pass over [T, H, C].
    """
    out_dtype = pl.FP32 if out_dtype is None else out_dtype
    # A is nilpotent at A^chunk, and the doubling reaches A^(2^ndouble). Off a power
    # of two the product stops short and the kernel returns a truncated inverse with
    # no other symptom, so refuse the shape instead.
    if chunk & (chunk - 1):
        raise ValueError(f"chunk must be a power of two, got {chunk}")
    if block & (block - 1) or block < 4 or chunk % block or block >= chunk:
        raise ValueError(f"block must be a power of two in [4, {chunk}), got {block}")
    nblk = chunk // block
    nd_in = block.bit_length() - 2         # X updates for (I + D)^-1, since D^block = 0
    nd_out = nblk.bit_length() - 2         # X updates for (I + M)^-1, since M^nblk = 0

    @(pl.jit.inline if inline else pl.jit)
    def gdn_solve_tril(
        a_in: pl.Tensor[[t, h, chunk], pl.FP16],
        neg_eye2: pl.Tensor[[2 * chunk, chunk], pl.FP16],
        m_diag: pl.Tensor[[chunk, chunk], pl.FP16],
        m_low: pl.Tensor[[chunk, chunk], pl.FP16],
        t_out: pl.Out[pl.Tensor[[t, h, chunk], out_dtype]],
    ):
        a_flat = pl.reshape(a_in, [t, h * chunk])
        t_flat = pl.reshape(t_out, [t, h * chunk])
        # slot_num=1: the mask multiplies are vector ops, so a cross-core ring
        # exists and its default depth reserves more of the vector buffer than
        # the tile needs.
        for c0 in pl.spmd(t // chunk, name_hint="solve_tril",
                          optimizations=[pl.cross_core_slot(slot_num=1)]):
            t0 = c0 * chunk
            for hh in pl.range(h):
                col = hh * chunk
                neg_i = neg_eye2[0:chunk, :]
                # The A slice is written out three times on purpose: one bound
                # value cannot feed several vector ops. D is taken here and N at
                # its use below -- holding both across the doubling puts the
                # vector buffer 40 KB over its 188416.
                dm = pl.mul(a_flat[t0 : t0 + chunk, col : col + chunk], m_diag[:, :])

                # --- Xd = (I + D)^-1. `I - D` as two single-K matmuls, not one
                # double-K: the [chunk, 2*chunk] concat operand is 64 KB of vector
                # buffer, and two of them are 8 KB over the limit.
                nd = pl.matmul(dm, neg_i, out_dtype=pl.FP32)
                xa = pl.matmul_acc(nd, neg_i, neg_i)               # X = I - D
                xc = pl.cast(xa, target_type=pl.FP16, mode="rint")
                yb = pl.matmul(dm, dm, out_dtype=pl.FP32)          # Y = D @ D
                yc = pl.cast(yb, target_type=pl.FP16, mode="rint")
                for _ in pl.unroll(nd_in - 1):
                    xa = pl.matmul_acc(xa, xc, yc)                 # X += X @ Y
                    xc = pl.cast(xa, target_type=pl.FP16, mode="rint")
                    yb = pl.matmul(yc, yc, out_dtype=pl.FP32)
                    yc = pl.cast(yb, target_type=pl.FP16, mode="rint")
                xd = pl.matmul_acc(xa, xc, yc)
                xdf = pl.cast(xd, target_type=pl.FP16, mode="rint")

                # --- M = Xd N, strictly block-lower, so M^nblk = 0
                nm = pl.mul(a_flat[t0 : t0 + chunk, col : col + chunk], m_low[:, :])
                mm = pl.matmul(xdf, nm, out_dtype=pl.FP32)
                mf = pl.cast(mm, target_type=pl.FP16, mode="rint")

                # --- (I + M)^-1 = (I - M)(I + M^2)(I + M^4)...
                nmm = pl.matmul(mf, neg_i, out_dtype=pl.FP32)
                pa = pl.matmul_acc(nmm, neg_i, neg_i)              # X = I - M
                pc = pl.cast(pa, target_type=pl.FP16, mode="rint")
                qb = pl.matmul(mf, mf, out_dtype=pl.FP32)          # Y = M @ M
                qc = pl.cast(qb, target_type=pl.FP16, mode="rint")
                for _ in pl.unroll(nd_out - 1):
                    pa = pl.matmul_acc(pa, pc, qc)
                    pc = pl.cast(pa, target_type=pl.FP16, mode="rint")
                    qb = pl.matmul(qc, qc, out_dtype=pl.FP32)
                    qc = pl.cast(qb, target_type=pl.FP16, mode="rint")
                pn = pl.matmul_acc(pa, pc, qc)
                pnf = pl.cast(pn, target_type=pl.FP16, mode="rint")

                # --- X = (I + M)^-1 Xd. No cast: the store carries the dtype, so
                # an FP16 t_out narrows the FP32 accumulator on the way out.
                t_flat[t0 : t0 + chunk, col : col + chunk] = pl.matmul(
                    pnf, xdf, out_dtype=pl.FP32)
        return t_out

    return gdn_solve_tril


gdn_solve_tril = build_kernel()


def blk_masks(chunk: int = CHUNK, block: int = BLOCK):
    """The block-diagonal indicator and its strictly-block-lower complement.

    Two constants rather than one plus a subtraction: `N = A - D` would need D
    live where the doubling has just finished with it, and holding it there is
    the 32 KB that does not fit.
    """
    import torch

    i = torch.arange(chunk)[:, None] // block
    j = torch.arange(chunk)[None, :] // block
    return (i == j).to(torch.float16), (i > j).to(torch.float16)


def neg_eye_stack(chunk: int):
    """`-I` stacked twice, `[2*chunk, chunk]`: the right operand of the merged setup.

    `[A | -I] @ [[-I], [-I]]` is `I - A`, so the same block serves both halves
    of the contraction and the kernel needs one constant rather than two.
    """
    import torch

    return -torch.cat([torch.eye(chunk, dtype=torch.float16)] * 2)


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK,
                       hg: int = HG, block: int = BLOCK, out_dtype=None):
    # hg only picks which reference chain to draw from; this stage reads no q or k.
    import torch
    from golden import TensorSpec

    import reference

    return [
        TensorSpec("a_in", [t, h, chunk], torch.float16,
                   init_value=reference.lazy("solve_tril", "a16", t, h, d, chunk, hg=hg)),
        TensorSpec("neg_eye2", [2 * chunk, chunk], torch.float16,
                   init_value=lambda: neg_eye_stack(chunk)),
        TensorSpec("m_diag", [chunk, chunk], torch.float16,
                   init_value=lambda: blk_masks(chunk, block)[0]),
        TensorSpec("m_low", [chunk, chunk], torch.float16,
                   init_value=lambda: blk_masks(chunk, block)[1]),
        TensorSpec("t_out", [t, h, chunk],
                   torch.float32 if out_dtype is None else out_dtype),
    ]


def golden_gdn_solve_tril(tensors):
    import reference

    chunk = tensors["a_in"].shape[-1]
    tensors["t_out"].copy_(reference.solve_tril(tensors["a_in"], chunk))


def _tri_inv_ok(actual, expected, **_kwargs):
    """megagdn-pto's criterion for this stage, plus the FP16 floor.

    The floor -- the error of the exact inverse merely rounded to FP16 -- is
    reported because the pipeline narrows this stage's FP32 output to FP16 before
    wy_fast, so it says how much of any budget that later step spends on its own.
    """
    import reference

    exp = expected.double()
    _, floor_detail = reference.stats_ok(exp.half(), exp)
    ok, detail = reference.stats_ok(actual, exp)
    print(f"[stats] {detail}  |  fp16 floor: {floor_detail}", flush=True)
    return ok, detail


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()

    result = run(
        fn=gdn_solve_tril,
        specs=build_tensor_specs(),
        golden_fn=golden_gdn_solve_tril,
        golden_data=args.golden_data,
        runtime_dir=args.runtime_dir,
        save_data=args.save_data,
        config=dict(platform=args.platform, device_id=args.device),
        rtol=1e-2,
        atol=1e-5,
        compare_fn={"t_out": _tri_inv_ok},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
