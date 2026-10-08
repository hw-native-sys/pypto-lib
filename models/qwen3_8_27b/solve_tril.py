# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet solve_tril: the unit-triangular inverse (I + A)^-1 per (chunk, head).

`A = D + N` with D block-diagonal in blocks of BLOCK_TILE and N strictly
block-lower, so

    I + A = (I + D)(I + Xd N),   Xd = (I + D)^-1
    (I + A)^-1 = (I + M)^-1 Xd,  M = Xd N

Each factor comes from the doubling `X = I - D; Y = D @ D; X += X @ Y; Y = Y @ Y`,
exact because D^BLOCK_TILE = 0 and M^(CHUNK_TILE/BLOCK_TILE) = 0.
"""
import pypto.language as pl

from config import GDN_TILING, PREFILL_SEQ, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
H = QWEN3_8_27B.linear_num_value_heads      # value heads
HG = QWEN3_8_27B.linear_num_key_heads       # QK heads; H // HG value heads share one
D = QWEN3_8_27B.linear_value_head_dim       # head dimension; unused here, drawn inputs match the pipeline
CHUNK_TILE = GDN_TILING.chunk               # chunk size in tokens, our tiling choice
A_WIDTH = H * CHUNK_TILE
BLOCK = 16              # doubling block; see the split above. 32 also passes, 64 does not
T = PREFILL_SEQ                             # tokens (single sequence, B = 1)


# A is nilpotent at A^CHUNK_TILE, and the doubling reaches A^(2^ndouble). Off a power
# of two the product stops short and the kernel returns a truncated inverse with
# no other symptom, so refuse the shape instead.
if CHUNK_TILE & (CHUNK_TILE - 1):
    raise ValueError(f"CHUNK_TILE must be a power of two, got {CHUNK_TILE}")
if BLOCK & (BLOCK - 1) or BLOCK < 4 or CHUNK_TILE % BLOCK or BLOCK >= CHUNK_TILE:
    raise ValueError(f"BLOCK must be a power of two in [4, {CHUNK_TILE}), got {BLOCK}")
NBLK = CHUNK_TILE // BLOCK
ND_IN = BLOCK.bit_length() - 2          # X updates for (I + D)^-1, since D^BLOCK = 0
ND_OUT = NBLK.bit_length() - 2          # X updates for (I + M)^-1, since M^NBLK = 0


def _gdn_solve_tril(
    a_in: pl.Tensor[[T_DYN, H, CHUNK_TILE], pl.FP16],
    eye: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    m_diag: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    m_low: pl.Tensor[[CHUNK_TILE, CHUNK_TILE], pl.FP16],
    t_out: pl.Out[pl.Tensor[[T_DYN, H, CHUNK_TILE], pl.FP16]],
):
    """The token count must be a multiple of CHUNK_TILE; a block walks one whole chunk."""
    a_in.bind_dynamic(0, T_DYN)
    t_out.bind_dynamic(0, T_DYN)
    t_dim = pl.tensor.dim(a_in, 0)
    a_flat = pl.reshape(a_in, [t_dim, A_WIDTH])
    t_flat = pl.reshape(t_out, [t_dim, A_WIDTH])
    # slot_num=1: the mask multiplies are vector ops, so a ring exists, and
    # its default depth reserves more of the vector buffer than the tile
    # needs (pypto#2903).
    for c0 in pl.spmd(t_dim // CHUNK_TILE, name_hint="solve_tril",
                      optimizations=[pl.cross_core_slot(slot_num=1)]):
        t0 = c0 * CHUNK_TILE
        for hh in pl.range(H):
            col = hh * CHUNK_TILE
            ident = eye[:, :]
            # The A slice is read twice: one bound value cannot feed several
            # vector ops. The masks carry a minus sign, so this is -D and -N
            # below, the negated form both matmuls want.
            dm = pl.mul(a_flat[t0 : t0 + CHUNK_TILE, col : col + CHUNK_TILE], m_diag[:, :])

            # Xd = (I + D)^-1, as two single-K matmuls rather than one double-K.
            nd = pl.matmul(dm, ident, out_dtype=pl.FP32)       # -D into an acc
            xa = pl.matmul_acc(nd, ident, ident)               # X = I - D
            xc = pl.cast(xa, target_type=pl.FP16, mode="rint")
            yb = pl.matmul(dm, dm, out_dtype=pl.FP32)          # Y = D @ D, sign squared
            yc = pl.cast(yb, target_type=pl.FP16, mode="rint")
            for _ in pl.unroll(ND_IN - 1):
                xa = pl.matmul_acc(xa, xc, yc)                 # X += X @ Y
                xc = pl.cast(xa, target_type=pl.FP16, mode="rint")
                yb = pl.matmul(yc, yc, out_dtype=pl.FP32)
                yc = pl.cast(yb, target_type=pl.FP16, mode="rint")
            xd = pl.matmul_acc(xa, xc, yc)
            xdf = pl.cast(xd, target_type=pl.FP16, mode="rint")

            # --- -M = Xd (-N), strictly block-lower, so M^nblk = 0. Taking the
            # negated N here means the product lands in the accumulator with
            # the sign `I - M` needs, so the identity costs one matmul rather
            # than a negate-then-add pair. (-M)^2 = M^2, so the squaring below
            # is unaffected.
            nm = pl.mul(a_flat[t0 : t0 + CHUNK_TILE, col : col + CHUNK_TILE], m_low[:, :])
            mn = pl.matmul(xdf, nm, out_dtype=pl.FP32)         # -M
            mf = pl.cast(mn, target_type=pl.FP16, mode="rint")

            # --- (I + M)^-1 = (I - M)(I + M^2)(I + M^4)...
            pa = pl.matmul_acc(mn, ident, ident)               # X = I - M
            pc = pl.cast(pa, target_type=pl.FP16, mode="rint")
            qb = pl.matmul(mf, mf, out_dtype=pl.FP32)          # Y = M @ M
            qc = pl.cast(qb, target_type=pl.FP16, mode="rint")
            for _ in pl.unroll(ND_OUT - 1):
                pa = pl.matmul_acc(pa, pc, qc)
                pc = pl.cast(pa, target_type=pl.FP16, mode="rint")
                qb = pl.matmul(qc, qc, out_dtype=pl.FP32)
                qc = pl.cast(qb, target_type=pl.FP16, mode="rint")
            pn = pl.matmul_acc(pa, pc, qc)
            pnf = pl.cast(pn, target_type=pl.FP16, mode="rint")

            # --- X = (I + M)^-1 Xd. No cast: the store carries the dtype, so
            # an FP16 t_out narrows the FP32 accumulator on the way out.
            t_out_tile = pl.matmul(pnf, xdf, out_dtype=pl.FP32)
            t_flat[t0 : t0 + CHUNK_TILE, col : col + CHUNK_TILE] = t_out_tile
    return t_out


gdn_solve_tril = pl.jit.inline(_gdn_solve_tril)
gdn_solve_tril_test = pl.jit(_gdn_solve_tril)


def blk_masks(chunk: int = CHUNK_TILE, block: int = BLOCK):
    """The negated block-diagonal indicator and its strictly-block-lower complement.

    Negated because both matmuls that consume them want `-D` and `-N`: that is
    what lets `matmul_acc(acc, I, I)` finish `I - D` and `I - M` in one matmul
    each. `D @ D` and `M @ M` square the sign away.

    Two constants rather than one plus a subtraction: `N = A - D` would need D
    live where the doubling has just finished with it, and holding it there is
    the 32 KB that does not fit.
    """
    import torch

    i = torch.arange(chunk)[:, None] // block
    j = torch.arange(chunk)[None, :] // block
    return -(i == j).to(torch.float16), -(i > j).to(torch.float16)


def eye_block(chunk: int):
    """`I`, `[chunk, chunk]`: the only constant the cube needs.

    `matmul_acc(acc, I, I)` adds the identity to an accumulator, which is how both
    `I - D` and `I - M` are formed. A positive identity serves as well as a
    negative one here, since the increment squares its sign.
    """
    import torch

    return torch.eye(chunk, dtype=torch.float16)


_INPUTS: dict = {}


def _inputs(t: int, h: int, d: int, chunk: int, hg: int) -> dict:
    """The chain this stage consumes, run here rather than shared with the other stages.

    Memoised so the several specs that draw from it pay for it once.
    """
    import torch

    import reference

    key = (t, h, d, chunk, hg)
    if key not in _INPUTS:
        st = reference.make_inputs(t, h, d, hg)
        g_sum = reference.cumsum(st["g"], chunk)
        st["a16"] = reference.kkt(st["k"], st["beta"], g_sum, chunk).to(torch.float16)
        _INPUTS[key] = st
    return _INPUTS[key]


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK_TILE,
                       hg: int = HG, block: int = BLOCK):
    # hg only picks which reference chain to draw from; this stage reads no q or k.
    import torch
    from golden import TensorSpec


    def draw(key, transform=None):
        def make():
            value = _inputs(t, h, d, chunk, hg)[key]
            return transform(value) if transform is not None else value

        return make


    return [
        TensorSpec("a_in", [t, h, chunk], torch.float16,
                   init_value=draw("a16")),
        TensorSpec("eye", [chunk, chunk], torch.float16,
                   init_value=lambda: eye_block(chunk)),
        TensorSpec("m_diag", [chunk, chunk], torch.float16,
                   init_value=lambda: blk_masks(chunk, block)[0]),
        TensorSpec("m_low", [chunk, chunk], torch.float16,
                   init_value=lambda: blk_masks(chunk, block)[1]),
        TensorSpec("t_out", [t, h, chunk], torch.float16),
    ]


def golden_gdn_solve_tril(tensors):
    import reference

    chunk = tensors["a_in"].shape[-1]
    tensors["t_out"].copy_(reference.solve_tril(tensors["a_in"], chunk))


def _tri_inv_ok(actual, expected, **_kwargs):
    """Relative Frobenius norm against the float64 golden, plus the FP16 rounding floor."""
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
    parser.add_argument("-p", "--platform", type=str, default="a2a3", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=T)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    args = parser.parse_args()
    if args.seq_len % CHUNK_TILE:
        parser.error(f"--seq-len must be a multiple of CHUNK_TILE={CHUNK_TILE}")

    result = run(
        fn=gdn_solve_tril_test,
        specs=build_tensor_specs(t=args.seq_len),
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
