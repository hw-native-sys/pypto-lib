# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Gated DeltaNet solve_tril: the unit-triangular inverse (I + A)^-1 per
(chunk, head), by the alternating Neumann product

    X = I - A;  Y = A @ A
    log2(CHUNK) - 1 times:  X = X + X @ Y;  Y = Y @ Y

which is exact, not truncated: A is strictly lower triangular, so A^CHUNK = 0 and
the product (I-A)(I+A^2)(I+A^4)...(I+A^(CHUNK/2)) terminates at the inverse.

Output is FP32, as the reference's is. The pipeline narrows it to FP16 before
wy_fast; doing that here would be a vector op and would drag the whole [C, C]
FP32 tile across the cube/vector boundary.
"""
import pypto.language as pl

# model config
T = 8192                # tokens (single sequence, B = 1)
H = 16                  # value heads
CHUNK = 128             # chunk size in tokens; A is [CHUNK, CHUNK] per head
D = 128                 # head dimension; unused here, drawn inputs match the pipeline


def build_kernel(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    """The stage kernel at one shape; `d` is unused and accepted for a uniform signature."""
    ndouble = chunk.bit_length() - 2       # X updates after X = I - A, log2(CHUNK) - 1

    @pl.jit
    def gdn_solve_tril(
        a_in: pl.Tensor[[t, h, chunk], pl.FP16],
        neg_eye: pl.Tensor[[chunk, chunk], pl.FP16],
        t_out: pl.Out[pl.Tensor[[t, h, chunk], pl.FP32]],
    ):
        a_flat = pl.reshape(a_in, [t, h * chunk])
        t_flat = pl.reshape(t_out, [t, h * chunk])
        neg_i = neg_eye[:, :]
        # No optimizations: the kernel has no vector op, so there is no cross-core
        # ring to size and nothing for pl.split to halve. Vec usage is 0.
        for c0 in pl.spmd(t // chunk, name_hint="solve_tril"):
            t0 = c0 * chunk
            for hh in pl.range(h):
                col = hh * chunk
                n16 = a_flat[t0 : t0 + chunk, col : col + chunk]
                xa = pl.matmul(n16, neg_i, out_dtype=pl.FP32)
                xa = pl.matmul_acc(xa, neg_i, neg_i)            # X = I - A
                x16 = pl.cast(xa, target_type=pl.FP16, mode="rint")
                y32 = pl.matmul(n16, n16, out_dtype=pl.FP32)
                y16 = pl.cast(y32, target_type=pl.FP16, mode="rint")
                for _ in pl.unroll(ndouble - 1):
                    # accumulate onto the FP32 X in place: no matmul is spent copying
                    # it into a fresh accumulator, and X stays FP32 across levels
                    xa = pl.matmul_acc(xa, x16, y16)            # X = X + X @ Y
                    x16 = pl.cast(xa, target_type=pl.FP16, mode="rint")
                    y32 = pl.matmul(y16, y16, out_dtype=pl.FP32)
                    y16 = pl.cast(y32, target_type=pl.FP16, mode="rint")
                xn = pl.matmul_acc(xa, x16, y16)                # last factor; Y is dead
                t_flat[t0 : t0 + chunk, col : col + chunk] = xn
        return t_out

    return gdn_solve_tril


gdn_solve_tril = build_kernel()


def build_tensor_specs(t: int = T, h: int = H, d: int = D, chunk: int = CHUNK):
    import torch
    from golden import TensorSpec

    from models.gdn import reference

    return [
        TensorSpec("a_in", [t, h, chunk], torch.float16,
                   init_value=reference.lazy("solve_tril", "a16", t, h, d, chunk)),
        TensorSpec("neg_eye", [chunk, chunk], torch.float16,
                   init_value=lambda: -torch.eye(chunk, dtype=torch.float16)),
        TensorSpec("t_out", [t, h, chunk], torch.float32, is_output=True),
    ]


def golden_gdn_solve_tril(tensors):
    from models.gdn import reference

    chunk = tensors["a_in"].shape[-1]
    tensors["t_out"].copy_(reference.solve_tril(tensors["a_in"], chunk))


def _tri_inv_ok(actual, expected, **_kwargs):
    """megagdn-pto's criterion for this stage, plus the FP16 floor.

    The floor -- the error of the exact inverse merely rounded to FP16 -- is
    reported because the pipeline narrows this stage's FP32 output to FP16 before
    wy_fast, so it says how much of any budget that later step spends on its own.
    """
    from models.gdn import reference

    exp = expected.double()
    _, floor_detail = reference.stats_ok(exp.half(), exp)
    ok, detail = reference.stats_ok(actual, exp)
    print(f"[stats] {detail}  |  fp16 floor: {floor_detail}", flush=True)
    return ok, detail


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
