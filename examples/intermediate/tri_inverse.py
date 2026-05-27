# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Triangular inverse — Cayley-Hamilton doubling algorithm.

Computes inv(I - A) for strict-lower-triangular A via the Cayley-Hamilton
doubling: X_{k+1} = X_k + X_k @ Y_k; Y_{k+1} = Y_k @ Y_k. After ceil(log2(n))
steps Y becomes zero (A is nilpotent) and X holds inv(I - A).

Kernel shape follows the paged_attention iter_args idiom:
- One InCore + one Orchestration function.
- State (X, Y) carried as GM tensor iter_args of a single pl.range.
- All ops on the cube engine: X update uses matmul_acc(X@Y_in_L0C, X_in_L0A,
  I_in_L0B) so X@Y is created in L0C by a fresh matmul, then the +X term is
  folded in via matmul_acc using X reloaded from Mat — avoiding the
  unsupported L0C → Mat tile-move on a2/a3.
- L0C → GM stores at end of each iter; iter_arg is the GM tensor handle.
"""

import pypto.language as pl


N = 128  # matrix dimension


def build_tri_inverse_program(n: int = N):
    """Build the @pl.program for n x n triangular inverse via Cayley-Hamilton.

    Supports n in {64, 128} on Ascend a2a3 today. For n > 128 the full-[n,n]
    tiles overflow L0 (Acc 128 KB and Right 64 KB caches); supporting larger
    n requires K-blocked matmuls inside the doubling loop.
    """
    if n not in (64, 128):
        raise ValueError(
            f"n={n} not supported. Current implementation uses full-[n,n] "
            f"tiles which overflow L0 caches at n>128. Supported: 64, 128."
        )
    n_steps = max(1, (n - 1).bit_length())  # 7 for n=128

    # Memory hierarchy on a2a3: GM (DDR) <-> Mat/Vec (L1) <-> Left/Right (L0A/B)
    # and Acc (L0C, cube output). Notable constraint: Acc -> Mat is *not*
    # a supported tile-move, so cube state can only leave L0C via pl.store
    # to GM.
    @pl.program
    class TriInverseProgram:
        @pl.function(type=pl.FunctionType.InCore)
        def cayley_core(
            A: pl.Tensor[[n, n], pl.FP32],
            identity: pl.Tensor[[n, n], pl.FP32],
            X_buf: pl.Out[pl.Tensor[[n, n], pl.FP32]],  # GM scratch carrying X
            Y_buf: pl.Out[pl.Tensor[[n, n], pl.FP32]],  # GM scratch carrying Y
        ) -> pl.Tensor[[n, n], pl.FP32]:
            # GM -> L1 Mat (one-shot; reloaded into L0A/L0B per cube op below).
            A_l1 = pl.load(A, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)
            I_l1 = pl.load(identity, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)

            # Seed X_buf <- I via an AIC-only matmul (I @ I = I -> L0C -> GM).
            # A direct pl.store of a Mat tile would route through Vec and
            # spawn an AIV function + 512 KB cube-to-vec pipe buffer — busting
            # the 192 KB AIV Vec budget. The matmul detour keeps it all AIC.
            seed_l0c = pl.matmul(
                pl.move(I_l1, target_memory=pl.MemorySpace.Left),
                pl.move(I_l1, target_memory=pl.MemorySpace.Right),
            )
            X_buf = pl.store(seed_l0c, [0, 0], X_buf)

            # Seed Y_buf <- A by the same trick (A @ I = A).
            A_l0c = pl.matmul(
                pl.move(A_l1, target_memory=pl.MemorySpace.Left),
                pl.move(I_l1, target_memory=pl.MemorySpace.Right),
            )
            Y_buf = pl.store(A_l0c, [0, 0], Y_buf)

            # Doubling loop. pl.range(..., init_values=(X, Y)) carries X / Y
            # as iter_args across iterations — they're GM tensor handles, not
            # L0 tiles, so the loop never needs to keep cube state alive
            # across boundaries.
            for k, (X_iter, Y_iter) in pl.range(n_steps, init_values=(X_buf, Y_buf)):
                X_mat = pl.load(X_iter, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)
                Y_mat = pl.load(Y_iter, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)

                # X_new = X + X @ Y, fused into one cube pipeline:
                #   (a) matmul(X_left, Y_right)         -> L0C  = X @ Y
                #   (b) matmul_acc(L0C, X_left, I_right) -> L0C += X
                # X is reloaded into Left for (b) because matmul consumes
                # the Left operand. Avoiding L0C->Mat is mandatory on a2a3.
                xy_l0c = pl.matmul(
                    pl.move(X_mat, target_memory=pl.MemorySpace.Left),
                    pl.move(Y_mat, target_memory=pl.MemorySpace.Right),
                )
                x_new_l0c = pl.matmul_acc(
                    xy_l0c,
                    pl.move(X_mat, target_memory=pl.MemorySpace.Left),
                    pl.move(I_l1, target_memory=pl.MemorySpace.Right),
                )
                X_new = pl.store(x_new_l0c, [0, 0], X_iter)

                # Y_new = Y @ Y.  L0C -> GM.
                y_new_l0c = pl.matmul(
                    pl.move(Y_mat, target_memory=pl.MemorySpace.Left),
                    pl.move(Y_mat, target_memory=pl.MemorySpace.Right),
                )
                Y_new = pl.store(y_new_l0c, [0, 0], Y_iter)

                # Rebind iter_args for the next iteration.
                (X_buf, Y_buf) = pl.yield_(X_new, Y_new)

            return X_buf

        @pl.function(type=pl.FunctionType.Orchestration)
        def tri_inverse(
            self,
            A: pl.Tensor[[n, n], pl.FP32],
            identity: pl.Tensor[[n, n], pl.FP32],
            out: pl.Out[pl.Tensor[[n, n], pl.FP32]],
        ) -> pl.Tensor[[n, n], pl.FP32]:
            # Y is internal state; allocate a per-call GM scratch tensor.
            Y_scratch = pl.create_tensor([n, n], dtype=pl.FP32)
            out = self.cayley_core(A, identity, out, Y_scratch)
            return out

    return TriInverseProgram


# ---------------------------------------------------------------------------
# Test harness — golden.run() wiring
# ---------------------------------------------------------------------------
def build_tensor_specs(n: int = N):
    """Three tensors: A (strict-lower), identity, out (write-only)."""
    import torch
    from golden import TensorSpec

    def init_strict_lower():
        gen = torch.Generator().manual_seed(0)
        scale = 1.0 / (4.0 * (n ** 0.5))
        A = torch.randn(n, n, dtype=torch.float32, generator=gen) * scale
        return A.tril(diagonal=-1).contiguous()

    def init_identity():
        return torch.eye(n, dtype=torch.float32).contiguous()

    return [
        TensorSpec("A", [n, n], torch.float32, init_value=init_strict_lower),
        TensorSpec("identity", [n, n], torch.float32, init_value=init_identity),
        TensorSpec("out", [n, n], torch.float32, is_output=True),
    ]


def golden_tri_inverse(tensors):
    """Reference: torch.linalg.inv(I - A) computed in FP32 on host."""
    import torch
    A = tensors["A"]
    n = A.shape[-1]
    eye = torch.eye(n, dtype=torch.float32)
    tensors["out"][:] = torch.linalg.inv(eye - A)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-n", "--matrix-size", type=int, default=N)
    args = parser.parse_args()

    result = run(
        program=build_tri_inverse_program(n=args.matrix_size),
        specs=build_tensor_specs(n=args.matrix_size),
        golden_fn=golden_tri_inverse,
        compile_cfg=dict(dump_passes=True),
        runtime_cfg=dict(platform=args.platform, device_id=args.device),
        rtol=2e-2, atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
