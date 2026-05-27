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


def build_tri_inverse_program_batched(n: int = N, batch: int = 1):
    """Batched build: invert `batch` independent strict-lower n x n matrices.

    Layout follows the paged_attention dispatch idiom
    (examples/models/07_paged_attention_multi_config.py:524-538):
      - A and out are flat 2D tensors of shape [batch * n, n]; the b'th
        matrix occupies rows [b*n : (b+1)*n].
      - Orchestration runs `for b in pl.range(batch)`, slices A and the
        scratch buffers at offset [b*n, 0], and calls the single-matrix
        InCore (cayley_core) once per element. The cube engine handles
        intra-matmul core parallelism inside each InCore call.
    """
    if n not in (64, 128):
        raise ValueError(
            f"n={n} not supported. Current implementation uses full-[n,n] "
            f"tiles which overflow L0 caches at n>128. Supported: 64, 128."
        )
    n_steps = max(1, (n - 1).bit_length())
    big = batch * n  # flat row-count covering all batch elements

    @pl.program
    class BatchedTriInverseProgram:
        # Same per-matrix InCore as the single-matrix program; copied here
        # because @pl.program classes don't share functions across programs.
        @pl.function(type=pl.FunctionType.InCore)
        def cayley_core(
            A: pl.Tensor[[n, n], pl.FP32],
            identity: pl.Tensor[[n, n], pl.FP32],
            X_buf: pl.Out[pl.Tensor[[n, n], pl.FP32]],
            Y_buf: pl.Out[pl.Tensor[[n, n], pl.FP32]],
        ) -> pl.Tensor[[n, n], pl.FP32]:
            A_l1 = pl.load(A, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)
            I_l1 = pl.load(identity, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)

            seed_l0c = pl.matmul(
                pl.move(I_l1, target_memory=pl.MemorySpace.Left),
                pl.move(I_l1, target_memory=pl.MemorySpace.Right),
            )
            X_buf = pl.store(seed_l0c, [0, 0], X_buf)

            A_l0c = pl.matmul(
                pl.move(A_l1, target_memory=pl.MemorySpace.Left),
                pl.move(I_l1, target_memory=pl.MemorySpace.Right),
            )
            Y_buf = pl.store(A_l0c, [0, 0], Y_buf)

            for k, (X_iter, Y_iter) in pl.range(n_steps, init_values=(X_buf, Y_buf)):
                X_mat = pl.load(X_iter, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)
                Y_mat = pl.load(Y_iter, [0, 0], [n, n], target_memory=pl.MemorySpace.Mat)

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

                y_new_l0c = pl.matmul(
                    pl.move(Y_mat, target_memory=pl.MemorySpace.Left),
                    pl.move(Y_mat, target_memory=pl.MemorySpace.Right),
                )
                Y_new = pl.store(y_new_l0c, [0, 0], Y_iter)

                (X_buf, Y_buf) = pl.yield_(X_new, Y_new)

            return X_buf

        @pl.function(type=pl.FunctionType.Orchestration)
        def tri_inverse(
            self,
            A: pl.Tensor[[big, n], pl.FP32],            # flat batch
            identity: pl.Tensor[[n, n], pl.FP32],       # shared across batch
            out: pl.Out[pl.Tensor[[big, n], pl.FP32]],  # flat batch
        ) -> pl.Tensor[[big, n], pl.FP32]:
            # Per-element Y scratch.  X uses the per-batch slice of `out`
            # directly as both the working buffer and the final result.
            Y_scratch = pl.create_tensor([big, n], dtype=pl.FP32)

            # `out` is carried as a pl.range iter_arg so each iter's
            # pl.assemble result flows into the next iter as the base
            # tensor.
            for b_idx, (out_iter,) in pl.range(batch, init_values=(out,)):
                offset = b_idx * n
                A_b         = pl.slice(A,         [n, n], [offset, 0])
                X_buf_b     = pl.slice(out_iter,  [n, n], [offset, 0])
                Y_scratch_b = pl.slice(Y_scratch, [n, n], [offset, 0])

                X_result = self.cayley_core(A_b, identity, X_buf_b, Y_scratch_b)

                out_new = pl.assemble(out_iter, X_result, [offset, 0])
                (out,) = pl.yield_(out_new)

            return out

    return BatchedTriInverseProgram


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


def build_batched_tensor_specs(n: int = N, batch: int = 1):
    """Specs for the batched program: A and out are flat [batch*n, n]."""
    import torch
    from golden import TensorSpec

    big = batch * n

    def init_strict_lower():
        gen = torch.Generator().manual_seed(0)
        scale = 1.0 / (4.0 * (n ** 0.5))
        # Independent random matrix per batch element, each strict-lower.
        A_flat = torch.empty(big, n, dtype=torch.float32)
        for b in range(batch):
            mat = torch.randn(n, n, dtype=torch.float32, generator=gen) * scale
            A_flat[b * n : (b + 1) * n] = mat.tril(diagonal=-1)
        return A_flat.contiguous()

    def init_identity():
        return torch.eye(n, dtype=torch.float32).contiguous()

    return [
        TensorSpec("A", [big, n], torch.float32, init_value=init_strict_lower),
        TensorSpec("identity", [n, n], torch.float32, init_value=init_identity),
        TensorSpec("out", [big, n], torch.float32, is_output=True),
    ]


def golden_tri_inverse_batched(tensors):
    """Reference: torch.linalg.inv(I - A_b) per batch element."""
    import torch
    A_flat = tensors["A"]
    n = A_flat.shape[-1]
    batch = A_flat.shape[0] // n
    eye = torch.eye(n, dtype=torch.float32)
    for b in range(batch):
        A_b = A_flat[b * n : (b + 1) * n]
        tensors["out"][b * n : (b + 1) * n] = torch.linalg.inv(eye - A_b)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-n", "--matrix-size", type=int, default=N)
    parser.add_argument("-b", "--batch", type=int, default=None,
                        help="If set, run only the batched build at this batch size. "
                             "If unset, run the default sweep: single-matrix kernel, "
                             "batched at B=2 (catches batch-index bugs), and batched "
                             "at B=4.")
    args = parser.parse_args()

    if args.batch is None:
        # Default sweep exercises both code paths in one CI invocation.
        cases = [("single", None), ("batched_b2", 2), ("batched_b4", 4)]
    else:
        cases = [(f"batched_b{args.batch}", args.batch)]

    failed: list[str] = []
    for label, b in cases:
        print(f"\n=== {label} ===", flush=True)
        if b is None:
            program = build_tri_inverse_program(n=args.matrix_size)
            specs   = build_tensor_specs(n=args.matrix_size)
            gfn     = golden_tri_inverse
        else:
            program = build_tri_inverse_program_batched(n=args.matrix_size, batch=b)
            specs   = build_batched_tensor_specs(n=args.matrix_size, batch=b)
            gfn     = golden_tri_inverse_batched

        result = run(
            program=program,
            specs=specs,
            golden_fn=gfn,
            compile_cfg=dict(dump_passes=True),
            runtime_cfg=dict(platform=args.platform, device_id=args.device),
            rtol=2e-2, atol=2e-2,
        )
        if not result.passed:
            failed.append(label)
            if result.error:
                print(result.error)

    if failed:
        print(f"\nFAILED cases: {failed}")
        raise SystemExit(1)
