# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Triangular inverse — Cayley-Hamilton doubling algorithm.

Computes inv(I - A) for strict-lower-triangular A in cube + vector pipe.

Algorithm (terminates exactly in ceil(log2(n)) steps because A^n = 0 by
Cayley-Hamilton when A is strict-lower triangular and therefore nilpotent):

    X0 = I, Y0 = A
    for k in range(ceil(log2(n))):
        X_{k+1} = X_k(I + Y_k) = X_k + X_k @ Y_k
        Y_{k+1} = Y_k @ Y_k
    # Result: X = inv(I - A)

Each doubling step computes 2 matmuls + 1 add; for n=128 -> 7 steps ->
14 matmuls + 7 adds total, all on [128, 128] tiles.

Sign convention: solves inv(I - A), matching pypto2 Qwen `inverse_pto`
in `models/qwen3_next/gated_delta_rule_impl.py`. The reference
pypto2 implementation lives in
gdn-tri-inverse/src/gdn_tri_inverse/pypto_tri_inv.py (uses Y0 = -A,
which solves inv(I + A); we use Y0 = A directly to match Qwen).
"""

import pypto.language as pl


# ---------------------------------------------------------------------------
# Algorithm parameters
# ---------------------------------------------------------------------------
N = 128  # matrix dimension; fixed for the Qwen3-Next chunked GDN tri-inverse
M_TILE = 32  # row-tile size for the M-parallel inner loop
K_TILE = 64  # K-reduction tile (gemm-style K-blocking inside each matmul)
M_CHUNK = 1  # M-tiles bundled per incore kernel


def build_tri_inverse_program(
    n: int = N,
    m_tile: int = M_TILE,
    k_tile: int = K_TILE,
    m_chunk: int = M_CHUNK,
    refinement_steps: int = 1,
):
    """Build the @pl.program for n x n triangular inverse.

    Specialises the doubling-step count via ceil(log2(n)).

    ``refinement_steps`` runs that many Newton iterative-refinement steps
    after the doubling loop: at each step ``X <- X + X (I - (I - A) X)``,
    which is quadratically convergent and crushes the FP16-cube roundoff
    error from O(eps) to O(eps^2).  Default 1 — bring n=64 inside the 2e-2
    tolerance and tighten accuracy at all sizes.  Set to 0 for raw speed
    when O(eps) accuracy is acceptable.
    """
    n_steps = max(1, (n - 1).bit_length())  # 7 for n=128
    if n % m_tile != 0:
        raise ValueError(f"n ({n}) must be a multiple of m_tile ({m_tile})")
    if n % k_tile != 0:
        raise ValueError(f"n ({n}) must be a multiple of k_tile ({k_tile})")
    k_blocks = n // k_tile  # 2 for n=128, k_tile=64

    @pl.program
    class TriInverseProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def tri_inverse(
            self,
            A: pl.Tensor[[n, n], pl.FP32],
            identity: pl.Tensor[[n, n], pl.FP32],
            out: pl.Out[pl.Tensor[[n, n], pl.FP32]],
        ) -> pl.Tensor[[n, n], pl.FP32]:
            """Compute out = inv(I - A) via Cayley-Hamilton doubling.

            Args:
                A:        strict-lower-triangular FP32 matrix of shape [n, n]
                identity: precomputed identity matrix of shape [n, n]
                          (bench passes torch.eye(n); avoids needing pl.eye)
                out:      destination tensor, written in place

            X and Y are kept in pl.create_tensor (GM) buffers so they flow
            across the doubling iterations.  Each iteration becomes its own
            pl.at scope (so the Vec budget is per-iter), and inside each
            scope an inner pl.parallel chunks the M dimension into
            [m_tile, n] row slabs — each parallel iter runs one
            [m_tile, n] @ [n, n] matmul, fitting comfortably in the 192 KB
            Vec budget.

            Y is double-buffered across two tensors (Y_a, Y_b) so the
            Y = Y @ Y step reads from one buffer and writes to the other.
            Without double-buffering the matmul would read the full Y while
            other M-parallel iters write to row slabs of that same tensor,
            which races; reading and writing distinct tensors removes the
            race without needing an explicit GM-to-GM snapshot copy.
            """
            X_state = pl.create_tensor([n, n], dtype=pl.FP32)
            Y_a = pl.create_tensor([n, n], dtype=pl.FP32)
            Y_b = pl.create_tensor([n, n], dtype=pl.FP32)
            X_acc = pl.create_tensor([n, n], dtype=pl.FP32)
            R_state = pl.create_tensor([n, n], dtype=pl.FP32)  # newton refinement residual

            with pl.at(level=pl.Level.CORE_GROUP,
                       optimization=pl.chunked_loop_optimizer,
                       name_hint="cayley_init"):
                for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                    id_row = pl.slice(identity, [m_tile, n], [mb, 0])
                    a_row = pl.slice(A, [m_tile, n], [mb, 0])
                    X_state = pl.assemble(X_state, id_row, [mb, 0])
                    Y_a = pl.assemble(Y_a, a_row, [mb, 0])

            for step in pl.unroll(n_steps):
                # Double-buffered Y: each step reads Y_a, writes Y_b, then
                # swaps the Python-level bindings so the next iteration reads
                # the freshly-written buffer.  No explicit snapshot/copy scope
                # needed — the read and write target distinct GM tensors.
                #
                # Split X update: matmul first (writes X_acc), then add(X, X_acc).
                # Keeping x_row + acc + x_new alive in one scope is the dominant
                # Vec cost at n=256 (3 row tiles of size m_tile*n*4 bytes); splitting
                # roughly halves the live set per scope and lets the kernel
                # compile for n>=128 with the same m_tile=32, k_tile=64 config.
                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="cayley_x_matmul"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        xa0 = pl.slice(X_state, [m_tile, k_tile], [mb, 0])
                        yb0 = pl.slice(Y_a, [k_tile, n], [0, 0])
                        acc = pl.matmul(xa0, yb0)
                        for kb in pl.range(1, k_blocks):
                            k0 = kb * k_tile
                            xa = pl.slice(X_state, [m_tile, k_tile], [mb, k0])
                            yb = pl.slice(Y_a, [k_tile, n], [k0, 0])
                            acc = pl.matmul_acc(acc, xa, yb)
                        X_acc = pl.assemble(X_acc, acc, [mb, 0])

                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="cayley_x_add"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        x_row = pl.slice(X_state, [m_tile, n], [mb, 0])
                        acc_row = pl.slice(X_acc, [m_tile, n], [mb, 0])
                        x_new_row = pl.add(x_row, acc_row)
                        X_state = pl.assemble(X_state, x_new_row, [mb, 0])

                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="cayley_y_square"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        ya0 = pl.slice(Y_a, [m_tile, k_tile], [mb, 0])
                        yb0 = pl.slice(Y_a, [k_tile, n], [0, 0])
                        acc = pl.matmul(ya0, yb0)
                        for kb in pl.range(1, k_blocks):
                            k0 = kb * k_tile
                            ya = pl.slice(Y_a, [m_tile, k_tile], [mb, k0])
                            yb = pl.slice(Y_a, [k_tile, n], [k0, 0])
                            acc = pl.matmul_acc(acc, ya, yb)
                        Y_b = pl.assemble(Y_b, acc, [mb, 0])

                Y_a, Y_b = Y_b, Y_a

            # Newton iterative refinement.  Reference numpy implementation:
            #     for _ in range(refinement_steps):
            #         R = I - A @ X        # using A_eff = (I - A); R = I - A_eff X
            #         X = X + X @ R
            # In our setting A_eff = (I - A), so:
            #     R = I - (I - A) X = I - X + A X
            # Computed in five scopes per step (K-blocked matmuls split as
            # matmul + add for the same Vec-budget reason as the doubling X
            # update).  Each step uses no extra GM beyond R_state + the
            # existing X_acc scratch.
            for _ in pl.unroll(refinement_steps):
                # Step 1: R_state = A @ X_state (K-blocked matmul)
                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="newton_ax_matmul"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        a0 = pl.slice(A, [m_tile, k_tile], [mb, 0])
                        x0 = pl.slice(X_state, [k_tile, n], [0, 0])
                        acc = pl.matmul(a0, x0)
                        for kb in pl.range(1, k_blocks):
                            k0 = kb * k_tile
                            a = pl.slice(A, [m_tile, k_tile], [mb, k0])
                            x = pl.slice(X_state, [k_tile, n], [k0, 0])
                            acc = pl.matmul_acc(acc, a, x)
                        R_state = pl.assemble(R_state, acc, [mb, 0])

                # Step 2: R_state = R_state - X_state  (= A X - X)
                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="newton_r_sub_x"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        r_row = pl.slice(R_state, [m_tile, n], [mb, 0])
                        x_row = pl.slice(X_state, [m_tile, n], [mb, 0])
                        diff = pl.sub(r_row, x_row)
                        R_state = pl.assemble(R_state, diff, [mb, 0])

                # Step 3: R_state = R_state + identity  (= A X - X + I = R)
                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="newton_r_add_i"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        r_row = pl.slice(R_state, [m_tile, n], [mb, 0])
                        id_row = pl.slice(identity, [m_tile, n], [mb, 0])
                        s = pl.add(r_row, id_row)
                        R_state = pl.assemble(R_state, s, [mb, 0])

                # Step 4: X_acc = X_state @ R_state (K-blocked matmul)
                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="newton_xr_matmul"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        xa0 = pl.slice(X_state, [m_tile, k_tile], [mb, 0])
                        rb0 = pl.slice(R_state, [k_tile, n], [0, 0])
                        acc = pl.matmul(xa0, rb0)
                        for kb in pl.range(1, k_blocks):
                            k0 = kb * k_tile
                            xa = pl.slice(X_state, [m_tile, k_tile], [mb, k0])
                            rb = pl.slice(R_state, [k_tile, n], [k0, 0])
                            acc = pl.matmul_acc(acc, xa, rb)
                        X_acc = pl.assemble(X_acc, acc, [mb, 0])

                # Step 5: X_state = X_state + X_acc  (= X + X R)
                with pl.at(level=pl.Level.CORE_GROUP,
                           optimization=pl.chunked_loop_optimizer,
                           name_hint="newton_x_add"):
                    for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                        x_row = pl.slice(X_state, [m_tile, n], [mb, 0])
                        xr_row = pl.slice(X_acc, [m_tile, n], [mb, 0])
                        x_new_row = pl.add(x_row, xr_row)
                        X_state = pl.assemble(X_state, x_new_row, [mb, 0])

            with pl.at(level=pl.Level.CORE_GROUP,
                       optimization=pl.chunked_loop_optimizer,
                       name_hint="cayley_out"):
                for mb in pl.parallel(0, n, m_tile, chunk=m_chunk):
                    x_row = pl.slice(X_state, [m_tile, n], [mb, 0])
                    out = pl.assemble(out, x_row, [mb, 0])
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
        # Scale 1/(4*sqrt(n)) targets ||A||_op ~= 0.5 by random-matrix
        # theory (2*sigma*sqrt(n) bound). A is mathematically nilpotent
        # so the doubling terminates exactly, but the Ascend 910B2 cube
        # unit uses FP16 multiply + FP32 accumulate, so each of the 14
        # chained matmuls compounds ~5e-4 relative error. Keeping ||A||
        # small bounds the intermediate-tile magnitudes.
        #
        # A local torch.Generator (seed=0) makes the test reproducible
        # without touching the global RNG; unseeded randn occasionally
        # samples a worst-case ||A|| ~ 0.7-0.9 that pushes the final
        # error past rtol/atol = 1e-2.
        gen = torch.Generator().manual_seed(0)
        scale = 1.0 / (4.0 * (n ** 0.5))
        A = torch.randn(n, n, dtype=torch.float32, generator=gen) * scale
        # Strict-lower: zero on/above the diagonal.
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
    parser.add_argument("--enable-l2-swimlane", action="store_true", default=False)
    parser.add_argument("-n", "--matrix-size", type=int, default=N,
                        help="Matrix dimension (default: 128). "
                             "Must satisfy A^n = 0 for the algorithm to terminate.")
    args = parser.parse_args()

    result = run(
        program=build_tri_inverse_program(n=args.matrix_size),
        specs=build_tensor_specs(n=args.matrix_size),
        golden_fn=golden_tri_inverse,
        compile_cfg=dict(dump_passes=True),
        runtime_cfg=dict(
            platform=args.platform,
            device_id=args.device,
            enable_l2_swimlane=args.enable_l2_swimlane,
        ),
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
