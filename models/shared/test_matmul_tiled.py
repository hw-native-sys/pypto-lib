# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Smoke test for shared matmul_tiled / matmul_tiled_4d primitives.

Usage:
    python models/shared/test_matmul_tiled.py --smoke           # compile-only

Run via ``task-submit`` for SIM golden verification:
    task-submit --device auto --run "source ~/workspace/scripts/activate && cd ~/workspace/subprojects/pypto-lib && python models/shared/test_matmul_tiled.py -p a2a3sim"
"""

import torch

import pypto.language as pl
from models.shared.matmul import matmul_tiled, matmul_tiled_4d


# ── Constants matching a typical Qwen3-32B decode configuration ──
BATCH = 16
HIDDEN = 8192
M = BATCH
K = HIDDEN
K_CHUNK = 128
N_CHUNK = 256
K_BLOCKS = K // K_CHUNK  # 64
N_BLOCKS = HIDDEN // N_CHUNK  # 32

# 4D variant constants (matching 32b_decode_4d.py)
HIDDEN_K_BLOCKS = K_BLOCKS  # 64
HIDDEN_N1_BLOCKS = N_BLOCKS  # 32
K_CHUNK_4D = 128
N1_CHUNK = 256
BATCH_4D = BATCH


@pl.program
class TestMatmulTiled:
    """2D matmul_tiled: Q-projection-like shapes.

    Matches the pattern from ``qwen3_32b_decode.py``: SPMD loop over N blocks,
    each iteration calls ``matmul_tiled`` then assembles into the output tensor.
    """

    @pl.function
    def main(self,
             a: pl.Tensor[[M, K], pl.BF16],
             b: pl.Tensor[[K, N_BLOCKS * N_CHUNK], pl.BF16],
             out: pl.Out[pl.Tensor[[M, N_BLOCKS * N_CHUNK], pl.FP32]],
             ) -> pl.Tensor[[M, N_BLOCKS * N_CHUNK], pl.FP32]:
        for qi in pl.spmd(N_BLOCKS, name_hint="n_loop"):
            q0 = qi * N_CHUNK
            y = matmul_tiled(
                a, b, q0,
                m=M, k_chunk=K_CHUNK, n_chunk=N_CHUNK, k_blocks=K_BLOCKS, stages=2,
            )
            out = pl.assemble(out, y, [0, q0])
        return out


@pl.program
class TestMatmulTiled4D:
    """4D matmul_tiled_4d: Q-projection-like shapes in block-major layout.

    Matches ``qwen3_32b_decode_4d.py``: parallel loop over N1 blocks, each
    iteration calls ``matmul_tiled_4d`` then assembles into output.
    """

    @pl.function
    def main(self,
             a: pl.Tensor[[HIDDEN_K_BLOCKS, 1, BATCH_4D, K_CHUNK_4D], pl.BF16],
             b: pl.Tensor[[HIDDEN_K_BLOCKS, HIDDEN_N1_BLOCKS, K_CHUNK_4D, N1_CHUNK], pl.BF16],
             out: pl.Out[pl.Tensor[[HIDDEN_N1_BLOCKS, 1, BATCH_4D, N1_CHUNK], pl.FP32]],
             ) -> pl.Tensor[[HIDDEN_N1_BLOCKS, 1, BATCH_4D, N1_CHUNK], pl.FP32]:
        for qb in pl.parallel(HIDDEN_N1_BLOCKS):
            y = matmul_tiled_4d(
                a, b, qb,
                batch=BATCH_4D, k_chunk=K_CHUNK_4D, n1_chunk=N1_CHUNK, k_blocks=HIDDEN_K_BLOCKS, stages=2,
            )
            out = pl.assemble(out, y, [qb, 0, 0, 0])
        return out


def build_tensor_specs():
    """Build TensorSpecs for the 2D test program."""
    from golden import TensorSpec

    return [
        TensorSpec("a", [M, K], torch.bfloat16,
                   init_value=lambda: torch.rand(M, K).bfloat16() - 0.5),
        TensorSpec("b", [K, N_BLOCKS * N_CHUNK], torch.bfloat16,
                   init_value=lambda: torch.rand(K, N_BLOCKS * N_CHUNK).bfloat16() - 0.5),
        TensorSpec("out", [M, N_BLOCKS * N_CHUNK], torch.float32,
                   is_output=True),
    ]


def build_tensor_specs_4d():
    from golden import TensorSpec

    return [
        TensorSpec("a", [HIDDEN_K_BLOCKS, 1, BATCH_4D, K_CHUNK_4D], torch.bfloat16,
                   init_value=lambda: torch.rand(HIDDEN_K_BLOCKS, 1, BATCH_4D, K_CHUNK_4D).bfloat16() - 0.5),
        TensorSpec("b", [HIDDEN_K_BLOCKS, HIDDEN_N1_BLOCKS, K_CHUNK_4D, N1_CHUNK], torch.bfloat16,
                   init_value=lambda: torch.rand(HIDDEN_K_BLOCKS, HIDDEN_N1_BLOCKS, K_CHUNK_4D, N1_CHUNK).bfloat16() - 0.5),
        TensorSpec("out", [HIDDEN_N1_BLOCKS, 1, BATCH_4D, N1_CHUNK], torch.float32,
                   is_output=True),
    ]


def golden_matmul_2d(tensors):
    a = tensors["a"]
    b = tensors["b"]
    tensors["out"][:] = (a.float() @ b.float()).float()


def golden_matmul_4d(tensors):
    a = tensors["a"]  # [K_BLOCKS, 1, BATCH, K_CHUNK]
    b = tensors["b"]  # [K_BLOCKS, N1_BLOCKS, K_CHUNK, N1_CHUNK]
    out = tensors["out"]
    for n1 in range(HIDDEN_N1_BLOCKS):
        acc = torch.zeros(BATCH_4D, N1_CHUNK, dtype=torch.float32)
        for kb in range(HIDDEN_K_BLOCKS):
            acc += a[kb, 0, :, :].float() @ b[kb, n1, :, :].float()
        out[n1, 0, :, :] = acc


if __name__ == "__main__":
    import argparse
    from golden import run

    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", default=False,
                        help="compile-only (no device)")
    parser.add_argument("--variant", type=str, default="2d", choices=["2d", "4d", "all"])
    args = parser.parse_args()

    if args.variant in ("2d", "all"):
        prog = TestMatmulTiled
        specs = build_tensor_specs()
        if args.smoke:
            result = run(program=prog, specs=specs, compile_cfg=dict(dump_passes=True), compile_only=True)
            if not result.passed:
                print(f"FAIL (2D): {result.error}")
                raise SystemExit(1)
            print("PASS (2D matmul_tiled)")
        else:
            result = run(program=prog, specs=specs, golden_fn=golden_matmul_2d,
                         compile_cfg=dict(dump_passes=True),
                         runtime_cfg=dict(platform=args.platform, device_id=args.device))
            if not result.passed:
                print(f"FAIL (2D): {result.error}")
                raise SystemExit(1)
            print("PASS (2D matmul_tiled golden)")

    if args.variant in ("4d", "all"):
        prog = TestMatmulTiled4D
        specs = build_tensor_specs_4d()
        if args.smoke:
            result = run(program=prog, specs=specs, compile_cfg=dict(dump_passes=True), compile_only=True)
            if not result.passed:
                print(f"FAIL (4D): {result.error}")
                raise SystemExit(1)
            print("PASS (4D matmul_tiled_4d)")
        else:
            result = run(program=prog, specs=specs, golden_fn=golden_matmul_4d,
                         compile_cfg=dict(dump_passes=True),
                         runtime_cfg=dict(platform=args.platform, device_id=args.device))
            if not result.passed:
                print(f"FAIL (4D): {result.error}")
                raise SystemExit(1)
            print("PASS (4D matmul_tiled_4d golden)")
