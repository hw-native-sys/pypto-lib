# Copyright (c) PyPTO Contributors.
# Predicate test: mixed kernel with UNSPLITTABLE vector operation.
#
# Contains a [1, Q_TILE] tensor consumed by row_max — the only non-unit axis
# is the reduction axis, so choose_split_axis returns UNSPLITTABLE and the
# operation must be wrapped in `if AIV_IDX == 0:` predication.

import os

import pypto.language as pl
import torch
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy
from pypto.runtime import RunConfig, RunResult, TensorSpec, run

Q_TILE = 16
BLOCK_SIZE = 128
HEAD_DIM = 128
BATCH = 64
BATCH_CHUNK = 8


def build_program():
    query_rows = BATCH
    key_rows = BLOCK_SIZE
    out_rows = BATCH

    @pl.program
    class PredicateTestProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def predicate_kernel(
            self,
            query: pl.Tensor[[query_rows, HEAD_DIM], pl.BF16],
            key: pl.Tensor[[key_rows, HEAD_DIM], pl.BF16],
            value: pl.Tensor[[key_rows, HEAD_DIM], pl.BF16],
            out: pl.Tensor[[out_rows, HEAD_DIM], pl.FP32],
        ) -> pl.Tensor[[out_rows, HEAD_DIM], pl.FP32]:

            with pl.auto_incore():
                for b_idx in pl.parallel(0, BATCH, 1, chunk=BATCH_CHUNK):
                    cur_offset = b_idx * Q_TILE

                    qi: pl.Tensor[[Q_TILE, HEAD_DIM], pl.BF16] = pl.slice(
                        query, [Q_TILE, HEAD_DIM], [cur_offset, 0]
                    )
                    kj: pl.Tensor[[BLOCK_SIZE, HEAD_DIM], pl.BF16] = pl.slice(
                        key, [BLOCK_SIZE, HEAD_DIM], [0, 0]
                    )
                    vj: pl.Tensor[[BLOCK_SIZE, HEAD_DIM], pl.BF16] = pl.slice(
                        value, [BLOCK_SIZE, HEAD_DIM], [0, 0]
                    )

                    # ── RED: Q*K^T matmul (AIC) ──
                    sij = pl.matmul(qi, kj, b_trans=True)  # [16, 128]

                    # ── GREEN: splittable softmax prep ──
                    mi = pl.row_max(sij)  # [16, 1] — splittable on axis 0

                    # ── GREEN: UNSPLITTABLE operation ──
                    # reshape [16, 1] → [1, 16]: axis 0 = 1 (too small)
                    # row_max consumes mi_flat with reduction on axis 1 (forbidden)
                    # → no valid split axis → UNSPLITTABLE → predication!
                    mi_flat = pl.reshape(mi, [1, Q_TILE])  # [1, 16]
                    global_max = pl.row_max(mi_flat)  # [1, 1] — also unsplittable

                    # ── GREEN: continue with splittable ops ──
                    centered = pl.sub(sij, global_max)  # [16, 128] broadcast
                    exp_vals = pl.exp(centered)
                    pij = pl.cast(exp_vals, target_type=pl.BF16)  # [16, 128]

                    # ── RED: P*V matmul (AIC) ──
                    oi = pl.matmul(pij, vj)  # [16, 128]

                    # ── GREEN: write back ──
                    out = pl.assemble(out, oi, [cur_offset, 0])

            return out

    return PredicateTestProgram


def main():
    program = build_program()

    tensor_specs = [
        TensorSpec("query", [BATCH, HEAD_DIM], torch.bfloat16, init_value=torch.randn),
        TensorSpec("key", [BLOCK_SIZE, HEAD_DIM], torch.bfloat16, init_value=torch.randn),
        TensorSpec("value", [BLOCK_SIZE, HEAD_DIM], torch.bfloat16, init_value=torch.randn),
        TensorSpec("out", [BATCH, HEAD_DIM], torch.float32, is_output=True),
    ]

    work_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "pa5_predicate_build")
    )

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=None,
        config=RunConfig(
            platform="a2a3",
            device_id=11,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=True,
            backend_type=BackendType.CCE,
            work_dir=work_dir,
        ),
    )
    print(f"Result: {result}")
    print("\nDone.")


if __name__ == "__main__":
    main()
