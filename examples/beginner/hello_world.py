# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hello World — the simplest PyPTO-Lib example.

Demonstrates the simplest use of auto_incore with a single parallel loop and a
scalar runtime parameter: a large matrix is split into row chunks, and each
chunk adds the same scalar ``a`` elementwise.

    output[r, c] = input[r, c] + a     for all (r, c)

The parallel loop with chunk= lets the compiler split the iteration space
into (chunk_loop, in_chunk_loop) and place the incore boundary automatically.
"""
import pypto.language as pl

ROWS = 1024
COLS = 512
ROW_CHUNK = 128


def build_hello_world_program(
    rows: int = ROWS,
    cols: int = COLS,
    row_chunk: int = ROW_CHUNK,
):
    @pl.program
    class HelloWorldProgram:
        @pl.function(type=pl.FunctionType.Opaque)
        def add_scalar(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            a: pl.Scalar[pl.FP32],
            y: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            with pl.at(level=pl.Level.CORE_GROUP, optimization=pl.chunked_loop_optimizer):
                for r in pl.parallel(0, rows, 1, chunk=row_chunk):
                    tile_x = pl.slice(x, [1, cols], [r, 0])
                    tile_y = pl.add(tile_x, a)
                    y = pl.assemble(y, tile_y, [r, 0])

            return y

    return HelloWorldProgram


def build_specs(
    rows: int = ROWS,
    cols: int = COLS,
    a: float = 1.0,
):
    import torch
    from golden import ScalarSpec, TensorSpec

    return [
        TensorSpec("x", [rows, cols], torch.float32, init_value=torch.randn),
        ScalarSpec("a", torch.float32, a),
        TensorSpec("y", [rows, cols], torch.float32, is_output=True),
    ]


def golden_hello_world(values):
    values["y"][:] = values["x"] + values["a"]


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    result = run(
        program=build_hello_world_program(),
        specs=build_specs(),
        golden_fn=golden_hello_world,
        config=RunConfig(
            rtol=1e-5,
            atol=1e-5,
            compile=dict(
                dump_passes=True,
            ),
            runtime=dict(
                platform=args.platform,
                device_id=args.device,
                runtime_profiling=args.runtime_profiling,
            ),
        ),
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)
