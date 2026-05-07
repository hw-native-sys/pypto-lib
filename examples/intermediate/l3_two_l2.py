# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""L3 dispatches two L2 — minimal dispatch / liveness smoke-test.

Verifies that a HOST Orchestrator (L3) can call two Orchestration (L2)
functions sequentially in the same iteration — the structural pattern used
in examples/models/qwen3/14b/qwen3_14b_gen_chunked.py — without deadlocking.

Architecture
------------
L1 tile_scale  : InCore, pl.load + pl.mul + pl.store on full [rows, cols] matrix
L1 tile_add    : InCore, pl.load + pl.add + pl.store on full [rows, cols] matrix
L2 func_scale  : Orchestration, calls tile_scale once (single InCore dispatch)
L2 func_add    : Orchestration, calls tile_add once   (single InCore dispatch)

Using the proven InCore+Orchestration pattern from test_l3_unroll_spike.py
(FourDispatchPlRangePureLoop, InPlaceLayerLoopSpike, etc.) to avoid triggering
the simulator 3rd-core deadlock caused by pl.parallel(0, 64, 16, chunk=1).

L3 host_orch executes ``num_steps`` chained applications per stream using
**in-place pl.unroll** (flat sequential dispatch, equivalent to handwritten
form — FourDispatchPureLoop / FourDispatchExternalScratch PASS patterns):

    step 0:  out = func(input, val, out)        (reads input, writes out)
    pl.unroll(num_steps - 1):
             out = func(out, val, out)           (in-place; unrolled at compile time)

WHY NOT pl.range FOR THE LOOP PORTION:
    ``pl.range`` generates a runtime Python ``for`` loop in host_orch.py.
    When combined with a pre-loop dispatch AND an unused loop variable (``_``),
    all loop-iteration TaskArgs are bit-for-bit identical (same tensor handles,
    same scalar = int(0) from int(float_add_val)).  The device scheduler cannot
    determine WAW order between iterations → deadlock.
    ``L3StackedMatmulFlat`` avoids this because the loop variable IS used as
    a scalar (changes 0→1), making iterations distinguishable.

    ``pl.unroll(N)`` compiles to N sequential flat dispatch calls (no runtime
    for loop in host_orch.py), matching the known-good handwritten pattern.

WHY NOT PING-PONG WITH pl.range:
    ``pl.range`` + ``if step % 2 == 0`` inside a HOST orchestrator triggers
    §2.3 of L3_DISTRIBUTED_CAPABILITIES.md: BinaryOp (%) and CompareOp (==)
    have no VisitExpr_ in distributed_codegen.cpp, so the condition is
    emitted as ``if 0:`` (always False). The loop always takes the ``else``
    branch, reads a scratch buffer that was never written, and the runtime
    deadlocks waiting for the missing producer.  ``test_l3_pingpong_spike.py``
    (Plan C) tests this exact pattern and is documented as FAIL in §5.

WHY NOT Python range:
    pypto's parser rejects any ``for`` loop not using a pypto iterator
    (pl.range, pl.parallel, pl.unroll, …).  Python's ``range()`` triggers:
    "For loop must use pl.range(), pl.parallel(), pl.unroll(), …"

WHY IN-PLACE IS SAFE (has_flag=True only):
    func_scale uses a single InCore dispatch per step.  The in-place
    ``out = func(out, val_t, out)`` pattern is identical across loop
    iterations; the scheduler resolves WAW ordering via ``block_dim=3``
    in DistributedConfig.  ``has_flag=False`` avoids this constraint
    entirely by using the ping-pong pattern (see below).

WHY PING-PONG FOR has_flag=False:
    In-place ``out = func(out, delta_t, out)`` makes dispatch 1 and
    dispatch 2 bit-for-bit identical (same tensor handles).  With
    ``block_dim=1`` (golden.run default) the scheduler cannot determine
    WAW order → deadlock.  Ping-pong alternates between ``buf`` and
    ``out_b`` as the write target, giving each dispatch a unique
    (input, output) handle pair.  Consecutive dispatches always differ,
    and non-consecutive dispatches share handles only after a full
    buf↔out_b cycle — the RAW chain (D_n writes X, D_{n+1} reads X)
    remains unambiguous so the scheduler can always order them.
    ``step % 2`` inside ``pl.unroll`` is evaluated at Python build time
    (compile-time constant), so no runtime BinaryOp/CompareOp is emitted
    and the §2.3 distributed_codegen limitation does not apply.

WHY TENSOR INSTEAD OF SCALAR FOR add_val / scale_factor:
    The distributed codegen emits ``add_scalar(int(fp32_value))`` for all
    FP32 scalars, truncating 0.1 → 0.  Passing the constant as a pre-filled
    [rows, cols] tensor (TensorSpec with init_value=full(add_val)) avoids
    this scalar-shipping bug and aligns with the tensor-only L2 call pattern
    used in all passing spike tests.

The script builds two distinct programs depending on ``has_flag``:

* ``has_flag=True``  — both streams (scale + add) with a runtime
  ``has_flag: pl.BOOL`` scalar held permanently True. Mirrors gen_chunked's
  runtime conditional shape.  Uses in-place ``pl.unroll`` (requires
  ``block_dim=3`` DistributedConfig to resolve identical TaskArgs).

* ``has_flag=False`` — single stream (add only), no ``out_a``,
  ``scale_factor``, or ``has_flag`` scalar.  Uses **ping-pong** between
  ``buf`` and ``out_b`` with ``pl.unroll``; ``step % 2`` is a
  compile-time Python constant so no runtime branch is emitted.
  Every dispatch has distinct tensor handles → scheduler sees an
  unambiguous RAW chain regardless of ``block_dim``.

``num_steps`` is the per-stream chain length and must be >= 3.
(The >= 3 guard keeps compatibility with the original test design intent;
 technically num_steps >= 2 is sufficient for pl.unroll(num_steps - 1).)

Mirrors the interleaved prefill + decode pattern in gen_chunked:
  - func_scale  ≡ qwen3_prefill_layer
  - func_add    ≡ qwen3_decode_layer
  - has_flag    ≡ has_prefill
  - num_steps   ≡ chunk_size

Golden (has_flag=True):
  out_a[r, c] = input_a[r, c] * scale_factor ^ num_steps
  out_b[r, c] = input_b[r, c] + add_val * num_steps

Golden (has_flag=False):
  out_b[r, c] = input_b[r, c] + add_val * num_steps
"""

# pyright: reportUndefinedVariable=false

import pypto.language as pl

ROWS = 64
COLS = 32
ROW_TILE = 16
NUM_STEPS = 3      # mirrors chunk_size in gen_chunked


def build_l3_two_l2_program(
    rows: int = ROWS,
    cols: int = COLS,
    row_tile: int = ROW_TILE,
    num_steps: int = NUM_STEPS,
    has_flag: bool = True,
):
    """Build the smoke-test program.

    ``has_flag`` is a Python build-time switch that selects between the
    full two-stream mirror of gen_chunked and a single-stream variant.
    ``has_flag=True`` uses in-place ``pl.unroll`` loops (requires
    ``block_dim=3`` DistributedConfig).  ``has_flag=False`` uses ping-pong
    with a ``buf`` scratch tensor so all dispatches have distinct handles.
    ``num_steps`` must be >= 3.
    """
    if num_steps < 3:
        raise ValueError(
            f"num_steps must be >= 3 (got {num_steps}); pre-loop seed + "
            "pl.unroll(num_steps-1) requires at least 2 unrolled iterations."
        )
    if has_flag:
        return _build_with_flag(rows, cols, row_tile, num_steps)
    return _build_no_flag(rows, cols, row_tile, num_steps)


def _build_with_flag(rows, cols, row_tile, num_steps):
    @pl.program
    class L3TwoL2Program:

        # ── L1: InCore tile kernels (single dispatch per matrix) ─────────────
        # Processes entire [rows, cols] in one shot — matches spike test pattern.
        @pl.function(type=pl.FunctionType.InCore)
        def tile_scale(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            scale_t: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            ta = pl.load(x, [0, 0], [rows, cols])
            tb = pl.load(scale_t, [0, 0], [rows, cols])
            tc = pl.mul(ta, tb)
            out_v = pl.store(tc, [0, 0], out)
            return out_v

        @pl.function(type=pl.FunctionType.InCore)
        def tile_add(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            delta_t: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            ta = pl.load(x, [0, 0], [rows, cols])
            tb = pl.load(delta_t, [0, 0], [rows, cols])
            tc = pl.add(ta, tb)
            out_v = pl.store(tc, [0, 0], out)
            return out_v

        # ── L2: Orchestration wrappers (each calls its InCore kernel once) ───
        @pl.function(type=pl.FunctionType.Orchestration)
        def func_scale(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            scale_t: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            out_v = self.tile_scale(x, scale_t, out)
            return out_v

        @pl.function(type=pl.FunctionType.Orchestration)
        def func_add(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            delta_t: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            out_v = self.tile_add(x, delta_t, out)
            return out_v

        # ── L3: HOST Orchestrator ───────────────────────────────────────────
        # Uses in-place pl.unroll loops (FourDispatchPureLoop pattern, PASS).
        # scale_t / delta_t are pre-filled [rows, cols] tensors (no FP32
        # scalar dispatch — avoids int(float) truncation bug in codegen).
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            input_a: pl.Tensor[[rows, cols], pl.FP32],
            input_b: pl.Tensor[[rows, cols], pl.FP32],
            scale_t: pl.Tensor[[rows, cols], pl.FP32],
            delta_t: pl.Tensor[[rows, cols], pl.FP32],
            out_a: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            out_b: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            # Stream A: out_a = input_a * scale_factor^num_steps
            out_a = self.func_scale(input_a, scale_t, out_a)
            for _ in pl.unroll(num_steps - 1):
                out_a = self.func_scale(out_a, scale_t, out_a)

            # Stream B: out_b = input_b + add_val * num_steps
            out_b = self.func_add(input_b, delta_t, out_b)
            for _ in pl.unroll(num_steps - 1):
                out_b = self.func_add(out_b, delta_t, out_b)

            return out_b

    return L3TwoL2Program


def _build_no_flag(rows, cols, row_tile, num_steps):
    @pl.program
    class L3TwoL2ProgramNoFlag:

        # ── L1: InCore tile kernel (single dispatch per matrix) ──────────────
        @pl.function(type=pl.FunctionType.InCore)
        def tile_add(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            delta_t: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            ta = pl.load(x, [0, 0], [rows, cols])
            tb = pl.load(delta_t, [0, 0], [rows, cols])
            tc = pl.add(ta, tb)
            out_v = pl.store(tc, [0, 0], out)
            return out_v

        # ── L2: Orchestration wrapper ─────────────────────────────────────────
        @pl.function(type=pl.FunctionType.Orchestration)
        def func_add(
            self,
            x: pl.Tensor[[rows, cols], pl.FP32],
            delta_t: pl.Tensor[[rows, cols], pl.FP32],
            out: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            out_v = self.tile_add(x, delta_t, out)
            return out_v

        # ── L3: HOST Orchestrator ─────────────────────────────────────────────
        # Ping-pong between buf and out_b so every dispatch has distinct
        # tensor handles.  step % 2 is a Python compile-time constant
        # (pl.unroll fully unrolls at build time), so no runtime branch is
        # emitted.  Seeding direction is chosen by num_steps parity so the
        # final result always lands in out_b:
        #
        #   odd  num_steps: seed→out_b, even step→buf,  odd step→out_b
        #   even num_steps: seed→buf,   even step→out_b, odd step→buf
        #
        # delta_t is a pre-filled [rows, cols] tensor — avoids FP32 scalar
        # dispatch (int(0.1)=0 truncation bug in distributed_codegen.cpp).
        @pl.function(level=pl.Level.HOST, role=pl.Role.Orchestrator)
        def host_orch(
            self,
            input_b: pl.Tensor[[rows, cols], pl.FP32],
            delta_t: pl.Tensor[[rows, cols], pl.FP32],
            buf: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
            out_b: pl.Out[pl.Tensor[[rows, cols], pl.FP32]],
        ) -> pl.Tensor[[rows, cols], pl.FP32]:
            if num_steps % 2 == 1:
                # Odd num_steps (e.g. 3): seed→out_b; last unrolled step→out_b
                out_b = self.func_add(input_b, delta_t, out_b)
                for step in pl.unroll(num_steps - 1):
                    if step % 2 == 0:
                        buf = self.func_add(out_b, delta_t, buf)
                    else:
                        out_b = self.func_add(buf, delta_t, out_b)
            else:
                # Even num_steps (e.g. 4): seed→buf; last unrolled step→out_b
                buf = self.func_add(input_b, delta_t, buf)
                for step in pl.unroll(num_steps - 1):
                    if step % 2 == 0:
                        out_b = self.func_add(buf, delta_t, out_b)
                    else:
                        buf = self.func_add(out_b, delta_t, buf)
            return out_b

    return L3TwoL2ProgramNoFlag


def build_specs(
    rows: int = ROWS,
    cols: int = COLS,
    scale_factor: float = 1.5,
    add_val: float = 0.1,
    has_flag: bool = True,
):
    """Build runtime tensor/scalar specs matching host_orch parameter order.

    scale_factor and add_val are passed as pre-filled [rows, cols] tensors
    (not scalars) to avoid the FP32 scalar shipping bug in distributed_codegen
    (int(0.1) = 0 truncation).

    has_flag=False includes a ``buf`` scratch tensor for the ping-pong pattern
    used in _build_no_flag.  ``buf`` is not validated (is_output=False).
    """
    import torch
    from golden import TensorSpec

    if has_flag:
        return [
            TensorSpec("input_a", [rows, cols], torch.float32, init_value=torch.randn),
            TensorSpec("input_b", [rows, cols], torch.float32, init_value=torch.randn),
            TensorSpec("scale_t", [rows, cols], torch.float32,
                       init_value=scale_factor),
            TensorSpec("delta_t", [rows, cols], torch.float32,
                       init_value=add_val),
            TensorSpec("out_a", [rows, cols], torch.float32, is_output=True),
            TensorSpec("out_b", [rows, cols], torch.float32, is_output=True),
        ]
    return [
        TensorSpec("input_b", [rows, cols], torch.float32, init_value=torch.randn),
        TensorSpec("delta_t", [rows, cols], torch.float32,
                   init_value=add_val),
        TensorSpec("buf", [rows, cols], torch.float32),   # ping-pong scratch
        TensorSpec("out_b", [rows, cols], torch.float32, is_output=True),
    ]


def golden_l3_two_l2(values, num_steps: int = NUM_STEPS, has_flag: bool = True):
    """CPU reference. Validates only the final ``out_a`` / ``out_b``.

    delta_t / scale_t are full [rows, cols] tensors; the golden uses them
    directly for element-wise operations.
    """
    delta_t = values["delta_t"]
    values["out_b"][:] = values["input_b"] + delta_t * num_steps

    if has_flag:
        scale_t = values["scale_t"]
        values["out_a"][:] = values["input_a"] * (scale_t ** num_steps)


if __name__ == "__main__":
    import argparse
    from golden import RunConfig, run
    from pypto.ir.distributed_compiled_program import DistributedConfig

    parser = argparse.ArgumentParser(
        description="Minimal L3-dispatches-two-L2 smoke test."
    )
    parser.add_argument(
        "-p", "--platform", type=str, default="a2a3",
        choices=["a2a3", "a2a3sim", "a5", "a5sim"],
    )
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--num-steps", type=int, default=NUM_STEPS,
        help="Number of loop iterations (chunk_size equivalent). Must be >= 3.",
    )
    parser.add_argument(
        "--scale-factor", type=float, default=1.5,
        help="Multiplication factor applied by func_scale each step.",
    )
    parser.add_argument(
        "--add-val", type=float, default=0.1,
        help="Addend applied by func_add each step.",
    )
    parser.add_argument(
        "--no-flag", dest="has_flag", action="store_false", default=True,
        help="Build the single-stream (add-only) program; out_a is dropped.",
    )
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--runtime-profiling", action="store_true", default=False)
    args = parser.parse_args()

    num_steps = args.num_steps
    has_flag = args.has_flag

    result = run(
        program=build_l3_two_l2_program(num_steps=num_steps, has_flag=has_flag),
        specs=build_specs(
            scale_factor=args.scale_factor,
            add_val=args.add_val,
            has_flag=has_flag,
        ),
        golden_fn=lambda v: golden_l3_two_l2(v, num_steps=num_steps, has_flag=has_flag),
        config=RunConfig(
            rtol=1e-3,
            atol=1e-3,
            compile_only=args.compile_only,
            compile=dict(
                dump_passes=True,
                # Match the passing spike-test DistributedConfig exactly.
                # block_dim=3 ensures all 3 AICore threads receive a block and
                # complete their startup handshake; block_dim=1 (default) leaves
                # Core 2 with no work and it never handshakes → deadlock.
                distributed_config=DistributedConfig(
                    num_sub_workers=0,
                    block_dim=3,
                    aicpu_thread_num=4,
                ),
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
