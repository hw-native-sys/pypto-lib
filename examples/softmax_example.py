# Copyright (c) PyPTO Contributors.
from __future__ import annotations

import os
import sys
from typing import Optional

# Use pypto from pypto3.0 instead of the one in /data/z00885570/pypto
pypto_path = '/data/z00885570/pypto3.0/pypto/python'
if pypto_path not in sys.path:
    sys.path.insert(0, pypto_path)

import pypto.language as pl

BATCH = 16
SEQ_LEN = 128
HIDDEN = 256
HIDDEN_CHUNK = 32


def build_softmax_program(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    hidden: int = HIDDEN,
):
    BATCH_CFG = batch
    SEQ_LEN_CFG = seq_len
    HIDDEN_CFG = hidden
    HIDDEN_BLOCKS = (HIDDEN_CFG + HIDDEN_CHUNK - 1) // HIDDEN_CHUNK

    @pl.program
    class SoftmaxExample:
        @pl.function(type=pl.FunctionType.Opaque)
        def softmax(
            self,
            input_tensor: pl.Tensor[[BATCH_CFG, SEQ_LEN_CFG, HIDDEN_CFG], pl.FP32],
            output_tensor: pl.Tensor[[BATCH_CFG, SEQ_LEN_CFG, HIDDEN_CFG], pl.FP32],
        ) -> pl.Tensor[[BATCH_CFG, SEQ_LEN_CFG, HIDDEN_CFG], pl.FP32]:
            with pl.auto_incore():
                max_vals = pl.create_tensor([BATCH_CFG, SEQ_LEN_CFG, 1], dtype=pl.FP32)
                max_vals = pl.mul(max_vals, -3.402823e38)
                exp_sum = pl.create_tensor([BATCH_CFG, SEQ_LEN_CFG, 1], dtype=pl.FP32)
                exp_sum = pl.mul(exp_sum, 0.0)

                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    for s in pl.range(SEQ_LEN_CFG):
                        row_max = pl.create_tensor([1, 1, 1], dtype=pl.FP32)
                        row_max = pl.mul(row_max, -3.402823e38)
                        
                        for hb in pl.range(HIDDEN_BLOCKS):
                            h0 = hb * HIDDEN_CHUNK
                            chunk = pl.slice(input_tensor, [1, 1, HIDDEN_CHUNK], [b, s, h0])
                            chunk_max = pl.reduce_max(chunk, axis=2)
                            row_max = pl.maximum(row_max, chunk_max)
                        
                        max_vals = pl.assemble(max_vals, row_max, [b, s, 0])
                        
                        row_sum = pl.create_tensor([1, 1, 1], dtype=pl.FP32)
                        row_sum = pl.mul(row_sum, 0.0)
                        
                        for hb in pl.range(HIDDEN_BLOCKS):
                            h0 = hb * HIDDEN_CHUNK
                            chunk = pl.slice(input_tensor, [1, 1, HIDDEN_CHUNK], [b, s, h0])
                            chunk_sub = pl.sub(chunk, row_max)
                            chunk_exp = pl.exp(chunk_sub)
                            chunk_sum = pl.reduce_sum(chunk_exp, axis=2)
                            row_sum = pl.add(row_sum, chunk_sum)
                        
                        exp_sum = pl.assemble(exp_sum, row_sum, [b, s, 0])

                for b in pl.parallel(0, BATCH_CFG, 1, chunk=4):
                    for s in pl.range(SEQ_LEN_CFG):
                        max_val = pl.slice(max_vals, [1, 1, 1], [b, s, 0])
                        sum_val = pl.slice(exp_sum, [1, 1, 1], [b, s, 0])
                        
                        for hb in pl.parallel(0, HIDDEN_BLOCKS, 1, chunk=4):
                            h0 = hb * HIDDEN_CHUNK
                            chunk = pl.slice(input_tensor, [1, 1, HIDDEN_CHUNK], [b, s, h0])
                            chunk_sub = pl.sub(chunk, max_val)
                            chunk_exp = pl.exp(chunk_sub)
                            chunk_out = pl.div(chunk_exp, sum_val)
                            output_tensor = pl.assemble(output_tensor, chunk_out, [b, s, h0])

            return output_tensor

    return SoftmaxExample


def build_tensor_specs(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    hidden: int = HIDDEN,
):
    import torch
    from pypto.runtime import TensorSpec

    return [
        TensorSpec("input_tensor", [batch, seq_len, hidden], torch.float32, init_value=torch.randn),
        TensorSpec("output_tensor", [batch, seq_len, hidden], torch.float32, is_output=True),
    ]


def compile_and_run(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    hidden: int = HIDDEN,
    platform: str = "a2a3",
    device_id: int = 11,
    work_dir: Optional[str] = None,
    dump_passes: bool = True,
):
    from pypto.backend import BackendType
    from pypto.ir.pass_manager import OptimizationStrategy
    from pypto.runtime import RunConfig, run

    program = build_softmax_program(
        batch=batch,
        seq_len=seq_len,
        hidden=hidden,
    )
    tensor_specs = build_tensor_specs(
        batch=batch,
        seq_len=seq_len,
        hidden=hidden,
    )

    if work_dir is None:
        work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "softmax_dump"))

    result = run(
        program=program,
        tensor_specs=tensor_specs,
        golden=None,
        config=RunConfig(
            platform=platform,
            device_id=device_id,
            rtol=2e-2,
            atol=2e-2,
            strategy=OptimizationStrategy.Default,
            dump_passes=dump_passes,
            backend_type=BackendType.CCE,
            work_dir=work_dir,
        ),
    )
    if not result.passed and result.error and "code_runner" in result.error:
        print("Result: COMPILE OK — device run skipped (code_runner not found).")
        print("  Generated kernels/orchestration:", work_dir)
        return result
    if not result.passed and result.error:
        print(f"Result: {result.error}")
        print("  Pass dumps may still have been written to:", work_dir)
    else:
        print("  Generated kernels/orchestration:", work_dir)
    return result


if __name__ == "__main__":
    compile_and_run()
