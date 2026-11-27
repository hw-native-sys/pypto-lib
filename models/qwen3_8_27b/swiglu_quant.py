# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The MLP's SwiGLU activation and the int8 hand-off to `down_proj`:

    a = silu(gate) * up ;  act_q[t] = rint(a * 127 / amax_t) ,  act_scale[t] = amax_t / 127

One token at a time, the intermediate row viewed as `[ROW_TILE, COL_TILE]`: a
`[1, 1]` reduction result is a 4-byte tile ptoas will not allocate.
"""
import pypto.language as pl

from config import GDN_QUANT, QWEN3_8_27B

# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")                 # tokens

# model config
INTER = QWEN3_8_27B.intermediate_size       # the axis the quantiser reduces over

# quantization
SCALE_MAX = GDN_QUANT.scale_max
AMAX_EPS = GDN_QUANT.amax_eps

# gate
MIN_EXACT_VS_GOLDEN = 0.95      # of act_q elements identical to the float64 reference's quantisation
# validation
DEFAULT_CASES = (16, 32)
BENCH_SEQ = 8192        # prefill tokens, selected explicitly with --seq-len

# tiling
TOK_TILE = 16           # tokens per block; 16 scalar scale writes fill one 64-byte line
COL_TILE = 128          # the intermediate row viewed as (INTER // COL_TILE) x COL_TILE


ROW_TILE = INTER // COL_TILE            # the activation row's other axis


def _qwen3_swiglu_quant(
    gate: pl.Tensor[[T_DYN, INTER], pl.BF16],
    up: pl.Tensor[[T_DYN, INTER], pl.BF16],
    act_q: pl.Out[pl.Tensor[[T_DYN, INTER], pl.INT8]],
    act_scale: pl.Out[pl.Tensor[[1, T_DYN], pl.FP32]],
):
    """The token count must be a multiple of TOK_TILE; a block walks one whole tile."""
    gate.bind_dynamic(0, T_DYN)
    up.bind_dynamic(0, T_DYN)
    act_q.bind_dynamic(0, T_DYN)
    act_scale.bind_dynamic(1, T_DYN)
    t_dim = pl.tensor.dim(gate, 0)
    for blk in pl.spmd(t_dim // TOK_TILE, name_hint="swiglu_quant"):
        t0 = blk * TOK_TILE
        ones_col = pl.full([ROW_TILE, 8], dtype=pl.FP32, value=1.0)
        for i in pl.pipeline(TOK_TILE, stage=1):
            r0 = t0 + i
            gate_run = pl.cast(gate[r0 : r0 + 1, :], target_type=pl.FP32)
            gate_tile = pl.reshape(gate_run, [ROW_TILE, COL_TILE])
            # silu as a divide: the reciprocal would be used once
            gate_act = pl.div(gate_tile, pl.add(pl.exp(pl.neg(gate_tile)), 1.0))
            up_run = pl.cast(up[r0 : r0 + 1, :], target_type=pl.FP32)
            up_tile = pl.reshape(up_run, [ROW_TILE, COL_TILE])
            act = pl.mul(gate_act, up_tile)

            # fold the token's maximum to [1, 8], clamp it there, then
            # broadcast down the row axis so row_max lands it in every row
            col_amax = pl.reshape(pl.col_max(pl.abs(act)), [COL_TILE // 8, 8])
            part_8 = pl.maximum(pl.col_max(col_amax), pl.full([1, 8], dtype=pl.FP32, value=AMAX_EPS))
            amax = pl.row_max(pl.col_expand_mul(ones_col, part_8))
            pl.write(act_scale, [0, r0], pl.read(amax, [0, 0]) / SCALE_MAX)

            # no clamp: amax is the row's own maximum, so |a| * SCALE_MAX / amax <= 127
            q = pl.row_expand_mul(act, pl.mul(pl.recip(amax, high_precision=True), SCALE_MAX))
            # INT32 then INT8, not FP32 straight to INT8: the direct cast rounds through
            # FP16, whose spacing at 127 is a sixteenth of an INT8 step, and that moves
            # 0.9% of the output by one step. `quant_x` carries the measurement.
            q32 = pl.cast(q, target_type=pl.INT32, mode="rint")
            q_i8 = pl.cast(q32, target_type=pl.INT8, mode="trunc")
            act_q[r0 : r0 + 1, :] = pl.reshape(q_i8, [1, INTER])
    return act_q


qwen3_swiglu_quant = pl.jit.inline(_qwen3_swiglu_quant)
qwen3_swiglu_quant_test = pl.jit(_qwen3_swiglu_quant)



def build_tensor_specs(t: int = BENCH_SEQ, inter: int = INTER):
    import torch
    from golden import TensorSpec

    def init_gate():
        gen = torch.Generator().manual_seed(42)
        return torch.randn(t, inter, generator=gen).to(torch.bfloat16)

    def init_up():
        gen = torch.Generator().manual_seed(43)
        return torch.randn(t, inter, generator=gen).to(torch.bfloat16)

    return [
        TensorSpec("gate", [t, inter], torch.bfloat16, init_value=init_gate),
        TensorSpec("up", [t, inter], torch.bfloat16, init_value=init_up),
        TensorSpec("act_q", [t, inter], torch.int8),
        TensorSpec("act_scale", [1, t], torch.float32),
    ]


def golden_qwen3_swiglu_quant(tensors):
    import reference

    act = reference.swiglu_act(tensors["gate"], tensors["up"])
    act_q, scale = reference.quantize_rows(act)
    tensors["act_q"].copy_(act_q)
    tensors["act_scale"].copy_(scale.reshape(1, -1))


if __name__ == "__main__":
    import argparse

    from golden import run

    from utils import int8_step_and_exact

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="one shape instead of DEFAULT_CASES")
    parser.add_argument("--col-tile", type=int, default=COL_TILE)
    parser.add_argument("--save-data", action="store_true", default=False)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--compile-only", action="store_true", default=False)
    args = parser.parse_args()
    cases = [args.seq_len] if args.seq_len else list(DEFAULT_CASES)
    if len(cases) > 1 and (args.golden_data or args.runtime_dir):
        parser.error("--golden-data and --runtime-dir carry one shape; name it with --seq-len")
    if args.seq_len and args.seq_len % TOK_TILE:
        parser.error(f"--seq-len must be a multiple of TOK_TILE={TOK_TILE}")

    for t in cases:
        print(f"[case] swiglu_quant T={t}", flush=True)
        result = run(
            fn=qwen3_swiglu_quant_test,
            specs=build_tensor_specs(t=t),
            golden_fn=golden_qwen3_swiglu_quant,
            golden_data=args.golden_data,
            runtime_dir=args.runtime_dir,
            save_data=args.save_data,
            compare_fn={"act_q": int8_step_and_exact(MIN_EXACT_VS_GOLDEN)},
            config=dict(
                backend_type=args.backend_type,
                execution_mode=args.execution_mode,
                device_id=args.device,
            ),
            rtol=1e-2,
            atol=1e-5,
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(f"FAILED at T={t}")
