# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The KDA epilogue: the gated output norm and the output projection.

``Glm5NextTextRMSNormGated`` normalises each 128-wide head in **strict FP32** (the
gamma is not downcast) and then multiplies by ``sigmoid(out_gate)`` before the heads
are flattened back to ``[T, KDA_H * KDA_DIM]`` and projected to ``[T, D]``.

``o_proj`` is head-sharded, so the projection is row-parallel and its result needs
the layer's TP all-reduce before mHC folds it back into the residual stream. Like
the rest of the KDA block, the weight stays BF16.

Two numerics notes:

* ``eps`` is ``rms_norm_eps`` = 1e-5, taken from the config. The Triton
  ``rms_norm_gated`` signature upstream defaults to 1e-6, so copying that default
  would be a silent tenfold error that no shape or dtype check catches.
* The gated result rounds to BF16 before ``o_proj``, matching the reference, which
  casts back to the activation dtype inside the norm. Carrying FP32 into the
  projection would be more accurate than the reference *and* four times the cube
  cost, for a difference that does not survive the all-reduce.

The norm runs over a ``[T * LOCAL_KDA_H, KDA_DIM]`` view, so one row is one head and
the reduction is a plain row sum. The gated rows are parked in GM and re-read as
``[T, LOCAL_KDA_QKV_DIM]`` for the projection, which avoids reshaping a tile on chip.
"""

import sys
from pathlib import Path

# Run directly and the script's own directory leads sys.path, where this model's
# ``golden.py`` shadows the repo-root ``golden`` harness package. Put the repo root
# first, exactly as models/deepseek_v4_1_flash does.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import FLASH, KDA_DIM, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM, T_DYN, D
from models.glm5_3_flash.golden import rms_norm_gated
from models.glm5_3_flash.linear import bf16_linear

EPS = FLASH.rms_norm_eps
KDA_DIM_INV = 1.0 / KDA_DIM

# tiling
T_TILE = 16  # tokens per task; the norm pass handles T_TILE * LOCAL_KDA_H rows
NORM_ROWS_TILE = 8
D_TILE = 128  # output columns per projection task


def golden_kda_output(
    core_attn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    out_gate: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    normalized = rms_norm_gated(core_attn_out, norm_weight, out_gate)
    flattened = normalized.reshape(*normalized.shape[:-2], -1)
    # Match the BF16 local linear output before TP reduction. FP32 is only
    # the exchange dtype; it carries the widened BF16 result.
    return (flattened.float() @ w_o.float().t()).to(torch.bfloat16).float()


@pl.jit.inline
def kda_output(
    core_attn_out: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    norm_weight: pl.Tensor[[KDA_DIM], pl.BF16],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    t_dim = pl.tensor.dim(core_attn_out, 0)
    tiles = (t_dim + T_TILE - 1) // T_TILE
    core_rows = pl.reshape(core_attn_out, [t_dim * LOCAL_KDA_H, KDA_DIM])
    gate_rows = pl.reshape(out_gate, [t_dim * LOCAL_KDA_H, KDA_DIM])
    weight_row = pl.reshape(norm_weight, [1, KDA_DIM])
    gated = pl.create_tensor([t_dim, LOCAL_KDA_QKV_DIM], dtype=pl.BF16)
    gated_rows = pl.reshape(gated, [t_dim * LOCAL_KDA_H, KDA_DIM])
    with pl.spmd((t_dim * LOCAL_KDA_H + NORM_ROWS_TILE - 1) // NORM_ROWS_TILE, name_hint='kda_output_norm'):
        task = pl.tile.get_block_idx()
        r0 = task * NORM_ROWS_TILE
        rows = pl.max(0, pl.min(NORM_ROWS_TILE, t_dim * LOCAL_KDA_H - r0))
        value_v1 = pl.slice(core_rows, [NORM_ROWS_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM])
        value = pl.cast(value_v1, pl.FP32)
        square_buffer = pl.create_tensor([NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        square_buffer[:, :] = pl.mul(value, value)
        square_sum = pl.row_sum(square_buffer)
        norm_argument = pl.create_tensor([1, NORM_ROWS_TILE], dtype=pl.FP32)
        part_v3 = pl.mul(square_sum, KDA_DIM_INV)
        part_v4 = pl.add(part_v3, EPS)
        norm_argument[:, :] = pl.reshape(part_v4, [1, NORM_ROWS_TILE])
        inverse = pl.rsqrt(norm_argument, high_precision=True)
        inv_rms = pl.reshape(inverse, [NORM_ROWS_TILE, 1])
        normed_v5 = pl.row_expand_mul(value, inv_rms)
        normed_v6 = pl.cast(weight_row[0:1, 0:KDA_DIM], pl.FP32)
        normed = pl.col_expand_mul(normed_v5, normed_v6)
        gate_v7 = pl.slice(gate_rows, [NORM_ROWS_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM])
        gate = pl.cast(gate_v7, pl.FP32)
        gate_buffer = pl.create_tensor([NORM_ROWS_TILE, KDA_DIM], dtype=pl.FP32)
        gate_buffer[:, :] = gate
        gate_sigmoid = pl.recip(pl.add(pl.exp(pl.neg(gate_buffer)), 1.0))
        activated = pl.mul(normed, gate_sigmoid)
        activated_bf16 = pl.cast(activated, pl.BF16, mode='rint')
        gated_rows[r0:r0 + NORM_ROWS_TILE, 0:KDA_DIM] = pl.set_validshape(activated_bf16, rows, KDA_DIM)
    with pl.spmd(tiles * (D // D_TILE), name_hint='kda_output_proj'):
        task = pl.tile.get_block_idx()
        t0 = task // (D // D_TILE) * T_TILE
        d0 = task % (D // D_TILE) * D_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        partial_v8 = pl.cast(D_TILE, pl.INDEX)
        partial = bf16_linear(gated, w_o, t0, d0, rows, partial_v8, T_TILE, D_TILE, LOCAL_KDA_QKV_DIM)
        local_bf16 = pl.cast(partial, pl.BF16, mode="rint")
        exchange_value = pl.cast(local_bf16, pl.FP32)
        output[t0:t0 + T_TILE, d0:d0 + D_TILE] = pl.set_validshape(exchange_value, rows, D_TILE)
    return output


@pl.jit
def kda_output_test(
    core_attn_out: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    norm_weight: pl.Tensor[[KDA_DIM], pl.BF16],
    out_gate: pl.Tensor[[T_DYN, LOCAL_KDA_H, KDA_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_KDA_QKV_DIM], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    output = kda_output(core_attn_out, norm_weight, out_gate, w_o, output)
    return output


def golden_kda_output_tensors(tensors) -> None:
    """Harness adapter: run the reference and write the output in place."""
    tensors["output"][:] = golden_kda_output(
        tensors["core_attn_out"], tensors["norm_weight"], tensors["out_gate"], tensors["w_o"]
    )



def compare_kda_output(actual, expected, **kwargs):
    """Require a BF16 local result carried through the FP32 exchange buffer."""
    from golden import ratio_allclose

    if not torch.equal(actual, actual.to(torch.bfloat16).float()):
        return False, "local projection was not rounded to BF16 before TP exchange"
    return ratio_allclose(max_error_ratio=0)(actual, expected, **kwargs)


def build_tensor_specs(tokens: int = 67):
    """A deliberately ragged token count, so the tail block is always exercised."""
    from golden import TensorSpec

    bf, f32 = torch.bfloat16, torch.float32
    del f32

    return [
        TensorSpec("core_attn_out", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 0.7).to(bf)),
        # Not all-ones: a unit gamma hides a dropped or mis-broadcast norm weight.
        TensorSpec("norm_weight", [KDA_DIM], bf,
                   init_value=lambda: (1.0 + torch.randn(KDA_DIM) * 0.3).to(bf)),
        # Wide enough to span both sigmoid tails.
        TensorSpec("out_gate", [tokens, LOCAL_KDA_H, KDA_DIM], bf,
                   init_value=lambda: (torch.randn(tokens, LOCAL_KDA_H, KDA_DIM) * 3.0).to(bf)),
        TensorSpec("w_o", [D, LOCAL_KDA_QKV_DIM], bf,
                   init_value=lambda: (torch.randn(D, LOCAL_KDA_QKV_DIM) * 0.02).to(bf)),
        TensorSpec("output", [tokens, D], torch.float32),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", type=str, default="ragged",
                        choices=["ragged", "aligned", "single"])
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    tokens = {"ragged": 67, "aligned": 64, "single": 1}[args.case]
    result = run(
        fn=kda_output_test,
        specs=build_tensor_specs(tokens),
        golden_fn=golden_kda_output_tensors,
        compare_fn={"output": compare_kda_output},
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


__all__ = [
    "golden_kda_output",
    "kda_output",
]
