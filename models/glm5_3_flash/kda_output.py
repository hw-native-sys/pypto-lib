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
the layer's TP16 all-reduce before mHC folds it back into the residual stream. Like
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

from models.glm5_3_flash.config import D, FLASH, KDA_DIM, LOCAL_KDA_H, LOCAL_KDA_QKV_DIM, T_DYN
from models.glm5_3_flash.golden import rms_norm_gated

EPS = FLASH.rms_norm_eps
KDA_DIM_INV = 1.0 / KDA_DIM

# tiling
T_TILE = 16  # tokens per task; the norm pass handles T_TILE * LOCAL_KDA_H rows
D_TILE = 128  # output columns per projection task: a [128, 512] BF16 weight tile is
# 128 KiB, and the pipeline double-buffers it inside L1's usable 448 KiB


def golden_kda_output(
    core_attn_out: torch.Tensor,
    norm_weight: torch.Tensor,
    out_gate: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    normalized = rms_norm_gated(core_attn_out, norm_weight, out_gate)
    flattened = normalized.reshape(*normalized.shape[:-2], -1)
    # The kernel emits an FP32 partial for attention_tp to reduce, so the golden
    # accumulates in FP32 rather than returning the BF16 input dtype.
    return torch.nn.functional.linear(flattened.float(), w_o.float())


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
    t_pad = tiles * T_TILE

    # One row per head, so the RMS reduction is a row sum over KDA_DIM.
    core_rows = pl.reshape(core_attn_out, [t_dim * LOCAL_KDA_H, KDA_DIM])
    gate_rows = pl.reshape(out_gate, [t_dim * LOCAL_KDA_H, KDA_DIM])
    weight_row = pl.reshape(norm_weight, [1, KDA_DIM])

    # Parked in GM at the padded token count so the projection can re-read it as
    # [T, LOCAL_KDA_QKV_DIM] without reshaping a tile on chip.
    gated_rows = pl.create_tensor([t_pad * LOCAL_KDA_H, KDA_DIM], dtype=pl.BF16)

    ROW_TILE = T_TILE * LOCAL_KDA_H
    with pl.spmd(tiles, name_hint="kda_output_norm"):
        task = pl.tile.get_block_idx()
        r0 = task * ROW_TILE
        rows = pl.min(ROW_TILE, t_dim * LOCAL_KDA_H - r0)
        value = pl.cast(
            pl.slice(core_rows, [ROW_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM]),
            pl.FP32,
        )
        # Strict FP32: the gamma is not downcast, and eps is rms_norm_eps.
        inv_rms = pl.rsqrt(
            pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), KDA_DIM_INV), EPS),
            high_precision=True,
        )
        normed = pl.col_expand_mul(
            pl.row_expand_mul(value, inv_rms), pl.cast(weight_row[0:1, 0:KDA_DIM], pl.FP32)
        )
        gate = pl.cast(
            pl.slice(gate_rows, [ROW_TILE, KDA_DIM], [r0, 0], valid_shape=[rows, KDA_DIM]),
            pl.FP32,
        )
        # sigmoid, spelled out: the traced body takes only pl.* calls.
        activated = pl.mul(normed, pl.recip(pl.add(pl.exp(pl.neg(gate)), 1.0)))
        gated_rows[r0 : r0 + ROW_TILE, 0:KDA_DIM] = pl.cast(activated, pl.BF16, mode="rint")

    # The padded tail rows are never written, so zero them: the projection reads whole
    # tiles and a recycled scratch row would otherwise reach the cube as garbage.
    if t_pad * LOCAL_KDA_H > t_dim * LOCAL_KDA_H:
        with pl.spmd(1, name_hint="kda_output_pad"):
            pad0 = pl.tile.get_block_idx() * ROW_TILE + t_dim * LOCAL_KDA_H
            gated_rows[pad0 : pad0 + ROW_TILE, 0:KDA_DIM] = pl.full(
                [ROW_TILE, KDA_DIM], dtype=pl.BF16, value=0.0
            )

    gated = pl.reshape(gated_rows, [t_pad, LOCAL_KDA_QKV_DIM])

    with pl.spmd(tiles * (D // D_TILE), name_hint="kda_output_proj"):
        task = pl.tile.get_block_idx()
        t0 = (task // (D // D_TILE)) * T_TILE
        d0 = (task % (D // D_TILE)) * D_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        # K is only LOCAL_KDA_QKV_DIM wide, so the whole contraction is one matmul and
        # no accumulator has to be seeded compact.
        partial = pl.matmul(
            gated[t0 : t0 + T_TILE, 0:LOCAL_KDA_QKV_DIM],
            w_o[d0 : d0 + D_TILE, 0:LOCAL_KDA_QKV_DIM],
            pl.FP32,
            b_trans=True,
        )
        output[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(partial, rows, D_TILE)

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
    parser.add_argument("-p", "--platform", type=str, default="a2a3",
                        choices=["a2a3", "a2a3sim", "a5", "a5sim"])
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
        config=dict(platform=args.platform, device_id=args.device),
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
