# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The MLA output projection and its TP16 reduction.

Prefill projects ``[T, LOCAL_H * V_DIM]`` through the plain ``o_proj``; decode
projects ``[T, LOCAL_H * KV_LORA]`` through the value-absorbed ``o_proj`` produced
by :mod:`models.glm5_3_flash.mla_prolog`. Both are row-parallel over heads, so the
result is a partial sum that needs an all-reduce across the 16 ranks before mHC
folds it back into the residual stream.

Both paths consume BF16 weights. When the checkpoint stores ``o_proj`` as W8A8
INT8, the loader dequantizes it with its per-output-channel scales before slicing
and absorption are complete. Neither epilog quantizes activations at runtime.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KV_LORA, LOCAL_H, T_DYN, V_DIM
from models.glm5_3_flash.linear import bf16_linear, golden_bf16_linear

MM_T_TILE = 16          # cube M tile; rows are a 16-row boxed tile
O_N_TILE = 256          # o_proj output-column tile over D


def golden_mla_epilog_prefill(
    attn_out: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """Head-space projection with the loader's prepared BF16 ``o_proj``."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return golden_bf16_linear(flattened, w_o)


def golden_mla_epilog_decode(
    attn_out: torch.Tensor,
    w_o_absorbed: torch.Tensor,
) -> torch.Tensor:
    """The absorbed path: ``kv_b_proj``'s value half is folded in, so BF16."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return golden_bf16_linear(flattened, w_o_absorbed)


@pl.jit.inline
def mla_epilog_prefill(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """Project head-space attention output, one block per output-column tile.

    ``w_o`` is stored ``[D, LOCAL_H * V_DIM]``, reduction dim last, so the matmul
    takes ``b_trans=True`` and each weight fragment loads K-contiguous.

    The result stays FP32: this rank owns ``LOCAL_H`` of the 64 heads, so the row is
    a partial sum that :mod:`models.glm5_3_flash.attention_tp` all-reduces across the
    16 ranks before mHC folds it into the residual stream. Rounding to BF16 here
    would round 16 times instead of once.

    Returns the partial sum so the caller can chain it.
    """
    t_dim = pl.tensor.dim(attn_out, 0)
    t_mm = ((t_dim + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
    attn_flat = pl.reshape(attn_out, [t_dim, LOCAL_H * V_DIM])
    for o_idx in pl.spmd(D // O_N_TILE, name_hint="mla_epilog_prefill"):
        n0 = o_idx * O_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = bf16_linear(attn_flat, w_o, t0, n0, valid_rows,
                              pl.cast(O_N_TILE, pl.INDEX), MM_T_TILE, O_N_TILE, LOCAL_H * V_DIM, True)
            output = pl.assemble(output, pl.set_validshape(acc, valid_rows, O_N_TILE), [t0, n0])
    return output


@pl.jit.inline
def mla_epilog_decode(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    w_o_absorbed: pl.Tensor[[D, LOCAL_H * KV_LORA], pl.BF16],
    output: pl.Tensor[[T_DYN, D], pl.FP32],
):
    """Project latent-space attention output through the value-absorbed ``o_proj``.

    Structurally the prefill epilog with a wider reduction: the context rows are
    ``KV_LORA`` = 512 instead of ``V_DIM`` = 256, because
    :func:`models.glm5_3_flash.mla_prolog.golden_absorb_output` folded ``kv_b_proj``'s
    value half into the weight, so the value expansion never happens per token. Same
    FP32 partial-sum contract.

    Returns the partial sum so the caller can chain it.
    """
    t_dim = pl.tensor.dim(attn_out, 0)
    t_mm = ((t_dim + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE
    attn_flat = pl.reshape(attn_out, [t_dim, LOCAL_H * KV_LORA])
    for o_idx in pl.spmd(D // O_N_TILE, name_hint="mla_epilog_decode"):
        n0 = o_idx * O_N_TILE
        for tc in pl.range(t_mm // MM_T_TILE):
            t0 = tc * MM_T_TILE
            valid_rows = pl.min(MM_T_TILE, t_dim - t0)
            acc = bf16_linear(attn_flat, w_o_absorbed, t0, n0, valid_rows,
                              pl.cast(O_N_TILE, pl.INDEX), MM_T_TILE, O_N_TILE, LOCAL_H * KV_LORA, True)
            output = pl.assemble(output, pl.set_validshape(acc, valid_rows, O_N_TILE), [t0, n0])
    return output


@pl.jit
def mla_epilog_prefill_test(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, V_DIM], pl.BF16],
    w_o: pl.Tensor[[D, LOCAL_H * V_DIM], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    """Run one head-space epilog for golden.run validation."""
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mla_epilog_prefill(attn_out, w_o, output)
    return output


@pl.jit
def mla_epilog_decode_test(
    attn_out: pl.Tensor[[T_DYN, LOCAL_H, KV_LORA], pl.BF16],
    w_o_absorbed: pl.Tensor[[D, LOCAL_H * KV_LORA], pl.BF16],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.FP32]],
):
    """Run one latent-space epilog for golden.run validation."""
    attn_out.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mla_epilog_decode(attn_out, w_o_absorbed, output)
    return output


def _epilog_specs(tokens: int, width: int, weight_name: str):
    """Build one deterministic epilog case at a given head-space width."""
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(73)

    def init_attn():
        return torch.randn(tokens, LOCAL_H, width, generator=generator).bfloat16()

    def init_weight():
        return (torch.randn(D, LOCAL_H * width, generator=generator) * 0.02).bfloat16()

    return [
        TensorSpec("attn_out", [tokens, LOCAL_H, width], torch.bfloat16, init_value=init_attn),
        TensorSpec(weight_name, [D, LOCAL_H * width], torch.bfloat16, init_value=init_weight),
        TensorSpec("output", [tokens, D], torch.float32),
    ]


def build_mla_epilog_prefill_specs(tokens: int = 20):
    """Head-space epilog: the reduction is ``LOCAL_H * V_DIM``."""
    return _epilog_specs(tokens, V_DIM, "w_o")


def build_mla_epilog_decode_specs(tokens: int = 20):
    """Latent-space epilog: the reduction is the wider ``LOCAL_H * KV_LORA``."""
    return _epilog_specs(tokens, KV_LORA, "w_o_absorbed")


def golden_mla_epilog_prefill_case(tensors):
    """Fill the expected partial sum for :func:`build_mla_epilog_prefill_specs`."""
    tensors["output"][:] = golden_mla_epilog_prefill(tensors["attn_out"], tensors["w_o"])


def golden_mla_epilog_decode_case(tensors):
    """Fill the expected partial sum for :func:`build_mla_epilog_decode_specs`."""
    tensors["output"][:] = golden_mla_epilog_decode(
        tensors["attn_out"], tensors["w_o_absorbed"]
    )


def main():
    """Prove the goldens on CPU, then validate both epilogs on device.

    The output is FP32 out of BF16 operands, so only the accumulation order separates it
    from the reference and the budget is correspondingly tight.
    """
    import argparse

    from golden import ratio_allclose, run
    from models.glm5_3_flash._golden_smoke import run_mla_epilog_goldens

    run_mla_epilog_goldens(golden_mla_epilog_prefill, golden_mla_epilog_decode)

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    compare = ratio_allclose(atol=1e-4, rtol=1e-3)
    cases = (
        ("prefill", mla_epilog_prefill_test, build_mla_epilog_prefill_specs(args.tokens),
         golden_mla_epilog_prefill_case),
        ("decode", mla_epilog_decode_test, build_mla_epilog_decode_specs(args.tokens),
         golden_mla_epilog_decode_case),
    )
    for name, fn, specs, golden_fn in cases:
        result = run(
            fn=fn,
            specs=specs,
            golden_fn=golden_fn,
            config={"backend_type": args.backend_type, "execution_mode": args.execution_mode, "device_id": args.device},
            rtol=1e-3,
            atol=1e-4,
            compare_fn={"output": compare},
            compile_only=args.compile_only,
        )
        print(f"{name}: {result}")
        if not result.passed:
            raise SystemExit(result.error or 1)


__all__ = [
    "build_mla_epilog_decode_specs",
    "build_mla_epilog_prefill_specs",
    "golden_mla_epilog_decode",
    "golden_mla_epilog_decode_case",
    "golden_mla_epilog_prefill",
    "golden_mla_epilog_prefill_case",
    "mla_epilog_decode",
    "mla_epilog_decode_test",
    "mla_epilog_prefill",
    "mla_epilog_prefill_test",
]


if __name__ == "__main__":
    main()
