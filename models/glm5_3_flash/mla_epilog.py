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
by :mod:`models.glm5_3_flash.mla_absorb`. Both are row-parallel over heads, so the
result is a partial sum that needs an all-reduce across the 16 ranks before mHC
folds it back into the residual stream.

``o_proj`` carries a ``weight_scale_inv`` in the released FP8 checkpoint but is
**BF16** in the deployment W8A8 one (``[4096, 8192]``, marked ``FLOAT`` in
``quant_model_description.json``), so neither path quantizes here.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

from models.glm5_3_flash.config import D, KV_LORA, LOCAL_H, T_DYN, V_DIM

MM_T_TILE = 16          # cube M tile; rows are a 16-row boxed tile
O_N_TILE = 256          # o_proj output-column tile over D
O_K_TILE = 256          # o_proj head-space reduction tile


def golden_mla_epilog_prefill(
    attn_out: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    """Head-space projection. ``o_proj`` is BF16 in the deployment checkpoint."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return torch.nn.functional.linear(flattened.float(), w_o.float())


def golden_mla_epilog_decode(
    attn_out: torch.Tensor,
    w_o_absorbed: torch.Tensor,
) -> torch.Tensor:
    """The absorbed path: ``kv_b_proj``'s value half is folded in, so BF16."""
    flattened = attn_out.reshape(*attn_out.shape[:-2], -1)
    return torch.nn.functional.linear(flattened.float(), w_o_absorbed.float())


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
            acc = pl.create_tensor([MM_T_TILE, O_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline((LOCAL_H * V_DIM) // O_K_TILE, stage=2):
                k0 = kb * O_K_TILE
                attn_tile = pl.slice(
                    attn_flat, [MM_T_TILE, O_K_TILE], [t0, k0], valid_shape=[valid_rows, O_K_TILE]
                )
                w_tile = w_o[n0 : n0 + O_N_TILE, k0 : k0 + O_K_TILE]
                acc = pl.matmul_acc(acc, attn_tile, w_tile, b_trans=True, init_cond=(kb == 0))
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
            acc = pl.create_tensor([MM_T_TILE, O_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline((LOCAL_H * KV_LORA) // O_K_TILE, stage=2):
                k0 = kb * O_K_TILE
                attn_tile = pl.slice(
                    attn_flat, [MM_T_TILE, O_K_TILE], [t0, k0], valid_shape=[valid_rows, O_K_TILE]
                )
                w_tile = w_o_absorbed[n0 : n0 + O_N_TILE, k0 : k0 + O_K_TILE]
                acc = pl.matmul_acc(acc, attn_tile, w_tile, b_trans=True, init_cond=(kb == 0))
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
    parser.add_argument("-p", "--platform", default="a2a3", choices=["a2a3", "a2a3sim"])
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
            config={"platform": args.platform, "device_id": args.device},
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
