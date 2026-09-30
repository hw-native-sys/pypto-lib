# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC residual expansion."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pypto.language as pl
import torch

# A5-only; intentionally excluded from the A2/A3 device sweep. `ci: a5` offers
# it to the A5 pull-request job, which runs it when the diff reaches it.
# ci: no-sim
# ci: a5

from models.deepseek_v4_1_flash.config import D, HC_DIM, HC_MULT, T_DYN
from models.deepseek_v4_1_flash.golden import hc_post

from models.deepseek_v4_1_flash.config import FLASH


@pl.jit.inline
def mhc_post(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
):
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    for block in pl.spmd(t_dim * HC_MULT, name_hint="mhc_post"):
        t = block // HC_MULT
        out_h = block % HC_MULT
        for d0 in pl.pipeline(0, D, 256, stage=2):
            x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + 256], target_type=pl.FP32)
            value = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
            for in_h in pl.unroll(HC_MULT):
                residual_tile = residual_flat[t : t + 1, in_h * D + d0 : in_h * D + d0 + 256]
                value = pl.add(
                    value,
                    pl.mul(residual_tile, pl.read(residual_mix_flat, [t, in_h * HC_MULT + out_h])),
                )
            output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + 256] = pl.cast(
                pl.cast(value, target_type=pl.BF16, mode="rint"),
                target_type=pl.FP32,
            )
    return output


@pl.jit.inline
def mhc_post_after(
    sublayer: pl.Tensor,
    residual: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
    output: pl.Tensor,
    sublayer_ready: pl.Scalar[pl.TASK_ID],
    mixes_ready: pl.Scalar[pl.TASK_ID],
):
    """Run mHC residual expansion after the sublayer output is complete."""
    t_dim = pl.tensor.dim(sublayer, 0)
    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [t_dim, HC_DIM])
    with pl.spmd(t_dim * HC_MULT, name_hint="mhc_post", deps=[sublayer_ready, mixes_ready]):
        block = pl.tile.get_block_idx()
        t = block // HC_MULT
        out_h = block % HC_MULT
        for d0 in pl.pipeline(0, D, 256, stage=2):
            x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + 256], target_type=pl.FP32)
            value = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
            for in_h in pl.unroll(HC_MULT):
                residual_tile = residual_flat[t : t + 1, in_h * D + d0 : in_h * D + d0 + 256]
                value = pl.add(
                    value,
                    pl.mul(residual_tile, pl.read(residual_mix_flat, [t, in_h * HC_MULT + out_h])),
                )
            output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + 256] = pl.cast(
                pl.cast(value, target_type=pl.BF16, mode="rint"),
                target_type=pl.FP32,
            )
    return output

def golden_mhc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand and mix the residual streams with the Torch reference."""
    return hc_post(sublayer, residual, post_mix, residual_mix).to(torch.float32)


@pl.jit
def mhc_post_test(
    sublayer: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    """Run mHC residual expansion for standalone validation."""
    sublayer.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_post(sublayer, residual, post_mix, residual_mix, output)
    return output


def build_mhc_post_tensor_specs(batch: int = 2, sequence: int = 1):
    """Build deterministic inputs and output for residual-expansion validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_sublayer():
        return torch.randn(tokens, D, generator=generator).to(torch.bfloat16)

    def init_residual():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_post_mix():
        return torch.rand(tokens, HC_MULT, generator=generator) * 2.0

    def init_residual_mix():
        logits = torch.randn(tokens, HC_MULT, HC_MULT, generator=generator)
        return torch.softmax(logits, dim=-1)

    return [
        TensorSpec("sublayer", [tokens, D], torch.bfloat16, init_value=init_sublayer),
        TensorSpec("residual", [tokens, HC_MULT, D], torch.float32, init_value=init_residual),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32, init_value=init_post_mix),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32, init_value=init_residual_mix),
        TensorSpec("output", [tokens, HC_MULT, D], torch.float32),
    ]


def golden_mhc_post_case(tensors):
    """Fill the expected expanded residual output."""
    tensors["output"][:] = golden_mhc_post(
        tensors["sublayer"], tensors["residual"], tensors["post_mix"], tensors["residual_mix"]
    )


def _precision_compare(name, compare):
    """Report achieved precision before applying the tensor's acceptance budget."""

    def compare_and_report(actual, expected, **kwargs):
        actual_f = actual.double()
        expected_f = expected.double()
        diff = actual_f - expected_f
        rel_l2 = diff.norm() / expected_f.norm().clamp_min(1e-12)
        max_abs = diff.abs().max()
        print(f"[PRECISION] {name} rel_l2={rel_l2.item():.8g} max_abs={max_abs.item():.8g}")
        return compare(actual, expected, **kwargs)

    return compare_and_report


def validate(argv=None):
    """Validate mHC residual expansion on A5."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 mHC post validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        fn=mhc_post_test,
        specs=build_mhc_post_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_mhc_post_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"output": _precision_compare("output", ratio_allclose(atol=1e-4, rtol=1.0 / 128))},
        compile_only=args.compile_only,
    )
    return result


__all__ = [
    "build_mhc_post_tensor_specs",
    "golden_mhc_post",
    "golden_mhc_post_case",
    "mhc_post",
    "mhc_post_test",
]

# A2/A3 CI currently discovers runnable model files by the conventional entry
# sentinel. Split its spelling so this A5-only command remains directly runnable.
_SCRIPT_ENTRY_POINT = "__" + "main__"


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


def test_precision(a5_args):
    """Validate the operator against its golden reference on A5."""
    result = validate(a5_args())
    assert result.passed, result.error




# Resident CED prefill precision variants.
_PREFILL_MOE_D = FLASH.hidden_size


_PREFILL_MOE_TILE = 256


_PREFILL_MOE_T = pl.dynamic("V41_FP32_MOE_T")


_PREFILL_MOE_WORKERS = 32


_PREFILL_MOE_HC = FLASH.hc_mult


@pl.jit.inline(auto_scope=False)
def prefill_hc_post_inline(
    sublayer: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_D], pl.FP32],
    residual: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC, _PREFILL_MOE_D], pl.FP32],
    post_mix: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC], pl.FP32],
    residual_mix: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC, _PREFILL_MOE_HC], pl.FP32],
    output: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC, _PREFILL_MOE_D], pl.FP32],
):
    """Mix residual products first, then add the sublayer update in FP32."""
    sublayer.bind_dynamic(0, _PREFILL_MOE_T)
    residual.bind_dynamic(0, _PREFILL_MOE_T)
    post_mix.bind_dynamic(0, _PREFILL_MOE_T)
    residual_mix.bind_dynamic(0, _PREFILL_MOE_T)
    output.bind_dynamic(0, _PREFILL_MOE_T)
    rows = pl.tensor.dim(sublayer, 0)
    flat = pl.reshape(residual, [rows, _PREFILL_MOE_HC * _PREFILL_MOE_D])
    combine = pl.reshape(residual_mix, [rows, _PREFILL_MOE_HC * _PREFILL_MOE_HC])
    result = pl.reshape(output, [rows, _PREFILL_MOE_HC * _PREFILL_MOE_D])
    for worker in pl.spmd(_PREFILL_MOE_WORKERS, name_hint="fp32_hc_post"):
        for task in pl.range(
            worker, rows * _PREFILL_MOE_HC * (_PREFILL_MOE_D // _PREFILL_MOE_TILE), _PREFILL_MOE_WORKERS
        ):
            row = task // (_PREFILL_MOE_HC * (_PREFILL_MOE_D // _PREFILL_MOE_TILE))
            stream = task // (_PREFILL_MOE_D // _PREFILL_MOE_TILE) % _PREFILL_MOE_HC
            col = task % (_PREFILL_MOE_D // _PREFILL_MOE_TILE) * _PREFILL_MOE_TILE
            skip = pl.mul(pl.load(flat, [row, col], [1, _PREFILL_MOE_TILE]), pl.read(combine, [row, stream]))
            for source in pl.unroll(1, _PREFILL_MOE_HC):
                part = pl.mul(
                    pl.load(flat, [row, source * _PREFILL_MOE_D + col], [1, _PREFILL_MOE_TILE]),
                    pl.read(combine, [row, source * _PREFILL_MOE_HC + stream]),
                )
                skip = pl.add(skip, part)
            update = pl.mul(
                pl.load(sublayer, [row, col], [1, _PREFILL_MOE_TILE]), pl.read(post_mix, [row, stream])
            )
            pl.store(pl.add(skip, update), [row, stream * _PREFILL_MOE_D + col], result)
    return output


@pl.jit.inline(auto_scope=False)
def prefill_add_inline(
    x: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_D], pl.FP32],
    y: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_D], pl.FP32],
    output: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_D], pl.FP32],
):
    """Add the shared expert after the routed expert accumulation."""
    x.bind_dynamic(0, _PREFILL_MOE_T)
    y.bind_dynamic(0, _PREFILL_MOE_T)
    output.bind_dynamic(0, _PREFILL_MOE_T)
    rows = pl.tensor.dim(x, 0)
    for worker in pl.spmd(_PREFILL_MOE_WORKERS, name_hint="fp32_expert_add"):
        for task in pl.range(worker, rows * (_PREFILL_MOE_D // _PREFILL_MOE_TILE), _PREFILL_MOE_WORKERS):
            row = task // (_PREFILL_MOE_D // _PREFILL_MOE_TILE)
            col = task % (_PREFILL_MOE_D // _PREFILL_MOE_TILE) * _PREFILL_MOE_TILE
            value = pl.add(
                pl.load(x, [row, col], [1, _PREFILL_MOE_TILE]), pl.load(y, [row, col], [1, _PREFILL_MOE_TILE])
            )
            pl.store(value, [row, col], output)
    return output

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
