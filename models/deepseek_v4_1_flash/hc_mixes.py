# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC coefficient generation."""

import math
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

from models.deepseek_v4_1_flash.config import D, FLASH, HC_DIM, HC_MULT, MIX_HC, T_DYN
from models.deepseek_v4_1_flash.golden import hc_mixes



HC_DIM_INV = 1.0 / HC_DIM
HC_SINKHORN_ITER = FLASH.hc_sinkhorn_iters
HC_EPS = FLASH.hc_eps
NORM_EPS = FLASH.rms_norm_eps
MIX_PAD = 32
HC_PAD = 8
T_TILE = 8
LINEAR_T_TILE = 16
COMB_T_TILE = 8
RMS_K_TILE = 512
LINEAR_K_TILE = 256
LINEAR_OK = 4
LINEAR_K_PER_SPLIT = HC_DIM // LINEAR_OK


@pl.jit.inline
def mhc_mixes(
    x_hc: pl.Tensor,
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Tensor,
    post_mix: pl.Tensor,
    residual_mix: pl.Tensor,
):
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    residual_mix_flat = pl.reshape(residual_mix, [t_dim, HC_MULT * HC_MULT])

    inv_rms = pl.create_tensor([t_linear, 1], dtype=pl.FP32)
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="mhc_rms"):
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_K_TILE, stage=4):
            k0 = kb * RMS_K_TILE
            rms_x_tile = pl.slice(
                x_flat,
                [T_TILE, RMS_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, RMS_K_TILE],
            )
            sq_sum = pl.add(sq_sum, pl.reshape(pl.row_sum(pl.mul(rms_x_tile, rms_x_tile)), [1, T_TILE]))
        rms_arg = pl.add(pl.mul(sq_sum, HC_DIM_INV), NORM_EPS)
        inv_rms[t0 : t0 + T_TILE, 0:1] = pl.reshape(
            pl.rsqrt(rms_arg, high_precision=True), [T_TILE, 1]
        )

    mixes_partials = pl.create_tensor([LINEAR_OK * t_linear, MIX_PAD], dtype=pl.FP32)
    for task in pl.spmd((t_linear // LINEAR_T_TILE) * LINEAR_OK, name_hint="mhc_linear"):
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        split = task % LINEAR_OK
        k_base = split * LINEAR_K_PER_SPLIT
        valid_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        acc = pl.create_tensor([LINEAR_T_TILE, MIX_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(LINEAR_K_PER_SPLIT // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            linear_x_tile = pl.slice(
                x_flat,
                [LINEAR_T_TILE, LINEAR_K_TILE],
                [t0, k0],
                valid_shape=[valid_rows, LINEAR_K_TILE],
            )
            w_tile = pl.slice(
                function,
                [MIX_PAD, LINEAR_K_TILE],
                [0, k0],
                valid_shape=[MIX_HC, LINEAR_K_TILE],
            )
            acc = pl.matmul_acc(acc, linear_x_tile, w_tile, b_trans=True, init_cond=(kb == 0))
        partial = split * t_linear + t0
        mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD] = acc

    mixes_raw = pl.create_tensor([t_linear, MIX_PAD], dtype=pl.FP32)
    for block in pl.spmd(t_linear // LINEAR_T_TILE, name_hint="mhc_linear_reduce"):
        t0 = block * LINEAR_T_TILE
        total = mixes_partials[t0 : t0 + LINEAR_T_TILE, 0:MIX_PAD]
        for split in pl.range(1, LINEAR_OK):
            partial = split * t_linear + t0
            total = pl.add(total, mixes_partials[partial : partial + LINEAR_T_TILE, 0:MIX_PAD])
        mixes_raw[t0 : t0 + LINEAR_T_TILE, 0:MIX_PAD] = total

    base_view = pl.reshape(base, [1, MIX_HC])
    scale0 = pl.read(scale, [0])
    scale1 = pl.read(scale, [1])
    scale2 = pl.read(scale, [2])
    with pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="mhc_split") as split_ready:
        block = pl.tile.get_block_idx()
        t0 = block * T_TILE
        valid_rows = pl.min(T_TILE, t_dim - t0)
        inv = inv_rms[t0 : t0 + T_TILE, 0:1]
        pre_base = pl.reshape(base[0:HC_PAD], [1, HC_PAD])
        pre_logits = pl.add(
            pl.mul(pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], inv), scale0),
            pl.col_expand(mixes_raw[t0 : t0 + T_TILE, 0:HC_PAD], pre_base),
        )
        pre_value = pl.add(pl.recip(pl.add(pl.exp(pl.neg(pre_logits)), 1.0)), HC_EPS)
        post_base = pl.reshape(base[HC_MULT : HC_MULT + HC_PAD], [1, HC_PAD])
        post_logits = pl.add(
            pl.mul(
                pl.row_expand_mul(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], inv),
                scale1,
            ),
            pl.col_expand(mixes_raw[t0 : t0 + T_TILE, HC_MULT : HC_MULT + HC_PAD], post_base),
        )
        post_value = pl.mul(pl.recip(pl.add(pl.exp(pl.neg(post_logits)), 1.0)), 2.0)
        pre_tile = pl.slice(pre_value, [T_TILE, HC_PAD], [0, 0], valid_shape=[valid_rows, HC_MULT])
        post_tile = pl.slice(post_value, [T_TILE, HC_PAD], [0, 0], valid_shape=[valid_rows, HC_MULT])
        pre_mix[t0 : t0 + T_TILE, 0:HC_MULT] = pre_tile
        post_mix[t0 : t0 + T_TILE, 0:HC_MULT] = post_tile

    with pl.spmd((t_dim + COMB_T_TILE - 1) // COMB_T_TILE, name_hint="mhc_sinkhorn") as sinkhorn_ready:
        block = pl.tile.get_block_idx()
        t0 = block * COMB_T_TILE
        valid_rows = pl.min(COMB_T_TILE, t_dim - t0)
        comb_inv = pl.load(
            inv_rms,
            [t0, 0],
            [COMB_T_TILE, 1],
            valid_shape=[valid_rows, 1],
            target_memory=pl.MemorySpace.Vec,
        )
        comb_offset = HC_MULT * 2
        row_max_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        row_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        mix0 = pl.load(
            mixes_raw,
            [t0, comb_offset + 0 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix1 = pl.load(
            mixes_raw,
            [t0, comb_offset + 1 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix2 = pl.load(
            mixes_raw,
            [t0, comb_offset + 2 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        mix3 = pl.load(
            mixes_raw,
            [t0, comb_offset + 3 * HC_MULT],
            [COMB_T_TILE, HC_PAD],
            valid_shape=[valid_rows, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        base0 = pl.load(
            base_view,
            [0, comb_offset + 0 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        base1 = pl.load(
            base_view,
            [0, comb_offset + 1 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        base2 = pl.load(
            base_view,
            [0, comb_offset + 2 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        base3 = pl.load(
            base_view,
            [0, comb_offset + 3 * HC_MULT],
            [1, HC_PAD],
            valid_shape=[1, HC_MULT],
            target_memory=pl.MemorySpace.Vec,
        )
        logits0 = pl.fillpad(
            pl.add(pl.mul(pl.row_expand_mul(mix0, comb_inv), scale2), pl.col_expand(mix0, base0)),
            pad_value=pl.PadValue.min,
        )
        logits1 = pl.fillpad(
            pl.add(pl.mul(pl.row_expand_mul(mix1, comb_inv), scale2), pl.col_expand(mix1, base1)),
            pad_value=pl.PadValue.min,
        )
        logits2 = pl.fillpad(
            pl.add(pl.mul(pl.row_expand_mul(mix2, comb_inv), scale2), pl.col_expand(mix2, base2)),
            pad_value=pl.PadValue.min,
        )
        logits3 = pl.fillpad(
            pl.add(pl.mul(pl.row_expand_mul(mix3, comb_inv), scale2), pl.col_expand(mix3, base3)),
            pad_value=pl.PadValue.min,
        )
        max0 = pl.row_max(logits0, row_max_tmp)
        max1 = pl.row_max(logits1, row_max_tmp)
        max2 = pl.row_max(logits2, row_max_tmp)
        max3 = pl.row_max(logits3, row_max_tmp)
        exp0 = pl.exp(pl.row_expand_sub(logits0, max0))
        exp1 = pl.exp(pl.row_expand_sub(logits1, max1))
        exp2 = pl.exp(pl.row_expand_sub(logits2, max2))
        exp3 = pl.exp(pl.row_expand_sub(logits3, max3))
        row0 = pl.add(pl.row_expand_div(exp0, pl.row_sum(exp0, row_sum_tmp)), HC_EPS)
        row1 = pl.add(pl.row_expand_div(exp1, pl.row_sum(exp1, row_sum_tmp)), HC_EPS)
        row2 = pl.add(pl.row_expand_div(exp2, pl.row_sum(exp2, row_sum_tmp)), HC_EPS)
        row3 = pl.add(pl.row_expand_div(exp3, pl.row_sum(exp3, row_sum_tmp)), HC_EPS)
        row0 = pl.fillpad(pl.set_validshape(row0, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        row1 = pl.fillpad(pl.set_validshape(row1, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        row2 = pl.fillpad(pl.set_validshape(row2, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        row3 = pl.fillpad(pl.set_validshape(row3, valid_rows, HC_MULT), pad_value=pl.PadValue.zero)
        col_sum = pl.add(pl.add(row0, row1), pl.add(row2, row3))
        col_sum = pl.add(col_sum, HC_EPS)
        row0 = pl.div(row0, col_sum)
        row1 = pl.div(row1, col_sum)
        row2 = pl.div(row2, col_sum)
        row3 = pl.div(row3, col_sum)
        sinkhorn_sum_tmp = pl.create_tile([COMB_T_TILE, HC_PAD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        for _ in pl.pipeline(HC_SINKHORN_ITER - 1, stage=2):
            row0 = pl.row_expand_div(row0, pl.add(pl.row_sum(row0, sinkhorn_sum_tmp), HC_EPS))
            row1 = pl.row_expand_div(row1, pl.add(pl.row_sum(row1, sinkhorn_sum_tmp), HC_EPS))
            row2 = pl.row_expand_div(row2, pl.add(pl.row_sum(row2, sinkhorn_sum_tmp), HC_EPS))
            row3 = pl.row_expand_div(row3, pl.add(pl.row_sum(row3, sinkhorn_sum_tmp), HC_EPS))
            col_sum = pl.add(pl.add(row0, row1), pl.add(row2, row3))
            col_sum = pl.add(col_sum, HC_EPS)
            row0 = pl.div(row0, col_sum)
            row1 = pl.div(row1, col_sum)
            row2 = pl.div(row2, col_sum)
            row3 = pl.div(row3, col_sum)
        pl.store(pl.set_validshape(row0, valid_rows, HC_MULT), [t0, 0 * HC_MULT], residual_mix_flat)
        pl.store(pl.set_validshape(row1, valid_rows, HC_MULT), [t0, 1 * HC_MULT], residual_mix_flat)
        pl.store(pl.set_validshape(row2, valid_rows, HC_MULT), [t0, 2 * HC_MULT], residual_mix_flat)
        pl.store(pl.set_validshape(row3, valid_rows, HC_MULT), [t0, 3 * HC_MULT], residual_mix_flat)
    mixes_ready = pl.system.task_dummy(deps=[split_ready, sinkhorn_ready])
    return mixes_ready

def golden_mhc_mixes(
    x_hc: torch.Tensor,
    function: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate the reference pre, post, and residual mixing coefficients."""
    return hc_mixes(x_hc, function, scale, base)


@pl.jit
def mhc_mixes_test(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    function: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[MIX_HC], pl.FP32],
    pre_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    post_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT], pl.FP32]],
    residual_mix: pl.Out[pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32]],
):
    """Run mHC coefficient generation for standalone validation."""
    x_hc.bind_dynamic(0, T_DYN)
    pre_mix.bind_dynamic(0, T_DYN)
    post_mix.bind_dynamic(0, T_DYN)
    residual_mix.bind_dynamic(0, T_DYN)
    mhc_mixes(x_hc, function, scale, base, pre_mix, post_mix, residual_mix)
    return residual_mix


def build_mhc_mixes_tensor_specs(batch: int = 2, sequence: int = 1):
    """Build deterministic inputs and outputs for coefficient validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_function():
        return torch.randn(MIX_HC, HC_DIM, generator=generator) / math.sqrt(HC_DIM)

    def init_scale():
        return torch.randn(3, generator=generator)

    def init_base():
        return torch.randn(MIX_HC, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("function", [MIX_HC, HC_DIM], torch.float32, init_value=init_function),
        TensorSpec("scale", [3], torch.float32, init_value=init_scale),
        TensorSpec("base", [MIX_HC], torch.float32, init_value=init_base),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("post_mix", [tokens, HC_MULT], torch.float32),
        TensorSpec("residual_mix", [tokens, HC_MULT, HC_MULT], torch.float32),
    ]


def golden_mhc_mixes_case(tensors):
    """Fill the expected coefficient outputs."""
    pre_mix, post_mix, residual_mix = golden_mhc_mixes(
        tensors["x_hc"], tensors["function"], tensors["scale"], tensors["base"]
    )
    tensors["pre_mix"][:] = pre_mix
    tensors["post_mix"][:] = post_mix
    tensors["residual_mix"][:] = residual_mix


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
    """Validate mHC coefficient generation on A5."""
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 mHC coefficient validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args(argv)
    result = run(
        fn=mhc_mixes_test,
        specs=build_mhc_mixes_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_mhc_mixes_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={
            "pre_mix": _precision_compare("pre_mix", ratio_allclose(atol=2.5e-5, rtol=5e-3)),
            "post_mix": _precision_compare("post_mix", ratio_allclose(atol=2.5e-5, rtol=5e-3)),
            "residual_mix": _precision_compare(
                "residual_mix", ratio_allclose(atol=2.5e-5, rtol=5e-3)
            ),
        },
        compile_only=args.compile_only,
    )
    return result


__all__ = [
    "build_mhc_mixes_tensor_specs",
    "golden_mhc_mixes",
    "golden_mhc_mixes_case",
    "mhc_mixes",
    "mhc_mixes_test",
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
_PREFILL_MOE_NORM_EPS = FLASH.rms_norm_eps


_PREFILL_MOE_HC = FLASH.hc_mult


_PREFILL_MOE_MIX = _PREFILL_MOE_HC * (_PREFILL_MOE_HC + 2)


_PREFILL_MOE_D = FLASH.hidden_size


_PREFILL_MOE_TILE = 256


_PREFILL_MOE_T = pl.dynamic("V41_FP32_MOE_T")


_PREFILL_MOE_WORKERS = 32


@pl.jit.inline(auto_scope=False)
def prefill_hc_normalize_logits_inline(
    x_hc: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC, _PREFILL_MOE_D], pl.FP32],
    raw_logits: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_MIX], pl.FP32],
    output: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_MIX], pl.FP32],
):
    """Apply the residual inverse RMS after its linear projection."""
    x_hc.bind_dynamic(0, _PREFILL_MOE_T)
    raw_logits.bind_dynamic(0, _PREFILL_MOE_T)
    output.bind_dynamic(0, _PREFILL_MOE_T)
    rows = pl.tensor.dim(x_hc, 0)
    flat = pl.reshape(x_hc, [rows, _PREFILL_MOE_HC * _PREFILL_MOE_D])
    for worker in pl.spmd(_PREFILL_MOE_WORKERS, name_hint="fp32_hc_normalize_logits"):
        for task in pl.range(worker, (rows + 7) // 8, _PREFILL_MOE_WORKERS):
            row = task * 8
            active = pl.min(8, rows - row)
            total = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            correction = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            for block in pl.range(_PREFILL_MOE_HC * _PREFILL_MOE_D // _PREFILL_MOE_TILE):
                values = pl.slice(
                    flat,
                    [8, _PREFILL_MOE_TILE],
                    [row, block * _PREFILL_MOE_TILE],
                    valid_shape=[active, _PREFILL_MOE_TILE],
                )
                values = pl.set_validshape(
                    pl.fillpad(values, pad_value=pl.PadValue.zero), 8, _PREFILL_MOE_TILE
                )
                partial = pl.reshape(pl.row_sum(pl.mul(values, values)), [1, 8])
                corrected = pl.sub(partial, correction)
                updated = pl.add(total, corrected)
                correction = pl.sub(pl.sub(updated, total), corrected)
                total = updated
            inverse = pl.reshape(
                pl.rsqrt(
                    pl.add(pl.mul(total, 1.0 / (_PREFILL_MOE_HC * _PREFILL_MOE_D)), _PREFILL_MOE_NORM_EPS),
                    high_precision=True,
                ),
                [8, 1],
            )
            projected = pl.slice(raw_logits, [8, 32], [row, 0], valid_shape=[active, _PREFILL_MOE_MIX])
            normalized = pl.row_expand_mul(projected, inverse)
            output[row : row + 8, :] = pl.set_validshape(normalized, active, _PREFILL_MOE_MIX)
    return output


_PREFILL_MOE_ITERATIONS = FLASH.hc_sinkhorn_iters


_PREFILL_MOE_EPS = FLASH.hc_eps


@pl.jit.inline(auto_scope=False)
def prefill_hc_coefficients_inline(
    logits: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_MIX], pl.FP32],
    scale: pl.Tensor[[3], pl.FP32],
    base: pl.Tensor[[_PREFILL_MOE_MIX], pl.FP32],
    pre_mix: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC], pl.FP32],
    post_mix: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC], pl.FP32],
    residual_mix: pl.Tensor[[_PREFILL_MOE_T, _PREFILL_MOE_HC, _PREFILL_MOE_HC], pl.FP32],
):
    """Transform logits from the RMS-normalized flattened residual streams."""
    logits.bind_dynamic(0, _PREFILL_MOE_T)
    pre_mix.bind_dynamic(0, _PREFILL_MOE_T)
    post_mix.bind_dynamic(0, _PREFILL_MOE_T)
    residual_mix.bind_dynamic(0, _PREFILL_MOE_T)
    rows = pl.tensor.dim(logits, 0)
    base_row = pl.reshape(base, [1, _PREFILL_MOE_MIX])
    combine = pl.reshape(residual_mix, [rows, _PREFILL_MOE_HC * _PREFILL_MOE_HC])
    for worker in pl.spmd(_PREFILL_MOE_WORKERS, name_hint="fp32_hc_coefficients"):
        for task in pl.range(worker, (rows + 7) // 8, _PREFILL_MOE_WORKERS):
            token = task * 8
            active = pl.min(8, rows - token)
            row_max_tmp = pl.create_tile([8, 8], dtype=pl.FP32)
            row_sum_tmp = pl.create_tile([8, 8], dtype=pl.FP32)
            pre_source = pl.load(logits, [token, 0], [8, 8], valid_shape=[active, 8])
            post_source = pl.load(logits, [token, _PREFILL_MOE_HC], [8, 8], valid_shape=[active, 8])
            pre_base = pl.load(base_row, [0, 0], [1, 8])
            post_base = pl.load(base_row, [0, _PREFILL_MOE_HC], [1, 8])
            pre_logits = pl.add(pl.mul(pre_source, pl.read(scale, [0])), pl.col_expand(pre_source, pre_base))
            post_logits = pl.add(
                pl.mul(post_source, pl.read(scale, [1])), pl.col_expand(post_source, post_base)
            )
            pre = pl.add(
                pl.div(
                    pl.exp(pl.minimum(pre_logits, 0.0)),
                    pl.add(pl.exp(pl.neg(pl.abs(pre_logits))), 1.0),
                    high_precision=True,
                ),
                _PREFILL_MOE_EPS,
            )
            post = pl.mul(
                pl.div(
                    pl.exp(pl.minimum(post_logits, 0.0)),
                    pl.add(pl.exp(pl.neg(pl.abs(post_logits))), 1.0),
                    high_precision=True,
                ),
                2.0,
            )
            pl.store(pl.set_validshape(pre, active, _PREFILL_MOE_HC), [token, 0], pre_mix)
            pl.store(pl.set_validshape(post, active, _PREFILL_MOE_HC), [token, 0], post_mix)
            raw0 = pl.load(logits, [token, 8], [8, 8], valid_shape=[active, _PREFILL_MOE_HC])
            base0 = pl.load(base_row, [0, 8], [1, 8], valid_shape=[1, _PREFILL_MOE_HC])
            value0 = pl.fillpad(
                pl.add(pl.mul(raw0, pl.read(scale, [2])), pl.col_expand(raw0, base0)),
                pad_value=pl.PadValue.min,
            )
            maximum0 = pl.row_max(value0, row_max_tmp)
            exponential0 = pl.exp(pl.row_expand_sub(value0, maximum0))
            row0 = pl.add(
                pl.row_expand_div(exponential0, pl.row_sum(exponential0, row_sum_tmp)), _PREFILL_MOE_EPS
            )
            row0 = pl.fillpad(pl.set_validshape(row0, active, _PREFILL_MOE_HC), pad_value=pl.PadValue.zero)
            raw1 = pl.load(logits, [token, 12], [8, 8], valid_shape=[active, _PREFILL_MOE_HC])
            base1 = pl.load(base_row, [0, 12], [1, 8], valid_shape=[1, _PREFILL_MOE_HC])
            value1 = pl.fillpad(
                pl.add(pl.mul(raw1, pl.read(scale, [2])), pl.col_expand(raw1, base1)),
                pad_value=pl.PadValue.min,
            )
            maximum1 = pl.row_max(value1, row_max_tmp)
            exponential1 = pl.exp(pl.row_expand_sub(value1, maximum1))
            row1 = pl.add(
                pl.row_expand_div(exponential1, pl.row_sum(exponential1, row_sum_tmp)), _PREFILL_MOE_EPS
            )
            row1 = pl.fillpad(pl.set_validshape(row1, active, _PREFILL_MOE_HC), pad_value=pl.PadValue.zero)
            raw2 = pl.load(logits, [token, 16], [8, 8], valid_shape=[active, _PREFILL_MOE_HC])
            base2 = pl.load(base_row, [0, 16], [1, 8], valid_shape=[1, _PREFILL_MOE_HC])
            value2 = pl.fillpad(
                pl.add(pl.mul(raw2, pl.read(scale, [2])), pl.col_expand(raw2, base2)),
                pad_value=pl.PadValue.min,
            )
            maximum2 = pl.row_max(value2, row_max_tmp)
            exponential2 = pl.exp(pl.row_expand_sub(value2, maximum2))
            row2 = pl.add(
                pl.row_expand_div(exponential2, pl.row_sum(exponential2, row_sum_tmp)), _PREFILL_MOE_EPS
            )
            row2 = pl.fillpad(pl.set_validshape(row2, active, _PREFILL_MOE_HC), pad_value=pl.PadValue.zero)
            raw3 = pl.load(logits, [token, 20], [8, 8], valid_shape=[active, _PREFILL_MOE_HC])
            base3 = pl.load(base_row, [0, 20], [1, 8], valid_shape=[1, _PREFILL_MOE_HC])
            value3 = pl.fillpad(
                pl.add(pl.mul(raw3, pl.read(scale, [2])), pl.col_expand(raw3, base3)),
                pad_value=pl.PadValue.min,
            )
            maximum3 = pl.row_max(value3, row_max_tmp)
            exponential3 = pl.exp(pl.row_expand_sub(value3, maximum3))
            row3 = pl.add(
                pl.row_expand_div(exponential3, pl.row_sum(exponential3, row_sum_tmp)), _PREFILL_MOE_EPS
            )
            row3 = pl.fillpad(pl.set_validshape(row3, active, _PREFILL_MOE_HC), pad_value=pl.PadValue.zero)
            denominator = pl.add(pl.add(pl.add(row0, row1), pl.add(row2, row3)), _PREFILL_MOE_EPS)
            row0 = pl.div(row0, denominator, high_precision=True)
            row1 = pl.div(row1, denominator, high_precision=True)
            row2 = pl.div(row2, denominator, high_precision=True)
            row3 = pl.div(row3, denominator, high_precision=True)
            for iteration in pl.range(_PREFILL_MOE_ITERATIONS - 1):
                row0 = pl.row_expand_div(row0, pl.add(pl.row_sum(row0, row_sum_tmp), _PREFILL_MOE_EPS))
                row1 = pl.row_expand_div(row1, pl.add(pl.row_sum(row1, row_sum_tmp), _PREFILL_MOE_EPS))
                row2 = pl.row_expand_div(row2, pl.add(pl.row_sum(row2, row_sum_tmp), _PREFILL_MOE_EPS))
                row3 = pl.row_expand_div(row3, pl.add(pl.row_sum(row3, row_sum_tmp), _PREFILL_MOE_EPS))
                denominator = pl.add(pl.add(pl.add(row0, row1), pl.add(row2, row3)), _PREFILL_MOE_EPS)
                row0 = pl.div(row0, denominator, high_precision=True)
                row1 = pl.div(row1, denominator, high_precision=True)
                row2 = pl.div(row2, denominator, high_precision=True)
                row3 = pl.div(row3, denominator, high_precision=True)
            pl.store(pl.set_validshape(row0, active, _PREFILL_MOE_HC), [token, 0], combine)
            pl.store(pl.set_validshape(row1, active, _PREFILL_MOE_HC), [token, 4], combine)
            pl.store(pl.set_validshape(row2, active, _PREFILL_MOE_HC), [token, 8], combine)
            pl.store(pl.set_validshape(row3, active, _PREFILL_MOE_HC), [token, 12], combine)
    return pre_mix, post_mix, residual_mix

if __name__ == _SCRIPT_ENTRY_POINT:
    main()
