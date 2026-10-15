# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC residual expansion."""

from utils import owner_rows
from config import TP_SIZE

import pypto.language as pl
import torch

from config import FLASH as M, SP_T_DYN, T_DYN


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim


def golden_hc_post(
    sublayer: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    residual_mix: torch.Tensor,
) -> torch.Tensor:
    """Expand a sublayer result and mix the four residual streams."""
    update = post_mix.unsqueeze(-1) * sublayer.unsqueeze(-2)
    skip = (residual_mix.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)
    return (update.float() + skip.float()).to(sublayer.dtype)


OUT_TILE = 2
D_TILE = 2560
D_TILES = D // D_TILE


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
    for block in pl.spmd(t_dim * (HC_MULT // OUT_TILE) * D_TILES, name_hint="mhc_post"):
        pair_block = block // D_TILES
        t = pair_block // (HC_MULT // OUT_TILE)
        out_base = (pair_block % (HC_MULT // OUT_TILE)) * OUT_TILE
        d0 = (block % D_TILES) * D_TILE
        x_tile = pl.cast(sublayer[t : t + 1, d0 : d0 + D_TILE], target_type=pl.FP32)
        residual0 = residual_flat[t : t + 1, d0 : d0 + D_TILE]
        residual1 = residual_flat[t : t + 1, D + d0 : D + d0 + D_TILE]
        residual2 = residual_flat[t : t + 1, 2 * D + d0 : 2 * D + d0 + D_TILE]
        residual3 = residual_flat[t : t + 1, 3 * D + d0 : 3 * D + d0 + D_TILE]
        for out_offset in pl.unroll(OUT_TILE):
            out_h = out_base + out_offset
            value = pl.mul(x_tile, pl.read(post_mix, [t, out_h]))
            value = pl.add(value, pl.mul(residual0, pl.read(residual_mix_flat, [t, out_h])))
            value = pl.add(value, pl.mul(residual1, pl.read(residual_mix_flat, [t, HC_MULT + out_h])))
            value = pl.add(value, pl.mul(residual2, pl.read(residual_mix_flat, [t, 2 * HC_MULT + out_h])))
            value = pl.add(value, pl.mul(residual3, pl.read(residual_mix_flat, [t, 3 * HC_MULT + out_h])))
            output_flat[t : t + 1, out_h * D + d0 : out_h * D + d0 + D_TILE] = pl.cast(
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
    return golden_hc_post(sublayer, residual, post_mix, residual_mix).to(torch.float32)


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
        compare_fn={"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
        compile_only=args.compile_only,
    )
    return result


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


@pl.jit.inline
def prefill_sp_post(
    sublayer: pl.Tensor[[SP_T_DYN, D], pl.BF16],
    residual: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    post_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[SP_T_DYN, HC_MULT, HC_MULT], pl.FP32],
    output: pl.Tensor[[SP_T_DYN, HC_MULT, D], pl.FP32],
    next_pre_mix: pl.Tensor[[SP_T_DYN, HC_MULT], pl.FP32],
    tp_rank: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Run local mHC-post, writing active values and zero padding in one producer.

    A later pure-output clear is not an ordering edge for the post task. Each
    residual element is instead written exactly once, with mhc_post's FP32
    accumulation order and single BF16 rounding boundary.
    """
    tokens = pl.tensor.dim(residual, 0)
    residual_flat = pl.reshape(residual, [tokens, HC_MULT * D])
    mix_flat = pl.reshape(residual_mix, [tokens, HC_MULT * HC_MULT])
    output_flat = pl.reshape(output, [tokens, HC_MULT * D])
    for block in pl.spmd(tokens * HC_MULT, name_hint="sp_mhc_post"):
        row = block // HC_MULT
        out_h = block % HC_MULT
        for col in pl.pipeline(0, D, 256, stage=2):
            value = pl.full([1, 256], dtype=pl.FP32, value=0.0)
            if tp_rank * tokens + row < num_tokens:
                hidden = pl.cast(sublayer[row:row + 1, col:col + 256], pl.FP32)
                value = pl.mul(hidden, pl.read(post_mix, [row, out_h]))
                for in_h in pl.unroll(HC_MULT):
                    lane = residual_flat[row:row + 1, in_h * D + col:in_h * D + col + 256]
                    value = pl.add(value, pl.mul(lane, pl.read(mix_flat, [row, in_h * HC_MULT + out_h])))
            output_flat[row:row + 1, out_h * D + col:out_h * D + col + 256] = pl.cast(
                pl.cast(value, pl.BF16, mode="rint"), pl.FP32,
            )
    for row in pl.spmd(tokens, name_hint="sp_mix_zero_padding"):
        if tp_rank * tokens + row >= num_tokens:
            zero_mix = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            next_pre_mix[row:row + 1, :] = pl.set_validshape(zero_mix, 1, HC_MULT)
    return output, next_pre_mix


def golden_attention_post(tensors, post, residual):
    """Populate the common post-Attention mHC boundary."""
    active = int(tensors["num_tokens"])
    sharded = "gathered" in tensors
    for rank in range(tensors["x_hc"].shape[0]):
        if sharded:
            _first, count, _width = owner_rows(tensors["attention_output"], rank, active, tp_size=TP_SIZE)
            tensors["attention_hidden"][rank].zero_()
            tensors["attention_output"][rank][count:].zero_()
            if count:
                tensors["attention_hidden"][rank][:count].copy_(
                    golden_mhc_post(
                        tensors["attention_output"][rank][:count],
                        tensors["x_hc"][rank][:count], post[rank][:count], residual[rank][:count]
                    )
                )
        else:
            tensors["attention_hidden"][rank].copy_(
                golden_mhc_post(
                    tensors["attention_output"][rank], tensors["x_hc"][rank], post[rank], residual[rank]
                )
            )


if __name__ == "__main__":
    main()
