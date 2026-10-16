# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""mHC stream collapse."""

from hc_mixes import golden_mhc_mixes, mhc_mixes
from rmsnorm import golden_rms_norm
from rmsnorm import rms_norm_tiled
from utils import owner_rows

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN, TP_SIZE


# model config
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
MIX_HC = M.mix_hc


D_TILE = 2560


def golden_hc_pre(x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse four HC streams with the kernel's pairwise FP32 reduction."""
    products = x.float() * pre_mix.float().unsqueeze(-1)
    first_pair = products[..., 0, :] + products[..., 1, :]
    second_pair = products[..., 2, :] + products[..., 3, :]
    return (first_pair + second_pair).to(x.dtype)


@pl.jit.inline
def mhc_pre(
    x_hc: pl.Tensor,
    pre_mix: pl.Tensor,
    output: pl.Tensor,
):
    t_dim = pl.tensor.dim(x_hc, 0)
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    for t in pl.spmd(t_dim, name_hint="mhc_pre", allow_early_resolve=True):
        pre0 = pl.read(pre_mix, [t, 0])
        pre1 = pl.read(pre_mix, [t, 1])
        pre2 = pl.read(pre_mix, [t, 2])
        pre3 = pl.read(pre_mix, [t, 3])
        for d0 in pl.pipeline(0, D, D_TILE, stage=2):
            x0 = pl.load(x_flat, [t, d0], [1, D_TILE])
            x1 = pl.load(x_flat, [t, D + d0], [1, D_TILE])
            x2 = pl.load(x_flat, [t, 2 * D + d0], [1, D_TILE])
            x3 = pl.load(x_flat, [t, 3 * D + d0], [1, D_TILE])
            y0 = pl.mul(x0, pre0)
            y1 = pl.mul(x1, pre1)
            y2 = pl.mul(x2, pre2)
            y3 = pl.mul(x3, pre3)
            y_tile = pl.add(pl.add(y0, y1), pl.add(y2, y3))
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            pl.store(y_bf16, [t, d0], output)
    return output


@pl.jit.inline
def attention_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_weight: pl.Tensor[[D], pl.BF16],
    next_pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    post_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    residual_mix: pl.Tensor[[T_DYN, HC_MULT, HC_MULT], pl.FP32],
    x: pl.Tensor[[T_DYN, D], pl.BF16],
):
    """Derive this sublayer's mixes, then collapse with the delayed pre-mix and normalize."""
    tokens = pl.tensor.dim(x_hc, 0)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, next_pre_mix, post_mix, residual_mix)
    collapsed = pl.create_tensor([tokens, D], dtype=pl.BF16)
    mhc_pre(x_hc, pre_mix, collapsed)
    norm_tokens = pl.tensor.dim(collapsed, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(collapsed, attn_norm_weight, x, norm_block, norm_tokens)
    return x


def golden_mhc_pre(x_hc: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
    """Collapse the HC streams with the Torch reference."""
    return golden_hc_pre(x_hc, pre_mix).to(torch.bfloat16)


@pl.jit
def l2_hc_pre(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    pre_mix: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, D], pl.BF16]],
):
    """Run mHC stream collapse for standalone validation."""
    x_hc.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    mhc_pre(x_hc, pre_mix, output)
    return output


def build_mhc_pre_tensor_specs(batch: int = 2, sequence: int = 1):
    """Build deterministic inputs and output for stream-collapse validation."""
    from golden import TensorSpec

    tokens = batch * sequence
    generator = torch.Generator().manual_seed(3)

    def init_x_hc():
        return torch.randn(tokens, HC_MULT, D, generator=generator)

    def init_pre_mix():
        return torch.rand(tokens, HC_MULT, generator=generator)

    return [
        TensorSpec("x_hc", [tokens, HC_MULT, D], torch.float32, init_value=init_x_hc),
        TensorSpec("pre_mix", [tokens, HC_MULT], torch.float32, init_value=init_pre_mix),
        TensorSpec("output", [tokens, D], torch.bfloat16),
    ]


def golden_mhc_pre_case(tensors):
    """Fill the expected collapsed stream output."""
    tensors["output"][:] = golden_mhc_pre(tensors["x_hc"], tensors["pre_mix"])


@pl.jit.inline
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """Clear physical rows that belong to an empty or partial TP slab.

    A TP rank with no logical token still publishes one fixed-capacity slab so
    every rank executes the same distributed window protocol.  Clearing the
    unused rows makes that padding deterministic and prevents an empty owner
    from publishing uninitialized data.
    """
    tokens = pl.tensor.dim(x, 0)
    if tokens > 0:
        for row in pl.spmd(tokens, name_hint="decode_sp_zero_padding"):
            if row >= valid_rows:
                x[row : row + 1, 0:D] = pl.full([1, D], dtype=pl.BF16, value=0.0)


@pl.jit.inline
def mhc_pre_norm(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    attn_norm_weight: pl.Tensor,
    attention_input: pl.Tensor,
    normalized_attention: pl.Tensor,
    owned_rows: pl.Scalar[pl.INT32],
):
    """Collapse the mHC streams and normalize the rows this call owns.

    The block runs between ``mhc_pre`` and the Attention operator, and it is the
    single implementation behind both the spec-driven boundary
    (:func:`attention_pre`) and the C1A compositions, which keep their mixes in
    the caller.  ``owned_rows`` is the caller's row count, never a global one:
    the replicated wiring passes the active batch size, the sequence-parallel
    wiring passes the rank's local count from ``decode_cp_allgather.slab_owner``.  It only masks
    the slab suffix, which is zeroed here so a published slab never carries a
    stale value in its padding rows.
    """
    mhc_pre(x_hc, incoming_pre_mix, attention_input)
    norm_tokens = pl.tensor.dim(attention_input, 0)
    for norm_block in pl.spmd((norm_tokens + 7) // 8, name_hint="rms_norm"):
        rms_norm_tiled(attention_input, attn_norm_weight, normalized_attention, norm_block, norm_tokens)
    zero_bf16_padding(normalized_attention, pl.min(owned_rows, pl.tensor.dim(normalized_attention, 0)))


@pl.jit.inline(auto_scope=False)
def attention_pre(
    x_hc: pl.Tensor,
    incoming_pre_mix: pl.Tensor,
    hc_attn_fn: pl.Tensor,
    hc_attn_scale: pl.Tensor,
    hc_attn_base: pl.Tensor,
    attn_norm_weight: pl.Tensor,
    attention_input: pl.Tensor,
    normalized_attention: pl.Tensor,
    attention_pre_mix: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Build the stable mHC-pre and RMSNorm boundary around any Attention leaf."""
    tokens = pl.tensor.dim(x_hc, 0)
    post_mix = pl.create_tensor([tokens, HC_MULT], dtype=pl.FP32)
    residual_mix = pl.create_tensor([tokens, HC_MULT, HC_MULT], dtype=pl.FP32)
    mhc_mixes(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, attention_pre_mix, post_mix, residual_mix)
    mhc_pre_norm(x_hc, incoming_pre_mix, attn_norm_weight, attention_input, normalized_attention, num_tokens)
    return post_mix, residual_mix


def golden_attention_pre(tensors):
    """Build normalized Attention inputs and return the mHC post state."""
    world = tensors["x_hc"].shape[0]
    active = int(tensors["num_tokens"])
    normalized, post, residual = [], [], []
    for rank in range(world):
        pre, post_mix, residual_mix = golden_mhc_mixes(
            tensors["x_hc"][rank],
            tensors["hc_attn_fn"][rank],
            tensors["hc_attn_scale"][rank],
            tensors["hc_attn_base"][rank],
        )
        tensors["attention_pre_mix"][rank].copy_(pre)
        collapsed = golden_mhc_pre(tensors["x_hc"][rank], tensors["incoming_pre_mix"][rank])
        normalized_rank = torch.full_like(collapsed, 13.0)
        normalized_rank[:active].copy_(golden_rms_norm(collapsed[:active], tensors["attn_norm_weight"][rank]))
        normalized.append(normalized_rank)
        post.append(post_mix)
        residual.append(residual_mix)
    return torch.stack(normalized), post, residual


def golden_attention_pre_sharded(tensors):
    """Build the local mHC/norm state and the gathered full-range input.

    Every rank collapses and normalizes its own slab of the active token range;
    the stacked return value carries the gathered rows each rank's leaf golden
    consumes, and ``tensors["gathered"]`` is filled so the harness can check the
    collector itself.
    """
    world = tensors["x_hc"].shape[0]
    active = int(tensors["num_tokens"])
    # Ownership follows the physical slab, while ``active`` only masks its
    # suffix.  This preserves rank mapping when active tokens are padded.
    normalized, post, residual = [], [], []
    for rank in range(world):
        pre, post_mix, residual_mix = golden_mhc_mixes(
            tensors["x_hc"][rank],
            tensors["hc_attn_fn"][rank],
            tensors["hc_attn_scale"][rank],
            tensors["hc_attn_base"][rank],
        )
        tensors["attention_pre_mix"][rank].copy_(pre)
        collapsed = golden_mhc_pre(tensors["x_hc"][rank], tensors["incoming_pre_mix"][rank])
        _first, count, _width = owner_rows(tensors["x_hc"], rank, active, tp_size=TP_SIZE)
        normalized_rank = torch.full_like(collapsed, 13.0)
        if count:
            normalized_rank[:count].copy_(
                golden_rms_norm(collapsed[:count], tensors["attn_norm_weight"][rank])
            )
        normalized.append(normalized_rank)
        post.append(post_mix)
        residual.append(residual_mix)
    per_rank_full = [None] * world
    for base in range(0, world, TP_SIZE):
        slabs = []
        for offset in range(TP_SIZE):
            rank = base + offset
            _first, count, _width = owner_rows(tensors["x_hc"], offset, active, tp_size=TP_SIZE)
            slabs.append(normalized[rank][:count])
        gathered = torch.cat(slabs, dim=0)[:active]
        for rank in range(base, base + TP_SIZE):
            tensors["gathered"][rank].zero_()
            tensors["gathered"][rank][: gathered.shape[0]].copy_(gathered)
            per_rank_full[rank] = gathered
    return torch.stack(per_rank_full), post, residual


if __name__ == "__main__":
    import argparse

    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="DeepSeek V4.1 mHC pre validation")
    parser.add_argument("-p", "--platform", default="a5", choices=["a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    result = run(
        fn=l2_hc_pre,
        specs=build_mhc_pre_tensor_specs(args.batch, args.sequence),
        golden_fn=golden_mhc_pre_case,
        config={"platform": args.platform, "device_id": args.device},
        rtol=1e-3,
        atol=1e-3,
        compare_fn={"output": ratio_allclose(atol=1e-4, rtol=1.0 / 128)},
        compile_only=args.compile_only,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
