# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Inverse rotary embedding and grouped BF16 output projection of attended heads."""

import pypto.language as pl
import torch

from config import FLASH as M, LOCAL_HEADS, LOCAL_O_GROUPS, T_DYN
from rope import rope_tail_bf16


# Dynamic shape variable.

# model config
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_HALF = M.qk_rope_head_dim // 2
HEADS_PER_GROUP = M.num_attention_heads // M.o_groups
GROUP_IN = HEADS_PER_GROUP * M.head_dim
O_LORA = M.o_lora_rank
LOCAL_O_WIDTH = LOCAL_O_GROUPS * O_LORA
LOCAL_ATTENDED = LOCAL_O_GROUPS * GROUP_IN

# tiling
T_TILE = 16
K_TILE = 128
N_TILE = 256
OUT_N_TILE = 512


@pl.jit.inline(auto_scope=False)
def o_proj(
    attended: pl.Tensor,
    wo_a: pl.Tensor,
    wo_b: pl.Tensor,
    cos: pl.Tensor,
    sin: pl.Tensor,
    output: pl.Tensor,
    groups: pl.constexpr,
    group_in: pl.constexpr,
    o_lora: pl.constexpr,
):
    """Inverse-rotate attended heads, project groups to LoRA width, then project to hidden."""
    t_dim = pl.tensor.dim(attended, 0)
    inverse_sin = pl.create_tensor([t_dim, ROPE_HALF], dtype=pl.FP32)
    for block in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="output_inverse_sine"):
        t0 = block * T_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        sine = pl.slice(sin, [T_TILE, ROPE_HALF], [t0, 0], valid_shape=[rows, ROPE_HALF])
        sine_neg = pl.neg(sine)
        inverse_sin[t0 : t0 + T_TILE, 0:ROPE_HALF] = sine_neg
    restored = pl.create_tensor([t_dim, groups * group_in], dtype=pl.BF16)
    restored = rope_tail_bf16(attended, cos, inverse_sin, restored, width=M.head_dim, heads=groups * group_in // M.head_dim)
    blocks = (t_dim + T_TILE - 1) // T_TILE
    mid = pl.create_tensor([t_dim, groups * o_lora], dtype=pl.BF16)
    for group in pl.range(groups):
        group_col = group * group_in
        mid_col = group * o_lora
        for block in pl.spmd(blocks, name_hint="o_proj_a"):
            t0 = block * T_TILE
            rows = pl.min(T_TILE, t_dim - t0)
            for n_block in pl.unroll(o_lora // N_TILE):
                n0 = n_block * N_TILE
                acc = pl.create_tensor([T_TILE, N_TILE], dtype=pl.FP32)
                for k0 in pl.pipeline(0, group_in, K_TILE, stage=2):
                    x_tile = pl.slice(restored, [T_TILE, K_TILE], [t0, group_col + k0], valid_shape=[rows, K_TILE])
                    x_tile = pl.set_validshape(pl.fillpad(x_tile, pad_value=pl.PadValue.zero), T_TILE, K_TILE)
                    w_tile_3d = pl.slice(wo_a, [1, N_TILE, K_TILE], [group, n0, k0])
                    w_tile = pl.reshape(w_tile_3d, [N_TILE, K_TILE])
                    acc = pl.matmul_acc(acc, x_tile, w_tile, b_trans=True, init_cond=(k0 == 0))
                rounded = pl.cast(acc, target_type=pl.BF16, mode="rint")
                mid[t0 : t0 + T_TILE, mid_col + n0 : mid_col + n0 + N_TILE] = pl.set_validshape(rounded, rows, N_TILE)
    # The token axis is scheduled through the SPMD block count (dynamic dim
    # outermost, constant divisors) instead of a sequential dynamic range: a
    # dynamic `pl.range` here carries a T_DYN iter_arg that the type checker
    # cannot compose with an inline MoE (KNOWN_PYPTO_ISSUES.md).
    n_tiles = D // OUT_N_TILE
    for fused in pl.spmd(blocks * n_tiles, name_hint="o_proj_b"):
        t_block = fused // n_tiles
        n_block = fused % n_tiles
        t0 = t_block * T_TILE
        n0 = n_block * OUT_N_TILE
        rows = pl.min(T_TILE, t_dim - t0)
        acc_b = pl.create_tensor([T_TILE, OUT_N_TILE], dtype=pl.FP32)
        for k0 in pl.pipeline(0, groups * o_lora, K_TILE, stage=2):
            x_tile_b = pl.slice(mid, [T_TILE, K_TILE], [t0, k0], valid_shape=[rows, K_TILE])
            x_tile_b = pl.set_validshape(pl.fillpad(x_tile_b, pad_value=pl.PadValue.zero), T_TILE, K_TILE)
            w_tile_b = pl.slice(wo_b, [OUT_N_TILE, K_TILE], [n0, k0])
            acc_b = pl.matmul_acc(acc_b, x_tile_b, w_tile_b, b_trans=True, init_cond=(k0 == 0))
        rounded_b = pl.cast(acc_b, target_type=pl.BF16, mode="rint")
        output[t0 : t0 + T_TILE, n0 : n0 + OUT_N_TILE] = pl.set_validshape(rounded_b, rows, OUT_N_TILE)
    return output


@pl.jit
def o_proj_test(
    attended: pl.Tensor[[T_DYN, LOCAL_ATTENDED], pl.BF16],
    wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[D, LOCAL_O_WIDTH], pl.BF16],
    cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    output: pl.Out[pl.Tensor[[T_DYN, 5120], pl.BF16]],
):
    attended.bind_dynamic(0, T_DYN)
    cos.bind_dynamic(0, T_DYN)
    sin.bind_dynamic(0, T_DYN)
    output.bind_dynamic(0, T_DYN)
    o_proj(attended, wo_a, wo_b, cos, sin, output, groups=LOCAL_O_GROUPS, group_in=GROUP_IN, o_lora=O_LORA)
    return output


def golden_o_proj(tensors):
    """Inverse-rotate BF16 heads before the two-stage grouped output projection."""
    heads = tensors["attended"].to(torch.bfloat16).float().reshape(-1, LOCAL_ATTENDED // M.head_dim, M.head_dim)
    pairs = heads[..., -M.qk_rope_head_dim:].unflatten(-1, (-1, 2))
    cosine = tensors["cos"].float().unsqueeze(1)
    sine = tensors["sin"].float().unsqueeze(1)
    real = pairs[..., 0] * cosine + pairs[..., 1] * sine
    imag = pairs[..., 1] * cosine - pairs[..., 0] * sine
    tail = torch.stack([real, imag], dim=-1).flatten(-2)
    restored = torch.cat([heads[..., :-M.qk_rope_head_dim], tail], dim=-1).to(torch.bfloat16)
    attended = restored.float().view(-1, LOCAL_O_GROUPS, GROUP_IN)
    wo_a = tensors["wo_a"].float()
    wo_b = tensors["wo_b"].float()
    mid = torch.einsum("tgi,goi->tgo", attended, wo_a).to(torch.bfloat16).float().reshape(-1, LOCAL_O_WIDTH)
    tensors["output"][:] = torch.matmul(mid, wo_b.t()).to(torch.bfloat16)


def build_tensor_specs(tokens: int = 16):
    from golden import TensorSpec

    angles = (torch.arange(tokens, dtype=torch.float32)[:, None] + 1) * (torch.arange(32, dtype=torch.float32)[None, :] + 1) * 0.03125
    return [
        TensorSpec("attended", [tokens, LOCAL_ATTENDED], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wo_a", [LOCAL_O_GROUPS, O_LORA, GROUP_IN], torch.bfloat16, init_value=lambda: torch.randn(LOCAL_O_GROUPS, O_LORA, GROUP_IN) * 0.02),
        TensorSpec("wo_b", [D, LOCAL_O_WIDTH], torch.bfloat16, init_value=lambda: torch.randn(D, LOCAL_O_WIDTH) * 0.02),
        TensorSpec("cos", [tokens, 32], torch.float32, init_value=angles.cos()),
        TensorSpec("sin", [tokens, 32], torch.float32, init_value=angles.sin()),
        TensorSpec("output", [tokens, 5120], torch.bfloat16),
    ]


def build_semantic_fixture(tokens):
    """Signed attended heads, nonidentity angles and asymmetric sparse Wo."""
    t = torch.arange(tokens, dtype=torch.float32)[:, None]
    channels = torch.arange(LOCAL_ATTENDED, dtype=torch.float32)[None, :]
    attended = ((channels.remainder(23) - 11) * 0.09375 + (t.remainder(7) - 3) * 0.03125).to(torch.bfloat16)
    pair = torch.arange(32, dtype=torch.float32)[None, :]
    angles = 0.31 + t * 0.173 + pair * 0.027
    wa = torch.zeros(LOCAL_O_GROUPS, O_LORA, GROUP_IN, dtype=torch.bfloat16)
    wb = torch.zeros(D, LOCAL_O_WIDTH, dtype=torch.bfloat16)
    for g in range(LOCAL_O_GROUPS):
        for row in range(1024):
            head = row % 8
            p = row // 8 % 32
            even = head * 512 + 448 + 2 * p
            wa[g, row, even] = (1 if row % 3 else -1) * (0.375 + row % 5 * 0.0625 + g * 0.125)
            wa[g, row, even + 1] = (1 if row % 2 else -1) * (0.15625 + row % 7 * 0.03125)
            wa[g, row, head * 512 + row * 17 % 448] = 0.09375 * (g + 1)
    for row in range(5120):
        wb[row, row % LOCAL_O_WIDTH] = 0.5 + row % 5 * 0.125
        wb[row, (row * 13 + 37) % LOCAL_O_WIDTH] = -0.28125 - row % 3 * 0.03125
    return {'attended': attended, 'cos': angles.cos(), 'sin': angles.sin(), 'wo_a': wa, 'wo_b': wb}


def _semantic_restored(tensors, mode='inverse'):
    """Complex multiplication, with explicit BF16 destination rounding."""
    x = tensors['attended'].reshape(-1, LOCAL_HEADS, 512).float()
    if mode == 'missing':
        return x.to(torch.bfloat16)
    if mode not in ('inverse', 'forward'):
        raise ValueError(mode)
    pairs = x[..., 448:].reshape(-1, LOCAL_HEADS, 32, 2).contiguous()
    z = torch.view_as_complex(pairs)
    signed_sin = -tensors['sin'] if mode == 'inverse' else tensors['sin']
    phase = torch.complex(tensors['cos'], signed_sin)[:, None, :]
    rotated = torch.view_as_real(z * phase).flatten(-2)
    return torch.cat([x[..., :448], rotated], dim=-1).to(torch.bfloat16)


def golden_o_proj_semantic(tensors, mode='inverse'):
    """Sparse three-term Wo_a/two-term Wo_b closed form, no candidate calls."""
    tokens = tensors['attended'].shape[0]
    x = _semantic_restored(tensors, mode).float().reshape(tokens, LOCAL_O_GROUPS, GROUP_IN)
    mid = torch.empty(tokens, LOCAL_O_GROUPS, O_LORA, dtype=torch.float32)
    for g in range(LOCAL_O_GROUPS):
        wa = tensors['wo_a'][g].float()
        nz = wa.nonzero()
        counts = torch.bincount(nz[:, 0], minlength=1024)
        if not torch.equal(counts, torch.full_like(counts, 3)):
            raise ValueError('expected exactly three Wo_a terms per row')
        columns = nz[:, 1].reshape(1024, 3)
        coefficients = wa.gather(1, columns)
        terms = x[:, g, columns] * coefficients[None, :, :]
        mid[:, g] = terms.sum(-1)
    mid = mid.to(torch.bfloat16).float().reshape(tokens, LOCAL_O_WIDTH)
    wb = tensors['wo_b'].float()
    nz = wb.nonzero()
    counts = torch.bincount(nz[:, 0], minlength=5120)
    if not torch.equal(counts, torch.full_like(counts, 2)):
        raise ValueError('expected exactly two Wo_b terms per row')
    columns = nz[:, 1].reshape(5120, 2)
    coefficients = wb.gather(1, columns)
    tensors['output'][:] = (mid[:, columns] * coefficients[None, :, :]).sum(-1).to(torch.bfloat16)


def build_semantic_tensor_specs(tokens):
    from golden import TensorSpec

    fixture = build_semantic_fixture(tokens)
    specs = [TensorSpec(name, list(fixture[name].shape), fixture[name].dtype, init_value=fixture[name]) for name in ("attended", "wo_a", "wo_b", "cos", "sin")]
    specs.append(TensorSpec("output", [tokens, D], torch.bfloat16))
    return specs


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    result = run(
        fn=o_proj_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=golden_o_proj,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=2e-2,
        atol=2e-2,
        compare_fn={"output": ratio_allclose(atol=2e-2, rtol=2e-2, max_error_ratio=5e-3)},
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)


    for semantic_tokens in (1, 7, 17):
        print(f"[CASE] inverse output RoPE original-semantics sparse T={semantic_tokens}", flush=True)
        result = run(
            fn=o_proj_test,
            specs=build_semantic_tensor_specs(semantic_tokens),
            golden_fn=golden_o_proj_semantic,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=0,
            atol=0,
            compare_fn={"output": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
