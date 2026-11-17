# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""W8A8 MLA projections with query-latent/KV RMSNorm and BF16 interleaved RoPE."""

import pypto.language as pl
import torch

from config import FLASH as M, LOCAL_HEADS, T_DYN
from gemm_w8a8 import w8a8_gemm
from quant import quantize_int8_per_token
from rmsnorm import rms_norm
from rope import rope_tail_bf16


# Dynamic shape variable.

# model config
D = M.hidden_size
Q_LORA = M.q_lora_rank
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
HEADS = LOCAL_HEADS

# tiling
T_TILE = 8


@pl.jit.inline(auto_scope=False)
def mla_proj(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[Q_LORA], pl.FP32],
    q_norm_w: pl.Tensor[[Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, HEADS * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[HEADS * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[HEAD_DIM], pl.FP32],
    kv_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_HALF], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_HALF], pl.FP32],
    q: pl.Tensor[[T_DYN, HEADS, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
):
    """Project one token block into MLA query, KV latent, and quantized query latent.

    ``qr_scale`` is the per-token dequantization scale of ``qr``; consumers that
    feed the query latent into another W8A8 GEMM (the lightning indexer) must use
    it rather than the input activation scale.
    """
    t_dim = pl.tensor.dim(x, 0)
    x_i8 = pl.create_tensor([t_dim, D], dtype=pl.INT8)
    x_scale = pl.create_tensor([t_dim, 1], dtype=pl.FP32)
    quantize_int8_per_token(x, x_i8, x_scale, width=D)

    qr_bf16 = pl.create_tensor([t_dim, Q_LORA], dtype=pl.BF16)
    w8a8_gemm(x_i8, x_scale, wq_a, wq_a_scale, qr_bf16, width=D, output_width=Q_LORA, output_fp32=False)
    qr_norm = pl.create_tensor([t_dim, Q_LORA], dtype=pl.BF16)
    rms_norm(qr_bf16, q_norm_w, qr_norm, width=Q_LORA, output_fp32=False, column_tile=256)
    quantize_int8_per_token(qr_norm, qr, qr_scale, width=Q_LORA)

    q_bf16 = pl.create_tensor([t_dim, HEADS * HEAD_DIM], dtype=pl.BF16)
    w8a8_gemm(qr, qr_scale, wq_b, wq_b_scale, q_bf16, width=Q_LORA, output_width=HEADS * HEAD_DIM, output_fp32=False)
    q_out = pl.reshape(q, [t_dim, HEADS * HEAD_DIM])
    rope_tail_bf16(q_bf16, rope_cos, rope_sin, q_out, width=HEAD_DIM, heads=HEADS)

    kv_bf16 = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    w8a8_gemm(x_i8, x_scale, wkv, wkv_scale, kv_bf16, width=D, output_width=HEAD_DIM, output_fp32=False)
    kv_norm = pl.create_tensor([t_dim, HEAD_DIM], dtype=pl.BF16)
    rms_norm(kv_bf16, kv_norm_w, kv_norm, width=HEAD_DIM, output_fp32=False, column_tile=256)
    rope_tail_bf16(kv_norm, rope_cos, rope_sin, kv, width=HEAD_DIM, heads=1)
    return q, kv, qr, qr_scale


@pl.jit
def mla_proj_test(
    x: pl.Tensor[[T_DYN, 5120], pl.BF16],
    wq_a: pl.Tensor[[5120, 1280], pl.INT8, pl.NZ],
    wq_a_scale: pl.Tensor[[1280], pl.FP32],
    q_norm_w: pl.Tensor[[1280], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, HEADS * HEAD_DIM], pl.INT8, pl.NZ],
    wq_b_scale: pl.Tensor[[HEADS * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[5120, 512], pl.INT8, pl.NZ],
    wkv_scale: pl.Tensor[[512], pl.FP32],
    kv_norm_w: pl.Tensor[[512], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, 32], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, 32], pl.FP32],
    q: pl.Out[pl.Tensor[[T_DYN, LOCAL_HEADS, 512], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, 512], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_DYN, 1280], pl.INT8]],
    qr_scale: pl.Out[pl.Tensor[[T_DYN, 1], pl.FP32]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    q.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    qr_scale.bind_dynamic(0, T_DYN)
    mla_proj(
        x,
        wq_a, wq_a_scale, q_norm_w,
        wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_w,
        rope_cos, rope_sin,
        q, kv, qr, qr_scale,
    )
    return q, kv, qr, qr_scale


def _golden_quant_per_token(x):
    from utils import int8_quant_per_token

    return int8_quant_per_token(x)


def _golden_mla_proj(tensors):
    from utils import unpack_nz

    x = tensors["x"].float()
    wq_a = unpack_nz(tensors["wq_a"]).float()
    wq_a_scale = tensors["wq_a_scale"].float()
    q_norm_w = tensors["q_norm_w"].float()
    wq_b = unpack_nz(tensors["wq_b"]).float()
    wq_b_scale = tensors["wq_b_scale"].float()
    wkv = unpack_nz(tensors["wkv"]).float()
    wkv_scale = tensors["wkv_scale"].float()
    kv_norm_w = tensors["kv_norm_w"].float()
    cos = tensors["rope_cos"].float()
    sin = tensors["rope_sin"].float()
    eps = M.rms_norm_eps

    def rms(value, gamma, width):
        inv = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
        return (value * inv * gamma).to(torch.bfloat16).float()

    def gemm(x_i8, x_scale, weight, weight_scale):
        acc = torch.matmul(x_i8.to(torch.int32), weight.to(torch.int32))
        return (acc.float() * x_scale * weight_scale.reshape(1, -1)).to(torch.bfloat16).float()

    def rope(value, cos_v, sin_v):
        pairs = value.unflatten(-1, (-1, 2))
        even, odd = pairs[..., 0], pairs[..., 1]
        cos_h = cos_v[..., :ROPE_HALF]
        sin_h = sin_v[..., :ROPE_HALF]
        while cos_h.ndim < even.ndim:
            cos_h = cos_h.unsqueeze(-2)
            sin_h = sin_h.unsqueeze(-2)
        real = (even * cos_h - odd * sin_h).to(torch.bfloat16)
        imag = (even * sin_h + odd * cos_h).to(torch.bfloat16)
        return torch.stack([real, imag], dim=-1).flatten(-2)

    x_i8, x_scale = _golden_quant_per_token(tensors["x"])
    qr_bf16 = gemm(x_i8, x_scale, wq_a, wq_a_scale)
    qr_norm = rms(qr_bf16, q_norm_w, Q_LORA)
    qr_i8, qr_scale = _golden_quant_per_token(qr_norm.to(torch.bfloat16))
    q_full = gemm(qr_i8, qr_scale, wq_b, wq_b_scale).view(-1, HEADS, HEAD_DIM)
    q_nope = q_full[..., :NOPE_DIM]
    q_rope = rope(q_full[..., NOPE_DIM:], cos.unsqueeze(1), sin.unsqueeze(1))
    tensors["q"][:] = torch.cat([q_nope, q_rope], dim=-1).to(torch.bfloat16)

    kv_bf16 = gemm(x_i8, x_scale, wkv, wkv_scale)
    kv_norm = rms(kv_bf16, kv_norm_w, HEAD_DIM)
    kv_rope = rope(kv_norm[..., NOPE_DIM:].unsqueeze(1), cos.unsqueeze(1), sin.unsqueeze(1)).squeeze(1)
    tensors["kv"][:] = torch.cat([kv_norm[..., :NOPE_DIM], kv_rope], dim=-1).to(torch.bfloat16)
    tensors["qr"][:] = qr_i8
    tensors["qr_scale"][:] = qr_scale


def _rope_tables(tokens: int):
    positions = torch.arange(tokens, dtype=torch.float32)
    exponent = torch.arange(0, ROPE_HALF, dtype=torch.float32) * 2.0 / ROPE_DIM
    inv_freq = 1.0 / torch.pow(torch.tensor(M.rope_theta, dtype=torch.float32), exponent)
    angle = positions.reshape(-1, 1) * inv_freq.reshape(1, -1)
    return torch.cos(angle), torch.sin(angle)


def build_tensor_specs(tokens: int = 32):
    from golden import TensorSpec

    def quantized(k: int, n: int):
        from utils import pack_nz, quant_w_per_channel

        return lambda: pack_nz(quant_w_per_channel(torch.randn(n, k) * 0.05)[0].transpose(0, 1).contiguous())

    def scale_of(k: int, n: int):
        from utils import quant_w_per_channel

        return lambda: quant_w_per_channel(torch.randn(n, k) * 0.05)[1]

    cos, sin = _rope_tables(tokens)
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=torch.randn),
        TensorSpec("wq_a", [D, Q_LORA], torch.int8, init_value=quantized(D, Q_LORA)),
        TensorSpec("wq_a_scale", [Q_LORA], torch.float32, init_value=scale_of(D, Q_LORA)),
        TensorSpec("q_norm_w", [Q_LORA], torch.bfloat16, init_value=lambda: torch.randn(Q_LORA) * 0.1 + 1.0),
        TensorSpec("wq_b", [Q_LORA, HEADS * HEAD_DIM], torch.int8, init_value=quantized(Q_LORA, HEADS * HEAD_DIM)),
        TensorSpec("wq_b_scale", [HEADS * HEAD_DIM], torch.float32, init_value=scale_of(Q_LORA, HEADS * HEAD_DIM)),
        TensorSpec("wkv", [D, HEAD_DIM], torch.int8, init_value=quantized(D, HEAD_DIM)),
        TensorSpec("wkv_scale", [HEAD_DIM], torch.float32, init_value=scale_of(D, HEAD_DIM)),
        TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=lambda: torch.randn(HEAD_DIM) * 0.1 + 1.0),
        TensorSpec("rope_cos", [tokens, ROPE_HALF], torch.float32, init_value=cos),
        TensorSpec("rope_sin", [tokens, ROPE_HALF], torch.float32, init_value=sin),
        TensorSpec("q", [tokens, HEADS, HEAD_DIM], torch.bfloat16),
        TensorSpec("kv", [tokens, HEAD_DIM], torch.bfloat16),
        TensorSpec("qr", [tokens, Q_LORA], torch.int8),
        TensorSpec("qr_scale", [tokens, 1], torch.float32),
    ]


def build_semantic_tensor_specs(tokens: int = 7):
    """Sparse signed projections with unit latent and distinct learned head magnitudes."""
    from golden import TensorSpec
    from utils import pack_nz

    sign = (1 - 2 * (torch.arange(tokens) % 2)).to(torch.bfloat16).reshape(-1, 1)
    columns = torch.arange(HEADS * HEAD_DIM)
    coefficients = (1 - 2 * (columns % 3 == 0)).to(torch.int8)
    scales = ((columns % 29 + 3) * (columns // HEAD_DIM % 4 + 1)).float() / 4096
    wq_a = torch.zeros(D, Q_LORA, dtype=torch.int8)
    wq_a[torch.arange(Q_LORA), torch.arange(Q_LORA)] = 127
    wq_b = torch.zeros(Q_LORA, HEADS * HEAD_DIM, dtype=torch.int8)
    wq_b[columns % Q_LORA, columns] = coefficients * 127
    wkv = torch.zeros(D, HEAD_DIM, dtype=torch.int8)
    wkv[torch.arange(HEAD_DIM), torch.arange(HEAD_DIM)] = 127
    angles = (torch.arange(tokens).float() + 17).reshape(-1, 1)
    angles = angles * (torch.arange(ROPE_HALF).float() + 1).reshape(1, -1) / 37
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=sign.expand(tokens, D).contiguous()),
        TensorSpec("wq_a", [D, Q_LORA], torch.int8, init_value=pack_nz(wq_a)),
        TensorSpec("wq_a_scale", [Q_LORA], torch.float32, init_value=1 / 127),
        TensorSpec("q_norm_w", [Q_LORA], torch.bfloat16, init_value=1),
        TensorSpec("wq_b", [Q_LORA, HEADS * HEAD_DIM], torch.int8, init_value=pack_nz(wq_b)),
        TensorSpec("wq_b_scale", [HEADS * HEAD_DIM], torch.float32, init_value=scales),
        TensorSpec("wkv", [D, HEAD_DIM], torch.int8, init_value=pack_nz(wkv)),
        TensorSpec("wkv_scale", [HEAD_DIM], torch.float32, init_value=1 / 127),
        TensorSpec("kv_norm_w", [HEAD_DIM], torch.bfloat16, init_value=1),
        TensorSpec("rope_cos", [tokens, ROPE_HALF], torch.float32, init_value=angles.cos()),
        TensorSpec("rope_sin", [tokens, ROPE_HALF], torch.float32, init_value=angles.sin()),
        TensorSpec("q", [tokens, HEADS, HEAD_DIM], torch.bfloat16),
        TensorSpec("kv", [tokens, HEAD_DIM], torch.bfloat16),
        TensorSpec("qr", [tokens, Q_LORA], torch.int8),
        TensorSpec("qr_scale", [tokens, 1], torch.float32),
    ]


def golden_mla_semantic(tensors):
    """Closed-form sparse fixture oracle, independent of GEMM/quantization/RMS helpers.

    The input and both normalized latents are signed units in BF16. Each Q
    column has one signed 127 coefficient, so its pre-RoPE value is exactly
    BF16(sign * 127 * scale). This oracle only applies to the semantic fixture.
    """
    sign = tensors["x"][:, :1].float()
    columns = torch.arange(HEADS * HEAD_DIM)
    coefficients = (1 - 2 * (columns % 3 == 0)).float()
    projected = sign * coefficients * (127 * tensors["wq_b_scale"])
    query = projected.to(torch.bfloat16).reshape(-1, HEADS, HEAD_DIM)
    latent = sign.expand(-1, HEAD_DIM).to(torch.bfloat16)
    phase = torch.complex(tensors["rope_cos"], tensors["rope_sin"])
    for value in (query, latent):
        pairs = value[..., NOPE_DIM:].float().reshape(*value.shape[:-1], ROPE_HALF, 2)
        complex_tail = torch.view_as_complex(pairs.contiguous())
        multiplier = phase.unsqueeze(1) if value.ndim == 3 else phase
        rotated = torch.view_as_real(complex_tail * multiplier).flatten(-2)
        value[..., NOPE_DIM:] = rotated.to(torch.bfloat16)
    tensors["q"][:] = query
    tensors["kv"][:] = latent
    tensors["qr"][:] = (sign * 127).to(torch.int8).expand_as(tensors["qr"])
    tensors["qr_scale"][:] = 1 / 127


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser()
    from config import add_parallel_arguments
    add_parallel_arguments(parser)
    parser.add_argument("-p", "--platform", dest="backend_type", choices=["a2a3", "a5"], default="a2a3")
    parser.add_argument("--sim", dest="execution_mode", action="store_true")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    result = run(
        fn=mla_proj_test,
        specs=build_tensor_specs(args.tokens),
        golden_fn=_golden_mla_proj,
        config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
        rtol=1e-2,
        atol=1e-2,
        compare_fn={
            "q": ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2),
            "kv": ratio_allclose(atol=2e-2, rtol=1.0 / 64, max_error_ratio=1e-2),
            "qr": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=1e-2),
        },
    )
    if not result.passed:
        if result.error:
            print(result.error)
        raise SystemExit(1)

    for semantic_tokens in (1, 7):
        print(f"[CASE] mla original-semantics signed-unit T={semantic_tokens}", flush=True)
        result = run(
            fn=mla_proj_test,
            specs=build_semantic_tensor_specs(semantic_tokens),
            golden_fn=golden_mla_semantic,
            config=dict(backend_type=args.backend_type, execution_mode=args.execution_mode, device_id=args.device),
            rtol=1e-5,
            atol=1e-5,
            compare_fn={name: ratio_allclose(atol=0, rtol=0, max_error_ratio=0) for name in ("q", "kv", "qr")},
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
