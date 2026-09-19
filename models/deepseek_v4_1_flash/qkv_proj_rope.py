# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""V4.1 Q/KV preprocessing with caller-owned scratch and no cache or TP communication."""

import pypto.language as pl

from models.deepseek_v4_1_flash.config import D, FLASH, HEAD_DIM, LOCAL_H, NOPE_DIM, Q_LORA, ROPE_DIM, T_DYN

EPS = FLASH.rms_norm_eps
MX_M_TILE = 32
N_TILE = 128
K_TILE = 256
WORKER_TILE = 64
PROJECTION_K_TILE = 32

def make_projection(width, output_width, output_dtype=pl.BF16):
    """Specialize an MXFP8 projection without expanding weights in HBM."""
    fp32_output = output_dtype == pl.FP32

    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP8E4M3FN],
        scale: pl.Tensor[[width // 32, output_width], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, output_width], output_dtype],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            for block in pl.spmd(output_width // N_TILE, name_hint="swa_mx_projection"):
                n0 = block * N_TILE
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                first = pl.load(x, [t0, 0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                # s = 2**ceil(log2(max(amax, 1e-4) / 448)); native quant_mx uses a different rule.
                firstq_values = pl.reshape(pl.cast(first, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                firstq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                firstq_maximum = pl.maximum(pl.row_max(pl.abs(firstq_values), tmp_tile=firstq_reduce_tmp), 1e-4)
                firstq_bits = pl.reinterpret_view(pl.mul(firstq_maximum, 1.0 / 448.0), pl.INT32)
                firstq_exponent = pl.shrs(pl.add(firstq_bits, 8388607), 23)
                firstq_scale = pl.reinterpret_view(pl.shls(firstq_exponent, 23), pl.FP32)
                firstq_quantized = pl.cast(pl.row_expand_div(firstq_values, firstq_scale), pl.FP8E4M3FN, mode="rint")
                firstq_payload = pl.reshape(firstq_quantized, [MX_M_TILE, K_TILE])
                firstq_signed_exponent = pl.sub(firstq_exponent, pl.mul(pl.shrs(firstq_exponent, 7), 256))
                firstq_codes = pl.reinterpret_view(pl.cast(firstq_signed_exponent, pl.INT8), pl.UINT8)
                firstq_flat = pl.reshape(firstq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                firstq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                firstq_packed = pl.tmov_x2zz(firstq_flat, firstq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8)
                a0 = firstq_payload
                sa0 = pl.reinterpret_view(firstq_packed, pl.FP8E8M0)
                b0 = pl.load(weight, [0, n0], [K_TILE, N_TILE])
                sb0 = pl.load(scale, [0, n0], [K_TILE // 32, N_TILE])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for kb in pl.range(1, width // K_TILE):
                    k0 = kb * K_TILE
                    values = pl.load(x, [t0, k0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                    values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                    nextq_values = pl.reshape(pl.cast(values, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                    nextq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                    nextq_maximum = pl.maximum(pl.row_max(pl.abs(nextq_values), tmp_tile=nextq_reduce_tmp), 1e-4)
                    nextq_bits = pl.reinterpret_view(pl.mul(nextq_maximum, 1.0 / 448.0), pl.INT32)
                    nextq_exponent = pl.shrs(pl.add(nextq_bits, 8388607), 23)
                    nextq_scale = pl.reinterpret_view(pl.shls(nextq_exponent, 23), pl.FP32)
                    nextq_quantized = pl.cast(pl.row_expand_div(nextq_values, nextq_scale), pl.FP8E4M3FN, mode="rint")
                    nextq_payload = pl.reshape(nextq_quantized, [MX_M_TILE, K_TILE])
                    nextq_signed_exponent = pl.sub(nextq_exponent, pl.mul(pl.shrs(nextq_exponent, 7), 256))
                    nextq_codes = pl.reinterpret_view(pl.cast(nextq_signed_exponent, pl.INT8), pl.UINT8)
                    nextq_flat = pl.reshape(nextq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                    nextq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                    nextq_packed = pl.tmov_x2zz(nextq_flat, nextq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8)
                    a = nextq_payload
                    sa = pl.reinterpret_view(nextq_packed, pl.FP8E8M0)
                    b = pl.load(weight, [k0, n0], [K_TILE, N_TILE])
                    sb = pl.load(scale, [k0 // 32, n0], [K_TILE // 32, N_TILE])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                if fp32_output:
                    output = pl.store(pl.set_validshape(pl.mul(acc, 1.0), rows, N_TILE), [t0, n0], output)
                else:
                    value = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    output = pl.store(pl.set_validshape(value, rows, N_TILE), [t0, n0], output)
        return output

    return project


def make_norm(width):
    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        if width == D:
            # Chunk the hidden dimension to retain aligned row reductions within Vec capacity.
            for block in pl.spmd((num_tokens + 7) // 8, name_hint="swa_hidden_rmsnorm"):
                t = block * 8
                rows = pl.min(8, num_tokens - t)
                square_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
                for chunk in pl.pipeline(width // 128, stage=2):
                    d0 = chunk * 128
                    source_chunk = pl.slice(x, [8, 128], [t, d0], valid_shape=[rows, 128])
                    source_chunk = pl.set_validshape(pl.fillpad(source_chunk, pad_value=pl.PadValue.zero), 8, 128)
                    value_chunk = pl.cast(source_chunk, pl.FP32)
                    chunk_sum = pl.reshape(pl.row_sum(pl.mul(value_chunk, value_chunk)), [1, 8])
                    square_sum = pl.add(square_sum, chunk_sum)
                inverse = pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / width), EPS), high_precision=True)
                inverse_col = pl.reshape(inverse, [8, 1])
                for chunk in pl.pipeline(width // 128, stage=2):
                    d0 = chunk * 128
                    source_chunk = pl.slice(x, [8, 128], [t, d0], valid_shape=[rows, 128])
                    source_chunk = pl.set_validshape(pl.fillpad(source_chunk, pad_value=pl.PadValue.zero), 8, 128)
                    value_chunk = pl.cast(source_chunk, pl.FP32)
                    gamma_chunk = pl.reshape(pl.cast(weight[d0:d0 + 128], pl.FP32), [1, 128])
                    result_chunk = pl.col_expand_mul(pl.row_expand_mul(value_chunk, inverse_col), gamma_chunk)
                    output[t:t + 8, d0:d0 + 128] = pl.set_validshape(pl.cast(result_chunk, pl.BF16, mode="rint"), rows, 128)
        else:
            for block in pl.spmd((num_tokens + 7) // 8, name_hint="swa_rmsnorm"):
                t = block * 8
                rows = pl.min(8, num_tokens - t)
                source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
                value = pl.cast(source, pl.FP32)
                inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / width),
                                     EPS), high_precision=True)
                gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
                normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
                output[t:t + 8, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, width)
        return output

    return normalize


def make_rope(heads, inverse=False):
    sign = -1.0 if inverse else 1.0

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for block in pl.spmd(num_tokens * heads, name_hint="swa_rope"):
            t = block // heads
            h = block % heads
            base = h * HEAD_DIM
            output[t:t + 1, base:base + NOPE_DIM] = x[t:t + 1, base:base + NOPE_DIM]
            tail = pl.cast(x[t:t + 1, base + NOPE_DIM:base + HEAD_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            c = cos[t:t + 1, :]
            s = pl.mul(sin[t:t + 1, :], sign)
            re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
            im = pl.add(pl.mul(even, s), pl.mul(odd, c))
            rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[t:t + 1, base + NOPE_DIM:base + HEAD_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rotate


project_qa = make_projection(D, Q_LORA)
project_qb = make_projection(Q_LORA, LOCAL_H * HEAD_DIM)
project_kv = make_projection(D, HEAD_DIM)
normalize_q = make_norm(Q_LORA)
normalize_kv = make_norm(HEAD_DIM)
rotate_q = make_rope(LOCAL_H)
rotate_kv = make_rope(1)


@pl.jit.inline
def prefill_project_qa(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    weight: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Accumulate scale-corrected group-32 QLoRA products in FP32."""
    scale_storage = pl.tensor.view(scale, [Q_LORA // 16, D // 2], layout=pl.ND)
    for mt in pl.parallel((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
        t0 = mt * MX_M_TILE
        for block in pl.spmd(Q_LORA // N_TILE, name_hint="prefill_swa_q_lora"):
            n0 = block * N_TILE
            rows = pl.min(MX_M_TILE, num_tokens - t0)
            acc = pl.tile.full([MX_M_TILE, N_TILE], dtype=pl.FP32, value=0.0)
            for kb in pl.range(D // (2 * PROJECTION_K_TILE)):
                # MX_B_NN stores adjacent K-group scales in interleaved pairs.
                raw = pl.load(scale_storage, [n0 // 16, kb * 32], [N_TILE // 16, 32])
                raw_u8 = pl.reinterpret_view(raw, pl.UINT8)
                codes = pl.cast(pl.reinterpret_view(raw_u8, pl.INT8), pl.INT32)
                codes = pl.ands(codes, 255)
                scale_bits = pl.maximum(pl.shls(codes, 23), 4194304)
                scale_pair = pl.reinterpret_view(scale_bits, pl.FP32)
                for half in pl.unroll(2):
                    k0 = (kb * 2 + half) * PROJECTION_K_TILE
                    if half == 0:
                        gathered_scale = pl.tile.gather_mask(scale_pair, mask_pattern=pl.tile.MaskPattern.P0101)
                    else:
                        gathered_scale = pl.tile.gather_mask(scale_pair, mask_pattern=pl.tile.MaskPattern.P1010)
                    sb = pl.reshape(gathered_scale, [1, N_TILE])
                    source = pl.load(x, [t0, k0], [MX_M_TILE, PROJECTION_K_TILE], valid_shape=[rows, PROJECTION_K_TILE])
                    source = pl.fillpad(source, pad_value=pl.PadValue.zero)
                    source = pl.set_validshape(source, MX_M_TILE, PROJECTION_K_TILE)
                    value = pl.cast(source, pl.FP32)
                    reduce_tmp = pl.create_tile([MX_M_TILE, PROJECTION_K_TILE], dtype=pl.FP32)
                    maximum = pl.row_max(pl.abs(value), tmp_tile=reduce_tmp)
                    maximum = pl.maximum(maximum, 1e-4)
                    bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                    exponent = pl.shrs(pl.add(bits, 8388607), 23)
                    sa = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                    quantized = pl.row_expand_div(value, sa)
                    payload = pl.cast(quantized, pl.FP8E4M3FN, mode="rint")
                    a = pl.cast(payload, pl.BF16)
                    weight_payload = pl.load(weight, [k0, n0], [PROJECTION_K_TILE, N_TILE])
                    b = pl.cast(weight_payload, pl.BF16)
                    dot = pl.matmul(a, b)
                    part = pl.row_expand_mul(dot, sa)
                    part = pl.col_expand_mul(part, sb)
                    acc = pl.add(acc, part)
            result = pl.cast(acc, pl.BF16, mode="rint")
            result = pl.set_validshape(result, rows, N_TILE)
            output = pl.store(result, [t0, n0], output)
    return output


def make_prefill_norm(width):
    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for worker in pl.spmd(WORKER_TILE, name_hint="prefill_swa_rmsnorm"):
            for block in pl.range(worker, (num_tokens + 7) // 8, WORKER_TILE):
                t = block * 8
                rows = pl.min(8, num_tokens - t)
                source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
                value = pl.cast(source, pl.FP32)
                square_sum = pl.row_sum(pl.mul(value, value))
                variance = pl.add(pl.mul(square_sum, 1.0 / width), EPS)
                inv = pl.rsqrt(variance, high_precision=True)
                gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
                normalized = pl.col_expand_mul(pl.row_expand_mul(value, inv), gamma)
                output[t:t + 8, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, width)
        return output

    return normalize


def make_prefill_rope(heads, inverse=False):
    sign = -1.0 if inverse else 1.0

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        for worker in pl.spmd(WORKER_TILE, name_hint="prefill_swa_rope"):
            for block in pl.range(worker, num_tokens * heads, WORKER_TILE):
                t = block // heads
                h = block % heads
                base = h * HEAD_DIM
                output[t:t + 1, base:base + NOPE_DIM] = x[t:t + 1, base:base + NOPE_DIM]
                tail = pl.cast(x[t:t + 1, base + NOPE_DIM:base + HEAD_DIM], pl.FP32)
                even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
                odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
                c = cos[t:t + 1, :]
                s = pl.mul(sin[t:t + 1, :], sign)
                re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
                im = pl.add(pl.mul(even, s), pl.mul(odd, c))
                rotated = pl.full([1, ROPE_DIM], dtype=pl.FP32, value=0.0)
                rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
                rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
                output[t:t + 1, base + NOPE_DIM:base + HEAD_DIM] = pl.cast(rotated, pl.BF16, mode="rint")
        return output

    return rotate


prefill_normalize_q = make_prefill_norm(Q_LORA)
prefill_normalize_kv = make_prefill_norm(HEAD_DIM)
prefill_rotate_q = make_prefill_rope(LOCAL_H)
prefill_rotate_kv = make_prefill_rope(1)


def make_qkv_kernels(project_latent, normalize_latent, rotate_query, normalize_window, rotate_window):
    """Keep caller-owned scratch and the prefill numerical/scheduling specialization."""

    @pl.jit.inline(auto_scope=False)
    def q_proj_qr(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        weight: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        projected: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        normalized: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        project_latent(x, weight, scale, projected, num_tokens)
        normalize_latent(projected, norm_weight, normalized, num_tokens)
        return normalized

    @pl.jit.inline(auto_scope=False)
    def q_proj_rope(
        normalized: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
        weight: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        projected: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        query: pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        project_qb(normalized, weight, scale, projected, num_tokens)
        rotate_query(projected, cos, sin, query, num_tokens)
        return query

    @pl.jit.inline(auto_scope=False)
    def kv_proj_rope(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        weight: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        projected: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
        normalized: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
        window_kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        project_kv(x, weight, scale, projected, num_tokens)
        normalize_window(projected, norm_weight, normalized, num_tokens)
        rotate_window(normalized, cos, sin, window_kv, num_tokens)
        return window_kv

    return q_proj_qr, q_proj_rope, kv_proj_rope


q_proj_qr, q_proj_rope, kv_proj_rope = make_qkv_kernels(
    project_qa, normalize_q, rotate_q, normalize_kv, rotate_kv,
)
prefill_q_proj_qr, prefill_q_proj_rope, prefill_kv_proj_rope = make_qkv_kernels(
    prefill_project_qa, prefill_normalize_q, prefill_rotate_q, prefill_normalize_kv, prefill_rotate_kv,
)
