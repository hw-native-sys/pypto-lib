# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""DeepSeek-V4 Q/KV LoRA + RoPE (dynamic shape): projects token-major
attention-normalized inputs for both decode and prefill attention paths."""


import pypto.language as pl

from config import ACTIVE as M, DECODE_BATCH, DECODE_SEQ, PREFILL_BATCH, PREFILL_SEQ


# Dynamic shape variables.
T_DYN = pl.dynamic("T_DYN")  # T = B * S


# model config
D = M.hidden_size
H = M.num_attention_heads
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
ROPE_HALF = ROPE_DIM // 2
NOPE_DIM = M.nope_head_dim
Q_LORA = M.q_lora_rank
EPS = M.rms_norm_eps
MAX_SEQ_LEN = M.max_position_embeddings
T_MAX = max(DECODE_BATCH * DECODE_SEQ, PREFILL_BATCH * PREFILL_SEQ)

# tiling
QUANT_TILE = 512
T_TILE = 8
KV_RMS_T_TILE = 8
Q_ROPE_H_TILE = 4



MX_M_TILE = 32
MX_K_TILE = 256
MX_N_TILE = 128
MX_QUANT_ROWS = 16


def make_mxfp8_projection(width, output_width):
    """Build a CANN-style native group-32 MXFP8 linear with BF16 output."""
    # Both loops below tile with floor division: a non-divisible output_width
    # would leave trailing columns unwritten and a non-divisible width would
    # drop the trailing K elements from the accumulation, both silently.
    assert width % MX_K_TILE == 0, f"MX projection K {width} % {MX_K_TILE}"
    assert output_width % MX_N_TILE == 0, f"MX projection N {output_width} % {MX_N_TILE}"
    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP8E4M3FN],
        scale: pl.Tensor[[width // 32, output_width], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        late_dep: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(x, 0)
        for mt in pl.parallel((tokens + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            with pl.spmd(output_width // MX_N_TILE, name_hint="mx_projection", deps=[late_dep]) as _project_tid:
                n0 = pl.tile.get_block_idx() * MX_N_TILE
                rows = pl.min(MX_M_TILE, tokens - t0)
                first = pl.load(x, [t0, 0], [MX_M_TILE, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, MX_K_TILE)
                a0, sa0 = pl.quant_mx(first, group_axis=1)
                b0 = pl.load(weight, [0, n0], [MX_K_TILE, MX_N_TILE])
                sb0 = pl.load(scale, [0, n0], [MX_K_TILE // 32, MX_N_TILE])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for kb in pl.range(1, width // MX_K_TILE):
                    k0 = kb * MX_K_TILE
                    values = pl.load(x, [t0, k0], [MX_M_TILE, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
                    values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), MX_M_TILE, MX_K_TILE)
                    aq, sa = pl.quant_mx(values, group_axis=1)
                    b = pl.load(weight, [k0, n0], [MX_K_TILE, MX_N_TILE])
                    sb = pl.load(scale, [k0 // 32, n0], [MX_K_TILE // 32, MX_N_TILE])
                    acc = pl.matmul_mx_acc(acc, aq, sa, b, sb)
                value = pl.cast(acc, target_type=pl.BF16, mode="rint")
                output = pl.store(pl.set_validshape(value, rows, MX_N_TILE), [t0, n0], output)
        return output

    return project


def make_mxfp8_projection_from_quantized(width, output_width):
    """Consume a quantized activation and its padded MX_A_ZZ scale bank."""
    assert width % MX_K_TILE == 0, f"MX projection K {width} % {MX_K_TILE}"
    assert output_width % MX_N_TILE == 0, f"MX projection N {output_width} % {MX_N_TILE}"
    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.FP8E4M3FN],
        x_scale: pl.Tensor[[T_MAX, width // 32], pl.FP8E8M0, pl.MX_A_ZZ],
        weight: pl.Tensor[[width, output_width], pl.FP8E4M3FN],
        scale: pl.Tensor[[width // 32, output_width], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        late_dep: pl.Scalar[pl.TASK_ID],
    ):
        tokens = pl.tensor.dim(x, 0)
        # Keep all token tiles in one SPMD group so its completion fences
        # every native MX task before ordinary BF16 matrix kernels start.
        with pl.spmd(output_width // MX_N_TILE, name_hint="mx_quantized_projection", deps=[late_dep]) as project_tid:
            n0 = pl.tile.get_block_idx() * MX_N_TILE
            for mt in pl.range((tokens + MX_M_TILE - 1) // MX_M_TILE):
                t0 = mt * MX_M_TILE
                rows = pl.min(MX_M_TILE, tokens - t0)
                a0 = pl.load(x, [t0, 0], [MX_M_TILE, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
                a0 = pl.set_validshape(pl.fillpad(a0, pad_value=pl.PadValue.zero), MX_M_TILE, MX_K_TILE)
                sa0 = pl.load(x_scale, [t0, 0], [MX_M_TILE, MX_K_TILE // 32])
                b0 = pl.load(weight, [0, n0], [MX_K_TILE, MX_N_TILE])
                sb0 = pl.load(scale, [0, n0], [MX_K_TILE // 32, MX_N_TILE])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for kb in pl.range(1, width // MX_K_TILE):
                    k0 = kb * MX_K_TILE
                    a = pl.load(x, [t0, k0], [MX_M_TILE, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
                    a = pl.set_validshape(pl.fillpad(a, pad_value=pl.PadValue.zero), MX_M_TILE, MX_K_TILE)
                    sa = pl.load(x_scale, [t0, k0 // 32], [MX_M_TILE, MX_K_TILE // 32])
                    b = pl.load(weight, [k0, n0], [MX_K_TILE, MX_N_TILE])
                    sb = pl.load(scale, [k0 // 32, n0], [MX_K_TILE // 32, MX_N_TILE])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                value = pl.cast(acc, target_type=pl.BF16, mode="rint")
                output = pl.store(pl.set_validshape(value, rows, MX_N_TILE), [t0, n0], output)
        return project_tid

    return project


@pl.jit.inline
def _normalize_qr(
    x: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    gamma: pl.Tensor[[Q_LORA], pl.BF16],
    output: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
):
    tokens = pl.tensor.dim(x, 0)
    for block in pl.spmd((tokens + 7) // 8, name_hint="qr_rmsnorm_bf16"):
        row = block * 8
        rows = pl.min(8, tokens - row)
        loaded = pl.slice(x, [8, Q_LORA], [row, 0], valid_shape=[rows, Q_LORA])
        loaded = pl.set_validshape(pl.fillpad(loaded, pad_value=pl.PadValue.zero), 8, Q_LORA)
        values = pl.cast(loaded, pl.FP32)
        inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(values, values)), 1.0 / Q_LORA), EPS), high_precision=True)
        weight = pl.reshape(pl.cast(gamma[:], pl.FP32), [1, Q_LORA])
        normalized = pl.col_expand_mul(pl.row_expand_mul(values, inv), weight)
        output[row:row + 8, :] = pl.set_validshape(pl.cast(normalized, pl.BF16, mode="rint"), rows, Q_LORA)
    return output


@pl.jit.inline
def _quantize_qr(
    x: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.FP8E4M3FN],
    qr_scale: pl.Tensor[[T_MAX, Q_LORA // 32], pl.FP8E8M0, pl.MX_A_ZZ],
):
    tokens = pl.tensor.dim(x, 0)
    # Separate payload and packed-scale writes keep each generated task's
    # output unambiguous when this helper is nested inside attention.
    for block in pl.spmd(((tokens + MX_QUANT_ROWS - 1) // MX_QUANT_ROWS) * (Q_LORA // MX_K_TILE), name_hint="qr_mx_payload"):
        t0 = (block // (Q_LORA // MX_K_TILE)) * MX_QUANT_ROWS
        k0 = (block % (Q_LORA // MX_K_TILE)) * MX_K_TILE
        rows = pl.min(MX_QUANT_ROWS, tokens - t0)
        loaded = pl.load(x, [t0, k0], [MX_QUANT_ROWS, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
        source = pl.set_validshape(pl.fillpad(loaded, pad_value=pl.PadValue.zero), MX_QUANT_ROWS, MX_K_TILE)
        payload, unused_scale = pl.quant_mx(source, group_axis=1)
        qr = pl.store(pl.set_validshape(payload, rows, MX_K_TILE), [t0, k0], qr)

    scale_elements = T_MAX * (Q_LORA // 32)
    backing = pl.tensor.view(qr_scale, [1, scale_elements], layout=pl.ND)
    for block in pl.spmd((T_MAX // MX_QUANT_ROWS) * (Q_LORA // MX_K_TILE), name_hint="qr_mx_scales"):
        t0 = (block // (Q_LORA // MX_K_TILE)) * MX_QUANT_ROWS
        k0 = (block % (Q_LORA // MX_K_TILE)) * MX_K_TILE
        offset = t0 * (Q_LORA // 32) + (k0 // 32) * MX_QUANT_ROWS
        if t0 < tokens:
            rows = pl.min(MX_QUANT_ROWS, tokens - t0)
            loaded = pl.load(x, [t0, k0], [MX_QUANT_ROWS, MX_K_TILE], valid_shape=[rows, MX_K_TILE])
            source = pl.set_validshape(pl.fillpad(loaded, pad_value=pl.PadValue.zero), MX_QUANT_ROWS, MX_K_TILE)
            unused_payload, scale = pl.quant_mx(source, group_axis=1)
            backing = pl.store(pl.reshape(scale, [1, MX_QUANT_ROWS * (MX_K_TILE // 32)]), [0, offset], backing)
        else:
            zero_codes = pl.tile.full([1, MX_QUANT_ROWS * (MX_K_TILE // 32)], dtype=pl.INT8, value=0)
            zero_codes_u8 = pl.reinterpret_view(zero_codes, pl.UINT8)
            zero_scale = pl.reinterpret_view(zero_codes_u8, pl.FP8E8M0)
            backing = pl.store(zero_scale, [0, offset], backing)
    return qr


def _make_norm_rope(heads, weighted):
    @pl.jit.inline
    def normalize_rotate(
        source: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
        gamma: pl.Tensor[[HEAD_DIM], pl.BF16],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
        output: pl.Tensor[[T_DYN, heads * HEAD_DIM], pl.BF16],
    ):
        tokens = pl.tensor.dim(source, 0)
        with pl.spmd(((tokens + 7) // 8) * heads, name_hint="qkv_norm_rope_bf16") as norm_tid:
            block = pl.tile.get_block_idx()
            row = (block // heads) * 8
            col = (block % heads) * HEAD_DIM
            rows = pl.min(8, tokens - row)
            loaded = pl.slice(source, [8, HEAD_DIM], [row, col], valid_shape=[rows, HEAD_DIM])
            loaded = pl.set_validshape(pl.fillpad(loaded, pad_value=pl.PadValue.zero), 8, HEAD_DIM)
            values = pl.cast(loaded, pl.FP32)
            inv = pl.rsqrt(pl.add(pl.mul(pl.row_sum(pl.mul(values, values)), 1.0 / HEAD_DIM), EPS), high_precision=True)
            normalized = pl.row_expand_mul(values, inv)
            if weighted:
                weight = pl.reshape(pl.cast(gamma[:], pl.FP32), [1, HEAD_DIM])
                normalized = pl.col_expand_mul(normalized, weight)
            # Both CANN RMSNorm and its preceding MX linear return BF16.
            rounded = pl.cast(normalized, pl.BF16, mode="rint")
            if weighted:
                # The KV cache keeps BF16 carriers after group-64 FP8 QDQ;
                # interleaved RoPE channels bypass this quantization.
                nonrope = pl.cast(rounded[:, :NOPE_DIM], pl.FP32)
                groups = pl.reshape(nonrope, [8 * (NOPE_DIM // 64), 64])
                maximum = pl.maximum(pl.row_max(pl.abs(groups)), 1e-4)
                bits = pl.reinterpret_view(pl.mul(maximum, 1.0 / 448.0), pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                normalized_mx = pl.row_expand_div(groups, scale)
                clipped = pl.minimum(pl.maximum(normalized_mx, -448.0), 448.0)
                payload = pl.cast(clipped, pl.FP8E4M3FN, mode="rint")
                restored = pl.row_expand_mul(pl.cast(payload, pl.FP32), scale)
                restored_rows = pl.reshape(restored, [8, NOPE_DIM])
                stored_nonrope = pl.cast(restored_rows, pl.BF16, mode="rint")
                output[row:row + 8, col:col + NOPE_DIM] = pl.set_validshape(stored_nonrope, rows, NOPE_DIM)
            else:
                output[row:row + 8, col:col + NOPE_DIM] = pl.slice(
                    rounded, [8, NOPE_DIM], [0, 0], valid_shape=[rows, NOPE_DIM],
                )
            tail = pl.cast(rounded[:, NOPE_DIM:HEAD_DIM], pl.FP32)
            even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
            odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
            cosine = pl.cast(rope_cos[row:row + 8, :ROPE_HALF], pl.FP32)
            sine = pl.cast(rope_sin[row:row + 8, :ROPE_HALF], pl.FP32)
            real = pl.sub(pl.mul(even, cosine), pl.mul(odd, sine))
            imag = pl.add(pl.mul(even, sine), pl.mul(odd, cosine))
            rotated = pl.full([8, ROPE_DIM], dtype=pl.FP32, value=0.0)
            rotated = pl.tensor.scatter(real, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
            rotated = pl.tensor.scatter(imag, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
            output[row:row + 8, col + NOPE_DIM:col + HEAD_DIM] = pl.set_validshape(
                pl.cast(rotated, pl.BF16, mode="rint"), rows, ROPE_DIM,
            )
        return norm_tid

    return normalize_rotate


project_qa = make_mxfp8_projection(D, Q_LORA)
project_kv = make_mxfp8_projection(D, HEAD_DIM)
project_qb = make_mxfp8_projection_from_quantized(Q_LORA, H * HEAD_DIM)
normalize_rope_q = _make_norm_rope(H, False)
normalize_rope_kv = _make_norm_rope(1, True)


@pl.jit.inline
def materialize_rope_rows(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_DIM], pl.BF16],
    position_ids: pl.Tensor[[T_DYN], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    rope_cos_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
):
    t_dim = pl.tensor.dim(position_ids, 0)
    for rope_t0 in pl.spmd(t_dim // KV_RMS_T_TILE, name_hint="qkv_rope_rows"):
        t0 = rope_t0 * KV_RMS_T_TILE
        for rope_dt in pl.range(KV_RMS_T_TILE):
            rope_t = t0 + rope_dt
            if rope_t < num_tokens:
                rope_pos_i32 = pl.read(position_ids, [rope_t])
                rope_pos = pl.cast(rope_pos_i32, pl.INDEX)
                rope_cos_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_cos[rope_pos : rope_pos + 1, 0:ROPE_DIM]
                rope_sin_t[rope_t : rope_t + 1, 0:ROPE_DIM] = freqs_sin[rope_pos : rope_pos + 1, 0:ROPE_DIM]


@pl.jit.inline
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.FP8E4M3FN],
    qr_scale: pl.Tensor[[T_MAX, Q_LORA // 32], pl.FP8E8M0, pl.MX_A_ZZ],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    tokens = pl.tensor.dim(x, 0)
    qa_bf16 = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    qr_bf16 = pl.create_tensor([tokens, Q_LORA], dtype=pl.BF16)
    q_projected = pl.create_tensor([tokens, H * HEAD_DIM], dtype=pl.BF16)
    kv_projected = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
    q_flat = pl.reshape(q, [tokens, H * HEAD_DIM])
    project_qa(x, wq_a, wq_a_scale, qa_bf16, late_dep)
    _normalize_qr(qa_bf16, gamma_cq, qr_bf16)
    _quantize_qr(qr_bf16, qr, qr_scale)
    project_qb(qr, qr_scale, wq_b, wq_b_scale, q_projected, late_dep)
    q_done = normalize_rope_q(q_projected, gamma_ckv, rope_cos, rope_sin, q_flat)
    project_kv(x, wkv, wkv_scale, kv_projected, late_dep)
    kv_done = normalize_rope_kv(kv_projected, gamma_ckv, rope_cos, rope_sin, kv)
    # Complete both native MX branches before any ordinary BF16 compressor.
    qkv_done = pl.system.task_dummy(deps=[q_done, kv_done])
    return qkv_done


@pl.jit
def qkv_proj_rope_test(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.FP8E4M3FN],
    wq_b_scale: pl.Tensor[[Q_LORA // 32, H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Out[pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    qr: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.FP8E4M3FN]],
    qr_scale: pl.Out[pl.Tensor[[T_MAX, Q_LORA // 32], pl.FP8E8M0, pl.MX_A_ZZ]],
):
    x.bind_dynamic(0, T_DYN)
    rope_cos.bind_dynamic(0, T_DYN)
    rope_sin.bind_dynamic(0, T_DYN)
    q.bind_dynamic(0, T_DYN)
    kv.bind_dynamic(0, T_DYN)
    qr.bind_dynamic(0, T_DYN)
    late_dep = pl.system.task_dummy(deps=[])
    qkv_proj_rope(
        x, wq_a, wq_a_scale, wq_b, wq_b_scale, wkv, wkv_scale,
        rope_cos, rope_sin, gamma_cq, gamma_ckv, q, kv, qr, qr_scale, late_dep,
    )
    return q


_A5_FP32_VECTOR_LANES = 64
_A5_CUBE_ACC_K = 16
_A5_CUBE_N_TILE = 128
_A5_CUBE_K_GROUP_TILE = 64


def _golden_a5_trowsum_fp32(values):
    """Mirror A5 TROWSUM's FP32 reduction order along the last dimension."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 FP32 TROWSUM golden requires float32, got {values.dtype}")
    if values.shape[-1] % _A5_FP32_VECTOR_LANES != 0:
        raise ValueError(
            f"A5 FP32 TROWSUM width must be divisible by {_A5_FP32_VECTOR_LANES}, "
            f"got {values.shape[-1]}"
        )

    groups = values.reshape(*values.shape[:-1], -1, _A5_FP32_VECTOR_LANES)
    while groups.shape[-1] > 1:
        pairs = groups.reshape(*groups.shape[:-1], -1, 2)
        groups = pairs[..., 0] + pairs[..., 1]

    group_sums = groups[..., 0]
    total = torch.zeros_like(group_sums[..., :1])
    for group in range(group_sums.shape[-1]):
        total += group_sums[..., group:group + 1]
    return total


def _golden_a5_high_precision_rsqrt(value):
    """Match A5 high-precision FP32 rsqrt without host FP32 double rounding."""
    import torch

    return torch.rsqrt(value.to(torch.float64)).to(torch.float32)


def _golden_a5_cube_bf16_matmul(lhs, rhs):
    """Return ``lhs @ rhs`` in A5 BF16 Cube's K16 MAD accumulation order."""
    import torch

    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("A5 Cube golden requires two rank-2 matrices")
    m_dim, k_dim = lhs.shape
    rhs_k_dim, n_dim = rhs.shape
    if k_dim != rhs_k_dim or k_dim % _A5_CUBE_ACC_K != 0:
        raise ValueError(
            f"A5 Cube golden requires equal K divisible by {_A5_CUBE_ACC_K}, "
            f"got {k_dim} and {rhs_k_dim}"
        )

    k_groups = k_dim // _A5_CUBE_ACC_K
    x_groups = lhs.to(torch.bfloat16).reshape(m_dim, k_groups, _A5_CUBE_ACC_K).double()
    w_groups = rhs.to(torch.bfloat16).T.contiguous().reshape(n_dim, k_groups, _A5_CUBE_ACC_K).double()
    out = torch.zeros(m_dim, n_dim, dtype=torch.float32, device=lhs.device)
    for n0 in range(0, n_dim, _A5_CUBE_N_TILE):
        n1 = min(n0 + _A5_CUBE_N_TILE, n_dim)
        acc = torch.zeros(m_dim, n1 - n0, dtype=torch.float32, device=lhs.device)
        for group0 in range(0, k_groups, _A5_CUBE_K_GROUP_TILE):
            group1 = min(group0 + _A5_CUBE_K_GROUP_TILE, k_groups)
            group_dots = torch.einsum(
                "mgk,ngk->mng",
                x_groups[:, group0:group1],
                w_groups[n0:n1, group0:group1],
            )
            for group in range(group1 - group0):
                acc = (acc.double() + group_dots[:, :, group]).float()
        out[:, n0:n1] = acc
    return out


def _golden_a5_chunked_rms_inv(values, *, chunk_size, eps):
    """A5 FP32 RMS: TROWSUM each chunk, add chunks ascending, then HP rsqrt."""
    import torch

    if values.dtype != torch.float32:
        raise ValueError(f"A5 RMS golden requires float32, got {values.dtype}")
    width = values.shape[-1]
    if width % chunk_size != 0:
        raise ValueError(f"A5 RMS width {width} must be divisible by chunk {chunk_size}")
    sq_sum = torch.zeros(
        *values.shape[:-1], 1, dtype=torch.float32, device=values.device
    )
    for k0 in range(0, width, chunk_size):
        chunk = values[..., k0:k0 + chunk_size]
        sq_sum += _golden_a5_trowsum_fp32(chunk * chunk)
    rms_arg = sq_sum * (1.0 / width) + eps
    return _golden_a5_high_precision_rsqrt(rms_arg)


def _golden_a5_q_head_rms_norm(q_full):
    """Mirror the fused Q dequant kernel's HEAD_DIM TROWSUM and HP rsqrt."""
    q_inv_rms = _golden_a5_chunked_rms_inv(q_full, chunk_size=HEAD_DIM, eps=EPS)
    return q_full * q_inv_rms


def _golden_native_mxfp8(values):
    """Independent OCP group-32 activation quantization: floor exponent + saturation."""
    import torch

    groups = values.float().reshape(*values.shape[:-1], values.shape[-1] // 32, 32)
    maximum = groups.abs().amax(-1)
    _, exponent = torch.frexp(maximum)
    codes = (exponent - 1 - 8 + 127).clamp(0, 255)
    codes = torch.where(maximum == 0, torch.zeros_like(codes), codes).to(torch.uint8)
    scales = torch.exp2(codes.float() - 127)
    quantized = (groups / scales.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return quantized.reshape(values.shape), codes


def _golden_mx_weight(weight, packed_scale):
    """Decode MX_B_NN storage independently of the checkpoint converter."""
    import torch

    groups, columns = packed_scale.shape
    logical = packed_scale.view(torch.uint8).reshape(columns // 16, groups // 2, 16, 2)
    logical = logical.permute(1, 3, 0, 2).contiguous().reshape(groups, columns)
    return weight.float() * torch.exp2(logical.float() - 127).repeat_interleave(32, dim=0)


def _golden_pack_qr_scale(codes):
    rows, groups = codes.shape
    return codes.reshape(rows // 16, 16, groups // 2, 2).permute(0, 2, 1, 3).contiguous().reshape(rows, groups)


def golden_qkv_proj_rope(tensors):
    """Independent Torch MXFP8 linears, BF16 RMS boundaries and interleaved RoPE."""
    import torch

    capacity = tensors["x"].shape[0]
    tokens = max(0, min(int(tensors.get("num_tokens", capacity)), capacity))
    for name in ("q", "kv", "qr", "qr_scale"):
        tensors[name].zero_()
    if tokens == 0:
        return
    x = tensors["x"][:tokens].to(torch.bfloat16)
    xq, xs = _golden_native_mxfp8(x)
    x_dequant = xq.float() * torch.exp2(xs.float() - 127).repeat_interleave(32, dim=-1)

    def norm(values, gamma=None):
        values = values.to(torch.bfloat16).float()
        result = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + EPS)
        if gamma is not None:
            result = result * gamma.to(torch.bfloat16).float()
        return result.to(torch.bfloat16)

    def rotate(values):
        values = values.to(torch.bfloat16)
        pairs = values[..., NOPE_DIM:].float().unflatten(-1, (-1, 2))
        cosine = tensors["rope_cos"][:tokens, :ROPE_HALF].float()
        sine = tensors["rope_sin"][:tokens, :ROPE_HALF].float()
        while cosine.ndim < pairs[..., 0].ndim:
            cosine = cosine.unsqueeze(-2)
            sine = sine.unsqueeze(-2)
        real = pairs[..., 0] * cosine - pairs[..., 1] * sine
        imag = pairs[..., 0] * sine + pairs[..., 1] * cosine
        rotary = torch.stack((real, imag), -1).flatten(-2).to(torch.bfloat16)
        return torch.cat((values[..., :NOPE_DIM], rotary), -1)

    qa = (x_dequant @ _golden_mx_weight(tensors["wq_a"], tensors["wq_a_scale"])).to(torch.bfloat16)
    normalized_qr = norm(qa, tensors["gamma_cq"])
    qr, qr_codes = _golden_native_mxfp8(normalized_qr)
    qr_dequant = qr.float() * torch.exp2(qr_codes.float() - 127).repeat_interleave(32, dim=-1)
    projected_q = (qr_dequant @ _golden_mx_weight(tensors["wq_b"], tensors["wq_b_scale"])).to(torch.bfloat16)
    q = rotate(norm(projected_q.reshape(tokens, H, HEAD_DIM)))
    projected_kv = (x_dequant @ _golden_mx_weight(tensors["wkv"], tensors["wkv_scale"])).to(torch.bfloat16)
    kv = rotate(norm(projected_kv, tensors["gamma_ckv"]))
    from kv_quant import reference_kv_quant_fp8

    kv[:, :NOPE_DIM] = reference_kv_quant_fp8(kv[:, :NOPE_DIM]).to(torch.bfloat16)
    padded_codes = torch.zeros(tensors["qr_scale"].shape, dtype=torch.uint8)
    padded_codes[:tokens] = qr_codes
    tensors["q"][:tokens] = q
    tensors["kv"][:tokens] = kv
    tensors["qr"][:tokens] = qr
    tensors["qr_scale"][:] = _golden_pack_qr_scale(padded_codes).view(torch.float8_e8m0fnu)


def _reference_q_from_quantized_qr(
    qr,
    qr_scale,
    wq_b,
    wq_b_scale,
    rope_cos,
    rope_sin,
):
    """Recompute the Q path downstream of the emitted INT8 QR boundary."""
    import torch

    t_dim = qr.shape[0]
    q_i32 = torch.matmul(qr.to(torch.int32), wq_b.to(torch.int32))
    q_scaled = q_i32.float() * qr_scale.float()
    q_scaled = q_scaled * wq_b_scale.float().view(1, -1)
    q_full = q_scaled.view(t_dim, H, HEAD_DIM)
    q_full = _golden_a5_q_head_rms_norm(q_full)

    q_pair = q_full[..., NOPE_DIM:].unflatten(-1, (-1, 2))
    q_even, q_odd = q_pair[..., 0], q_pair[..., 1]
    cos = rope_cos.float()[..., :ROPE_HALF].unsqueeze(-2)
    sin = rope_sin.float()[..., :ROPE_HALF].unsqueeze(-2)
    y_even = (q_even * cos - q_odd * sin).to(torch.bfloat16)
    y_odd = (q_even * sin + q_odd * cos).to(torch.bfloat16)
    q_rope = torch.stack([y_even, y_odd], dim=-1).flatten(-2)
    return torch.cat([q_full[..., :NOPE_DIM], q_rope], dim=-1).to(torch.bfloat16)


def quantized_qr_compare(
    *,
    max_code_step=1,
    max_changed_ratio=0.005,
    max_changed_per_row_ratio=0.005,
    max_show=10,
):
    """Bound both the magnitude and population of QR quantization-boundary changes."""
    import torch

    if max_code_step < 0:
        raise ValueError(f"max_code_step must be non-negative, got {max_code_step}")
    if not 0.0 <= max_changed_ratio <= 1.0:
        raise ValueError(
            f"max_changed_ratio must be in [0, 1], got {max_changed_ratio}"
        )
    if not 0.0 <= max_changed_per_row_ratio <= 1.0:
        raise ValueError(
            "max_changed_per_row_ratio must be in [0, 1], got "
            f"{max_changed_per_row_ratio}"
        )

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu()
        expected = expected.cpu()
        if actual.shape != expected.shape:
            return False, (
                f"    QR shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if actual.dtype != torch.int8 or expected.dtype != torch.int8:
            return False, (
                f"    QR comparator requires int8 tensors, got "
                f"{actual.dtype} and {expected.dtype}"
            )
        diff = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
        changed = diff != 0
        changed_count = int(changed.count_nonzero().item())
        changed_limit = round(max_changed_ratio * diff.numel())
        changed_per_row = changed.reshape(-1, changed.shape[-1]).sum(dim=-1)
        row_limit = int(max_changed_per_row_ratio * changed.shape[-1])
        overfull_rows = changed_per_row > row_limit
        overfull_row_count = int(overfull_rows.count_nonzero().item())
        too_large = diff > max_code_step
        too_large_count = int(too_large.count_nonzero().item())
        if (
            changed_count <= changed_limit
            and overfull_row_count == 0
            and too_large_count == 0
        ):
            return True, ""

        changed_indices = changed.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        lines = []
        for index in changed_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={int(flat_actual[index])} "
                f"expected={int(flat_expected[index])} "
                f"code_step={int(flat_diff[index])}"
            )
        return False, (
            f"    QR quantization-boundary mismatch: changed={changed_count}/"
            f"{diff.numel()} (allowed<={max_changed_ratio:.4%}, "
            f"threshold={changed_limit}), code_step>{max_code_step}: "
            f"{too_large_count}, rows>{row_limit} changed codes: "
            f"{overfull_row_count}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"quantized_qr_compare(max_code_step={max_code_step},"
        f"max_changed_ratio={max_changed_ratio},"
        f"max_changed_per_row_ratio={max_changed_per_row_ratio})"
    )
    return compare


def qr_scale_compare(
    *,
    atol=2.5e-5,
    rtol=5e-3,
    max_error_ratio=0.0,
    max_show=10,
):
    """Validate QR dequant scales with aggregate and per-row bounds."""
    import torch

    if atol < 0 or rtol < 0:
        raise ValueError("QR scale tolerances must be non-negative")
    if not 0.0 <= max_error_ratio <= 1.0:
        raise ValueError(
            f"max_error_ratio must be in [0, 1], got {max_error_ratio}"
        )
    scale_atol = atol
    scale_rtol = rtol

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del actual_outputs, expected_outputs, inputs, rtol, atol
        actual = actual.cpu().to(torch.float32)
        expected = expected.cpu().to(torch.float32)
        if actual.shape != expected.shape:
            return False, (
                f"    QR scale shape mismatch: {tuple(actual.shape)} vs "
                f"{tuple(expected.shape)}"
            )
        if not torch.isfinite(actual).all().item() or not torch.isfinite(expected).all().item():
            return False, "    QR scales contain NaN or Inf"
        if (actual <= 0).any().item() or (expected <= 0).any().item():
            return False, "    QR scales must be positive"

        diff = (actual - expected).abs()
        tolerance = scale_atol + scale_rtol * expected.abs()
        bad = diff > tolerance
        bad_count = int(bad.count_nonzero().item())
        threshold = round(max_error_ratio * actual.numel())
        hard_bad = diff > (2.0 * tolerance)
        hard_bad_count = int(hard_bad.count_nonzero().item())
        if bad_count <= threshold and hard_bad_count == 0:
            return True, ""

        bad_indices = bad.flatten().nonzero(as_tuple=False).flatten()
        flat_actual = actual.flatten()
        flat_expected = expected.flatten()
        flat_diff = diff.flatten()
        flat_tolerance = tolerance.flatten()
        lines = []
        for index in bad_indices[:max_show].tolist():
            lines.append(
                f"      [{index}] actual={float(flat_actual[index]):.8g} "
                f"expected={float(flat_expected[index]):.8g} "
                f"diff={float(flat_diff[index]):.4g} "
                f"tol={float(flat_tolerance[index]):.4g}"
            )
        return False, (
            f"    QR scale mismatch: bad={bad_count}/{actual.numel()} "
            f"(allowed<={max_error_ratio:.4%}, threshold={threshold}), "
            f"hard_bad={hard_bad_count}, atol={scale_atol}, rtol={scale_rtol}\n"
            + "\n".join(lines)
        )

    compare.__name__ = (
        f"qr_scale_compare(atol={scale_atol},rtol={scale_rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def q_from_runtime_qr_compare(
    *,
    atol=1e-4,
    rtol=1.0 / 128,
    max_error_ratio=0.005,
):
    """Validate Q against a reference conditioned on the emitted QR codes."""
    from golden import ratio_allclose

    base_compare = ratio_allclose(atol=atol, rtol=rtol, max_error_ratio=max_error_ratio)

    def compare(
        actual,
        expected,
        *,
        actual_outputs,
        expected_outputs,
        inputs,
        rtol,
        atol,
    ):
        del expected
        required_outputs = ("qr", "qr_scale")
        required_inputs = ("wq_b", "wq_b_scale", "rope_cos", "rope_sin")
        missing_outputs = [name for name in required_outputs if name not in actual_outputs]
        missing_inputs = [name for name in required_inputs if name not in inputs]
        if missing_outputs or missing_inputs:
            return False, (
                "    conditioned Q comparator is missing "
                f"outputs={missing_outputs}, inputs={missing_inputs}"
            )

        conditioned = _reference_q_from_quantized_qr(
            actual_outputs["qr"].cpu(),
            actual_outputs["qr_scale"].cpu(),
            inputs["wq_b"].cpu(),
            inputs["wq_b_scale"].cpu(),
            inputs["rope_cos"].cpu(),
            inputs["rope_sin"].cpu(),
        )
        if actual.shape != conditioned.shape:
            return False, (
                f"    conditioned Q shape mismatch: actual={tuple(actual.shape)} "
                f"reference={tuple(conditioned.shape)}"
            )
        ok, detail = base_compare(
            actual,
            conditioned,
            actual_outputs=actual_outputs,
            expected_outputs=expected_outputs,
            inputs=inputs,
            rtol=rtol,
            atol=atol,
        )
        if ok:
            return True, ""
        return False, "    Q downstream of emitted QR does not match:\n" + detail

    compare.__name__ = (
        f"q_from_runtime_qr_compare(atol={atol},rtol={rtol},"
        f"max_error_ratio={max_error_ratio})"
    )
    return compare


def build_tensor_specs(B, S):
    import torch
    from golden import TensorSpec

    tokens = B * S
    padded_tokens = T_MAX

    def weight_pair(width, columns):
        source = torch.empty(columns, width, dtype=torch.bfloat16).uniform_(-0.1, 0.1)
        payload, codes = _golden_native_mxfp8(source)
        logical = codes.T.contiguous()
        groups = width // 32
        packed = logical.reshape(groups // 2, 2, columns // 16, 16).permute(2, 0, 3, 1)
        return payload.T.contiguous(), packed.contiguous().reshape(groups, columns).view(torch.float8_e8m0fnu)

    qa, qa_scale = weight_pair(D, Q_LORA)
    qb, qb_scale = weight_pair(Q_LORA, H * HEAD_DIM)
    kv, kv_scale = weight_pair(D, HEAD_DIM)
    return [
        TensorSpec("x", [tokens, D], torch.bfloat16, init_value=lambda: torch.randn(tokens, D).to(torch.bfloat16)),
        TensorSpec("wq_a", [D, Q_LORA], torch.float8_e4m3fn, init_value=lambda: qa),
        TensorSpec("wq_a_scale", [D // 32, Q_LORA], torch.float8_e8m0fnu, init_value=lambda: qa_scale),
        TensorSpec("wq_b", [Q_LORA, H * HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: qb),
        TensorSpec("wq_b_scale", [Q_LORA // 32, H * HEAD_DIM], torch.float8_e8m0fnu, init_value=lambda: qb_scale),
        TensorSpec("wkv", [D, HEAD_DIM], torch.float8_e4m3fn, init_value=lambda: kv),
        TensorSpec("wkv_scale", [D // 32, HEAD_DIM], torch.float8_e8m0fnu, init_value=lambda: kv_scale),
        TensorSpec("rope_cos", [tokens, ROPE_DIM], torch.bfloat16,
                   init_value=lambda: torch.rand(tokens, ROPE_DIM).to(torch.bfloat16)),
        TensorSpec("rope_sin", [tokens, ROPE_DIM], torch.bfloat16,
                   init_value=lambda: torch.rand(tokens, ROPE_DIM).to(torch.bfloat16)),
        TensorSpec("gamma_cq", [Q_LORA], torch.bfloat16,
                   init_value=lambda: (1.0 + 0.1 * torch.randn(Q_LORA)).to(torch.bfloat16)),
        TensorSpec("gamma_ckv", [HEAD_DIM], torch.bfloat16,
                   init_value=lambda: (1.0 + 0.1 * torch.randn(HEAD_DIM)).to(torch.bfloat16)),
        TensorSpec("q", [tokens, H, HEAD_DIM], torch.bfloat16),
        TensorSpec("kv", [tokens, HEAD_DIM], torch.bfloat16),
        TensorSpec("qr", [tokens, Q_LORA], torch.float8_e4m3fn),
        TensorSpec("qr_scale", [padded_tokens, Q_LORA // 32], torch.float8_e8m0fnu),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    MODES = {
        "decode":  (DECODE_BATCH, DECODE_SEQ),
        "prefill": (PREFILL_BATCH, PREFILL_SEQ),
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--platform", type=str, default="a5", choices=["a2a3", "a2a3sim", "a5", "a5sim"])
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument(
        "--mode", choices=["decode", "prefill", "all"], default="all",
        help="Use decode or prefill batch sizes, or 'all' to test both.",
    )
    parser.add_argument(
        "--enable-chip-swimlane", type=int, choices=[0, 1, 2, 4], default=0,
        help="chip swimlane level: 0=off, 1=per-kernel AICore timing "
        "(prints the per-function Task Statistics table), 2=+AICPU timing.",
    )
    parser.add_argument("--runtime-dir", type=str, default=None)
    parser.add_argument("--golden-data", type=str, default=None)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--compile-only", action="store_true", default=False)
    parser.add_argument("--dump-passes", action="store_true", default=False)
    args = parser.parse_args()

    modes_to_run = list(MODES.keys()) if args.mode == "all" else [args.mode]

    for mode_name in modes_to_run:
        B, S = MODES[mode_name]
        print(f"--- qkv_proj_rope {mode_name}: B={B}, S={S} ---")
        result = run(
            fn=qkv_proj_rope_test,
            specs=build_tensor_specs(B, S),
            golden_fn=golden_qkv_proj_rope,
            rtol=5e-3,
            atol=5e-3,
            compare_fn={
                "q": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.005),
                "kv": ratio_allclose(atol=1e-3, rtol=1.0 / 64, max_error_ratio=0.005),
                "qr": ratio_allclose(atol=1e-3, rtol=1.0 / 8, max_error_ratio=0.005),
                "qr_scale": ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0),
            },
            runtime_dir=args.runtime_dir,
            golden_data=args.golden_data,
            save_data=args.save_data,
            config=dict(
                dump_passes=args.dump_passes,
                platform=args.platform,
                device_id=args.device,
                enable_chip_swimlane=args.enable_chip_swimlane,
            ),
            compile_only=args.compile_only,
        )
        if not result.passed:
            if result.error:
                print(result.error)
            raise SystemExit(1)
