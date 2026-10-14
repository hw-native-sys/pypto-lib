# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Attention projection, RMSNorm, and RoPE kernel factories."""

import pypto.language as pl
import torch

from config import FLASH as M, T_DYN, TP_SIZE


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
INDEX_TOPK = M.index_topk


EPS = M.rms_norm_eps
M_TILE = 16
MX_M_TILE = 32
N_TILE = 128
K_TILE = 256


from quantization import _prefill_operand
from quantization import prefill_quantize_dequantize
import torch.nn.functional as F


def prefill_linear(x, weight, bias=None, *, precision="fp32", format="fp32"):
    """Contract output-major weights with explicit released matrix roles.

    Quantized roles use FP8 activations and sequential FP32 group-32 partials.
    Decoded weights retain their exact power-of-two scales. Dense FP32 roles
    (HC, router, ratio-two compressor and head) never narrow their outputs.
    """
    if format not in ("fp32", "bf16", "mxfp8", "mxfp4"):
        raise ValueError("unknown checkpoint linear format")
    if precision not in ("fp32", "official"):
        raise ValueError("precision must be 'fp32' or 'official'")
    if precision == "official" and format in ("mxfp8", "mxfp4"):
        if bias is not None or x.shape[-1] % 32:
            raise ValueError("official quantized linears require no bias and a group-32 width")
        quantized = prefill_quantize_dequantize(x, "mxfp8")
        output = x.new_zeros((*x.shape[:-1], weight.shape[0]), dtype=torch.float32)
        for start in range(0, x.shape[-1], 32):
            partial = F.linear(
                _prefill_operand(quantized[..., start:start + 32]), _prefill_operand(weight[..., start:start + 32])
            ).float()
            output = output + partial
        return output.bfloat16().float()
    if precision == "official" and format == "bf16":
        x, weight = x.bfloat16().float(), weight.bfloat16().float()
    bias = None if bias is None else _prefill_operand(bias)
    output = F.linear(_prefill_operand(x), _prefill_operand(weight), bias).float()
    return output.bfloat16().float() if precision == "official" and format == "bf16" else output


def prefill_matmul(left, right):
    """Matrix/vector/batched contraction with FP32 operands and result."""
    return torch.matmul(_prefill_operand(left), _prefill_operand(right)).float()


def prefill_einsum(equation, *operands):
    """Evaluate an explicit Einstein contraction with one final FP32 rounding."""
    return torch.einsum(equation, *(_prefill_operand(value) for value in operands)).float()


def prefill_publish(cache, values, slots):
    """Publish only caller-owned physical rows, preserving all other bytes."""
    if cache.dtype != torch.float32 or values.dtype != torch.float32:
        raise ValueError("FP32 publication cannot narrow values or caches")
    flat = cache.flatten(0, 1)
    valid = slots >= 0
    flat[slots[valid].long()] = values[valid].reshape_as(flat[slots[valid].long()])


def _prefill_online_attention(query, window, window_indices, compressed, compressed_indices, sink, extents):
    """Released online64 order, including holes and unrounded FP32 denominator."""
    query = query.bfloat16().float()
    output = torch.empty_like(query)
    window = window.flatten(0, 1).bfloat16().float()
    compressed = compressed.flatten(0, 1).bfloat16().float() if compressed is not None else None
    for token in range(query.shape[0]):
        window_width = int(extents[token, 0]) if extents is not None else window_indices.shape[1]
        global_width = int(extents[token, 1]) if extents is not None else (
            compressed_indices.shape[1] if compressed_indices is not None else 0
        )
        indices = [window_indices[token, :window_width]]
        caches = [window]
        if global_width:
            indices.append(compressed_indices[token, :global_width])
            caches.append(compressed)
        masks = [rows >= 0 for rows in indices]
        keys = torch.cat([
            torch.where(mask[:, None], cache[rows.clamp_min(0).long()], 0)
            for rows, mask, cache in zip(indices, masks, caches, strict=True)
        ])
        valid = torch.cat(masks)
        padding = -keys.shape[0] % 64
        keys = F.pad(keys, (0, 0, 0, padding))
        valid = F.pad(valid, (0, padding), value=False)
        maximum = query.new_full((query.shape[1],), -1e30)
        denominator = torch.zeros_like(maximum)
        accumulator = torch.zeros_like(query[token])
        for start in range(0, keys.shape[0], 64):
            block = keys[start:start + 64]
            mask = valid[start:start + 64]
            scores = prefill_matmul(query[token], block.T) * query.shape[-1] ** -0.5
            scores = scores.masked_fill(~mask[None, :], -torch.inf)
            updated_maximum = torch.maximum(maximum, scores.amax(-1))
            alpha = torch.exp(maximum - updated_maximum)
            probability = torch.exp(scores - updated_maximum[:, None])
            denominator = denominator * alpha + probability.sum(-1)
            accumulator = accumulator * alpha[:, None] + prefill_matmul(probability.bfloat16().float(), block)
            maximum = updated_maximum
        denominator = denominator + torch.exp(sink - maximum)
        output[token] = (accumulator / denominator[:, None]).bfloat16().float()
    return output


def prefill_sparse_attention_reference(
    query, window, window_indices, compressed, compressed_indices, sink, *, precision="fp32", extents=None
):
    """Independent global FP32 or released online64 mixed-precision attention."""
    if precision == "official":
        return _prefill_online_attention(query, window, window_indices, compressed, compressed_indices, sink, extents)
    if query.dtype != torch.float32:
        raise ValueError("FP32 attention requires FP32 queries")
    output = torch.empty_like(query)
    window_rows = window.flatten(0, 1)
    global_rows = compressed.flatten(0, 1) if compressed is not None else None
    for token in range(query.shape[0]):
        rows = window_indices[token]
        selected = [window_rows[rows[rows >= 0].long()]]
        if global_rows is not None and compressed_indices is not None:
            rows = compressed_indices[token]
            selected.append(global_rows[rows[rows >= 0].long()])
        keys = torch.cat(selected)
        scores = prefill_matmul(query[token], keys.T) * query.shape[-1] ** -0.5
        probability = torch.cat((scores, sink[:, None]), dim=-1).softmax(dim=-1)[:, :-1]
        output[token] = prefill_matmul(probability, keys)
    return output


def _prefill_project(x, weights, name, precision):
    format = weights["formats"][name] if precision == "official" else "fp32"
    return prefill_linear(x, weights[name], precision=precision, format=format)


def golden_paged_sparse_attention_stats(
    q: torch.Tensor,
    cache: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token-major online-softmax statistics from physical paged rows."""
    flat_cache = cache.flatten(0, 1).squeeze(-2)
    safe = indices.clamp_min(0).to(torch.long)
    selected = flat_cache[safe]
    logits = torch.einsum("thd,tkd->thk", q.float(), selected.float()) * q.shape[-1] ** -0.5
    logits = logits.masked_fill(indices.unsqueeze(-2) < 0, -torch.inf)
    maximum = logits.amax(dim=-1)
    finite_maximum = maximum.masked_fill(~torch.isfinite(maximum), 0.0)
    exponentials = torch.exp(logits - finite_maximum.unsqueeze(-1))
    exponentials = exponentials.masked_fill(~torch.isfinite(logits), 0.0)
    denominator = exponentials.sum(dim=-1)
    numerator = torch.einsum("thk,tkd->thd", exponentials, selected.float())
    return maximum, denominator, numerator


def golden_paged_sparse_attention(
    q: torch.Tensor,
    window_cache: torch.Tensor,
    window_indices: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_indices: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Merge token-major paged window and compressed attention with one sink."""
    parts = (
        golden_paged_sparse_attention_stats(q, window_cache, window_indices),
        golden_paged_sparse_attention_stats(q, compressed_cache, compressed_indices),
    )
    return golden_merge_attention_stats(parts, sink).to(q.dtype)


def golden_merge_attention_stats(
    parts: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    sink: torch.Tensor,
) -> torch.Tensor:
    """Merge sparse sources with one attention sink in a common softmax denominator."""
    maxima = [part[0] for part in parts]
    sink_value = sink.float().view(*([1] * (maxima[0].ndim - 1)), -1)
    global_maximum = torch.maximum(torch.stack(maxima).amax(dim=0), sink_value)
    denominator = torch.exp(sink_value - global_maximum)
    numerator = torch.zeros_like(parts[0][2])
    for maximum, local_sum, local_numerator in parts:
        scale = torch.exp(maximum - global_maximum)
        scale = scale.masked_fill(~torch.isfinite(maximum), 0.0)
        denominator = denominator + local_sum * scale
        numerator = numerator + local_numerator * scale.unsqueeze(-1)
    return numerator / denominator.unsqueeze(-1)


def make_mx_projection(width, output_width, output_dtype=pl.BF16, *, name_hint="attention_mx_projection"):
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
            for block in pl.spmd(output_width // N_TILE, name_hint=name_hint):
                n0 = block * N_TILE
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                first = pl.load(x, [t0, 0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                firstq_values = pl.reshape(pl.cast(first, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                firstq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                firstq_maximum = pl.maximum(pl.row_max(pl.abs(firstq_values), tmp_tile=firstq_reduce_tmp), 1e-4)
                firstq_bits = pl.reinterpret_view(pl.mul(firstq_maximum, 1.0 / 448.0), pl.INT32)
                firstq_exponent = pl.shrs(pl.add(firstq_bits, 8388607), 23)
                firstq_scale = pl.reinterpret_view(pl.shls(firstq_exponent, 23), pl.FP32)
                firstq_quantized = pl.cast(
                    pl.row_expand_div(firstq_values, firstq_scale), pl.FP8E4M3FN, mode="rint"
                )
                firstq_payload = pl.reshape(firstq_quantized, [MX_M_TILE, K_TILE])
                firstq_signed_exponent = pl.sub(firstq_exponent, pl.mul(pl.shrs(firstq_exponent, 7), 256))
                firstq_codes = pl.reinterpret_view(pl.cast(firstq_signed_exponent, pl.INT8), pl.UINT8)
                firstq_flat = pl.reshape(firstq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                firstq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                firstq_packed = pl.tmov_x2zz(
                    firstq_flat, firstq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                )
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
                    nextq_maximum = pl.maximum(
                        pl.row_max(pl.abs(nextq_values), tmp_tile=nextq_reduce_tmp), 1e-4
                    )
                    nextq_bits = pl.reinterpret_view(pl.mul(nextq_maximum, 1.0 / 448.0), pl.INT32)
                    nextq_exponent = pl.shrs(pl.add(nextq_bits, 8388607), 23)
                    nextq_scale = pl.reinterpret_view(pl.shls(nextq_exponent, 23), pl.FP32)
                    nextq_quantized = pl.cast(
                        pl.row_expand_div(nextq_values, nextq_scale), pl.FP8E4M3FN, mode="rint"
                    )
                    nextq_payload = pl.reshape(nextq_quantized, [MX_M_TILE, K_TILE])
                    nextq_signed_exponent = pl.sub(
                        nextq_exponent, pl.mul(pl.shrs(nextq_exponent, 7), 256)
                    )
                    nextq_codes = pl.reinterpret_view(pl.cast(nextq_signed_exponent, pl.INT8), pl.UINT8)
                    nextq_flat = pl.reshape(nextq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                    nextq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                    nextq_packed = pl.tmov_x2zz(
                        nextq_flat, nextq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                    )
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


def _make_norm_block(width):
    @pl.jit.inline
    def normalize_block(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        block: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        t = block * 8
        rows = pl.min(8, num_tokens - t)
        if width == D:
            square_sum = pl.full([1, 8], dtype=pl.FP32, value=0.0)
            for chunk in pl.pipeline(width // 128, stage=2):
                d0 = chunk * 128
                source_chunk = pl.slice(x, [8, 128], [t, d0], valid_shape=[rows, 128])
                source_chunk = pl.set_validshape(
                    pl.fillpad(source_chunk, pad_value=pl.PadValue.zero), 8, 128
                )
                value_chunk = pl.cast(source_chunk, pl.FP32)
                chunk_sum = pl.reshape(pl.row_sum(pl.mul(value_chunk, value_chunk)), [1, 8])
                square_sum = pl.add(square_sum, chunk_sum)
            inverse = pl.rsqrt(pl.add(pl.mul(square_sum, 1.0 / width), EPS), high_precision=True)
            inverse_col = pl.reshape(inverse, [8, 1])
            for chunk in pl.pipeline(width // 128, stage=2):
                d0 = chunk * 128
                source_chunk = pl.slice(x, [8, 128], [t, d0], valid_shape=[rows, 128])
                source_chunk = pl.set_validshape(
                    pl.fillpad(source_chunk, pad_value=pl.PadValue.zero), 8, 128
                )
                value_chunk = pl.cast(source_chunk, pl.FP32)
                gamma_chunk = pl.reshape(pl.cast(weight[d0 : d0 + 128], pl.FP32), [1, 128])
                result_chunk = pl.col_expand_mul(
                    pl.row_expand_mul(value_chunk, inverse_col), gamma_chunk
                )
                output[t : t + 8, d0 : d0 + 128] = pl.set_validshape(
                    pl.cast(result_chunk, pl.BF16, mode="rint"), rows, 128
                )
        else:
            source = pl.slice(x, [8, width], [t, 0], valid_shape=[rows, width])
            source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 8, width)
            value = pl.cast(source, pl.FP32)
            row_inverse = pl.rsqrt(
                pl.add(pl.mul(pl.row_sum(pl.mul(value, value)), 1.0 / width), EPS),
                high_precision=True,
            )
            gamma = pl.reshape(pl.cast(weight[:], pl.FP32), [1, width])
            normalized = pl.col_expand_mul(pl.row_expand_mul(value, row_inverse), gamma)
            output[t : t + 8, :] = pl.set_validshape(
                pl.cast(normalized, pl.BF16, mode="rint"), rows, width
            )
        return output

    return normalize_block


def make_norm(width, *, max_workers=None, name_hint="attention_rmsnorm"):
    """Specialize RMSNorm with direct or capped strided SPMD scheduling."""
    normalize_block = _make_norm_block(width)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = (num_tokens + 7) // 8
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                normalize_block(x, weight, output, block, num_tokens)
        return output

    return normalize


def make_norm_with_deps(width, *, max_workers=None, name_hint="attention_rmsnorm"):
    """Specialize RMSNorm with an explicit producer dependency."""
    normalize_block = _make_norm_block(width)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def normalize(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width], pl.BF16],
        output: pl.Tensor[[T_DYN, width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = (num_tokens + 7) // 8
        workers = pl.min(blocks, worker_limit)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as norm_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, workers):
                normalize_block(x, weight, output, block, num_tokens)
        return norm_tid

    return normalize


def _make_rope_block(heads, inverse, head_dim, rope_dim):
    sign = -1.0 if inverse else 1.0
    nope_dim = head_dim - rope_dim

    @pl.jit.inline
    def rotate_block(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        block: pl.Scalar[pl.INT32],
    ):
        t = block // heads
        h = block % heads
        base = h * head_dim
        output[t : t + 1, base : base + nope_dim] = x[t : t + 1, base : base + nope_dim]
        tail = pl.cast(x[t : t + 1, base + nope_dim : base + head_dim], pl.FP32)
        even = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P0101)
        odd = pl.gather(tail, mask_pattern=pl.tile.MaskPattern.P1010)
        c = cos[t : t + 1, :]
        s = pl.mul(sin[t : t + 1, :], sign)
        re = pl.sub(pl.mul(even, c), pl.mul(odd, s))
        im = pl.add(pl.mul(even, s), pl.mul(odd, c))
        rotated = pl.full([1, rope_dim], dtype=pl.FP32, value=0.0)
        rotated = pl.tensor.scatter(re, mask_pattern=pl.tile.MaskPattern.P0101, dst=rotated)
        rotated = pl.tensor.scatter(im, mask_pattern=pl.tile.MaskPattern.P1010, dst=rotated)
        output[t : t + 1, base + nope_dim : base + head_dim] = pl.cast(
            rotated, pl.BF16, mode="rint"
        )
        return output

    return rotate_block


def make_rope(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with direct or capped strided scheduling."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return output

    return rotate


def make_rope_with_deps(
    heads,
    inverse=False,
    *,
    head_dim=HEAD_DIM,
    rope_dim=ROPE_DIM,
    max_workers=None,
    name_hint="attention_rope",
):
    """Specialize adjacent-pair RoPE with an explicit producer dependency."""
    rotate_block = _make_rope_block(heads, inverse, head_dim, rope_dim)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def rotate(
        x: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        cos: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        sin: pl.Tensor[[T_DYN, rope_dim // 2], pl.FP32],
        output: pl.Tensor[[T_DYN, heads * head_dim], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = num_tokens * heads
        workers = pl.min(blocks, worker_limit)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as rope_tid:
            worker = pl.tile.get_block_idx()
            for block in pl.range(worker, blocks, workers):
                rotate_block(x, cos, sin, output, block)
        return rope_tid

    return rotate


def _make_bf16_projection_block(width, output_width):
    n_tile = min(N_TILE, output_width)
    k_tile = min(K_TILE, width)

    @pl.jit.inline
    def project_block(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        task: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        output_tiles = output_width // n_tile
        row = task // output_tiles * M_TILE
        col = task % output_tiles * n_tile
        rows = pl.min(M_TILE, num_tokens - row)
        acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
        for k in pl.range(0, width, k_tile):
            a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
            b = pl.slice(weight, [k_tile, n_tile], [k, col])
            acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
        output[row : row + M_TILE, col : col + n_tile] = pl.set_validshape(
            pl.cast(acc, pl.BF16, mode="rint"), rows, n_tile
        )
        return output

    return project_block


def make_bf16_projection(
    width,
    output_width,
    *,
    max_workers=None,
    name_hint="attention_bf16_projection",
):
    """Specialize a BF16 projection with an FP32 accumulator."""
    project_block = _make_bf16_projection_block(width, output_width)
    n_tile = min(N_TILE, output_width)
    worker_limit = 2**31 - 1 if max_workers is None else max_workers

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        workers = pl.min(blocks, worker_limit)
        for worker in pl.spmd(workers, name_hint=name_hint):
            for task in pl.range(worker, blocks, workers):
                project_block(source, weight, output, task, num_tokens)
        return output

    return project


def make_bf16_projection_staged(width, output_width):
    """Stage FP32 projection values before a separate BF16 narrowing task.

    The BF16 operand passes through Vector before Cube starts its result transfer
    (hw-native-sys/pypto#2829). Store FP32 values through Vector as well: direct
    Acc-to-GM stores were unstable in the concurrent A5 prefill workload.
    Accumulation and RNE rounding are unchanged.
    """
    n_tile = min(N_TILE, output_width)
    k_tile = min(K_TILE, width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
    ):
        tokens = pl.tensor.dim(source, 0)
        accumulated = pl.create_tensor([tokens, output_width], dtype=pl.FP32)
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cube") as cube_tid:
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            acc = pl.create_tensor([M_TILE, n_tile], dtype=pl.FP32)
            for k in pl.range(0, width, k_tile):
                a = pl.slice(source, [M_TILE, k_tile], [row, k], valid_shape=[rows, k_tile])
                # Preserve BF16 values while establishing Vector-to-Cube startup.
                a = pl.cast(pl.cast(a, pl.FP32), pl.BF16, mode="rint")
                b = pl.slice(weight, [k_tile, n_tile], [k, col])
                acc = pl.matmul_acc(acc, a, b, init_cond=(k == 0))
            accumulated[row : row + M_TILE, col : col + n_tile] = pl.set_validshape(pl.mul(acc, 1.0), rows, n_tile)
        with pl.spmd(blocks, name_hint="prefill_bf16_projection_cast", deps=[cube_tid]):
            block = pl.tile.get_block_idx()
            row = block // (output_width // n_tile) * M_TILE
            col = block % (output_width // n_tile) * n_tile
            rows = pl.min(M_TILE, num_tokens - row)
            value = pl.slice(accumulated, [M_TILE, n_tile], [row, col], valid_shape=[rows, n_tile])
            output[row : row + M_TILE, col : col + n_tile] = pl.cast(value, pl.BF16, mode="rint")
        return output

    return project


def make_bf16_projection_with_deps(
    width,
    output_width,
    *,
    workers=32,
    name_hint="attention_bf16_projection",
):
    """Specialize a BF16 projection with an explicit producer dependency."""
    project_block = _make_bf16_projection_block(width, output_width)
    n_tile = min(N_TILE, output_width)

    @pl.jit.inline
    def project(
        source: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.BF16],
        output: pl.Tensor[[T_DYN, output_width], pl.BF16],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        blocks = (num_tokens + M_TILE - 1) // M_TILE * (output_width // n_tile)
        with pl.spmd(workers, name_hint=name_hint, deps=[ready]) as projection_tid:
            worker = pl.tile.get_block_idx()
            for task in pl.range(worker, blocks, workers):
                project_block(source, weight, output, task, num_tokens)
        return projection_tid

    return project


__all__ = [
    "EPS",
    "K_TILE",
    "M_TILE",
    "MX_M_TILE",
    "N_TILE",
    "make_bf16_projection",
    "make_bf16_projection_staged",
    "make_bf16_projection_with_deps",
    "make_norm",
    "make_norm_with_deps",
    "make_mx_projection",
    "make_mx_projection_with_deps",
    "make_rope",
    "make_rope_with_deps",
]


def make_mx_projection_with_deps(width, output_width, output_dtype=pl.BF16, *,
    n_tile=N_TILE, name_hint="attention_mx_projection",
):
    """Specialize an MXFP8 projection with an explicit producer dependency."""
    fp32_output = output_dtype == pl.FP32

    @pl.jit.inline
    def project(
        x: pl.Tensor[[T_DYN, width], pl.BF16],
        weight: pl.Tensor[[width, output_width], pl.FP8E4M3FN],
        scale: pl.Tensor[[width // 32, output_width], pl.FP8E8M0, pl.MX_B_NN],
        output: pl.Tensor[[T_DYN, output_width], output_dtype],
        num_tokens: pl.Scalar[pl.INT32],
        ready: pl.Scalar[pl.TASK_ID],
    ):
        joined_tid = ready
        for mt in pl.range((num_tokens + MX_M_TILE - 1) // MX_M_TILE):
            t0 = mt * MX_M_TILE
            with pl.spmd(output_width // n_tile, name_hint=name_hint, deps=[ready]) as project_tid:
                block = pl.tile.get_block_idx()
                n0 = block * n_tile
                rows = pl.min(MX_M_TILE, num_tokens - t0)
                first = pl.load(x, [t0, 0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                first = pl.set_validshape(pl.fillpad(first, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                firstq_values = pl.reshape(pl.cast(first, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                firstq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                firstq_maximum = pl.maximum(pl.row_max(pl.abs(firstq_values), tmp_tile=firstq_reduce_tmp), 1e-4)
                firstq_bits = pl.reinterpret_view(pl.mul(firstq_maximum, 1.0 / 448.0), pl.INT32)
                firstq_exponent = pl.shrs(pl.add(firstq_bits, 8388607), 23)
                firstq_scale = pl.reinterpret_view(pl.shls(firstq_exponent, 23), pl.FP32)
                firstq_quantized = pl.cast(
                    pl.row_expand_div(firstq_values, firstq_scale), pl.FP8E4M3FN, mode="rint"
                )
                firstq_payload = pl.reshape(firstq_quantized, [MX_M_TILE, K_TILE])
                firstq_signed_exponent = pl.sub(firstq_exponent, pl.mul(pl.shrs(firstq_exponent, 7), 256))
                firstq_codes = pl.reinterpret_view(pl.cast(firstq_signed_exponent, pl.INT8), pl.UINT8)
                firstq_flat = pl.reshape(firstq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                firstq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                firstq_packed = pl.tmov_x2zz(
                    firstq_flat, firstq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                )
                a0 = firstq_payload
                sa0 = pl.reinterpret_view(firstq_packed, pl.FP8E8M0)
                b0 = pl.load(weight, [0, n0], [K_TILE, n_tile])
                sb0 = pl.load(scale, [0, n0], [K_TILE // 32, n_tile])
                acc = pl.matmul_mx(a0, sa0, b0, sb0)
                for kb in pl.range(1, width // K_TILE):
                    k0 = kb * K_TILE
                    values = pl.load(x, [t0, k0], [MX_M_TILE, K_TILE], valid_shape=[rows, K_TILE])
                    values = pl.set_validshape(pl.fillpad(values, pad_value=pl.PadValue.zero), MX_M_TILE, K_TILE)
                    nextq_values = pl.reshape(pl.cast(values, pl.FP32), [MX_M_TILE * (K_TILE // 32), 32])
                    nextq_reduce_tmp = pl.create_tile([MX_M_TILE * (K_TILE // 32), 32], dtype=pl.FP32)
                    nextq_maximum = pl.maximum(
                        pl.row_max(pl.abs(nextq_values), tmp_tile=nextq_reduce_tmp), 1e-4
                    )
                    nextq_bits = pl.reinterpret_view(pl.mul(nextq_maximum, 1.0 / 448.0), pl.INT32)
                    nextq_exponent = pl.shrs(pl.add(nextq_bits, 8388607), 23)
                    nextq_scale = pl.reinterpret_view(pl.shls(nextq_exponent, 23), pl.FP32)
                    nextq_quantized = pl.cast(
                        pl.row_expand_div(nextq_values, nextq_scale), pl.FP8E4M3FN, mode="rint"
                    )
                    nextq_payload = pl.reshape(nextq_quantized, [MX_M_TILE, K_TILE])
                    nextq_signed_exponent = pl.sub(
                        nextq_exponent, pl.mul(pl.shrs(nextq_exponent, 7), 256)
                    )
                    nextq_codes = pl.reinterpret_view(pl.cast(nextq_signed_exponent, pl.INT8), pl.UINT8)
                    nextq_flat = pl.reshape(nextq_codes, [1, MX_M_TILE * (K_TILE // 32)])
                    nextq_tmp = pl.create_tile([1, 96], dtype=pl.UINT8)
                    nextq_packed = pl.tmov_x2zz(
                        nextq_flat, nextq_tmp, group_axis=1, dst_rows=MX_M_TILE, dst_cols=8
                    )
                    a = nextq_payload
                    sa = pl.reinterpret_view(nextq_packed, pl.FP8E8M0)
                    b = pl.load(weight, [k0, n0], [K_TILE, n_tile])
                    sb = pl.load(scale, [k0 // 32, n0], [K_TILE // 32, n_tile])
                    acc = pl.matmul_mx_acc(acc, a, sa, b, sb)
                if fp32_output:
                    output = pl.store(pl.set_validshape(pl.mul(acc, 1.0), rows, n_tile), [t0, n0], output)
                else:
                    value = pl.cast(acc, target_type=pl.BF16, mode="rint")
                    output = pl.store(pl.set_validshape(value, rows, n_tile), [t0, n0], output)
            joined_tid = pl.system.task_dummy(deps=[joined_tid, project_tid])
        return joined_tid

    return project
